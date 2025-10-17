#!/usr/bin/env python3
import argparse
from collections import deque
import math
import os

import brevitas.nn as qnn
import torch
from torch import nn
import yaml

from network import networks


# --------------------------
# Boilerplate
# --------------------------
def setcache(m):
    m.cache_inference_quant_bias = True


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(
        "Extract per-Residual CODE shifts to map {x, fn(x)} scales to the new input_quant scale"
    )
    p.add_argument("--model_dir", type=str, required=True,
                   help="Trained model directory (needs files: model.pth, config.yml)")
    return vars(p.parse_args())


def load_model(config, model_pth, device):
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    model.apply(setcache)
    state = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def generate_input(config, device):
    C, H, W = config["input_channels"], config["input_h"], config["input_w"]
    return torch.zeros(1, C, H, W, device=device)


def _scale_of(qt):
    """
    Extract numeric scale from a Brevitas QuantTensor or return None.
    Works for inputs (pre) and outputs (post) of QuantIdentity.
    """
    if qt is None:
        return None
    s = getattr(qt, "scale", None)
    if s is None:
        return None
    try:
        return float(s.item()) if hasattr(s, "item") else float(s)
    except Exception:
        return None


# --------------------------
# Header writer (2-D array)
# --------------------------
class Residual2DWriter:
    def __init__(self, out_dir, filename="scaleRES.h", array_name="scaleRES"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self.array_name = array_name
        if os.path.exists(self.path):
            os.remove(self.path)
        self.rows = 0
        self._opened = False

    def _open_if_needed(self):
        if self._opened:
            return
        with open(self.path, "a") as f:
            f.write("// Auto-generated from quantized model\n")
            f.write("// scaleRES[layerID][2] = { x_shift, fn_shift }\n")
            f.write("// Each entry is the CODE shift k to map s_in -> s_q:\n")
            f.write("//   k = round(log2(s_q / s_in))\n")
            f.write("//   k > 0 : code RIGHT shift by k  (>>)  [s_q > s_in]\n")
            f.write("//   k < 0 : code LEFT  shift by |k| (<<) [s_q < s_in]\n")
            f.write(f"static const int {self.array_name}[][2] = {{\n")
        self._opened = True

    def add_row(self, x_shift: int, fn_shift: int, comment: str = ""):
        self._open_if_needed()
        with open(self.path, "a") as f:
            if comment:
                f.write(f"  /* layer {self.rows}: {comment} */ ")
            else:
                f.write(f"  /* layer {self.rows} */ ")
            f.write(f"{{ {int(x_shift)}, {int(fn_shift)} }},\n")
        self.rows += 1

    def close(self):
        if not self._opened:
            with open(self.path, "a") as f:
                f.write(f"static const int {self.array_name}[][2] = {{}};\n")
                f.write(f"static const int {self.array_name}_ROWS = 0;\n")
            return
        with open(self.path, "a") as f:
            f.write("};\n")
            f.write(f"static const int {self.array_name}_ROWS = {self.rows};\n")


# --------------------------
# Residual hooks
# --------------------------
def attach_residual_hooks(model: nn.Module, output_dir="./output"):
    """
    Targets modules that look like your Residual wrapper:
      - attribute `input_quant` (QuantIdentity), called twice per forward
      - attribute `fn` (Module)
    For each `input_quant` call we record:
        s_in  = scale of the *input* QuantTensor
        s_q   = scale of the *output* QuantTensor
    Then per Residual we compute CODE shifts:
        k_x  = round(log2(s_qx  / s_x ))
        k_fn = round(log2(s_qfn / s_fn))
    and write a row { k_x, k_fn }.
    """
    writer = Residual2DWriter(output_dir)
    name_of = {m: n for n, m in model.named_modules()}

    buffers = {}
    hooks = []

    # Detect Residual modules structurally
    residuals = []
    for m in model.modules():
        if (
            hasattr(m, "input_quant")
            and isinstance(m.input_quant, qnn.QuantIdentity)
            and hasattr(m, "fn")
            and isinstance(m.fn, nn.Module)
            and m.__class__.__name__.lower().startswith("residual")
        ):
            residuals.append(m)

    def make_iq_hook(resid_mod):
        buf = buffers[resid_mod]

        def iq_hook(module, inputs, output):
            in_qt = inputs[0] if (isinstance(inputs, tuple) and len(inputs) > 0) else None
            s_in = _scale_of(in_qt)
            s_q = _scale_of(output)
            buf.append((s_in, s_q))
        return iq_hook

    def resid_hook(module, inputs, output):
        # Expect two entries appended in order of calls:
        #   first = fn(x) path, second = x path  (per your Residual.forward)
        name = name_of.get(module, "<Residual>")
        buf = buffers[module]

        s_in_fn, s_q_fn = buf.popleft() if len(buf) else (None, None)
        s_in_x,  s_q_x  = buf.popleft() if len(buf) else (None, None)

        def code_shift(s_in, s_q):
            if s_in is None or s_q is None or s_in <= 0 or s_q <= 0:
                return 0
            # Signed CODE shift: +k => >>k,  -k => <<|k|
            return int(round(math.log2(s_q / s_in)))

        k_fn = code_shift(s_in_fn, s_q_fn)
        k_x  = code_shift(s_in_x,  s_q_x)

        writer.add_row(k_x, k_fn, comment=name)

    # Register hooks
    for resid in residuals:
        buffers[resid] = deque()
        hooks.append(resid.input_quant.register_forward_hook(make_iq_hook(resid)))
        hooks.append(resid.register_forward_hook(resid_hook))

    def cleanup():
        writer.close()
        for h in hooks:
            h.remove()

    return cleanup

def residual_scale_extractor(device, model_dir):
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)
    dummy = generate_input(config, device)

    # Warm-up pass to build quantized graph
    _ = model(dummy)

    # Collect residual shifts on another pass
    out_dir = os.path.join(model_dir, "weights")
    cleanup = attach_residual_hooks(model, out_dir)
    _ = model(dummy)
    cleanup()

    print(f"[OK] Wrote residual shifts to: {os.path.join(out_dir, 'scaleRES.h')}")


