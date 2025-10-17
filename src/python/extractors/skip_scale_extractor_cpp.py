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
        "Extract decoder skip-add CODE shifts: map {xK, dK} to the new x_quant scale"
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
class Skip2DWriter:
    def __init__(self, out_dir, filename="scaleSKIP.h", array_name="scaleSKIP"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, filename)
        self.array_name = array_name
        if os.path.exists(self.path):
            os.remove(self.path)
        self._opened = False
        self.rows = 0

    def _open_if_needed(self):
        if self._opened:
            return
        with open(self.path, "a") as f:
            f.write("// Auto-generated from quantized model\n")
            f.write("// scaleSKIP[layerID][2] = { x_shift, up_shift }\n")
            f.write("// Each entry is the CODE shift k to map s_in -> s_q:\n")
            f.write("//   k = round(log2(s_q / s_in))\n")
            f.write("//   k > 0 : code RIGHT shift by k  (>>)  [s_q > s_in]\n")
            f.write("//   k < 0 : code LEFT  shift by |k| (<<) [s_q < s_in]\n")
            f.write(f"static const int {self.array_name}[][2] = {{\n")
        self._opened = True

    def add_row(self, x_shift: int, up_shift: int, comment: str = ""):
        self._open_if_needed()
        with open(self.path, "a") as f:
            if comment:
                f.write(f"  /* layer {self.rows}: {comment} */ ")
            else:
                f.write(f"  /* layer {self.rows} */ ")
            f.write(f"{{ {int(x_shift)}, {int(up_shift)} }},\n")
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
# Skip-connection hooks
# --------------------------
def attach_skip_hooks(model: nn.Module, output_dir="./output"):
    """
    Write one row per *true* Brevitas QuantConv2d (groups == 1) forward call, in execution order.
      - If the conv is inside an Up_conv* (i.e., within a FusionConv), compute real shifts
        from the two most-recent x_quant calls: {x_shift, up_shift}.
      - Otherwise write {0, 0}.
    All other layers are ignored.
    """
    writer = Skip2DWriter(output_dir)
    name_of = {m: n for n, m in model.named_modules()}
    module_of = dict(model.named_modules())

    # FIFO of (s_in, s_q) captured at x_quant
    qcalls = deque()
    hooks = []

    # ---- 1) hook x_quant ----
    x_quants = []
    for n, m in model.named_modules():
        if n.endswith("x_quant") and isinstance(m, qnn.QuantIdentity):
            x_quants.append(m)
    if not x_quants:
        for m in model.modules():
            if isinstance(m, qnn.QuantIdentity) and getattr(m, "name", "") == "x_quant":
                x_quants.append(m)

    def xq_hook(module, inputs, output):
        in_qt = inputs[0] if (isinstance(inputs, tuple) and len(inputs) > 0) else None
        s_in = _scale_of(in_qt)
        s_q = _scale_of(output)
        qcalls.append((s_in, s_q))

    for xq in x_quants:
        hooks.append(xq.register_forward_hook(xq_hook))

    # ---- 2) hook only 'true' QuantConv2d ----
    def _kernel_gt1(mod) -> bool:
        ks = getattr(mod, "kernel_size", None)
        if ks is None:
            return False
        if isinstance(ks, tuple | list):
            return any(int(k) > 1 for k in ks)
        try:
            return int(ks) > 1
        except Exception:
            return False

    # ---- 2) hook only 'true' QuantConv2d (groups == 1) ----
    def _is_true_quant_conv2d(mod: nn.Module) -> bool:
        return isinstance(mod, qnn.QuantConv2d) and int(getattr(mod, "groups", 1)) == 1 and _kernel_gt1(mod)

    def _is_within_upconv(mod: nn.Module) -> bool:
        """True if this conv is under a FusionConv (i.e., an Up_convK block)."""
        full_name = name_of.get(mod, "")
        parts = full_name.split(".") if full_name else []

        # Heuristic 1: name contains something like Up_conv*
        for p in parts:
            pl = p.lower()
            if pl.startswith(("up_conv", "upconv")):
                return True

        # Heuristic 2: any ancestor is a FusionConv
        for i in range(1, len(parts)):  # parents only
            parent_name = ".".join(parts[:i])
            parent = module_of.get(parent_name)
            if parent is not None and parent.__class__.__name__ == "FusionConv":
                return True
        return False

    def _code_shift(s_in, s_q):
        if s_in is None or s_q is None or s_in <= 0 or s_q <= 0:
            return 0
        return int(round(math.log2(s_q / s_in)))

    def conv_hook(module, inputs, output):
        conv_name = name_of.get(module, "<QuantConv2d>")
        if _is_within_upconv(module):
            # Consume the two x_quant calls corresponding to (xK, dK)
            s_in_x,  s_q_x  = qcalls.popleft() if len(qcalls) else (None, None)
            s_in_up, s_q_up = qcalls.popleft() if len(qcalls) else (None, None)
            k_x  = _code_shift(s_in_x,  s_q_x)
            k_up = _code_shift(s_in_up, s_q_up)
            writer.add_row(k_x, k_up, comment=conv_name)
        else:
            # True QuantConv2d but not inside Up_conv* -> zeros
            writer.add_row(0, 0, comment=conv_name)

    for m in model.modules():
        if _is_true_quant_conv2d(m):
            hooks.append(m.register_forward_hook(conv_hook))

    def cleanup():
        writer.close()
        for h in hooks:
            h.remove()

    return cleanup




def skip_scale_extractor(device, model_dir):
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)
    dummy = generate_input(config, device)

    # Warm-up to build quant graph
    _ = model(dummy)

    out_dir = os.path.join(model_dir, "weights")
    cleanup = attach_skip_hooks(model, out_dir)

    # One pass to record the four skip adds (Up_conv5/4/3/2)
    _ = model(dummy)

    cleanup()
    print(f"[OK] Skip-add shifts written to: {os.path.join(out_dir, 'scaleSKIP.h')}")
