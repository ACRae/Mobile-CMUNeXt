import argparse
from collections import deque
import math
import os

import brevitas.nn as qnn
import numpy as np
import torch
from torch import nn
import yaml

from network import networks


def setcache(m):
    m.cache_inference_quant_bias = True


def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser()

    # base
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory (needs files: model.pth, config.yml)",
        required=True,
    )
    return vars(parser.parse_args())


def load_model(config, model_pth, device):
    """
    Load the trained model.
    """
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    model.apply(setcache)
    state_dict = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict, strict=True)
    model.to(device)
    return model.eval()


def generate_input(config, device):
    C = config["input_channels"]
    H = config["input_h"]
    W = config["input_w"]
    return torch.zeros(1, C, H, W).to(device)


class ScaleShiftCppBase:
    def __init__(self, output_dir, scale_filename):
        self.scale_name = os.path.splitext(scale_filename)[0]
        os.makedirs(output_dir, exist_ok=True)
        self.scale_file = os.path.join(output_dir, scale_filename)
        if os.path.exists(self.scale_file):
            os.remove(self.scale_file)
        self.s_counter = 0
        self._total_vals = 0

    def write_scale_shift(self, shift):
        # Accept scalar or array-like; write as flat ints
        if isinstance(shift, list | tuple | np.ndarray):
            arr = np.asarray(shift, dtype=np.int32).ravel()
        else:
            arr = np.array([int(shift)], dtype=np.int32)

        with open(self.scale_file, "a") as f:
            if self.s_counter == 0:
                f.write("// Auto-generated from quantized model\n")
                f.write("// Flat list of shift amounts (n_in - n_out)\n")
                f.write(f"static const int {self.scale_name}[] = {{")
            f.write(f"\n/* idx: {self.s_counter}, count: {arr.size} */\n    ")
            f.write(", ".join(str(int(v)) for v in arr) + ",")

        self.s_counter += 1
        self._total_vals += arr.size

    def cleanup(self):
        if self.s_counter > 0:
            with open(self.scale_file, "a") as f:
                f.write("\n};\n")
                f.write(f"// total values: {self._total_vals}\n")

class ScaleShiftPWCpp(ScaleShiftCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "scalePW.h")

class ScaleShiftDWCpp(ScaleShiftCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "scaleDW.h")

class ScaleShift3DCpp(ScaleShiftCppBase):
    def __init__(self, output_dir):
        super().__init__(output_dir, "scale3D.h")


def _scale_of(obj):
    s = getattr(obj, "scale", None)
    if s is None:
        return None
    return float(s.item()) if hasattr(s, "item") else float(s)

def attach_hooks(model: nn.Module, output_dir="./output"):
    scaleDWCpp = ScaleShiftDWCpp(output_dir)
    scalePWCpp = ScaleShiftPWCpp(output_dir)
    scale3DCpp = ScaleShift3DCpp(output_dir)

    # Map module -> qualified name (to spot 'output_quant')
    name_by_module = {m: n for n, m in model.named_modules()}

    # Stack of conv infos to pair with the next quant op
    conv_info_stack = deque()  # each item: {"kind": "DW"/"PW"/"C3D", "s_conv": float or None}

    def conv_hook(module, inputs, output):
        # classify conv
        if module.groups == module.in_channels:
            kind = "DW"
        elif module.kernel_size == (1, 1) and module.groups == 1:
            kind = "PW"
        else:
            kind = "C3D"

        s_conv = _scale_of(output)  # Conv's QuantTensor output scale
        conv_info_stack.append({"kind": kind, "s_conv": s_conv})

    def _write_shift_from(conv_info, s_out):
        kind = conv_info["kind"]
        s_in = conv_info["s_conv"]
        if s_in is None or s_out is None or s_in <= 0 or s_out <= 0:
            shift = 0
        else:
            shift = int(round(math.log2(s_out / s_in)))
        writer = {"DW": scaleDWCpp, "PW": scalePWCpp, "C3D": scale3DCpp}[kind]
        writer.write_scale_shift(shift)

    def relu_hook(module, inputs, output):
        # Pair with the most recent conv
        conv_info = conv_info_stack.pop() if conv_info_stack else None
        if conv_info is None:
            return  # nothing to pair
        s_out = _scale_of(output)  # ReLU output scale
        _write_shift_from(conv_info, s_out)

    def qid_hook(module, inputs, output):
        """
        Only compute shift for the FINAL QuantIdentity (your output quantizer),
        using its scale instead of a ReLU’s.
        """
        name = name_by_module.get(module, "")
        if not (name == "output_quant" or name.endswith(".output_quant")):
            return  # ignore other QuantIdentity modules

        conv_info = conv_info_stack.pop() if conv_info_stack else None
        if conv_info is None:
            return
        s_out = _scale_of(output)  # QuantIdentity output scale
        _write_shift_from(conv_info, s_out)

    # Register hooks
    for m in model.modules():
        if isinstance(m, qnn.QuantConv2d):
            m.register_forward_hook(conv_hook)
        elif isinstance(m, qnn.QuantReLU):
            m.register_forward_hook(relu_hook)
        elif isinstance(m, qnn.QuantIdentity):
            m.register_forward_hook(qid_hook)

    def cleanup():
        # Any conv not followed by ReLU or the final QuantIdentity → shift = 0
        while conv_info_stack:
            conv_info = conv_info_stack.popleft()
            _write_shift_from(conv_info, s_out=None)
        scaleDWCpp.cleanup()
        scalePWCpp.cleanup()
        scale3DCpp.cleanup()

    return cleanup


def conv_scale_extractor(device, model_dir):
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)
    input_ = generate_input(config, device)

    output_dir = os.path.join(model_dir, "weights")

    model(input_)

    cleanup = attach_hooks(model, output_dir)
    model(input_)
    cleanup()
