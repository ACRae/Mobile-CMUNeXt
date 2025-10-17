import argparse
import os

import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor
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


def fixed_point_info(scale_tensor, bit_width, signed=True):
    """Calculate fixed-point format (integer bits, fractional bits) from scale and bit width."""
    # Handle tensor or scalar scale values
    scale_val = scale_tensor.item() if hasattr(scale_tensor, "item") else float(scale_tensor)

    # Handle edge case where scale is 0
    if scale_val == 0:
        int_bits = 0
        frac_bits = bit_width - (1 if signed else 0)
        return int_bits, frac_bits

    # Calculate maximum representable value
    max_val = (2 ** (bit_width - 1 if signed else bit_width)) - 1

    # Calculate integer bits needed to represent the maximum scaled value
    max_scaled_val = max_val * scale_val
    int_bits = max(0, int(np.floor(np.log2(max_scaled_val))) + 1)

    # Calculate fractional bits
    frac_bits = (bit_width - (1 if signed else 0)) - int_bits

    # Ensure fractional bits are non-negative
    if frac_bits < 0:
        int_bits = bit_width - (1 if signed else 0)
        frac_bits = 0

    return int_bits, frac_bits


def make_named_hook(name_map, log_file):
    def debug_hook(module, input, output):
        data = name_map.get(module, "Unnamed")
        name = data["name"]
        weight_quant = data["weight_quant"]
        bias_quant = data["bias_quant"]
        with open(log_file, "a") as f:
            print(f"\nModule: {name} ({module.__class__.__name__})", file=f)

            if isinstance(module, qnn.QuantConv2d):  # Quant Tensor
                tensor = input[0]
                scale = getattr(tensor, "scale", None)
                signed = getattr(tensor, "signed_t", None)
                bit_width = getattr(tensor, "bit_width", None)

                int_bits, frac_bits = fixed_point_info(
                    scale.item() if hasattr(scale, "item") else scale,
                    bit_width.item() if hasattr(bit_width, "item") else bit_width,
                    signed=signed,
                )
                print(
                    f"  * Quant Input shape: {tuple(tensor.shape)}, "
                    f"bit_width={bit_width.item() if hasattr(bit_width, 'item') else bit_width}, "
                    f"fixed-point format=Q{int(int_bits)}.{int(frac_bits)}, "
                    f"scale={scale.item() if hasattr(scale, 'item') else scale}, "
                    f"signed={signed}",
                    file=f,
                )
            else:
                if isinstance(input, tuple):
                    input = input[0]

                print(f"  * Input shape: {input.shape}", file=f)

            if weight_quant:
                quant_tensor = weight_quant
                scale = getattr(quant_tensor, "scale", None)
                signed = getattr(quant_tensor, "signed_t", None)
                # weights = getattr(quant_tensor, "value", None)
                bit_width = getattr(quant_tensor, "bit_width", None)

                int_bits, frac_bits = fixed_point_info(
                    scale.item() if hasattr(scale, "item") else scale,
                    bit_width.item() if hasattr(bit_width, "item") else bit_width,
                    signed=signed,
                )
                print(
                    f"  * Quant Weight shape: {tuple(quant_tensor.shape)}, "
                    f"bit_width={bit_width.item() if hasattr(bit_width, 'item') else bit_width}, "
                    f"fixed-point format=Q{int(int_bits)}.{int(frac_bits)}, "
                    f"scale={scale.item() if hasattr(scale, 'item') else scale}, "
                    f"signed={signed}",
                    file=f,
                )
            elif hasattr(module, "weight"):
                print(f"  * Weight shape: {input.shape}", file=f)

            if bias_quant:
                try:
                    quant_tensor = bias_quant
                    scale = getattr(quant_tensor, "scale", None)
                    signed = getattr(quant_tensor, "signed_t", None)
                    bit_width = getattr(quant_tensor, "bit_width", None)

                    int_bits, frac_bits = fixed_point_info(
                        scale.item() if hasattr(scale, "item") else scale,
                        bit_width.item() if hasattr(bit_width, "item") else bit_width,
                        signed=signed,
                    )
                    print(
                        f"  * Quant Bias shape: {tuple(quant_tensor.shape)}, "
                        f"bit_width={bit_width.item() if hasattr(bit_width, 'item') else bit_width}, "
                        f"fixed-point format=Q{int(int_bits)}.{int(frac_bits)}, "
                        f"scale={scale.item() if hasattr(scale, 'item') else scale}, "
                        f"signed={signed}",
                        file=f,
                    )
                except RuntimeError as e:
                    print(f"  * Skipping bias quant info: {e}", file=f)

            if isinstance(output, QuantTensor):  # Quant Tensor
                tensor = getattr(output, "value", None)
                scale = getattr(output, "scale", None)
                signed = getattr(output, "signed_t", None)
                bit_width = getattr(output, "bit_width", None)

                int_bits, frac_bits = fixed_point_info(
                    scale.item() if hasattr(scale, "item") else scale,
                    bit_width.item() if hasattr(bit_width, "item") else bit_width,
                    signed=signed,
                )
                print(
                    f"  * Quant Output shape: {tuple(tensor.shape)}, "
                    f"bit_width={bit_width.item() if hasattr(bit_width, 'item') else bit_width}, "
                    f"fixed-point format=Q{int(int_bits)}.{int(frac_bits)}, "
                    f"scale={scale.item() if hasattr(scale, 'item') else scale}, "
                    f"signed={signed}",
                    file=f,
                )
            else:
                if isinstance(output, tuple):
                    output = output[0]
                print(f"  * Output shape: {output.shape}", file=f)

    return debug_hook


def preprocess_model_data(model: nn.Module):
    """
    Returns a dict mapping module object to its fully qualified name.
    """
    data = {}
    for name, module in model.named_modules():
        quant_weight = module.quant_weight() if hasattr(module, "weight_quant") else None
        quant_bias = (
            module.quant_bias()
            if hasattr(module, "bias_quant") and hasattr(module, "bias") and module.bias is not None
            else None
        )
        data[module] = {"name": name, "weight_quant": quant_weight, "bias_quant": quant_bias}
    return data


def attach_hooks(model: nn.Module, target_classes=(qnn.QuantConv2d,), log_file="model_debug_log.txt"):
    model_data = preprocess_model_data(model)
    hook_fn = make_named_hook(model_data, log_file)

    open(log_file, "w").close()  # Clear the log file at the start

    for module in model_data:
        if isinstance(module, target_classes) and not isinstance(module, nn.Sequential):
            module.register_forward_hook(hook_fn)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    model_dir = args["model_dir"]
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)
    input_ = generate_input(config, device)

    output_path = os.path.join(model_dir, "model_description.txt")

    model(input_)

    target_classes = (
        qnn.QuantConv2d,
        qnn.QuantReLU,
        qnn.QuantUpsamplingBilinear2d,
        qnn.QuantIdentity
    )# MODIFY IF NEEDED

    attach_hooks(model, target_classes, output_path)
    model(input_)


if __name__ == "__main__":
    main()
