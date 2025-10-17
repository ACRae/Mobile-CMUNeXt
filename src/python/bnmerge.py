import argparse
from datetime import datetime
import os

import brevitas.nn as qnn
import brevitas.nn.utils as butils
import torch
from torch import nn
import yaml

from network.networks import get_model


def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory (needs files: model.pth, config.yml)",
        required=True,
    )
    parser.add_argument("--output_dir", type=str, help="Output directory", default="./")
    return vars(parser.parse_args())


def load_model(config, model_pth, device):
    """Load the trained model."""
    model = get_model(model_name=config["model"], config=config).to(device)
    state_dict = torch.load(model_pth, map_location=device)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model.eval()


def merge_bn_with_brevitas(conv, bn, conv_name, bn_name):
    print(f"Merging BN '{bn_name}' into Conv '{conv_name}'")
    butils.merge_bn(conv, bn)

    # Reset and replace BN with Identity to remove it from graph
    bn.reset_parameters()
    with torch.no_grad():
        bn.weight.fill_(1.0)
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)


def fold_bn_recursively(model):
    prev_module = None
    prev_name = None

    for name, module in model.named_children():
        fold_bn_recursively(module)

        if isinstance(prev_module, nn.Conv2d | qnn.QuantConv2d) and isinstance(module, nn.BatchNorm2d):
            merge_bn_with_brevitas(prev_module, module, prev_name, name)
            setattr(model, name, qnn.QuantIdentity(act_quant=None, return_quant_tensor=True))  # Remove BN from model

        prev_module = module
        prev_name = name

    return model


def save_model_pth(model, output_path):
    """Save the updated model state dict."""
    torch.save(model.state_dict(), output_path)


def main():
    args = parse_args()
    model_dir = args["model_dir"]
    output_dir = args["output_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")

    model = load_model(config, model_pth, device)
    model = fold_bn_recursively(model)

    model_name = config["custom_name"] or config["model"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    output_dir = os.path.join(output_dir, "merged_models", model_name, config["dataset_name"], timestamp)
    os.makedirs(output_dir, exist_ok=True)

    output_model_path = os.path.join(output_dir, "model.pth")
    save_model_pth(model, output_model_path)
    print(f"Saved merged model to: {output_dir}")

    config["bn"] = False
    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"Saved new config to: {output_dir}")


if __name__ == "__main__":
    main()

# python bnmerge.py --model_dir
# ../../saved_models/Mobile-CMUNeXt-Quant-RELU-BN-ACT-W8/FIVES2022/2025-05-26_13h54m18s/ --output ./
