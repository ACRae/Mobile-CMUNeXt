from argparse import ArgumentParser
import os

from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
from torchinfo import summary
import yaml

from network import networks


def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)

def parse_args():
    parser = ArgumentParser()
    # base
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory (needs files: model.pth, config.yml)",
        required=True,
    )
    return vars(parser.parse_args())


def main():
    args = parse_args()
    model_dir = args["model_dir"]
    config = load_yaml(os.path.join(model_dir, "config.yml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = networks.get_model(config["model"], config).to(device)

    input_size = (
        config["batch_size"],
        config["input_channels"],
        config["input_w"],
        config["input_h"],
    )

    print(f"Input shape: {input_size}")

    # ---- TorchInfo Summary ----
    summary(model, input_size=input_size, depth=8)

    # ---- FVCore MACs per-layer ----
    dummy_input = torch.randn(*input_size).to(device)
    flops = FlopCountAnalysis(model, dummy_input)

    print("\n🔍 MACs per layer (from fvcore):")
    print(flop_count_table(flops, max_depth=8, show_param_shapes=False))

    # Optionally print the model architecture
    # print("\n🧠 Model architecture:")
    # print(model)


if __name__ == "__main__":
    """
    Example usage: python profiler.py --model Mobile-CMUNeXt-RELU
    """
    main()
