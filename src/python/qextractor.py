import argparse

import torch

from extractors.conv_scale_extractor_cpp import conv_scale_extractor
from extractors.residual_scale_extractor_cpp import residual_scale_extractor
from extractors.skip_scale_extractor_cpp import skip_scale_extractor
from extractors.weights_extractor_cpp import weight_extractor


def parse_args():
    p = argparse.ArgumentParser("Extract all quantized parameters")
    p.add_argument("--model_dir", type=str, required=True,
                   help="Trained model directory (needs files: model.pth, config.yml)")
    return vars(p.parse_args())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    model_dir = args["model_dir"]
    weight_extractor(device, model_dir)
    conv_scale_extractor(device, model_dir)
    skip_scale_extractor(device, model_dir)
    residual_scale_extractor(device, model_dir)


if __name__ == "__main__":
    main()
