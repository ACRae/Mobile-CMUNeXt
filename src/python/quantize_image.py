import argparse
import os
from pathlib import Path

from albumentations import Compose, Normalize, Resize
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import StatsOp
import brevitas.nn as qnn
from brevitas.quant.solver import ActQuantSolver
import cv2
from dependencies import value
import numpy as np
import torch
from torch import nn
import yaml


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseQuant(ExtendedInjector):
    bit_width_impl_type = BitWidthImplType.CONST
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = False
    bit_width = None
    signed = True
    narrow_range = True

    @value
    def quant_type():
        return QuantType.INT


class ActQuant(BaseQuant, ActQuantSolver):
    signed = True


def preprocess_image(config, image_path):
    transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            Normalize(),  # divides by 255 internally with default params
        ]
    )

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure RGB

    img = transform(image=img)["image"]         # H, W, C, float32 in [0,1]
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # [1, H, W, C]
    return img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Folder with images to convert",
        required=True,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Trained model directory (needs files: model.pth, config.yml)"
    )
    return vars(parser.parse_args())


class InputQuant(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=8, return_quant_tensor=True
        )

    def forward(self, x):
        return self.input_quant(x)


def find_images(images_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


def quanttensor_to_int8_bytes(qt):
    """
    Convert a Brevitas QuantTensor (per-tensor scale) into signed int8 bytes (1 byte/value).
    Uses narrow range [-127, 127] to match config.
    """
    v = qt.value.detach()
    s = qt.scale.detach()

    qmin, qmax = -127, 127
    q = torch.round(v / s).clamp_(qmin, qmax).to(torch.int8)
    return q.cpu().numpy().flatten(order="C").tobytes()


def main():
    args = parse_args()
    model_dir = args["model_dir"]

    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")

    images_dir = Path(args["images_dir"]).expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"--images_dir not found or not a directory: {images_dir}")

    # Load quant module once
    state_dict = torch.load(model_pth, map_location=DEVICE, weights_only=True)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k.startswith("input_quant.")}

    quant = InputQuant().to(DEVICE)
    quant.load_state_dict(state_dict=filtered_state_dict, strict=True)
    quant.eval()

    images = find_images(images_dir)
    if not images:
        print(f"No images found in {images_dir}")
        return

    print(f"Processing {len(images)} image(s) from {images_dir} ...")

    for img_path in images:
        try:
            # Preprocess (CPU tensor; keep on CPU so .numpy() works later)
            img = preprocess_image(config, str(img_path))  # [1, H, W, C], float32

            # Save NON-quantized float tensor
            float_out = img_path.with_name(img_path.stem + "_float.bin")
            with open(float_out, "wb") as f:
                f.write(img.numpy().astype(np.float32).flatten(order="C").tobytes())

            # Quantize (can run on CPU or move to GPU first)
            with torch.no_grad():
                qt = quant(img.to(DEVICE))

            # Save QUANTIZED int8 codes (1 byte/value)
            quant_out = img_path.with_name(img_path.stem + "_quant_int8.bin")
            with open(quant_out, "wb") as f:
                f.write(quanttensor_to_int8_bytes(qt))

            print(f"✓ {img_path.name} → {float_out.name} (float32), {quant_out.name} (int8)")

        except Exception as e:
            print(f"✗ Failed on {img_path.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
