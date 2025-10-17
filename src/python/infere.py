import argparse
from glob import glob
import os

from albumentations import Compose, Normalize, Resize
import cv2
import numpy as np
import torch
import yaml

from network import networks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Trained model directory (needs files: model.pth, config.yml)",
        required=True,
    )
    parser.add_argument("--input", type=str, help="Input file to test the model", required=True)
    parser.add_argument("--mask", type=str, help="Mask file to test the model")
    parser.add_argument("--output_dir", type=str, help="Output directory", default="./")
    parser.add_argument(
        "--recursive", type=bool, help="Recursively infere every model in model_dir", default=False
    )
    parser.add_argument("--dataset", type=str, help="Dataset for recursive inference")

    # NEW: overlay controls
    parser.add_argument(
        "--overlay_style",
        type=str,
        choices=["outline", "fill"],
        default="outline",
        help="How to draw overlay: 'outline' (contours) or 'fill' (alpha-blended regions).",
    )
    parser.add_argument(
        "--outline_thickness",
        type=int,
        default=1,
        help="Contour thickness when overlay_style=outline.",
    )
    parser.add_argument(
        "--fill_alpha",
        type=float,
        default=0.4,
        help="Alpha for filled overlays when overlay_style=fill (0..1).",
    )

    args = parser.parse_args()
    if args.recursive and not args.dataset:
        parser.error("--recursive argument requires --dataset")

    return vars(args)


def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


def preprocess_image(config, image_path):
    transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            Normalize(),
        ]
    )
    img = cv2.imread(image_path)  # BGR
    img = transform(image=img)["image"]
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
    return img


def preprocess_mask(config, mask_path):
    transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            Normalize(),
        ]
    )
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = transform(image=mask)["image"]
    mask = mask.astype(np.float32) / 255.0
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    return mask


def foreground_accuracy(output, mask):
    output = torch.sigmoid(output).cpu().numpy()
    output = (output >= 0.5).astype(np.uint8)

    mask = torch.sigmoid(mask).cpu().numpy()
    mask = (mask >= 0.5).astype(np.uint8)

    if output.shape != mask.shape:
        raise ValueError(f"Mismatched shapes: {output.shape} vs {mask.shape}")

    tp = np.logical_and(output == 1, mask == 1).sum()
    fp = np.logical_and(output == 1, mask == 0).sum()
    fn = np.logical_and(output == 0, mask == 1).sum()

    denominator = tp + fp + fn
    if denominator == 0:
        return float("nan")

    return tp / denominator


def load_model(config, model_pth, device):
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    state_dict = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model.eval()


def _base_output_name(config):
    return "_".join([config["dataset_name"], config["custom_name"] or config["model"]]) + config["mask_ext"]


def save_output(config, output, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output = torch.sigmoid(output).cpu().numpy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0
    output_name = _base_output_name(config)
    output_path = os.path.join(output_dir, output_name)

    for i in range(len(output)):
        for c in range(config["num_classes"]):
            cv2.imwrite(output_path, (output[i, c] * 255).astype("uint8"))


def save_faccuracy(config, acc, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_name = "_".join([config["dataset_name"], config["custom_name"] or config["model"]]) + "_facc.txt"
    output_path = os.path.join(output_dir, output_name)
    with open(output_path, "w") as f:
        f.write(f"Foreground ACC: {acc}\n")


def save_overlay(
    config,
    output,
    input_path,
    mask_path,
    output_dir,
    overlay_style: str = "outline",
    outline_thickness: int = 1,
    fill_alpha: float = 0.4,
):
    """
    Renders ground-truth mask first (green), then prediction (red) on top.
    Works for both 'outline' and 'fill' styles so inference always appears above the mask.
    """
    os.makedirs(output_dir, exist_ok=True)

    H, W = int(config["input_h"]), int(config["input_w"])
    orig = cv2.imread(input_path)  # BGR
    if orig is None:
        return
    base_img = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)

    # Build union prediction foreground [H, W]
    pred = torch.sigmoid(output).detach().cpu().numpy()
    if pred.ndim != 4:
        return
    pred_bin = (pred >= 0.5).astype(np.uint8)
    pred_fore = pred_bin[0].any(axis=0).astype(np.uint8)  # [H, W]

    # Optional GT mask
    gt_fore = None
    if mask_path:
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is not None:
            gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)
            _, gt_thr = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)
            gt_fore = (gt_thr > 0).astype(np.uint8)

    overlay = base_img.copy()

    if overlay_style == "outline":
        def outline_from_binary(bin_mask: np.ndarray, thickness: int = 1) -> np.ndarray:
            if bin_mask is None:
                return None
            bin_u8 = (bin_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            edge = np.zeros_like(bin_u8)
            if contours:
                cv2.drawContours(edge, contours, -1, 255, thickness)
            return edge

        # Draw order: GT first, Pred second (on top)
        gt_edge = outline_from_binary(gt_fore, outline_thickness) if gt_fore is not None else None
        pred_edge = outline_from_binary(pred_fore, outline_thickness)

        if gt_edge is not None:
            overlay[gt_edge > 0] = (0, 255, 0)  # green
        overlay[pred_edge > 0] = (0, 0, 255)    # red

    else:  # 'fill'
        out = overlay.astype(np.float32)

        # Draw order: GT first, Pred second (on top)
        if gt_fore is not None:
            gmask = gt_fore.astype(bool)[..., None]
            green = np.zeros_like(out); green[..., 1] = 255.0
            out = out * (1.0 - fill_alpha * gmask) + green * (fill_alpha * gmask)

        rmask = pred_fore.astype(bool)[..., None]
        red = np.zeros_like(out); red[..., 2] = 255.0
        out = out * (1.0 - fill_alpha * rmask) + red * (fill_alpha * rmask)

        overlay = np.clip(out, 0, 255).astype(np.uint8)

    # Save with _overlay suffix (same base name/extension)
    mask_name = "_".join([config["dataset_name"], config["custom_name"] or config["model"]]) + config["mask_ext"]
    root, ext = os.path.splitext(mask_name)
    overlay_name = f"{root}_overlay{ext}"
    cv2.imwrite(os.path.join(output_dir, overlay_name), overlay)



def infere_model(model_dir, input_path, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha):
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    model_pth = os.path.join(model_dir, "model.pth")
    model = load_model(config, model_pth, device)

    input_tensor = preprocess_image(config, input_path).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    if mask_path:
        mask_tensor = preprocess_mask(config, mask_path).to(device)
        acc = foreground_accuracy(output, mask_tensor)
        print(f"Foreground ACC: {acc}")
        save_faccuracy(config, acc, output_dir=output_dir)

    save_output(config, output=output, output_dir=output_dir)
    save_overlay(
        config,
        output=output,
        input_path=input_path,
        mask_path=mask_path,
        output_dir=output_dir,
        overlay_style=overlay_style,
        outline_thickness=outline_thickness,
        fill_alpha=fill_alpha,
    )


def recursive_infere(data_dir, dataset, input_path, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha):
    for model_path in glob(f"{data_dir}/*/"):
        dataset_dir = os.path.join(model_path, dataset)
        if os.path.exists(dataset_dir):
            timestamp_dirs = sorted(glob(f"{dataset_dir}/*/"), reverse=True)
            if timestamp_dirs:
                ts_dir = timestamp_dirs[0]
                config_file = os.path.join(ts_dir, "config.yml")
                model_file = os.path.join(ts_dir, "model.pth")
                if os.path.exists(config_file) and os.path.exists(model_file):
                    infere_model(ts_dir, input_path, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha)


def infere(data_dir, input_path, dataset, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha, recursive=False):
    if recursive:
        recursive_infere(data_dir, dataset, input_path, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha)
    else:
        infere_model(data_dir, input_path, mask_path, output_dir, device, overlay_style, outline_thickness, fill_alpha)


def main():
    args = parse_args()
    model_dir = args["model_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infere(
        model_dir,
        args["input"],
        args["dataset"],
        args["mask"],
        args["output_dir"],
        device,
        args["overlay_style"],
        args["outline_thickness"],
        args["fill_alpha"],
        recursive=args["recursive"],
    )


if __name__ == "__main__":
    main()

# examples:
# outline (default thickness=1)
# python infere.py --model_dir ../../saved_models/ --input ./test/input/20_A.png --output_dir ./out
# fill with alpha=0.5
# python infere.py --model_dir ../../saved_models/ --input ./test/input/20_A.png --output_dir ./out --overlay_style fill --fill_alpha 0.5
