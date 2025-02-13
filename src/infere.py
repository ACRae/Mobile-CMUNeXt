import argparse

from albumentations import Compose, Normalize, Resize
import cv2
import numpy as np
import torch

from network import networks
from network.networks import get_model
from utils.metrics import iou_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, help="Trained model pth file", required=True)
    parser.add_argument("--model", type=str, choices=networks.__all__, help="model", required=True)
    parser.add_argument("--test_file", type=str, help="File to test the model", required=True)

    parser.add_argument("--mask_file", type=str, help="File to test the model", required=True)

    parser.add_argument("--output_file", type=str, help="Output segmentation mask file", default="output.png")
    parser.add_argument("--input_channels", default=3, type=int, help="Model Input channels")
    parser.add_argument("--num_classes", default=1, type=int, help="Model Number of classes")
    parser.add_argument("--input_w", default=256, type=int, help="Model Image width")
    parser.add_argument("--input_h", default=256, type=int, help="Model Image height")

    # Quantization
    parser.add_argument("--act_bit_width", default=8, type=int, help="Quantization Activation Bits")
    parser.add_argument("--weight_bit_width", default=8, type=int, help="Quantization Weight Bits")
    parser.add_argument("--bias_bit_width", default=16, type=int, help="Quantization Bias Bits")

    return vars(parser.parse_args())


def preprocess_image(config, image_path):
    transform = Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            Normalize(),
        ]
    )
    img = cv2.imread(image_path)  # Ensure the image is in RGB format

    img = transform(image=img)["image"]
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img).unsqueeze(0)
    return img


def metrics(output, mask):
    iou, dice, SE, PC, F1, _, ACC = iou_score(output, mask)
    print(
        f"Val Dice: {dice:.4f} | "
        f"Val IoU: {iou:.4f} | "
        f"Val SE: {SE:.4f}, Val PC: {PC:.4f}, "
        f"Val F1: {F1:.4f}, Val ACC: {ACC:.4f}"
    )


def load_model(config, device):
    """Load the trained model."""
    model = get_model(model_name=config["model"], config=config).to(device)
    state_dict = torch.load(config["model_file"], map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model.eval()


def save_output(config, output, output_path):
    output = torch.sigmoid(output).cpu().numpy()
    output[output >= 0.5] = 1
    output[output < 0.5] = 0

    for i in range(len(output)):
        for c in range(config["num_classes"]):
            cv2.imwrite(output_path, (output[i, c] * 255).astype("uint8"))


def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    input_tensor = preprocess_image(config, config["test_file"]).to(device)
    mask_tensor = preprocess_image(config, config["mask_file"]).to(device)

    with torch.inference_mode():
        output = model(input_tensor)

    metrics(output, mask_tensor)

    save_output(config, output=output, output_path=config["output_file"])
    print(f"Segmentation mask saved to {config['output_file']}.")


if __name__ == "__main__":
    main()
