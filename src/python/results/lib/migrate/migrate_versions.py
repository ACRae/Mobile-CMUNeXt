import argparse
from glob import glob
import os
import sys
import traceback

from thop import clever_format
import torch
import yaml

from network import networks
from profilers.params import macs_and_params


def load_yaml(file_path) -> dict:
    with open(file_path) as file:
        return yaml.safe_load(file)


def profile_network(config: dict, device, metrics_path):
    config.pop("MACS")
    config.pop("PARAMS")

    MACS, PARAMS = macs_and_params(
        config["input_channels"],
        config["input_w"],
        config["input_h"],
        model=networks.get_model(config["model"], config).to(device),
        device=device,
    )

    clever_macs, clever_params = clever_format([MACS, PARAMS], "%.3f")

    profile_network = {
        "MACS": MACS,
        "PARAMS": PARAMS,
        "MACS (Formatted)": clever_macs,
        "PARAMS (Formatted)": clever_params,
    }

    with open(os.path.join(metrics_path, "profile.yml"), "w") as f:
        yaml.dump(profile_network, f, sort_keys=False)


def migrate_metrics(metrics: dict, config: dict, device):
    metrics["epoch"] = metrics.pop("epoch")
    metrics["loss"] = metrics.pop("val_loss")
    metrics["iou"] = metrics.pop("val_iou")
    metrics["dice"] = metrics.pop("val_dice")
    metrics["PC"] = metrics.pop("PC")
    metrics["F1"] = metrics.pop("F1")
    metrics["AAC"] = metrics.pop("AAC")


def migrate_files(metrics_path: str, metrics: dict, config: dict):
    old_metrics = os.path.join(metrics_path, "best_metrics.yml")
    os.remove(old_metrics)

    with open(os.path.join(metrics_path, "config.yml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    with open(os.path.join(metrics_path, "metrics.yml"), "w") as f:
        yaml.dump(metrics, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze model metrics and generate CSV reports")
    parser.add_argument("--data", required=True, help="Path to the models directory")
    args = parser.parse_args()

    models_dir = args.data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_path in glob(f"{models_dir}/*/"):
        for dataset_path in glob(f"{os.path.normpath(model_path)}/*/"):

            timestamp_dirs = sorted(glob(f"{dataset_path}/*/"), reverse=True)

            for ts_dir in timestamp_dirs:
                best_file = os.path.join(ts_dir, "best_metrics.yml")
                config_file = os.path.join(ts_dir, "config.yml")
                if os.path.exists(best_file) and os.path.exists(config_file):
                    best_data = load_yaml(best_file)
                    config_data = load_yaml(config_file)
                    try:
                        migrate_metrics(best_data, config_data, device)
                        profile_network(config_data, device, ts_dir)
                        migrate_files(ts_dir, best_data, config_data)
                    except (KeyError, NameError):
                        print("Error!")
                        print(traceback.format_exc())
                        sys.exit()


if __name__ == "__main__":
    main()
