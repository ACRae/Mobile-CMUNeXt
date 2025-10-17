import argparse
from collections import OrderedDict
from datetime import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
import yaml

from dataloader.dataloader import get_dataloaders
from network import networks
from optimizers import optimizers, schedulers
from train import train_network
from utils import losses
from utils.util import str2bool
from validation import validate_network


dir_path = os.path.dirname(os.path.realpath(__file__))


def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    parser.add_argument(
        "--epochs", default=50, type=int, metavar="N", help="number of total epochs to run", required=True
    )
    parser.add_argument(
        "--batch_size", default=8, type=int, metavar="N", help="mini-batch size (default: 8)", required=True
    )
    parser.add_argument("--data_dir", type=str, default="../../data", help="data dir", required=True)
    parser.add_argument("--verbose", default=True, type=str2bool, help="Verbose mode")
    parser.add_argument(
        "--output_name", default="finetune_models", type=str, help="Model output directory name"
    )

    return vars(parser.parse_args())


def print_dict(config):
    print("-" * 20)
    for key in config:
        print(f"{key}: {config[key]}")
    print("-" * 20)
    print()


def print_logs(epoch, total_epcochs, train_log, val_log):
    print(
        f"Epoch [{epoch}/{total_epcochs}] | "
        f"Train Loss: {train_log['loss']:.4f}, Train IoU: {train_log['iou']:.4f} | "
        f"Val Loss: {val_log['val_loss']:.4f}, Val IoU: {val_log['val_iou']:.4f} | "
        f"Val SE: {val_log['SE']:.4f}, Val PC: {val_log['PC']:.4f}, "
        f"Val F1: {val_log['F1']:.4f}, Val ACC: {val_log['AAC']:.4f}"
    )


def setup(config):
    seed_torch(config["seed"])

    config["datetime"] = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    config["model_dir"] = os.path.join(
        dir_path,
        config["output_name"],
        config["custom_name"] or config["model"],
        config["dataset_name"],
        config["datetime"],
    )

    os.makedirs(config["model_dir"], exist_ok=True)
    with open(os.path.join(config["model_dir"], "config.yml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    if config["verbose"]:
        print_dict(config)


def save(model: torch.nn.Module, path: str, data):
    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
    with open(os.path.join(path, "metrics.yml"), "w") as f:
        yaml.dump(data, f, sort_keys=False)
    print("=> saved best model")


def load_model(config, model_pth, device):
    """Load the trained model."""
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    state_dict = torch.load(model_pth, map_location=device)
    model.load_state_dict(state_dict=state_dict, strict=False)
    return model.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    model_dir = args["model_dir"]
    config = load_yaml(os.path.join(model_dir, "config.yml"))
    config.update(args)
    config["model_dir"] = os.path.join(
        model_dir,
        config["output_name"],
        config["custom_name"] or config["model"],
        config["dataset_name"],
        config["datetime"],
    )

    model_pth = os.path.join(model_dir, "model.pth")
    setup(config)

    model = load_model(config, model_pth, device)
    trainloader, valloader = get_dataloaders(config)

    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
            ("val_dice", []),
            ("SE", []),
            ("PC", []),
            ("F1", []),
            ("AAC", []),
            ("best", []),
        ]
    )

    optimizer = optimizers.get_optimizer(config["optimizer"], model_params=model.parameters(), **config)
    criterion = losses.__dict__["BCEDiceLoss"]().to(device)
    scheduler = schedulers.get_scheduler(scheduler_name=config["scheduler"], optmzr=optimizer, **config)

    best_iou = 0
    trigger = 0

    for epoch in range(config["epochs"]):
        print("Epoch [%d/%d]" % (epoch, config["epochs"]))
        train_log = train_network(
            model,
            trainloader,
            optimizer,
            criterion,
            scheduler,
            device,
            verbose=config["verbose"],
        )
        val_log = validate_network(model, valloader, criterion, device, verbose=config["verbose"])
        print_logs(epoch, config["epochs"], train_log, val_log)

        log["epoch"].append(epoch)
        log["lr"].append(config["lr"])
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(val_log["val_loss"])
        log["val_iou"].append(val_log["val_iou"])
        log["val_dice"].append(val_log["val_dice"])
        log["SE"].append(val_log["SE"])
        log["PC"].append(val_log["PC"])
        log["F1"].append(val_log["F1"])
        log["AAC"].append(val_log["AAC"])

        trigger += 1

        if val_log["val_iou"] > best_iou:
            best_iou = val_log["val_iou"]
            log["best"].append(True)
            save_log = {
                "epoch": epoch,
                "loss": float(log["val_loss"][-1]),
                "iou": float(log["val_iou"][-1]),
                "dice": float(log["val_dice"][-1]),
                "PC": float(log["PC"][-1]),
                "F1": float(log["F1"][-1]),
                "AAC": float(log["AAC"][-1]),
            }
            save(model, config["model_dir"], save_log)
            trigger = 0
        else:
            log["best"].append(False)

        pd.DataFrame(log).to_csv(os.path.join(config["model_dir"], "log.csv"), index=False)

        if 0 <= config["early_stopping"] <= trigger:
            print("=> Early stopping")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config["end_datetime"] = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    with open(os.path.join(config["model_dir"], "config.yml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)


if __name__ == "__main__":
    main()


# python finetuning.py --model_dir
# ./merged_models/Mobile-CMUNeXt-Quant-RELU-BN-ACT-W8/FIVES2022/2025-05-27_15h17m50s/
# --epochs 10 --batch_size 2 --data_dir ../../data/ --custom_name ...
