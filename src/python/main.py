import argparse
from collections import OrderedDict
from datetime import datetime
import os
import random

import numpy as np
import pandas as pd
from thop import clever_format
import torch
import yaml

from dataloader.dataloader import get_dataloaders
from network import networks
from optimizers import optimizers, schedulers
from profilers.params import macs_and_params
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


def parse_args():
    parser = argparse.ArgumentParser()

    # base
    parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--batch_size", default=8, type=int, metavar="N", help="mini-batch size (default: 8)")
    parser.add_argument("--description", default=None, type=str)
    parser.add_argument("--seed", default=41, type=int, help="PyTorch/Numpy seed")
    parser.add_argument("--verbose", default=True, type=str2bool, help="Verbose mode")
    parser.add_argument("--output_name", default="saved_models", type=str, help="Model output directory name")
    parser.add_argument("--custom_name", default=None, type=str, help="Custom model name")

    # model
    parser.add_argument("--model", type=str, default="CMUNeXt", choices=networks.__all__, help="model")
    parser.add_argument("--input_channels", default=3, type=int, help="input channels")
    parser.add_argument("--num_classes", default=1, type=int, help="number of classes")
    parser.add_argument("--input_w", default=256, type=int, help="image width")
    parser.add_argument("--input_h", default=256, type=int, help="image height")
    parser.add_argument("--dims", nargs='+', type=int, help="Dimensions of the channels")
    parser.add_argument("--depths", nargs='+', type=int, help="Depth of the blocks")
    parser.add_argument("--kernels", nargs='+', type=int, help="Kernels list")

    # dataset
    parser.add_argument("--data_dir", type=str, default="../../data", help="data dir")
    parser.add_argument("--dataset_name", type=str, default="ISIC2016", help="dataset_name")
    parser.add_argument("--img_ext", default=".png", help="image file extension")
    parser.add_argument("--mask_ext", default=".png", help="mask file extension")

    # optimizers
    parser.add_argument("--optimizer", default="Adam", choices=optimizers.__all__, help="optimizer")
    parser.add_argument("--lr", default=1e-3, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--nesterov", default=False, type=str2bool, help="nesterov")

    # schedulers
    parser.add_argument("--scheduler", default="CosineAnnealingLR", choices=schedulers.__all__)
    parser.add_argument("--T_max", default=1e-3, type=float, metavar="TM")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="minimum learning rate")
    parser.add_argument("--eta_min", default=1e-5, type=float, help="eta minimum learning rate")
    parser.add_argument("--exponent", default=0.9, type=float, help="lambda lr exponent value")
    parser.add_argument("--factor", default=0.1, type=float)
    parser.add_argument("--patience", default=2, type=int)
    parser.add_argument("--milestones", default="1,2", type=str)
    parser.add_argument("--gamma", default=2 / 3, type=float)
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file")
    parser.add_argument(
        "--early_stopping", default=-1, type=int, metavar="N", help="early stopping (default: -1)"
    )
    parser.add_argument("--num_workers", default=8, type=int)

    # Quantization
    parser.add_argument("--act_bit_width", default=8, type=int, help="Quantization Activation Bits")
    parser.add_argument("--weight_bit_width", default=8, type=int, help="Quantization Weight Bits")
    parser.add_argument("--bias_bit_width", default=16, type=int, help="Quantization Bias Bits")

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


def profile_network(config, device):
    MACS, PARAMS = macs_and_params(
        config["input_channels"],
        config["input_w"],
        config["input_h"],
        model=networks.get_non_quant_model(config["model"], config).to(device),
        device=device,
    )

    clever_macs, clever_params = clever_format([MACS, PARAMS], "%.3f")

    profile_network = {
        "MACS": MACS,
        "PARAMS": PARAMS,
        "MACS (Formatted)": clever_macs,
        "PARAMS (Formatted)": clever_params,
    }

    path = config["model_dir"]

    with open(os.path.join(path, "profile.yml"), "w") as f:
        yaml.dump(profile_network, f, sort_keys=False)

    print_dict(profile_network)


def main():
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup(config)
    profile_network(config, device)

    model = networks.get_model(config["model"], config).to(device)
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
