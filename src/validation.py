from collections import OrderedDict

import torch
from tqdm import tqdm

from utils.metrics import iou_score
from utils.util import AverageMeter


def validate_network(model, valloader, criterion, device, verbose=True):
    avg_meters = {
        "val_loss": AverageMeter(),
        "val_iou": AverageMeter(),
        "val_dice": AverageMeter(),
        "SE": AverageMeter(),
        "PC": AverageMeter(),
        "F1": AverageMeter(),
        "AAC": AverageMeter(),
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(valloader)) if verbose else None
        for sampled_batch in valloader:
            input, target = sampled_batch["image"], sampled_batch["label"]
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = criterion(output, target)

            iou, dice, SE, PC, F1, _, ACC = iou_score(output, target)
            avg_meters["val_loss"].update(loss.item(), input.size(0))
            avg_meters["val_iou"].update(iou, input.size(0))
            avg_meters["val_dice"].update(dice, input.size(0))
            avg_meters["SE"].update(SE, input.size(0))
            avg_meters["PC"].update(PC, input.size(0))
            avg_meters["F1"].update(F1, input.size(0))
            avg_meters["AAC"].update(ACC, input.size(0))

            if verbose:
                postfix = OrderedDict(
                    [
                        ("val_loss", avg_meters["val_loss"].avg),
                        ("val_iou", avg_meters["val_iou"].avg),
                        ("val_dice", avg_meters["val_dice"].avg),
                        ("SE", avg_meters["SE"].avg),
                        ("PC", avg_meters["PC"].avg),
                        ("F1", avg_meters["F1"].avg),
                        ("AAC", avg_meters["AAC"].avg),
                    ]
                )
                pbar.set_postfix(postfix)
                pbar.update(1)
        if verbose:
            pbar.close()

    return OrderedDict(
        [
            ("val_loss", avg_meters["val_loss"].avg),
            ("val_iou", avg_meters["val_iou"].avg),
            ("val_dice", avg_meters["val_dice"].avg),
            ("SE", avg_meters["SE"].avg),
            ("PC", avg_meters["PC"].avg),
            ("F1", avg_meters["F1"].avg),
            ("AAC", avg_meters["AAC"].avg),
        ]
    )
