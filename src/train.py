from collections import OrderedDict

from tqdm import tqdm

from utils.metrics import iou_score
from utils.util import AverageMeter


def train_network(
    model,
    trainloader,
    optimizer,
    criterion,
    scheduler,
    device,
    scheduler_type="batch",
    verbose=True,
):
    model.train()
    avg_meters = {
        "loss": AverageMeter(),
        "iou": AverageMeter(),
    }

    pbar = tqdm(total=len(trainloader)) if verbose else None

    for sampled_batch in trainloader:
        volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
        volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

        outputs = model(volume_batch)
        loss = criterion(outputs, label_batch)
        iou, *_ = iou_score(outputs, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters["loss"].update(loss.item(), volume_batch.size(0))
        avg_meters["iou"].update(iou, volume_batch.size(0))

        # Scheduler update based on the type
        if scheduler_type == "batch":
            scheduler.step()

        if verbose:
            postfix = OrderedDict(
                [
                    ("loss", avg_meters["loss"].avg),
                    ("iou", avg_meters["iou"].avg),
                ]
            )
            pbar.set_postfix(postfix)
            pbar.update(1)

    if verbose:
        pbar.close()

    # If scheduler is epoch-based, call it here (e.g., ReduceLROnPlateau)
    if scheduler_type == "epoch":
        scheduler.step(avg_meters["loss"].avg)

    return OrderedDict([("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)])
