import os

from albumentations import Compose, Flip, Normalize, RandomRotate90, Resize
from torch.utils.data import DataLoader

from dataloader.dataset.DatasetSplit import DatasetSplit
from dataloader.dataset.MedicalDatasets import MedicalDataSets
from dataloader.dataset.MedicalDatasetsSplit import MedicalDataSetsSplit


datasets = {
    "ISIC2016": MedicalDataSetsSplit,
    "ISIC2017": MedicalDataSets,
    "FIVES2022": MedicalDataSetsSplit,
}


def get_transform(config, dataset_split=DatasetSplit.TRAIN):
    if dataset_split is DatasetSplit.TRAIN:
        return Compose(
            [
                RandomRotate90(),
                Flip(),
                Resize(config["input_h"], config["input_w"]),
                Normalize(),  # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
            ]
        )
    return Compose(
        [
            Resize(config["input_h"], config["input_w"]),
            Normalize(),
        ]
    )


def get_dataloaders(config):
    dataset_root_train = os.path.join(config["data_dir"], config["dataset_name"], "train")
    if not (os.path.exists(dataset_root_train)) and len(os.listdir(dataset_root_train)) == 0:
        raise ValueError("No train dataset folder/data found!")

    dataset_class = datasets.get(config["dataset_name"], MedicalDataSetsSplit)

    if dataset_class == MedicalDataSetsSplit:
        MedicalDataSetsSplit.dataset_split(
            config["data_dir"], config["dataset_name"], config["img_ext"], val_size=0.3
        )

    db_train = dataset_class(
        data_dir=config["data_dir"],
        dataset_name=config["dataset_name"],
        transform=get_transform(config, dataset_split=DatasetSplit.TRAIN),
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        split=DatasetSplit.TRAIN,
    )

    trainloader = DataLoader(
        db_train,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    db_val = dataset_class(
        data_dir=config["data_dir"],
        dataset_name=config["dataset_name"],
        transform=get_transform(config, dataset_split=DatasetSplit.VAL),
        img_ext=config["img_ext"],
        mask_ext=config["mask_ext"],
        num_classes=config["num_classes"],
        split=DatasetSplit.VAL,
    )

    valloader = DataLoader(
        db_val,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    print(f"Train size: {len(db_train)}, Validation size: {len(db_val)}")

    return trainloader, valloader
