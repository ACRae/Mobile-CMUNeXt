from glob import glob
import os
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from dataloader.dataset.DatasetSplit import DatasetSplit


class MedicalDataSetsSplit(Dataset):
    def __init__(
        self,
        data_dir="./data",
        dataset_name="isic",
        img_ext=".png",
        mask_ext=".png",
        split: DatasetSplit = DatasetSplit.TRAIN,
        num_classes=1,
        transform=None,
    ):
        """
        Args:
            data_dir: Directory to data/ folder
            dataset_name: Name of the dataset (should match the folder name)
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            validation: Flag to indicate if it's a validation dataset.
            transform (Compose, optional): Albumentations transforms. Defaults to None.

        Folder structure:
            <dataset_name>
            └──train
               ├──images
               └──masks
                   └──0
                   └──1
                   └──...
                   └──<num classes-1>
        """
        self._base_dir = os.path.join(data_dir, dataset_name)
        self.img_dir = os.path.join(self._base_dir, "train", "images")
        self.mask_dir = os.path.join(self._base_dir, "train", "masks")
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        # Verify folder structure and raise error if invalid
        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Directory structure not found: {self.img_dir} or {self.mask_dir}")

        if split == DatasetSplit.TRAIN:
            with open(os.path.join(self._base_dir, f"{dataset_name}_train.txt")) as f1:
                self.img_ids = f1.readlines()
            self.img_ids = [item.replace("\n", "") for item in self.img_ids]

        elif split == DatasetSplit.VAL:
            with open(os.path.join(self._base_dir, f"{dataset_name}_val.txt")) as f:
                self.img_ids = f.readlines()
            self.img_ids = [item.replace("\n", "") for item in self.img_ids]
        else:
            raise ValueError("Invalid split value!")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        case = self.img_ids[idx]

        img_path = os.path.join(self.img_dir, case + self.img_ext)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        mask = []
        for i in range(self.num_classes):
            mask_class_dir = os.path.join(self.mask_dir, str(i))
            mask_file = os.path.join(mask_class_dir, case + self.mask_ext)

            if os.path.isfile(mask_file):
                mask.append(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)[..., None])
            else:
                matching_mask = glob(os.path.join(mask_class_dir, f"*{case}*{self.mask_ext}"))
                if matching_mask:
                    mask.append(cv2.imread(matching_mask[0], cv2.IMREAD_GRAYSCALE)[..., None])
                else:
                    raise FileNotFoundError(f"Mask not found for case: {case} in {mask_class_dir}")

        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        mask = mask.astype(np.float32) / 255.0
        mask = mask.transpose(2, 0, 1)

        return {"image": img, "label": mask, "case": case}

    @staticmethod
    def dataset_split(data_dir="./data", dataset_name="isic", img_ext=".png", val_size=0.3):
        """
        Splits the dataset into training and validation sets and stores the results in temporary
        files.

        Args:
            data_dir: Root directory of the dataset.
            dataset_name: Name of the dataset.
            img_ext: Image file extension (default: .png).
            val_size: Validation set size as a fraction of the total dataset.
        """
        root = os.path.join(data_dir, dataset_name)

        img_ids = glob(os.path.join(root, "train", "images", "*" + img_ext))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        assert len(img_ids) > 0, "No images found! Maybe check image extension [--img_ext]"

        train_img_ids, val_img_ids = train_test_split(
            img_ids, test_size=val_size, random_state=random.randint(0, 1024)
        )

        train_file_path = os.path.join(root, f"{dataset_name}_train.txt")
        val_file_path = os.path.join(root, f"{dataset_name}_val.txt")

        if not os.path.exists(train_file_path) or not os.path.exists(train_file_path):
            with open(train_file_path, "w") as train_file:
                for img_id in train_img_ids:
                    train_file.write(img_id + "\n")
                print(f"Built train file successfully, path is: {train_file_path}")

            with open(val_file_path, "w") as val_file:
                for img_id in val_img_ids:
                    val_file.write(img_id + "\n")
                print(f"Built validation file successfully, path is: {val_file_path}")
