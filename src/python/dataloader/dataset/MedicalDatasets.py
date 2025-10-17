from glob import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataloader.dataset.DatasetSplit import DatasetSplit


class MedicalDataSets(Dataset):
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
            ├── train
            │   ├── images
            │   └── masks
            │       └──0
            │       └──1
            │       └──...
            │       └──<num classes-1>
            ├── test (optional)
            │    ├── images
            │    └── masks
            │        └──0
            │        └──1
            │        └──...
            │        └──<num classes-1>
            └── valdation (optional)
                 ├── images
                 └── masks
                     └──0
                     └──1
                     └──...
                     └──<num classes-1>
        """
        self._base_dir = os.path.join(data_dir, dataset_name)
        self.img_dir = os.path.join(self._base_dir, split.value, "images")
        self.mask_dir = os.path.join(self._base_dir, split.value, "masks")
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

        # Verify folder structure and raise error if invalid
        if not os.path.exists(self.img_dir) or not os.path.exists(self.mask_dir):
            raise ValueError(f"Directory structure not found: {self.img_dir} or {self.mask_dir}")

        self.img_ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in glob(os.path.join(self.img_dir, "*" + self.img_ext))
        ]

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
