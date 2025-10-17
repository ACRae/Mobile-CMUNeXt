import argparse
import os
import zipfile

import requests
from tqdm import tqdm


def str2bool(v):
    if v.lower() in ["true", 1]:
        return True
    if v.lower() in ["false", 0]:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def download_file(url, target_dir):
    """Download file from the given URL to the target directory with tqdm progress bar."""
    os.makedirs(target_dir, exist_ok=True)
    local_file_path = os.path.join(target_dir, os.path.basename(url))

    # Download the file with progress bar
    with requests.get(url, stream=True, timeout=5000) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        with open(local_file_path, "wb") as f, tqdm(
            desc=f"Downloading {os.path.basename(url)}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in r.iter_content(chunk_size=1024):
                f.write(data)
                bar.update(len(data))

    return local_file_path


def unzip_file(zip_file_path, target_dir, delete_after=True):
    """Unzip the downloaded file to the target directory."""
    print(f"Extracting {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print(f"Extraction complete for {os.path.basename(zip_file_path)}.")

    # Optionally remove the zip file after extraction
    if delete_after:
        os.remove(zip_file_path)
