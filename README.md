# Mobile-CMUNeXt: Fast Semantic Segmentation for Diagnosis of Medical Images

Semantic segmentation of medical images is a critical task in modern healthcare, enabling precise identification and localization of anatomical structures and pathological regions. This technique classifies image regions at the pixel level, offering a powerful tool for rapid and accurate diagnosis. Despite many advancements in deep learning models, the deployment of these systems in real-world clinical settings is hindered by the computational demands of traditional software implementations, leading to inefficient and slower processing times. Field-Programmable Gate Arrays (FPGAs) offer a promising solution by providing programmable hardware-level acceleration suited for deep learning inference tasks. FPGAs enable parallel processing, low latency, and energy-efficient computation, making them ideal for real-time applications such as tumor detection, organ delineation, and lesion classification.

The objective of this work is to design and implement an efficient and fast semantic segmentation network tailored for deployment on an FPGA. By optimizing both the network architecture and its hardware implementation, we aim to overcome current limitations in computational efficiency and latency.

> [!IMPORTANT]
> This work is still under development

## Requirements
* Python 3.11.10


## Installation

### Creating a Virtual Environment
```bash
python -m venv .env # or python3
source .env/bin/activate
```

### Installing Dependencies
```bash
pip install -r requirements
```

### Linting
```bash
ruff check . --fix
black . --line-length 110
```


## Training
```bash
python main.py --model CMUNeXt-S --data_dir ../data --dataset_name ISIC2016 --img_ext .jpg --mask_ext .png

python main.py --model Mobile-CMUNeXt-Quant --data_dir ../data --dataset_name FIVES2022 --act_bit_width 4 --weight_bit_width 4
```
