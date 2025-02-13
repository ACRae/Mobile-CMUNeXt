# Mobile-CMUNeXt: Fast Semantic Segmentation Diagnosis of Medical Images

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