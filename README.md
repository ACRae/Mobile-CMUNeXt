# Mobile-CMUNeXt - Semantic Segmentation of Medical Images for Fast Diagnosis

Real-time semantic segmentation of medical images with FPGA acceleration.

This repository provides the complete software and hardware stack for real-time semantic segmentation of medical images, leveraging lightweight neural networks, quantization-aware training, and custom FPGA accelerators. It is the official implementation for the associated Master's thesis.

## Thesis Information

| Field | Details |
|-------|---------|
| **Title** | Semantic Segmentation of Medical Images for Fast Diagnosis |
| **Author** | António Maria Ferreira de Oliveira Carvalho |
| **Degree** | M.Sc. Computer Science and Engineering (*Engenharia Informática e de Computadores*) |
| **Institution** | Instituto Superior de Engenharia de Lisboa (ISEL), Instituto Politécnico de Lisboa (IPL) |
| **Supervisor** | Dr. Mário Pereira Véstias |
| **Date** | December 2025 |

## Publication

This work has been published in:

> **A. Carvalho, M. Véstias**, "Fast Semantic Segmentation of Medical Images," *Proceedings of the 9th International Young Engineers Forum on Electrical and Computer Engineering (YEF-ECE)*, Caparica/Lisbon, Portugal, 2025, pp. 56–61. DOI: 10.1109/YEF-ECE66503.2025.11117287

## System Overview

Mobile-CMUNeXt implements a complete hardware–software co-design pipeline:

```
Training (PyTorch) 
    ↓
Quantization-Aware Training (Brevitas)
    ↓
Model Export (Fixed-point parameters)
    ↓
HLS Synthesis (Vitis HLS)
    ↓
FPGA Deployment (Vivado)
    ↓
Real-time Inference (ZU3EG MPSoC)
```

**Target Hardware:** Avnet Ultra96-V2 (Zynq UltraScale+ ZU3EG MPSoC)


## Repository Structure

```
├── docs/                  # Documentation and notes (hardware, software, configs)
│   ├── Mobile-CMUNeXt/    # Thesis manuscript and related docs
│   └── xilinx/            # Xilinx/Vitis/Vivado notes and guides
│
├── network/
│   └── train/             # Training logs and checkpoints (per dataset)
│
├── src/
│   ├── python/            # Main Python source code
│   │   ├── *.py           # Training, inference, quantization, utilities
│   │   ├── dataloader/    # Data loading modules
│   │   ├── extractors/    # Quantization/scale extractors
│   │   ├── infere/        # Inference input/output
│   │   ├── network/       # Network architectures (incl. Mobile-CMUNeXt)
│   │   ├── optimizers/    # Custom optimizers and schedulers
│   │   ├── plot_net/      # Network plotting utilities
│   │   ├── profilers/     # Model profiling tools
│   │   ├── quantized_images/ # Quantized image utilities
│   │   ├── results/       # Results, exports, and notebooks
│   │   └── utils/         # Losses, metrics, general utilities
│   │
│   ├── scripts/               # Top-level automation and benchmarking scripts
│   │   ├── *.csv              # Experiment and network configuration tables
│   │   ├── train.sh           # Training automation
│   │   ├── dataset/           # Dataset download scripts
│   │   ├── logs/              # Log files
│   │   └── train/             # Training script wrappers
│   │
│   ├── vitis/             # FPGA implementation (Vitis HLS, app, packages)
│   │   ├── Makefile
│   │   ├── app/           # Vitis app configuration
│   │   ├── hls/           # HLS accelerator cores (C/C++)
│   │   ├── include/       # HLS headers
│   │   └── packages/      # Vitis packages
│   │
│   └── vivado/            # Vivado FPGA project files and designs
│       ├── mc-fives.xpr
│       └── design/
│
├── LICENSE
├── README.md
```

### Key Folders

- `src/python/` — All core Python code for training, quantization, inference, and utilities
- `src/scripts/` — Helper scripts for datasets, training, and automation
- `src/vitis/` — Vitis HLS and app code for FPGA accelerator implementation
- `src/vivado/` — Vivado project files for hardware deployment
- `docs/` — Documentation, notes, and thesis manuscript
- `network/train/` — Training logs and results (organized by dataset)

## Quick Start


### Prerequisites

- Python 3.11+ (required)
- PyTorch 1.9+
- Brevitas (for quantization-aware training)
- Xilinx Vivado 2022.1+ and Vitis HLS (for hardware synthesis)

---

### 1. Clone the Repository

```bash
git clone https://github.com/ACRae/Mobile-CMUNeXt.git
cd Mobile-CMUNeXt
```


### 2. Prepare Python Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv .env
source .env/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r src/python/requirements.txt
```

---


### 3. Prepare Datasets

Organize your datasets as follows (outside the repo or in a `data/` folder):

```
data/
├── BUSI/
├── ISIC2016/
└── FIVES2022/
```

Each dataset should contain images and corresponding masks. Update paths in your training/inference commands as needed.

You can use the dataset download scripts in `src/scripts/dataset/`:

```bash
bash src/scripts/dataset/download_busi_2019.sh
bash src/scripts/dataset/download_fives_2022.sh
bash src/scripts/dataset/download_isic_2016.sh
```

---


### 4. Train a Floating-Point Model

To train a full-precision model (from `src/python/`):

```bash
cd src/python
python main.py \
  --model Mobile-CMUNeXt-RELU \
  --data_dir ../../data \
  --dataset_name ISIC2016 \
  --img_ext .jpg \
  --mask_ext .png
```

Adjust `--data_dir`, dataset name, and extensions as needed.

---


### 5. Train a Quantized Model

To train a quantized model:

```bash
cd src/python
python main.py \
  --model Mobile-CMUNeXt-Quant-RELU-BN-ACT \
  --data_dir ../../data \
  --dataset_name FIVES2022 \
  --act_bit_width 4 \
  --weight_bit_width 4
```

Adjust bit widths and dataset as needed.

---


### 6. Export Quantized Weights and Scales for FPGA

Export quantized weights and scales for hardware deployment:

```bash
cd src/python
python qextractor.py \
  --model_dir <TRAINED_MODEL_DIR>
```



## Key Features

- Lightweight, efficient neural network for embedded/FPGA deployment
- Quantization-aware training for fixed-point hardware
- Custom HLS accelerators (pointwise, depthwise, 3D conv)
- Multi-dataset support (BUSI, ISIC, FIVES)
- End-to-end reproducibility: training → quantization → hardware


## Reproducibility

All experiments in the thesis can be reproduced:

- Training configs and hyperparameters are provided in scripts and code
- Pre-trained weight exports match thesis results
- FPGA synthesis requires Xilinx Vivado 2022.1+ and Vitis HLS
- Dataset download scripts included in `src/scripts/dataset/`


## Documentation

- See `docs/` for thesis, hardware/software notes, and configuration guides
- Inline documentation and example usage in each module


## License & Citation

This repository is provided for **academic and research use only**.

If you use this work, please cite:

```bibtex
@inproceedings{carvalho2025mobile,
  author={Carvalho, A. and V\'{e}stias, M.},
  title={Fast Semantic Segmentation of Medical Images},
  booktitle={Proceedings of the 9th International Young Engineers Forum on Electrical and Computer Engineering (YEF-ECE)},
  year={2025},
  pages={56--61},
  doi={10.1109/YEF-ECE66503.2025.11117287}
}
```


## Contact

For questions or issues:

- Open an issue on GitHub
- Contact: Instituto Superior de Engenharia de Lisboa (ISEL)
- Author: António Maria Ferreira de Oliveira Carvalho
