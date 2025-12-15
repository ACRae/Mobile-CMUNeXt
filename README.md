# Mobile-CMUNeXt - Semantic Segmentation of Medical Images for Fast Diagnosis

Real-time semantic segmentation of medical images with FPGA acceleration.

This repository contains the complete software and hardware implementation of a Master's thesis combining lightweight neural network design, quantization-aware training, and custom FPGA accelerators for fast medical image diagnosis.

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

### Core Training & Evaluation

- **`training/`** — PyTorch training pipelines
  - Dataset loading and augmentation
  - Floating-point and quantization-aware training
  - Loss functions and training loops

- **`validation/`** — Evaluation frameworks
  - Metric computation (IoU, Dice, pixel accuracy)
  - Multi-dataset benchmarking (BUSI, ISIC, FIVES)

- **`inference/`** — Inference utilities
  - Test-time model execution
  - Segmentation mask generation and visualization

### Models & Architecture

- **`models/`** — PyTorch network implementations
  - Original CMUNeXt
  - Optimized Mobile-CMUNeXt variants
  - Ablation study configurations

- **`datasets/`** — Dataset utilities
  - Medical image loaders (BUSI, ISIC, FIVES)
  - Preprocessing and augmentation pipelines
  - *Note: Dataset files not included; follow download instructions in dataset utilities*

### Quantization & Export

- **`quantization/`** — Quantization-aware training (QAT)
  - Brevitas-based quantizer configuration
  - Batch normalization folding utilities
  - Fixed-point precision tuning

- **`export/`** — Model deployment preparation
  - Fixed-point weight extraction
  - FPGA parameter file generation (C headers)
  - Hardware interface configuration

### FPGA Implementation

- **`hls/`** — High-Level Synthesis accelerator cores
  - Pointwise convolution (C/C++)
  - Depthwise convolution (C/C++)
  - 3D convolution operators
  - Designed for Vitis HLS synthesis

- **`hardware/`** — System-level integration
  - Accelerator interfaces and memory mapping
  - Control logic and data movers
  - Vivado project integration

### Utilities

- **`scripts/`** — Automation and benchmarking
  - Experiment reproducibility scripts
  - Performance profiling tools
  - Hardware synthesis automation

## Quick Start

### Prerequisites

- Python 3.8+, PyTorch 1.9+
- Xilinx Vivado 2022.1+ and Vitis HLS (for hardware)
- Brevitas for quantization-aware training

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

Install dependencies from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The repository requires at minimum **Python 3.11**.

---

### 3. Prepare Datasets

Organize your medical image dataset in a folder structure such as:

```
data/
├── BUSI/
├── ISIC2016/
└── FIVES2022/
```

Each dataset should contain images and corresponding masks. Update paths in your training or inference scripts accordingly.

---

### 4. Train a Floating-Point Model

To train the full-precision Mobile-CMUNeXt or CMUNeXt model:

```bash
python main.py \
    --model Mobile-CMUNeXt-RELU \
    --data_dir ./data \
    --dataset_name ISIC2016 \
    --img_ext .jpg \
    --mask_ext .png
```

This runs training on the specified dataset. Replace `ISIC2016`, image extensions, and other parameters to match your dataset.([GitHub][1])

---

### 5. Train a Quantized Model

The repository supports quantized variants. To train with quantization:

```bash
python main.py \
    --model Mobile-CMUNeXt-Quant-RELU-BN-ACT \
    --data_dir ./data \
    --dataset_name FIVES2022 \
    --act_bit_width 4 \
    --weight_bit_width 4
```

Adjust `--act_bit_width` and `--weight_bit_width` for your target precision.

---

### 6. Export Quantized Weights and Scales for FPGA

```bash
python qextractor \
    --model_dir <TRAINED_MODEL_DIR> 
```


## Key Features

- Lightweight architecture optimized for embedded deployment
- Quantization-aware training for fixed-point FPGA execution
- Custom HLS accelerators for convolution kernels
- Multi-dataset evaluation on BUSI, ISIC, and FIVES
- End-to-end reproducibility from training to hardware deployment

## Reproducibility

All experiments in the thesis can be reproduced using this repository:

- Training configurations and hyperparameters are provided
- Pre-trained weight exports match thesis results
- FPGA synthesis requires Xilinx toolchains (Vivado 2022.1+, Vitis HLS)
- Instructions for dataset acquisition included in `datasets/`

## Documentation

Refer to the thesis manuscript for:

- Detailed architecture descriptions
- Quantization methodology and ablation studies
- Hardware implementation details
- Experimental results and comparisons

Individual modules include inline documentation and example usage.

## License & Citation

This repository is provided for **academic and research use**.

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

For questions about the thesis or repository:

- Open an issue on GitHub
- Contact: Instituto Superior de Engenharia de Lisboa (ISEL)
- Author: António Maria Ferreira de Oliveira Carvalho
