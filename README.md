# Mobile-CMUNeXt

**Mobile-CMUNeXt** is a research and deployment project focused on efficient convolutional neural networks (CNNs) for embedded and FPGA systems.
It combines **Python-based training and quantization** with **hardware acceleration using Xilinx tools** (Vivado and Vitis).

---

## 📂 Project Structure

| Folder | Description |
|--------|--------------|
| `src/` | Source code for both software and hardware implementation |
| `network/` | Network architectures and pre-trained models |
| `docs/` | Documentation and hardware deployment notes |
| `LICENSE` | License file for the project |

---

## 🚀 Getting Started

1. **Set up Python environment**
   ```bash
   cd src/python
   pip install -r requirements.txt

    Train or test the model

    python main.py

    FPGA / Hardware build

        Open Vivado or Vitis from the src/ folder as described in their README files.

🧩 Requirements

    Python 3.11+

    Xilinx Vivado & Vitis 2025.1 (for hardware builds)

🧠 Notes

This project bridges software-level CNN model development with FPGA hardware deployment for optimized inference.


---

### ⚙️ **`Mobile-CMUNeXt/src/README.md`** — *Source Code Overview*

```markdown
# Source Code

This folder contains all **source code** used for both the software (Python) and hardware (FPGA) parts of Mobile-CMUNeXt.

---

## 📁 Structure

| Folder | Purpose |
|---------|----------|
| `python/` | Python scripts for training, quantization, profiling, and inference |
| `vitis/` | Vitis project files for building the hardware accelerator |
| `vivado/` | Vivado design files for FPGA synthesis and implementation |

---

## 🧩 Tip

Start with the `python/` folder to train and evaluate the model, then move to `vitis/` or `vivado/` for hardware acceleration.

