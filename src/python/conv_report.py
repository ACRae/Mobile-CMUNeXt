#!/usr/bin/env python3
# export_conv_report.py
import argparse, os
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
import pandas as pd
import yaml

import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor

# your project model factory
from network import networks


def setcache(m):
    m.cache_inference_quant_bias = True


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config: dict, model_pth: str, device: torch.device) -> nn.Module:
    model = networks.get_model(model_name=config["model"], config=config).to(device)
    model.apply(setcache)
    state_dict = torch.load(model_pth, map_location=device, weights_only=True)
    model.load_state_dict(state_dict=state_dict, strict=True)
    return model.eval()


def make_dummy_input(config: dict, device: torch.device) -> torch.Tensor:
    C, H, W = int(config["input_channels"]), int(config["input_h"]), int(config["input_w"])
    return torch.zeros(1, C, H, W, device=device)


# ---------- helpers ----------

def is_pool_module(m: nn.Module) -> bool:
    cn = m.__class__.__name__
    return isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)) or "MaxPool" in cn


def is_upsample_module(m: nn.Module) -> bool:
    cn = m.__class__.__name__
    return isinstance(m, (nn.Upsample,)) or ("Upsample" in cn) or ("Upsampling" in cn)


def get_io_shapes(inp_obj, out_obj) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    def to_tensor(o):
        if isinstance(o, QuantTensor):
            return o.value
        if isinstance(o, (tuple, list)) and o:
            o = o[0]
        return o
    tin = to_tensor(inp_obj)
    tout = to_tensor(out_obj)
    if not torch.is_tensor(tin) or not torch.is_tensor(tout):
        return tuple(), tuple()
    return tuple(tin.shape), tuple(tout.shape)


def padding_to_tuple(pad) -> Tuple[int, int]:
    if isinstance(pad, tuple):
        return tuple(int(x) for x in pad)
    return (int(pad), int(pad))


def classify_conv(module: nn.Module) -> str:
    """
    Return one of: 'Depthwise', 'Pointwise', 'Convolution3D'
    - Depthwise: groups == in_channels (typical DW conv)
    - Pointwise: kernel 1x1 and groups == 1
    - Convolution3D: kernel 3x3 and groups == 1 (standard 3x3 conv)
    Anything else is mapped to the closest bucket you asked for:
      - if 1x1 but grouped -> treat as Pointwise
      - otherwise treat as Convolution3D (standard conv category)
    """
    # Support Brevitas QuantConv2d and plain Conv2d
    if isinstance(module, (nn.Conv2d, qnn.QuantConv2d)):
        k = module.kernel_size if hasattr(module, "kernel_size") else None
        groups = int(getattr(module, "groups", 1))
        in_ch = int(getattr(module, "in_channels", 0))

        if k == (1, 1):
            # even if grouped, you asked to label by role; we call it pointwise
            return "Pointwise"
        if groups == in_ch and in_ch > 0:
            return "Depthwise"
        if k == (3, 3) and groups == 1:
            return "Convolution3D"
        # fallback to your 3×3 bucket for other kernels
        return "Convolution3D"

    # If someone actually uses nn.Conv3d / QuantConv3d, also call it Convolution3D
    if isinstance(module, (nn.Conv3d, getattr(qnn, "QuantConv3d", tuple()))):
        return "Convolution3D"

    return "Convolution3D"


def find_prev(exe_log: List[nn.Module], pred) -> nn.Module | None:
    """
    Scan backward (before the current module which is at exe_log[-1]) to find the
    nearest earlier module matching 'pred'.
    """
    # Current module is at -1; start from -2 backwards
    for i in range(len(exe_log) - 2, -1, -1):
        m = exe_log[i]
        if pred(m):
            return m
    return None


# ---------- main collector ----------

def collect_conv_report(model: nn.Module, x: torch.Tensor) -> List[Dict[str, Any]]:
    name_map = {m: n for n, m in model.named_modules()}

    results: List[Dict[str, Any]] = []
    execution_log: List[nn.Module] = []

    # register a "common" hook on *all* modules to capture true execution order
    common_hooks = []
    for m in model.modules():
        common_hooks.append(m.register_forward_hook(lambda mod, inp, out: execution_log.append(mod)))

    # register a dedicated hook on conv modules to collect rows
    conv_types = (nn.Conv2d, qnn.QuantConv2d, nn.Conv3d)
    conv_hooks = []

    def conv_hook(mod, inp, out):
        # 'execution_log' already has 'mod' appended by the common hook,
        # so the *previous* executed module (if any) is at index -2.
        name = name_map.get(mod, "(unknown)")
        ctype = classify_conv(mod)

        in_shape, out_shape = get_io_shapes(inp[0] if isinstance(inp, tuple) else inp, out)
        wshape = tuple(mod.weight.shape) if getattr(mod, "weight", None) is not None else tuple()
        bshape = tuple(mod.bias.shape) if getattr(mod, "bias", None) is not None else tuple()
        pad = padding_to_tuple(getattr(mod, "padding", (0, 0)))

        prev_pool = find_prev(execution_log, is_pool_module) is not None if ctype == "Depthwise" else False
        prev_ups  = find_prev(execution_log, is_upsample_module) is not None if ctype == "Convolution3D" else False

        results.append({
            "Conv Name": name,
            "Type": ctype,
            "Input Dims": in_shape,
            "Output Dims": out_shape,
            "Kernel Dims": wshape,     # (out_c, in_c/groups, kH, kW) for Conv2d
            "Bias Dims": bshape,       # (out_c,) or ()
            "Padding": pad,            # (pad_h, pad_w)
            "Prev Is MaxPool": bool(prev_pool),
            "Prev Is Upsample": bool(prev_ups),
        })

    for m in model.modules():
        if isinstance(m, conv_types):
            conv_hooks.append(m.register_forward_hook(conv_hook))

    with torch.no_grad():
        _ = model(x)

    # cleanup hooks
    for h in conv_hooks + common_hooks:
        h.remove()

    return results


def save_to_excel(rows: List[Dict[str, Any]], path: str):
    # stringify tuples for Excel
    def fmt(v):
        if isinstance(v, tuple):
            try:
                return "x".join(str(int(x)) for x in v)
            except Exception:
                return str(v)
        return v
    rows_fmt = [{k: fmt(v) for k, v in r.items()} for r in rows]
    df = pd.DataFrame(rows_fmt, columns=[
        "Conv Name", "Type", "Input Dims", "Output Dims",
        "Kernel Dims", "Bias Dims", "Padding",
        "Prev Is MaxPool", "Prev Is Upsample"
    ])
    with pd.ExcelWriter(path, engine="openpyxl") as wr:
        df.to_excel(wr, index=False, sheet_name="Convolutions")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory with model.pth and config.yml")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(os.path.join(args.model_dir, "config.yml"))
    model = load_model(cfg, os.path.join(args.model_dir, "model.pth"), device)
    dummy = make_dummy_input(cfg, device)

    rows = collect_conv_report(model, dummy)

    # print a compact table
    try:
        from tabulate import tabulate
        print(tabulate(rows, headers="keys", tablefmt="github"))
    except Exception:
        print(rows)

    out_xlsx = os.path.join(args.model_dir, "conv_report.xlsx")
    save_to_excel(rows, out_xlsx)
    print(f"\nSaved Excel report to: {out_xlsx}")


if __name__ == "__main__":
    main()
