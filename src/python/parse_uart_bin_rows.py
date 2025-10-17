# parse_uart_bin_rows.py
import argparse
import os
import re

import cv2
import numpy as np


def parse_uart_bin_rows(lines, expect_w=None, expect_h=None):
    rows = []
    w_hdr = h_hdr = None
    row_re = re.compile(r'^\s*(\d+):\s*([01\s]+)')

    for l_ in lines:
        line = l_.rstrip('\r\n')
        if line.startswith("BEGIN_BIN"):
            parts = line.split()
            if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
                w_hdr, h_hdr = int(parts[1]), int(parts[2])
            continue
        if line.startswith("END_BIN"):
            break
        m = row_re.match(line)
        if not m:
            continue
        bits = m.group(2).replace(' ', '')
        rows.append([1 if c == '1' else 0 for c in bits])

    if not rows:
        raise ValueError("No rows parsed")

    w = expect_w or w_hdr or len(rows[0])
    h = expect_h or h_hdr or len(rows)

    # normalize each row width
    for i, r in enumerate(rows):
        if len(r) < w:
            rows[i] = r + [0] * (w - len(r))
        elif len(r) > w:
            rows[i] = r[:w]

    # normalize height
    if len(rows) < h:
        rows.extend([[0] * w for _ in range(h - len(rows))])
    elif len(rows) > h:
        rows = rows[:h]

    return (np.array(rows, dtype=np.uint8) * 255)


def process_file(txt_path, expect_w=None, expect_h=None):
    """Parse one .txt file and save as PNG"""
    out_path = os.path.splitext(txt_path)[0] + ".png"
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    try:
        img = parse_uart_bin_rows(lines, expect_w, expect_h)
        ok = cv2.imwrite(out_path, img)
        if not ok:
            print(f"❌ cv2.imwrite failed: {out_path}")
        else:
            print(f"✅ Wrote {out_path}  shape={img.shape}")
    except Exception as e:
        print(f"⚠️ Failed to parse {txt_path}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Parse all UART .txt logs in a folder into PNGs")
    ap.add_argument("folder", help="Folder containing .txt files")
    ap.add_argument("--w", type=int, help="override width")
    ap.add_argument("--h", type=int, help="override height")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a directory: {folder}")

    txt_files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".txt")]
    )

    if not txt_files:
        print(f"No .txt files found in {folder}")
        return

    print(f"Found {len(txt_files)} .txt files in {folder}")
    for txt_path in txt_files:
        process_file(txt_path, args.w, args.h)


if __name__ == "__main__":
    main()
