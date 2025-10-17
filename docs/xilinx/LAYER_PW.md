# Layer PW
Layer PW processes pointwise convolutions.

Processes two [H][C] lines by multiplying 8 channels with 8 filters at a time.

## Format

- **Input:** [H][W][InC]
- **Output:** [H][W][outC]
- **Kernel:** [layerID][outC][inC]
- **Bias:** [layerID][outC]

### Configurations


| Input Dim   | Output Dim | Kernel Dim | Bias |
| ----------- |------------|------------|------|
| 256x256x8   | 256x256x32 | 1          | 32   |
| 256x256x32  | 256x256x8  | 1          | 8    |
| 128x128x8   | 128x128x32 | 1          | 32   |
| 128x128x32  | 128x128x8  | 1          | 8    |
| 64x64x8     | 64x64x32   | 1          | 64   |
| 64x64x32    | 64x64x8    | 1          | 16   |
| 32x32x16    | 32x32x64   | 1          | 64   |
| 32x32x64    | 32x32x16   | 1          | 16   |
| 16x16x16    | 16x16x64   | 1          | 64   |
| 16x16x64    | 16x16x16   | 1          | 16   |
| 256x256x8   | 256x256x8  | 1          | 8    |

- **Note:** Kernel is the same for HW or SW
- Output dimension should be the same as input dimension
- Last layer should have 256x256x8 but the values that matter should be only 256x256x1,
the bias will only have 1 value and the others would be 0

## Layer Count
25
