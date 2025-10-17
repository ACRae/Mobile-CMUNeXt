# Layer DW
Layer DW processes depthwise convolutions

## Format

- **Input:** [H][W][C]
- **Output:** [H][W][C]
- **Kernel:** [layerID][K1][K2][outC]
- **Bias:** [layerID][outC]

### Configurations

| Input Dim  | Padded Input Dim | Output Dim | Kernel Dim | Bias | Padding | Maxpool |
|------------|------------------|------------|------------|------|---------| ------- |
|||Encoder 1|||||
| 256x256x8  | 258x258x8        | 256x256x8  | 3x3x8      | 8    | 1       | 0       |
|||Encoder 2|||||
| 256x256x8  | 130x130x8        | 128x128x8  | 3x3x8      | 8    | 1       | 1       |
| 128x128x8  | 130x130x8        | 128x128x8  | 3x3x8      | 8    | 1       | 0       |
|||Encoder 3|||||
| 128x128x8  | 70x70x8          | 64x64x8    | 7x7x8      | 8    | 3       | 1       |
| 64x64x8    | 70x70x8          | 64x64x8    | 7x7x8      | 8    | 3       | 0       |
|||Encoder 4|||||
| 64x64x16   | 38x38x16         | 32x32x16   | 7x7x16     | 16   | 3       | 1       |
| 32x32x16   | 38x38x16         | 32x32x16   | 7x7x16     | 16   | 3       | 0       |
|||Encoder 5|||||
| 32x32x16   | 24x24x16         | 16x16x16   | 9x9x16     | 16   | 4       | 1       |
| 16x16x16   | 24x24x16         | 16x16x16   | 9x9x16     | 16   | 4       | 0       |

##

- **Note:** DW Channels can only be 8 or 16
    - Output dimension should be the same as input dimension


## Layer Count
16
