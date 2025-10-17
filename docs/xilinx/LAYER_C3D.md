# Layer C3D
Layer C3D processes normal 3D convolutions

## Format

- **Input:** [H][W][inC]
- **Output:** [H][W][outC]
- **Kernel:** [layerID][K1][K2][outC][inC]
- **Bias**: [layerID][outC]

### Configurations

| Input Dim  | Output Dim | Kernel Dim | Bias | Padding | Upsample | Skip Con |
| ---------- |------------|------------|------|---------|----------|----------|
|||First Layer|||||
| 256x256x3  | 256x256x8  | 3x3x3x8    | 8    | 1       |0         |0         |
|||Encoders|||||
| 256x256x8  | 256x256x8  | 3x3x8x8    | 8    | 1       |0         |0         |
| 128x128x8  | 128x128x8  | 3x3x8x8    | 8    | 1       |0         |0         |
| 64x64x8    | 64x64x16   | 3x3x16x8   | 16   | 1       |0         |0         |
| 32x32x16   | 32x32x16   | 3x3x16x16  | 16   | 1       |0         |0         |
| 16x16x16   | 16x16x24   | 3x3x24x16  | 24   | 1       |0         |0         |
|||Decoders|||||
| 16x16x24   | 32x32x16   | 3x3x16x24  | 16   | 1       |1         |0         |
| 32x32x16   | 32x32x16   | 3x3x16x16  | 16   | 1       |0         |1         |
| 32x32x16   | 64x64x16   | 3x3x16x16  | 16   | 1       |1         |0         |
| 64x64x16   | 64x64x16   | 3x3x16x16  | 16   | 1       |0         |1         |
| 64x64x16   | 128x128x8  | 3x3x8x16   | 8    | 1       |1         |0         |
| 128x128x8  | 128x128x8  | 3x3x8x8    | 8    | 1       |0         |1         |
| 128x128x8  | 256x256x8  | 3x3x8x8    | 8    | 1       |1         |0         |
| 256x256x8  | 256x256x8  | 3x3x8x8    | 8    | 1       |0         |1         |




- Output dimension should be the same as input dimension
- Padding is always 1
**Note** - Missing upsample part in table, so if upsample is 1 than the mapsize needs to be ajusted
- Should produce skip connection if applicable


## Layer Count
14
