#include "utils.h"


void fill_map4D_contiguous_sequential(int8_t* map, int w_size, int z_size, int y_size, int x_size) {
    uint8_t value = 0;
    for (int w = 0; w < w_size; w++) {
        for (int z = 0; z < z_size; z++) {
            for (int y = 0; y < y_size; y++) {
                for (int x = 0; x < x_size; x++) {
                    // Linear index: w*(Z*Y*X) + z*(Y*X) + y*X + x
                    int index = w * (z_size * y_size * x_size) +
                                z * (y_size * x_size) +
                                y * x_size +
                                x;
                    map[index] = (int8_t)value;
                    value++;
                }
            }
        }
    }
}



void fill_map4D_sequential(int8_t**** map, int w_size, int z_size, int y_size, int x_size) {
    uint8_t value = 0;
    for (int w = 0; w < w_size; w++) {
        for (int z = 0; z < z_size; z++) {
            for (int y = 0; y < y_size; y++) {
                for (int x = 0; x < x_size; x++) {
                    map[w][z][y][x] = (int8_t)value;
                    value++;
                }
            }
        }
    }
}


void fill_map3D_contiguous_sequential(int8_t* map, int z_size, int y_size, int x_size) {
    uint8_t value = 0;
    for (int z = 0; z < z_size; z++) {
        for (int y = 0; y < y_size; y++) {
            for (int x = 0; x < x_size; x++) {
                // Calculate linear index: z * (Y * X) + y * X + x
                int index = z * (y_size * x_size) + y * x_size + x;
                map[index] = (int8_t)value;
                value++;
            }
        }
    }
}

void fill_map3D_sequential(int8_t*** map, int z_size, int y_size, int x_size) {
    uint8_t value = 0;
    for (int z = 0; z < z_size; z++) {
        for (int y = 0; y < y_size; y++) {
            for (int x = 0; x < x_size; x++) {
                map[z][y][x] = (int8_t)value;  // Use 3D indexing directly
                value++;
            }
        }
    }
}



void fill_map2D_sequential(int8_t** map, int y_size, int x_size) {
    uint8_t value = 0;
    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            map[y][x] = (int8_t)value;  // Use 2D indexing
            value++;
        }
    }
}


void fill_map2D_contiguous_sequential(int8_t* map, int y_size, int x_size) {
    uint8_t value = 0;
    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            map[y * x_size + x] = (int8_t)value;
            value++;
        }
    }
}



#define MAX_ELEMENTS 1000000000000000 // Adjust as "reasonable number" threshold


void print_map2D(int8_t** map, int y_size, int x_size) {
    int total = y_size * x_size;
    printf("int8_t map[%d][%d] = {\n", y_size, x_size);
    int skip_middle = total > MAX_ELEMENTS;

    for (int y = 0; y < y_size; y++) {
        if (skip_middle && y == 3 && y_size > 6) {
            printf("    ...\n");
            y = y_size - 3;
        }
        printf("    {");
        for (int x = 0; x < x_size; x++) {
            printf("%d", map[y][x]);  // Direct 2D indexing instead of manual calculation
            if (x < x_size - 1) printf(", ");
        }
        printf("}");
        if (y < y_size - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}



void print_map3D(int8_t*** map, int z_size, int y_size, int x_size) {
    int total = z_size * y_size * x_size;
    printf("int8_t map[%d][%d][%d] = {\n", z_size, y_size, x_size);

    int skip_middle = total > MAX_ELEMENTS;
    for (int z = 0; z < z_size; z++) {
        if (skip_middle && z == 3 && z_size > 6) {
            printf("    ...\n");
            z = z_size - 3;
        }
        printf("    {");
        for (int y = 0; y < y_size; y++) {
            printf("{");
            for (int x = 0; x < x_size; x++) {
                printf("%d", map[z][y][x]);
                if (x < x_size - 1) printf(", ");
            }
            printf("}");
            if (y < y_size - 1) printf(", ");
        }
        printf("}");
        if (z < z_size - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}

void print_map4D(int8_t**** map, int d1, int d2, int d3, int d4) {
    int total = d1 * d2 * d3 * d4;
    printf("int8_t map[%d][%d][%d][%d] = {\n", d1, d2, d3, d4);

    int skip_middle = total > MAX_ELEMENTS;
    for (int i = 0; i < d1; i++) {
        if (skip_middle && i == 3 && d1 > 6) {
            printf("    ...\n");
            i = d1 - 3;
        }
        printf("    {");  // open d2
        for (int j = 0; j < d2; j++) {
            printf("{");  // open d3
            for (int k = 0; k < d3; k++) {
                printf("{");  // open d4
                for (int l = 0; l < d4; l++) {
                    printf("%d", map[i][j][k][l]);
                    if (l < d4 - 1) printf(", ");
                }
                printf("}");
                if (k < d3 - 1) printf(", ");
            }
            printf("}");
            if (j < d2 - 1) printf(", ");
        }
        printf("}");
        if (i < d1 - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}



void print_map3D_contiguous(int8_t* map, int z_size, int y_size, int x_size) {
    int total = z_size * y_size * x_size;
    printf("int8_t map[%d][%d][%d] = {\n", z_size, y_size, x_size);

    int skip_middle = total > MAX_ELEMENTS;
    for (int z = 0; z < z_size; z++) {
        if (skip_middle && z == 3 && z_size > 6) {
            printf("    ...\n");
            z = z_size - 3;
        }
        printf("    {");
        for (int y = 0; y < y_size; y++) {
            printf("{");
            for (int x = 0; x < x_size; x++) {
                int index = z * (y_size * x_size) + y * x_size + x;
                printf("%d", map[index]);
                if (x < x_size - 1) printf(", ");
            }
            printf("}");
            if (y < y_size - 1) printf(", ");
        }
        printf("}");
        if (z < z_size - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}

void print_map4D_contiguous(int8_t* map, int dim0, int dim1, int dim2, int dim3) {
    int total = dim0 * dim1 * dim2 * dim3;
    printf("int8_t map[%d][%d][%d][%d] = {\n", dim0, dim1, dim2, dim3);

    int skip_middle = total > MAX_ELEMENTS;
    for (int i0 = 0; i0 < dim0; i0++) {
        if (skip_middle && i0 == 3 && dim0 > 6) {
            printf("    ...\n");
            i0 = dim0 - 3;
        }
        printf("    {");
        for (int i1 = 0; i1 < dim1; i1++) {
            printf("{");
            for (int i2 = 0; i2 < dim2; i2++) {
                printf("{");
                for (int i3 = 0; i3 < dim3; i3++) {
                    int index = i0 * (dim1 * dim2 * dim3)
                              + i1 * (dim2 * dim3)
                              + i2 * dim3
                              + i3;
                    printf("%d", map[index]);
                    if (i3 < dim3 - 1) printf(", ");
                }
                printf("}");
                if (i2 < dim2 - 1) printf(", ");
            }
            printf("}");
            if (i1 < dim1 - 1) printf(", ");
        }
        printf("}");
        if (i0 < dim0 - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}


void print_map_YXZ(int8_t* map, int y_size, int x_size, int z_size) {
    printf("int8_t map[%d][%d][%d] = {\n", y_size, x_size, z_size);
    for (int y = 0; y < y_size; y++) {
        printf("    {");
        for (int x = 0; x < x_size; x++) {
            printf("{");
            for (int z = 0; z < z_size; z++) {
                int index = y * (x_size * z_size) + x * z_size + z;
                printf("%d", map[index]);
                if (z < z_size - 1) printf(", ");
            }
            printf("}");
            if (x < x_size - 1) printf(", ");
        }
        printf("}");
        if (y < y_size - 1) printf(",");
        printf("\n");
    }
    printf("};\n");
}

uint64_t pack8bytes(int8_t* vals) {
    uint64_t result = 0;
    uint8_t converted;
    for (int i = 0; i < 8; i++) {
        converted = (uint8_t) vals[i];
        result |= ((uint64_t) converted) << (8 * i);
    }
    return result;
}

void transpose_YXZ_contiguous(int8_t* in, int8_t* out, int z_size, int y_size, int x_size) {
    int idx = 0;
    for (int y = 0; y < y_size; y++) {
        for (int x = 0; x < x_size; x++) {
            for (int z = 0; z < z_size; z++) {
                int offset = z * y_size * x_size + y * x_size + x;
                out[idx++] = in[offset];
            }
        }
    }
}

void transpose_YXZ(int8_t* in, int8_t*** out, int z_size, int y_size, int x_size) {
    int idx = 0;
    for (int z = 0; z < z_size; z++) {         // C
        for (int y = 0; y < y_size; y++) {     // H
            for (int x = 0; x < x_size; x++) { // W
                out[y][x][z] = in[idx++];
            }
        }
    }
}


// Transpose weights from (outC, inC, kH, kW) to (kH, kW, outC, inC)
// This matches the Python: transpose(2, 3, 0, 1)
void transpose_weights_4d(
    int8_t* in,      // input 1D flattened array in (outC, inC, kH, kW) order
    int8_t**** out,  // output as 4D pointer in (kH, kW, outC, inC) order
    int outC, int inC, int kH, int kW
) {
    int idx = 0;
    for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inC; ic++) {
            for (int h = 0; h < kH; h++) {
                for (int w = 0; w < kW; w++) {
                    // Input index order: (oc, ic, h, w)
                    int8_t val = in[idx++];
                    // Output index order: (h, w, oc, ic) - CORRECTED
                    out[h][w][oc][ic] = val;
                }
            }
        }
    }
}



void transpose_XY_contiguous(int8_t* in, int8_t* out, int y_size, int x_size) {
    int idx = 0;
    for (int x = 0; x < x_size; x++) {
        for (int y = 0; y < y_size; y++) {
            int in_idx = y * x_size + x;  // Calculate flattened index assuming row-major (YX) layout
            out[idx++] = in[in_idx];
        }
    }
}

void print_YXZ_order_map(int8_t* map, int z_size, int y_size, int x_size) {
    printf("Map in XZ order [Z=%d, Y=%d, X=%d]:\n", z_size, y_size, x_size);

    for (int y = 0; y < y_size; y++) {
        printf("Y=%d: ", y);
        for (int x = 0; x < x_size; x++) {
            printf("X%d[", x);
            for (int z = 0; z < z_size; z++) {
                int offset = z * y_size * x_size + y * x_size + x;
                printf("%3d", map[offset]);
                if (z < z_size - 1) printf(",");
            }
            printf("]");
            if (x < x_size - 1) printf(" ");
        }
        printf("\n");
    }
}

void print_flattened_XZ(uint8_t* flattened_array, int z_size, int y_size, int x_size) {
    printf("\nFlattened XZ array:\n");
    int total_size = z_size * y_size * x_size;

    for (int i = 0; i < total_size; i++) {
        printf("%3d", flattened_array[i]);

        // Add separators to show the structure
        if ((i + 1) % z_size == 0) {
            printf(" | ");  // End of Z group
        } else {
            printf(" ");
        }

        // New line after each Y row
        if ((i + 1) % (z_size * x_size) == 0) {
            printf("\n");
        }
    }
    printf("\n");
}



// Pads a 3D tensor (Z × Y × X) with 0s on top, bottom, left, and right of each 2D slice.
// Input shape:  [Z][Y][X] (flattened as input[Z * Y * X])
// Output shape: [Z][Y+2][X+2] (flattened as output[Z * (Y+2) * (X+2)])
void pad_tensor_3d(
    int8_t* output,     // Pre-allocated output buffer [Z][Y+2*pad][X+2*pad]
    int8_t* input,      // Input buffer [Z][Y][X]
    int Z, int Y, int X,// Dimensions of input
    int pad             // Padding size (same for all sides)
) {
    int outY = Y + 2 * pad;
    int outX = X + 2 * pad;

    for (int z = 0; z < Z; ++z) {
        // Zero the entire output slice for this z
        memset(&output[z * outY * outX], 0, outY * outX * sizeof(int8_t));

        for (int y = 0; y < Y; ++y) {
            for (int x = 0; x < X; ++x) {
                int in_idx = z * Y * X + y * X + x;
                int out_idx = z * outY * outX + (y + pad) * outX + (x + pad);
                output[out_idx] = input[in_idx];
            }
        }
    }
}



// Applies 2x2 maxpool with stride 2 to a 3D tensor (Z × Y × X)
// Input shape: [Z][Y][X] (flattened as input[Z * Y * X])
// Output shape: [Z][Y/2][X/2] (flattened as output[Z * (Y/2) * (X/2)])
// Assumes Y and X are even numbers for simplicity
void maxpool_2x2_stride2_3d(
    int8_t* output,     // Pre-allocated output buffer [Z][Y/2][X/2]
    int8_t* input,      // Input buffer [Z][Y][X]
    int Z, int Y, int X // Dimensions of input
) {
    int outY = Y / 2;
    int outX = X / 2;

    for (int z = 0; z < Z; ++z) {
        for (int out_y = 0; out_y < outY; ++out_y) {
            for (int out_x = 0; out_x < outX; ++out_x) {

                // Calculate input coordinates for the 2x2 window
                int in_y = out_y * 2;
                int in_x = out_x * 2;

                // Get the 4 values in the 2x2 window
                int in_idx_00 = z * Y * X + (in_y + 0) * X + (in_x + 0);  // top-left
                int in_idx_01 = z * Y * X + (in_y + 0) * X + (in_x + 1);  // top-right
                int in_idx_10 = z * Y * X + (in_y + 1) * X + (in_x + 0);  // bottom-left
                int in_idx_11 = z * Y * X + (in_y + 1) * X + (in_x + 1);  // bottom-right

                int8_t val_00 = input[in_idx_00];
                int8_t val_01 = input[in_idx_01];
                int8_t val_10 = input[in_idx_10];
                int8_t val_11 = input[in_idx_11];

                // Find maximum of the 4 values
                int8_t max_val = val_00;
                if (val_01 > max_val) max_val = val_01;
                if (val_10 > max_val) max_val = val_10;
                if (val_11 > max_val) max_val = val_11;

                // Store result
                int out_idx = z * outY * outX + out_y * outX + out_x;
                output[out_idx] = max_val;
            }
        }
    }
}


bool compare_HW_SW(hls::stream<ap_uint<64>> &fifo_out, int8_t *map_flat_YXZ, int total_size)
{
    int mismatch_count = 0;
    int total_chunks = total_size / 8;

    // Temporary buffer so we can replay values later
    std::vector<ap_uint<64>> buffer;
    buffer.reserve(total_chunks);

    for (int i = 0; i < total_chunks; i++) {
        if (fifo_out.empty()) {
            printf("-------------------------------------\n");
            printf("HW - SW Comparison failed, fifo is empty\n");
            return false;
        }
        ap_uint<64> hw_val = fifo_out.read();
        buffer.push_back(hw_val); // save copy

        uint64_t sw_val = pack8bytes(&map_flat_YXZ[i * 8]);

        if (i < 10) {
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n",
                   i, (unsigned long long)hw_val,
                   i, (unsigned long long)sw_val);
        }

        if ((uint64_t)hw_val != sw_val) {
            mismatch_count++;
        }
    }

    // Replay values back into the stream
    for (int i = 0; i < total_chunks; i++) {
        fifo_out.write(buffer[i]);
    }

    printf("-------------------------------------\n");
    if (mismatch_count == 0) {
        printf("All outputs match! (%d chunks)\n", total_chunks);
        return true;
    } else {
        printf("Mismatches found: %d out of %d chunks\n",
               mismatch_count, total_chunks);
    }
    return false;
}


bool compare_HW_SW_bus64_t(hls::stream<bus64_t> &fifo_out, int8_t *map_flat_YXZ, int total_size)
{
    int mismatch_count = 0;
    int read = 0;
    int total_chunks = total_size / 8;
    bus64_t hw_val;

    for (int i = 0; i < total_chunks; i++) {
        if (fifo_out.empty()) {
            printf("-------------------------------------\n");
            printf("HW - SW Comparison failed, fifo is empty\n");
            return false;
        }
        hw_val = fifo_out.read();
        read++;

        uint64_t sw_val = pack8bytes(&map_flat_YXZ[i * 8]);

        if (i < 10) {
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n",
                   i, (unsigned long long)hw_val.data,
                   i, (unsigned long long)sw_val);
        }

        if (!hw_val.last.is_zero()) {
            printf("TLAST = 1; CHUNK = %d\n", i);
        }

        if ((uint64_t)hw_val.data != sw_val) {
            mismatch_count++;
        }
    }

    printf("WRITE_WORDS = %d\n", read);
    printf("-------------------------------------\n");
    if (mismatch_count == 0) {
        printf("All outputs match! (%d chunks)\n", total_chunks);
        return true;
    } else {
        printf("Mismatches found: %d out of %d chunks\n",
               mismatch_count, total_chunks);
    }
    return false;
}


bool compare_HW_SW_256(hls::stream<ap_uint<256>> &fifo_out, int8_t *map_flat_YXZ, int total_size)
{
    int mismatch_count = 0;
    int total_chunks = total_size / 32; // 256 bits = 32 bytes
    ap_uint<256> tmp;

    // Temporary buffer to store values so we can write them back later
    std::vector<ap_uint<256>> buffer;
    buffer.reserve(total_chunks);

    for (int i = 0; i < total_chunks; i++) {
        if (fifo_out.empty()) {
            printf("-------------------------------------\n");
            printf("HW - SW Comparison failed, fifo is empty\n");
            return false;
        }
        tmp = fifo_out.read();
        buffer.push_back(tmp); // keep a copy

        // Extract 4 × 64-bit chunks from 256-bit word
        uint64_t hw_val0 = tmp.range( 63,   0);
        uint64_t hw_val1 = tmp.range(127,  64);
        uint64_t hw_val2 = tmp.range(191, 128);
        uint64_t hw_val3 = tmp.range(255, 192);

        // Pack corresponding SW chunks
        uint64_t sw_val0 = pack8bytes(&map_flat_YXZ[i * 32 +  0]);
        uint64_t sw_val1 = pack8bytes(&map_flat_YXZ[i * 32 +  8]);
        uint64_t sw_val2 = pack8bytes(&map_flat_YXZ[i * 32 + 16]);
        uint64_t sw_val3 = pack8bytes(&map_flat_YXZ[i * 32 + 24]);

        // Debug print first 10 chunks
        if (i*4 >= 256 && i*4 < 266) {
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n", i*4+0, hw_val0, i*4+0, sw_val0);
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n", i*4+1, hw_val1, i*4+1, sw_val1);
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n", i*4+2, hw_val2, i*4+2, sw_val2);
            printf("HW[%3d]: 0x%016llx | SW[%3d]: 0x%016llx\n", i*4+3, hw_val3, i*4+3, sw_val3);
        }

        // Compare all 4 subchunks
        if (hw_val0 != sw_val0) mismatch_count++;
        if (hw_val1 != sw_val1) mismatch_count++;
        if (hw_val2 != sw_val2) mismatch_count++;
        if (hw_val3 != sw_val3) mismatch_count++;
    }

    // Replay values back into the stream
    for (int i = 0; i < total_chunks; i++) {
        fifo_out.write(buffer[i]);
    }

    printf("-------------------------------------\n");
    if (mismatch_count == 0) {
        printf("All outputs match! (%d 64-bit chunks)\n", total_chunks*4);
        return true;
    } else {
        printf("Mismatches found: %d out of %d chunks\n", mismatch_count, total_chunks*4);
    }
    return false;
}



void transpose_and_compare_hls_bus64_t(
    int8_t* sw_data,          // software output
    hls::stream<bus64_t> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
) {
    int total_size = channels * height * width;
    int8_t* sw_transposed = new int8_t[total_size];

    // Transpose SW output
    transpose_YXZ_contiguous(sw_data, sw_transposed, channels, height, width);

    // Compare with HW stream
    assert(compare_HW_SW_bus64_t(hw_stream, sw_transposed, total_size));

    delete[] sw_transposed;
}




void transpose_and_compare_hls(
    int8_t* sw_data,          // software output
    hls::stream<ap_uint<64>> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
) {
    int total_size = channels * height * width;
    int8_t* sw_transposed = new int8_t[total_size];

    // Transpose SW output
    transpose_YXZ_contiguous(sw_data, sw_transposed, channels, height, width);

    // Compare with HW stream
    assert(compare_HW_SW(hw_stream, sw_transposed, total_size));

    delete[] sw_transposed;
}


void transpose_and_compare_hls_256(
    int8_t* sw_data,          // software output
    hls::stream<ap_uint<256>> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
) {
    int total_size = channels * height * width;
    int8_t* sw_transposed = new int8_t[total_size];

    // Transpose SW output
    transpose_YXZ_contiguous(sw_data, sw_transposed, channels, height, width);

    // Compare with HW stream
    assert(compare_HW_SW_256(hw_stream, sw_transposed, total_size));

    delete[] sw_transposed;
}



int8_t*** allocate_map_3d(int h, int w, int c) {
    int8_t*** map = (int8_t***)malloc(h * sizeof(int8_t**));

    for (int i = 0; i < h; i++) {
        map[i] = (int8_t**)malloc(w * sizeof(int8_t*));
        for (int j = 0; j < w; j++) {
            map[i][j] = (int8_t*)malloc(c * sizeof(int8_t));
        }
    }
    return map;
}


int8_t**** allocate_map_4d(int d1, int d2, int d3, int d4) {
    int8_t**** map = (int8_t****)malloc(d1 * sizeof(int8_t***));

    for (int i = 0; i < d1; i++) {
        map[i] = (int8_t***)malloc(d2 * sizeof(int8_t**));
        for (int j = 0; j < d2; j++) {
            map[i][j] = (int8_t**)malloc(d3 * sizeof(int8_t*));
            for (int k = 0; k < d3; k++) {
                map[i][j][k] = (int8_t*)malloc(d4 * sizeof(int8_t));
            }
        }
    }
    return map;
}



int8_t** allocate_map_2d(int h, int w) {
    int8_t** map = (int8_t**)malloc(h * sizeof(int8_t*));
    for (int i = 0; i < h; i++) {
        map[i] = (int8_t*)malloc(w * sizeof(int8_t));
    }
    return map;
}



// Function to print ap_uint<64> as hex by extracting individual bytes
void print_ap_uint64_hex(ap_uint<64>& value, const char* label = "") {
    printf("%s", label);
    ap_uint<8> val;

    // Extract each byte (8 bits) from the 64-bit value
    // Print from most significant byte (7) to least significant byte (0)
    for (int i = 7; i >= 0; i--) {
        val = value.range(i*8 + 7, i*8);
        uint8_t byte_val = (uint8_t) val;
        printf("%02X", byte_val);
    }
    printf("\n");
}
