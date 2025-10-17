// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Function declarations

void uart_print_bits_bin_rows(const int8_t *buf, uint32_t w, uint32_t h);
void fill_map4D_contiguous_sequential(int8_t* map, int w_size, int z_size, int y_size, int x_size);
void fill_map4D_sequential(int8_t**** map, int w_size, int z_size, int y_size, int x_size);

void fill_map3D_contiguous_sequential(int8_t* map, int z_size, int y_size, int x_size);
void fill_map3D_sequential(int8_t*** map, int z_size, int y_size, int x_size);

void fill_map2D_contiguous_sequential(int8_t* map, int y_size, int x_size);
void fill_map2D_sequential(int8_t** map, int y_size, int x_size);

void print_map2D_contiguous(int8_t* map, int y_size, int x_size);
void print_map2D(int8_t** map, int y_size, int x_size);
void print_map3D(int8_t*** map, int z_size, int y_size, int x_size);
void print_map4D(int8_t**** map, int d1, int d2, int d3, int d4);
void print_map3D_contiguous(int8_t* map, int z_size, int y_size, int x_size);
void print_map4D_contiguous(int8_t* map, int dim0, int dim1, int dim2, int dim3);

void print_map_YXZ(int8_t* map, int y_size, int x_size, int z_size);
uint64_t pack8bytes(int8_t* vals);

// Additional utility functions
void transpose_YXZ_contiguous(int8_t* in, int8_t* out, int z_size, int y_size, int x_size);
void transpose_YXZ(int8_t* in, int8_t*** out, int z_size, int y_size, int x_size);
void transpose_weights_4d(
    int8_t* in,         // input 1D flatten array
    int8_t**** out,     // output as 4D pointer
    int outC, int inC, int kH, int kW
);
void transpose_XY_contiguous(int8_t* in, int8_t* out, int y_size, int x_size);
void print_YXZ_order_map(int8_t* map, int z_size, int y_size, int x_size);
void print_flattened_XZ(uint8_t* flattened_array, int z_size, int y_size, int x_size);

/*
bool compare_HW_SW(hls::stream<ap_uint<64>> &fifo_out, int8_t *map_flat_YXZ, int total_size);

void transpose_and_compare_hls(
    int8_t* sw_data,          // software output
    hls::stream<ap_uint<64>> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
);

void transpose_and_compare_hls_bus64_t(
    int8_t* sw_data,          // software output
    hls::stream<bus64_t> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
);

bool compare_HW_SW_bus64_t(hls::stream<bus64_t> &fifo_out, int8_t *map_flat_YXZ, int total_size);

bool compare_HW_SW_256(hls::stream<ap_uint<256>> &fifo_out, int8_t *map_flat_YXZ, int total_size);


void transpose_and_compare_hls_256(
    int8_t* sw_data,          // software output
    hls::stream<ap_uint<256>> &hw_stream,    // hardware HLS stream
    int channels,
    int height,
    int width
);
*/


// Inline utility functions
static inline int calculate_3d_index(int z, int y, int x, int y_size, int x_size) {
    return z * (y_size * x_size) + y * x_size + x;
}

static inline uint8_t get_map_value(uint8_t* map, int z, int y, int x, int y_size, int x_size) {
    int index = calculate_3d_index(z, y, x, y_size, x_size);
    return map[index];
}

static inline void set_map_value(uint8_t* map, int z, int y, int x, int y_size, int x_size, uint8_t value) {
    int index = calculate_3d_index(z, y, x, y_size, x_size);
    map[index] = value;
}


// void print_ap_uint64_hex(ap_uint<64>& value, const char* label);



// Macros
#define GET_3D_INDEX(z, y, x, y_size, x_size) ((z) * ((y_size) * (x_size)) + (y) * (x_size) + (x))
#define CLAMP8_RELU(x) ((x) > INT8_MAX ? INT8_MAX : ((x) < 0 ? 0 : (x)))
// const acc_t APINT8_MAX = 127;
// const acc_t APINT8_MIN = 0;
// #define CLAMP8_RELU_AP(x) ((x) > APINT8_MAX ? APINT8_MAX : ((x) < APINT8_MIN ? APINT8_MIN : (x)))

#define OUTPUT_DIMENSION(INPUT_DIMENSION, PADDING, KERNEL_SIZE) \
    ((INPUT_DIMENSION) + 2 * (PADDING) - (KERNEL_SIZE) + 1)


#endif // UTILS_H
