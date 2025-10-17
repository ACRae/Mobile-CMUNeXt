// utils.h
#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cassert>
#include "ap_int.h"
#include "hls_stream.h"
#include "layers.h"

// Function declarations

template <typename T>
static inline T safe_shift(T x, int shift) {
    return (shift >= 0) ? (x >> shift) : (x << -shift);
}



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
void pad_tensor_3d(
    int8_t* output,            // Pre-allocated output buffer [Z][Y+2][X+2]
    int8_t* input,       // Input buffer [Z][Y][X]
    int Z, int Y, int X,         // Dimensions of input
    int pad 
);
void maxpool_2x2_stride2_3d(
    int8_t* output,     // Pre-allocated output buffer [Z][Y/2][X/2]
    int8_t* input,      // Input buffer [Z][Y][X]
    int Z, int Y, int X // Dimensions of input
);
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

int8_t**** allocate_map_4d(int d1, int d2, int d3, int d4);
int8_t*** allocate_map_3d(int h, int w, int c);
int8_t** allocate_map_2d(int h, int w);


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


void print_ap_uint64_hex(ap_uint<64>& value, const char* label);



/**
 * @brief Structure to hold parameters for a 2D convolution layer.
 */
typedef struct {
    // Input tensor dimensions
    int in_channels;     // Number of input channels (e.g., 3 for RGB)
    int in_height;       // Height of the input feature map
    int in_width;        // Width of the input feature map

    // Output tensor dimensions
    int out_channels;    // Number of output channels (e.g., number of filters)

    // Convolution parameters
    int kernel_size;     // Size of the convolution kernel (assumed square, e.g., 3)
    // int padding;
    // int stride;

    int relu_shift;
    
    /*
    // Quantization fractional bits
    int input_frac_bits;   // Fractional bits for input (e.g., 7)
    int weight_frac_bits;  // Fractional bits for weights (e.g., 0)
    int bias_frac_bits;    // Fractional bits for bias (e.g., 14)
    int act_frac_bits;     // Fractional bits for activation/output (e.g., 7)
    
    
    // Reference macro definitions (used to compute values above):
    
    #define INPUT_CHANNELS    3
    #define INPUT_HEIGHT      256
    #define INPUT_WIDTH       256
    #define INPUT_SIZE        (INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS)

    #define KERNEL_SIZE       3
    #define PADDING           1 
    #define STRIDE            1

    #define OUTPUT_CHANNELS   8
    #define OUTPUT_HEIGHT     ((INPUT_HEIGHT + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)
    #define OUTPUT_WIDTH      ((INPUT_WIDTH + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1)
    #define OUTPUT_SIZE       (OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS)

    #define WEIGHT_FRAC_BITS  0
    #define BIAS_FRAC_BITS    14
    #define INPUT_FRAC_BITS   7
    #define ACT_FRAC_BITS     7
    #define SCALE_SHIFT       (BIAS_FRAC_BITS - (INPUT_FRAC_BITS + WEIGHT_FRAC_BITS))

    #define WEIGHTS_SIZE      (OUTPUT_CHANNELS * INPUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE)
    #define BIAS_SIZE         OUTPUT_CHANNELS
    */
} conv2d_args;




// Macros
#define GET_3D_INDEX(z, y, x, y_size, x_size) ((z) * ((y_size) * (x_size)) + (y) * (x_size) + (x))
#define CLAMP8_RELU(x) ((x) > INT8_MAX ? INT8_MAX : ((x) < 0 ? 0 : (x)))
// const acc_t APINT8_MAX = 127;
// const acc_t APINT8_MIN = 0;
// #define CLAMP8_RELU_AP(x) ((x) > APINT8_MAX ? APINT8_MAX : ((x) < APINT8_MIN ? APINT8_MIN : (x)))

#define OUTPUT_DIMENSION(INPUT_DIMENSION, PADDING, KERNEL_SIZE) \
    ((INPUT_DIMENSION) + 2 * (PADDING) - (KERNEL_SIZE) + 1)

// Debug macros
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

#endif // UTILS_H
