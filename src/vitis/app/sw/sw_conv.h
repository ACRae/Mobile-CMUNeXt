#include <stdint.h>
#ifndef SW_CONV_H
#define SW_CONV_H

#define OUTPUT_DIMENSION(INPUT_DIMENSION, PADDING, KERNEL_SIZE) \
    ((INPUT_DIMENSION) + 2 * (PADDING) - (KERNEL_SIZE) + 1)

#define CLAMP8_RELU(x) ((x) > INT8_MAX ? INT8_MAX : ((x) < 0 ? 0 : (x)))

typedef struct {
    // Input tensor dimensions
    int in_channels;     // Number of input channels (e.g., 3 for RGB)
    int in_height;       // Height of the input feature map
    int in_width;        // Width of the input feature map

    // Output tensor dimensions
    int out_channels;    // Number of output channels (e.g., number of filters)

    // Convolution parameters
    int kernel_size;     // Size of the convolution kernel (assumed square, e.g., 3)
    int relu_shift;
} conv2d_args;


void sw_conv2D_relu_nopad(
    int8_t* input,
    int8_t* weights,
    int16_t* bias,
    int8_t* output,
    conv2d_args* args
);

void sw_conv_dw(
    int8_t* input,
    int8_t* weights,
    int16_t* bias,
    int8_t* output,
    conv2d_args* args
);

// void maxpool_2x2_stride2_3d(
//     int8_t* output,     // Pre-allocated output buffer [Z][Y/2][X/2]
//     int8_t* input,      // Input buffer [Z][Y][X]
//     int Z, int Y, int X // Dimensions of input
// );

// void pad_tensor_3d(
//     int8_t* output,            // Pre-allocated output buffer [Z][Y+2][X+2]
//     int8_t* input,       // Input buffer [Z][Y][X]
//     int Z, int Y, int X,         // Dimensions of input
//     int pad 
// );


#endif // SW_CONV_H