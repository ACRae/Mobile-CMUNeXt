#include "sw_conv.h"
#include <string.h>

void sw_conv2D_relu_nopad(
    int8_t* input,
    int8_t* weights,
    int16_t* bias,
    int8_t* output,
    conv2d_args* args
) {
    const int output_height = args->in_height - args->kernel_size + 1; // no padding
    const int output_width  = args->in_width  - args->kernel_size + 1; // no padding

    for (int oc = 0; oc < args->out_channels; ++oc) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                int32_t accum = (int32_t) bias[oc];

                for (int ic = 0; ic < args->in_channels; ++ic) {
                    for (int x = 0; x < args->kernel_size; ++x) {
                        for (int y = 0; y < args->kernel_size; ++y) {
                            int input_idx =
                                ic * args->in_height * args->in_width +
                                (i + x) * args->in_width + (j + y);

                            int weight_idx =
                                oc * args->in_channels * args->kernel_size * args->kernel_size +
                                ic * args->kernel_size * args->kernel_size +
                                x * args->kernel_size + y;

                            int8_t input_val = input[input_idx];
                            int8_t weight_val = weights[weight_idx];
                            int32_t product = ((int32_t)weight_val * input_val);

                            accum += product;
                        }
                    }
                }

                int out_idx =
                    oc * output_height * output_width +
                    i * output_width + j;

                // BEGIN: RELU
                int32_t scaled = accum >> args->relu_shift;
                output[out_idx] = (int8_t) CLAMP8_RELU(scaled);

            }
        }
    }
}


void sw_conv_dw(
    int8_t* input,
    int8_t* weights,
    int16_t* bias,
    int8_t* output,
    conv2d_args* args
) {
    // Output dims without padding
    const int output_height = (args->in_height - args->kernel_size) + 1;
    const int output_width  = (args->in_width  - args->kernel_size) + 1;

    for (int c = 0; c < args->in_channels; ++c) {
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {

                int32_t accum = (int32_t) bias[c];

                // Convolution kernel
                for (int x = 0; x < args->kernel_size; ++x) {
                    for (int y = 0; y < args->kernel_size; ++y) {
                        int input_x = i + x;
                        int input_y = j + y;

                        int input_idx =
                            c * args->in_height * args->in_width +
                            input_x * args->in_width + input_y;

                        int weight_idx =
                            c * args->kernel_size * args->kernel_size +
                            x * args->kernel_size + y;

                        int8_t  input_val  = input[input_idx];
                        int8_t  weight_val = weights[weight_idx];
                        int32_t product    = ((int32_t) weight_val * input_val);

                        accum += product;
                    }
                }

                int out_idx =
                    c * output_height * output_width +
                    i * output_width + j;

                // BEGIN: RELU
                int8_t out;
                if (accum <= 0) {
                    out = 0;
                } else {
                    int32_t scaled = accum >> args->relu_shift;  // rouding instead of truncating
                    out = (int8_t) CLAMP8_RELU(scaled);
                }
                // END: RELU

                output[out_idx] = out;
            }
        }
    }
}


// Pads a 3D tensor (Z × Y × X) with 0s on top, bottom, left, and right of each 2D slice.
// Input shape:  [Z][Y][X] (flattened as input[Z * Y * X])
// Output shape: [Z][Y+2][X+2] (flattened as output[Z * (Y+2) * (X+2)])
// void pad_tensor_3d(
//     int8_t* output,     // Pre-allocated output buffer [Z][Y+2*pad][X+2*pad]
//     int8_t* input,      // Input buffer [Z][Y][X]
//     int Z, int Y, int X,// Dimensions of input
//     int pad             // Padding size (same for all sides)
// ) {
//     int outY = Y + 2 * pad;
//     int outX = X + 2 * pad;

//     for (int z = 0; z < Z; ++z) {
//         // Zero the entire output slice for this z
//         memset(&output[z * outY * outX], 0, outY * outX * sizeof(int8_t));

//         for (int y = 0; y < Y; ++y) {
//             for (int x = 0; x < X; ++x) {
//                 int in_idx = z * Y * X + y * X + x;
//                 int out_idx = z * outY * outX + (y + pad) * outX + (x + pad);
//                 output[out_idx] = input[in_idx];
//             }
//         }
//     }
// }
