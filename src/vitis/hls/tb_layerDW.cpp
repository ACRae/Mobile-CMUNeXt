#include <cstdint>
#include <cstdio>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "layerDW.h"
#include "utils.h"

#define MAXPOOL_STRIDE 2
#define MAXPOOL_KERNEL 2
#define MAX(a, b) (((data_t)(a) > (data_t)(b)) ? (a) : (b))

/*
NO PADDING VERSION
*/
void SW_depthwise(
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
                    int32_t scaled = accum >> args->relu_shift;
                    out = (int8_t) CLAMP8_RELU(scaled);
                }
                // END: RELU

                output[out_idx] = out;
            }
        }
    }
}

void SW_readData(
    int8_t* inMap,
    int8_t* outMap,
    int8_t* resMap,
    int in_channels,
    int map_size,
    int padding,
    bool maxpool
) {
    int8_t* in_ptr = inMap;
    int C = in_channels;
    int H = map_size;
    int W = map_size;
    int8_t* maxMap = NULL;

    if (maxpool) {
        int pooled_H = H / 2;
        int pooled_W = W / 2;
        size_t pooled_size = pooled_H * pooled_W * C * sizeof(int8_t);

        maxMap = (int8_t*) malloc(pooled_size);
        maxpool_2x2_stride2_3d(maxMap, inMap, C, H, W);

        // copy maxMap into resMap
        memcpy(resMap, maxMap, pooled_size);

        in_ptr = maxMap;
        H = pooled_H;
        W = pooled_W;
    } else {
        size_t map_bytes = C * H * W * sizeof(int8_t);
        memcpy(resMap, inMap, map_bytes);
    }

    pad_tensor_3d(outMap, in_ptr, C, H, W, padding);

    if (maxMap) {
        free(maxMap);
    }
}




void SW_writeData(
    int8_t* result,
    const int8_t* map1, // dephwise
    const int8_t* map2, // residual 
    int inC, int H, int W,
    int layer_ID
) {
    int size = inC * H * W;
    for (int i = 0; i < size; i++) {
        int8_t v1 = safe_shift(map1[i], scaleRES[layer_ID][1]);
        int8_t v2 = safe_shift(map2[i], scaleRES[layer_ID][0]);
        int16_t sum = v1 + v2;
        // clamp back to int8_t range [-128, 127]
        if (sum > 127) sum = 127;
        if (sum < -128) sum = -128;
        result[i] = (int8_t)sum;
    }
}


void test_DW(int map_size, int inC, int kernel_size, int padding, bool maxpool, int layer_id)
{

    int scale = scaleDW[layer_id];

    int map_size_maxpool = (maxpool == 1)
        ? (int) ((map_size - MAXPOOL_KERNEL) / MAXPOOL_STRIDE + 1)
        : (int) map_size;

    int map_size_maxpool_pad = map_size_maxpool+2*padding;

    int out_dim = OUTPUT_DIMENSION(map_size_maxpool, padding, kernel_size);
    assert(out_dim == map_size_maxpool && "Output dimension should be the same as input dimension!");


    int8_t SW_inMap[inC * map_size * map_size];
    fill_map3D_contiguous_sequential(SW_inMap, inC, map_size, map_size);
    
    int16_t bias[inC];
    for(int16_t i = 0; i < inC; i++) bias[i] = biasDW[layer_id][i];


    int8_t SW_kernel[inC * kernel_size * kernel_size];
    int idx = 0;
    for (int c = 0; c < inC; c++)
        for (int k1 = 0; k1 < kernel_size; k1++)
            for (int k2 = 0; k2 < kernel_size; k2++)
                SW_kernel[idx++] = weightsDW[layer_id][k1][k2][c];

    int8_t SW_outResidual[inC * map_size_maxpool * map_size_maxpool];
    int8_t SW_outMaxPad[inC * map_size_maxpool_pad * map_size_maxpool_pad];
    SW_readData(SW_inMap, SW_outMaxPad, SW_outResidual, inC, map_size, padding, maxpool);



    conv2d_args args = {
        .in_channels      = inC,
        .in_height        = map_size_maxpool_pad,
        .in_width         = map_size_maxpool_pad,
        .out_channels     = inC,
        .kernel_size      = kernel_size,
        .relu_shift = scale
    };

    int8_t SW_outDepthwise[inC * map_size_maxpool * map_size_maxpool];
    SW_depthwise(SW_outMaxPad, SW_kernel, bias, SW_outDepthwise, &args);


    int8_t SW_outWrite[inC * map_size_maxpool * map_size_maxpool];
    SW_writeData(SW_outWrite, SW_outDepthwise, SW_outResidual, inC, out_dim, out_dim, layer_id);



    hls::stream<bus64_t> strm_in;
    hls::stream<bus64_t> strm_out;
    hls::stream<ap_uint<64>> fifo_in;
    hls::stream<ap_uint<256>> fifo_out1;
    hls::stream<ap_uint<64>> fifo_out2;
    hls::stream<ap_uint<64>> res_connection;

    config_t config;

    config.range(8,0) = map_size;   // map size
    config.range(13,9) = layer_id;  // layer id
    config.range(15,14) = (inC / 8); // input channel
    config.range(19,16) = kernel_size; // kernel size
    config.range(22,20) = padding; // padding
    config.range(23,23) = maxpool; // maxpool


    ap_uint<64> tmp;
    bus64_t tmpb64;

    int8_t HW_inMap[inC * map_size * map_size];
    transpose_YXZ_contiguous(SW_inMap, HW_inMap, inC, map_size, map_size);

    // Write input
    for (int i = 0; i < inC * map_size * map_size / 8; i++) {
        tmpb64.data.range(63, 0) = pack8bytes(&HW_inMap[i * 8]);
        strm_in.write(tmpb64);
    }

    // HW_readData(strm_in, fifo_in, res_connection, config);
    // transpose_and_compare_hls(SW_outMaxPad, fifo_in, inC, map_size_maxpool_pad, map_size_maxpool_pad);
    // transpose_and_compare_hls(SW_outResidual, res_connection, inC, map_size_maxpool, map_size_maxpool);

    // HW_depthWise(fifo_in, fifo_out1, res_connection, config);
    // transpose_and_compare_hls_256(SW_outDepthwise, fifo_out1, inC, out_dim, out_dim);

    // HW_tb_writeData(strm_out, fifo_out1, res_connection, config);
    // HW_tb_layerDW(strm_in, strm_out, config, bias, HW_kernel3D);
    HW_layerDW(strm_in, strm_out, config);
    transpose_and_compare_hls_bus64_t(SW_outWrite, strm_out, inC, out_dim, out_dim);

}


int main() {
    // test_multiple_readData();

    /*
        256x256x8 k3 p1
        128x128x8 k3 p1
        64x64x16 k7 p3
        32x32x16 k9 p3
        16x16x16 k9 p4
    */

    // Encoder 1
    test_DW(256, 8, 3, 1, 0, 0);

    // Encoder 2
    test_DW(256, 8, 3, 1, 1, 1); // maxpool 
    test_DW(128, 8, 3, 1, 0, 2);
    
    // Encoder 3
    test_DW(128, 8, 7, 3, 1, 3); // maxpool
    test_DW(64, 8, 7, 3, 0, 4);    
    
    // Encoder 4
    test_DW(64, 16, 7, 3, 1, 5); // maxpool
    test_DW(32, 16, 7, 3, 0, 6);

    // Encoder 5
    test_DW(32, 16, 9, 4, 1, 7); // maxpool
    test_DW(16, 16, 9, 4, 0, 8);


    return 0;
}
