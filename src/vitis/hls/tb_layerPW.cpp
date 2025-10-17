#include <cstdint>
#include <stdio.h>
#include "utils.h"
#include "layers.h"
#include "layerPW.h"


void conv2D(
    int8_t* input,
    int8_t* weights, // [outC][inC][kernel_h][kernel_w]
    int16_t* bias,
    int8_t* output,
    conv2d_args* args,
    bool lastLayer
)
{
    int count = 0;
    const int output_height = OUTPUT_DIMENSION(args->in_height, 0, args->kernel_size);
    const int output_width = OUTPUT_DIMENSION(args->in_width, 0, args->kernel_size);

    for (int oc = 0; oc < args->out_channels; ++oc) {

        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                int32_t accum = (int32_t)bias[oc];

                for (int ic = 0; ic < args->in_channels; ++ic) {
                    for (int x = 0; x < args->kernel_size; ++x) {
                        for (int y = 0; y < args->kernel_size; ++y) {
                            int input_x = i + x;
                            int input_y = j + y;

                            int input_idx =
                                ic * args->in_height * args->in_width +
                                input_x * args->in_width + input_y;

                            int weight_idx =
                                oc * args->in_channels * args->kernel_size *
                                args->kernel_size +
                                ic * args->kernel_size * args->kernel_size +
                                x * args->kernel_size + y;

                            int8_t input_val = input[input_idx];
                            int8_t weight_val = weights[weight_idx];
                            int32_t product =
                                ((int32_t)weight_val * input_val);

                            accum += product;
                        }
                    }
                }

                int out_idx =
                    oc * output_height * output_width + i * output_width + j;
                int32_t scaled = accum >> args->relu_shift;
                
                if (lastLayer) {
                    // Sigmoid+0.5 threshold == (logit >= 0)
                    output[out_idx] = (accum >= 0);
                } else {
                    // BEGIN: RELU
                    output[out_idx] = (int8_t) CLAMP8_RELU(scaled);
                }
            }
        }
    }
}


void test_pointWise(int inC, int H, int W, int outC, int layer_ID, int lastLayer)
{
    // kernel dim for pointwise == inChannel * outChannel
    // for the input of the pointwise HW we need to flatten into outChannel * inChannel
    // 8x8x8 -> 8x8x32
    int scale = scalePW[layer_ID];
    
    int kernel_size = 1;
    conv2d_args args;
    args.in_channels = inC;
    args.in_height = H;
    args.in_width = W;
    args.out_channels = outC;
    args.relu_shift = scale;
    args.kernel_size = kernel_size;

    int out_dim = OUTPUT_DIMENSION(H, 0, kernel_size);

    int8_t inMap[inC * H * W];
    int8_t kernel[outC * inC];
    //int8_t** kernel2D = allocate_map_2d(outC, inC);
    int16_t bias[outC];
    int8_t outMap[outC * out_dim * out_dim];
    int8_t inMapFlat[inC * H * W];

    fill_map3D_contiguous_sequential(inMap, inC, H, W);
    //fill_map2D_contiguous_sequential(kernel, outC, inC);
    //fill_map2D_sequential(kernel2D, outC, inC);
    for (int16_t i = 0; i < outC; i++) bias[i] = biasPW[layer_ID][i];

    int idx = 0; 
    for (int oc = 0; oc < outC; oc++) {
        for (int ic = 0; ic < inC; ic++) {
            kernel[idx++] = weightsPW[layer_ID][oc][ic];
        }
    }

    conv2D(inMap, kernel, bias, outMap, &args, lastLayer);

    config_t config;
    config.range(8,0) = H;       // map size
    config.range(13,9) = layer_ID;      // layer id
    config.range(17,14) = inC / 8;    // input channels
    config.range(23,18) = lastLayer ? 1 : (outC / 8);    // output channels
    config.range(24,24) = lastLayer;     // last layer flag


    hls::stream<bus64_t> strm_in;
    hls::stream<bus64_t> strm_out;

    transpose_YXZ_contiguous(inMap, inMapFlat, inC, H, W);
    for (int i = 0; i < (inC * H * W) / 8; i++) {
        bus64_t tmp;
        tmp.data.range(63, 0) = pack8bytes(&inMapFlat[i * 8]);
        strm_in.write(tmp);
    }

    //HW_tb_layerPW(strm_in, strm_out, config, bias, kernel2D);
    HW_layerPW(strm_in, strm_out, config);
    transpose_and_compare_hls_bus64_t(outMap, strm_out, lastLayer ? 1 : outC, H, W);
}


/*
1  - 256x256x8
2  - 256x256x32
3  - 128x128x8
4  - 128x128x32
5  - 64x64x8
6  - 64x64x32
7  - 32x32x16
8  - 32x32x64
9  - 16x16x16
9  - 16x16x64
10 - 256x256x1
*/
int main()
{
    printf("*************************************\n");
    printf("*----------TESTBENCH START----------*\n");
    printf("*************************************\n");


    test_pointWise(8, 256, 256, 32, 1, 0);
    test_pointWise(32, 256, 256, 8, 2, 0);

    test_pointWise(8, 128, 128, 32, 3, 0);
    test_pointWise(32, 128, 128, 8, 4, 0);

    test_pointWise(8, 64, 64, 32, 5, 0);
    test_pointWise(32, 64, 64, 8, 6, 0);

    test_pointWise(16, 32, 32, 64, 7, 0);
    test_pointWise(64, 32, 32, 16, 8, 0);

    test_pointWise(16, 16, 16, 64, 9, 0);
    test_pointWise(64, 16, 16, 16, 10, 0);

    // Classifier
    test_pointWise(8, 256, 256, 1, 24, 1);

    printf("*************************************\n");
    printf("*----------TESTBENCH FINISH---------*\n");
    printf("*************************************\n");
}
