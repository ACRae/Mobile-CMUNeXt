#include <cstdint>
#include <stdio.h>
#include "ap_int.h"
#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "layers.h"
#include "utils.h"
#include "layerC3D.h"

#define KERNEL_SIZE 3
#define PAD 1


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



void SW_tb_conv3D(
    int8_t* in,
    int8_t* out,
    config_t config,
    int16_t *bias,
    int8_t *weights
) {



    int map_size =      config.range(8,0).to_uint();        // input dimensions
    int layer_ID =      config.range(13,9).to_uint();       // layer identifier
    int channel =       config.range(15,14).to_uint()*8;       // channel
    int filters =       config.range(17,16).to_uint()*8;
    bool upsample =     config.range(18,18).to_uint();
    bool firstLayer =   config.range(19,19).to_uint();
    bool skipCon =      config.range(20,20).to_uint();

    int scale = scale3D[layer_ID];



    int map_size_upsample;
    map_size_upsample = upsample ? map_size * 2 : map_size;

    int map_size_up_pad = map_size_upsample+2;

    conv2d_args args = {
        .in_channels      = channel,
        .in_height        = map_size_up_pad,
        .in_width         = map_size_up_pad,
        .out_channels     = filters,
        .kernel_size      = KERNEL_SIZE,
        .relu_shift = scale
    };

    sw_conv2D_relu_nopad(in, weights, bias, out, &args);
}


void bilinear_upsample_fixed(
    int8_t* input_image, // [C][H][W]
    int8_t* output_image,
    int input_height,
    int input_width,
    int num_channels
) {
    int iy0, ix0, iy1, ix1;
    int y_lerp, x_lerp;
    int output_height = input_height * 2;
    int output_width = input_width * 2;

    for (int oh = 0; oh < output_height; ++oh) {
        if (oh == 0) {
            iy0 = 0;
            iy1 = 0;
        } else {
            iy0 = (oh - 1) >> 1;
            iy1 = iy0 + 1;
        }
        if (iy1 >= input_height) iy1 = input_height - 1;
        y_lerp = (oh % 2 == 0) ? 3 : 1;

        for (int ow = 0; ow < output_width; ++ow) {
            if (ow == 0) {
                ix0 = 0;
                ix1 = 0;
            } else {
                ix0 = (ow - 1) >> 1;
                ix1 = ix0 + 1;
            }
            if (ix1 >= input_width) ix1 = input_width - 1;
            x_lerp = (ow % 2 == 0) ? 3 : 1;

            // Loop over all channels for the current output pixel
            for (int c = 0; c < num_channels; ++c) {
                // Calculate base offset for this channel
                int channel_offset = c * input_height * input_width;

                // Calculate indices for the four corner pixels in [C][H][W] layout
                int idx00 = channel_offset + iy0 * input_width + ix0;
                int idx01 = channel_offset + iy0 * input_width + ix1;
                int idx10 = channel_offset + iy1 * input_width + ix0;
                int idx11 = channel_offset + iy1 * input_width + ix1;

                // Perform bilinear interpolation
                int16_t top    = input_image[idx00] * (4 - x_lerp) + input_image[idx01] * x_lerp;
                int16_t bottom = input_image[idx10] * (4 - x_lerp) + input_image[idx11] * x_lerp;
                int16_t value  = top * (4 - y_lerp) + bottom * y_lerp;

                value = value >> 4;
                if (value > 127) value = 127;
                if (value < -128) value = -128;

                // Calculate output index in [C][H][W] layout
                int out_idx = c * output_height * output_width + oh * output_width + ow;
                output_image[out_idx] = (int8_t)value;
            }
        }
    }
}

void SW_tb_upsample(
    int8_t* input_image, // [C][H][W]
    int8_t* res_con,
    int8_t* output_image,
    int map_size,
    int channels,
    bool firstLayer,
    bool upsample,
    bool residual,
    int layer_ID
) {
    int inC = (firstLayer == 1) ? 3 : channels;
    if(upsample) {
        bilinear_upsample_fixed(input_image, output_image, map_size, map_size, channels);
    }
    else {
        int size = map_size * map_size * inC;
        for (int i = 0; i < size; i++) {
            int8_t value = input_image[i];
            if(residual) {
                int8_t v1 = safe_shift(input_image[i], scaleSKIP[layer_ID][1]);
                int8_t v2 = safe_shift(res_con[i], scaleSKIP[layer_ID][0]);
                int16_t sum = v1 + v2;
                if (sum > 127) sum = 127;
                if (sum < -128) sum = -128;
                value = (int8_t)sum;
            }
            output_image[i] = value;
        }
    }
}


void SW_readData(
    int8_t* input, // [C][H][W]
    int8_t* output, // [C][H][W]
    int in_channel,
    int map_size,
    bool firstLayer
){
    if (firstLayer){
        // PAD TO 258 AND ADD ZEROS TO EXPAND FROM 3 CHANNELS TO 8
        int input_h = 256;
        int input_w = 256;
        int input_c = 3;
        int output_h = 258;
        int output_w = 258;
        int output_c = 8;

        // Initialize output to zeros
        for (int c = 0; c < output_c; c++) {
            for (int h = 0; h < output_h; h++) {
                for (int w = 0; w < output_w; w++) {
                    output[c * output_h * output_w + h * output_w + w] = 0;
                }
            }
        }

        // Copy input data to output with padding (1 pixel on all sides)
        // Only copy to first 3 channels, channels 4-7 remain zero
        for (int c = 0; c < input_c; c++) {
            for (int h = 0; h < input_h; h++) {
                for (int w = 0; w < input_w; w++) {
                    int input_idx = c * input_h * input_w + h * input_w + w;
                    int output_idx = c * output_h * output_w + (h + 1) * output_w + (w + 1);
                    output[output_idx] = input[input_idx];
                }
            }
        }

    } else {
        pad_tensor_3d(output, input, in_channel, map_size, map_size, 1);
    }
}

void SW_layerC3D(
    int8_t* in,
    int8_t* res,
    int8_t* out,
    config_t config,
    int16_t *bias,
    int8_t *weight
){
    int map_size =      config.range(8,0).to_uint();        // input dimensions
    int layer_ID =      config.range(13,9).to_uint();       // layer identifier
    int channel =       config.range(15,14).to_uint()*8;       // channel
    int filters =       config.range(17,16).to_uint()*8;
    bool upsample =     config.range(18,18).to_uint();
    bool firstLayer =   config.range(19,19).to_uint();
    bool skipCon =      config.range(20,20).to_uint();
    hls::stream<bus64_t> strm_in;
    hls::stream<bus64_t> res_in;

    int map_size_upsample;
    map_size_upsample = upsample ? map_size * 2 : map_size;

    int map_size_up_pad = map_size_upsample+2;

    int8_t up_out[channel * map_size_upsample * map_size_upsample];
    SW_tb_upsample(in, res, up_out, map_size, channel, firstLayer, upsample, skipCon, layer_ID);

    int8_t pad_out[channel * map_size_up_pad * map_size_up_pad];
    SW_readData(up_out, pad_out, channel, map_size_upsample, firstLayer);

    SW_tb_conv3D(pad_out, out, config, bias, weight);
}


void test_layerC3D(
    int map_size,
    int inC,
    int outC,
    bool firstLayer,
    bool upsample,
    bool residual,
    int layer_id
) {
    hls::stream<bus64_t> strm_in;
    hls::stream<bus64_t> res_in;
    hls::stream<bus64_t> str_out;

    int inC_S = (firstLayer == 1) ? 3 : inC;

    int8_t inMap[inC_S * map_size * map_size]; // Input [C][H][W]
    int8_t flatten_map[inC_S * map_size * map_size]; // Flattened input for stream

    int map_size_upsample;
    map_size_upsample = upsample ? map_size * 2 : map_size;

    int8_t outMap[outC * map_size_upsample * map_size_upsample]; // Input [C][H][W]


    config_t config;
    // config.range(8,0) = map_size;   // map size
    // config.range(13,9) = layer_id;  // layer id
    // config.range(19,18) = (inC / 8); // input channel
    // config.range(21,20) = (outC / 8); // output channel
    // config.range(22,22) = upsample ? 1 : 0; // upsample
    // config.range(24,24) = firstLayer ? 1 : 0; // first layer
    // config.range(25,25) = residual ? 1 : 0; // Sum skip con

    config.range(8,0) = map_size;   // map size
    config.range(13,9) = layer_id;  // layer id
    config.range(15,14) = (inC / 8); // input channel
    config.range(17,16) = (outC / 8); // output channel
    config.range(18,18) = upsample ? 1 : 0; // upsample
    config.range(19,19) = firstLayer ? 1 : 0; // first layer
    config.range(20,20) = residual ? 1 : 0; // Sum skip con

    // Fill and flatten input
    fill_map3D_contiguous_sequential(inMap, inC_S, map_size, map_size);
    transpose_YXZ_contiguous(inMap, flatten_map, inC_S, map_size, map_size);
    

    

    int16_t bias[outC];
    for (int16_t i = 0; i < outC; i++) bias[i] = bias3D[layer_id][i];

    int8_t kernel[KERNEL_SIZE*KERNEL_SIZE*outC*inC];
    int idx = 0;
    for (int oc = 0; oc < outC; ++oc)
        for (int ic = 0; ic < inC; ++ic)
            for (int k1 = 0; k1 < KERNEL_SIZE; ++k1)
                for (int k2 = 0; k2 < KERNEL_SIZE; ++k2)
                    kernel[idx++] = weights3D[layer_id][k1][k2][oc][ic]; 

    // Stream input to DUT
    for (int i = 0; i < (inC_S * map_size * map_size) / 8; i++) {
        bus64_t tmp;
        tmp.data.range(63, 0) = pack8bytes(&flatten_map[i * 8]);

        tmp.keep = 0xFF;
        tmp.strb = 0xFF;
        if(i == ((inC_S * map_size * map_size) / 8)-1) tmp.last = 1;
        else tmp.last = 0;

        strm_in.write(tmp);
        if (residual) res_in.write(tmp);
    }

    SW_layerC3D(inMap, inMap, outMap, config, bias, kernel);
    HW_layerC3D(strm_in, res_in, str_out, config);
    transpose_and_compare_hls_bus64_t(outMap, str_out, outC, map_size_upsample, map_size_upsample);
}



int main()
{
    printf("*************************************\n");
    printf("*----------TESTBENCH START----------*\n");
    printf("*************************************\n");

    // ---------- Encoders ------------ //
    test_layerC3D(256, 8, 8, 1, 0, 0, 0); // takes a little bit
    test_layerC3D(256, 8, 8, 0, 0, 0, 1);
    test_layerC3D(128, 8, 8, 0, 0, 0, 2);
    test_layerC3D(64, 8, 16, 0, 0, 0, 3);
    test_layerC3D(32, 16, 16, 0, 0, 0, 4);
    test_layerC3D(16, 16, 24, 0, 0, 0, 5);
    // ---------- Decoders ------------ //
    test_layerC3D(16, 24, 16, 0, 1, 0, 6); // upsample 16x16x24 -> 32x32x16
    test_layerC3D(32, 16, 16, 0, 0, 1, 7); // skip con 32x32x16 -> 32x32x16
    test_layerC3D(32, 16, 16, 0, 1, 0, 8); // upsample 32x32x16 -> 64x64x16
    test_layerC3D(64, 16, 16, 0, 0, 1, 9); // skip con 64x64x16 -> 64x64x16
    test_layerC3D(64, 16, 8, 0, 1, 0, 10);  // upsample 64x64x16 -> 128x128x8
    test_layerC3D(128, 8, 8, 0, 0, 1, 11);  // skip con 128x128x8 -> 128x128x8
    test_layerC3D(128, 8, 8, 0, 1, 0, 12);  // upsample 128x128x8 -> 256x256x8
    test_layerC3D(256, 8, 8, 0, 0, 1, 13);  // skip con 256x256x8 -> 256x256x8


    printf("*************************************\n");
    printf("*----------TESTBENCH FINISH---------*\n");
    printf("*************************************\n");
    return 0;
}
