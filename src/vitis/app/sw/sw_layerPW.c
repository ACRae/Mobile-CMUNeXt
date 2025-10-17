#include "sw_layerPW.h"
#include "FIVES_biasPW.h"
#include "FIVES_weightsPW.h"
#include "FIVES_scalePW.h"
#include <stdint.h>

#define KERNEL_SIZE 1

void sw_layer_pw(
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t LAST_LAYER,
    int8_t *in_buf,   // [H][W][InC]  (HWC)
    int8_t *out_buf   // [H][W][OutC] (HWC)
) {
    int SCALE = scalePW[LAYER_ID];
    for (unsigned int i = 0; i < MAP_SIZE; ++i) {
        for (unsigned int j = 0; j < MAP_SIZE; ++j) {
            const int in_hw_base = (i * MAP_SIZE + j) * CIN;   // HWC base for this pixel
            const int out_hw_base = (i * MAP_SIZE + j) * COUT;

            for (unsigned int oc = 0; oc < COUT; ++oc) {
                int32_t accum = (int32_t)biasPW[LAYER_ID][oc];

                // 1x1 kernel => sum over input channels only
                for (unsigned int ic = 0; ic < CIN; ++ic) {
                    const int8_t input_val  = in_buf[in_hw_base + ic];         // [H][W][ic]
                    const int8_t weight_val = weightsPW[LAYER_ID][oc][ic];     // [oc][ic]
                    accum += (int32_t)input_val * (int32_t)weight_val;
                }

                // write [H][W][oc]
                int32_t scaled = accum >> SCALE;
                if (LAST_LAYER) {
                    // Sigmoid+0.5 threshold == (logit >= 0)
                    out_buf[out_hw_base + oc] = (accum >= 0);
                } else {
                    // BEGIN: RELU
                    out_buf[out_hw_base + oc] = (int8_t) CLAMP8_RELU(scaled);
                }
            }
        }
    }
}