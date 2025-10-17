#include <stdint.h>
#include <string.h>
#include "FIVES_bias3D.h"
#include "FIVES_weights3D.h"
#include "FIVES_scaleSKIP.h"
#include "FIVES_scale3D.h"

void sw_layerC3D(
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t UPSAMPLE,     // 0/1
    uint32_t FIRST_LAYER,  // 0/1
    uint32_t SKIP_CON,     // 0/1
    const int8_t *in_buf,  // [H][W][CIN]
    const int8_t *skip_buf,// same shape as in_buf (ignored if FIRST_LAYER or UPSAMPLE==1)
    int8_t *out_buf        // [Hout][Wout][COUT]
);