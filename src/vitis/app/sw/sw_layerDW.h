#include <stdint.h>
#include <string.h>
#include "sw_conv.h"
#include "FIVES_biasDW.h"
#include "FIVES_weightsDW.h"
#include "FIVES_scaleRES.h"
#include "FIVES_scaleDW.h"


#define MAXPOOL_STRIDE 2
#define MAXPOOL_KERNEL 2
#define MAX(a, b) (((data_t)(a) > (data_t)(b)) ? (a) : (b))


int sw_layer_dw(
    uint32_t MAP_SIZE,
    uint32_t CIN,
    uint32_t LAYER_ID,
    uint32_t KERNEL_SIZE,
    uint32_t PAD,
    uint32_t MAXPOOL,
    int8_t *in_buf,
    int8_t *out_buf
);