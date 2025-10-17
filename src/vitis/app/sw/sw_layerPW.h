#include <stdint.h>
#define CLAMP8_RELU(x) ((x) > INT8_MAX ? INT8_MAX : ((x) < 0 ? 0 : (x)))

void sw_layer_pw(
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t LAST_LAYER,
    int8_t *in_buf,   // [H][W][InC]  (HWC)
    int8_t *out_buf   // [H][W][OutC] (HWC)
);