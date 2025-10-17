#ifndef LAYERPW_H
#define LAYERPW_H

#include <stdint.h>
#include "xhw_layerpw.h"
#include "xhw_layerpw_hw.h"
#include "xaxidma.h"
#include "../utils/hls_utils.h"
#include "../platform.h"

#define DMA_PW_DEV_ID    XPAR_AXI_DMA_0_BASEADDR // AXI_DMA_0

// HLS Module Base Addresses (for control registers)
#define HLS_PW_BASEADDR    XPAR_XHW_LAYERPW_0_BASEADDR   // 0xa0010000


static inline uint32_t pack_pw_cfg(
    uint32_t map_size,    // bits 8–0
    uint32_t layer_id,    // bits 13–9
    uint32_t channel,     // bits 17–14 (input channels)
    uint32_t filters,     // bits 23–18 (output channels)
    uint32_t lastLayer    // bit 24
)
{
    uint32_t c = 0;
    c |= (map_size   & 0x1FFu) << 0;   // bits 8–0
    c |= (layer_id   & 0x1Fu)  << 9;   // bits 13–9
    c |= (channel    & 0xFu)   << 14;  // bits 17–14
    c |= (filters    & 0x3Fu)  << 18;  // bits 23–18
    c |= (lastLayer  & 0x1u)   << 24;  // bit 24
    return c;
}


int hw_layer_pw(
    XAxiDma *DmaPW,
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t LAST_LAYER,
    int8_t *in_buf,
    int8_t *out_buf
);


#endif