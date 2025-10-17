#ifndef LAYERDW_H
#define LAYERDW_H


#include <stdint.h>
#include "xhw_layerdw.h"
#include "xhw_layerdw_hw.h"
#include "xaxidma.h"
#include "../utils/hls_utils.h"
#include "../platform.h"


#define DMA_DW_DEV_ID    XPAR_AXI_DMA_1_BASEADDR  // AXI_DMA_1
#define HLS_DW_BASEADDR   XPAR_XHW_LAYERDW_0_BASEADDR   // 0xa0030000

static inline uint32_t pack_dw_cfg(
    uint32_t map_size,
    uint32_t layer_id,
    uint32_t channel,
    uint32_t kernel_size,
    uint32_t pad,
    uint32_t maxpool)
{
    uint32_t c = 0;
    c |= (map_size    & 0x1FFu) << 0;   // bits 8–0
    c |= (layer_id    & 0x1Fu)  << 9;   // bits 13–9
    c |= (channel     & 0x3u)   << 14;  // bits 15–14
    c |= (kernel_size & 0xFu)   << 16;  // bits 19–16
    c |= (pad         & 0x7u)   << 20;  // bits 22–20
    c |= (maxpool     & 0x1u)   << 23;  // bit 23
    return c;
}


int hw_layer_dw(
    XAxiDma *DmaDW,
    uint32_t MAP_SIZE,
    uint32_t CIN,
    uint32_t LAYER_ID,
    uint32_t KERNEL_SIZE,
    uint32_t PAD,
    uint32_t MAXPOOL,
    int8_t *in_buf,
    int8_t *out_buf
);

#endif