#ifndef LAYERC3D_H
#define LAYERC3D_H


#include <stdint.h>
#include "xhw_layerc3d.h"
#include "xhw_layerc3d_hw.h"
#include "xaxidma.h"
#include "../utils/hls_utils.h"
#include "../platform.h"

#define DMA_C3D_DEV_ID    XPAR_AXI_DMA_2_BASEADDR  // AXI_DMA_2
#define DMA_C3D_SKIP_DEV_ID    XPAR_AXI_DMA_3_BASEADDR  // AXI_DMA_3


#define HLS_C3D_BASEADDR    XPAR_XHW_LAYERC3D_0_BASEADDR  // 0xa0060000


/* Pack your config words (exact fields from your spec) */
static inline uint32_t pack_c3d_cfg(
    uint32_t map_size,
    uint32_t layer_id,
    uint32_t channel,
    uint32_t filters,
    uint32_t upsample,
    uint32_t firstLayer,
    uint32_t skipCon
) {
    uint32_t c = 0;
    c |= (map_size   & 0x1FFu) << 0;   // bits 8–0
    c |= (layer_id   & 0x1Fu)  << 9;   // bits 13–9
    c |= (channel    & 0x3u)   << 14;  // bits 15–14
    c |= (filters    & 0x3u)   << 16;  // bits 17–16
    c |= (upsample   & 0x1u)   << 18;  // bit 18
    c |= (firstLayer & 0x1u)   << 19;  // bit 19
    c |= (skipCon    & 0x1u)   << 20;  // bit 20
    return c;
}


int hw_layer_c3d(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t UPSAMPLE,
    uint32_t FIRST_LAYER,
    uint32_t SKIP_CON,
    int8_t *in_buf,
    int8_t *skip_buf,
    int8_t *out_buf
);


#endif