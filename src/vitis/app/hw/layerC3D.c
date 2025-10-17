#include "layerC3D.h"
#include "hls_utils.h"
#include <stdint.h>

int hw_layer_c3d(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN_,
    uint32_t COUT,
    uint32_t UPSAMPLE,
    uint32_t FIRST_LAYER,
    uint32_t SKIP_CON,
    int8_t *in_buf,
    int8_t *skip_buf,
    int8_t *out_buf
) {
    /* 8-bit channels on 64-bit AXIS => 8 values/beat.
    * So "groups" = channels / 8.
    */
    const uint32_t HW_CIN = (FIRST_LAYER == 1) ? 8 : CIN_;
    const uint32_t MAP_SIZE_OUT = MAP_SIZE * (1 + UPSAMPLE);
    const uint32_t BYTES_IN = (MAP_SIZE*MAP_SIZE*CIN_);
    const uint32_t BYTES_SKIP = BYTES_IN;
    const uint32_t BYTES_OUT = (MAP_SIZE_OUT*MAP_SIZE_OUT*COUT);
    
    log_info("Input buffer:  0x%08x, size: %d bytes\r\n", in_buf, BYTES_IN);
    log_info("Output buffer: 0x%08x, size: %d bytes\r\n", out_buf, BYTES_OUT);

    /* Reset DMA before use */
    XAxiDma_Reset(DmaC3D);
    while (!XAxiDma_ResetIsDone(DmaC3D)) {
        log_info("Waiting for DMA reset...\r\n");
    }
    if (SKIP_CON) {
        XAxiDma_Reset(DmaC3D_SKIP);
        while (!XAxiDma_ResetIsDone(DmaC3D_SKIP)) {
            log_info("Waiting for SKIP DMA reset...\r\n");
        }
    }
    log_info("DMA reset completed\r\n");

    /* Cache ops */
    Xil_DCacheFlushRange((UINTPTR)in_buf,  BYTES_IN);
    if (SKIP_CON) Xil_DCacheFlushRange((UINTPTR)skip_buf, BYTES_SKIP);
    Xil_DCacheInvalidateRange((UINTPTR)out_buf, BYTES_OUT);

    uint32_t cfg = pack_c3d_cfg(
        MAP_SIZE, 
        LAYER_ID,
        (HW_CIN/8), 
        (COUT/8), 
        UPSAMPLE, 
        FIRST_LAYER,
        SKIP_CON
    );

    log_debug("LAYER_C3D: %dx%dx%d -> %dx%dx%d; layerid:%d; upsample:%d; first:%d; skip_con:%d", 
        MAP_SIZE, MAP_SIZE, CIN_, 
        MAP_SIZE_OUT, MAP_SIZE_OUT, COUT,
        LAYER_ID, UPSAMPLE, FIRST_LAYER, SKIP_CON
    );


    write_cfg_and_start(HLS_C3D_BASEADDR, cfg);
    log_info("HLS configured and started\r\n");
    log_info("Starting DMA transfers - S2MM first, then MM2S...\r\n");

    /* start S2MM to receive processed data */
    if (dma_s2mm_start(DmaC3D, (UINTPTR)out_buf, BYTES_OUT) != XST_SUCCESS) {
        log_error("S2MM start failed\r\n");
        cleanup_platform(); 
        return -2;
    }
    log_info("S2MM started (receiving processed data)\r\n");

    if (dma_mm2s_start(DmaC3D, (UINTPTR)in_buf, BYTES_IN) != XST_SUCCESS) {
        log_error("MM2S start failed\r\n");
        cleanup_platform(); 
        return -3;
    }
    log_info("MM2S started (feeding data to HLS)\r\n");

    if (SKIP_CON) {
        if (!skip_buf) {
            log_error("skip_buf is NULL but SKIP is enabled\r\n");
            return -4;
        }
        if (dma_mm2s_start(DmaC3D_SKIP, (UINTPTR)skip_buf, BYTES_SKIP) != XST_SUCCESS) {
            log_error("MM2S (skip) start failed\r\n");
            return -3;
        }
    }
    log_info("MM2S started (feeding data to HLS)\r\n");
    
    log_info("Waiting for DMA transfers to complete...\r\n");
    while (
        XAxiDma_Busy(DmaC3D, XAXIDMA_DEVICE_TO_DMA) || 
        XAxiDma_Busy(DmaC3D, XAXIDMA_DMA_TO_DEVICE) ||
        (SKIP_CON && XAxiDma_Busy(DmaC3D_SKIP, XAXIDMA_DMA_TO_DEVICE))
    ) {
        // WAIT
    }
    log_info("DMA transfers completed\r\n");


    /* Optional: also check if HLS is done */
    // for (volatile uint32_t t = 0; t < 1000000U; ++t) {
    //     if (done(HLS_C3D_BASEADDR)) {
    //         DEBUG_INFO("HLS processing completed\r\n");
    //         break;
    //     }
    // }

    /* Check HLS status */
    uint32_t hls_status = Xil_In32(HLS_C3D_BASEADDR + HLS_AP_CTRL);
    log_info("Final HLS status: 0x%08x\r\n", hls_status);
    return 0;
}