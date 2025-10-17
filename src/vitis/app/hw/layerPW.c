#include "layerPW.h"

int hw_layer_pw(
    XAxiDma *DmaPW,
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t LAST_LAYER,
    int8_t *in_buf,
    int8_t *out_buf
){
    /* 8-bit channels on 64-bit AXIS => 8 values/beat.
    * So "groups" = channels / 8.
    */
    const uint32_t GROUP_COUT = LAST_LAYER ? 1 : (COUT/8);
    const uint32_t BYTES_IN = (MAP_SIZE*MAP_SIZE*CIN);
    const uint32_t BYTES_OUT = (MAP_SIZE*MAP_SIZE*COUT);
    
    log_info("Input buffer:  0x%08x, size: %d bytes\r\n", in_buf, BYTES_IN);
    log_info("Output buffer: 0x%08x, size: %d bytes\r\n", out_buf, BYTES_OUT);

    /* Reset DMA before use */
    XAxiDma_Reset(DmaPW);
    while (!XAxiDma_ResetIsDone(DmaPW)) {
        log_info("Waiting for DMA reset...\r\n");
    }
    log_info("DMA reset completed\r\n");

    /* Cache ops */
    Xil_DCacheFlushRange((UINTPTR)in_buf, BYTES_IN);
    Xil_DCacheInvalidateRange((UINTPTR)out_buf, BYTES_OUT);

    uint32_t cfg = pack_pw_cfg(
        MAP_SIZE,
        LAYER_ID,
        (CIN/8),
        GROUP_COUT,
        LAST_LAYER
    );

    log_debug("LAYER_PW: %dx%dx%d -> %dx%dx%d; layerid:%d; last_layer:%d", 
        MAP_SIZE, MAP_SIZE, CIN, 
        MAP_SIZE, MAP_SIZE, COUT,
        LAYER_ID, LAST_LAYER
    );

    write_cfg_and_start(HLS_PW_BASEADDR, cfg);
    log_info("HLS configured and started\r\n");

    log_info("Starting DMA transfers - S2MM first, then MM2S...\r\n");

    /* start S2MM to receive processed data */
    if (dma_s2mm_start(DmaPW, (UINTPTR)out_buf, BYTES_OUT) != XST_SUCCESS) {
        log_error("S2MM start failed\r\n");
        cleanup_platform(); 
        return -2;
    }
    log_info("S2MM started (receiving processed data)\r\n");

    
    if (dma_mm2s_start(DmaPW, (UINTPTR)in_buf, BYTES_IN) != XST_SUCCESS) {
        log_error("MM2S start failed\r\n");
        cleanup_platform(); 
        return -3;
    }
    log_info("MM2S started (feeding data to HLS)\r\n");
    

    log_info("Waiting for DMA transfers to complete...\r\n");
    while (XAxiDma_Busy(DmaPW, XAXIDMA_DEVICE_TO_DMA) || XAxiDma_Busy(DmaPW, XAXIDMA_DMA_TO_DEVICE)) 
    {
        // WAIT
    }
    log_info("DMA transfers completed\r\n");


    /* Optional: also check if HLS is done */
    // for (volatile uint32_t t = 0; t < 1000000U; ++t) {
    //     if (done(HLS_PW_BASEADDR)) {
    //         DEBUG_INFO("HLS processing completed\r\n");
    //         break;
    //     }
    // }

    /* Check HLS status */
    uint32_t hls_status = Xil_In32(HLS_PW_BASEADDR + HLS_AP_CTRL);
    log_info("Final HLS status: 0x%08x\r\n", hls_status);
    return 0;
}
