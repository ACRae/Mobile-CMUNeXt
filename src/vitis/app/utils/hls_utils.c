#include "hls_utils.h"
#include <stdint.h>
#include <xil_printf.h>
#include <xil_types.h>


/* DMA helpers - NON-BLOCKING versions with enhanced error checking */
int dma_mm2s_start(XAxiDma *D, uintptr_t buf, uint32_t nbytes)
{
    if (buf & 0x3F) {  // 64-byte alignment check
        log_error("MM2S buffer not 64-byte aligned: 0x%08x\r\n", (uint32_t)buf);
        return XST_FAILURE;
    }
    if (nbytes & 0x7) {  // 8-byte (64-bit) alignment check
        log_error("MM2S size not 8-byte aligned: %d\r\n", nbytes);
        return XST_FAILURE;
    }
    
    log_info("MM2S: buf=0x%08x, size=%d bytes\r\n", (uint32_t)buf, nbytes);
    int s = XAxiDma_SimpleTransfer(D, buf, nbytes, XAXIDMA_DMA_TO_DEVICE);
    if (s != XST_SUCCESS) {
        log_error("MM2S transfer start failed with status %d\r\n", s);
    }
    return s;
}

int dma_s2mm_start(XAxiDma *D, uintptr_t buf, uint32_t nbytes)
{
    if (buf & 0x3F) {  // 64-byte alignment check
        log_error("S2MM buffer not 64-byte aligned: 0x%08x\r\n", (uint32_t)buf);
        return XST_FAILURE;
    }
    if (nbytes & 0x7) {  // 8-byte (64-bit) alignment check
        log_error("S2MM size not 8-byte aligned: %d\r\n", nbytes);
        return XST_FAILURE;
    }
    
    log_info("S2MM: buf=0x%08x, size=%d bytes\r\n", (uint32_t)buf, nbytes);
    int s = XAxiDma_SimpleTransfer(D, buf, nbytes, XAXIDMA_DEVICE_TO_DMA);
    if (s != XST_SUCCESS) {
        log_error("S2MM transfer start failed with status %d\r\n", s);
        uint32_t s2mm_status = XAxiDma_ReadReg(D->RegBase, XAXIDMA_RX_OFFSET + XAXIDMA_SR_OFFSET);
        log_error("S2MM Status Register: 0x%08x\r\n", s2mm_status);
    }
    return s;
}



/* Helper: configure & init one DMA instance (simple mode, no interrupts) */
int dma_init(XAxiDma *D, uintptr_t dev_id)
{
    XAxiDma_Config *cfg = XAxiDma_LookupConfig(dev_id);
    if (!cfg) {
        log_error("DMA config lookup failed for device %d\r\n", dev_id);
        return XST_FAILURE;
    }
    int s = XAxiDma_CfgInitialize(D, cfg);
    if (s != XST_SUCCESS) {
        log_error("DMA init failed with status %d\r\n", s);
        return s;
    }
    if (XAxiDma_HasSg(D)) {
        log_error("DMA has scatter-gather (we want simple mode)\r\n");
        return XST_FAILURE; // we want simple mode
    }
    return XST_SUCCESS;
}


void sample_buffer(int8_t* out_buf, uint32_t nbytes){
    /* See results */
    Xil_DCacheInvalidateRange((UINTPTR)out_buf, nbytes);
    uint64_t *out64 = (uint64_t *)out_buf;
    xil_printf("Output sample:\r\n");
    for (int i = 0; i < 8 && i < (int)(nbytes/8); ++i) {
        xil_printf(" out[%d] = 0x%016llx\r\n", i, (unsigned long long)out64[i]);
    }
}