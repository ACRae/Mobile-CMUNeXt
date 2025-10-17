#ifndef HLS_UTILS_H
#define HLS_UTILS_H

#include "debug.h"
#include "xaxidma.h"
#include "xparameters.h"
#include <stdio.h>

/* HLS s_axilite offsets (Vivado HLS default map) */
#define HLS_AP_CTRL      0x00  // [0]=ap_start, [1]=ap_done, [2]=ap_idle, [7]=auto_restart
#define HLS_GIE          0x04
#define HLS_IER          0x08
#define HLS_ISR          0x0C
#define HLS_ARG_CONFIG   0x10  // first argument 'config' (32-bit)

/* ---- HLS control helpers ---- */
static inline void write_cfg_and_start(u32 hls_baseaddr, uint32_t cfg)
{
    log_info("Writing config 0x%08x to HLS at 0x%08x\r\n", cfg, hls_baseaddr);
    Xil_Out32(hls_baseaddr + HLS_ARG_CONFIG, cfg);
    /* ap_start = 1, (no auto_restart for single frame) */
    Xil_Out32(hls_baseaddr + HLS_AP_CTRL, 0x1);
}

static inline int done(u32 hls_baseaddr)
{
    return (Xil_In32(hls_baseaddr + HLS_AP_CTRL) >> 1) & 0x1;
}

int dma_mm2s_start(XAxiDma *D, uintptr_t buf, uint32_t nbytes);
int dma_s2mm_start(XAxiDma *D, uintptr_t buf, uint32_t nbytes);
int dma_init(XAxiDma *D, uintptr_t dev_id);
void sample_buffer(int8_t* out_buf, uint32_t nbytes);

#endif /* HLS_UTILS_H */
