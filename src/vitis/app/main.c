#include "xaxidma.h"
#include "xparameters.h"
#include <stdint.h>
#include <stdio.h>
#include <xstatus.h>
#include "platform.h"
#include "xil_printf.h"
#include "hw/layerC3D.h"
#include "hw/layerDW.h"
#include "hw/layerPW.h"
#include "utils/hls_utils.h"
#include "utils/time_utils.h"
#include "utils/debug.h"
#include "utils/sw_utils.h"
#include "sw/sw_layerPW.h"
#include "test/test_cores.h"
#include "hw/Mobile-CMUNeXt.h"
#include "images/embedded/192_N_quant_int8.h"
#include "images/embedded/20_A_quant_int8.h"
#include "images/embedded/85_D_quant_int8.h"
#include "images/embedded/86_D_quant_int8.h"
#include "images/embedded/87_D_quant_int8.h"
#include "images/embedded/88_D_quant_int8.h"
#include "images/embedded/89_D_quant_int8.h"

static XAxiDma DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP;


static int DmaInitAll()
{
    if(dma_init(&DmaPW, DMA_PW_DEV_ID) != XST_SUCCESS) {
        log_error("PW DMA init failed\r\n");
        cleanup_platform();
        return XST_FAILURE;
    }
    if(dma_init(&DmaDW, DMA_DW_DEV_ID) != XST_SUCCESS)  {
        log_error("DW DMA init failed\r\n");
        cleanup_platform();
        return XST_FAILURE;
    }
    if(dma_init(&DmaC3D, DMA_C3D_DEV_ID) != XST_SUCCESS)  {
        log_error("C3D DMA init failed\r\n");
        cleanup_platform();
        return XST_FAILURE;
    }
    if(dma_init(&DmaC3D_SKIP, DMA_C3D_SKIP_DEV_ID) != XST_SUCCESS)  {
        log_error("C3D SKIP DMA init failed\r\n");
        cleanup_platform();
        return XST_FAILURE;
    }
    return XST_SUCCESS;
}


#define MAX_BUFFER 256*256*32
#define IMAGE_SIZE 256*256*3

/* Aligned buffers for DMA - try larger alignment */
static int8_t hw_buffA[MAX_BUFFER]  __attribute__((aligned(64)));
static int8_t hw_buffB[MAX_BUFFER] __attribute__((aligned(64)));
static int8_t out_buff[256*256*1] __attribute__((aligned(64)));

static int8_t sw_buffA[MAX_BUFFER] __attribute__((aligned(64)));

/* Define buffers for skip connections */
static int8_t skip_buf4[32*32*16]  __attribute__((aligned(64)));
static int8_t skip_buf3[64*64*16]  __attribute__((aligned(64)));
static int8_t skip_buf2[128*128*8]  __attribute__((aligned(64)));
static int8_t skip_buf1[256*256*8]  __attribute__((aligned(64)));



void infere(
    const char* name,
    unsigned char *img
){
    memset(hw_buffA, 0, sizeof(hw_buffA));
    memset(hw_buffB, 0, sizeof(hw_buffB));
    memset(sw_buffA, 0, sizeof(sw_buffA));

    memcpy(hw_buffA, img, IMAGE_SIZE);
    memcpy(sw_buffA, img, IMAGE_SIZE);

    xil_printf("INFERRING IMAGE: %s\n", name);
    TIME_FUNCTION(
        "Mobile-CMUNeXt",
        MobileCMUNeXt(
            &DmaPW, &DmaDW, &DmaC3D, &DmaC3D_SKIP,
            hw_buffA, hw_buffB, out_buff,
            skip_buf1, skip_buf2, skip_buf3, skip_buf4
        );
    );
    uart_print_bits_bin_rows(out_buff, 256, 256);
}


void test_network() {
    test_all_configs_layerPW(&DmaPW, hw_buffA, hw_buffB, sw_buffA);
    test_all_configs_layerDW(&DmaDW,  hw_buffA, hw_buffB, sw_buffA);
    test_all_configs_layerC3D(&DmaC3D, &DmaC3D_SKIP, hw_buffA, skip_buf1, hw_buffB, sw_buffA);
}


int main()
{
    init_platform();
    log_set_level(LOG_LEVEL_DEBUG);
    log_disable();

    if(DmaInitAll() == XST_FAILURE) return -1;
    
    //test_network();

    infere("IMG_192_N", img_192_N_quant_int8);
    infere("IMG_20_A", img_20_A_quant_int8);
    infere("IMG_85_D", img_85_D_quant_int8);
    infere("IMG_86_D", img_86_D_quant_int8);
    infere("IMG_87_D", img_87_D_quant_int8);
    infere("IMG_88_D", img_88_D_quant_int8);
    infere("IMG_89_D", img_89_D_quant_int8);

    cleanup_platform();
    return 0;
}