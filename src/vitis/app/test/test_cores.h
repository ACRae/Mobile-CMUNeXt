#include "xaxidma.h"
#include <stdint.h>

void test_layerPW(
    XAxiDma *DmaPW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);


void test_layerDW(
    XAxiDma *DmaDW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);


void test_layerC3D(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    int8_t *hw_in_buf,
    int8_t *hw_skip_buf,   // pass NULL if no skip
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);



void test_all_configs_layerPW(
    XAxiDma *DmaPW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);


void test_all_configs_layerDW(
    XAxiDma *DmaDW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);

void test_all_configs_layerC3D(
    XAxiDma *DmaC3D, 
    XAxiDma *DmaC3D_SKIP,
    int8_t *hw_in_buf,
    int8_t *hw_skip_buf,   // provide valid data for skip cases
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
);
