#include "ap_int.h"
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include "hls_math.h"
#include "layers.h"
#include "FIVES_bias3D.h"
#include "FIVES_weights3D.h"
#include "FIVES_scaleSKIP.h"
#include "FIVES_scale3D.h"

void HW_upsample(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &skip_con_in,
    hls::stream<ap_uint<64>> &fifo_out,
    config_t config
);


void HW_readData(
    hls::stream<ap_uint<64>> &fifo_in, //[H][W][C]
    hls::stream<ap_uint<64>> &fifo_out,
    ap_uint<32> config
);


void HW_layerC3D(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &skip_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
);