// layerDW.h

#include "ap_int.h"
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include "hls_math.h"
#include "layers.h"
#include "FIVES_weightsDW.h"
#include "FIVES_biasDW.h"
#include "FIVES_scaleRES.h"
#include "FIVES_scaleDW.h"

void HW_readData(
    hls::stream<bus64_t> &strm_in,
    hls::stream<ap_uint<64>> &fifo_out,
    hls::stream<ap_uint<64>> &res_con,
    ap_uint<32> config
);

void HW_depthWise(
    hls::stream<ap_uint<64>> &fifo_in,
    hls::stream<ap_uint<256>> &fifo_out,
    hls::stream<ap_uint<64>> &res_con,
    ap_uint<32> config
);

void HW_layerDW(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
);