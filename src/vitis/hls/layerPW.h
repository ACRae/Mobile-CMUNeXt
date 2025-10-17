// layerPW.h
#include "ap_int.h"
#include "hls_stream.h"
#include <ap_axi_sdata.h>
#include "hls_math.h"
#include "layers.h"
#include "FIVES_weightsPW.h"
#include "FIVES_biasPW.h"
#include "FIVES_scalePW.h"

void HW_layerPW(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
);
