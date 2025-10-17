#include "layerC3D.h"
#include "layerDW.h"
#include "layerPW.h"


void MobileCMUNeXt(
    XAxiDma *DmaPW,
    XAxiDma *DmaDW,
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    int8_t  *bufA,           // working buffer A
    int8_t  *bufB,           // working buffer B
    int8_t  *outBuff,
    int8_t  *skip_buf1,
    int8_t  *skip_buf2,
    int8_t  *skip_buf3,
    int8_t  *skip_buf4
);