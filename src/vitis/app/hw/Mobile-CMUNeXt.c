#include "Mobile-CMUNeXt.h"
#include "FIVES_scaleDW.h"
#include "FIVES_scalePW.h"
#include "FIVES_scale3D.h"
#include <stdint.h>
#include <string.h>

static inline void swap_i8(int8_t **a, int8_t **b){ int8_t *t=*a; *a=*b; *b=t; }

/*
 * MobileCMUNeXt block layout per depth:
 *   DW(3x3, SAME, residual inside core)   [C -> C]
 *   DW(3x3, SAME, residual inside core)   [C -> C]
 *   PW(1x1 + ReLU) expand                 [C -> 4C]
 *   PW(1x1 + ReLU) project                [4C -> C]
 * tail:
 *   C3D(3x3 SAME + ReLU, optional external skip) [C -> COUT]
 *
 * layer_id_* are IN/OUT single counters (not arrays):
 *   - *layer_id_dw  is bumped twice per depth
 *   - *layer_id_pw  is bumped twice per depth
 *   - *layer_id_c3d is bumped once at the tail
 */
void MobileCMUNeXtBlock(
    XAxiDma *DmaPW,
    XAxiDma *DmaDW,
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    int8_t  *hw_in_buf,
    int8_t  *hw_out_buf,
    int8_t  *skip_buf,          // optional skip for tail C3D
    int8_t  *layer_id_dw,       // IN/OUT: DW id counter
    int8_t  *layer_id_pw,       // IN/OUT: PW id counter
    int8_t  *layer_id_c3d,      // IN/OUT: C3D id counter
    uint32_t depth,
    uint32_t MAP_SIZE,
    uint32_t KERNEL_SIZE,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t MAXPOOL            // 0 or 1 (applied ONCE at block entry)
){
    int8_t *cur = hw_in_buf;
    int8_t *nxt = hw_out_buf;

    const uint32_t PAD = KERNEL_SIZE >> 1;   // SAME for odd K
    const uint32_t C4  = CIN << 2;           // expand factor inside block

    uint32_t dw_id  = (uint8_t)(*layer_id_dw);
    uint32_t pw_id  = (uint8_t)(*layer_id_pw);
    uint32_t c3d_id = (uint8_t)(*layer_id_c3d);

    // Effective map after the (optional) maxpool on DW #1
    const uint32_t MAP0 = MAP_SIZE;
    const uint32_t MAP1 = MAXPOOL ? (MAP_SIZE >> 1) : MAP_SIZE;

    for (uint32_t i = 0; i < depth; ++i) {
        // DW #1 @ MAP0 with MAXPOOL
        {
            const uint32_t LID   = dw_id;
            hw_layer_dw(DmaDW, i == 0 ? MAP0 : MAP1, CIN, LID, KERNEL_SIZE, PAD, i == 0 ? MAXPOOL : 0, cur, nxt);
            dw_id++;
            swap_i8(&cur, &nxt);
        }
        // DW #2 @ MAP1, NO maxpool
        {
            const uint32_t LID   = dw_id;
            hw_layer_dw(DmaDW, MAP1, CIN, LID, KERNEL_SIZE, PAD, /*MAXPOOL=*/0, cur, nxt);
            dw_id++;
            swap_i8(&cur, &nxt);
        }
        // PW expand: C -> 4C @ MAP1
        {
            const uint32_t LID   = pw_id;
            hw_layer_pw(DmaPW, MAP1, LID, CIN, C4, /*LAST=*/0, cur, nxt);
            pw_id++;
            swap_i8(&cur, &nxt);
        }
        // PW project: 4C -> C @ MAP1
        {
            const uint32_t LID   = pw_id;
            hw_layer_pw(DmaPW, MAP1, LID, C4, CIN, /*LAST=*/0, cur, nxt);
            pw_id++;
            swap_i8(&cur, &nxt);
        }
    }

    // Tail C3D: C -> COUT @ MAP1, optional external skip
    {
        const uint32_t LID       = c3d_id++;
        const uint32_t UPSAMPLE  = 0;
        const uint32_t FIRST     = 0;
        const uint32_t SKIP_CON  = 0;  // <— always zero here

        hw_layer_c3d(DmaC3D, DmaC3D_SKIP, MAP1, LID,
                     CIN, COUT, UPSAMPLE, FIRST, SKIP_CON,
                     cur, NULL, nxt);
        swap_i8(&cur, &nxt);
    }

    // Ensure final in hw_out_buf
    if (cur != hw_out_buf) {
        const size_t bytes = (size_t)MAP1 * MAP1 * COUT;
        memcpy(hw_out_buf, cur, bytes);
    }

    // If a skip stash buffer is provided, copy the final output into it for decoder fusion later
    if (skip_buf) {
        const size_t bytes = (size_t)MAP1 * MAP1 * COUT;
        memcpy(skip_buf, cur, bytes);
    }

    *layer_id_dw  = (int8_t)dw_id;
    *layer_id_pw  = (int8_t)pw_id;
    *layer_id_c3d = (int8_t)c3d_id;
}


// Upsample-by-2 (bilinear) + 3x3 SAME conv + ReLU
// Input  : [MAP_SIZE x MAP_SIZE x CIN]
// Output : [2*MAP_SIZE x 2*MAP_SIZE x COUT]
void UpConv(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,   // unused here (skip disabled), kept for API symmetry
    int8_t  *hw_in_buf,
    int8_t  *hw_out_buf,
    int8_t  *layer_id_c3d,  // IN/OUT: C3D id counter
    uint32_t MAP_SIZE,
    uint32_t CIN,
    uint32_t COUT
) {
    // Current C3D layer id and scale
    uint32_t LID   = (uint8_t)(*layer_id_c3d);

    // Configure C3D for: UPSAMPLE=1, FIRST_LAYER=0, SKIP_CON=0
    const uint32_t UPSAMPLE  = 1;
    const uint32_t FIRST     = 0;
    const uint32_t SKIP_CON  = 0;

    // hw_layer_c3d(DmaC3D, DmaC3D_SKIP, MAP, SCALE, LID,
    //              CIN, COUT, UPSAMPLE, FIRST, SKIP_CON, in, skip, out)
    (void)DmaC3D_SKIP; // not used when SKIP_CON==0
    (void)hw_layer_c3d(DmaC3D, DmaC3D_SKIP, MAP_SIZE, LID,
                       CIN, COUT, UPSAMPLE, FIRST, SKIP_CON,
                       hw_in_buf, NULL, hw_out_buf);

    // Advance C3D layer id counter
    *layer_id_c3d = (int8_t)(LID + 1);
}


/*
 * SkipFusion:
 *   C3D 3x3 SAME + ReLU with external skip:   [CIN -> CIN]
 *   PW  1x1 + ReLU expand:                    [CIN -> 4*COUT]
 *   PW  1x1 + ReLU project:                   [4*COUT -> COUT]
 */
void SkipFusion(
    XAxiDma *DmaPW,
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    int8_t  *hw_in_buf,
    int8_t  *hw_out_buf,
    int8_t  *skip_buf,          // REQUIRED for SkipFusion
    int8_t  *layer_id_pw,       // IN/OUT: PW id counter
    int8_t  *layer_id_c3d,      // IN/OUT: C3D id counter
    uint32_t MAP_SIZE,
    uint32_t CIN,
    uint32_t COUT
) {
    // --- required preconditions (every SkipFusion has a skip) ---
    if (!skip_buf || !DmaC3D_SKIP) {
        xil_printf("SkipFusion: missing required skip inputs (skip_buf or DmaC3D_SKIP)\r\n");
        return;
    }
    if (!DmaPW || !DmaC3D || !hw_in_buf || !hw_out_buf || !layer_id_pw || !layer_id_c3d) {
        xil_printf("SkipFusion: invalid arguments\r\n");
        return;
    }

    // ping–pong
    int8_t *cur = hw_in_buf;
    int8_t *nxt = hw_out_buf;

    uint32_t c3d_id = (uint8_t)(*layer_id_c3d);
    uint32_t pw_id  = (uint8_t)(*layer_id_pw);

    const uint32_t UPSAMPLE = 0;
    const uint32_t FIRST    = 0;
    const uint32_t SKIP_CON = 1;

    // --- 3x3 SAME + ReLU with external skip: CIN -> CIN ---
    {
        uint32_t LID   = c3d_id;                        // use current id
        hw_layer_c3d(DmaC3D, DmaC3D_SKIP,
                     MAP_SIZE, LID,
                     CIN, /*COUT=*/CIN,
                     UPSAMPLE, FIRST, SKIP_CON,
                     cur, skip_buf, nxt);
        c3d_id++;                                       // bump AFTER call
        swap_i8(&cur, &nxt);
    }

    // --- PW expand: CIN -> 4*COUT ---
    const uint32_t C4 = COUT << 2;
    {
        uint32_t LID   = pw_id;
        hw_layer_pw(DmaPW, MAP_SIZE, LID,
                    /*in*/CIN, /*out*/C4, /*LAST=*/0,
                    cur, nxt);
        pw_id++;                                        // bump AFTER call
        swap_i8(&cur, &nxt);
    }

    // --- PW project: 4*COUT -> COUT ---
    {
        uint32_t LID   = pw_id;
        hw_layer_pw(DmaPW, MAP_SIZE, LID,
                    /*in*/C4, /*out*/COUT, /*LAST=*/0,
                    cur, nxt);
        pw_id++;                                        // bump AFTER call
        swap_i8(&cur, &nxt);
    }

    // Ensure final tensor is in hw_out_buf
    if (cur != hw_out_buf) {
        const size_t bytes = (size_t)MAP_SIZE * MAP_SIZE * COUT;
        memcpy(hw_out_buf, cur, bytes);
    }

    // write back updated counters
    *layer_id_c3d = (int8_t)c3d_id;
    *layer_id_pw  = (int8_t)pw_id;
}


void Stem(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,   // unused when SKIP_CON==0, kept for API symmetry
    int8_t  *hw_in_buf,
    int8_t  *hw_out_buf,
    int8_t  *layer_id_c3d  // IN/OUT: C3D id counter
){
    if (!DmaC3D || !hw_in_buf || !hw_out_buf || !layer_id_c3d) {
        xil_printf("Stem: invalid arguments\r\n");
        return;
    }

    // Use current layer id, then advance after the call
    uint32_t LID   = (uint8_t)(*layer_id_c3d);

    const uint32_t MAP_SIZE = 256;
    const uint32_t UPSAMPLE = 0;   // no upsample
    const uint32_t FIRST    = 1;   // enable special 3->8 + pad path in HW
    const uint32_t SKIP_CON = 0;   // no external skip on the stem

    // hw_layer_c3d(DmaC3D, DmaC3D_SKIP, MAP, SCALE, LID,
    //              CIN, COUT, UPSAMPLE, FIRST, SKIP_CON, in, skip, out)
    (void)DmaC3D_SKIP; // wrapper may ignore when SKIP_CON==0
    hw_layer_c3d(DmaC3D, DmaC3D_SKIP,
                 MAP_SIZE, LID,
                 /*CIN=*/3, /*COUT=*/8,
                 UPSAMPLE, FIRST, SKIP_CON,
                 hw_in_buf, /*skip*/NULL, hw_out_buf);

    *layer_id_c3d = (int8_t)(LID + 1);
}


// Final 1x1 pointwise with LAST=1.
// Runs: in [MAP_SIZE x MAP_SIZE x CIN] -> out [MAP_SIZE x MAP_SIZE x COUT]
void Classifier(
    XAxiDma *DmaPW,
    int8_t  *hw_in_buf,
    int8_t  *hw_out_buf,
    int8_t  *layer_id_pw   // IN/OUT: PW id counter
){
    if (!DmaPW || !hw_in_buf || !hw_out_buf || !layer_id_pw) {
        xil_printf("Classifier: invalid arguments\r\n");
        return;
    }

    const uint32_t MAP_SIZE = 256;
    const uint32_t CIN  = 8;
    const uint32_t COUT = 1;   

    // Use current PW layer id, then advance after the call
    uint32_t LID   = (uint8_t)(*layer_id_pw);

    // hw_layer_pw(DMA, MAP, LID, SCALE, CIN, COUT, LAST, in, out)
    hw_layer_pw(DmaPW, MAP_SIZE, LID, CIN, COUT, /*LAST=*/1, hw_in_buf, hw_out_buf);

    *layer_id_pw = (int8_t)(LID + 1);
}


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
) {
    const int8_t DEPTHS[] = {3, 1, 1, 1, 2};
    const int8_t DIMS[] = {8, 8, 16, 16, 24};
    const int8_t KERNELS[] = {3, 3, 7, 7, 9};

    int8_t dw_id = 0, pw_id = 0, c3d_id = 0;
    int8_t *in  = bufA;
    int8_t *out = bufB;

    Stem(DmaC3D, DmaC3D_SKIP, in, out, &c3d_id);
    swap_i8(&in, &out);

    MobileCMUNeXtBlock(
        DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP, 
        in, out, skip_buf1,  &dw_id, &pw_id, &c3d_id,
        DEPTHS[0], 256, KERNELS[0], DIMS[0], DIMS[0], /*MAXPOOL*/ 0
    );
    swap_i8(&in, &out);
    
    MobileCMUNeXtBlock(
        DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP, 
        in, out, skip_buf2, &dw_id, &pw_id, &c3d_id, 
        DEPTHS[1], 256, KERNELS[1], DIMS[0], DIMS[1], /*MAXPOOL*/ 1
    );
    swap_i8(&in, &out);

    
    MobileCMUNeXtBlock(
        DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP, 
        in, out, skip_buf3, &dw_id, &pw_id, &c3d_id,  
        DEPTHS[2], 128, KERNELS[2], DIMS[1], DIMS[2], /*MAXPOOL*/ 1
    );
    swap_i8(&in, &out);
    
    MobileCMUNeXtBlock(
        DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP, 
        in, out, skip_buf4, &dw_id, &pw_id, &c3d_id,  
        DEPTHS[3], 64, KERNELS[3], DIMS[2], DIMS[3], /*MAXPOOL*/ 1
    );
    swap_i8(&in, &out);

    MobileCMUNeXtBlock(
        DmaPW, DmaDW, DmaC3D, DmaC3D_SKIP, 
        in, out, NULL, &dw_id, &pw_id, &c3d_id, 
        DEPTHS[4], 32, KERNELS[4], DIMS[3], DIMS[4], /*MAXPOOL*/ 1
    );
    swap_i8(&in, &out);
    
    UpConv(DmaC3D, DmaC3D_SKIP, in, out, &c3d_id, 16, DIMS[4], DIMS[3]);
    swap_i8(&in, &out);
    SkipFusion(DmaPW, DmaC3D, DmaC3D_SKIP, in, out, skip_buf4, &pw_id, &c3d_id, 32, DIMS[3], DIMS[3]);
    swap_i8(&in, &out);

    UpConv(DmaC3D, DmaC3D_SKIP, in, out, &c3d_id, 32, DIMS[3], DIMS[2]);
    swap_i8(&in, &out);
    SkipFusion(DmaPW, DmaC3D, DmaC3D_SKIP, in, out, skip_buf3, &pw_id, &c3d_id, 64, DIMS[2], DIMS[2]);
    swap_i8(&in, &out);

    UpConv(DmaC3D, DmaC3D_SKIP, in, out, &c3d_id, 64, DIMS[2], DIMS[1]);
    swap_i8(&in, &out);
    SkipFusion(DmaPW, DmaC3D, DmaC3D_SKIP, in, out, skip_buf2, &pw_id, &c3d_id, 128, DIMS[1], DIMS[1]);
    swap_i8(&in, &out);

    UpConv(DmaC3D, DmaC3D_SKIP, in, out, &c3d_id, 128, DIMS[1], DIMS[0]);
    swap_i8(&in, &out);
    SkipFusion(DmaPW, DmaC3D, DmaC3D_SKIP, in, out, skip_buf1, &pw_id, &c3d_id, 256, DIMS[0], DIMS[0]);
    swap_i8(&in, &out);

    Classifier(DmaPW, in, outBuff, &pw_id);
}