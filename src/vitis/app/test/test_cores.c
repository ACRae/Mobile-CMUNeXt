#include "test_cores.h"
#include <stdint.h>
#include <stdio.h>
#include <xstatus.h>
#include "xil_printf.h"
#include "../utils/hls_utils.h"
#include "../utils/time_utils.h"
#include "../utils/debug.h"
#include "../utils/sw_utils.h"
#include "../sw/sw_layerPW.h"
#include "../sw/sw_layerDW.h"
#include "../hw/layerPW.h"
#include "../hw/layerDW.h"
#include "../sw/sw_layerC3D.h"
#include "../hw/layerC3D.h"


// returns 1 if equal, 0 if any mismatch
int buffers_equal_i8(const int8_t *a, const int8_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return 0;
    }
    return 1;
}


void test_layerPW(
    XAxiDma *DmaPW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
) {
    xil_printf("Starting PW Test:\r\n");
    TIME_FUNCTION(
        "HW Layer PW - 1",
        hw_layer_pw(
            DmaPW, 
            64,     /* MAP_SIZE */ 
            0,      /* LAYER_ID */
            16,     /* CIN */ 
            64,     /* COUT */
            0, 
            hw_in_buf, 
            hw_out_buf
        )
    );
    sample_buffer(hw_out_buf, 64*64*64);


    TIME_FUNCTION(
        "SW layer PW - 1",
        sw_layer_pw( 
            64,     /* MAP_SIZE */ 
            0,      /* LAYER_ID */
            16,     /* CIN */ 
            64,     /* COUT */
            0,
            hw_in_buf, 
            sw_out_buf
        )
    );
    sample_buffer(sw_out_buf, 64*64*64);

    if(buffers_equal_i8(hw_out_buf, sw_out_buf, 64*64*64)) {
        xil_printf("[SUCESS] HW and SW buffers Match!\r\n");
    }
    else {
        xil_printf("[FAILURE] HW and SW buffers DONT Match!\r\n");
    }
}



// DW test – same style as your PW test
void test_layerDW(
    XAxiDma *DmaDW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
) {
    // Choose params so HW and SW produce same H/W (SAME padding)
    const uint32_t MAP_SIZE    = 64;
    const uint32_t CIN         = 16;
    const uint32_t LAYER_ID    = 0;
    const uint32_t KERNEL_SIZE = 3;
    const uint32_t PAD         = 1;   // SAME for K=3
    const uint32_t MAXPOOL     = 0;   // keep off for 1:1 size

    const size_t   BYTES = (size_t)MAP_SIZE * MAP_SIZE * CIN;

    xil_printf("Starting DW Test:\r\n");

    TIME_FUNCTION(
        "HW Layer DW - 1",
        hw_layer_dw(
            DmaDW,
            MAP_SIZE,
            CIN,
            LAYER_ID,
            KERNEL_SIZE,
            PAD,
            MAXPOOL,
            hw_in_buf,
            hw_out_buf
        )
    );
    sample_buffer(hw_out_buf, BYTES);

    TIME_FUNCTION(
        "SW Layer DW - 1",
        sw_layer_dw(
            MAP_SIZE,
            CIN,
            LAYER_ID,
            KERNEL_SIZE,
            PAD,
            MAXPOOL,
            hw_in_buf,
            sw_out_buf
        )
    );
    sample_buffer(sw_out_buf, BYTES);

    if (buffers_equal_i8(hw_out_buf, sw_out_buf, BYTES)) {
        xil_printf("[SUCCESS] HW and SW buffers Match!\r\n");
    } else {
        xil_printf("[FAILURE] HW and SW buffers DON'T Match!\r\n");
    }
}



void test_layerC3D(
    XAxiDma *DmaC3D,
    XAxiDma *DmaC3D_SKIP,
    int8_t *hw_in_buf,
    int8_t *hw_skip_buf,   // pass NULL if no skip
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
) {
    /* Choose params that match your HLS:
       - CIN & COUT must be multiples of 8.
       - in HLS cfg, fields are groups-of-8 (your hw wrapper should divide by 8).
       - With K=3 and PAD=1 inside the block, spatial size is preserved after the pad+conv.
    */
    const uint32_t MAP_SIZE    = 64;
    const uint32_t LAYER_ID    = 0;
    const uint32_t CIN         = 16;   // multiple of 8
    const uint32_t COUT        = 16;   // multiple of 8
    const uint32_t UPSAMPLE    = 0;    // set 1 to test the upsample path
    const uint32_t FIRST_LAYER = 0;    // set 1 only if your SW/HW expand 3->8 inside the block
    const uint32_t SKIP_CON    = (hw_skip_buf != NULL) ? 1u : 0u;

    const uint32_t H = MAP_SIZE * (1u + UPSAMPLE);
    const uint32_t W = MAP_SIZE * (1u + UPSAMPLE);
    const size_t OUT_ELEMS = (size_t)H * W * COUT;

    xil_printf("Starting C3D Test:\r\n");

    TIME_FUNCTION(
        "HW Layer C3D - 1",
        hw_layer_c3d(
            DmaC3D,
            DmaC3D_SKIP,
            MAP_SIZE,
            LAYER_ID,
            CIN,
            COUT,
            UPSAMPLE,
            FIRST_LAYER,
            SKIP_CON,
            hw_in_buf,
            hw_skip_buf,    // only used if SKIP_CON==1 and UPSAMPLE==0 and FIRST_LAYER==0 (per your HLS)
            hw_out_buf
        )
    );
    sample_buffer(hw_out_buf, OUT_ELEMS);

    TIME_FUNCTION(
        "SW Layer C3D - 1",
        sw_layerC3D(
            MAP_SIZE,
            LAYER_ID,
            CIN,
            COUT,
            UPSAMPLE,
            FIRST_LAYER,
            SKIP_CON,
            hw_in_buf,
            hw_skip_buf,
            sw_out_buf
        )
    );
    sample_buffer(sw_out_buf, OUT_ELEMS);

    if (buffers_equal_i8(hw_out_buf, sw_out_buf, OUT_ELEMS)) {
        xil_printf("[SUCCESS] HW and SW buffers Match!\r\n");
    } else {
        xil_printf("[FAILURE] HW and SW buffers DON'T Match!\r\n");
    }
}



typedef struct {
    uint32_t map;
    uint32_t layer;
    uint32_t cin;
    uint32_t cout;
    uint32_t lastLayer;
} PwCfg;

static void run_pw_case(
    int case_id,
    XAxiDma *DmaPW,
    const PwCfg *cfg,
    int8_t *in,
    int8_t *hw_out,
    int8_t *sw_out
){
    char lbl_hw[48], lbl_sw[48];
    snprintf(lbl_hw, sizeof(lbl_hw), "[HW] Layer PW - %d", case_id);
    snprintf(lbl_sw, sizeof(lbl_sw), "[SW] Layer PW - %d", case_id);

    // Run HW
    TIME_FUNCTION(
        lbl_hw,
        hw_layer_pw(
            DmaPW,
            cfg->map, cfg->layer, cfg->cin, cfg->cout, cfg->lastLayer,
            in, hw_out
        )
    );

    // Run SW (write into sw_out, not hw_out)
    TIME_FUNCTION(
        lbl_sw,
        sw_layer_pw(
            cfg->map, cfg->layer, cfg->cin, cfg->cout, cfg->lastLayer,
            in, sw_out
        )
    );

    // Compare
    const size_t elems = (size_t)cfg->map * cfg->map * cfg->cout;
    if (buffers_equal_i8(hw_out, sw_out, elems)) {
        xil_printf("  Case %d: [OK] %lux%lu CIN=%lu -> COUT=%lu\r\n",
                   case_id, (unsigned long)cfg->map, (unsigned long)cfg->map,
                   (unsigned long)cfg->cin, (unsigned long)cfg->cout);
    } else {
        xil_printf("  Case %d: [MISMATCH] %lux%lu CIN=%lu -> COUT=%lu\r\n",
                   case_id, (unsigned long)cfg->map, (unsigned long)cfg->map,
                   (unsigned long)cfg->cin, (unsigned long)cfg->cout);
    }
}

void test_all_configs_layerPW(
    XAxiDma *DmaPW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
){
    static const PwCfg cases[] = {
        // map, layer, scale, cin, cout, lastlayer
        {256, 0,  8, 32, 0},
        {256, 0, 32,  8, 0},
        {128, 0,  8, 32, 0},
        {128, 0, 32,  8, 0},
        { 64, 0,  8, 32, 0},
        { 64, 0, 32,  8, 0},
        { 32, 0, 16, 64, 0},
        { 32, 0, 64, 16, 0},
        { 16, 0, 16, 64, 0},
        { 16, 0, 64, 16, 0},
        {256, 24, 8, 1,  1}, // last layer
    };

    xil_printf("Starting PW sweep (%d cases):\r\n", (int)(sizeof(cases)/sizeof(cases[0])));

    for (int i = 0; i < (int)(sizeof(cases)/sizeof(cases[0])); ++i) {
        run_pw_case(i+1, DmaPW, &cases[i], hw_in_buf, hw_out_buf, sw_out_buf);
    }

    xil_printf("------------------ PW Test DONE ------------------\r\n");
}






// ===== DW sweep =====
typedef struct {
    uint32_t map;        // input H=W before pool
    uint32_t cin;        // channels in = channels out
    uint32_t layer;
    uint32_t k;          // kernel size
    uint32_t pad;        // padding on each side
    uint32_t maxpool;    // 0/1 (2x2 s2)
    const char *name;    // label
} DwCfg;

static void run_dw_case(
    int idx, XAxiDma *DmaDW, const DwCfg *cfg,
    int8_t *in, int8_t *hw_out, int8_t *sw_out
) {
    // Output spatial size equals (map / (maxpool?2:1)) when pad==(k-1)/2 and stride=1
    const uint32_t H0 = cfg->map / (cfg->maxpool ? 2u : 1u);
    const uint32_t W0 = H0;
    const size_t ELEMS = (size_t)H0 * W0 * cfg->cin;

    char lbl_hw[64], lbl_sw[64];
    snprintf(lbl_hw, sizeof(lbl_hw), "[HW] DW - %d %s", idx, cfg->name);
    snprintf(lbl_sw, sizeof(lbl_sw), "[SW] DW - %d %s", idx, cfg->name);

    TIME_FUNCTION(
        lbl_hw,
        hw_layer_dw(
            DmaDW,
            /* MAP_SIZE   */ cfg->map,
            /* CIN        */ cfg->cin,
            /* LAYER_ID   */ cfg->layer,
            /* KERNEL     */ cfg->k,
            /* PAD        */ cfg->pad,
            /* MAXPOOL    */ cfg->maxpool,
            in, hw_out
        )
    );


    TIME_FUNCTION(
        lbl_sw,
        sw_layer_dw(
            cfg->map, cfg->cin, cfg->layer,
            cfg->k, cfg->pad, cfg->maxpool,
            in, sw_out
        )
    );

    if (buffers_equal_i8(hw_out, sw_out, ELEMS)) {
        xil_printf("  DW Case %d [%s]: OK (%lux%lu x C=%lu)\r\n",
            idx, cfg->name, (unsigned long)H0, (unsigned long)W0, (unsigned long)cfg->cin);
    } else {
        xil_printf("  DW Case %d [%s]: MISMATCH\r\n", idx, cfg->name);
    }
}

void test_all_configs_layerDW(
    XAxiDma *DmaDW,
    int8_t *hw_in_buf,
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
){
    static const DwCfg cases[] = {
        // Encoder 1
        {256,  8, 2, 3, 1, 0, "Enc1 256->256 K3 P1 pool0"},
        // Encoder 2
        {256,  8, 2, 3, 1, 1, "Enc2 256->128 K3 P1 pool1"},
        {128,  8, 2, 3, 1, 0, "Enc2 128->128 K3 P1 pool0"},
        // Encoder 3
        {128,  8, 2, 7, 3, 1, "Enc3 128->64  K7 P3 pool1"},
        { 64,  8, 2, 7, 3, 0, "Enc3 64->64   K7 P3 pool0"},
        // Encoder 4
        { 64, 16, 2, 7, 3, 1, "Enc4 64->32   K7 P3 pool1"},
        { 32, 16, 2, 7, 3, 0, "Enc4 32->32   K7 P3 pool0"},
        // Encoder 5
        { 32, 16, 2, 9, 4, 1, "Enc5 32->16   K9 P4 pool1"},
        { 16, 16, 2, 9, 4, 0, "Enc5 16->16   K9 P4 pool0"},
    };

    const int N = (int)(sizeof(cases)/sizeof(cases[0]));
    xil_printf("Starting DW sweep (%d cases):\r\n", N);
    for (int i = 0; i < N; ++i)
        run_dw_case(i+1, DmaDW, &cases[i], hw_in_buf, hw_out_buf, sw_out_buf);
    xil_printf("------------------ DW Test DONE ------------------\r\n");
}


// ===== C3D sweep =====
typedef struct {
    uint32_t map;        // input H=W BEFORE upsample
    uint32_t cin;
    uint32_t cout;
    uint32_t layer;
    uint32_t upsample;     // 0/1
    uint32_t first_layer;  // 0/1
    uint32_t skip_con;     // 0/1
    const char *name;
} C3dCfg;

static void run_c3d_case(
    int idx,
    XAxiDma *DmaC3D, XAxiDma *DmaC3D_SKIP,
    const C3dCfg *cfg,
    int8_t *in, int8_t *skip, int8_t *hw_out, int8_t *sw_out
){
    const uint32_t H = cfg->map * (1u + cfg->upsample);
    const uint32_t W = H;
    const size_t OUT_ELEMS = (size_t)H * W * cfg->cout;

    char lbl_hw[64], lbl_sw[64];
    snprintf(lbl_hw, sizeof(lbl_hw), "[HW] C3D - %d %s", idx, cfg->name);
    snprintf(lbl_sw, sizeof(lbl_sw), "[SW] C3D - %d %s", idx, cfg->name);

    TIME_FUNCTION(
        lbl_hw,
        hw_layer_c3d(
            DmaC3D, DmaC3D_SKIP,
            cfg->map, cfg->layer,
            cfg->cin, cfg->cout,
            cfg->upsample, cfg->first_layer, cfg->skip_con,
            in,
            (cfg->skip_con ? skip : NULL),
            hw_out
        )
    );

    TIME_FUNCTION(
        lbl_sw,
        sw_layerC3D(
            cfg->map, cfg->layer,
            cfg->cin, cfg->cout,
            cfg->upsample, cfg->first_layer, cfg->skip_con,
            in,
            (cfg->skip_con ? skip : NULL),
            sw_out
        )
    );

    if (buffers_equal_i8(hw_out, sw_out, OUT_ELEMS)) {
        xil_printf("  C3D Case %d [%s]: OK (%lux%lu x C=%lu)\r\n",
            idx, cfg->name, (unsigned long)H, (unsigned long)W, (unsigned long)cfg->cout);
    } else {
        xil_printf("  C3D Case %d [%s]: MISMATCH\r\n", idx, cfg->name);
    }
}

void test_all_configs_layerC3D(
    XAxiDma *DmaC3D, 
    XAxiDma *DmaC3D_SKIP,
    int8_t *hw_in_buf,
    int8_t *hw_skip_buf,   // provide valid data for skip cases
    int8_t *hw_out_buf,
    int8_t *sw_out_buf
){
    // SCALE set to 0 for all; adjust if you want fixed shifts.
    static const C3dCfg cases[] = {
        // First Layer
        {256,  3,  8, 0, 0, 1, 0, "First 256x256x3 -> 256x256x8, up0, skip0"},
        // Encoders
        {256,  8,  8, 0, 0, 0, 0, "Enc 256->256 C8->8 up0 skip0"},
        {128,  8,  8, 0, 0, 0, 0, "Enc 128->128 C8->8 up0 skip0"},
        { 64,  8, 16, 0, 0, 0, 0, "Enc 64->64  C8->16 up0 skip0"},
        { 32, 16, 16, 0, 0, 0, 0, "Enc 32->32 C16->16 up0 skip0"},
        { 16, 16, 24, 0, 0, 0, 0, "Enc 16->16 C16->24 up0 skip0"},
        // Decoders (upsample and/or skip as specified)
        { 16, 24, 16, 0, 1, 0, 0, "Dec 16->32  up1 skip0 C24->16"},
        { 32, 16, 16, 0, 0, 0, 1, "Dec 32->32  up0 skip1 C16->16"},
        { 32, 16, 16, 0, 1, 0, 0, "Dec 32->64  up1 skip0 C16->16"},
        { 64, 16, 16, 0, 0, 0, 1, "Dec 64->64  up0 skip1 C16->16"},
        { 64, 16,  8, 0, 1, 0, 0, "Dec 64->128 up1 skip0 C16->8"},
        {128,  8,  8, 0, 0, 0, 1, "Dec 128->128 up0 skip1 C8->8"},
        {128,  8,  8, 0, 1, 0, 0, "Dec 128->256 up1 skip0 C8->8"},
        {256,  8,  8, 0, 0, 0, 1, "Dec 256->256 up0 skip1 C8->8"},
    };

    const int N = (int)(sizeof(cases)/sizeof(cases[0]));
    xil_printf("Starting C3D sweep (%d cases):\r\n", N);
    for (int i = 0; i < N; ++i)
        run_c3d_case(i+1, DmaC3D, DmaC3D_SKIP, &cases[i],
                     hw_in_buf, hw_skip_buf, hw_out_buf, sw_out_buf);
    xil_printf("------------------ C3D Test DONE -----------------\r\n");
}
