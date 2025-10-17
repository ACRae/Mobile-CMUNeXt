#include <stdint.h>
#include <string.h>
#include "sw_layerDW.h"

// ---- CONFIGURABLE LIMITS (override with -DDW_MAX_* if needed) ----
#ifndef DW_MAX_MAP_SIZE
#define DW_MAX_MAP_SIZE 256
#endif
#ifndef DW_MAX_CIN
#define DW_MAX_CIN 64
#endif


// Static pool buffer (worst-case when MAP_SIZE=DW_MAX_MAP_SIZE and MAXPOOL=1)
static int8_t s_pool[(DW_MAX_MAP_SIZE/2) * (DW_MAX_MAP_SIZE/2) * DW_MAX_CIN];

// --- Helpers (match HLS semantics) ---
static inline int8_t clamp8_relu_i32(int32_t x) {
    if (x <= 0) return 0;
    if (x > 127) return 127;
    return (int8_t)x;
}
static inline int8_t sat_add_i8(int8_t a, int8_t b) {
    int16_t s = (int16_t)a + (int16_t)b;
    if (s > 127) return 127;
    if (s < -128) return -128;
    return (int8_t)s;
}
// Shift like HLS safe_shift_ap on 8-bit values: arithmetic >>, saturating <<.
static inline int8_t safe_shift_i8(int8_t v, int s) {
    if (s >= 0) {
        // arithmetic right shift on signed 8b
        return (int8_t)((int16_t)v >> s);
    } else {
        int sh = -s;
        int16_t w = (int16_t)v << sh;
        if (w > 127) w = 127;
        if (w < -128) w = -128;
        return (int8_t)w;
    }
}

/* 2x2 stride-2 maxpool, HWC: in[H][W][C] -> out[H/2][W/2][C] */
static void maxpool_2x2_stride2_hwc(
    int8_t *out, const int8_t *in,
    int H, int W, int C
){
    const int outH = H >> 1;
    const int outW = W >> 1;
    for (int oy = 0; oy < outH; ++oy) {
        const int iy = oy << 1;
        for (int ox = 0; ox < outW; ++ox) {
            const int ix = ox << 1;
            const int base00 = (iy * W + ix) * C;
            const int base01 = base00 + C;                 // (iy, ix+1)
            const int base10 = ((iy + 1) * W + ix) * C;    // (iy+1, ix)
            const int base11 = base10 + C;                 // (iy+1, ix+1)
            int8_t *dst = &out[(oy * outW + ox) * C];
            for (int c = 0; c < C; ++c) {
                int8_t m = in[base00 + c];
                int8_t v;
                v = in[base01 + c]; if (v > m) m = v;
                v = in[base10 + c]; if (v > m) m = v;
                v = in[base11 + c]; if (v > m) m = v;
                dst[c] = m;
            }
        }
    }
}

/*
 * HLS-compatible depthwise layer (HWC).
 * - Optional 2x2 s2 maxpool (pre-conv).
 * - SAME-sized depthwise conv with zero-padding (no rounding on >> SCALE).
 * - ReLU on DW output.
 * - Residual add with per-branch scales: scaleRES[layer][1] for DW path,
 *   scaleRES[layer][0] for residual path.
 * Output dims always equal src dims after (optional) maxpool.
 */
int sw_layer_dw(
    uint32_t MAP_SIZE,
    uint32_t CIN,
    uint32_t LAYER_ID,
    uint32_t KERNEL_SIZE,  // e.g., 3/7/9
    uint32_t PAD,          // typically (K-1)/2 for SAME
    uint32_t MAXPOOL,      // 0/1
    int8_t *in_buf,        // [MAP_SIZE][MAP_SIZE][C]
    int8_t *out_buf        // [Hsrc][Wsrc][C]  (SAME as src after optional pool)
){
    // Minimal sanity (avoid UB); match HLS leniency otherwise.
    if (!MAP_SIZE || !CIN || !KERNEL_SIZE) return -1;
    if (CIN > DW_MAX_CIN || MAP_SIZE > DW_MAX_MAP_SIZE) return -2;
    if ((MAXPOOL & 1u) && (MAP_SIZE & 1u)) return -3; // need even for 2x2 s2

    // Source dims after optional maxpool (HLS: map_size >> maxpool)
    const int C    = (int)CIN;
    const int Hsrc = (int)MAP_SIZE >> (MAXPOOL & 1u);
    const int Wsrc = Hsrc;
    const int K    = (int)KERNEL_SIZE;
    const int pad  = (int)PAD;

    const int SCALE = scaleDW[LAYER_ID];

    // Prepare conv source: either pooled(input) or input
    const int8_t *src = in_buf;
    if (MAXPOOL & 1u) {
        maxpool_2x2_stride2_hwc(s_pool, in_buf, (int)MAP_SIZE, (int)MAP_SIZE, C);
        src = s_pool;
    }

    // SAME-sized DW conv over zero-padded src; then ReLU; then scaled residual add.
    // Output tensor shape: [Hsrc][Wsrc][C]
    for (int y = 0; y < Hsrc; ++y) {
        for (int x = 0; x < Wsrc; ++x) {
            const int out_base = (y * Wsrc + x) * C;

            for (int c = 0; c < C; ++c) {
                int32_t acc = (int32_t)biasDW[LAYER_ID][c];

                // Convolution with explicit zero-padding
                for (int ky = 0; ky < K; ++ky) {
                    const int iy = y + ky - pad;
                    if ((unsigned)iy >= (unsigned)Hsrc) continue;
                    const int row_base = iy * Wsrc;

                    for (int kx = 0; kx < K; ++kx) {
                        const int ix = x + kx - pad;
                        if ((unsigned)ix >= (unsigned)Wsrc) continue;

                        const int in_idx = (row_base + ix) * C + c; // HWC
                        const int8_t vin = src[in_idx];
                        const int8_t w   = weightsDW[LAYER_ID][ky][kx][c];
                        acc += (int32_t)vin * (int32_t)w;
                    }
                }

                // Match HLS: arithmetic >> SCALE (no rounding) then ReLU (0..127)
                const int32_t shifted = (SCALE ? (acc >> (int)SCALE) : acc);
                const int8_t  dw_val  = clamp8_relu_i32(shifted);

                // Per-branch extra scaling (HLS HW_writeData)
                const int8_t dw_scaled  = safe_shift_i8(dw_val,              (int)scaleRES[LAYER_ID][1]);
                const int8_t res_scaled = safe_shift_i8(src[(y*Wsrc + x)*C + c], (int)scaleRES[LAYER_ID][0]);

                out_buf[out_base + c] = sat_add_i8(dw_scaled, res_scaled);
            }
        }
    }

    return 0;
}
