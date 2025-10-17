#include <stdint.h>
#include <string.h>
#include "sw_conv.h"
#include "sw_layerC3D.h"

// ---------- TUNE THESE TO YOUR MAX CASES ----------
#ifndef C3D_MAX_MAP
#define C3D_MAX_MAP   256     // max MAP_SIZE (before upsample)
#endif
#ifndef C3D_MAX_CIN
#define C3D_MAX_CIN   64      // max input channels
#endif
#ifndef C3D_MAX_COUT
#define C3D_MAX_COUT  64      // max output channels
#endif

// For first layer, HW expands 3->8 channels. Ensure the padded stage supports 8.
#if C3D_MAX_CIN < 8
#define C3D_MAX_CIN_PAD 8
#else
#define C3D_MAX_CIN_PAD C3D_MAX_CIN
#endif

#define KERNEL_SIZE 3
#define PAD         1


// Worst-case scratch sizes (2× upsample, then +2 pad)
static int8_t s_up      [(C3D_MAX_MAP*2) * (C3D_MAX_MAP*2) * C3D_MAX_CIN];           // upsample or copy(+skip)
static int8_t s_pad     [(C3D_MAX_MAP*2+2) * (C3D_MAX_MAP*2+2) * C3D_MAX_CIN_PAD];   // +1/side pad and optional 3->8 expand

// ---------- helpers ----------
static inline int8_t clamp8_relu_i32(int32_t x) {
    if (x <= 0)  return 0;
    if (x > 127) return 127;
    return (int8_t)x;
}
static inline int8_t sat_add_i8(int8_t a, int8_t b) {
    int16_t s = (int16_t)a + (int16_t)b;
    if (s > 127)  return 127;
    if (s < -128) return -128;
    return (int8_t)s;
}
// HLS-safe shift on int8: arithmetic >> for s>=0; saturating << for s<0.
static inline int8_t safe_shift_i8(int8_t v, int s) {
    if (s >= 0) {
        return (int8_t)((int16_t)v >> s);
    } else {
        int16_t w = (int16_t)v << (-s);
        if (w > 127)  w = 127;
        if (w < -128) w = -128;
        return (int8_t)w;
    }
}

/* -------- Bilinear 2× upsample (Q4 weights: 3/1), HWC --------
   in : [H][W][C]
   out: [2H][2W][C]
*/
static void upsample2x_bilinear_hwc(
    const int8_t *in, int8_t *out,
    int H, int W, int C
){
    const int Ho = H << 1;
    const int Wo = W << 1;

    for (int oh = 0; oh < Ho; ++oh) {
        int iy0, iy1;
        if (oh == 0) { iy0 = iy1 = 0; } else { iy0 = (oh - 1) >> 1; iy1 = iy0 + 1; if (iy1 >= H) iy1 = H - 1; }
        const int y_lerp = ((oh & 1) == 0) ? 3 : 1;

        for (int ow = 0; ow < Wo; ++ow) {
            int ix0, ix1;
            if (ow == 0) { ix0 = ix1 = 0; } else { ix0 = (ow - 1) >> 1; ix1 = ix0 + 1; if (ix1 >= W) ix1 = W - 1; }
            const int x_lerp = ((ow & 1) == 0) ? 3 : 1;

            const int b00 = (iy0 * W + ix0) * C;
            const int b01 = (iy0 * W + ix1) * C;
            const int b10 = (iy1 * W + ix0) * C;
            const int b11 = (iy1 * W + ix1) * C;
            const int bo  = (oh  * Wo + ow ) * C;

            for (int c = 0; c < C; ++c) {
                int16_t a = in[b00 + c];
                int16_t b = in[b01 + c];
                int16_t d = in[b10 + c];
                int16_t e = in[b11 + c];

                int16_t top    = a * (4 - x_lerp) + b * x_lerp;
                int16_t bottom = d * (4 - x_lerp) + e * x_lerp;
                int16_t val    = top * (4 - y_lerp) + bottom * y_lerp;

                int16_t q = val >> 4;                    // no rounding (matches HLS)
                if (q > 127)  q = 127;
                if (q < -128) q = -128;
                out[bo + c] = (int8_t)q;
            }
        }
    }
}

/* -------- Optional skip add with per-branch shifts (HLS-matched), HWC -------- */
static void add_skip_inplace_hwc_scaled(
    int8_t *dst, const int8_t *skip,
    int H, int W, int C,
    uint32_t LAYER_ID
){
    const int N = H * W * C;
    const int s_in   = (int)scaleSKIP[LAYER_ID][1];
    const int s_skip = (int)scaleSKIP[LAYER_ID][0];

    for (int i = 0; i < N; ++i) {
        int8_t a = safe_shift_i8(dst[i],  s_in);
        int8_t b = safe_shift_i8(skip[i], s_skip);
        dst[i] = sat_add_i8(a, b);
    }
}

/* -------- Pad HWC tensor by PAD on H/W; optional 3->8 expansion for first layer --------
   in : [H][W][Cin_eff]  (Cin_eff = 3 if FIRST_LAYER==1 else CIN)
   out: [H+2][W+2][Cpad] (Cpad = 8 if FIRST_LAYER==1 else CIN)
*/
static void pad_hwc_with_optional_expand8(
    int8_t *out, const int8_t *in,
    int H, int W, int CIN_eff,
    int PAD_hw, int FIRST_LAYER
){
    const int Cpad = FIRST_LAYER ? 8 : CIN_eff;
    const int Ho = H + (PAD_hw << 1);
    const int Wo = W + (PAD_hw << 1);

    memset(out, 0, (size_t)Ho * Wo * Cpad);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int in_base  = (y * W + x) * CIN_eff;
            const int out_base = ((y + PAD_hw) * Wo + (x + PAD_hw)) * Cpad;

            if (FIRST_LAYER) {
                // Copy RGB into first 3 channels; others stay zero (matches HLS stem path)
                out[out_base + 0] = in[in_base + 0];
                out[out_base + 1] = in[in_base + 1];
                out[out_base + 2] = in[in_base + 2];
            } else {
                memcpy(&out[out_base], &in[in_base], (size_t)CIN_eff);
            }
        }
    }
}

/* -------- 3×3 conv, stride=1, NO implicit padding, HWC --------
   in_padded : [Hpad][Wpad][Cin_conv] (already padded by +1)
   out       : [H][W][Cout] where H = Hpad - 2, W = Wpad - 2
   Shift is arithmetic >> SCALE **without rounding** (matches HLS).
*/
static void sw_conv2D_relu_nopad_C3D(
    const int8_t *in_padded,
    int8_t *out,
    uint32_t Hpad, uint32_t Wpad,
    uint32_t CIN_conv, uint32_t COUT,
    uint32_t LAYER_ID, uint32_t SCALE
){
    const int Hout = (int)Hpad - KERNEL_SIZE + 1;
    const int Wout = (int)Wpad - KERNEL_SIZE + 1;
    const int Cin  = (int)CIN_conv;
    const int Cout = (int)COUT;
    const int Wp   = (int)Wpad;

    for (int y = 0; y < Hout; ++y) {
        for (int x = 0; x < Wout; ++x) {
            const int in_row_base = y * Wp + x; // top-left of 3x3 window (in pixels)
            const int out_base    = (y * Wout + x) * Cout;

            for (int oc = 0; oc < Cout; ++oc) {
                int32_t acc = (int32_t)bias3D[LAYER_ID][oc];

                for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                    const int row_off = (in_row_base + ky * Wp) * Cin;
                    for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                        const int pix_base = row_off + kx * Cin;

                        // Accumulate over input channels
                        for (int ic = 0; ic < Cin; ++ic) {
                            int8_t vin = in_padded[pix_base + ic];
                            int8_t w   = weights3D[LAYER_ID][ky][kx][oc][ic];
                            acc += (int32_t)vin * (int32_t)w;
                        }
                    }
                }

                // Arithmetic >> SCALE (no rounding) + ReLU + clamp to int8
                const int32_t shifted = SCALE ? (acc >> (int)SCALE) : acc;
                out[out_base + oc] = clamp8_relu_i32(shifted);
            }
        }
    }
}

/* -------- Orchestrator: upsample? -> optional scaled skip add (not on first layer) -> pad -> conv --------
   Input : in_buf  [MAP_SIZE][MAP_SIZE][CIN]
   Skip  : skip_buf same H/W/CIN as input (only used if UPSAMPLE==0 && SKIP_CON==1 && FIRST_LAYER==0)
   Output: out_buf [H’][W’][COUT] with H’ = W’ = (UPSAMPLE? 2*MAP_SIZE : MAP_SIZE)
*/
void sw_layerC3D(
    uint32_t MAP_SIZE,
    uint32_t LAYER_ID,
    uint32_t CIN,
    uint32_t COUT,
    uint32_t UPSAMPLE,     // 0/1
    uint32_t FIRST_LAYER,  // 0/1
    uint32_t SKIP_CON,     // 0/1
    const int8_t *restrict in_buf,   // [H][W][CIN]
    const int8_t *restrict skip_buf, // same shape as in_buf
    int8_t *restrict out_buf         // [Hout][Wout][COUT]
){
    // ---- derive sizes (mirror HLS) ----
    const int H0  = (int)MAP_SIZE;
    const int W0  = (int)MAP_SIZE;
    const int Cin = (int)CIN;
    const int H1  = UPSAMPLE ? (H0 << 1) : H0;  // after upsample / copy
    const int W1  = UPSAMPLE ? (W0 << 1) : W0;
    const int Cin_eff  = FIRST_LAYER ? 3 : Cin;   // how many channels we read
    const int Cin_conv = FIRST_LAYER ? 8 : Cin_eff;
    const int SCALE = scale3D[LAYER_ID];

    // ---- stage 1: upsample or copy ----
    if (UPSAMPLE) {
        // s_up: [H1][W1][Cin_eff]
        upsample2x_bilinear_hwc(in_buf, s_up, H0, W0, Cin_eff);
        // NOTE: skip add is NOT done when upsample==1 (matches HLS)
    } else {
        // direct copy (H0,W0,Cin_eff) -> s_up(H1,W1,Cin_eff)
        memcpy(s_up, in_buf, (size_t)H0 * W0 * Cin_eff);
        if (SKIP_CON && !FIRST_LAYER) {
            add_skip_inplace_hwc_scaled(s_up, skip_buf, H0, W0, Cin_eff, LAYER_ID);
        }
    }

    // ---- stage 2: +1/side pad; expand 3->8 for first layer ----
    pad_hwc_with_optional_expand8(s_pad, s_up, H1, W1, Cin_eff, PAD, FIRST_LAYER);

    // ---- stage 3: 3x3 conv (no implicit pad), ReLU ----
    // out dims = (H1+2 - 3 + 1) x (W1+2 - 3 + 1) = H1 x W1
    sw_conv2D_relu_nopad_C3D(
        s_pad, out_buf,
        (uint32_t)(H1 + 2), (uint32_t)(W1 + 2),
        (uint32_t)Cin_conv, COUT,
        LAYER_ID, SCALE
    );
}
