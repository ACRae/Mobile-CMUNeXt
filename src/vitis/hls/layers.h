#ifndef LAYERS_H
#define LAYERS_H

#include <hls_stream.h>
#include <ap_axi_sdata.h>

#define MSIZE 4

typedef ap_int<8> data_t;
typedef ap_int<16> mult_t;
typedef ap_int<24> acc_t;
typedef ap_uint<32> config_t;

typedef ap_int<64> ch_data_t;
typedef ap_axis<128,0,0,0> bus_t;
typedef ap_axis<64,0,0,0> bus64_t;

const acc_t APINT8_MAX = 127;
const acc_t APINT8_RELU_MIN = 0;
const acc_t APINT8_MIN = -128;

static inline acc_t CLAMP8_RELU_AP(acc_t x) {
    return (x > APINT8_MAX) ? APINT8_MAX : ((x < APINT8_RELU_MIN) ? APINT8_RELU_MIN : x);
}

template <typename T>
inline T safe_shift_ap(const T &x, int shift) {
    if (shift == 0)
        return x;
    else if (shift > 0)
        return (x >> shift);
    else
        return (x << (-shift));
}


#define image_W 256      // Height
#define image_H 256      // Width
#define MAX_K 9
#define layersDW 16
#define layersPW 25
#define max_kernel_size 9
// #define MAX_CHANNELS 8

#endif