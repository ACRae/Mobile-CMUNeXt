#include "layerC3D.h"

void HW_upsample(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &skip_con_in,
    hls::stream<ap_uint<64>> &fifo_out,
    config_t config
) {
#pragma HLS interface axis port=strm_in
#pragma HLS interface axis port=skip_con_in
#pragma HLS INLINE off

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    unsigned int layer_ID = config.range(13,9).to_uint();       // layer identifier
    unsigned int channel = config.range(15,14).to_uint();       // channel
    ap_uint<1> upsample = config.range(18,18);
    ap_uint<1> firstLayer = config.range(19,19);
    ap_uint<1> skipCon = config.range(20,20);



    // 4-line rolling buffer; second dim sized to your worst-case (map_size*channel <= 128)
    ap_uint<64> mapa_inC2D[4][128];
#pragma HLS ARRAY_PARTITION variable=mapa_inC2D complete dim=1
#pragma HLS bind_storage  variable=mapa_inC2D type=RAM_2P impl=bram

    int iy0, ix0;
    int iy1, ix1;
    ap_int<3>  y_lerp, x_lerp;
    data_t a, b, c, d;

    bus64_t tmp1, tmp2;
    int last_loaded_y = 3; // (4-1)

    if (upsample == 1){
        // Prime the 4 rolling rows
        rd_Lines1a: for (int i = 0; i < 4; i++) {
            rd_Lines1b: for (int j = 0; j < map_size*channel; j++){
#pragma HLS LOOP_TRIPCOUNT max=128
#pragma HLS PIPELINE II=1
                tmp1 = strm_in.read();
                mapa_inC2D[i][j] = tmp1.data;
            }
        }

        // For each output row (2x height)
        for (ap_uint<9> oh = 0; oh < map_size*2; ++oh) {
#pragma HLS LOOP_TRIPCOUNT max=256

            if (oh == 0) { iy0 = 0; iy1 = 0; }
            else { iy0 = (oh - 1) >> 1; iy1 = iy0 + 1; }
            if (iy1 >= map_size) iy1 = map_size - 1;

            y_lerp = ((oh & 1) == 0) ? 3 : 1; // 4x fixed-point weights: 3/1

            // Load the next needed input line into the rolling buffer (before we pipeline the inner loop)
            if (iy1 > last_loaded_y) {
                last_loaded_y++;
                for (int j = 0; j < map_size * channel; j++) {
#pragma HLS PIPELINE II=1
                    tmp1 = strm_in.read();
                    mapa_inC2D[last_loaded_y & 3][j] = tmp1.data;
                }
            }

            // For each output col (2x width)
            for (ap_uint<9> ow = 0; ow < map_size*2; ++ow) {
#pragma HLS LOOP_TRIPCOUNT max=256

                if (ow == 0) { ix0 = 0; ix1 = 0; }
                else { ix0 = (ow - 1) >> 1; ix1 = ix0 + 1; }
                if (ix1 >= map_size) ix1 = map_size - 1;

                x_lerp = ((ow & 1) == 0) ? 3 : 1;

                // Pipeline the writer loop: 1 output word per cycle
                upsample_ch: for (int ch = 0; ch < 3; ++ch) {
#pragma HLS LOOP_TRIPCOUNT max=3
#pragma HLS PIPELINE II=1
                    if (ch >= (int)channel) continue;

                    // Cache the four 64-bit neighbors ONCE (avoid multi-reads per lane)
                    ap_uint<64> wa0 = mapa_inC2D[iy0 & 3][ix0 * channel + ch];
                    ap_uint<64> wb0 = mapa_inC2D[iy0 & 3][ix1 * channel + ch];
                    ap_uint<64> wa1 = mapa_inC2D[iy1 & 3][ix0 * channel + ch];
                    ap_uint<64> wb1 = mapa_inC2D[iy1 & 3][ix1 * channel + ch];

                    // Compute 8 lanes fully in parallel and pack to 64b
                    ap_uint<64> out_buf = 0;

                    interp_lanes: for (ap_uint<4> lane = 0; lane < 8; lane++) {
#pragma HLS UNROLL
                        data_t aa = (data_t)wa0.range(lane*8+7, lane*8);
                        data_t bb = (data_t)wb0.range(lane*8+7, lane*8);
                        data_t cc = (data_t)wa1.range(lane*8+7, lane*8);
                        data_t dd = (data_t)wb1.range(lane*8+7, lane*8);

                        // Bilinear interpolation (Q4 weights: 4-x, x)
                        ap_int<12> top    = aa * (4 - x_lerp) + bb * x_lerp;
                        ap_int<12> bottom = cc * (4 - x_lerp) + dd * x_lerp;
                        ap_int<12> value  = top * (4 - y_lerp) + bottom * y_lerp;
                        value = value >> 4;

                        // Saturate to int8
                        ap_int<9> v9 = value;
                        data_t v8 = (v9 > 127) ? (data_t)127 : (v9 < -128) ? (data_t)-128 : (data_t)v9;

                        out_buf.range(lane*8+7, lane*8) = v8.range(7, 0);
                    }

                    fifo_out.write(out_buf);
                } // ch
            } // ow
        } // oh
    }
    else {
        int line_read = map_size*channel;
        if (firstLayer == 1) line_read = ((256*3)/8);

        rd_all_1: for (ap_int<10> i = 0; i < map_size; i++) {
            #pragma HLS LOOP_TRIPCOUNT max=256
            rd_all_2: for (ap_int<10> j = 0; j < line_read; j++){
                #pragma HLS LOOP_TRIPCOUNT max=256
                #pragma HLS PIPELINE II=1

                tmp1 = strm_in.read();

                // Skip connection is either on for whole frame or off; no per-iter hazard
                const bool perform_skip = (skipCon == 1) && (firstLayer == 0);
                if (perform_skip) {
                    tmp2 = skip_con_in.read();

                skip_add: for (ap_uint<4> c = 0; c < 8; c++) {
                        #pragma HLS UNROLL
                        data_t v1 = (data_t) tmp1.data.range(c*8+7, c*8);
                        data_t v2 = (data_t) tmp2.data.range(c*8+7, c*8);
                    
                        v1 = safe_shift_ap(v1, scaleSKIP[layer_ID][1]); // input
                        v2 = safe_shift_ap(v2, scaleSKIP[layer_ID][0]); // skip

                        ap_int<9> sum = (ap_int<9>)v1 + (ap_int<9>)v2;
                        ap_int<8> sat = (sum > 127) ? (ap_int<8>)127 : (sum < -128) ? (ap_int<8>)-128 : (ap_int<8>)sum;

                        tmp1.data.range(c*8+7, c*8) = sat.range(7, 0);
                    }
                }

                fifo_out.write(tmp1.data);
            }
        }
    }
}





void HW_readData(
    hls::stream<ap_uint<64>> &fifo_in, //[H][W][C]
    hls::stream<ap_uint<64>> &fifo_out,
    ap_uint<32> config
) {

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    ap_uint<2> channel = config.range(15,14);       // channel
    ap_uint<1> upsample = config.range(18,18);
    ap_uint<1> firstLayer = config.range(19,19);


    ap_uint<9> map_size_upsample = map_size + (map_size * upsample);

    ap_uint<64> auxBuf;
    ap_uint<64> zeros = 0;
    ap_uint<64> tmp;

    if (firstLayer == 1){ // For Stem
        pad1: for (ap_uint<9> i = 0; i < 258; i++)  // send zeros for padding
            fifo_out.write(zeros);

        wr_lines: for (ap_uint<9> j = 0; j < 256; j++){ // Y
            fifo_out.write(zeros); // left padding

            // scratch buffer for the 8 output 64b words produced per k
            ap_uint<64> out_words[8];
            //#pragma HLS ARRAY_PARTITION variable=out_words complete dim=1

            for (ap_uint<6> k = 0; k < 32; k++) {
                // ---- Pack phase (no stream writes here) ----
                ap_uint<64> tmp0 = fifo_in.read();
                ap_uint<64> tmp1 = fifo_in.read();
                ap_uint<64> tmp2 = fifo_in.read();

                ap_uint<64> auxBuf = 0;

                // fill out_words[0..7] exactly like your current sequence of writes:
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp0.range(23,0));
                out_words[0] = auxBuf;
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp0.range(47,24));
                out_words[1] = auxBuf;
                auxBuf.range(15,0) = tmp0.range(63,48);
                auxBuf.range(63,16) = ((ap_uint<40>)0, tmp1.range(7,0));
                out_words[2] = auxBuf;
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp1.range(31,8));
                out_words[3] = auxBuf;
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp1.range(55,32));
                out_words[4] = auxBuf;
                auxBuf.range(7,0) = tmp1.range(63,56);
                auxBuf.range(63,8) = ((ap_uint<40>)0, tmp2.range(15,0));
                out_words[5] = auxBuf;
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp2.range(39,16));
                out_words[6] = auxBuf;
                auxBuf.range(63,0) = ((ap_uint<40>)0, tmp2.range(63,40));
                out_words[7] = auxBuf;

                // ---- Emit phase: exactly one write per iteration (II=1 achievable) ----
                emit: for (int t = 0; t < 8; t++) {
                    #pragma HLS PIPELINE II=1
                    fifo_out.write(out_words[t]);
                }
            }

            fifo_out.write(zeros); // right padding
        }

        pad2: for (ap_uint<9> i = 0; i < 258; i++)  // send zeros for padding
            #pragma HLS PIPELINE II=1
            fifo_out.write(zeros);
    }
    else {
        padTop: for (ap_uint<9> i = 0; i < (map_size_upsample+2)*channel; i++)  // send zeros for padding
            #pragma HLS LOOP_TRIPCOUNT max=258
            fifo_out.write(zeros);

        wr_linesb: for (ap_uint<9> j = 0; j < map_size_upsample; j++){
            #pragma HLS LOOP_TRIPCOUNT max=256

            padLeft: for (ap_uint<2> k = 0; k < channel;k++){
                #pragma HLS LOOP_TRIPCOUNT max=3
                fifo_out.write(zeros);
            }

            rd_datab: for (ap_uint<9> k = 0; k < map_size_upsample*channel; k++){
                #pragma HLS LOOP_TRIPCOUNT max=256
                tmp = fifo_in.read();
                fifo_out.write(tmp.range(63,0));
            }

            padRight: for (ap_uint<2> k = 0; k < channel;k++) {
                #pragma HLS LOOP_TRIPCOUNT max=3 // 8*3 = 24
                fifo_out.write(zeros);
            }
        }
        padBottom: for (ap_uint<9> i = 0; i < (map_size_upsample+2)*channel; i++){  // send zeros for padding
            #pragma HLS LOOP_TRIPCOUNT max=258
            fifo_out.write(zeros);
        }
    }
}



#define MAX_LINES   3             // 3 lines are enough for 3x3
#define KERNEL_SIZE 3
#define PAD         1
#define MAX_CH      3             // channel field is 2 bits -> [0..3] => max 3
#define MAX_W_PADDED 258          // worst-case width with +2 pad
#define BUF_COLS    (MAX_W_PADDED * MAX_CH)

void HW_conv3D(
    hls::stream<ap_uint<64>> &fifo_in,
    hls::stream<ap_uint<64>> &fifo_out1,
    hls::stream<ap_uint<64>> &fifo_out2,
    ap_uint<32> config
) {

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    const unsigned int layer_ID = config.range(13,9).to_uint();       // layer identifier
    const unsigned int channel = config.range(15,14).to_uint();       // channel
    ap_uint<2> filters = config.range(17,16);       // channel
    ap_uint<1> upsample = config.range(18,18);

    const int scale = scale3D[layer_ID].to_int();

    ap_uint<9> map_size_upsample = map_size + (map_size * upsample);
    ap_uint<9> map_size_up_padded = (map_size_upsample+2*PAD);

    ap_uint<17> maxReads =  map_size_up_padded*map_size_up_padded*channel;
    ap_uint<17> readCounter = 0;

    ap_uint<64> mapa_inC2D[MAX_LINES][BUF_COLS];
#pragma HLS bind_storage variable=mapa_inC2D type=RAM_2P impl=bram

    // Two-dimensional accumulators: [8 filters][9 kernel positions]
    acc_t acc1[8][KERNEL_SIZE*KERNEL_SIZE];
    acc_t acc2[8][KERNEL_SIZE*KERNEL_SIZE];
    #pragma HLS ARRAY_PARTITION variable=acc1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc2 complete dim=0

    ap_uint<64> tmp, tmp1;

    rd_Lines1a: for (int i = 0; i < MAX_LINES; i++){
        #pragma HLS LOOP_TRIPCOUNT min=3 max=3
        rd_Lines1b: for (int j = 0; j < map_size_up_padded*channel; j++){
            #pragma HLS LOOP_TRIPCOUNT max=258
            #pragma HLS PIPELINE
            tmp = fifo_in.read();
            readCounter++;
            mapa_inC2D[i][j] = tmp.range(63,0);
        }
    }

    map_line: for (ap_uint<9> i = 0; i < map_size_upsample; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=256
        map_col: for (int j = 0; j < map_size_upsample; j+=2) {
        #pragma HLS LOOP_TRIPCOUNT max=128

            ft_opt: for (int fx = 0; fx < filters; fx++) {
                #pragma HLS LOOP_TRIPCOUNT max=3

                // Initialize all accumulator positions to zero
                init_acc: for (int f = 0; f < 8; f++) {
                    #pragma HLS UNROLL
                    for (int k = 0; k < KERNEL_SIZE*KERNEL_SIZE; k++) {
                        #pragma HLS UNROLL
                        acc1[f][k] = 0;
                        acc2[f][k] = 0;
                    }
                }

                ch: for (int ch_idx = 0; ch_idx < channel; ch_idx++){
                #pragma HLS LOOP_TRIPCOUNT max=2

                    ch_kx_ky: for (int idx = 0; idx < KERNEL_SIZE * KERNEL_SIZE; idx++) {
                    #pragma HLS PIPELINE II=1
                        int ki = idx / KERNEL_SIZE;
                        int kj = idx % KERNEL_SIZE;
                        
                        int ii = i + ki;
                        int jj  = j + kj;
                        int jj2 = jj + 1;
                        if (jj2 >= (int)map_size_up_padded) jj2 = (int)map_size_up_padded - 1;

                        int map_idx1 = jj  * channel + ch_idx;
                        int map_idx2 = jj2 * channel + ch_idx;

                        ap_uint<64> v1 = mapa_inC2D[(ii%MAX_LINES)][map_idx1];
                        ap_uint<64> v2 = mapa_inC2D[(ii%MAX_LINES)][map_idx2];

                        data_mac: for (int c = 0; c < 8; c++) {
                            #pragma HLS UNROLL
                            data_t d1 = (data_t)v1.range(c*8+7, c*8);
                            data_t d2 = (data_t)v2.range(c*8+7, c*8);

                            filters_mac: for (int f = 0; f < 8; f++){
                                #pragma HLS UNROLL
                                data_t w1 = weights3D[layer_ID][ki][kj][fx*8+f][ch_idx*8+c];
                                acc_t w1_mul1, w1_mul2;
                                // Bind specific operations to DSP48 with latency control
                                #pragma HLS BIND_OP variable=w1_mul1 op=mul impl=dsp latency=3
                                #pragma HLS BIND_OP variable=w1_mul2 op=mul impl=dsp latency=3

                                w1_mul1 = d1 * w1;
                                w1_mul2 = d2 * w1;

                                // Accumulate to separate position for each kernel element
                                acc1[f][idx] += w1_mul1;
                                acc2[f][idx] += w1_mul2;
                            }
                        }
                    }
                }

                // Reduction tree to sum all kernel positions and add bias once
                acc_t final_acc1[8], final_acc2[8];
                #pragma HLS ARRAY_PARTITION variable=final_acc1 complete
                #pragma HLS ARRAY_PARTITION variable=final_acc2 complete

                reduce: for (int f = 0; f < 8; f++) {
                    #pragma HLS UNROLL
                    // Start with bias
                    final_acc1[f] = bias3D[layer_ID][fx*8+f];
                    final_acc2[f] = bias3D[layer_ID][fx*8+f];
                    
                    // Sum all kernel positions
                    for (int k = 0; k < KERNEL_SIZE*KERNEL_SIZE; k++) {
                        #pragma HLS UNROLL
                        final_acc1[f] += acc1[f][k];
                        final_acc2[f] += acc2[f][k];
                    }
                }

                data_t buf1[8], buf2[8];
                #pragma HLS ARRAY_PARTITION variable=buf1 complete
                #pragma HLS ARRAY_PARTITION variable=buf2 complete

                res: for (ap_uint<4> c = 0; c < 8; c++){
                    #pragma HLS UNROLL
                    data_t v1 = (data_t)CLAMP8_RELU_AP(final_acc1[c] >> scale);
                    data_t v2 = (data_t)CLAMP8_RELU_AP(final_acc2[c] >> scale);
                    buf1[c] = v1;
                    buf2[c] = v2;
                }

                ap_uint<64> bufout1, bufout2;
                store: for (ap_uint<4> p = 0; p < 8; p++){
                    #pragma HLS UNROLL
                    bufout1.range(p*8+7, p*8) = buf1[p].range(7,0);
                    bufout2.range(p*8+7, p*8) = buf2[p].range(7,0);
                }

                fifo_out1.write(bufout1);
                fifo_out2.write(bufout2);
            }
        }

        if(readCounter < maxReads) {
            int line = (i + MAX_LINES);
            for (int x = 0; x < map_size_up_padded*channel; x++){
                #pragma HLS LOOP_TRIPCOUNT max=258
                #pragma HLS PIPELINE II=1
                tmp1 = fifo_in.read();
                readCounter++;
                mapa_inC2D[(line%MAX_LINES)][x].range(63,0) = tmp1.range(63,0);
            }
        }
    }
}


void HW_writeData(
    hls::stream<bus64_t>      &strm_out,
    hls::stream<ap_uint<64>>  &fifo_in1,
    hls::stream<ap_uint<64>>  &fifo_in2,
    config_t                   config
){
#pragma HLS INTERFACE axis port=strm_out
#pragma HLS INLINE off

    const unsigned int map_size = config.range(8,0).to_uint();        // input dimensions
    const unsigned int filters = config.range(17,16).to_uint();       // channel
    const unsigned int upsample = config.range(18,18).to_uint();

    const unsigned int map_size_up = map_size * (1 + upsample);
    const unsigned int half_j      = (map_size_up >> 1);

    // total number of 64-bit words produced for this frame
    const unsigned int TOTAL = filters * map_size_up * map_size_up;

    unsigned emitted = 0;

    bus64_t outw;
    outw.keep = 0xFF;
    outw.strb = 0xFF;
    outw.last = 0;

RowLoop:
    for (unsigned i = 0; i < map_size_up; ++i) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=256
    ColPairLoop:
        for (unsigned j = 0; j < half_j; ++j) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=128

            // Emit the first column group (fifo_in1): filters words
        Filt1Loop:
            for (unsigned k = 0; k < filters; ++k) {
            #pragma HLS PIPELINE II=1
                ap_uint<64> w = fifo_in1.read();
                outw.data = w;
                // TLAST only on very last word of the frame
                outw.last = ((emitted + 1) == TOTAL);
                strm_out.write(outw);
                emitted++;
            }

            // Emit the second column group (fifo_in2): filters words
        Filt2Loop:
            for (unsigned k = 0; k < filters; ++k) {
            #pragma HLS PIPELINE II=1
                ap_uint<64> w = fifo_in2.read();
                outw.data = w;
                outw.last = ((emitted + 1) == TOTAL);
                strm_out.write(outw);
                emitted++;
            }
        }
    }
}



void HW_layerC3D(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &skip_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
){
#pragma HLS interface axis port=strm_in
#pragma HLS interface axis port=skip_in
#pragma HLS interface axis port=strm_out
#pragma HLS INTERFACE s_axilite port=return bundle=AXILite
#pragma HLS INTERFACE s_axilite port=config bundle=AXILite
#pragma HLS DATAFLOW

static hls::stream<ap_uint<64>> fifoA("fifoA");
static hls::stream<ap_uint<64>> fifoB("fifoB");
static hls::stream<ap_uint<64>> fifo1("fifo1");
static hls::stream<ap_uint<64>> fifo2("fifo2");

#pragma HLS STREAM variable=fifoA depth=2048
#pragma HLS STREAM variable=fifoB depth=2064
#pragma HLS STREAM variable=fifo1 depth=2048
#pragma HLS STREAM variable=fifo2 depth=2048

    HW_upsample(strm_in, skip_in, fifoA, config);
    HW_readData(fifoA, fifoB, config);
    HW_conv3D(fifoB, fifo1, fifo2, config);
    HW_writeData(strm_out, fifo1, fifo2, config);
}

