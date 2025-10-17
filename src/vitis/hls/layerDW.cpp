// layerDW.cpp
#include "layerDW.h"

#define MAX(a, b) (((data_t)(a) > (data_t)(b)) ? (a) : (b))

void HW_readData(
    hls::stream<bus64_t> &strm_in,
    hls::stream<ap_uint<64>> &fifo_out,
    hls::stream<ap_uint<64>> &res_con,
    ap_uint<32> config
) {
#pragma HLS interface axis port=strm_in

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    ap_uint<2> channel = config.range(15,14);
    ap_uint<3> pad = config.range(22,20);
    ap_uint<1> maxpool = config.range(23,23);


    ap_uint<9> map_size_maxpool = map_size >> maxpool;


    ap_uint<64> buffer0_ch0[128], buffer0_ch1[128];
    ap_uint<64> buffer1_ch0[128], buffer1_ch1[128];


    ap_uint<64> zeros = 0;
    bus64_t tmp1, tmp2;
    ap_uint<64> buf[4];
#pragma HLS ARRAY_PARTITION variable=buf type=complete dim=1

    // Write top padding
    paddTop: for (ap_uint<9> i = 0; i < (map_size_maxpool + pad*2) * channel * pad; i++) {
        #pragma HLS LOOP_TRIPCOUNT max=(256+2)
        fifo_out.write(zeros);
    }

    // Process data rows
    rd_line: for (ap_uint<9> i = 0; i < map_size; i++) {
        #pragma HLS LOOP_TRIPCOUNT max=256

        bool write_row = (maxpool == 0) || (i % 2 == 0);
        bool process_maxpool = (maxpool == 1) && (i % 2 == 1);

        // Left padding
        if (pad > 0 && write_row) {
            paddLeft: for (ap_uint<4> j = 0; j < pad * channel; j++) {
                #pragma HLS LOOP_TRIPCOUNT max=8
                fifo_out.write(zeros);
            }
        }

        // Process map data
        if (channel == 1) {
            // Single channel processing (unchanged)
            rd_map: for (int j = 0; j < map_size; j += 2) {
                #pragma HLS LOOP_TRIPCOUNT max=128
                #pragma HLS PIPELINE II=2

                strm_in.read(tmp1);
                strm_in.read(tmp2);

                if (maxpool == 0) {
                    fifo_out.write(tmp1.data);
                    fifo_out.write(tmp2.data);
                    res_con.write(tmp1.data);
                    res_con.write(tmp2.data);
                } else {
                    int idx = j / 2;

                    if ((i & 1) == 0) {
                        buffer0_ch0[idx] = tmp1.data.range(63, 0);
                        buffer1_ch0[idx] = tmp2.data.range(63, 0);
                    } else {
                        for (ap_uint<4> k = 0; k < 8; k++) {
                            #pragma HLS UNROLL
                            data_t curr1 = tmp1.data.range(k*8+7, k*8);
                            data_t curr2 = tmp2.data.range(k*8+7, k*8);
                            data_t prev1 = buffer0_ch0[idx].range(k*8+7, k*8);
                            data_t prev2 = buffer1_ch0[idx].range(k*8+7, k*8);

                            data_t max_val = MAX(MAX(prev1, prev2), MAX(curr1, curr2));
                            buffer0_ch0[idx].range(k*8+7, k*8) = max_val;
                        }

                        fifo_out.write(buffer0_ch0[idx]);
                        res_con.write(buffer0_ch0[idx]);
                    }
                }
            }
        } else {
            // Multi-channel processing with separate channel buffers
            rd_map2: for (int j = 0; j < map_size; j += 2) {
                #pragma HLS LOOP_TRIPCOUNT max=16
                #pragma HLS PIPELINE II=4

                // Read 4 data elements for 2 channels
                buf[0] = strm_in.read().data;
                buf[1] = strm_in.read().data;
                buf[2] = strm_in.read().data;
                buf[3] = strm_in.read().data;

                if (maxpool == 0) {
                    fifo_out.write(buf[0]);
                    fifo_out.write(buf[1]);
                    fifo_out.write(buf[2]);
                    fifo_out.write(buf[3]);

                    res_con.write(buf[0]);
                    res_con.write(buf[1]);
                    res_con.write(buf[2]);
                    res_con.write(buf[3]);
                } else {
                    int idx = j / 2;

                    if (i % 2 == 0) {
                        // Store first row data in separate channel buffers
                        buffer0_ch0[idx] = buf[0];
                        buffer0_ch1[idx] = buf[1];
                        buffer1_ch0[idx] = buf[2];
                        buffer1_ch1[idx] = buf[3];
                    } else {
                        // Process max pooling with separate buffers (no conflicts)
                        ap_uint<64> result0 = 0, result1 = 0;

                        maxpool_ch0: for (ap_uint<4> k = 0; k < 8; k++) {
                            #pragma HLS UNROLL
                            data_t max0 = MAX(buffer0_ch0[idx].range(k*8+7, k*8), buf[0].range(k*8+7, k*8));
                            data_t max1 = MAX(buffer1_ch0[idx].range(k*8+7, k*8), buf[2].range(k*8+7, k*8));
                            result0.range(k*8+7, k*8) = MAX(max0, max1);
                        }

                        maxpool_ch1: for (ap_uint<4> k = 0; k < 8; k++) {
                            #pragma HLS UNROLL
                            data_t max0 = MAX(buffer0_ch1[idx].range(k*8+7, k*8), buf[1].range(k*8+7, k*8));
                            data_t max1 = MAX(buffer1_ch1[idx].range(k*8+7, k*8), buf[3].range(k*8+7, k*8));
                            result1.range(k*8+7, k*8) = MAX(max0, max1);
                        }

                        if (process_maxpool) {
                            fifo_out.write(result0);
                            fifo_out.write(result1);
                            res_con.write(result0);
                            res_con.write(result1);
                        }
                    }
                }
            }
        }

        // Right padding
        if (pad > 0 && ((maxpool == 0) || (i % 2 == 1))) {
            paddRight: for (ap_uint<4> j = 0; j < pad * channel; j++) {
                #pragma HLS LOOP_TRIPCOUNT max=8
                fifo_out.write(zeros);
            }
        }
    }

    // Write bottom padding
    paddBottom: for (ap_uint<9> i = 0; i < (map_size_maxpool + pad*2) * channel * pad; i++) {
        #pragma HLS LOOP_TRIPCOUNT max=(32+2)*2
        fifo_out.write(zeros);
    }
}





/*
VITIS HLS CONV2D DEPTHWISE IMPLEMENTATION
OUTPUT DIM SHOULD BE SAME AS INPUT
*/
#define MAX_LINES 10
void HW_depthWise(
        hls::stream<ap_uint<64>> &fifo_in,
        hls::stream<ap_uint<256>> &fifo_out1, // convolution output
        hls::stream<ap_uint<64>> &res_con, // residual connection
        config_t config
    )
{
    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    unsigned int layer_ID = config.range(13,9).to_uint();       // layer identifier
    unsigned int channel = config.range(15,14).to_uint();
    ap_uint<4> kernel_size = config.range(19,16);
    ap_uint<3> pad = config.range(22,20);
    ap_uint<1> maxpool = config.range(23,23);


    const int scale = scaleDW[layer_ID].to_int();


    ap_uint<64> mapa_inDW[MAX_LINES][258];
#pragma HLS BIND_STORAGE variable=mapa_inDW type=ram_s2p impl=lutram // important, use LUTs instead of BRAMs

    ap_uint<64> tmp1, tmp2;

    ap_uint<9> map_size_maxpool = map_size >> maxpool;

    ap_uint<17> maxReads = ((map_size_maxpool+2*pad) * (map_size_maxpool+2*pad)) * channel; // correct
    ap_uint<17> readCounter = 0;

    ap_uint<5> initialLines;
    if (kernel_size == 3) initialLines = 4;
    if (kernel_size == 7) initialLines = 8;
    if (kernel_size == 9) initialLines = 10;

    int cl = (channel == 1) ? 4 : 2;
    int ck = (channel == 1) ? 0 : 8;

    // Doing 4 accumulations per channel at a time
    acc_t acc1[8], acc2[8], acc3[8], acc4[8];
    ap_uint<256> buffer;

    rd_k_lines: for (int i = 0; i < initialLines; i++){
        #pragma HLS LOOP_TRIPCOUNT max=10
        for (int j  = 0; j < (map_size_maxpool+2*pad)*channel; j++){
            #pragma HLS LOOP_TRIPCOUNT max=1024
            tmp1 = fifo_in.read();
            readCounter++;
            mapa_inDW[i][j].range(63,0) = tmp1.range(63,0);
        }
    }

    map_line: for (int i = 0; i < map_size_maxpool; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=256

        map_col: for (int j = 0; j < map_size_maxpool; j+=cl) { // columns (step by cl = 4 or 2)
            #pragma HLS LOOP_TRIPCOUNT max=64

            bias_init: for(int c = 0; c < 8; c++) {
                #pragma HLS UNROLL
                mult_t b1 = biasDW[layer_ID][c];
                mult_t b2 = biasDW[layer_ID][c+ck];
                acc1[c] = b1;
                acc2[c] = b2;
                acc3[c] = b1;
                acc4[c] = b2;
            }

            k1a: for (int ki = 0; ki < kernel_size; ki++) { // k-height
                #pragma HLS LOOP_TRIPCOUNT max=9
                int  ii = i + ki; // correct

                k2a: for (int  kj = 0; kj < kernel_size; kj++) { // k-width
                    #pragma HLS LOOP_TRIPCOUNT max=9
                    #pragma HLS PIPELINE

                    int  jj = (j + kj) * channel;
                    ch1: for (int c = 0; c < 8; c++){ // process 8 channels in parallel and 4 accums at a time. that means that 64 * 4 == 256
                        data_t v1 = mapa_inDW[(ii%MAX_LINES)][(jj+0)].range(c*8+7, c*8);
                        data_t v2 = mapa_inDW[(ii%MAX_LINES)][(jj+1)].range(c*8+7, c*8);
                        data_t v3 = mapa_inDW[(ii%MAX_LINES)][(jj+2)].range(c*8+7, c*8);
                        data_t v4 = mapa_inDW[(ii%MAX_LINES)][(jj+3)].range(c*8+7, c*8);

                        data_t w1 = weightsDW[layer_ID][ki][kj][c];
                        data_t w2 = weightsDW[layer_ID][ki][kj][c+ck];

                        acc1[c] += v1*w1;
                        acc2[c] += v2*w2;
                        acc3[c] += v3*w1;
                        acc4[c] += v4*w2;
                    }

                }
            }

            relu: for (int c = 0; c < 8; c++){
                #pragma HLS UNROLL
                data_t v1 = (data_t)CLAMP8_RELU_AP(acc1[c] >> scale);
                data_t v2 = (data_t)CLAMP8_RELU_AP(acc2[c] >> scale);
                data_t v3 = (data_t)CLAMP8_RELU_AP(acc3[c] >> scale);
                data_t v4 = (data_t)CLAMP8_RELU_AP(acc4[c] >> scale);
                buffer.range(c*8+7,c*8) = v1.range(7,0);
                buffer.range(c*8+7+64,c*8+64) = v2.range(7,0);
                buffer.range(c*8+7+128,c*8+128) = v3.range(7,0);
                buffer.range(c*8+7+192,c*8+192) = v4.range(7,0);
            }

            fifo_out1.write(buffer);
        }
        // read 1 line
        if(readCounter < maxReads) {
            int line = (i + initialLines);
            for (int x = 0; x < (map_size_maxpool+2*pad)*channel; x++){
                #pragma HLS LOOP_TRIPCOUNT max=1024
                tmp1 = fifo_in.read();
                readCounter++;
                mapa_inDW[(line%MAX_LINES)][x].range(63,0) = tmp1.range(63,0);
            }
        }
    }
}




void HW_writeData(
      hls::stream<bus64_t> &strm_out,
      hls::stream<ap_uint<256>> &fifo_in1,
      hls::stream<ap_uint<64>> &res_con,
      ap_uint<32> config
) {
#pragma HLS interface axis port=strm_out

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    unsigned int layer_ID = config.range(13,9).to_uint();       // layer identifier
    ap_uint<2> channel = config.range(15,14);
    ap_uint<1> maxpool = config.range(23,23);

    ap_uint<9> map_size_maxpool = map_size >> maxpool;

    ap_uint<128> buffer[128];
    ap_uint<64> buf, tmpB0, tmpB1;
    ap_uint<256> tmp1, tmp2;
    bus64_t tmpo;


    if (channel == 1){
        wr_line: for (int i = 0; i < map_size_maxpool*map_size_maxpool/4; i++){
            #pragma HLS LOOP_TRIPCOUNT max=16384
            #pragma HLS PIPELINE II=4

            tmp1 = fifo_in1.read();

            tmpo.keep = 0xFF;
            tmpo.strb = 0xFF;

            for (ap_uint<3> j = 0; j < 4; j++){ // 256/8 == 4
                tmpB0 = res_con.read(); // read 1 more

                for (ap_uint<4> c = 0; c < 8; c++) {
                    data_t v1 = tmp1.range(c*8+7 + j*64, c*8 + j*64);
                    data_t v2 = tmpB0.range(c*8+7, c*8);
                    
                    v1 = safe_shift_ap(v1, scaleRES[layer_ID][1]);
                    v2 = safe_shift_ap(v2, scaleRES[layer_ID][0]);

                    ap_int<9> sum = v1 + v2;
                    if (sum > 127) sum = 127;
                    if (sum < -128) sum = -128;

                    buf.range(8*c+7, 8*c) = sum.range(7,0);
                }
                if (i == map_size_maxpool*map_size_maxpool/4-1 && j == 4-1)
                    tmpo.last = 1;
                else
                    tmpo.last = 0;

                tmpo.data.range(63,0) = buf.range(63,0);
                strm_out.write(tmpo);
            }
        }
    }
    else{
        wr_line2: for (ap_uint<11> i = 0; i < map_size_maxpool*map_size_maxpool/4; i++) {
            #pragma HLS LOOP_TRIPCOUNT max=1024
            #pragma HLS PIPELINE II=8

            // --- Drain residuals EARLY ---
            ap_uint<64> r0[4], r1[4];
            #pragma HLS ARRAY_PARTITION variable=r0 complete
            #pragma HLS ARRAY_PARTITION variable=r1 complete
            for (int j=0;j<4;j++) { r0[j] = res_con.read(); }  // for first 256b
            for (int j=0;j<4;j++) { r1[j] = res_con.read(); }  // for second 256b

            // Now fetch conv results
            ap_uint<256> t0 = fifo_in1.read();
            ap_uint<256> t1 = fifo_in1.read();

            bus64_t o = {0}; o.keep = 0xFF; o.strb = 0xFF;

            // first 256b -> 4×64b using r0[]
            for (int j=0;j<4;j++) {
                ap_uint<64> buf = 0;
                for (int c=0;c<8;c++) {
                    data_t v1 = t0.range(c*8+7 + j*64, c*8 + j*64);
                    data_t v2 = r0[j].range(c*8+7, c*8);
                    
                    v1 = safe_shift_ap(v1, scaleRES[layer_ID][1]);
                    v2 = safe_shift_ap(v2, scaleRES[layer_ID][0]);
                    
                    ap_int<9> s = v1 + v2;

                    if (s > 127) s = 127; else if (s < -128) s = -128;
                    buf.range(8*c+7, 8*c) = s.range(7,0);
                }
                o.data = buf;
                o.last = 0;
                strm_out.write(o);
            }

            // second 256b -> 4×64b using r1[]
            for (int j=0;j<4;j++) {
                ap_uint<64> buf = 0;
                for (int c=0;c<8;c++) {
                    data_t v1 = t1.range(c*8+7 + j*64, c*8 + j*64);
                    data_t v2 = r1[j].range(c*8+7, c*8);

                    v1 = safe_shift_ap(v1, scaleRES[layer_ID][1]);
                    v2 = safe_shift_ap(v2, scaleRES[layer_ID][0]);
                    
                    ap_int<9> s = v1 + v2;

                    if (s > 127) s = 127; else if (s < -128) s = -128;
                    buf.range(8*c+7, 8*c) = s.range(7,0);
                }
                o.data = buf;
                o.last = (i == map_size_maxpool*map_size_maxpool/4-1 && j == 3) ? 1 : 0;
                strm_out.write(o);
            }
        }
    }
}




void HW_layerDW(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
) {
#pragma HLS interface axis port=strm_in
#pragma HLS interface axis port=strm_out
#pragma HLS INTERFACE s_axilite port=return bundle=AXILite
#pragma HLS INTERFACE s_axilite port=config bundle=AXILite
#pragma HLS DATAFLOW

static hls::stream<ap_uint<64>> f1("f1");
static hls::stream<ap_uint<256>> f2("f2");
static hls::stream<ap_uint<64>> r_fifo("r_con");

#pragma HLS STREAM variable=f1 depth=2048       // Guaranteed startup buffering
#pragma HLS STREAM variable=f2 depth=2048        // Ample rate mismatch buffer
#pragma HLS STREAM variable=r_fifo depth=4096   // Safe residual buffering

    HW_readData(strm_in, f1, r_fifo, config);
    HW_depthWise(f1, f2, r_fifo, config);
    HW_writeData(strm_out, f2, r_fifo, config);
}

