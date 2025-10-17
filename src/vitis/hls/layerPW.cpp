// layerPW.cpp
#include "layerPW.h"

void HW_pointWise(
    hls::stream<bus64_t>& strm_in,
    hls::stream<ap_uint<64>>& fifo_outA0,
    hls::stream<ap_uint<64>>& fifo_outA1,
    hls::stream<ap_uint<64>>& fifo_outB0,
    hls::stream<ap_uint<64>>& fifo_outB1,
    ap_uint<32> config
) {
#pragma HLS interface axis port=strm_in

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    unsigned int layer_ID = config.range(13,9).to_uint();       // layer identifier
    unsigned int channel = config.range(17,14).to_uint();      // in channels
    ap_uint<6> filters = config.range(23,18);      // out channels
    ap_uint<1> lastLayer = config.range(24,24);    // last layer flag

    const int scale = scalePW[layer_ID].to_int();

    acc_t acc1[8], acc2[8];
    #pragma HLS ARRAY_PARTITION variable=acc1 complete
    #pragma HLS ARRAY_PARTITION variable=acc2 complete

    ap_uint<64> line_buffer[2][256*4];

    bool fifoSel = false;

    // Process two lines at a time
    process_pairs: for (ap_uint<9> i = 0; i < map_size; i += 2) {
        #pragma HLS LOOP_TRIPCOUNT max=128

        for (int l = 0; l < 2; l++) {
            #pragma HLS UNROLL
            for (int j = 0; j < map_size * channel; j++) {
                #pragma HLS LOOP_TRIPCOUNT max=(256*4) min=1
                #pragma HLS PIPELINE II=1
                bus64_t value = strm_in.read();
                line_buffer[l][j].range(63, 0) = value.data.range(63, 0);
            }
        }

        // Process the two loaded lines
        map_col_opt: for (int j = 0; j < map_size; j++) {
            #pragma HLS LOOP_TRIPCOUNT max=256

            ft_opt: for (int fx = 0; fx < filters; fx++) {
                #pragma HLS LOOP_TRIPCOUNT max=8

                bias_init: for (int f = 0; f < 8; f++) {
                    #pragma HLS UNROLL
                    mult_t bias = biasPW[layer_ID][fx*8+f];
                    acc1[f] = bias;
                    acc2[f] = bias;
                }

                // Process input channels
                ch_opt: for (int ch_idx = 0; ch_idx < channel; ch_idx++) {
                    #pragma HLS LOOP_TRIPCOUNT max=8
                    #pragma HLS PIPELINE

                    int map_idx = (j * channel) + ch_idx;
                    ap_uint<64> data1 = line_buffer[0][map_idx];
                    ap_uint<64> data2 = line_buffer[1][map_idx];

                    // Unroll both data processing and MAC completely
                    data_mac: for (int c = 0; c < 8; c++) {
                        #pragma HLS UNROLL
                        data_t d1 = data1.range(c*8+7, c*8);
                        data_t d2 = data2.range(c*8+7, c*8);

                        filters_mac: for (int f = 0; f < 8; f++) {
                            #pragma HLS UNROLL
                            data_t weight = weightsPW[layer_ID][fx*8+f][ch_idx*8+c];
                            acc1[f] += d1 * weight;
                            acc2[f] += d2 * weight;
                        }
                    }
                }

                // Scale, pack and output in one step
                ap_uint<64> out1 = 0, out2 = 0;
                output_pack: for (int p = 0; p < 8; p++) {
                    #pragma HLS UNROLL
                    if (lastLayer) {
                        // sigmoid+0.5 threshold == sign test on logits
                        data_t b1 = (acc1[p] >= 0);
                        data_t b2 = (acc2[p] >= 0);

                        out1.range(p*8+7, p*8) = b1.range(7, 0);
                        out2.range(p*8+7, p*8) = b2.range(7, 0);
                    } else {
                        // normal hidden layers: ReLU + clamp to int8
                        data_t v1 = (data_t)CLAMP8_RELU_AP(acc1[p] >> scale);
                        data_t v2 = (data_t)CLAMP8_RELU_AP(acc2[p] >> scale);
                        out1.range(p*8+7, p*8) = v1.range(7, 0);
                        out2.range(p*8+7, p*8) = v2.range(7, 0);
                    }
                }

                // Alternate FIFO writes
                if (fifoSel) {
                    fifo_outA1.write(out1);
                    fifo_outB1.write(out2);
                } else {
                    fifo_outA0.write(out1);
                    fifo_outB0.write(out2);
                }
                fifoSel = !fifoSel;
            }
        }
    }
}

// CORRECT
void HW_writeData(
    hls::stream<bus64_t>& strm_out,
    hls::stream<ap_uint<64>>& fifo_outA0,
    hls::stream<ap_uint<64>>& fifo_outA1,
    hls::stream<ap_uint<64>>& fifo_outB0,
    hls::stream<ap_uint<64>>& fifo_outB1,
    config_t config
){
#pragma HLS interface axis port=strm_out

    ap_uint<9> map_size = config.range(8,0);        // input dimensions
    ap_uint<6> filters = config.range(23,18);      // out channels
    ap_uint<16> total_iterations = (map_size * filters) >> 1; // divide by 2
    ap_uint<16> last_iteration = total_iterations - 1;
    ap_uint<1> lastLayer = config.range(24,24);    // last layer flag

    bus64_t bus_template;
    bus_template.keep = 0xFF;
    bus_template.strb = 0xFF;
    bus_template.last = 0;

    if (lastLayer) {
        // Emit HxWx1 in HWC order: 8 consecutive pixels (same row) per 64-bit word.
        const ap_uint<16> words_per_row = map_size >> 3; // 8 pixels per word
        const ap_uint<16> total_output_words = (map_size * map_size) >> 3;

        ap_uint<16> word_count = 0;

        // process rows in pairs (A FIFOs = row y, B FIFOs = row y+1)
        for (ap_uint<9> y = 0; y < map_size; y += 2) {
        #pragma HLS LOOP_TRIPCOUNT max=256
            // --- Row y (use A0/A1 only) ---
            for (ap_uint<16> x = 0; x < words_per_row; ++x) {
            #pragma HLS LOOP_TRIPCOUNT max=32
            #pragma HLS PIPELINE II=4
                ap_uint<64> w1 = fifo_outA0.read();
                ap_uint<64> w2 = fifo_outA1.read();
                ap_uint<64> w3 = fifo_outA0.read();
                ap_uint<64> w4 = fifo_outA1.read();
                ap_uint<64> w5 = fifo_outA0.read();
                ap_uint<64> w6 = fifo_outA1.read();
                ap_uint<64> w7 = fifo_outA0.read();
                ap_uint<64> w8 = fifo_outA1.read();

                ap_uint<64> out_word = 0;
                out_word.range( 7, 0) = w1.range(7, 0);
                out_word.range(15, 8) = w2.range(7, 0);
                out_word.range(23,16) = w3.range(7, 0);
                out_word.range(31,24) = w4.range(7, 0);
                out_word.range(39,32) = w5.range(7, 0);
                out_word.range(47,40) = w6.range(7, 0);
                out_word.range(55,48) = w7.range(7, 0);
                out_word.range(63,56) = w8.range(7, 0);

                bus64_t out = bus_template;
                out.data = out_word;
                out.last = (word_count == total_output_words - 1) ? 1 : 0;
                strm_out.write(out);
                ++word_count;
            }

            // --- Row y+1 (use B0/B1 only) ---
            if (y + 1 < map_size) {
                for (ap_uint<16> x = 0; x < words_per_row; ++x) {
                #pragma HLS LOOP_TRIPCOUNT max=32
                #pragma HLS PIPELINE II=4
                    ap_uint<64> w1 = fifo_outB0.read();
                    ap_uint<64> w2 = fifo_outB1.read();
                    ap_uint<64> w3 = fifo_outB0.read();
                    ap_uint<64> w4 = fifo_outB1.read();
                    ap_uint<64> w5 = fifo_outB0.read();
                    ap_uint<64> w6 = fifo_outB1.read();
                    ap_uint<64> w7 = fifo_outB0.read();
                    ap_uint<64> w8 = fifo_outB1.read();

                    ap_uint<64> out_word = 0;
                    out_word.range( 7, 0) = w1.range(7, 0);
                    out_word.range(15, 8) = w2.range(7, 0);
                    out_word.range(23,16) = w3.range(7, 0);
                    out_word.range(31,24) = w4.range(7, 0);
                    out_word.range(39,32) = w5.range(7, 0);
                    out_word.range(47,40) = w6.range(7, 0);
                    out_word.range(55,48) = w7.range(7, 0);
                    out_word.range(63,56) = w8.range(7, 0);

                    bus64_t out = bus_template;
                    out.data = out_word;
                    out.last = (word_count == total_output_words - 1) ? 1 : 0;
                    strm_out.write(out);
                    ++word_count;
                }
            }
        }
    } else {
        LOOP_Y: for (ap_uint<9> y = 0; y < map_size; y++) {
            #pragma HLS LOOP_TRIPCOUNT max=256

            const bool use_A_fifos = (y & 1) == 0;
            const bool is_last_row = (y == map_size - 1);

            LOOP_X: for (ap_uint<16> x = 0; x < total_iterations; x++) {
                #pragma HLS LOOP_TRIPCOUNT max=512 //(256*4)/2
                #pragma HLS PIPELINE II=2

                // Use local variables to help with timing
                bus64_t tmp1 = bus_template;
                bus64_t tmp2 = bus_template;

                // Conditional read based on pre-calculated flag
                if (use_A_fifos) {
                    tmp1.data = fifo_outA0.read();
                    tmp2.data = fifo_outA1.read();
                } else {
                    tmp1.data = fifo_outB0.read();
                    tmp2.data = fifo_outB1.read();
                }

                const bool is_last_element = (is_last_row && (x == last_iteration));
                tmp2.last = is_last_element ? 1 : 0;

                strm_out.write(tmp1);
                strm_out.write(tmp2);
            }
        }
    }
}




void HW_layerPW(
    hls::stream<bus64_t> &strm_in,
    hls::stream<bus64_t> &strm_out,
    config_t config
) {
    #pragma HLS interface axis port=strm_in
    #pragma HLS interface axis port=strm_out
    #pragma HLS INTERFACE s_axilite port=return bundle=AXILite
    #pragma HLS INTERFACE s_axilite port=config bundle=AXILite
    #pragma HLS DATAFLOW

    static hls::stream<ap_uint<64>> fifoA0("fifoA0");
    static hls::stream<ap_uint<64>> fifoA1("fifoA1");
    static hls::stream<ap_uint<64>> fifoB0("fifoB0");
    static hls::stream<ap_uint<64>> fifoB1("fifoB1");

#pragma HLS STREAM variable=fifoA0 depth=1024
#pragma HLS STREAM variable=fifoA1 depth=1024  
#pragma HLS STREAM variable=fifoB0 depth=1024
#pragma HLS STREAM variable=fifoB1 depth=1024


    HW_pointWise(strm_in, fifoA0, fifoA1, fifoB0, fifoB1, config);
    HW_writeData(strm_out, fifoA0, fifoA1, fifoB0, fifoB1, config);
}
