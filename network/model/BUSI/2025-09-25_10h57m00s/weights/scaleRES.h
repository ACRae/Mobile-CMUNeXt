// Auto-generated from quantized model
// scaleRES[layerID][2] = { x_shift, fn_shift }
// Each entry is the CODE shift k to map s_in -> s_q:
//   k = round(log2(s_q / s_in))
//   k > 0 : code RIGHT shift by k  (>>)  [s_q > s_in]
//   k < 0 : code LEFT  shift by |k| (<<) [s_q < s_in]
static const int scaleRES[][2] = {
  /* layer 0: encoder1.block.0.0 */ { 1, 0 },
  /* layer 1: encoder1.block.0.1 */ { 0, -1 },
  /* layer 2: encoder1.block.1.0 */ { -1, 0 },
  /* layer 3: encoder1.block.1.1 */ { 1, 1 },
  /* layer 4: encoder1.block.2.0 */ { 0, 1 },
  /* layer 5: encoder1.block.2.1 */ { 0, -1 },
  /* layer 6: encoder2.block.0.0 */ { 0, 0 },
  /* layer 7: encoder2.block.0.1 */ { 0, 1 },
  /* layer 8: encoder3.block.0.0 */ { 0, 0 },
  /* layer 9: encoder3.block.0.1 */ { 0, 0 },
  /* layer 10: encoder4.block.0.0 */ { 0, 0 },
  /* layer 11: encoder4.block.0.1 */ { 0, 0 },
  /* layer 12: encoder5.block.0.0 */ { 0, 0 },
  /* layer 13: encoder5.block.0.1 */ { 1, 0 },
  /* layer 14: encoder5.block.1.0 */ { 0, -1 },
  /* layer 15: encoder5.block.1.1 */ { 1, -1 },
};
static const int scaleRES_ROWS = 16;
