

// Auto-generated from quantized model
// scaleRES[layerID][2] = { x_shift, fn_shift }
// Each shift is the RIGHT shift k needed to convert s_in -> s_q:
//     k = round(log2(s_in / s_q))   (negative means left-shift)
const static int scaleRES[][2] = {
  /* layer 0: encoder1.block.0.0 */ { 1, -1 },
  /* layer 1: encoder1.block.0.1 */ { 1, 0 },
  /* layer 2: encoder1.block.1.0 */ { 0, 0 },
  /* layer 3: encoder1.block.1.1 */ { 0, 0 },
  /* layer 4: encoder1.block.2.0 */ { -1, 0 },
  /* layer 5: encoder1.block.2.1 */ { 1, 0 },
  /* layer 6: encoder2.block.0.0 */ { 0, 0 },
  /* layer 7: encoder2.block.0.1 */ { 0, 0 },
  /* layer 8: encoder3.block.0.0 */ { 0, 1 },
  /* layer 9: encoder3.block.0.1 */ { 0, 0 },
  /* layer 10: encoder4.block.0.0 */ { 0, 1 },
  /* layer 11: encoder4.block.0.1 */ { 0, 1 },
  /* layer 12: encoder5.block.0.0 */ { -1, 0 },
  /* layer 13: encoder5.block.0.1 */ { 1, 1 },
  /* layer 14: encoder5.block.1.0 */ { 0, 0 },
  /* layer 15: encoder5.block.1.1 */ { 0, 0 },
};
