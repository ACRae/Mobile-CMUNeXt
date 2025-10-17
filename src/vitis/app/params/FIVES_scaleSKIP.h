

// Auto-generated from quantized model
// scaleSKIP[layerID][2] = { x_shift, up_shift }
// Each entry is the CODE shift k to map s_in -> s_q:
//   k = round(log2(s_q / s_in))
//   k > 0 : code RIGHT shift by k  (>>)  [s_q > s_in]
//   k < 0 : code LEFT  shift by |k| (<<) [s_q < s_in]
const static int scaleSKIP[14][2] = {
  /* layer 0: stem.conv.0 */ { 0, 0 },
  /* layer 1: encoder1.up.conv.0 */ { 0, 0 },
  /* layer 2: encoder2.up.conv.0 */ { 0, 0 },
  /* layer 3: encoder3.up.conv.0 */ { 0, 0 },
  /* layer 4: encoder4.up.conv.0 */ { 0, 0 },
  /* layer 5: encoder5.up.conv.0 */ { 0, 0 },
  /* layer 6: Up5.up.1 */ { 0, 0 },
  /* layer 7: Up_conv5.conv.0 */ { 1, 1 },
  /* layer 8: Up4.up.1 */ { 0, 0 },
  /* layer 9: Up_conv4.conv.0 */ { 0, 1 },
  /* layer 10: Up3.up.1 */ { 0, 0 },
  /* layer 11: Up_conv3.conv.0 */ { 0, 0 },
  /* layer 12: Up2.up.1 */ { 0, 0 },
  /* layer 13: Up_conv2.conv.0 */ { 0, 0 },
};
