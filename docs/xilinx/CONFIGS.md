# Configurations

Each layer type (**C3D, DW, PW**) encodes its parameters inside a packed `config` 32 bit uint.
The tables below list each field, its **bit range**, **width**, and meaning.

---

## C3D Configuration

| Field        | Bit Range | Width | Description        |
| ------------ | --------- | ----- | ------------------ |
| `map_size`   | 8–0       | 9     | Input dimensions   |
| `layer_ID`   | 13–9      | 5     | Layer identifier   |
| `channel`    | 15–14     | 2     | Input channel      |
| `filters`    | 17–16     | 2     | Output channel     |
| `upsample`   | 18–18     | 1     | Upsampling flag    |
| `firstLayer` | 19–19     | 1     | First layer flag   |
| `skipCon`    | 20–20     | 1     | Sum skip con Flag  |


ap_uint<9> map_size = config.range(8,0);        // input dimensions
ap_uint<5> layer_ID = config.range(13,9);       // layer identifier
ap_uint<2> channel = config.range(15,14);       // channel
ap_uint<2> filters = config.range(17,16);       // channel
ap_uint<1> upsample = config.range(18,18);
ap_uint<1> firstLayer = config.range(19,19);
ap_uint<1> skipCon = config.range(20,20);


config.range(8,0) = ;   // map size
config.range(13,9) = ;  // layer id
config.range(15,14) = ; // input channel
config.range(17,16) = ; // output channel
config.range(18,18) = ; // upsample
config.range(19,19) = ; // first layer
config.range(20,20) = ; // Sum skip con


---

## DW (Depthwise) Configuration

| Field         | Bit Range | Width | Description      |
| ------------- | --------- | ----- | ---------------- |
| `map_size`    | 8–0       | 9     | Input dimensions |
| `layer_ID`    | 13–9      | 5     | Layer identifier |
| `channel`     | 15–14     | 2     | Input channel    |
| `kernel_size` | 19–16     | 4     | Kernel size      |
| `pad`         | 22–20     | 3     | Padding size     |
| `maxpool`     | 23–23     | 1     | Maxpool flag     |


ap_uint<9> map_size = config.range(8,0);        // input dimensions
ap_uint<5> layer_ID = config.range(13,9);       // layer identifier
ap_uint<2> channel = config.range(15,14);
ap_uint<4> kernel_size = config.range(19,16);
ap_uint<3> pad = config.range(22,20);
ap_uint<1> maxpool = config.range(23,23);


config.range(8,0) = ;   // map size
config.range(13,9) = ;  // layer id
config.range(15,14) = ; // input channel
config.range(19,16) = ; // kernel size
config.range(22,20) = ; // padding
config.range(23,23) = ; // maxpool

---

## PW (Pointwise) Configuration

| Field      | Bit Range | Width | Description      |
| ---------- | --------- | ----- | ---------------- |
| `map_size` | 8–0       | 9     | Input dimensions |
| `layer_ID` | 13–9      | 5     | Layer identifier |
| `channel`  | 17–14     | 4     | Input channels   |
| `filters`  | 23–18     | 6     | Output channels  |
| `lastLayer`| 24–24     | 1     | Last Layer Flag  |

---

ap_uint<9> map_size = config.range(8,0);        // input dimensions
ap_uint<5> layer_ID = config.range(13,9);       // layer identifier
ap_uint<4> channel = config.range(17,14);      // in channels
ap_uint<6> filters = config.range(23,18);      // out channels
ap_uint<1> lastLayer = config.range(24,24);    // last layer flag

config.range(8,0) = ;       // map size
config.range(13,9) = ;      // layer id
config.range(17,14) = ;     // input channels
config.range(23,18) = ;     // output channels
config.range(24,24) = ;     // last layer flag
