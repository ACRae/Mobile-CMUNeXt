import brevitas.nn as qnn
from torch import nn

from network.mobile_cmunext.quantized.quant_config import (
    ActQuant,
    BiasQuant,
    WeightQuant,
    get_act_bit_width,
    get_weight_bit_width,
    set_bit_widths,
)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.input_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=get_act_bit_width(), return_quant_tensor=True
        )

    def forward(self, x):
        fn_x = self.fn(x)
        qfn_x = self.input_quant(fn_x)
        qx = self.input_quant(x)
        return qfn_x + qx


class MobileCMUNeXtBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=1, k=3, qact=qnn.QuantReLU):
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    # First depthwise convolution
                    Residual(
                        nn.Sequential(
                            qnn.QuantConv2d(
                                in_channels=ch_in,
                                out_channels=ch_in,
                                kernel_size=k,
                                stride=1,
                                padding=(k // 2, k // 2),
                                groups=ch_in,
                                bias=True,
                                bias_quant=BiasQuant,
                                weight_quant=WeightQuant,
                                weight_bit_width=get_weight_bit_width(),
                                return_quant_tensor=True,
                            ),
                            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
                            qact(
                                act_quant=ActQuant,
                                bit_width=get_act_bit_width(),
                                return_quant_tensor=True,
                            ),
                        )
                    ),
                    # Second depthwise convolution
                    Residual(
                        nn.Sequential(
                            qnn.QuantConv2d(
                                in_channels=ch_in,
                                out_channels=ch_in,
                                kernel_size=k,
                                stride=1,
                                padding=(k // 2, k // 2),
                                groups=ch_in,
                                bias=True,
                                bias_quant=BiasQuant,
                                weight_quant=WeightQuant,
                                weight_bit_width=get_weight_bit_width(),
                                return_quant_tensor=True,
                            ),
                            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
                            qact(
                                act_quant=ActQuant,
                                bit_width=get_act_bit_width(),
                                return_quant_tensor=True,
                            ),
                        )
                    ),
                    qnn.QuantConv2d(
                        in_channels=ch_in,
                        out_channels=ch_in * 4,
                        kernel_size=1,
                        bias=True,
                        bias_quant=BiasQuant,
                        weight_quant=WeightQuant,
                        weight_bit_width=get_weight_bit_width(),
                        return_quant_tensor=True,
                    ),
                    qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
                    qact(
                        act_quant=ActQuant,
                        bit_width=get_act_bit_width(),
                        return_quant_tensor=True,
                    ),
                    qnn.QuantConv2d(
                        in_channels=ch_in * 4,
                        out_channels=ch_in,
                        kernel_size=1,
                        bias=True,
                        bias_quant=BiasQuant,
                        weight_quant=WeightQuant,
                        weight_bit_width=get_weight_bit_width(),
                        return_quant_tensor=True,
                    ),
                    qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
                    qact(
                        act_quant=ActQuant,
                        bit_width=get_act_bit_width(),
                        return_quant_tensor=True,
                    ),
                )
                for _ in range(depth)
            ]
        )
        self.up = ConvBlock(ch_in, ch_out, qact=qact)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, qact=qnn.QuantReLU):
        super().__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                bias_quant=BiasQuant,
                weight_quant=WeightQuant,
                weight_bit_width=get_weight_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
            qact(
                act_quant=ActQuant,
                bit_width=get_act_bit_width(),
                return_quant_tensor=True,
            ),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out, qact=qnn.QuantReLU):
        super().__init__()
        self.up = nn.Sequential(
            qnn.QuantUpsamplingBilinear2d(scale_factor=2, return_quant_tensor=True),
            qnn.QuantConv2d(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                bias_quant=BiasQuant,
                weight_quant=WeightQuant,
                weight_bit_width=get_weight_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
            qact(
                act_quant=ActQuant,
                bit_width=get_act_bit_width(),
                inplace=True,
                return_quant_tensor=True,
            ),
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FusionConv(nn.Module):
    def __init__(self, ch_in, ch_out, qact=qnn.QuantReLU):
        super().__init__()
        self.conv = nn.Sequential(
            qnn.QuantConv2d(
                in_channels=ch_in,
                out_channels=ch_in,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=True,
                bias_quant=BiasQuant,
                weight_quant=WeightQuant,
                weight_bit_width=get_weight_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
            qact(
                act_quant=ActQuant,
                bit_width=get_act_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantConv2d(
                in_channels=ch_in,
                out_channels=ch_out * 4,
                kernel_size=(1, 1),
                bias=True,
                bias_quant=BiasQuant,
                weight_quant=WeightQuant,
                weight_bit_width=get_weight_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
            qact(
                act_quant=ActQuant,
                bit_width=get_act_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantConv2d(
                in_channels=ch_out * 4,
                out_channels=ch_out,
                kernel_size=(1, 1),
                bias=True,
                bias_quant=BiasQuant,
                weight_quant=WeightQuant,
                weight_bit_width=get_weight_bit_width(),
                return_quant_tensor=True,
            ),
            qnn.QuantIdentity(act_quant=None, return_quant_tensor=True),
            qact(
                act_quant=ActQuant,
                bit_width=get_act_bit_width(),
                return_quant_tensor=True,
            ),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobileCMUNeXt_Quant_ACT(nn.Module):
    def __init__(
        self,
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        input_channel=3,
        num_classes=1,
        dims=[16, 32, 128, 160, 256],
        depths=[1, 1, 1, 3, 1],
        kernels=[3, 3, 7, 7, 7],
        qact=qnn.QuantReLU,
    ):
        """
        Args:
            input_channel : input channel.
            num_classes: output channel.
            dims: length of channels
            depths: length of cmunext blocks
            kernels: kernal size of cmunext blocks

        Improvements Done:
            * Altered Concat to sum and altered shapes
            * Changed Activation Function to Hardswish
            * Added a second depthwise convolution
        """
        super().__init__()

        set_bit_widths(weight_bit_width, act_bit_width, bias_bit_width)

        self.input_quant = qnn.QuantIdentity(act_quant=ActQuant, bit_width=8, return_quant_tensor=True)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = ConvBlock(ch_in=input_channel, ch_out=dims[0], qact=qact)
        self.encoder1 = MobileCMUNeXtBlock(
            ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0], qact=qact
        )
        self.encoder2 = MobileCMUNeXtBlock(
            ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1], qact=qact
        )
        self.encoder3 = MobileCMUNeXtBlock(
            ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2], qact=qact
        )
        self.encoder4 = MobileCMUNeXtBlock(
            ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3], qact=qact
        )
        self.encoder5 = MobileCMUNeXtBlock(
            ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4], qact=qact
        )
        # Decoder
        self.Up5 = UpConv(ch_in=dims[4], ch_out=dims[3], qact=qact)
        self.Up_conv5 = FusionConv(ch_in=dims[3], ch_out=dims[3], qact=qact)
        self.Up4 = UpConv(ch_in=dims[3], ch_out=dims[2], qact=qact)
        self.Up_conv4 = FusionConv(ch_in=dims[2], ch_out=dims[2], qact=qact)
        self.Up3 = UpConv(ch_in=dims[2], ch_out=dims[1], qact=qact)
        self.Up_conv3 = FusionConv(ch_in=dims[1], ch_out=dims[1], qact=qact)
        self.Up2 = UpConv(ch_in=dims[1], ch_out=dims[0], qact=qact)
        self.Up_conv2 = FusionConv(ch_in=dims[0], ch_out=dims[0], qact=qact)
        self.Conv_1x1 = qnn.QuantConv2d(
            in_channels=dims[0],
            out_channels=num_classes,
            kernel_size=1,
            padding=0,
            bias=True,
            bias_quant=BiasQuant,
            weight_quant=WeightQuant,
            weight_bit_width=get_weight_bit_width(),
            return_quant_tensor=True,
        )
        self.x_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=get_act_bit_width(), return_quant_tensor=True
        )
        self.output_quant = qnn.QuantIdentity(act_quant=ActQuant, bit_width=8, return_quant_tensor=True)

    def forward(self, x):
        x1 = self.input_quant(x)
        x1 = self.stem(x1)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5) # uspample (C3D) output scale OK
        x4 = self.x_quant(x4)
        d5 = self.x_quant(d5)
        d5 = x4 + d5
        d5 = self.Up_conv5(d5) # fusion (C3D) skip con scale MODIFY

        d4 = self.Up4(d5)
        x3 = self.x_quant(x3)
        d4 = self.x_quant(d4)
        d4 = x3 + d4
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.x_quant(x2)
        d3 = self.x_quant(d3)
        d3 = x2 + d3
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.x_quant(x1)
        d2 = self.x_quant(d2)
        d2 = x1 + d2
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        d1 = self.output_quant(d1)
        return d1.tensor
