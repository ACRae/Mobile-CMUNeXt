import brevitas.nn as qnn
from torch import nn

from network.NetworkABC import NetworkABC
from network.QuantizedNetworkABC import QuantizedNetworkABC

from .mobilecmunext import MobileCMUNeXt
from .mobilecmunext_bn_act import MobileCMUNeXt_BN_ACT
from .quantized.mobile_cmunext_quant_act import MobileCMUNeXt_Quant_ACT
from .quantized.mobile_cmunext_quant_bn_act import MobileCMUNeXt_Quant_BN_ACT


class MobileCMUNeXt_Hardswish_Network(NetworkABC):
    def __init__(self, depths, dims, kernels, input_channels=3, num_classes=1) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            kernels=kernels
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=self.depths if self.depths is not None else [3, 1, 1, 1, 2],
            kernels=self.kernels if self.kernels is not None else [3, 3, 7, 7, 9],
            upsampling_mode="bilinear",
            act=nn.Hardswish,
        )


class MobileCMUNeXt_Nearest_Hardswish_Network(NetworkABC):
    def __init__(self, depths, dims, kernels, input_channels=3, num_classes=1) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            kernels=kernels
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=self.depths if self.depths is not None else [3, 1, 1, 1, 2],
            kernels=self.kernels if self.kernels is not None else [3, 3, 7, 7, 9],
            upsampling_mode="nearest",
            act=nn.ReLU,
        )


class MobileCMUNeXt_RELU_Network(NetworkABC):
    def __init__(self, depths, dims, kernels, input_channels=3, num_classes=1) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            kernels=kernels
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=self.depths if self.depths is not None else [3, 1, 1, 1, 2],
            kernels=self.kernels if self.kernels is not None else [3, 3, 7, 7, 9],
            upsampling_mode="bilinear",
            act=nn.ReLU,
        )


class MobileCMUNeXt_RELU_BN_ACT_Network(NetworkABC):
    def __init__(self, depths, dims, kernels, input_channels=3, num_classes=1) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            kernels=kernels
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt_BN_ACT(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=self.depths if self.depths is not None else [3, 1, 1, 1, 2],
            kernels=self.kernels if self.kernels is not None else [3, 3, 7, 7, 9],
            upsampling_mode="bilinear",
            act=nn.ReLU,
        )


class MobileCMUNeXt_LeakyRELU_Network(NetworkABC):
    def __init__(self, depths, dims, kernels, input_channels=3, num_classes=1) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            kernels=kernels
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=self.depths if self.depths is not None else [3, 1, 1, 1, 2],
            kernels=self.kernels if self.kernels is not None else [3, 3, 7, 7, 9],
            upsampling_mode="bilinear",
            act=nn.LeakyReLU,
        )

class MobileCMUNeXt_Quant_RELU_BN_ACT_Network(QuantizedNetworkABC):
    def __init__(
        self,
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        input_channels=3,
        num_classes=1,
        bn=True,
        dims=None
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bias_bit_width=bias_bit_width,
            bn=bn,
            dims=dims
        )

    def get_network(self) -> nn.Module:
        if self.bn:
            return MobileCMUNeXt_Quant_BN_ACT(
                input_channel=self.input_channels,
                num_classes=self.num_classes,
                dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
                depths=[3, 1, 1, 1, 2],
                kernels=[3, 3, 7, 7, 9],
                weight_bit_width=self.weight_bit_width,
                act_bit_width=self.act_bit_width,
                bias_bit_width=self.bias_bit_width,
                qact=qnn.QuantReLU,
            )
        return MobileCMUNeXt_Quant_ACT(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=self.dims if self.dims is not None else [8, 10, 12, 16, 24],
            depths=[3, 1, 1, 1, 2],
            kernels=[3, 3, 7, 7, 9],
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            bias_bit_width=self.bias_bit_width,
            qact=qnn.QuantReLU,
        )
