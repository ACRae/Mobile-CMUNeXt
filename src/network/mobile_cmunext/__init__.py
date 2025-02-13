from torch import nn

from network.NetworkABC import NetworkABC
from network.QuantizedNetworkABC import QuantizedNetworkABC

from .mobilecmunext import MobileCMUNeXt
from .quantized.mobile_cmunext_quant import MobileCMUNeXt_Quant


class MobileCMUNeXt_Network(NetworkABC):
    def __init__(self, input_channels=3, num_classes=1) -> None:
        super().__init__(input_channels, num_classes)

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 10, 12, 16, 24],
            depths=[3, 1, 1, 1, 2],
            kernels=[3, 3, 7, 7, 9],
            upsampling_mode="bilinear",
        )


class MobileCMUNeXt_Quant_Network(QuantizedNetworkABC):
    def __init__(
        self,
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        input_channels=3,
        num_classes=1,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            bias_bit_width=bias_bit_width,
        )

    def get_network(self) -> nn.Module:
        return MobileCMUNeXt_Quant(
            input_channel=self.input_channels,
            num_classes=self.num_classes,
            dims=[8, 10, 12, 16, 24],
            depths=[3, 1, 1, 1, 2],
            kernels=[3, 3, 7, 7, 9],
            weight_bit_width=self.weight_bit_width,
            act_bit_width=self.act_bit_width,
            bias_bit_width=self.bias_bit_width,
        )


__all__ = ["MobileCMUNeXt_Network", "MobileCMUNeXt_Quant_Network"]
