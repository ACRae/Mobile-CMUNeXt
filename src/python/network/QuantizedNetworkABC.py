from abc import ABC, abstractmethod

from torch import nn


class QuantizedNetworkABC(ABC):
    def __init__(
        self,
        weight_bit_width,
        act_bit_width,
        bias_bit_width,
        input_channels=3,
        num_classes=1,
        input_h=256,
        input_w=256,
        bn=True,
        dims=None,
        depths=None,
        kernels=None,
    ) -> None:
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_h = input_h
        self.input_w = input_w
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.bias_bit_width = bias_bit_width
        self.bn = bn
        self.dims = dims
        self.depths = depths
        self.kernels = kernels

    @abstractmethod
    def get_network(self) -> nn.Module:
        """
        Genreic method to obtain netowrk
        """
