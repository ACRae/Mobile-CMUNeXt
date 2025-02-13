import brevitas.nn as qnn
from brevitas.nn.quant_layer import ActQuantType
from torch import nn

from .quant_config import ActQuant


class QuantMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        act_quant: ActQuantType | None,
        bit_width=8,
        return_quant_tensor=True,
    ):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=bit_width, return_quant_tensor=False
        )
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            bit_width=bit_width,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.max(x)
        x = self.output_quant(x)
        return x
