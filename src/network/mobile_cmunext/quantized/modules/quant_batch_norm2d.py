import brevitas.nn as qnn
from brevitas.nn.quant_layer import ActQuantType
from torch import nn

from .quant_config import ActQuant


class QuantBatchNorm2d(nn.Module):
    def __init__(self, in_channels, act_quant: ActQuantType | None, bit_width=8, return_quant_tensor=True):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=bit_width, return_quant_tensor=False
        )
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, bit_width=bit_width
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.bn(x)
        x = self.output_quant(x)
        return x
