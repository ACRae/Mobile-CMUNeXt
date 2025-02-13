import brevitas.nn as qnn
from brevitas.nn.quant_layer import ActQuantType
from torch import Tensor, nn

from .quant_config import ActQuant


class QuantHardswish(nn.Module):

    def __init__(
        self, act_quant: ActQuantType | None, bit_width=8, return_quant_tensor=True, inplace: bool = False
    ):
        super().__init__()
        self.inplace = inplace
        self.input_quant = qnn.QuantIdentity(
            act_quant=ActQuant, bit_width=bit_width, return_quant_tensor=False
        )
        self.output_quant = qnn.QuantIdentity(
            act_quant=act_quant, return_quant_tensor=return_quant_tensor, bit_width=bit_width
        )

    def _quant_hardswish(self, x: Tensor) -> Tensor:
        x = x if self.inplace else x.clone()
        QUANT_MUL = 1.75

        # Calculate min(6, x + 3)
        shifted = x + 3
        shifted.clamp_(max=6)

        # Calculate max(0, min(6, x + 3))
        shifted.clamp_(min=0)

        # Multiply by x and divide by 6
        x.mul_(shifted).mul_(QUANT_MUL)
        return x

    def forward(self, input: Tensor) -> Tensor:
        x = self.input_quant(input)
        x = self._quant_hardswish(x)
        x = self.output_quant(x)
        return x
