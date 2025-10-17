from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType, RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import StatsOp
from brevitas.quant import IntBias
from brevitas.quant.solver import ActQuantSolver, WeightQuantSolver
from dependencies import value


WEIGHT_BIT_WIDTH = 8
ACT_BIT_WIDTH = 8  # more sensitive
BIAS_BIT_WIDTH = 16  # no need to alter


def set_bit_widths(
    weight_bit_width=8,
    act_bit_width=8,
    bias_bit_width=16,
):
    global WEIGHT_BIT_WIDTH  # noqa: PLW0603
    global ACT_BIT_WIDTH  # noqa: PLW0603
    global BIAS_BIT_WIDTH  # noqa: PLW0603

    WEIGHT_BIT_WIDTH = weight_bit_width
    ACT_BIT_WIDTH = act_bit_width
    BIAS_BIT_WIDTH = bias_bit_width


def get_weight_bit_width():
    global WEIGHT_BIT_WIDTH  # noqa: PLW0602
    return WEIGHT_BIT_WIDTH


def get_act_bit_width():
    global ACT_BIT_WIDTH  # noqa: PLW0602
    return ACT_BIT_WIDTH


class BaseQuant(ExtendedInjector):
    """
    A base quantization configuration class that defines common properties
    for different quantization types (weights, activations, biases).
    """

    bit_width_impl_type = BitWidthImplType.CONST
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    zero_point_impl = ZeroZeroPoint
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = False
    bit_width = None
    signed = True
    narrow_range = True

    @value
    def quant_type():
        """
        Determines the quantization type based on the weight bit-width.
        """
        return QuantType.BINARY if WEIGHT_BIT_WIDTH == 1 else QuantType.INT


class WeightQuant(BaseQuant, WeightQuantSolver):
    """
    Quantization configuration for weights.
    """

    scaling_const = 1.0


class ActQuant(BaseQuant, ActQuantSolver):
    """
    Quantization configuration for activations.
    """

    signed = True


class BiasQuant(BaseQuant, IntBias):
    """
    Quantization configuration for biases.
    """

    bit_width = BIAS_BIT_WIDTH
