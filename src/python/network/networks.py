import inspect

from torch import nn

from network.NetworkABC import NetworkABC
from network.QuantizedNetworkABC import QuantizedNetworkABC
from network.attention_unet import AttentionUNet_Network
from network.cfpnetm import CFPNetM_Network
from network.cmunext import (
    CMUNeXtAddS_Network,
    CMUNeXtL_Hardswish_Network,
    CMUNeXtL_LeakyRELU_Network,
    CMUNeXtL_Network,
    CMUNeXtL_RELU_Network,
    CMUNeXtS_Hardswish_Network,
    CMUNeXtS_Kernel_3_Network,
    CMUNeXtS_LeakyRELU_Network,
    CMUNeXtS_Network,
    CMUNeXtS_RELU_Network,
    CMUNeXtXXS_Network,
    CMUNeXt_Hardswish_Network,
    CMUNeXt_LeakyRELU_Network,
    CMUNeXt_Network,
    CMUNeXt_RELU_Network,
)
from network.givted_net import GIVTEDNet_Network
from network.lucf_net import LUCFNet_Network
from network.mobile_cmunext import (
    MobileCMUNeXt_Hardswish_Network,
    MobileCMUNeXt_LeakyRELU_Network,
    MobileCMUNeXt_Nearest_Hardswish_Network,
    MobileCMUNeXt_Quant_RELU_BN_ACT_Network,
    MobileCMUNeXt_RELU_BN_ACT_Network,
    MobileCMUNeXt_RELU_Network,
)
from network.transunet import TrasnUnet_Network
from network.ulite.network import ULite_Network
from network.unet import UNet_Network
from network.unetpp import UNetpp_Network
from network.unetv2 import UNetV2_Network
from network.unext import UNeXtS_Network, UNeXt_Network


networks: dict[str, NetworkABC] = {
    "CMUNeXt": CMUNeXt_Network,
    "CMUNeXt-LeakyRELU": CMUNeXt_LeakyRELU_Network,
    "CMUNeXt-RELU": CMUNeXt_RELU_Network,
    "CMUNeXt-Hardswish": CMUNeXt_Hardswish_Network,

    "CMUNeXt-L": CMUNeXtL_Network,
    "CMUNeXt-L-LeakyRELU": CMUNeXtL_LeakyRELU_Network,
    "CMUNeXt-L-RELU": CMUNeXtL_RELU_Network,
    "CMUNeXt-L-Hardswish": CMUNeXtL_Hardswish_Network,

    "CMUNeXt-S": CMUNeXtS_Network,
    "CMUNeXt-S-LeakyRELU": CMUNeXtS_LeakyRELU_Network,
    "CMUNeXt-S-RELU": CMUNeXtS_RELU_Network,
    "CMUNeXt-S-Hardswish": CMUNeXtS_Hardswish_Network,

    "CMUNeXt-S-Kernel-3": CMUNeXtS_Kernel_3_Network,
    "CMUNeXt-XXS": CMUNeXtXXS_Network,
    "CMUNeXt-Add-S": CMUNeXtAddS_Network,
    "Mobile-CMUNeXt-Hardswish": MobileCMUNeXt_Hardswish_Network,
    "Mobile-CMUNeXt-RELU": MobileCMUNeXt_RELU_Network,
    "Mobile-CMUNeXt-RELU-BN-ACT": MobileCMUNeXt_RELU_BN_ACT_Network,
    "Mobile-CMUNeXt-Leaky-RELU": MobileCMUNeXt_LeakyRELU_Network,
    "Mobile-CMUNeXt-Nearest-Hardswish": MobileCMUNeXt_Nearest_Hardswish_Network,
    "CFPNetM": CFPNetM_Network,
    "UNeXt": UNeXt_Network,
    "UNeXt-S": UNeXtS_Network,
    "TransUnet": TrasnUnet_Network,
    "UNet_V2": UNetV2_Network,
    "AttentionUNet": AttentionUNet_Network,
    "UNet++": UNetpp_Network,
    "UNet": UNet_Network,
    "ULite": ULite_Network,
    "LUCF-Net": LUCFNet_Network,
    "GIVTED-Net": GIVTEDNet_Network,
}


quant_networks: dict[str, QuantizedNetworkABC] = {
    "Mobile-CMUNeXt-Quant-RELU-BN-ACT": MobileCMUNeXt_Quant_RELU_BN_ACT_Network,
}

__all__ = list(networks.keys()) + list(quant_networks.keys())


def get_non_quant_model(model_name: str, config: dict) -> nn.Module:
    """
    Retrieve a model instance based on the model name and configuration.

    Args:
        model_name (str): The name of the model.
        config (dict): A dictionary containing additional parameters to configure the model.

    Returns:
        Network instance or raises an error if the model is not found.
    """
    # Attempt to find the model class in quantized and standard networks
    if "-Quant" in model_name:
        model_name = model_name.replace("-Quant", "")

    model_class = networks.get(model_name)

    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in networks.")

    # Filter configuration parameters to those accepted by the model class
    valid_params = inspect.signature(model_class.__init__).parameters
    filtered_kwargs = {k: v for k, v in config.items() if k in valid_params}

    # Instantiate and return the model
    return model_class(**filtered_kwargs).get_network()


def get_model(model_name: str, config: dict) -> nn.Module:
    """
    Retrieve a model instance based on the model name and configuration.

    Args:
        model_name (str): The name of the model.
        config (dict): A dictionary containing additional parameters to configure the model.

    Returns:
        Network instance or raises an error if the model is not found.
    """
    # Attempt to find the model class in quantized and standard networks
    model_class = quant_networks.get(model_name) or networks.get(model_name)

    if model_class is None:
        raise ValueError(f"Model '{model_name}' not found in networks.")

    # Filter configuration parameters to those accepted by the model class
    valid_params = inspect.signature(model_class).parameters
    filtered_kwargs = {k: v for k, v in config.items() if k in valid_params}

    # Instantiate and return the model
    return model_class(**filtered_kwargs).get_network()
