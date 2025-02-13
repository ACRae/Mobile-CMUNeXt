import inspect

from torch import nn

from network.NetworkABC import NetworkABC
from network.QuantizedNetworkABC import QuantizedNetworkABC
from network.attention_unet import AttentionUNet_Network
from network.cfpnetm import CFPNetM_Network
from network.cmunext import (
    CMUNeXtAddS_Network,
    CMUNeXtL_Network,
    CMUNeXtS_Kernel_3_Network,
    CMUNeXtS_Network,
    CMUNeXt_Network,
)
from network.givted_net import GIVTEDNet_Network
from network.lucf_net import LUCFNet_Network
from network.mobile_cmunext import MobileCMUNeXt_Network, MobileCMUNeXt_Quant_Network
from network.transunet import TrasnUnet_Network
from network.ulite.network import ULite_Network
from network.unet import UNet_Network
from network.unetpp import UNetpp_Network
from network.unetv2 import UNetV2_Network
from network.unext import UNeXtS_Network, UNeXt_Network


networks: dict[str, NetworkABC] = {
    "CMUNeXt": CMUNeXt_Network,
    "CMUNeXt-L": CMUNeXtL_Network,
    "CMUNeXt-S": CMUNeXtS_Network,
    "CMUNeXt-S-Kernel-3": CMUNeXtS_Kernel_3_Network,
    "CMUNeXt-Add-S": CMUNeXtAddS_Network,
    "Mobile-CMUNeXt": MobileCMUNeXt_Network,
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
    "Mobile-CMUNeXt-Quant": MobileCMUNeXt_Quant_Network,
}

__all__ = list(networks.keys()) + list(quant_networks.keys())


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
