from thop import clever_format, profile
import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_macs_and_params(*input_dims, model, device):
    """
    Calculate the number of Multiply-Accumulate Operations (MACs)
        and model parameters for a given model.

    Args:
        *input_dims (int): The dimensions of the input tensor (height, width, channels)
                            that the model expects. These dimensions will be used to create
                            a dummy input tensor to profile the model.
        model (nn.Module): The PyTorch model for which MACs and parameters need to be calculated.
        device (torch.device): The device (CPU or GPU) where the model and input tensor
            will be allocated.

    Returns:
        tuple: A tuple containing:
            - macs (str): The number of MACs (Multiply-Accumulate Operations) formatted
                to 3 decimal places.
            - params (str): The total number of model parameters formatted to 3 decimal places.
    """
    input_tensor = torch.randn(1, *input_dims).to(device)
    params = count_params(model)
    macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    model.state_dict().clear()
    return macs, params
