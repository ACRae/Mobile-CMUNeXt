import inspect

from torch import optim


optimizers = {"SGD": optim.SGD, "Adam": optim.Adam, "AdamW": optim.AdamW}

__all__ = list(optimizers.keys())


def get_optimizer(optimizer_name, model_params, **kwargs):
    if optimizer_name not in optimizers:
        raise NotImplementedError(
            f"Optimizer '{optimizer_name}' is not implemented. Available schedulers: \
                {list(optimizers.keys())}"
        )

    optimizer_class = optimizers[optimizer_name]
    valid_params = inspect.signature(optimizer_class).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return optimizer_class(params=model_params, **filtered_kwargs)
