import inspect

from torch.optim import lr_scheduler


schedulers = {
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "MultiStepLR": lr_scheduler.MultiStepLR,
}

__all__ = list(schedulers.keys())


def get_scheduler(scheduler_name, optmzr, **kwargs):
    if scheduler_name not in schedulers:
        raise NotImplementedError(
            f"Scheduler '{scheduler_name}' is not implemented. Available schedulers: {list(schedulers.keys())}"
        )

    scheduler_class = schedulers[scheduler_name]
    valid_params = list(inspect.signature(scheduler_class).parameters)
    valid_params.remove("optimizer")
    valid_params.remove("verbose")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return scheduler_class(optimizer=optmzr, **filtered_kwargs)
