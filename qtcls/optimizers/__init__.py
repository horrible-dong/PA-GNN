# Copyright (c) QIU Tian. All rights reserved.

from torch.optim import *


def build_optimizer(args, params):
    optimizer_name = args.optimizer.lower()

    if optimizer_name == 'sgd':
        return SGD(params, **args.optimizer_kwargs)

    if optimizer_name == 'adam':
        return Adam(params, **args.optimizer_kwargs)

    if optimizer_name == 'adamw':
        return AdamW(params, **args.optimizer_kwargs)

    if optimizer_name == 'rmsprop':
        return RMSprop(params, **args.optimizer_kwargs)

    raise ValueError(f"Optimizer '{optimizer_name}' is not found.")
