import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import (CALLBACK_REGISTRY,
                                             LR_SCHEDULER_REGISTRY,
                                             OPTIMIZER_REGISTRY, LightningCLI)

import helpers.callbacks as custom_callbacks
import helpers.lr_scheduler as custom_lr_scheduler

CALLBACK_REGISTRY.register_classes(
    pl.callbacks, pl.callbacks.Callback, custom_callbacks)
LR_SCHEDULER_REGISTRY.register_classes(
    torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler, custom_lr_scheduler)


class ConditioningLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes, nested_key="optimizer", link_to="model.init_args.optimizer_init")
        parser.add_lr_scheduler_args(
            LR_SCHEDULER_REGISTRY.classes, nested_key="lr_scheduler", link_to="model.init_args.lr_scheduler_init")
