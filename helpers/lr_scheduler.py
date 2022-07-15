from typing import List

import torch
from pytorch_lightning.utilities.cli import LR_SCHEDULER_REGISTRY
from torch.optim import Optimizer


@LR_SCHEDULER_REGISTRY
class LinearWarmupPolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    """Decays the learning rate according to a polynomial schedule. When last_epoch=-1,
    sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epochs (int): Number of training epochs (epoch-based training) or
            training iterations (iteration-based training).
        power (float): Exponent of the polynomial schedule.
        min_lr (float): Final learning rate.
        last_epoch (int): The index of last epoch / iteration. Default: -1.

    Example:
        >>> max_epochs = 100
        >>> scheduler = PolynomialLR(optimizer, max_epochs=max_epochs)
        >>> for epoch in range(max_epochs):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self,
                 optimizer: Optimizer,
                 max_steps: int = None,
                 warmup_iters: int = 1500,
                 warmup_ratio: float = 1e-6,
                 power=0.9,
                 min_lr=0.,
                 last_epoch=-1):
        self.max_updates = max_steps
        self.warmup_iters = warmup_iters
        self.warmup_ratio = warmup_ratio
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:

        # warmup phase
        if self.last_epoch < self.warmup_iters:
            k = (1 - self.last_epoch / self.warmup_iters) * \
                (1 - self.warmup_ratio)
            return [_lr * (1 - k) for _lr in self.base_lrs]

        # poly phase
        else:
            coeff = (1 - (self.last_epoch - self.warmup_iters) /
                     float(self.max_updates - self.warmup_iters)) ** self.power
            return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]
