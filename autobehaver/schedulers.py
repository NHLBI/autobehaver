"""Schedule PosePred learning rates with batch steps and warmup transitions."""

from __future__ import annotations

from typing import Optional

import torch
from torch.optim.optimizer import Optimizer

__all__ = [
    "InverseSquareRootDecayScheduler",
    "CosineAnnealingScheduler",
    "LinearAnnealingScheduler",
    "EpochCosineScheduler",
]


class EpochCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Apply loader-sized warmup followed by cosine decay without restarts.

    Step after each optimizer batch. Convert configured epoch counts using
    the actual training loader, including files reassigned by ``--noval``.
    """

    def __init__(self, optimizer, batches_per_epoch, warmup_epochs, decay_epochs,
                 eta_min=0.0, last_epoch=-1):
        if batches_per_epoch < 1 or warmup_epochs < 0 or decay_epochs < 1:
            raise ValueError("Epoch cosine requires a nonempty loader, nonnegative warmup and positive decay.")
        self.warmup_steps = int(batches_per_epoch) * int(warmup_epochs)
        self.decay_steps = int(batches_per_epoch) * int(decay_epochs)
        self.eta_min = float(eta_min)
        if any(self.eta_min > group['lr'] for group in optimizer.param_groups):
            raise ValueError("eta_min must not exceed the base learning rate.")
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        import math
        step = max(0, self.last_epoch)
        if self.warmup_steps and step < self.warmup_steps:
            factor = 1e-5 + (1.0 - 1e-5) * step / self.warmup_steps
            return [base * factor for base in self.base_lrs]
        fraction = min(1.0, max(0.0, (step - self.warmup_steps) / self.decay_steps))
        return [self.eta_min + (base - self.eta_min) * (1 + math.cos(math.pi * fraction)) / 2
                for base in self.base_lrs]


class _SequentialLRMixin:
    """Helpers expected by ``SequentialLR`` during scheduler transitions."""

    scheduler: object

    def _update_lr(self, epoch: Optional[int] = None) -> None:
        """Update learning rates at a scheduler transition.

        ``SequentialLR`` uses this instead of ``step()`` when handing control to
        the next scheduler. Prefer the wrapped scheduler's native ``_update_lr``
        when available; otherwise fall back to ``step(epoch)``.
        """
        inner_update = getattr(self.scheduler, "_update_lr", None)
        if callable(inner_update):
            inner_update(epoch)
        elif epoch is None:
            self.step()
        else:
            self.step(epoch)

    def _initial_step(self) -> None:
        inner_initial = getattr(self.scheduler, "_initial_step", None)
        if callable(inner_initial):
            inner_initial()
        else:  # pragma: no cover - defensive path.
            self.step()

    @property
    def base_lrs(self):
        return getattr(self.scheduler, "base_lrs", [group["lr"] for group in self.optimizer.param_groups])

    @property
    def _last_lr(self):
        return getattr(self.scheduler, "_last_lr", [group["lr"] for group in self.optimizer.param_groups])


class InverseSquareRootDecayScheduler(_SequentialLRMixin):
    """Inverse-square-root LR decay with ``SequentialLR`` support."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        lr_lambda = lambda step: (step + 1) ** -0.5
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.scheduler.step()
        else:
            self.scheduler.step(epoch)

    def get_last_lr(self) -> list[float]:
        return list(self.scheduler.get_last_lr())

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    @last_epoch.setter
    def last_epoch(self, value):
        self.scheduler.last_epoch = value


class CosineAnnealingScheduler(_SequentialLRMixin):
    """Cosine warm-restart scheduler with ``SequentialLR`` support."""

    def __init__(self, optimizer: Optimizer, t_0_batches: int, t_mult: float = 1, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, int(t_0_batches)),
            T_mult=int(max(1, t_mult)),
            eta_min=eta_min,
        )

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.scheduler.step()
        else:
            self.scheduler.step(epoch)

    def get_last_lr(self) -> list[float]:
        return list(self.scheduler.get_last_lr())

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch

    @last_epoch.setter
    def last_epoch(self, value):
        self.scheduler.last_epoch = value


class LinearAnnealingScheduler:
    """Decay linearly from base_lr to eta_min and restart after t_0_batches.

    Multiply each cycle length by t_mult while retaining the cycle base LR.
    Expose the transition hooks expected by SequentialLR.
    """

    def __init__(self, optimizer: Optimizer, t_0_batches: int, t_mult: float = 1.0, eta_min: float = 0.0):
        self.optimizer = optimizer
        self.t_0 = max(1, int(t_0_batches))
        self.t_mult = float(max(1.0, t_mult))
        self.eta_min = float(eta_min)
        self.base_lrs = [group.get("initial_lr", group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = -1
        self.cycle = 0
        self.cycle_start = 0
        self.cycle_length = float(self.t_0)
        self._last_lr = list(self.base_lrs)

    def _current_lrs(self, epoch: int) -> list[float]:
        step = max(0, int(epoch))

        while step - self.cycle_start >= self.cycle_length:
            self.cycle += 1
            self.cycle_start += self.cycle_length
            self.cycle_length *= self.t_mult

        cycle_pos = step - self.cycle_start
        cycle_len = max(float(self.cycle_length), 1.0)
        frac = min(max(cycle_pos / cycle_len, 0.0), 1.0)
        return [base_lr - (base_lr - self.eta_min) * frac for base_lr in self.base_lrs]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is not None:
            self.last_epoch = int(epoch)
        else:
            self.last_epoch += 1

        lrs = self._current_lrs(self.last_epoch)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self._last_lr = list(lrs)

    def _update_lr(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.step()
        else:
            self.step(epoch)

    def _initial_step(self) -> None:
        self.last_epoch = -1
        self.step()

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict:
        return {
            "last_epoch": self.last_epoch,
            "cycle": self.cycle,
            "cycle_start": self.cycle_start,
            "cycle_length": self.cycle_length,
            "base_lrs": list(self.base_lrs),
            "_last_lr": list(self._last_lr),
            "t_0": self.t_0,
            "t_mult": self.t_mult,
            "eta_min": self.eta_min,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.last_epoch = int(state_dict.get("last_epoch", self.last_epoch))
        self.cycle = int(state_dict.get("cycle", self.cycle))
        self.cycle_start = int(state_dict.get("cycle_start", self.cycle_start))
        self.cycle_length = float(state_dict.get("cycle_length", self.cycle_length))
        self.base_lrs = list(state_dict.get("base_lrs", self.base_lrs))
        self._last_lr = list(state_dict.get("_last_lr", self._last_lr))
        self.t_0 = int(state_dict.get("t_0", self.t_0))
        self.t_mult = float(state_dict.get("t_mult", self.t_mult))
        self.eta_min = float(state_dict.get("eta_min", self.eta_min))
