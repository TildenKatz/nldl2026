from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import Field
from torch import Tensor
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
)

from src.config.base import FrozenConfig

type Params = Iterable[Tensor] | Iterable[dict[str, Tensor]]


class AdamConfig(FrozenConfig):
    type: Literal["adam"] = "adam"
    lr: float
    weight_decay: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def create_optimizer(self, params: Params) -> Optimizer:
        return Adam(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )


class AdamWConfig(FrozenConfig):
    type: Literal["adamw"] = "adamw"
    lr: float
    weight_decay: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def create_optimizer(self, params: Params) -> Optimizer:
        return AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
        )


OptimizerConfig = Annotated[
    AdamConfig | AdamWConfig,
    Field(discriminator="type"),
]


class CosineSchedulerConfig(FrozenConfig):
    type: Literal["cosine"] = "cosine"
    warmup_epochs: int
    warmup_start_factor: float
    eta_min: float = 0.0

    def create_scheduler(self, optimizer: Optimizer, max_epochs: int) -> LRScheduler:
        if self.warmup_epochs <= 0:
            return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=self.eta_min)

        warmup = LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - self.warmup_epochs,
            eta_min=self.eta_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs],
        )


class StepSchedulerConfig(FrozenConfig):
    type: Literal["step"] = "step"
    step_size: int
    gamma: float = 0.1

    def create_scheduler(
        self,
        optimizer: Optimizer,
        max_epochs: int,  # noqa: ARG002
    ) -> LRScheduler:
        return StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)


SchedulerConfig = Annotated[
    CosineSchedulerConfig | StepSchedulerConfig,
    Field(discriminator="type"),
]
