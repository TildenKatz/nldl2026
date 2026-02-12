from typing import Annotated, Literal

import torch.nn.functional as F
from pydantic import Field
from torch import Tensor

from src.config.base import FrozenConfig


class MSELossConfig(FrozenConfig):
    type: Literal["mse"] = "mse"

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)


class MAELossConfig(FrozenConfig):
    type: Literal["mae"] = "mae"

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(pred, target)


class HuberLossConfig(FrozenConfig):
    type: Literal["huber"] = "huber"
    delta: float = 1.0

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class CrossEntropyLossConfig(FrozenConfig):
    type: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: float = 0.0

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            pred,
            target.long().squeeze(-1),
            label_smoothing=self.label_smoothing,
        )


LossConfig = Annotated[
    MSELossConfig | MAELossConfig | HuberLossConfig | CrossEntropyLossConfig,
    Field(discriminator="type"),
]
