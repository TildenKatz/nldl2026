from typing import Annotated, Literal

from pydantic import Field, model_validator

from src.config.base import FrozenConfig
from src.config.data import DataConfig, DVMDataConfig, WellsDataConfig
from src.config.losses import LossConfig
from src.config.models import EncoderConfig, ImageModelConfig
from src.config.optimizers import (
    OptimizerConfig,
    SchedulerConfig,
)


class TrainingConfig(FrozenConfig):
    batch_size: int
    max_epochs: int

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None

    early_stopping_patience: int
    early_stopping_min_delta: float

    seed: int


class SupervisedExperimentConfig(FrozenConfig):
    experiment_type: Literal["supervised"] = "supervised"
    experiment_name: str
    run_name: str | None = None

    data: DataConfig
    model: ImageModelConfig
    training: TrainingConfig
    loss: LossConfig

    @model_validator(mode="after")
    def validate_loss_matches_task(self) -> "SupervisedExperimentConfig":
        """Ensure loss function matches the task type."""
        if isinstance(self.data, WellsDataConfig) and self.loss.type == "cross_entropy":
            raise ValueError(
                "Cannot use cross_entropy loss with regression task (wells)",
            )

        if isinstance(self.data, DVMDataConfig) and self.loss.type in (
            "mse",
            "mae",
            "huber",
        ):
            raise ValueError(
                f"Cannot use {self.loss.type} loss with classification task (dvm)",
            )
        return self


class CLIPAlignmentConfig(FrozenConfig):
    experiment_type: Literal["clip_alignment"] = "clip_alignment"
    experiment_name: str
    run_name: str | None = None

    encoder: EncoderConfig
    freeze_encoder: bool

    tabular_hidden_dim: int
    tabular_output_dim: int

    projection_dim: int
    projection_hidden_dim: int

    temperature: float
    label_smoothing: float

    training: TrainingConfig

    data: DataConfig


ExperimentConfig = Annotated[
    SupervisedExperimentConfig | CLIPAlignmentConfig,
    Field(discriminator="experiment_type"),
]
