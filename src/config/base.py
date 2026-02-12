from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InfrastructureSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    dino_weight_dir: Path
    dino_repo_dir: Path

    wells_data_dir: Path
    dvm_data_dir: Path

    num_workers: int = 4
    accelerator: Literal["auto", "cpu", "gpu", "mps"]
    devices: int | str = "auto"
    precision: Literal["64-true", "32-true", "16-mixed", "bf16-mixed"] = "32-true"
    fast_dev_run: bool = False

    wandb_project: str = "nldl-winter-school"
    wandb_entity: str | None = None
    wandb_save_dir: Path
    wandb_data_dir: Path | None = None
    wandb_cache_dir: Path | None = None


infra = InfrastructureSettings()  # type: ignore[call-arg]
