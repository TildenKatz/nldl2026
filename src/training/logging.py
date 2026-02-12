import os
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger

from src.config.base import infra
from src.config.experiments import (
    CLIPAlignmentConfig,
    ExperimentConfig,
    SupervisedExperimentConfig,
)
from src.config.models import DinoEncoderConfig, EncoderConfig


def _encoder_tags(encoder: EncoderConfig, *, frozen: bool) -> list[str]:
    tags = [encoder.type, encoder.variant]
    if isinstance(encoder, DinoEncoderConfig) and encoder.lora is not None:
        tags.append("lora")
    if frozen:
        tags.append("frozen")
    return tags


def _extract_tags(config: ExperimentConfig) -> list[str]:
    tags = [config.experiment_type, config.data.type]

    match config:
        case CLIPAlignmentConfig(encoder=encoder, freeze_encoder=is_frozen):
            tags.extend(_encoder_tags(encoder, frozen=is_frozen))
        case SupervisedExperimentConfig(model=model):
            tags.extend(_encoder_tags(model.encoder, frozen=model.freeze_encoder))

    return tags


def create_wandb_logger(
    config: ExperimentConfig,
    *,
    resume_id: str | None,
) -> WandbLogger:
    group = f"{config.experiment_type}/{config.data.type}"
    tags = _extract_tags(config)

    wandb_logger = WandbLogger(
        project=infra.wandb_project,
        entity=infra.wandb_entity,
        name=config.run_name or config.experiment_name,
        group=group,
        tags=tags,
        config=config.model_dump(),
        log_model="all",
        save_dir=str(infra.wandb_save_dir),
        id=resume_id,
        resume="must" if resume_id else "never",
    )

    _log_artifacts(wandb_logger)

    return wandb_logger


def _log_artifacts(wandb_logger: WandbLogger) -> None:
    uv_lock = Path("uv.lock")
    if uv_lock.exists():
        wandb_logger.experiment.save(str(uv_lock), policy="now")

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        slurm_script = os.environ.get("SLURM_JOB_NAME", "")
        if slurm_script:
            slurm_path = Path(slurm_script)
            if slurm_path.exists():
                wandb_logger.experiment.save(str(slurm_path), policy="now")

        wandb_logger.experiment.config.update(
            {
                "slurm/job_id": slurm_job_id,
                "slurm/array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm/array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "slurm/job_name": os.environ.get("SLURM_JOB_NAME"),
                "slurm/nodelist": os.environ.get("SLURM_NODELIST"),
                "slurm/num_gpus": os.environ.get("SLURM_GPUS_ON_NODE"),
            },
            allow_val_change=True,
        )
