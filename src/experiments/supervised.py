from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.plugins.environments import SLURMEnvironment

import wandb
from src.config.base import infra
from src.config.experiments import SupervisedExperimentConfig
from src.config.factory import create_image_model
from src.training.data_module import DataModule
from src.training.logging import create_wandb_logger
from src.training.supervised_module import SupervisedLightningModule


def run_supervised(
    config: SupervisedExperimentConfig,
    ckpt_path: Path | None,
    wandb_run_id: str | None,
) -> SupervisedLightningModule:
    print("Starting supervised experiment")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Config:\n{config.model_dump_json(indent=2)}")
    print(f"Settings:\n{infra.model_dump_json(indent=2)}")

    L.seed_everything(config.training.seed, workers=True)

    data = config.data.load(seed=config.training.seed)

    data = data._replace(
        train_image_transform=config.data.aug_transform,
        eval_image_transform=config.data.eval_transform,
    )

    print(f"Task: {data.task}, Output dim: {data.output_dim}")

    model = create_image_model(config.model, output_dim=data.output_dim)
    print(f"Created model: {model}")

    lightning_module = SupervisedLightningModule(
        model=model,
        config=config,
        task=data.task,
        num_classes=data.output_dim if data.task == "classification" else None,
        target_names=data.target_names,
    )

    data_module = DataModule.from_config(
        config=data,
        batch_size=config.training.batch_size,
        num_workers=infra.num_workers,
    )

    wandb_logger = create_wandb_logger(config, resume_id=wandb_run_id)

    ckpt_dir = infra.wandb_save_dir / "checkpoints" / wandb_logger.experiment.id

    monitor_metric = "val/acc" if data.task == "classification" else "val/loss"
    monitor_mode = "max" if data.task == "classification" else "min"

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor=monitor_metric,
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode=monitor_mode,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        default_root_dir=ckpt_dir,
        max_epochs=config.training.max_epochs,
        accelerator=infra.accelerator,
        devices=infra.devices,
        precision=infra.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        plugins=[SLURMEnvironment(auto_requeue=False)],
        deterministic=True,
        enable_progress_bar=True,
        fast_dev_run=infra.fast_dev_run,
    )

    print("Starting training...")
    trainer.fit(
        lightning_module,
        data_module,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    print("Running test evaluation...")
    trainer.test(lightning_module, data_module, ckpt_path=None)

    wandb.finish()

    print("Supervised experiment complete.")
    return lightning_module
