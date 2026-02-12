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
from src.config.experiments import CLIPAlignmentConfig
from src.data.types import ComposeTransform, OneHotEncoder
from src.models.clip_model import CLIPModel
from src.models.tabular_encoder import TabularEncoder
from src.training.clip_module import CLIPLightningModule
from src.training.contrastive_augmentations import SCARFAugmentation
from src.training.data_module import DataModule
from src.training.logging import create_wandb_logger


def run_clip(
    config: CLIPAlignmentConfig,
    ckpt_path: Path | None,
    wandb_run_id: str | None,
) -> CLIPLightningModule:
    print("Starting CLIP pre-training experiment")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Config:\n{config.model_dump_json(indent=2)}")
    print(f"Settings:\n{infra.model_dump_json(indent=2)}")

    L.seed_everything(config.training.seed, workers=True)

    data = config.data.load(seed=config.training.seed)

    print(
        f"Data split: {data.train.length} train, "
        f"{data.val.length} val, {data.test.length} test",
    )

    print("Extracting training features for SCARF augmentation...")
    train_features = data.train.extract_features()
    scarf_augmentation = SCARFAugmentation.from_samples(
        features=train_features,
        corruption_rate=0.3,
    )

    if data.field_lengths is not None:
        one_hot = OneHotEncoder(data.field_lengths)
        train_tabular_transform = ComposeTransform(scarf_augmentation, one_hot)
        eval_tabular_transform = one_hot
    else:
        train_tabular_transform = scarf_augmentation
        eval_tabular_transform = data.eval_tabular_transform

    data = data._replace(
        train_image_transform=config.data.aug_transform,
        eval_image_transform=config.data.eval_transform,
        train_tabular_transform=train_tabular_transform,
        eval_tabular_transform=eval_tabular_transform,
    )

    image_encoder = config.encoder.create_encoder()

    if config.freeze_encoder:
        for param in image_encoder.parameters():
            param.requires_grad = False

    tabular_encoder = TabularEncoder(
        input_dim=data.feature_dim,
        hidden_dim=config.tabular_hidden_dim,
        output_dim=config.tabular_output_dim,
    )

    clip_model = CLIPModel(
        image_encoder=image_encoder,
        tabular_encoder=tabular_encoder,
        image_embed_dim=config.encoder.embed_dim,
        tabular_embed_dim=config.tabular_output_dim,
        projection_dim=config.projection_dim,
        projection_hidden_dim=config.projection_hidden_dim,
    )

    lightning_module = CLIPLightningModule(
        model=clip_model,
        config=config,
    )

    data_module = DataModule.from_config(
        config=data,
        batch_size=config.training.batch_size,
        num_workers=infra.num_workers,
    )

    wandb_logger = create_wandb_logger(config, resume_id=wandb_run_id)

    ckpt_dir = infra.wandb_save_dir / "checkpoints" / wandb_logger.experiment.id

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/loss",
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
            mode="min",
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

    print("Starting CLIP pre-training...")
    trainer.fit(
        lightning_module,
        data_module,
        ckpt_path=ckpt_path,
        weights_only=False,
    )

    print("Running test evaluation...")
    trainer.test(lightning_module, data_module, ckpt_path=None)

    best_checkpoint = checkpoint_callback.best_model_path
    print(f"Best checkpoint saved at: {best_checkpoint}")

    wandb.finish()

    print("CLIP pre-training complete.")
    return lightning_module
