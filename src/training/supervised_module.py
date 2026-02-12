import io
from typing import Literal

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from PIL import Image
from torch import Tensor
from torchmetrics.classification import MulticlassAccuracy

import wandb
from src.config.experiments import SupervisedExperimentConfig
from src.models.linear_classifier import ImagePredictionModel

Task = Literal["regression", "classification"]

_EPS = 1e-8
_R2_MIN_VARIANCE = 1e-6


def compute_regression_metrics(
    predictions: Tensor,
    targets: Tensor,
) -> dict[str, Tensor]:
    mae = F.l1_loss(predictions, targets)
    mse = F.mse_loss(predictions, targets)
    rmse = mse.sqrt()

    rss = ((targets - predictions) ** 2).sum()
    tss = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - rss / (tss + _EPS)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def compute_per_target_metrics(
    predictions: Tensor,
    targets: Tensor,
    target_names: list[str],
) -> dict[str, Tensor]:
    metrics = {}
    for i, name in enumerate(target_names):
        col_pred = predictions[:, i]
        col_target = targets[:, i]

        col_mse = F.mse_loss(col_pred, col_target)
        col_rmse = col_mse.sqrt()

        col_rss = ((col_target - col_pred) ** 2).sum()
        col_tss = ((col_target - col_target.mean()) ** 2).sum()

        if col_tss < _R2_MIN_VARIANCE:
            col_r2 = torch.tensor(0.0, device=col_pred.device)
        else:
            col_r2 = 1 - col_rss / col_tss

        # Use shorter name for cleaner logging
        short_name = name.replace("XRD_WP_", "")
        metrics[f"target/{short_name}/rmse"] = col_rmse
        metrics[f"target/{short_name}/r2"] = col_r2

    return metrics


class SupervisedLightningModule(L.LightningModule):
    def __init__(
        self,
        model: ImagePredictionModel,
        config: SupervisedExperimentConfig,
        task: Task,
        num_classes: int | None = None,
        target_names: list[str] | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self.task = task
        self.target_names = target_names or []
        self.class_names = class_names or []
        self._batch_size = config.training.batch_size
        self.save_hyperparameters(ignore=["model", "config"])
        self.save_hyperparameters(config.model_dump())

        self._loss_fn = config.loss

        if task == "classification" and num_classes:
            self.train_acc = MulticlassAccuracy(num_classes=num_classes)
            self.val_acc = MulticlassAccuracy(num_classes=num_classes)
            self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        self._val_predictions: list[Tensor] = []
        self._val_targets: list[Tensor] = []
        self._test_predictions: list[Tensor] = []
        self._test_targets: list[Tensor] = []
        self._val_sample_images: list[Tensor] = []
        self._val_sample_preds: list[Tensor] = []
        self._val_sample_targets: list[Tensor] = []

    @property
    def samples_seen(self) -> int:
        return self.global_step * self._batch_size

    def on_fit_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric("samples_seen")
            self.logger.experiment.define_metric("*", step_metric="samples_seen")

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def _compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self._loss_fn(predictions, targets)

    def _shared_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
    ) -> Tensor:
        images, _, targets = batch
        predictions = self(images)

        loss = self._compute_loss(predictions, targets)
        self.log("samples_seen", float(self.samples_seen), sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        with torch.no_grad():
            if self.task == "classification":
                preds = predictions.argmax(dim=-1)
                targets_int = targets.long().squeeze(-1)

                acc_metric = getattr(self, f"{stage}_acc", None)
                if acc_metric:
                    acc_metric(preds, targets_int)
                    self.log(f"{stage}/acc", acc_metric, sync_dist=True, prog_bar=True)

                if stage == "val":
                    self._val_predictions.append(preds.detach().cpu())
                    self._val_targets.append(targets_int.detach().cpu())
                    self._reservoir_sample(images, preds, targets_int, batch_idx)
            else:
                metrics = compute_regression_metrics(predictions, targets)
                for name, value in metrics.items():
                    self.log(f"{stage}/{name}", value, sync_dist=True)

                if stage == "val":
                    self._val_predictions.append(predictions.detach().cpu())
                    self._val_targets.append(targets.detach().cpu())
                    self._reservoir_sample(images, predictions, targets, batch_idx)
                elif stage == "test":
                    self._test_predictions.append(predictions.detach().cpu())
                    self._test_targets.append(targets.detach().cpu())

        return loss

    def _reservoir_sample(
        self,
        images: Tensor,
        predictions: Tensor,
        targets: Tensor,
        batch_idx: int,
        k: int = 8,
    ) -> None:
        batch_size = images.size(0)
        n_seen = batch_idx * batch_size

        for i in range(batch_size):
            n_seen += 1
            if len(self._val_sample_images) < k:
                self._val_sample_images.append(images[i].detach().cpu())
                self._val_sample_preds.append(predictions[i].detach().cpu())
                self._val_sample_targets.append(targets[i].detach().cpu())
            else:
                j = int(torch.randint(0, n_seen, (1,)).item())
                if j < k:
                    self._val_sample_images[j] = images[i].detach().cpu()
                    self._val_sample_preds[j] = predictions[i].detach().cpu()
                    self._val_sample_targets[j] = targets[i].detach().cpu()

    def _compute_epoch_end_metrics(self, stage: str) -> None:
        if stage == "val":
            preds_list, targets_list = self._val_predictions, self._val_targets
        elif stage == "test":
            preds_list, targets_list = self._test_predictions, self._test_targets
        else:
            return

        if not preds_list:
            return

        all_preds = torch.cat(preds_list, dim=0)
        all_targets = torch.cat(targets_list, dim=0)

        if self.task == "regression" and self.target_names:
            per_target = compute_per_target_metrics(
                all_preds,
                all_targets,
                self.target_names,
            )
            for name, value in per_target.items():
                self.log(f"{stage}/{name}", value, sync_dist=True)

            self._log_prediction_scatter(all_preds, all_targets, stage)
        elif self.task == "classification":
            self._log_confusion_matrix(all_preds, all_targets, stage)

        if stage == "val" and self._val_sample_images:
            sample_images = torch.stack(self._val_sample_images)
            sample_preds = torch.stack(self._val_sample_preds)
            sample_targets = torch.stack(self._val_sample_targets)
            self._log_sample_predictions(
                sample_images,
                sample_preds,
                sample_targets,
                stage,
            )

        if stage == "val":
            self._val_predictions.clear()
            self._val_targets.clear()
            self._val_sample_images.clear()
            self._val_sample_preds.clear()
            self._val_sample_targets.clear()
        elif stage == "test":
            self._test_predictions.clear()
            self._test_targets.clear()

    def _log_confusion_matrix(
        self,
        all_preds: Tensor,
        all_targets: Tensor,
        stage: str,
    ) -> None:
        logger = self.logger
        if not isinstance(logger, WandbLogger):
            return

        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        num_classes = int(max(preds_np.max(), targets_np.max())) + 1

        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, target in zip(preds_np, targets_np, strict=True):
            conf_matrix[int(target), int(pred)] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(conf_matrix, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.colorbar(im, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close(fig)

        logger.experiment.log(
            {
                f"{stage}/confusion_matrix": wandb.Image(Image.open(buf)),
                "samples_seen": self.samples_seen,
            },
            commit=False,
        )

    def _log_prediction_scatter(
        self,
        all_preds: Tensor,
        all_targets: Tensor,
        stage: str,
    ) -> None:
        logger = self.logger
        if not isinstance(logger, WandbLogger):
            return

        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        num_targets = preds_np.shape[1] if preds_np.ndim > 1 else 1

        cols = min(3, num_targets)
        rows = (num_targets + cols - 1) // cols
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(6 * cols, 6 * rows),
            squeeze=False,
        )

        for i in range(num_targets):
            ax = axes[i // cols, i % cols]
            pred_col = preds_np[:, i] if preds_np.ndim > 1 else preds_np
            target_col = targets_np[:, i] if targets_np.ndim > 1 else targets_np

            ax.scatter(target_col, pred_col, alpha=0.3, s=10)

            min_val = min(target_col.min(), pred_col.min())
            max_val = max(target_col.max(), pred_col.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)

            name = self.target_names[i] if i < len(self.target_names) else f"Target {i}"
            short_name = name.replace("XRD_WP_", "")
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title(short_name)

        for i in range(num_targets, rows * cols):
            axes[i // cols, i % cols].set_visible(False)

        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        plt.close(fig)

        logger.experiment.log(
            {
                f"{stage}/prediction_scatter": wandb.Image(Image.open(buf)),
                "samples_seen": self.samples_seen,
            },
            commit=False,
        )

    def _log_sample_predictions(
        self,
        images: Tensor,
        predictions: Tensor,
        targets: Tensor,
        stage: str,
    ) -> None:
        logger = self.logger
        if not isinstance(logger, WandbLogger):
            return

        images_np = images.numpy()
        preds_np = predictions.numpy()
        targets_np = targets.numpy()

        def format_class(idx: int) -> str:
            if self.class_names and idx < len(self.class_names):
                return self.class_names[idx]
            return str(idx)

        data = []
        for i in range(min(8, len(images_np))):
            img = images_np[i]
            if img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            img = (img * 255).astype(np.uint8)

            if self.task == "classification":
                pred_str = format_class(int(preds_np[i]))
                target_str = format_class(int(targets_np[i]))
            else:
                pred_str = ", ".join(f"{v:.2f}" for v in preds_np[i])
                target_str = ", ".join(f"{v:.2f}" for v in targets_np[i])

            data.append([wandb.Image(img), pred_str, target_str])

        table = wandb.Table(
            columns=["Image", "Prediction", "Ground Truth"],
            data=data,
        )
        logger.experiment.log(
            {
                f"{stage}/sample_predictions": table,
                "samples_seen": self.samples_seen,
            },
            commit=False,
        )

    def training_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def on_validation_epoch_end(self) -> None:
        self._compute_epoch_end_metrics("val")

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def on_test_epoch_end(self) -> None:
        self._compute_epoch_end_metrics("test")

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        params = (p for p in self.parameters() if p.requires_grad)

        optimizer = self.config.training.optimizer.create_optimizer(params)

        if self.config.training.scheduler is None:
            return {"optimizer": optimizer}  # type: ignore[return-value]

        scheduler = self.config.training.scheduler.create_scheduler(
            optimizer,
            self.config.training.max_epochs,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
