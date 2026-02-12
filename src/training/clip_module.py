import io

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from PIL import Image
from torch import Tensor

import wandb
from src.config.experiments import CLIPAlignmentConfig
from src.models.clip_model import CLIPModel
from src.training.losses import clip_loss


class CLIPLightningModule(L.LightningModule):
    def __init__(
        self,
        model: CLIPModel,
        config: CLIPAlignmentConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.config = config
        self._batch_size = config.training.batch_size
        self.save_hyperparameters(ignore=["model"])
        self.save_hyperparameters(config.model_dump())

    @property
    def samples_seen(self) -> int:
        return self.global_step * self._batch_size

    def on_fit_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.define_metric("samples_seen")
            self.logger.experiment.define_metric("*", step_metric="samples_seen")

    def forward(
        self,
        images: Tensor,
        tabular: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.model(images, tabular)

    def _shared_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        stage: str,
        batch_idx: int,
    ) -> Tensor:
        images, tabular, _ = batch
        _, _, image_proj, tabular_proj = self(images, tabular)

        loss, loss_i2t, loss_t2i = clip_loss(
            image_proj,
            tabular_proj,
            temperature=self.config.temperature,
            label_smoothing=self.config.label_smoothing,
        )

        self.log("samples_seen", float(self.samples_seen), sync_dist=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/loss_i2t", loss_i2t, sync_dist=True)
        self.log(f"{stage}/loss_t2i", loss_t2i, sync_dist=True)

        with torch.no_grad():
            self._log_retrieval_metrics(image_proj, tabular_proj, stage)
            self._log_similarity_matrix(image_proj, tabular_proj, stage, batch_idx)
            self._log_embedding_stats(image_proj, tabular_proj, stage)

        return loss

    def _log_retrieval_metrics(
        self,
        image_proj: Tensor,
        tabular_proj: Tensor,
        stage: str,
    ) -> None:
        batch_size = image_proj.size(0)
        similarity = image_proj @ tabular_proj.T

        i2t_ranks = (
            (
                similarity.argsort(dim=1, descending=True)
                == torch.arange(batch_size, device=self.device).unsqueeze(1)
            )
            .nonzero()[:, 1]
            .float()
        )
        t2i_ranks = (
            (
                similarity.T.argsort(dim=1, descending=True)
                == torch.arange(batch_size, device=self.device).unsqueeze(1)
            )
            .nonzero()[:, 1]
            .float()
        )

        for k in [1, 5, 10]:
            if batch_size >= k:
                self.log(
                    f"{stage}/i2t_R@{k}",
                    (i2t_ranks < k).float().mean(),
                    sync_dist=True,
                )
                self.log(
                    f"{stage}/t2i_R@{k}",
                    (t2i_ranks < k).float().mean(),
                    sync_dist=True,
                )

        i2t_mrr = (1.0 / (i2t_ranks + 1)).mean()
        t2i_mrr = (1.0 / (t2i_ranks + 1)).mean()
        self.log(f"{stage}/i2t_mrr", i2t_mrr, sync_dist=True)
        self.log(f"{stage}/t2i_mrr", t2i_mrr, sync_dist=True)

    def _log_similarity_matrix(
        self,
        image_proj: Tensor,
        tabular_proj: Tensor,
        stage: str,
        batch_idx: int,
    ) -> None:
        if stage != "val" or batch_idx % 50 != 0:
            return
        logger = self.logger
        if not isinstance(logger, WandbLogger):
            return

        similarity = (image_proj @ tabular_proj.T).cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(similarity, cmap="viridis", aspect="auto")
        ax.set_xlabel("Tabular")
        ax.set_ylabel("Image")
        ax.set_title("Image-Tabular Similarity Matrix")
        fig.colorbar(im, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close(fig)

        logger.experiment.log(
            {
                "val/similarity_matrix": wandb.Image(Image.open(buf)),
                "samples_seen": self.samples_seen,
            },
            commit=False,
        )

    def _log_embedding_stats(
        self,
        image_proj: Tensor,
        tabular_proj: Tensor,
        stage: str,
    ) -> None:
        logger = self.logger
        if not isinstance(logger, WandbLogger):
            return

        image_norms = image_proj.norm(dim=1).cpu().numpy()
        tabular_norms = tabular_proj.norm(dim=1).cpu().numpy()

        metrics: dict = {
            f"{stage}/image_proj_norm_mean": float(image_norms.mean()),
            f"{stage}/image_proj_norm_std": float(image_norms.std()),
            f"{stage}/tabular_proj_norm_mean": float(tabular_norms.mean()),
            f"{stage}/tabular_proj_norm_std": float(tabular_norms.std()),
        }

        try:
            metrics[f"{stage}/image_proj_norms"] = wandb.Histogram(
                image_norms.tolist(),
            )
            metrics[f"{stage}/tabular_proj_norms"] = wandb.Histogram(
                tabular_norms.tolist(),
            )
        except ValueError:
            pass

        metrics["samples_seen"] = self.samples_seen
        logger.experiment.log(metrics, commit=False)

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

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        return self._shared_step(batch, "test", batch_idx)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        params = filter(lambda p: p.requires_grad, self.parameters())

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
