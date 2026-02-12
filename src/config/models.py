from typing import Annotated, Literal

import torch
from pydantic import Field, TypeAdapter, computed_field
from torch import nn
from torchvision import models

import wandb
from src.config.base import FrozenConfig, infra
from src.models.linear_classifier import LinearClassifier
from src.models.lora import create_lora_model, merge_and_reset_lora


class LoRAConfig(FrozenConfig):
    rank: int
    alpha: float | None = None
    dropout: float = 0.0
    target_modules: list[str] = Field(default=["qkv", "proj"])


class DinoEncoderConfig(FrozenConfig):
    """Configuration for DINOv3 vision transformer encoder."""

    type: Literal["dino"] = "dino"
    variant: Literal["vits16", "vits16plus", "vitb16", "vitl16"]
    lora: LoRAConfig | None

    @property
    def model_name(self) -> str:
        return f"dinov3_{self.variant}"

    @property
    def embed_dim(self) -> int:
        dims = {"vits16": 384, "vits16plus": 384, "vitb16": 768, "vitl16": 1024}
        return dims[self.variant]

    @property
    def weight_file(self) -> str:
        return f"dinov3_{self.variant}.pth"

    def create_encoder(self) -> nn.Module:
        weight_path = infra.dino_weight_dir / self.weight_file

        print(f"Loading DINOv3 weights from {weight_path}")

        model = torch.hub.load(
            infra.dino_repo_dir.as_posix(),
            self.model_name,
            source="local",
            weights=(weight_path).as_posix(),
        )
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model)}")

        if self.lora is None:
            return model

        return create_lora_model(
            model,
            rank=self.lora.rank,
            alpha=self.lora.alpha,
            dropout=self.lora.dropout,
        )


class ResNetEncoderConfig(FrozenConfig):
    """Configuration for ResNet encoder."""

    type: Literal["resnet"] = "resnet"
    variant: Literal["resnet18", "resnet34", "resnet50", "resnet101"]

    @property
    def embed_dim(self) -> int:
        dims = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}
        return dims[self.variant]

    def create_encoder(self) -> nn.Module:
        """Create ResNet encoder (without final FC layer)."""
        model = getattr(models, self.variant)(weights="DEFAULT")

        # Remove the final FC layer, return features
        model.fc = nn.Identity()

        return model


_LiteralConfig = Annotated[
    DinoEncoderConfig | ResNetEncoderConfig,
    Field(discriminator="type"),
]


class ArtifactConfig(FrozenConfig):
    type: Literal["wandb"] = "wandb"
    run_id: str
    lora: LoRAConfig | None = None

    def _full_run_path(self) -> str:
        if "/" in self.run_id:
            return self.run_id
        entity = infra.wandb_entity or wandb.Api().default_entity
        return f"{entity}/{infra.wandb_project}/{self.run_id}"

    @computed_field
    @property
    def _source_config(self) -> "_LiteralConfig":
        run = wandb.Api().run(self._full_run_path())
        return TypeAdapter(_LiteralConfig).validate_python(
            run.config["encoder"],
        )

    @property
    def embed_dim(self) -> int:
        return self._source_config.embed_dim

    @property
    def variant(self) -> str:
        return self._source_config.variant

    def _load_state(self, artifact: wandb.Artifact) -> dict:
        cache_dir = infra.wandb_save_dir / "artifacts" / artifact.name.replace(":", "_")
        cache_dir.mkdir(parents=True, exist_ok=True)

        artifact_dir = artifact.download(root=str(cache_dir))
        state = torch.load(
            f"{artifact_dir}/model.ckpt",
            map_location="cpu",
            weights_only=False,
        )

        return {
            k.removeprefix("model.image_encoder."): v
            for k, v in state["state_dict"].items()
            if k.startswith("model.image_encoder.")
        }

    def create_encoder(self) -> nn.Module:
        run = wandb.Api().run(self._full_run_path())

        try:
            artifact = next(
                a
                for a in run.logged_artifacts()
                if a.type == "model" and "best" in a.aliases
            )
        except StopIteration:
            raise ValueError(
                f"No 'best' model artifact found in run {self.run_id}",
            ) from None

        model = self._source_config.create_encoder()
        state = self._load_state(artifact)

        mismatch = model.load_state_dict(state, strict=True)

        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise ValueError(
                f"State dict keys mismatch: missing {mismatch.missing_keys}, "
                f"unexpected {mismatch.unexpected_keys}",
            )

        if self.lora is not None:
            if not isinstance(self._source_config, DinoEncoderConfig):
                raise ValueError("LoRA can only be added to DINO encoders")
            if self._source_config.lora is not None:
                merge_and_reset_lora(model, self.lora.rank, self.lora.alpha)
            else:
                model = create_lora_model(
                    model,
                    rank=self.lora.rank,
                    alpha=self.lora.alpha,
                    dropout=self.lora.dropout,
                )

        return model


EncoderConfig = Annotated[
    DinoEncoderConfig | ResNetEncoderConfig | ArtifactConfig,
    Field(discriminator="type"),
]


class LinearHeadConfig(FrozenConfig):
    hidden_dim: int | None
    dropout: float = 0.0
    bounded_outputs: bool

    def create_head(self, input_dim: int, output_dim: int) -> nn.Module:
        return LinearClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self.hidden_dim,
            bounded_outputs=self.bounded_outputs,
            dropout=self.dropout,
        )


class ImageModelConfig(FrozenConfig):
    encoder: EncoderConfig
    head: LinearHeadConfig
    freeze_encoder: bool
