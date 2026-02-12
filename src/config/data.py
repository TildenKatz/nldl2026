from typing import Annotated, Literal

import numpy as np
import torch
from PIL import Image
from pydantic import Field
from torch import Tensor
from torchvision.transforms import v2

from src.config.base import FrozenConfig, infra
from src.data.dvm import get_dvm_config
from src.data.types import DatasetConfig, Transform
from src.data.wells import get_rwi_config


class RandomPatchSelect:
    def __call__(self, patches: Tensor) -> Image.Image:
        idx = int(torch.randint(len(patches), ()))
        return Image.fromarray(np.array(patches[idx], dtype=np.uint8))


class FixedPatchSelect:
    def __call__(self, patches: Tensor) -> Image.Image:
        return Image.fromarray(np.array(patches[0], dtype=np.uint8))


class WellsDataConfig(FrozenConfig):
    type: Literal["wells"] = "wells"
    task: Literal["regression"] = "regression"

    xrd_columns: list[str] | None = Field(
        default=[
            "XRD_WP_QZ",
            "XRD_WP_TSILI",
            "XRD_WP_KFS",
            "XRD_WP_PL",
            "XRD_WP_ZEO",
            "XRD_WP_TCB",
            "XRD_WP_CAL",
            "XRD_WP_DOL",
            "XRD_WP_SD",
            "XRD_WP_PY",
            "XRD_WP_GP",
            "XRD_WP_JRS",
            "XRD_WP_TCLAY",
            "XRD_WP_AMRPH",
            "XRD_WP_OTHHLD",
            "XRD_WP_TOTHER",
        ],
    )

    def load(self, seed: int) -> "DatasetConfig":  # noqa: ARG002
        return get_rwi_config(
            features_dir=infra.wells_data_dir,
            xrd_columns=self.xrd_columns,
        )

    @property
    def eval_transform(self) -> Transform:
        return v2.Compose(
            [
                FixedPatchSelect(),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    @property
    def aug_transform(self) -> Transform:
        return v2.Compose(
            [
                RandomPatchSelect(),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )


class DVMDataConfig(FrozenConfig):
    type: Literal["dvm"] = "dvm"
    task: Literal["classification"] = "classification"

    def load(self, seed: int) -> "DatasetConfig":  # noqa: ARG002
        return get_dvm_config(features_dir=infra.dvm_data_dir)

    @property
    def eval_transform(self) -> Transform:
        return v2.Compose(
            [
                v2.Resize((128, 128)),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

    @property
    def aug_transform(self) -> Transform:
        return v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=128,
                    scale=(0.08, 1.0),
                ),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(
                    brightness=0.8,
                    contrast=0.8,
                    saturation=0.8,
                    hue=0.0,
                ),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(
                    kernel_size=29,
                    sigma=(0.1, 2.0),
                ),
                v2.ToTensor(),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )


DataConfig = Annotated[
    WellsDataConfig | DVMDataConfig,
    Field(discriminator="type"),
]
