from functools import partial
from pathlib import Path

from pydantic import BaseModel
from torch import Tensor

from src.data.types import DatasetConfig, DatasetSetup


class RWIMetadata(BaseModel):
    xrd: list[str]
    xrf: list[str]
    xrf_means: dict[str, float]
    xrf_stds: dict[str, float]
    split_sizes: dict[str, int]
    split_wells: dict[str, list[str]]
    shard_counts: dict[str, int]


def _shard_pattern(n_shards: int) -> str:
    return f"{{0000..{n_shards - 1:04d}}}"


def get_rwi_config(
    features_dir: Path,
    xrd_columns: list[str] | None,
) -> "DatasetConfig":
    metadata = RWIMetadata.model_validate_json(
        (features_dir / "metadata.json").read_text(),
    )

    xrd_columns = xrd_columns or metadata.xrd

    xrd_idx = [metadata.xrd.index(col) for col in xrd_columns]

    def select_xrd(y: Tensor) -> Tensor:
        return y[xrd_idx] / 100.0

    base_dataset = partial(
        DatasetSetup,
        decode="torch",
        to_tuple=("patches.pth", "features.pth", "label.pth"),
    )

    return DatasetConfig(
        train=base_dataset(
            root=features_dir
            / f"train-{_shard_pattern(metadata.shard_counts['train'])}.tar",
            length=metadata.split_sizes["train"],
            shuffle=True,
        ),
        val=base_dataset(
            root=features_dir
            / f"val-{_shard_pattern(metadata.shard_counts['val'])}.tar",
            length=metadata.split_sizes["val"],
        ),
        test=base_dataset(
            root=features_dir
            / f"test-{_shard_pattern(metadata.shard_counts['test'])}.tar",
            length=metadata.split_sizes["test"],
        ),
        target_names=xrd_columns,
        feature_dim=len(metadata.xrf),
        output_dim=len(xrd_columns),
        task="regression",
        label_transform=select_xrd,
    )
