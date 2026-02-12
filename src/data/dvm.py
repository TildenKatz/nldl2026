from functools import partial
from pathlib import Path

from pydantic import BaseModel

from src.data.types import DatasetConfig, DatasetSetup


class DVMMetadata(BaseModel):
    n_classes: int
    n_continuous: int
    n_categorical: int
    continuous_cols: list[str]
    categorical_cols: list[str]
    cat_sizes: dict[str, int]
    split_sizes: dict[str, int]
    train_means: dict[str, float]
    train_stds: dict[str, float]
    cat_mappings: dict[str, dict[str, int]]
    label_to_model: dict[int, str]
    label_to_id: dict[int, str]
    shard_counts: dict[str, int]


def _shard_pattern(n_shards: int) -> str:
    return f"{{0000..{n_shards - 1:04d}}}"


def get_dvm_config(features_dir: Path) -> "DatasetConfig":
    metadata = DVMMetadata.model_validate_json(
        (features_dir / "metadata.json").read_text(),
    )

    # nb: assumes ordering
    field_lengths = (
        *(1 for _ in metadata.continuous_cols),
        *metadata.cat_sizes.values(),
    )

    base_dataset = partial(
        DatasetSetup,
        decode="pil",
        to_tuple=("image.jpg", "features.pth", "label.pth"),
    )

    return DatasetConfig(
        train=base_dataset(
            root=features_dir
            / f"dvm_fronts_train-{_shard_pattern(metadata.shard_counts['train'])}.tar",
            length=metadata.split_sizes["train"],
            shuffle=True,
        ),
        val=base_dataset(
            root=features_dir
            / f"dvm_fronts_val-{_shard_pattern(metadata.shard_counts['val'])}.tar",
            length=metadata.split_sizes["val"],
        ),
        test=base_dataset(
            root=features_dir
            / f"dvm_fronts_test-{_shard_pattern(metadata.shard_counts['test'])}.tar",
            length=metadata.split_sizes["test"],
        ),
        target_names=list(metadata.label_to_model.values()),
        field_lengths=field_lengths,
        feature_dim=sum(field_lengths),
        output_dim=metadata.n_classes,
        task="classification",
    )
