from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, NamedTuple
from warnings import filterwarnings

import torch
import webdataset as wds
from torch import Tensor
from tqdm import tqdm

filterwarnings("ignore", message=r"\.with_length\(\) only sets", module="webdataset")

type Transform = Callable[[Any], Tensor]


def identity(x: Tensor) -> Tensor:
    return x


class DataBatch(NamedTuple):
    images: Tensor  # (B, C, H, W)
    features: Tensor  # (B, feature_dim) - xrf, car specs, etc.
    targets: Tensor  # (B, target_dim) - xrd values, class labels, etc.


class DatasetSetup(NamedTuple):
    root: Path
    decode: str | None = None
    to_tuple: tuple[str, ...] | None = None
    length: int | None = None
    shuffle: bool = False

    def extract_features(self) -> Tensor:
        ds = wds.WebDataset(  # type: ignore
            self.root.as_posix(),
            shardshuffle=False,
        )
        if self.decode:
            ds = ds.decode(self.decode)
        if self.to_tuple:
            ds = ds.to_tuple(*self.to_tuple)

        return torch.stack(
            [
                feat
                for _, feat, _ in tqdm(
                    ds,
                    desc="Extracting features",
                    total=self.length,
                )
            ],
        )

    def build(
        self,
        *,
        batch_size: int,
        num_workers: int,
        transforms: tuple[Transform, Transform, Transform] | None,
    ) -> wds.WebLoader:  # type: ignore[name-defined]
        ds = wds.WebDataset(  # type: ignore[attr-defined]
            self.root.as_posix(),
            shardshuffle=1000 if self.shuffle else False,
            nodesplitter=wds.split_by_node if self.shuffle else None,  # type: ignore[attr-defined]
            workersplitter=wds.split_by_worker if self.shuffle else None,  # type: ignore[attr-defined]
        )

        if self.decode:
            ds = ds.decode(self.decode)
        if self.to_tuple:
            ds = ds.to_tuple(*self.to_tuple)
        if transforms is not None:
            ds = ds.map_tuple(*transforms)

        if self.shuffle:
            ds = ds.shuffle(1000)

        ds = ds.batched(batch_size, partial=not self.shuffle)

        loader = wds.WebLoader(  # type: ignore[attr-defined]
            ds,
            batch_size=None,
            num_workers=0,
            pin_memory=True,
        )

        if self.length:
            if self.shuffle:
                loader = loader.with_length(self.length // batch_size)
            else:
                loader = loader.with_length((self.length + batch_size - 1) // batch_size)

        return loader


class OneHotEncoder:
    def __init__(self, field_lengths: tuple[int, ...]) -> None:
        lengths = torch.tensor(field_lengths)
        offsets = lengths.cumsum(0) - lengths

        continuous = lengths == 1
        self.continuous_src = continuous.nonzero().squeeze(-1)
        self.continuous_dst = offsets[continuous]

        categorical = ~continuous
        self.categorical_src = categorical.nonzero().squeeze(-1)
        self.categorical_offsets = offsets[categorical]

        self.output_dim = int(lengths.sum())

    def __call__(self, x: Tensor) -> Tensor:
        out = x.new_zeros(*x.shape[:-1], self.output_dim)
        out[..., self.continuous_dst] = x[..., self.continuous_src]

        cat_vals = x[..., self.categorical_src].long()
        valid = cat_vals >= 0
        indices = self.categorical_offsets + cat_vals.clamp(min=0)
        out.scatter_(-1, indices, valid.float())
        return out


class ComposeTransform:
    def __init__(self, first: Transform, second: Transform) -> None:
        self.first = first
        self.second = second

    def __call__(self, x: Tensor) -> Tensor:
        return self.second(self.first(x))


class DatasetConfig(NamedTuple):
    task: Literal["regression", "classification"]
    feature_dim: int
    output_dim: int
    target_names: list[str]

    train: DatasetSetup
    val: DatasetSetup
    test: DatasetSetup

    field_lengths: tuple[int, ...] | None = None

    train_image_transform: Transform = identity
    eval_image_transform: Transform = identity
    train_tabular_transform: Transform = identity
    eval_tabular_transform: Transform = identity
    label_transform: Transform = identity
