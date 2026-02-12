import lightning as L
from torch.utils.data import DataLoader

from src.data.types import DataBatch, DatasetConfig, DatasetSetup, Transform


class DataModule(L.LightningDataModule):

    def __init__(
        self,
        train_dataset: DatasetSetup,
        val_dataset: DatasetSetup,
        test_dataset: DatasetSetup,
        batch_size: int,
        num_workers: int,
        train_image_transform: Transform,
        eval_image_transform: Transform,
        train_tabular_transform: Transform,
        eval_tabular_transform: Transform,
        label_transform: Transform,
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_image_transform = train_image_transform
        self.eval_image_transform = eval_image_transform
        self.train_tabular_transform = train_tabular_transform
        self.eval_tabular_transform = eval_tabular_transform
        self.label_transform = label_transform

    @classmethod
    def from_config(
        cls,
        config: DatasetConfig,
        batch_size: int,
        num_workers: int,
    ) -> "DataModule":
        return cls(
            train_dataset=config.train,
            val_dataset=config.val,
            test_dataset=config.test,
            batch_size=batch_size,
            num_workers=num_workers,
            train_image_transform=config.train_image_transform,
            eval_image_transform=config.eval_image_transform,
            train_tabular_transform=config.train_tabular_transform,
            eval_tabular_transform=config.eval_tabular_transform,
            label_transform=config.label_transform,
        )

    def train_dataloader(self) -> DataLoader[DataBatch]:
        return self.train_dataset.build(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transforms=(
                self.train_image_transform,
                self.train_tabular_transform,
                self.label_transform,
            ),
        )

    def val_dataloader(self) -> DataLoader[DataBatch]:
        return self.val_dataset.build(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transforms=(
                self.eval_image_transform,
                self.eval_tabular_transform,
                self.label_transform,
            ),
        )

    def test_dataloader(self) -> DataLoader[DataBatch]:
        return self.test_dataset.build(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            transforms=(
                self.eval_image_transform,
                self.eval_tabular_transform,
                self.label_transform,
            ),
        )
