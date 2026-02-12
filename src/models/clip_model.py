from typing import NamedTuple, Protocol

import torch.nn.functional as F
from torch import Tensor, nn


class Encoder(Protocol):
    def __call__(self, x: Tensor) -> Tensor: ...


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        use_hidden: bool,
    ) -> None:
        super().__init__()
        if use_hidden:
            self.projection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class ClipOutput(NamedTuple):
    image_embeddings: Tensor
    tabular_embeddings: Tensor
    image_projections: Tensor
    tabular_projections: Tensor


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        tabular_encoder: nn.Module,
        image_embed_dim: int,
        tabular_embed_dim: int,
        projection_dim: int,
        projection_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder

        self.image_projection = ProjectionHead(
            image_embed_dim,
            projection_hidden_dim,
            projection_dim,
            use_hidden=True,
        )
        self.tabular_projection = ProjectionHead(
            tabular_embed_dim,
            projection_hidden_dim,
            projection_dim,
            use_hidden=False,
        )

    def encode_image(self, images: Tensor) -> Tensor:
        return self.image_encoder(images)

    def encode_tabular(self, tabular: Tensor) -> Tensor:
        return self.tabular_encoder(tabular)

    def project_image(self, embeddings: Tensor) -> Tensor:
        return F.normalize(self.image_projection(embeddings), dim=-1)

    def project_tabular(self, embeddings: Tensor) -> Tensor:
        return F.normalize(self.tabular_projection(embeddings), dim=-1)

    def forward(self, images: Tensor, tabular: Tensor) -> ClipOutput:
        image_embeddings = self.encode_image(images)
        tabular_embeddings = self.encode_tabular(tabular)

        image_projections = self.project_image(image_embeddings)
        tabular_projections = self.project_tabular(tabular_embeddings)

        return ClipOutput(
            image_embeddings=image_embeddings,
            tabular_embeddings=tabular_embeddings,
            image_projections=image_projections,
            tabular_projections=tabular_projections,
        )
