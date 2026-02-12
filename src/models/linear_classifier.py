from torch import Tensor, nn


class LinearClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None,
        dropout: float,
        bounded_outputs: bool,
    ) -> None:
        super().__init__()

        if bounded_outputs:
            final_layer = nn.Sigmoid()
        else:
            final_layer = nn.Identity()

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                final_layer,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, output_dim),
                final_layer,
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class ImagePredictionModel(nn.Module):
    """Image encoder + head for prediction tasks."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, images: Tensor) -> Tensor:
        features = self.encoder(images)
        return self.head(features)
