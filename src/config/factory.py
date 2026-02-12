from src.config.models import ImageModelConfig
from src.models.linear_classifier import ImagePredictionModel


def create_image_model(
    config: ImageModelConfig,
    output_dim: int,
) -> ImagePredictionModel:
    encoder = config.encoder.create_encoder()

    if config.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    head = config.head.create_head(
        input_dim=config.encoder.embed_dim,
        output_dim=output_dim,
    )

    return ImagePredictionModel(
        encoder=encoder,
        head=head,
    )
