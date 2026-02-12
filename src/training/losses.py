import torch
import torch.nn.functional as F
from torch import Tensor


def clip_loss(
    image_projections: Tensor,
    tabular_projections: Tensor,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    logits_per_image = image_projections @ tabular_projections.T / temperature
    logits_per_tabular = logits_per_image.T

    batch_size = image_projections.size(0)
    labels = torch.arange(batch_size, device=image_projections.device)

    loss_i2t = F.cross_entropy(
        logits_per_image,
        labels,
        label_smoothing=label_smoothing,
    )
    loss_t2i = F.cross_entropy(
        logits_per_tabular,
        labels,
        label_smoothing=label_smoothing,
    )

    return (loss_i2t + loss_t2i) / 2, loss_i2t, loss_t2i


def info_nce_loss(
    anchor: Tensor,
    positive: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    logits = anchor @ positive.T / temperature
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return F.cross_entropy(logits, labels)
