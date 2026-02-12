import torch
from torch import Tensor


class SCARFAugmentation:
    def __init__(
        self,
        corruption_rate: float,
        feature_values: Tensor,
    ) -> None:
        self.corruption_rate = corruption_rate
        self.feature_values = feature_values

    @classmethod
    def from_samples(
        cls,
        features: Tensor,
        corruption_rate: float,
    ) -> "SCARFAugmentation":
        return cls(corruption_rate=corruption_rate, feature_values=features)

    def __call__(self, x: Tensor) -> Tensor:
        if self.corruption_rate <= 0:
            return x

        mask = torch.rand(x.shape[-1]) < self.corruption_rate

        random_idx = torch.randint(0, len(self.feature_values), ())
        random_values = self.feature_values[random_idx]

        return torch.where(mask, random_values, x)
