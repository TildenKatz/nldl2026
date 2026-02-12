import math

import torch
from torch import nn


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.in_features = in_features
        self.out_features = out_features

        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.register_buffer("W0", None)

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.W0 is not None:
            result = nn.functional.linear(x, self.W0)
        else:
            result = 0

        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_output * self.scaling

    def merge_weights(self) -> None:
        if self._merged or self.W0 is None:
            return 
        
        with torch.no_grad():
            merged = self.W0 + (self.lora_B @ self.lora_A) * self.scaling
            self.register_buffer("W0", merged.detach().clone())

        self._merged = True


    def unmerge_weights(self) -> None:
        if not self._merged or self.W0 is  None:
            return 
        
        with torch.no_grad():
            unmerged = self.W0 - (self.lora_B @ self.lora_A) * self.scaling
            self.register_buffer("W0", unmerged.detach().clone())

        self._merged = False


class LoRALinear(nn.Module):
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.original_layer = original_layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        self.lora.register_buffer("W0", original_layer.weight.data.clone())

        self.bias = original_layer.bias

    @property
    def in_features(self) -> int:
        return self.original_layer.in_features

    @property
    def out_features(self) -> int:
        return self.original_layer.out_features

    @property
    def weight(self) -> torch.Tensor:
        return self.original_layer.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.lora(x)

        if self.bias is not None:
            output = output + self.bias

        return output


def apply_lora_to_block(
    block: nn.Module,
    rank: int,
    alpha: float | None,
    dropout: float,
    target_modules: list[str],
) -> None:
    if hasattr(block, "attn"):
        attn = block.attn  # type: ignore
    elif hasattr(block, "self_attn"):
        attn = block.self_attn  # type: ignore
    else:
        return

    for module_name in target_modules:
        if not hasattr(attn, module_name):
            continue

        original_layer = getattr(attn, module_name)

        if isinstance(original_layer, nn.Linear):
            lora_layer = LoRALinear(
                original_layer=original_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            setattr(attn, module_name, lora_layer)

    if hasattr(block, "mlp"):
        mlp = block.mlp
        if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):  # type: ignore
            original_layer = mlp.fc1  # type: ignore
            lora_layer = LoRALinear(
                original_layer=original_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            mlp.fc1 = lora_layer  # type: ignore


def apply_lora_to_all_layers(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> None:
    if target_modules is None:
        target_modules = ["qkv"] 

    blocks: nn.ModuleList

    if hasattr(model, "blocks"):
        blocks = model.blocks  # type: ignore
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        blocks = model.encoder.layers  # type: ignore
    else:
        raise ValueError("Cannot find transformer blocks in model")

    print(
        f"Applying low-rank LoRA (r={rank}) to ALL {len(blocks)} transformer layers...",
    )

    for block in blocks:
        apply_lora_to_block(
            block,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
        )


def create_lora_model(
    base_model: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> nn.Module:
    apply_lora_to_all_layers(
        base_model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    for name, param in base_model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False

    return base_model


def merge_and_reset_lora(model: nn.Module, rank: int, alpha: float | None) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora = module.lora
            lora.merge_weights()

            in_features = lora.in_features
            out_features = lora.out_features

            lora.rank = rank
            lora.alpha = alpha if alpha is not None else rank
            lora.scaling = lora.alpha / lora.rank

            lora.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            lora.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(lora.lora_A, a=math.sqrt(5))

            lora._merged = False
