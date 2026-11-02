"""Minimal LoRA for nn.Linear layers (attention qkv/proj of ViT/Swin encoders)."""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora(module: nn.Module, r: int = 8, alpha: int = 16, targets=("qkv", "proj")) -> nn.Module:
    """Freeze all parameters, then wrap every nn.Linear whose attribute name is in `targets`."""
    for p in module.parameters():
        p.requires_grad_(False)
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(parent, name, LoRALinear(child, r, alpha))
    return module
