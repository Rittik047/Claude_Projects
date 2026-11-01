"""Compact 1-D ResNet for 12-lead ECG (spec §4.2)."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, kernel: int):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm1d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(out)) + self.skip(x))


class ResNet1d(nn.Module):
    def __init__(
        self,
        n_leads: int = 12,
        n_classes: int = 5,
        width: int = 64,
        blocks: tuple[int, ...] = (2, 2, 2, 2),
        kernel: int = 7,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, width, 15, 2, 7, bias=False), nn.BatchNorm1d(width), nn.ReLU()
        )
        layers, c = [], width
        for i, n in enumerate(blocks):
            c_out = width * 2 ** min(i, 1)  # channels w, 2w, 2w, 2w: stays < 2M params
            for j in range(n):
                layers.append(BasicBlock1d(c, c_out, 2 if (j == 0 and i > 0) else 1, kernel))
                c = c_out
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.layers(self.stem(x)).mean(dim=-1))
