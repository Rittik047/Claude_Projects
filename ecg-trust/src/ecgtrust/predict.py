"""Inference from saved checkpoints. Each checkpoint normalises with its own train stats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ecgtrust.models.train import build_model, predict_logits_array


def load_checkpoint(path: Path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def predict_logits(checkpoints: list[Path], x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """(M, N, 5) logits for M checkpoints."""
    torch.set_num_threads(4)
    out = []
    for path in checkpoints:
        model, ck = load_checkpoint(path)
        out.append(predict_logits_array(model, x, ck["lead_mean"], ck["lead_std"], batch_size))
    return np.stack(out)
