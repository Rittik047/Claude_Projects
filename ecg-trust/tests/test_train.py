import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecgtrust.config import load_config
from ecgtrust.models.train import train_model

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture()
def tiny_cfg(tmp_path):
    train = {
        "width": 8,
        "blocks": [1, 1],
        "batch_size": 16,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "max_epochs": 25,
        "patience": 25,
    }
    return load_config(CFG, artifacts=str(tmp_path), train=train)


def synthetic_ecg(n=64, seed=0, t=250):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 5)) < 0.4).astype(np.float32)
    x = rng.normal(0, 0.1, (n, 12, t)).astype(np.float32)
    tt = np.arange(t) / 100
    for c in range(5):  # class c ↔ 2+c Hz sinusoid in lead c
        x[:, c] += y[:, c, None] * np.sin(2 * np.pi * (2 + c) * tt)[None]
    return x, y


def first_epoch_loss(cfg, seed, out):
    x, y = synthetic_ecg()
    train_model(cfg, x, y, x, y, seed=seed, out_dir=out, max_epochs=1)
    return json.loads((out / "train_log.jsonl").read_text().splitlines()[0])["train_loss"]


def test_overfits_tiny_dataset(tmp_path, tiny_cfg):
    x, y = synthetic_ecg()
    ckpt = train_model(tiny_cfg, x, y, x, y, seed=0, out_dir=tmp_path)
    assert torch.load(ckpt, weights_only=False)["val_select_macro_auroc"] > 0.95


def test_same_seed_same_first_epoch_loss(tmp_path, tiny_cfg):
    a = first_epoch_loss(tiny_cfg, seed=1, out=tmp_path / "a")
    b = first_epoch_loss(tiny_cfg, seed=1, out=tmp_path / "b")
    assert abs(a - b) < 1e-6


def test_stats_come_from_train_split(tmp_path, tiny_cfg):
    x_tr, y_tr = synthetic_ecg(seed=2)
    x_va, y_va = synthetic_ecg(seed=3)
    x_va = x_va + 5.0  # validation statistics must not leak into normalisation
    ckpt = torch.load(
        train_model(tiny_cfg, x_tr, y_tr, x_va, y_va, 0, tmp_path, max_epochs=1), weights_only=False
    )
    assert np.allclose(ckpt["lead_mean"], x_tr.mean(axis=(0, 2)), atol=1e-5)
