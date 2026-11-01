import shutil
from pathlib import Path

import numpy as np
import pytest

from ecgtrust.config import load_config
from ecgtrust.data.ptbxl import load_metadata
from ecgtrust.data.splits import make_splits
from ecgtrust.models.ensemble import ensemble_proba, load_preds, train_resnet_seeds

FIX = Path(__file__).parent / "fixtures"
CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_ensemble_is_mean_of_member_probs():
    lg = np.array([[[0.0] * 5], [[2.0] * 5]])
    assert np.allclose(ensemble_proba(lg), (0.5 + 1 / (1 + np.exp(-2))) / 2)


def test_temperature_applied_per_class():
    lg = np.ones((1, 1, 5))
    t = np.array([1, 2, 1, 1, 1.0])
    assert np.isclose(ensemble_proba(lg, t)[0, 1], 1 / (1 + np.exp(-0.5)))


@pytest.fixture()
def tiny_real_layout_cfg(tmp_path):
    train = {
        "width": 4,
        "blocks": [1],
        "batch_size": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "patience": 1,
    }
    cfg = load_config(
        CFG,
        data_root=str(tmp_path / "data"),
        artifacts=str(tmp_path / "art"),
        reports=str(tmp_path / "rep"),
        train=train,
        seeds=[0],
    )
    shutil.copytree(FIX / "ptbxl", cfg.data_root / "ptbxl")
    meta = load_metadata(cfg.data_root / "ptbxl")
    cfg.artifacts.mkdir(parents=True)
    make_splits(meta).to_frame().to_parquet(cfg.artifacts / "splits.parquet")
    rng = np.random.default_rng(0)
    np.save(cfg.artifacts / "ptbxl_x100.npy", rng.normal(size=(len(meta), 12, 1000)).astype("f4"))
    return cfg, make_splits(meta)


def test_train_cli_never_touches_test(monkeypatch, tiny_real_layout_cfg):
    cfg, splits = tiny_real_layout_cfg
    seen = []
    import ecgtrust.predict as pred

    real = pred.predict_logits

    def spy(ckpts, x, **k):
        seen.append(len(x))
        return real(ckpts, x, **k)

    monkeypatch.setattr(pred, "predict_logits", spy)
    train_resnet_seeds(cfg, [0])
    n_val = int(splits.isin(["val_select", "val_calib"]).sum())
    assert sum(seen) == n_val  # no test/external rows ever requested
    p = load_preds(cfg.artifacts / "preds" / "resnet_val_select.npz")
    assert p["logits"].shape[0] == 1 and p["logits"].shape[2] == 5
    assert set(p) >= {"logits", "y", "ecg_id", "patient_id"}
