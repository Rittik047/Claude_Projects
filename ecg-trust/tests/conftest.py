"""Shared fixtures: a synthetic PTB-XL-like layout with stored validation predictions."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgtrust import CLASSES
from ecgtrust.config import load_config
from ecgtrust.data.ptbxl import load_metadata
from ecgtrust.data.splits import make_splits
from ecgtrust.models.ensemble import save_preds

FIX = Path(__file__).parent / "fixtures"
CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
CODES = {"NORM": "NORM", "MI": "IMI", "STTC": "NDT", "CD": "LAFB", "HYP": "LVH"}


def write_synthetic_ptbxl(root: Path, n: int = 600, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIX / "ptbxl" / "scp_statements.csv", root / "scp_statements.csv")
    patient = np.arange(n) // 2  # two records per patient
    rows = []
    for i in range(n):
        labels = [c for c in CLASSES if rng.random() < 0.35] or ["NORM"]
        rows.append(
            {
                "ecg_id": i + 1,
                "patient_id": float(patient[i]),
                "age": float(rng.choice([35, 50, 70, 85, 300])),
                "sex": int(rng.integers(0, 2)),
                "device": rng.choice(["CS-12   E", "AT-6 C 5.5", "TINY"], p=[0.5, 0.48, 0.02]),
                "scp_codes": str({CODES[c]: 100.0 for c in labels}),
                "strat_fold": int(patient[i] % 10) + 1,
                "validated_by_human": True,
                "baseline_drift": "x" if rng.random() < 0.2 else None,
                "static_noise": "y" if rng.random() < 0.2 else None,
                "burst_noise": None,
                "electrodes_problems": None,
                "pacemaker": "ja" if rng.random() < 0.05 else None,
                "filename_lr": f"records100/{i:05d}_lr",
            }
        )
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)


@pytest.fixture()
def tmp_cfg_val(tmp_path):
    cfg = load_config(
        CFG,
        data_root=str(tmp_path / "data"),
        artifacts=str(tmp_path / "art"),
        reports=str(tmp_path / "rep"),
        n_boot=50,
    )
    write_synthetic_ptbxl(cfg.data_root / "ptbxl")
    meta = load_metadata(cfg.data_root / "ptbxl")
    splits = make_splits(meta)
    cfg.artifacts.mkdir(parents=True)
    splits.to_frame().to_parquet(cfg.artifacts / "splits.parquet")
    from ecgtrust.data.ptbxl import load_scp, superclass_labels

    y = superclass_labels(meta, load_scp(cfg.data_root / "ptbxl")).to_numpy().astype(np.float32)
    rng = np.random.default_rng(1)
    for s in ("val_select", "val_calib"):
        m = (splits == s).to_numpy()
        logits = (y[m] * 4 - 2)[None] * 1.5 + rng.normal(0, 1.5, (3, m.sum(), 5))  # overconfident
        save_preds(
            cfg.artifacts / "preds" / f"resnet_{s}.npz",
            logits,
            y[m],
            meta.index.to_numpy()[m],
            meta.patient_id.to_numpy()[m],
        )
    return cfg


@pytest.fixture()
def tmp_cfg_unfrozen(tmp_cfg_val):
    return tmp_cfg_val
