"""Seed ensemble (Lakshminarayanan et al. 2017) and the validation prediction store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ecgtrust import predict
from ecgtrust.config import Config


def ensemble_proba(logits: np.ndarray, temperature: np.ndarray | None = None) -> np.ndarray:
    """Mean over members of sigmoid(logits / T); logits (M, N, 5), T (5,) or None."""
    t = np.ones(logits.shape[-1]) if temperature is None else np.asarray(temperature, float)
    return (1.0 / (1.0 + np.exp(-logits / t))).mean(axis=0)


def save_preds(path: Path, logits, y, ecg_id, patient_id) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, logits=logits, y=y, ecg_id=ecg_id, patient_id=patient_id)


def load_preds(path: Path) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def checkpoint_paths(cfg: Config, seeds: list[int] | None = None) -> list[Path]:
    seeds = cfg.seeds if seeds is None else seeds
    return [cfg.artifacts / "models" / f"seed{s}" / f"resnet_seed{s}.pt" for s in seeds]


def train_resnet_seeds(cfg: Config, seeds: list[int]) -> list[Path]:
    """Train one ResNet per seed on `train`; store logits for val_select and val_calib only."""
    from ecgtrust.evaluate.metrics import macro_auroc
    from ecgtrust.models.train import train_model
    from ecgtrust.train_cli import load_labelled

    meta, y, splits = load_labelled(cfg)
    x = np.load(cfg.artifacts / "ptbxl_x100.npy", mmap_mode="r")
    rows = {
        s: np.flatnonzero((splits == s).to_numpy()) for s in ("train", "val_select", "val_calib")
    }
    yv = y.to_numpy().astype(np.float32)
    ckpts = []
    for seed in seeds:
        out = cfg.artifacts / "models" / f"seed{seed}"
        ck = train_model(
            cfg,
            x[rows["train"]],
            yv[rows["train"]],
            x[rows["val_select"]],
            yv[rows["val_select"]],
            seed=seed,
            out_dir=out,
        )
        ckpts.append(ck)
        print(f"seed {seed}: {ck}", flush=True)
    summary = {}
    for s in ("val_select", "val_calib"):
        r = rows[s]
        lg = predict.predict_logits(ckpts, x[r])
        save_preds(
            cfg.artifacts / "preds" / f"resnet_{s}.npz",
            lg,
            yv[r],
            meta.index.to_numpy()[r],
            meta["patient_id"].to_numpy()[r],
        )
        summary[s] = {
            "ensemble": macro_auroc(yv[r], ensemble_proba(lg)),
            "members": [
                macro_auroc(yv[r], ensemble_proba(lg[i : i + 1])) for i in range(len(ckpts))
            ],
        }
    (cfg.reports / "val").mkdir(parents=True, exist_ok=True)
    (cfg.reports / "val" / "resnet_summary.json").write_text(json.dumps(summary, indent=2))
    return ckpts
