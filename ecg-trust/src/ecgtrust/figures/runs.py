"""Load the frozen predictions of each evaluation (no new evaluation is performed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.calibrate.conformal import ConformalThresholds
from ecgtrust.models.ensemble import ensemble_proba, load_preds


@dataclass
class Run:
    name: str  # display name
    key: str  # report folder name
    y: np.ndarray
    logits: np.ndarray
    temps: np.ndarray
    thr: dict  # alpha -> ConformalThresholds

    @property
    def p_pre(self) -> np.ndarray:
        return ensemble_proba(self.logits)

    @property
    def p(self) -> np.ndarray:
        return ensemble_proba(self.logits, self.temps)


def _calib(cfg, model: str):
    d = cfg.artifacts / "calibration"
    d = d if model == "resnet" else d / model
    if not (d / "T.npy").exists():
        return None, None
    thr = {
        str(a): ConformalThresholds.from_json(json.loads((d / f"conformal_{a}.json").read_text()))
        for a in cfg.alphas
    }
    return np.load(d / "T.npy"), thr


def load_runs(cfg) -> list[Run]:
    preds = cfg.artifacts / "preds"
    out = []
    t_r, thr_r = _calib(cfg, "resnet")
    t_f, thr_f = _calib(cfg, "fusion")
    f = preds / "resnet_test.npz"
    if f.exists() and t_r is not None:
        d = load_preds(f)
        out.append(Run("ResNet · internal test", "test", d["y"], d["logits"], t_r, thr_r))
        if t_f is not None and (cfg.artifacts / "models" / "fusion.json").exists():
            from ecgtrust.models.fusion import fused_test_logits

            z = fused_test_logits(cfg, d["logits"], d["ecg_id"])
            out.append(Run("Fusion · internal test", "test_fusion", d["y"], z, t_f, thr_f))
    f = preds / "resnet_external_all.npz"
    em_f = cfg.artifacts / "external_meta.parquet"
    if f.exists() and em_f.exists() and t_r is not None:
        with np.load(f) as z:  # record ids are strings (object array): read numeric keys only
            d = {k: z[k] for k in ("logits", "y")}
        em = pd.read_parquet(em_f)
        lab = em.loc[em.excluded_reason == "", "has_label"].to_numpy()
        out.append(
            Run("ResNet · external", "external", d["y"][lab], d["logits"][:, lab], t_r, thr_r)
        )
    return out


def val_select(cfg) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Name -> (y, p) on val_select for every model that has stored predictions."""
    preds: Path = cfg.artifacts / "preds"
    out = {}
    for name, f in (("12SL baseline", "baseline_12sl"), ("ECGDeli baseline", "baseline_ecgdeli")):
        if (preds / f"{f}_val_select.npz").exists():
            with np.load(preds / f"{f}_val_select.npz") as d:
                out[name] = (d["y"], d["p"])
    for name, f in (("ResNet ensemble", "resnet"), ("Fusion", "fusion")):
        if (preds / f"{f}_val_select.npz").exists():
            d = load_preds(preds / f"{f}_val_select.npz")
            out[name] = (d["y"], ensemble_proba(d["logits"]))
    return out


__all__ = ["CLASSES", "Run", "load_runs", "val_select"]
