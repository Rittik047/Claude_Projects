"""Late fusion of the ResNet ensemble with the GE 12SL-measurement baseline.

    z = w * logit(p_resnet) + (1 - w) * logit(p_12sl)

A single scalar `w` is chosen on `val_select` (macro-AUROC over an 11-point grid). Calibration
and conformal thresholds are then fit on `val_calib` exactly as for the ResNet, so `test`
stays untouched. The fused output is stored as a one-member "logits" array, which lets the
calibration and reporting code run unchanged.

The fusion needs GE 12SL measurements, which only PTB-XL+ provides. It is therefore a
PTB-XL-only research model: the served model and the external evaluation stay ResNet-only.
"""

from __future__ import annotations

import json

import numpy as np

from ecgtrust.config import Config
from ecgtrust.evaluate.metrics import macro_auroc, patient_bootstrap
from ecgtrust.models.ensemble import ensemble_proba, load_preds, save_preds

GRID = tuple(round(float(w), 2) for w in np.linspace(0.0, 1.0, 11))
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fuse(resnet_logits: np.ndarray, p_base: np.ndarray, w: float) -> np.ndarray:
    """(M, N, 5) ResNet logits + (N, 5) baseline probabilities -> (1, N, 5) fused logits."""
    z = w * logit(ensemble_proba(resnet_logits)) + (1 - w) * logit(p_base)
    return z[None].astype(np.float32)


def select_weight(
    resnet_logits: np.ndarray, p_base: np.ndarray, y: np.ndarray, grid=GRID
) -> tuple[float, dict]:
    scores = {w: macro_auroc(y, ensemble_proba(fuse(resnet_logits, p_base, w))) for w in grid}
    return max(scores, key=scores.get), scores


def _pair(cfg: Config, split: str) -> tuple[dict, np.ndarray]:
    r = load_preds(cfg.artifacts / "preds" / f"resnet_{split}.npz")
    with np.load(cfg.artifacts / "preds" / f"baseline_12sl_{split}.npz") as b:
        if not np.array_equal(b["ecg_id"], r["ecg_id"]):
            raise ValueError(f"ResNet and 12SL predictions do not align on {split}")
        return r, b["p"]


def fusion_weight(cfg: Config) -> float:
    return float(json.loads((cfg.artifacts / "models" / "fusion.json").read_text())["w"])


def build_fusion(cfg: Config) -> dict:
    """Select w on val_select; store fused predictions for both validation splits."""
    r, pb = _pair(cfg, "val_select")
    w, scores = select_weight(r["logits"], pb, r["y"])
    (cfg.artifacts / "models").mkdir(parents=True, exist_ok=True)
    (cfg.artifacts / "models" / "fusion.json").write_text(
        json.dumps({"w": w, "selected_on": "val_select", "grid_macro_auroc": scores}, indent=2)
    )
    summary = {"w": w, "grid_macro_auroc": scores}
    for s in ("val_select", "val_calib"):
        rs, ps = (r, pb) if s == "val_select" else _pair(cfg, s)
        z = fuse(rs["logits"], ps, w)
        save_preds(
            cfg.artifacts / "preds" / f"fusion_{s}.npz", z, rs["y"], rs["ecg_id"], rs["patient_id"]
        )
        summary[s] = list(
            patient_bootstrap(
                rs["y"], ensemble_proba(z), rs["patient_id"], macro_auroc, n=cfg.n_boot
            )
        )
    (cfg.reports / "val").mkdir(parents=True, exist_ok=True)
    (cfg.reports / "val" / "fusion_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def fused_test_logits(cfg: Config, resnet_logits: np.ndarray, ecg_ids: np.ndarray) -> np.ndarray:
    """Fuse ResNet test logits with the frozen 12SL baseline's test probabilities."""
    import joblib

    from ecgtrust.features.ptbxl_plus import load_features

    model = joblib.load(cfg.artifacts / "models" / "baseline_12sl.joblib")
    feats = load_features(cfg.dataset_dir("ptbxl_plus"), "12sl").reindex(ecg_ids)
    return fuse(resnet_logits, model.predict_proba(feats), fusion_weight(cfg))
