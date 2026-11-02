"""`ecgtrust train`: fits models on `train` and stores predictions for the validation splits
only. It never computes predictions for `test` or `external` (those go through
`ecgtrust evaluate` after the freeze)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.config import Config, load_config
from ecgtrust.data.ptbxl import load_metadata, load_scp, superclass_labels
from ecgtrust.evaluate.metrics import macro_auroc, patient_bootstrap, per_class_auroc

VAL_SPLITS = ("val_select", "val_calib")


def load_labelled(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    root = cfg.dataset_dir("ptbxl")
    meta = load_metadata(root)
    y = superclass_labels(meta, load_scp(root))
    splits = pd.read_parquet(cfg.artifacts / "splits.parquet")["split"].reindex(meta.index)
    return meta, y, splits


def summarize(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int) -> dict:
    return {
        "n": int(len(y)),
        "macro_auroc": list(patient_bootstrap(y, p, groups, macro_auroc, n=n_boot)),
        "per_class_auroc": [float(v) for v in per_class_auroc(y, p)],
        "classes": list(CLASSES),
    }


def train_baseline(cfg: Config, source: str, seed: int = 0) -> dict:
    from ecgtrust.features.ptbxl_plus import load_features
    from ecgtrust.models.baseline import fit_baseline

    meta, y, splits = load_labelled(cfg)
    feats = load_features(cfg.dataset_dir("ptbxl_plus"), source).reindex(meta.index)
    tr = splits == "train"
    model = fit_baseline(feats[tr], y[tr], seed=seed)
    (cfg.artifacts / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.artifacts / "models" / f"baseline_{source}.joblib")
    (cfg.artifacts / "preds").mkdir(parents=True, exist_ok=True)
    out = {"features": source}
    for s in VAL_SPLITS:
        m = splits == s
        p = model.predict_proba(feats[m])
        np.savez(
            cfg.artifacts / "preds" / f"baseline_{source}_{s}.npz",
            p=p,
            y=y[m].to_numpy(),
            ecg_id=meta.index[m].to_numpy(),
            patient_id=meta.loc[m, "patient_id"].to_numpy(),
        )
        if s == "val_select":
            out.update(
                summarize(y[m].to_numpy(), p, meta.loc[m, "patient_id"].to_numpy(), cfg.n_boot)
            )
    (cfg.reports / "val").mkdir(parents=True, exist_ok=True)
    (cfg.reports / "val" / f"baseline_{source}.json").write_text(json.dumps(out, indent=2))
    return out


def register(sub) -> None:
    p = sub.add_parser("train", help="train a model on the train split")
    p.add_argument("--model", required=True, choices=["baseline", "resnet", "fusion"])
    p.add_argument("--features", choices=["12sl", "ecgdeli"], default="12sl")
    p.add_argument("--seeds", type=int, nargs="+", default=None)

    def run(args):
        cfg = load_config(args.config)
        if args.model == "baseline":
            r = train_baseline(cfg, args.features)
            print(f"baseline[{args.features}] val_select macro-AUROC {r['macro_auroc']}")
        elif args.model == "fusion":
            from ecgtrust.models.fusion import build_fusion

            r = build_fusion(cfg)
            print(f"fusion w={r['w']} val_select macro-AUROC {r['val_select']}")
        else:
            from ecgtrust.models.ensemble import train_resnet_seeds

            train_resnet_seeds(cfg, args.seeds or cfg.seeds)

    p.set_defaults(func=run)


__all__ = ["load_labelled", "summarize", "train_baseline", "register", "Path"]
