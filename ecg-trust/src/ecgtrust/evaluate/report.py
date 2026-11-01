"""`ecgtrust evaluate --split ...`: discrimination, calibration, conformal and subgroup report.

Temperature and conformal thresholds are always fit on `val_calib` only. `test` and
`external` are locked behind the freeze manifest and logged as single-use runs.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.calibrate.conformal import coverage, decide, fit_conformal
from ecgtrust.calibrate.metrics import brier, ece
from ecgtrust.calibrate.temperature import fit_temperature
from ecgtrust.config import Config
from ecgtrust.evaluate import plots
from ecgtrust.evaluate.metrics import (
    bootstrap_samples,
    macro_auroc,
    patient_bootstrap,
    per_class_auprc,
    per_class_auroc,
)
from ecgtrust.evaluate.subgroups import DEFAULT_COLUMNS, subgroup_table
from ecgtrust.freeze import freeze_hash, log_test_run, require_freeze
from ecgtrust.models.ensemble import checkpoint_paths, ensemble_proba, load_preds, save_preds

EXTRA_MODULES = (
    "ecgtrust.robustness.run",
    "ecgtrust.robustness.occlusion",
    "ecgtrust.evaluate.comparator",
)


def extra_analyses() -> list:
    """(name, fn(cfg, split, ctx)) from modules exposing ANALYSIS; missing modules are skipped."""
    import importlib

    out = []
    for mod in EXTRA_MODULES:
        try:
            out.append(importlib.import_module(mod).ANALYSIS)
        except ModuleNotFoundError:
            continue
    return out


def _clean(o):
    """JSON-safe: NaN/inf → None, numpy → python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def fit_calibrators(cfg: Config) -> tuple[np.ndarray, dict]:
    cal = load_preds(cfg.artifacts / "preds" / "resnet_val_calib.npz")
    temps = fit_temperature(cal["logits"], cal["y"])
    p_cal = ensemble_proba(cal["logits"], temps)
    thr = {str(a): fit_conformal(p_cal, cal["y"], a) for a in cfg.alphas}
    out = cfg.artifacts / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "T.npy", temps)
    for a, t in thr.items():
        (out / f"conformal_{a}.json").write_text(json.dumps(_clean(t.to_json())))
    return temps, thr


def _split_inputs(cfg: Config, split: str, rerun_reason: str | None):
    """Return (logits, y, groups, meta_rows, extra) for the split."""
    from ecgtrust import predict
    from ecgtrust.train_cli import load_labelled

    if split in ("val_select", "val_calib"):
        d = load_preds(cfg.artifacts / "preds" / f"resnet_{split}.npz")
        meta, _, _ = load_labelled(cfg)
        return d["logits"], d["y"], d["patient_id"], meta.loc[d["ecg_id"]], {}
    log_test_run(cfg, split, freeze_hash(cfg), rerun_reason)
    ckpts = checkpoint_paths(cfg)
    if split == "test":
        meta, y, splits = load_labelled(cfg)
        r = np.flatnonzero((splits == "test").to_numpy())
        x = np.load(cfg.artifacts / "ptbxl_x100.npy", mmap_mode="r")
        lg = predict.predict_logits(ckpts, x[r])
        yv = y.to_numpy().astype(np.float32)[r]
        save_preds(
            cfg.artifacts / "preds" / "resnet_test.npz",
            lg,
            yv,
            meta.index.to_numpy()[r],
            meta.patient_id.to_numpy()[r],
        )
        return lg, yv, meta.patient_id.to_numpy()[r], meta.iloc[r], {}
    em = pd.read_parquet(cfg.artifacts / "external_meta.parquet")
    em = em[em.excluded_reason == ""]
    x = np.load(cfg.artifacts / "external_x100.npy", mmap_mode="r")
    rows = em.cache_row.to_numpy()
    lg_all = predict.predict_logits(ckpts, x[np.sort(rows)])
    lg_all = lg_all[:, np.argsort(np.argsort(rows))]
    save_preds(
        cfg.artifacts / "preds" / "resnet_external_all.npz",
        lg_all,
        em[list(CLASSES)].to_numpy().astype(np.float32),
        em.record.to_numpy(),
        em.record.to_numpy(),
    )
    lab = em.has_label.to_numpy()
    lib = em.has_label_liberal.to_numpy()
    extra = {
        "liberal": (
            lg_all[:, lib],
            em.loc[lib, [c + "_liberal" for c in CLASSES]].to_numpy().astype(np.float32),
        )
    }
    y = em.loc[lab, list(CLASSES)].to_numpy().astype(np.float32)
    return lg_all[:, lab], y, em.record.to_numpy()[lab], em[lab], extra


def _discrimination(y, p, groups, n_boot) -> dict:
    per = []
    for c in range(len(CLASSES)):

        def auc_c(yy, pp, c=c):
            return float(per_class_auroc(yy[:, c : c + 1], pp[:, c : c + 1])[0])

        bs = (
            bootstrap_samples(y, p, groups, auc_c, n=n_boot) if np.unique(y[:, c]).size == 2 else []
        )
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)
        per.append(
            {
                "class": CLASSES[c],
                "n_pos": int(y[:, c].sum()),
                "auroc": auc_c(y, p),
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    return {
        "macro_auroc": dict(
            zip(
                ("point", "lo", "hi"),
                patient_bootstrap(y, p, groups, macro_auroc, n=n_boot),
                strict=True,
            )
        ),
        "per_class": per,
        "auprc": per_class_auprc(y, p).tolist(),
    }


def _calibration(y, p_pre, p_post, bins) -> dict:
    pre = [ece(y[:, c], p_pre[:, c], bins) for c in range(5)]
    post = [ece(y[:, c], p_post[:, c], bins) for c in range(5)]
    return {
        "ece_pre": pre,
        "ece_post": post,
        "macro_ece_pre": float(np.mean(pre)),
        "macro_ece_post": float(np.mean(post)),
        "brier_pre": [brier(y[:, c], p_pre[:, c]) for c in range(5)],
        "brier_post": [brier(y[:, c], p_post[:, c]) for c in range(5)],
    }


def evaluate_split(
    cfg: Config, split: str, model: str = "resnet", rerun_reason: str | None = None
) -> dict:
    manifest = require_freeze(cfg, split)
    temps, thr = fit_calibrators(cfg)
    logits, y, groups, meta_rows, extra = _split_inputs(cfg, split, rerun_reason)
    p_pre, p_post = ensemble_proba(logits), ensemble_proba(logits, temps)
    out_dir = cfg.reports / split
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {
        "split": split,
        "n": int(len(y)),
        "temperature": temps.tolist(),
        "freeze": manifest.get("git_commit"),
        "discrimination": _discrimination(y, p_post, groups, cfg.n_boot),
        "calibration": _calibration(y, p_pre, p_post, cfg.ece_bins),
        "conformal": {
            a: {k: v.tolist() for k, v in coverage(y, p_post, t).items()} for a, t in thr.items()
        },
    }
    min_pos = cfg.external_min_pos if split == "external" else 30
    if split == "external":
        res["prevalence"] = {
            c: {
                "n_pos": int(y[:, i].sum()),
                "status": "ok" if y[:, i].sum() >= cfg.external_min_pos else "insufficient",
            }
            for i, c in enumerate(CLASSES)
        }
        lg_l, y_l = extra["liberal"]
        res["liberal_sensitivity"] = {
            "n": int(len(y_l)),
            "per_class_auroc": per_class_auroc(y_l, ensemble_proba(lg_l, temps)).tolist(),
        }
        cols = ["sex", "age_band"]
    else:
        cols = DEFAULT_COLUMNS
    sub = subgroup_table(
        meta_rows, y, p_post, cols, groups, min_pos=min(min_pos, 30), n_boot=cfg.n_boot
    )
    sub.to_csv(out_dir / "subgroups.csv", index=False)
    ctx = {
        "y": y,
        "p": p_post,
        "logits": logits,
        "temps": temps,
        "thr": thr,
        "meta": meta_rows,
        "groups": groups,
        "decisions": decide(p_post, thr[str(max(cfg.alphas))]),
    }
    have_ckpts = all(p.exists() for p in checkpoint_paths(cfg))
    for name, fn in extra_analyses():
        res[name] = fn(cfg, split, ctx) if have_ckpts else {"skipped": "no checkpoints"}
    aucs = [r["auroc"] for r in res["discrimination"]["per_class"]]
    plots.roc_plot(y, p_post, aucs, out_dir / "roc.png", f"ROC — {split}")
    plots.reliability_plot(
        y,
        p_pre,
        res["calibration"]["ece_pre"],
        out_dir / "reliability_pre.png",
        f"Reliability before scaling — {split}",
        cfg.ece_bins,
    )
    plots.reliability_plot(
        y,
        p_post,
        res["calibration"]["ece_post"],
        out_dir / "reliability_post.png",
        f"Reliability after temperature scaling — {split}",
        cfg.ece_bins,
    )
    (out_dir / "metrics.json").write_text(json.dumps(_clean(res), indent=2))
    (out_dir / "summary.md").write_text(render_summary(_clean(res)))
    return res


def render_summary(r: dict) -> str:
    d, c = r["discrimination"], r["calibration"]
    m = d["macro_auroc"]
    lines = [
        f"# Evaluation — {r['split']} (n = {r['n']})",
        "",
        f"Macro-AUROC **{m['point']:.3f}** [95% CI {m['lo']:.3f}, {m['hi']:.3f}] "
        "(patient-level bootstrap).",
        "",
        "| class | positives | AUROC [95% CI] | AUPRC | ECE pre → post |",
        "|---|---|---|---|---|",
    ]
    for i, row in enumerate(d["per_class"]):
        au = (
            "n/a"
            if row["auroc"] is None
            else (
                f"{row['auroc']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]"
                if row["lo"] is not None
                else f"{row['auroc']:.3f}"
            )
        )
        ap = d["auprc"][i]
        lines.append(
            f"| {row['class']} | {row['n_pos']} | {au} | "
            f"{'n/a' if ap is None else f'{ap:.3f}'} | "
            f"{c['ece_pre'][i]:.3f} → {c['ece_post'][i]:.3f} |"
        )
    lines += [
        "",
        f"Macro ECE {c['macro_ece_pre']:.3f} → {c['macro_ece_post']:.3f}; "
        f"temperatures {', '.join(f'{t:.2f}' for t in r['temperature'])}.",
        "",
    ]
    for a, cv in r["conformal"].items():
        fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(
            f"Conformal α={a}: coverage(pos) {', '.join(map(fmt, cv['cov_pos']))}; "
            f"refer rate {', '.join(map(fmt, cv['refer_rate']))}."
        )
    return "\n".join(lines) + "\n"


def register(sub) -> None:
    p = sub.add_parser("evaluate", help="evaluate a split (test/external require a freeze)")
    p.add_argument("--split", required=True, choices=["val_select", "test", "external"])
    p.add_argument("--rerun-reason", default=None)

    def run(args):
        from ecgtrust.config import load_config

        r = evaluate_split(load_config(args.config), args.split, rerun_reason=args.rerun_reason)
        print(
            Path(load_config(args.config).reports / args.split / "summary.md").read_text()
            if r
            else ""
        )

    p.set_defaults(func=run)
