"""Guide figures, generated only from saved reports (never recomputed from models)."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecgtrust import CLASSES  # noqa: E402

INK, MUTED, ACCENT = "#1f2a44", "#8a94a6", "#0072B2"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
    }
)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def prevalence(summary: dict, path: Path) -> Path | None:
    if "ptbxl" not in summary:
        return None
    series = {
        "PTB-XL train": summary["ptbxl"]["prevalence"]["train"],
        "PTB-XL test": summary["ptbxl"]["prevalence"]["test"],
    }
    ext = summary.get("external", {})
    if "prevalence_strict" in ext:
        series["External (strict map)"] = ext["prevalence_strict"]
    if "prevalence_liberal" in ext:
        series["External (liberal map)"] = ext["prevalence_liberal"]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    w = 0.8 / len(series)
    for i, (name, prev) in enumerate(series.items()):
        ax.bar(
            np.arange(5) + i * w, [prev[c] * 100 for c in CLASSES], w, label=name, color=PALETTE[i]
        )
    ax.set_xticks(np.arange(5) + w * (len(series) - 1) / 2, CLASSES)
    ax.set_ylabel("records with label (%)")
    ax.set_title("Label prevalence: development data vs external validation set")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    return _save(fig, path)


def validation_by_fold(summary: dict, path: Path) -> Path | None:
    v = summary.get("ptbxl", {}).get("validated_by_human")
    if not v:
        return None
    folds = sorted(v, key=int)
    fig, ax = plt.subplots(figsize=(6, 2.3))
    ax.bar(
        folds, [v[f] * 100 for f in folds], color=[ACCENT if int(f) >= 9 else MUTED for f in folds]
    )
    ax.set(
        xlabel="PTB-XL fold",
        ylabel="human-validated (%)",
        ylim=(0, 105),
        title="Folds 9–10 (validation/test) are fully cardiologist-validated",
    )
    return _save(fig, path)


def model_selection(val: dict, path: Path) -> Path | None:
    rows = []
    for k, label in (
        ("baseline_12sl", "GBM on GE 12SL features"),
        ("baseline_ecgdeli", "GBM on ECGDeli features"),
    ):
        if k in val:
            a = val[k]["macro_auroc"]
            rows.append((label, a[0], a[1], a[2]))
    if "resnet" in val:
        for i, m in enumerate(val["resnet"]["val_select"]["members"]):
            rows.append((f"ResNet seed {i}", m, None, None))
        e = val["resnet"]["val_select"]["ensemble"]
        rows.append(("ResNet 3-seed ensemble", e, None, None))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(rows) + 0.9))
    for i, (label, pt, lo, hi) in enumerate(rows):
        c = ACCENT if "ensemble" in label else (PALETTE[1] if "GBM" in label else MUTED)
        ax.plot(pt, i, "o", color=c, ms=7)
        if lo is not None:
            ax.hlines(i, lo, hi, color=c, lw=2)
        ax.text(pt, i + 0.28, f"{pt:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("macro-AUROC on val_select (95% CI where available)")
    ax.set_title("Model selection on validation data only")
    return _save(fig, path)


def subgroup_forest(csv: Path, path: Path) -> Path | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    df = df[(df.status == "ok") & (df["column"].isin(["sex_label", "age_band", "sex"]))]
    if df.empty:
        return None
    df = df.assign(label=df["column"].str.replace("_label", "") + ": " + df.group.astype(str))
    labels = sorted(df.label.unique())
    fig, axes = plt.subplots(1, 5, figsize=(9.5, 0.32 * len(labels) + 1.2), sharey=True)
    for k, (ax, c) in enumerate(zip(axes, CLASSES, strict=True)):
        d = df[df["class"] == c].set_index("label").reindex(labels)
        yy = np.arange(len(labels))
        ax.hlines(yy, d.lo, d.hi, color=PALETTE[k], lw=2)
        ax.plot(d.auroc, yy, "o", color=PALETTE[k], ms=4)
        ax.set_title(c)
        ax.set_xlim(max(0.5, np.nanmin(d.lo) - 0.03), 1.0)
        ax.tick_params(labelsize=7)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.supxlabel("AUROC with patient-bootstrap 95% CI", fontsize=8)
    fig.suptitle("Subgroup performance (descriptive)", fontweight="bold")
    return _save(fig, path)


def internal_vs_external(test: dict, ext: dict, path: Path) -> Path | None:
    if not test or not ext:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.9))
    x, w = np.arange(5), 0.38
    for i, (m, name) in enumerate(((test, "internal test"), (ext, "external"))):
        au = [
            r["auroc"] if r["auroc"] is not None else np.nan
            for r in m["discrimination"]["per_class"]
        ]
        axes[0].bar(x + i * w, au, w, label=name, color=PALETTE[i])
        axes[1].bar(x + i * w, m["calibration"]["ece_post"], w, label=name, color=PALETTE[i])
    for ax, lab in zip(axes, ("AUROC", "ECE after temperature scaling"), strict=True):
        ax.set_xticks(x + w / 2, CLASSES)
        ax.set_ylabel(lab)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("What changes under dataset shift", fontweight="bold")
    return _save(fig, path)


def comparator(csv: Path, path: Path) -> Path | None:
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(df))
    ax.plot(df["12sl_sens"], y, "s", color=PALETTE[1], label="GE 12SL sensitivity")
    ax.plot(
        df["model_sens_at_12sl_spec"],
        y,
        "o",
        color=ACCENT,
        label="model sensitivity at 12SL's specificity",
    )
    for i, r in df.iterrows():
        ax.annotate(
            "",
            (r["model_sens_at_12sl_spec"], i),
            (r["12sl_sens"], i),
            arrowprops={"arrowstyle": "->", "color": MUTED},
        )
    ax.set_yticks(y, df["class"])
    ax.invert_yaxis()
    ax.set_xlabel("sensitivity")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Model vs a commercial interpretation algorithm, at matched specificity")
    return _save(fig, path)


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    if src.exists():
        shutil.copy(src, dst)
        return dst
    return None
