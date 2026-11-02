"""01_data: cohort, labels, demographics, raw signals and 12SL measurements."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from ecgtrust import CLASSES, FS, LEADS
from ecgtrust.figures.style import CLASS_COLORS, INK, MUTED, SPLIT_COLORS, plt, save, subplots

SPLITS = ("train", "val_select", "val_calib", "test")
NOISE_COLS = ("baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "pacemaker")
GLOBAL_12SL = {
    "HR_Ventr_Global": "Heart rate (bpm)",
    "QRS_Dur_Global": "QRS duration (ms)",
    "QT_IntCorr_Global": "QTc (ms)",
    "PR_Int_Global": "PR interval (ms)",
    "R_AxisFrontal_Global": "Frontal QRS axis (°)",
    "T_AxisFront_Global": "Frontal T axis (°)",
}


def _single_label(y: pd.DataFrame) -> pd.Series:
    """Class name for records carrying exactly one superclass, else None."""
    one = y.sum(axis=1) == 1
    return y[one].idxmax(axis=1).reindex(y.index)


def prevalence_by_split(y, splits, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    width = 0.2
    for i, s in enumerate(SPLITS):
        m = (splits == s).to_numpy()
        frac = y[m].mean().to_numpy() * 100
        xs = np.arange(len(CLASSES)) + (i - 1.5) * width
        ax.bar(xs, frac, width, color=SPLIT_COLORS[s], label=f"{s} (n={m.sum():,})")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_ylabel("records with label (%)")
    ax.set_title("Superclass prevalence is stable across patient-level splits")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    return save(fig, out / "class_prevalence_by_split.png")


def cooccurrence(y, out: Path) -> Path:
    v = y.to_numpy().astype(float)
    cond = (v.T @ v) / v.sum(0)[:, None]  # row i: P(column j | row i)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cond, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{cond[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cond[i, j] > 0.6 else INK,
                fontsize=8,
            )
    ax.set_xticks(range(5), CLASSES)
    ax.set_yticks(range(5), CLASSES)
    ax.set_xlabel("also labelled with")
    ax.set_ylabel("given label")
    ax.set_title("Label co-occurrence  P(col | row)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, out / "label_cooccurrence.png")


def combinations(y, out: Path, top: int = 15) -> Path:
    combos = Counter(
        "+".join(c for c in CLASSES if row[c]) or "(none)" for _, row in y.iterrows()
    ).most_common(top)
    labels, counts = zip(*combos, strict=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(
        range(len(labels))[::-1], counts, color=[CLASS_COLORS.get(lbl, MUTED) for lbl in labels]
    )
    ax.set_yticks(range(len(labels))[::-1], labels, fontsize=8)
    for i, c in enumerate(counts):
        ax.text(c, len(labels) - 1 - i, f" {c:,}", va="center", fontsize=7, color=INK)
    ax.set_xlabel("records")
    ax.set_title(f"Most common label combinations (top {top}); multi-label is the norm")
    return save(fig, out / "label_combinations.png")


def age_by_class(meta, y, out: Path) -> Path:
    age = meta["age"].where(meta["age"] < 300)  # PTB-XL encodes >89 years as 300
    data = [age[y[c].astype(bool)].dropna().to_numpy() for c in CLASSES]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], CLASSES, strict=True):
        body.set_facecolor(CLASS_COLORS[c])
        body.set_alpha(0.6)
    parts["cmedians"].set_color(INK)
    ax.set_xticks(range(1, 6), CLASSES)
    ax.set_ylabel("age (years)")
    ax.set_title("Age by superclass (NORM is younger; >89 y excluded, coded 300)")
    return save(fig, out / "age_by_class.png")


def sex_by_class(meta, y, out: Path) -> Path:
    female = [(meta.loc[y[c].astype(bool), "sex"] == 1).mean() * 100 for c in CLASSES]
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(CLASSES, female, color="#CC79A7", label="female")
    ax.bar(CLASSES, [100 - f for f in female], bottom=female, color="#56B4E9", label="male")
    for i, f in enumerate(female):
        ax.text(i, f / 2, f"{f:.0f}%", ha="center", color="white", fontsize=8)
    ax.set_ylabel("% of records")
    ax.set_title("Sex composition by superclass")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    return save(fig, out / "sex_by_class.png")


def devices(meta, out: Path) -> Path:
    vc = meta["device"].astype(str).str.strip().value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(vc.index, vc.to_numpy(), color="#0072B2")
    ax.set_yscale("log")
    ax.set_ylabel("records (log)")
    ax.set_title("Recording device")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    return save(fig, out / "device_distribution.png")


def noise_annotations(meta, y, out: Path) -> Path:
    cols = [c for c in NOISE_COLS if c in meta]
    frac = pd.DataFrame(
        {c: [meta.loc[y[k].astype(bool), c].notna().mean() * 100 for k in CLASSES] for c in cols},
        index=CLASSES,
    )
    fig, ax = plt.subplots(figsize=(7, 3.2))
    frac.plot.bar(ax=ax, color=["#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#009E73"], width=0.8)
    ax.set_ylabel("records annotated (%)")
    ax.set_title("Signal-quality annotations recorded by PTB-XL technicians")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    return save(fig, out / "signal_quality_annotations.png")


def plot_ecg(ax_grid, sig: np.ndarray, color: str, shade: np.ndarray | None = None) -> None:
    """12-lead strip: sig (12, T) in mV; optional per-sample shade (12, T) in [0, 1]."""
    t = np.arange(sig.shape[1]) / FS
    for i, ax in enumerate(ax_grid):
        if shade is not None:
            ax.imshow(
                shade[i][None],
                aspect="auto",
                cmap="Reds",
                vmin=0,
                vmax=1,
                alpha=0.75,
                extent=(0, t[-1], sig[i].min() - 0.2, sig[i].max() + 0.2),
            )
        ax.plot(t, sig[i], color=color, lw=0.7)
        ax.set_ylabel(LEADS[i], rotation=0, labelpad=14, fontsize=8)
        ax.set_yticks([])
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.grid(axis="x", color="#f1c6c6", lw=0.4)
        if i < len(ax_grid) - 1:
            ax.set_xticks([])
    ax_grid[-1].set_xlabel("time (s)")


def example_ecgs(x, meta, y, splits, out: Path) -> list[Path]:
    single = _single_label(y)
    files = []
    for c in CLASSES:
        cand = np.flatnonzero(((single == c) & (splits == "train")).to_numpy())
        if not len(cand):
            continue
        row = cand[0]
        fig, axes = plt.subplots(12, 1, figsize=(8, 9), sharex=True)
        plot_ecg(axes, np.asarray(x[row]), CLASS_COLORS[c])
        fig.suptitle(
            f"Example {c} ECG (ecg_id {meta.index[row]}, train split, 100 Hz)", fontweight="bold"
        )
        files.append(save(fig, out / "examples" / f"example_ecg_{c}.png"))
    return files


def spectra(x, y, splits, out: Path, n: int = 400) -> Path:
    rng = np.random.default_rng(0)
    single = _single_label(y)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for c in CLASSES:
        idx = np.flatnonzero(((single == c) & (splits == "train")).to_numpy())
        idx = np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
        f, pxx = welch(np.asarray(x[idx, 1]), fs=FS, nperseg=256, axis=-1)
        ax.semilogy(f, pxx.mean(0), color=CLASS_COLORS[c], label=c, lw=1.2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD, lead II (mV²/Hz)")
    ax.set_title("Mean power spectrum by class (single-label train records)")
    ax.legend(frameon=False, ncols=5, fontsize=8)
    return save(fig, out / "power_spectrum_by_class.png")


def amplitude_heatmap(x, y, splits, out: Path, n: int = 400) -> Path:
    rng = np.random.default_rng(0)
    single = _single_label(y)
    rows = []
    for c in CLASSES:
        idx = np.flatnonzero(((single == c) & (splits == "train")).to_numpy())
        idx = np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
        xs = np.asarray(x[idx])
        rows.append(np.median(xs.max(-1) - xs.min(-1), axis=0))
    m = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    ax.set_xticks(range(12), LEADS)
    ax.set_yticks(range(5), CLASSES)
    for i in range(5):
        for j in range(12):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="white")
    fig.colorbar(im, ax=ax, label="median peak-to-peak (mV)")
    ax.set_title("Signal amplitude per lead and class (HYP: larger precordial voltages)")
    return save(fig, out / "lead_amplitude_by_class.png")


def global_12sl(feats: pd.DataFrame, y, out: Path) -> Path:
    single = _single_label(y)
    cols = [c for c in GLOBAL_12SL if c in feats]
    fig, axes = subplots(len(cols), ncols=3, w=3.3, h=2.6)
    for ax, col in zip(axes, cols, strict=False):
        data = [feats.loc[single == c, col].dropna().to_numpy() for c in CLASSES]
        lo, hi = np.nanpercentile(feats[col], [1, 99])
        data = [d[(d >= lo) & (d <= hi)] for d in data]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], CLASSES, strict=True):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.set_xticks(range(1, 6), CLASSES, fontsize=7)
        ax.set_title(GLOBAL_12SL[col], fontsize=9)
    fig.suptitle("GE 12SL global measurements by class (single-label records)", fontweight="bold")
    return save(fig, out / "12sl_measurements_by_class.png")


def external_comparison(y, splits, ext: pd.DataFrame, out: Path) -> list[Path]:
    files = []
    test = y[(splits == "test").to_numpy()].mean() * 100
    strict = ext.loc[ext.has_label, list(CLASSES)].mean() * 100
    lib = ext.loc[ext.has_label_liberal, [c + "_liberal" for c in CLASSES]].mean().to_numpy() * 100
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    w = 0.27
    xs = np.arange(5)
    ax.bar(xs - w, test.to_numpy(), w, color="#D55E00", label="PTB-XL test")
    ax.bar(
        xs,
        strict.to_numpy(),
        w,
        color="#009E73",
        label=f"external, strict map (n={int(ext.has_label.sum()):,})",
    )
    ax.bar(
        xs + w,
        lib,
        w,
        color="#56B4E9",
        label=f"external, liberal map (n={int(ext.has_label_liberal.sum()):,})",
    )
    ax.set_xticks(xs, CLASSES)
    ax.set_ylabel("prevalence (%)")
    ax.set_title("Label shift: PTB-XL (Germany) vs ecg-arrhythmia (China)")
    ax.legend(frameon=False, fontsize=7)
    files.append(save(fig, out / "external_vs_ptbxl_prevalence.png"))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(
        ext.age.dropna().clip(0, 100),
        bins=40,
        density=True,
        alpha=0.6,
        color="#009E73",
        label="external",
    )
    ax.set_xlabel("age (years)")
    ax.set_ylabel("density")
    ax.set_title("External cohort age distribution")
    ax.legend(frameon=False)
    files.append(save(fig, out / "external_age.png"))

    fig, ax = plt.subplots(figsize=(6, 2.8))
    reason = ext.excluded_reason.str.split(":").str[0].replace("", "kept")
    excl = reason.value_counts()
    ax.barh(
        excl.index.astype(str),
        excl.to_numpy(),
        color=["#009E73" if i == "kept" else MUTED for i in excl.index],
    )
    ax.set_xlabel("records")
    ax.set_xscale("log")
    ax.set_title("External records kept vs excluded (by reason, log scale)")
    files.append(save(fig, out / "external_exclusions.png"))
    return files


def make(cfg, out: Path) -> list[Path]:
    from ecgtrust.train_cli import load_labelled

    meta, y, splits = load_labelled(cfg)
    y = y.astype(int)
    files = [
        prevalence_by_split(y, splits, out),
        cooccurrence(y, out),
        combinations(y, out),
        age_by_class(meta, y, out),
        sex_by_class(meta, y, out),
        devices(meta, out),
        noise_annotations(meta, y, out),
    ]
    xf = cfg.artifacts / "ptbxl_x100.npy"
    if xf.exists():
        x = np.load(xf, mmap_mode="r")
        files += example_ecgs(x, meta, y, splits, out)
        files += [spectra(x, y, splits, out), amplitude_heatmap(x, y, splits, out)]
    try:
        from ecgtrust.features.ptbxl_plus import load_features

        feats = load_features(cfg.dataset_dir("ptbxl_plus"), "12sl").reindex(meta.index)
        files.append(global_12sl(feats, y, out))
    except FileNotFoundError:
        pass
    ef = cfg.artifacts / "external_meta.parquet"
    if ef.exists():
        files += external_comparison(y, splits, pd.read_parquet(ef), out)
    return files
