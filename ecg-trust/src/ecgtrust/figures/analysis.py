"""02-06, 08: training, performance, calibration, abstention, robustness, comparator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ecgtrust import CLASSES, LEADS
from ecgtrust.calibrate.conformal import coverage, decide
from ecgtrust.calibrate.metrics import ece, reliability_curve
from ecgtrust.evaluate.metrics import per_class_auroc
from ecgtrust.figures.runs import load_runs, val_select
from ecgtrust.figures.style import (
    CLASS_COLORS,
    INK,
    MUTED,
    RUN_COLORS,
    plt,
    save,
    subplots,
)

SEED_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def _run_color(name: str) -> str:
    return next((c for k, c in RUN_COLORS.items() if k.lower() in name.lower()), MUTED)


# ---------------------------------------------------------------- 02_training
def training(cfg, out: Path) -> list[Path]:
    files = []
    logs = {
        s: pd.read_json(cfg.artifacts / "models" / f"seed{s}" / "train_log.jsonl", lines=True)
        for s in cfg.seeds
        if (cfg.artifacts / "models" / f"seed{s}" / "train_log.jsonl").exists()
    }
    if logs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
        for (s, d), col in zip(logs.items(), SEED_COLORS, strict=False):
            a1.plot(d.epoch, d.train_loss, color=col, label=f"seed {s}")
            a2.plot(d.epoch, d.val_select_macro_auroc, color=col, label=f"seed {s}")
            b = d.val_select_macro_auroc.idxmax()
            a2.scatter(d.epoch[b], d.val_select_macro_auroc[b], color=col, zorder=3, s=25)
        a1.set(xlabel="epoch", ylabel="training BCE loss", title="Training loss")
        a2.set(
            xlabel="epoch",
            ylabel="val_select macro-AUROC",
            title="Validation AUROC (dot = checkpoint kept)",
        )
        a1.legend(frameon=False)
        files.append(save(fig, out / "learning_curves.png"))

        fig, ax = plt.subplots(figsize=(5, 3))
        for (s, d), col in zip(logs.items(), SEED_COLORS, strict=False):
            ax.plot(d.epoch, d.seconds, color=col, label=f"seed {s}")
        ax.set(xlabel="epoch", ylabel="seconds / epoch", title="Compute: CPU epoch time")
        ax.legend(frameon=False)
        files.append(save(fig, out / "epoch_time.png"))

    f = cfg.reports / "val" / "resnet_summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for i, split in enumerate(("val_select", "val_calib")):
            m = s[split]["members"]
            ax.scatter([i] * len(m), m, color=SEED_COLORS[: len(m)], s=30, zorder=3)
            ax.scatter(i, s[split]["ensemble"], marker="D", s=60, color=INK, zorder=4)
        ax.set_xticks([0, 1], ["val_select", "val_calib"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("macro-AUROC")
        ax.set_title("Deep ensemble beats every member (◆ ensemble, ● seeds)")
        files.append(save(fig, out / "ensemble_vs_members.png"))

    f = cfg.artifacts / "models" / "fusion.json"
    if f.exists():
        g = json.loads(f.read_text())
        w = np.array([float(k) for k in g["grid_macro_auroc"]])
        a = np.array(list(g["grid_macro_auroc"].values()))
        fig, ax = plt.subplots(figsize=(5.5, 3))
        ax.plot(w, a, "o-", color="#D55E00")
        ax.axvline(g["w"], color=MUTED, ls="--")
        ax.text(g["w"], a.min(), f"  chosen w = {g['w']}", color=INK, fontsize=8)
        ax.set(
            xlabel="weight on ResNet logit (w)",
            ylabel="val_select macro-AUROC",
            title="Fusion weight: w=0 is 12SL features only, w=1 is ResNet only",
        )
        files.append(save(fig, out / "fusion_weight_selection.png"))

    vs = val_select(cfg)
    if vs:
        fig, ax = plt.subplots(figsize=(7.5, 3.3))
        width = 0.8 / len(vs)
        cols = ["#8a94a6", "#56B4E9", "#0072B2", "#D55E00"]
        for i, ((name, (y, p)), col) in enumerate(zip(vs.items(), cols, strict=False)):
            auc = per_class_auroc(y, p)
            xs = np.arange(5) + (i - (len(vs) - 1) / 2) * width
            ax.bar(xs, auc, width, color=col, label=f"{name} ({np.nanmean(auc):.3f})")
        ax.set_xticks(range(5), CLASSES)
        ax.set_ylim(0.85, 0.98)
        ax.set_ylabel("AUROC (val_select)")
        ax.set_title("Model selection on validation (legend: macro-AUROC)")
        ax.legend(frameon=False, fontsize=7, ncols=2)
        files.append(save(fig, out / "model_comparison_val.png"))
    return files


# ------------------------------------------------------------- 03_performance
def _curves(run, out: Path, kind: str) -> Path:
    fig, axes = subplots(5, ncols=5, w=2.5, h=2.6)
    for c, ax in enumerate(axes):
        y, p = run.y[:, c], run.p[:, c]
        if np.unique(y).size < 2:
            ax.set_title(f"{CLASSES[c]} (no positives)")
            continue
        if kind == "roc":
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=CLASS_COLORS[CLASSES[c]])
            ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=0.8)
            ax.set(xlabel="1 − specificity", ylabel="sensitivity" if c == 0 else None)
            ax.set_title(f"{CLASSES[c]}  AUROC {per_class_auroc(y[:, None], p[:, None])[0]:.3f}")
        else:
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=CLASS_COLORS[CLASSES[c]])
            ax.axhline(y.mean(), color=MUTED, ls=":", lw=0.8)
            ax.set(xlabel="recall", ylabel="precision" if c == 0 else None)
            ax.set_title(f"{CLASSES[c]}  prevalence {y.mean():.2f}")
        ax.set_aspect("equal")
    fig.suptitle(
        f"{'ROC' if kind == 'roc' else 'Precision–recall'} curves — {run.name}", fontweight="bold"
    )
    return save(fig, out / f"{kind}_{run.key}.png")


def _forest(cfg, out: Path) -> Path | None:
    rows = []
    for key, label in (
        ("test", "ResNet · test"),
        ("test_fusion", "Fusion · test"),
        ("external", "ResNet · external"),
    ):
        f = cfg.reports / key / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            for r in m["discrimination"]["per_class"]:
                rows.append((label, r["class"], r["auroc"], r["lo"], r["hi"]))
            mm = m["discrimination"]["macro_auroc"]
            rows.append((label, "macro", mm["point"], mm["lo"], mm["hi"]))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["run", "class", "auc", "lo", "hi"]).dropna()
    runs = list(dict.fromkeys(df.run))
    cats = [*CLASSES, "macro"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(runs):
        d = df[df.run == r].set_index("class").reindex(cats)
        ys = np.arange(len(cats)) + (i - (len(runs) - 1) / 2) * 0.22
        ax.errorbar(
            d.auc,
            ys,
            xerr=[d.auc - d.lo, d.hi - d.auc],
            fmt="o",
            ms=4,
            capsize=2,
            color=_run_color(r),
            label=r,
        )
    ax.axvline(0.93, color=MUTED, ls=":", lw=0.8)
    ax.text(0.93, len(cats) - 0.4, " published ≈0.93", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC with 95% patient-bootstrap CI")
    ax.set_title("Discrimination per class: internal vs external")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    return save(fig, out / "auroc_forest_all_runs.png")


def _score_hist(run, out: Path) -> Path:
    fig, axes = subplots(5, ncols=5, w=2.5, h=2.3)
    for c, ax in enumerate(axes):
        y = run.y[:, c].astype(bool)
        bins = np.linspace(0, 1, 26)
        ax.hist(run.p[~y, c], bins, color=MUTED, alpha=0.7, density=True, label="absent")
        ax.hist(
            run.p[y, c],
            bins,
            color=CLASS_COLORS[CLASSES[c]],
            alpha=0.7,
            density=True,
            label="present",
        )
        ax.set_yscale("log")
        ax.set_title(CLASSES[c])
        ax.set_xlabel("calibrated probability")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Score separation — {run.name}", fontweight="bold")
    return save(fig, out / f"score_distributions_{run.key}.png")


def _subgroups(cfg, key: str, name: str, out: Path) -> Path | None:
    f = cfg.reports / key / "subgroups.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[(df.status == "ok") & df.lo.notna()]
    cols = list(dict.fromkeys(df.column))
    fig, axes = subplots(len(cols), ncols=min(3, len(cols)), w=3.6, h=3)
    for ax, col in zip(axes, cols, strict=False):
        d = df[df.column == col]
        groups = list(dict.fromkeys(d.group))
        for j, c in enumerate(CLASSES):
            dc = d[d["class"] == c].set_index("group").reindex(groups)
            ys = np.arange(len(groups)) + (j - 2) * 0.14
            ax.errorbar(
                dc.auroc,
                ys,
                xerr=[dc.auroc - dc.lo, dc.hi - dc.auroc],
                fmt="o",
                ms=3,
                capsize=1.5,
                color=CLASS_COLORS[c],
                label=c,
                lw=0.8,
            )
        ax.set_yticks(range(len(groups)), [str(g) for g in groups], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(col.replace("_label", "").replace("_", " "))
        ax.set_xlabel("AUROC")
    axes[0].legend(frameon=False, fontsize=6, ncols=2)
    fig.suptitle(f"Subgroup AUROC (95% CI) — {name}", fontweight="bold")
    return save(fig, out / f"subgroups_{key}.png")


def _decision_matrix(run, out: Path, alpha: str) -> Path:
    d = decide(run.p, run.thr[alpha])
    fig, axes = subplots(5, ncols=5, w=2.3, h=2.2)
    for c, ax in enumerate(axes):
        y = run.y[:, c].astype(bool)
        m = np.array(
            [
                [(d[y == t, c] == k).sum() for k in ("present", "refer", "absent")]
                for t in (True, False)
            ]
        )
        ax.imshow(m / m.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if m[i, j] / m[i].sum() > 0.6 else INK,
                )
        ax.set_xticks(range(3), ["present", "refer", "absent"], fontsize=7)
        ax.set_yticks([0, 1], ["true +", "true −"], fontsize=7)
        ax.set_title(CLASSES[c])
    fig.suptitle(f"Three-way decisions vs truth at α={alpha} — {run.name}", fontweight="bold")
    return save(fig, out / f"decision_matrix_{run.key}_alpha{alpha}.png")


# ------------------------------------------------------------- 04_calibration
def _reliability_grid(run, cfg, out: Path) -> Path:
    fig, axes = subplots(5, ncols=5, w=2.5, h=2.6)
    for c, ax in enumerate(axes):
        y = run.y[:, c]
        for p, col, lab in (
            (run.p_pre[:, c], MUTED, "before"),
            (run.p[:, c], CLASS_COLORS[CLASSES[c]], "after"),
        ):
            mp, fo, _ = reliability_curve(y, p, cfg.ece_bins)
            ax.plot(
                mp,
                fo,
                "o-",
                ms=2.5,
                color=col,
                lw=1,
                label=f"{lab} ECE {ece(y, p, cfg.ece_bins):.3f}",
            )
        ax.plot([0, 1], [0, 1], color=INK, ls=":", lw=0.7)
        ax.set_title(CLASSES[c])
        ax.set_xlabel("predicted")
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_aspect("equal")
    axes[0].set_ylabel("observed frequency")
    fig.suptitle(f"Reliability before vs after temperature scaling — {run.name}", fontweight="bold")
    return save(fig, out / f"reliability_{run.key}.png")


def _ece_bars(runs, cfg, out: Path) -> Path:
    fig, axes = subplots(len(runs), ncols=len(runs), w=3.3, h=2.8)
    for ax, run in zip(axes, runs, strict=False):
        pre = [ece(run.y[:, c], run.p_pre[:, c], cfg.ece_bins) for c in range(5)]
        post = [ece(run.y[:, c], run.p[:, c], cfg.ece_bins) for c in range(5)]
        xs = np.arange(5)
        ax.bar(xs - 0.2, pre, 0.4, color=MUTED, label="before")
        ax.bar(xs + 0.2, post, 0.4, color=_run_color(run.name), label="after")
        ax.set_xticks(xs, CLASSES, fontsize=7)
        ax.set_title(run.name, fontsize=9)
        ax.set_ylabel("ECE (15 equal-mass bins)")
    axes[0].legend(frameon=False, fontsize=7)
    return save(fig, out / "ece_per_class.png")


def _temperatures(cfg, out: Path) -> Path | None:
    vals = {}
    for name, d in (
        ("ResNet", cfg.artifacts / "calibration"),
        ("Fusion", cfg.artifacts / "calibration" / "fusion"),
    ):
        if (d / "T.npy").exists():
            vals[name] = np.load(d / "T.npy")
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3))
    for i, (name, t) in enumerate(vals.items()):
        ax.bar(np.arange(5) + (i - 0.5) * 0.38, t, 0.38, color=RUN_COLORS[name], label=name)
    ax.axhline(1, color=INK, ls=":", lw=0.8)
    ax.set_xticks(range(5), CLASSES)
    ax.set_ylabel("temperature T")
    ax.set_title("Per-class temperatures (T>1 softens over-confidence)")
    ax.legend(frameon=False, fontsize=8)
    return save(fig, out / "temperatures.png")


# -------------------------------------------------------------- 05_abstention
def _coverage(runs, cfg, out: Path) -> list[Path]:
    files = []
    fig, axes = subplots(len(runs), ncols=len(runs), w=3.3, h=3)
    for ax, run in zip(axes, runs, strict=False):
        for c in range(5):
            cov = [coverage(run.y, run.p, run.thr[str(a)])["cov_pos"][c] for a in cfg.alphas]
            ax.plot(
                [1 - a for a in cfg.alphas],
                cov,
                "o-",
                color=CLASS_COLORS[CLASSES[c]],
                label=CLASSES[c],
                ms=4,
            )
        lo, hi = 1 - max(cfg.alphas) - 0.05, 1
        ax.plot([lo, hi], [lo, hi], color=INK, ls=":", lw=0.8)
        ax.set(
            xlabel="target coverage 1 − α",
            ylabel="empirical coverage (positives)",
            title=run.name,
            xlim=(lo, hi),
            ylim=(lo - 0.05, hi),
        )
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Label-conditional conformal coverage (on the diagonal = guarantee met)", fontweight="bold"
    )
    files.append(save(fig, out / "coverage_vs_target.png"))

    fig, axes = subplots(len(runs), ncols=len(runs), w=3.3, h=2.8)
    for ax, run in zip(axes, runs, strict=False):
        for i, a in enumerate(cfg.alphas):
            rr = coverage(run.y, run.p, run.thr[str(a)])["refer_rate"]
            ax.bar(
                np.arange(5) + (i - 0.5) * 0.38,
                rr * 100,
                0.38,
                color=["#E69F00", "#0072B2"][i % 2],
                label=f"α={a}",
            )
        ax.set_xticks(range(5), CLASSES, fontsize=7)
        ax.set(ylabel="referred to clinician (%)", title=run.name)
    axes[0].legend(frameon=False, fontsize=7)
    files.append(save(fig, out / "refer_rate.png"))

    a = str(max(cfg.alphas))
    fig, axes = subplots(len(runs), ncols=len(runs), w=3.3, h=2.8)
    for ax, run in zip(axes, runs, strict=False):
        d = decide(run.p, run.thr[a])
        bottom = np.zeros(5)
        for k, col in (("present", "#D55E00"), ("refer", "#E69F00"), ("absent", "#0072B2")):
            frac = (d == k).mean(0) * 100
            ax.bar(CLASSES, frac, bottom=bottom, color=col, label=k)
            bottom += frac
        ax.set(ylabel="% of ECGs", title=run.name)
        ax.tick_params(axis="x", labelsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Decision mix at α={a}", fontweight="bold")
    files.append(save(fig, out / "decision_mix.png"))

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for run in runs:
        for al in cfg.alphas:
            cv = coverage(run.y, run.p, run.thr[str(al)])
            ax.scatter(
                cv["refer_rate"] * 100,
                cv["acc_non_referred"] * 100,
                s=30,
                color=_run_color(run.name),
                marker="o" if al == max(cfg.alphas) else "s",
                label=f"{run.name}, α={al}",
            )
    ax.set(
        xlabel="refer rate (%)",
        ylabel="accuracy on non-referred (%)",
        title="Trade-off: referring more buys accuracy on the rest",
    )
    ax.legend(frameon=False, fontsize=6)
    files.append(save(fig, out / "accuracy_vs_refer.png"))
    return files


# -------------------------------------------------------------- 06_robustness
def robustness(cfg, out: Path) -> list[Path]:
    files = []
    f = cfg.reports / "test" / "robustness.csv"
    if f.exists():
        df = pd.read_csv(f)
        clean = df[df.noise == "clean"].iloc[0]
        d = df[df.noise != "clean"].copy()
        d["snr"] = d.level.astype(float)
        names = {"bw": "baseline wander", "emg": "muscle (EMG)", "em": "electrode motion"}
        cols = {"bw": "#0072B2", "emg": "#D55E00", "em": "#009E73"}
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for metric, ax, lab in (
            ("macro_auroc", axes[0], "macro-AUROC"),
            ("macro_ece", axes[1], "macro ECE"),
            ("mean_refer_rate", axes[2], "mean refer rate"),
        ):
            for n, g in d.groupby("noise"):
                ax.plot(g.snr, g[metric], "o-", ms=3, color=cols.get(n), label=names.get(n, n))
            ax.axhline(clean[metric], color=MUTED, ls=":", lw=0.8)
            ax.invert_xaxis()
            ax.set(xlabel="SNR (dB) → noisier", ylabel=lab)
        axes[0].legend(frameon=False, fontsize=7)
        fig.suptitle(
            "Robustness to synthetic noise (internal test, ResNet; dotted = clean)",
            fontweight="bold",
        )
        files.append(save(fig, out / "noise_robustness_panels.png"))
    f = cfg.reports / "test" / "occlusion.csv"
    if f.exists():
        df = pd.read_csv(f).pivot(index="class", columns="lead", values="auroc_drop")
        df = df.reindex(index=list(CLASSES), columns=list(LEADS))
        fig, ax = plt.subplots(figsize=(8, 2.8))
        lim = np.nanmax(np.abs(df.to_numpy()))
        im = ax.imshow(df.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(5):
            for j in range(12):
                ax.text(j, i, f"{df.iat[i, j]:.3f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(12), LEADS)
        ax.set_yticks(range(5), CLASSES)
        fig.colorbar(im, ax=ax, label="AUROC drop when lead zeroed")
        ax.set_title("Lead occlusion (internal test): which leads each class depends on")
        files.append(save(fig, out / "lead_occlusion_heatmap.png"))
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ax.bar(LEADS, df.mean(0).to_numpy(), color="#0072B2")
        ax.set(ylabel="mean AUROC drop", title="Average reliance on each lead")
        files.append(save(fig, out / "lead_occlusion_mean.png"))
    for name in ("robustness.png", "occlusion.png"):
        src = cfg.reports / "test" / name
        if src.exists():
            files.append(Path(shutil.copy(src, out / f"evaluate_{name}")))
    return files


# ------------------------------------------------------------- 08_comparator
def comparator(cfg, out: Path) -> list[Path]:
    files = []
    for key, label in (("test", "ResNet"), ("test_fusion", "Fusion")):
        f = cfg.reports / key / "comparator.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        xs = np.arange(len(df))
        ax.bar(xs - 0.2, df["12sl_sens"], 0.4, color=MUTED, label="GE 12SL statements")
        err = [
            df.model_sens_at_12sl_spec - (df["12sl_sens"] + df.lo),
            (df["12sl_sens"] + df.hi) - df.model_sens_at_12sl_spec,
        ]
        ax.bar(
            xs + 0.2,
            df.model_sens_at_12sl_spec,
            0.4,
            color=RUN_COLORS[label],
            yerr=np.clip(err, 0, None),
            capsize=2,
            label=f"{label} at 12SL's specificity",
        )
        for i, r in df.iterrows():
            ax.text(
                i,
                max(r["12sl_sens"], r.model_sens_at_12sl_spec) + 0.06,
                f"spec {r['12sl_spec']:.2f}\nΔ {r['diff']:+.2f}",
                ha="center",
                fontsize=6.5,
            )
        ax.set_xticks(xs, df["class"])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("sensitivity")
        ax.set_title(f"Matched-specificity comparison with GE 12SL — {label}, internal test")
        ax.legend(frameon=False, fontsize=7, loc="lower right")
        files.append(save(fig, out / f"vs_12sl_{key}.png"))
    return files


def make(cfg, out_root: Path, groups) -> dict[str, list[Path]]:
    made: dict[str, list[Path]] = {}
    if "02_training" in groups:
        made["02_training"] = training(cfg, out_root / "02_training")
    runs = load_runs(cfg)
    if "03_performance" in groups:
        out = out_root / "03_performance"
        files = [f for f in [_forest(cfg, out)] if f]
        for r in runs:
            files += [_curves(r, out, "roc"), _curves(r, out, "pr"), _score_hist(r, out)]
            files += [f for f in [_subgroups(cfg, r.key, r.name, out)] if f]
            files += [_decision_matrix(r, out, str(max(cfg.alphas)))]
        made["03_performance"] = files
    if "04_calibration" in groups:
        out = out_root / "04_calibration"
        files = [_reliability_grid(r, cfg, out) for r in runs]
        files += [_ece_bars(runs, cfg, out)] if runs else []
        files += [f for f in [_temperatures(cfg, out)] if f]
        made["04_calibration"] = files
    if "05_abstention" in groups and runs:
        made["05_abstention"] = _coverage(runs, cfg, out_root / "05_abstention")
    if "06_robustness" in groups:
        (out_root / "06_robustness").mkdir(parents=True, exist_ok=True)
        made["06_robustness"] = robustness(cfg, out_root / "06_robustness")
    if "08_comparator" in groups:
        made["08_comparator"] = comparator(cfg, out_root / "08_comparator")
    return made
