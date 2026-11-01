"""Robustness curves: AUROC, ECE and conformal refer rate versus noise level."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecgtrust.calibrate.conformal import ConformalThresholds, decide  # noqa: E402
from ecgtrust.calibrate.metrics import ece  # noqa: E402
from ecgtrust.evaluate.metrics import macro_auroc  # noqa: E402
from ecgtrust.models.ensemble import ensemble_proba  # noqa: E402
from ecgtrust.robustness.noise import GENERATORS, add_noise_at_snr, lead_dropout  # noqa: E402

LABELS = {"bw": "baseline wander", "emg": "muscle (EMG-like)", "em": "electrode motion"}


def robustness_curve(
    x: np.ndarray,
    y: np.ndarray,
    predict_logits: Callable[[np.ndarray], np.ndarray],
    temps: np.ndarray,
    thr: ConformalThresholds,
    snr_db=(24, 18, 12, 6, 0, -6),
    noise_types=("bw", "emg", "em"),
    dropout_k=(1, 2, 4),
    seed: int = 0,
    fs: int = 100,
) -> pd.DataFrame:
    def row(noise, level, xn):
        p = ensemble_proba(predict_logits(xn), temps)
        return {
            "noise": noise,
            "level": str(level),
            "macro_auroc": macro_auroc(y, p),
            "macro_ece": float(np.mean([ece(y[:, c], p[:, c]) for c in range(y.shape[1])])),
            "mean_refer_rate": float((decide(p, thr) == "refer").mean()),
        }

    rows = [row("clean", "clean", x)]
    for nt in noise_types:
        gen = GENERATORS[nt]
        for snr in snr_db:
            rng = np.random.default_rng(seed)
            xn = np.stack([add_noise_at_snr(xi, gen(xi.shape, fs, rng), snr) for xi in x])
            rows.append(row(nt, snr, xn.astype(np.float32)))
    for k in dropout_k:
        rng = np.random.default_rng(seed)
        rows.append(row("dropout", k, np.stack([lead_dropout(xi, k, rng) for xi in x])))
    return pd.DataFrame(rows)


def robustness_plot(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    clean = df[df.noise == "clean"].iloc[0]
    colours = {"bw": "#0072B2", "emg": "#D55E00", "em": "#009E73"}
    for nt, c in colours.items():
        d = df[df.noise == nt]
        if d.empty:
            continue
        snr = d.level.astype(float)
        axes[0].plot(snr, d.macro_auroc, marker="o", color=c, label=LABELS[nt])
        axes[1].plot(snr, d.mean_refer_rate, marker="o", color=c, label=LABELS[nt])
    axes[0].axhline(clean.macro_auroc, ls=":", c="grey", label="clean")
    axes[1].axhline(clean.mean_refer_rate, ls=":", c="grey", label="clean")
    for ax, lab in zip(axes, ["macro-AUROC", "conformal refer rate (α = 0.10)"], strict=True):
        ax.set(xlabel="SNR (dB)", ylabel=lab)
        ax.invert_xaxis()
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Robustness to synthetic noise (lower SNR = noisier, to the right)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _split_signals(cfg, ctx) -> np.ndarray:
    idx = pd.read_parquet(cfg.artifacts / "ptbxl_index.parquet")["ecg_id"].to_numpy()
    pos = pd.Series(np.arange(len(idx)), index=idx).loc[ctx["meta"].index].to_numpy()
    x = np.load(cfg.artifacts / "ptbxl_x100.npy", mmap_mode="r")
    order = np.argsort(pos)
    out = np.empty((len(pos), 12, x.shape[-1]), np.float32)
    out[order] = x[pos[order]]
    return out


def analysis(cfg, split: str, ctx: dict) -> dict | None:
    if split not in ("val_select", "test"):
        return None
    from ecgtrust import predict
    from ecgtrust.models.ensemble import checkpoint_paths

    ckpts = checkpoint_paths(cfg)
    thr = ctx["thr"][str(max(cfg.alphas))]
    df = robustness_curve(
        _split_signals(cfg, ctx),
        ctx["y"],
        lambda xb: predict.predict_logits(ckpts, xb),
        ctx["temps"],
        thr,
        snr_db=cfg.snr_db,
    )
    out = cfg.reports / split
    df.to_csv(out / "robustness.csv", index=False)
    robustness_plot(df, out / "robustness.png")
    return df.to_dict(orient="records")


ANALYSIS = ("robustness", analysis)
