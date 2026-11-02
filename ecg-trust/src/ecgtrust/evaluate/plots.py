"""Report figures (matplotlib, Agg). Colour-blind-safe palette, one colour per class."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from ecgtrust import CLASSES  # noqa: E402
from ecgtrust.calibrate.metrics import reliability_curve  # noqa: E402

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
plt.rcParams.update(
    {"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def roc_plot(y: np.ndarray, p: np.ndarray, aurocs, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for c, name in enumerate(CLASSES):
        if np.unique(y[:, c]).size == 2:
            fpr, tpr, _ = roc_curve(y[:, c], p[:, c])
            ax.plot(fpr, tpr, color=PALETTE[c], lw=1.6, label=f"{name} ({aurocs[c]:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=title)
    ax.legend(loc="lower right", frameon=False, title="AUROC")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def reliability_plot(
    y: np.ndarray, p: np.ndarray, eces, path: Path, title: str, n_bins: int = 15
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for c, name in enumerate(CLASSES):
        mp, obs, _ = reliability_curve(y[:, c], p[:, c], n_bins)
        ax.plot(
            mp, obs, marker="o", ms=3, color=PALETTE[c], lw=1.4, label=f"{name} (ECE {eces[c]:.3f})"
        )
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed frequency",
        title=title,
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
