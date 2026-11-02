"""Shared figure style (matches the guide) and small helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, ACCENT = "#1f2a44", "#8a94a6", "#0072B2"
# Okabe-Ito: colour-blind safe, one colour per diagnostic superclass.
CLASS_COLORS = {
    "NORM": "#0072B2",
    "MI": "#D55E00",
    "STTC": "#009E73",
    "CD": "#CC79A7",
    "HYP": "#E69F00",
}
RUN_COLORS = {"ResNet": "#0072B2", "Fusion": "#D55E00", "External": "#009E73", "12SL": "#8a94a6"}
SPLIT_COLORS = {
    "train": "#0072B2",
    "val_select": "#56B4E9",
    "val_calib": "#009E73",
    "test": "#D55E00",
}

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
        "savefig.dpi": 160,
    }
)
import logging  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def subplots(n: int, ncols: int = 3, w: float = 3.4, h: float = 2.8, **kw):
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat
