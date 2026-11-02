"""Shared figure style: validated reference palette (dataviz skill), thin marks, recessive axes.

Categorical slots are assigned in FIXED order and follow the entity (e.g. model P0 is always
slot 1), never the rank. Sequential = blue ramp; diverging = blue <-> red with a gray midpoint.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("bmf_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("bmf_div", ["#256abf", "#86b6ef", "#f0efec", "#ec8f8f", "#c33636"])
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# entity -> fixed slot (colour follows the entity everywhere in the paper)
MODEL_COLORS = {"P0": CAT[0], "P1": CAT[1], "P2": CAT[2], "P3": CAT[3], "P4": CAT[4], "P5": CAT[6], "O1": CAT[5]}
PRIMARY_ORDER = ["lung", "breast", "melanoma", "gastrointestinal", "renal", "prostate", "other"]
PRIMARY_COLORS = {p: CAT[i] for i, p in enumerate(PRIMARY_ORDER)}
RANO_COLORS = {"CR": CAT[2], "PR": CAT[0], "SD": "#b9b8b0", "PD": CAT[7]}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.family": "DejaVu Sans", "figure.dpi": 100, "text.color": INK,
    })


def fig(w=5.0, h=3.4, **kw):
    apply_style()
    return plt.subplots(figsize=(w, h), constrained_layout=True, **kw)


def save(f, out: Path | str) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f.savefig(out.with_suffix(".png"), dpi=300)
    f.savefig(out.with_suffix(".pdf"))
    plt.close(f)
    return out.with_suffix(".png")


def bar_kw():
    return dict(edgecolor=SURFACE, linewidth=2)        # 2px surface gap between adjacent fills
