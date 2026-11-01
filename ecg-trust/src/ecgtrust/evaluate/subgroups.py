"""Descriptive subgroup metrics with patient-level bootstrap CIs (spec §5.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.calibrate.metrics import ece
from ecgtrust.evaluate.metrics import bootstrap_samples, per_class_auroc

BINARY_FLAG_COLUMNS = {
    "static_noise",
    "baseline_drift",
    "burst_noise",
    "electrodes_problems",
    "pacemaker",
}
DEFAULT_COLUMNS = ["sex_label", "age_band", "device", *sorted(BINARY_FLAG_COLUMNS)]


def _group_values(meta: pd.DataFrame, col: str) -> pd.Series:
    if col in BINARY_FLAG_COLUMNS:
        return meta[col].notna().map({True: "yes", False: "no"})
    v = meta[col].astype(str)
    return v.str.strip() if col == "device" else v


def _auroc(y, p):
    return float(per_class_auroc(y, p)[0])


def subgroup_table(
    meta: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    columns: list[str],
    groups: np.ndarray,
    min_pos: int = 30,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """One row per (column, group, class). Metrics are NaN with status 'insufficient' when the
    group has fewer than `min_pos` positives or negatives for that class."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows = []
    for col in columns:
        if col not in meta:
            continue
        vals = _group_values(meta.reset_index(drop=True), col).to_numpy()
        for g in sorted(set(vals)):
            m = vals == g
            for c, name in enumerate(CLASSES):
                yc, pc = y[m, c], p[m, c]
                n_pos = int(yc.sum())
                row = {
                    "column": col,
                    "group": g,
                    "class": name,
                    "n": int(m.sum()),
                    "n_pos": n_pos,
                    "auroc": np.nan,
                    "lo": np.nan,
                    "hi": np.nan,
                    "ece": np.nan,
                    "status": "insufficient",
                }
                if n_pos >= min_pos and m.sum() - n_pos >= min_pos:
                    bs = bootstrap_samples(yc, pc, groups[m], _auroc, n=n_boot)
                    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                    row.update(
                        auroc=_auroc(yc, pc),
                        lo=float(lo),
                        hi=float(hi),
                        ece=ece(yc, pc),
                        status="ok",
                    )
                rows.append(row)
    return pd.DataFrame(rows)
