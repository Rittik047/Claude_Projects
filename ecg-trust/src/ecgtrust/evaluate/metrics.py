"""Discrimination metrics and a patient-level (cluster) bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _per_class(fn, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim == 1:
        y, p = y[:, None], p[:, None]
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if np.unique(y[:, c]).size == 2:
            out[c] = fn(y[:, c], p[:, c])
    return out


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """AUROC per column; NaN where a column has a single label value."""
    return _per_class(roc_auc_score, y, p)


def per_class_auprc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return _per_class(average_precision_score, y, p)


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    auc = per_class_auroc(y, p)
    return float(np.nanmean(auc)) if np.isfinite(auc).any() else float("nan")


def bootstrap_samples(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Metric over `n` resamples of whole groups (patients) drawn with replacement."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for b in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        vals[b] = metric(y[idx], p[idx])
    return vals


def patient_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, 2.5 %, 97.5 %) percentile CI from a patient-level bootstrap."""
    vals = bootstrap_samples(y, p, groups, metric, n, seed)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(metric(y, p)), float(lo), float(hi)
