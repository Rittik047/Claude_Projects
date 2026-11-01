"""Binary calibration metrics with equal-mass (quantile) bins."""

from __future__ import annotations

import numpy as np


def _bins(p: np.ndarray, n_bins: int) -> list[np.ndarray]:
    order = np.argsort(p, kind="stable")
    return [b for b in np.array_split(order, n_bins) if len(b)]


def reliability_curve(y, p, n_bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean predicted, observed frequency, count) per equal-mass bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = _bins(p, n_bins)
    return (
        np.array([p[b].mean() for b in bins]),
        np.array([y[b].mean() for b in bins]),
        np.array([len(b) for b in bins]),
    )


def ece(y, p, n_bins: int = 15) -> float:
    mp, obs, cnt = reliability_curve(y, p, n_bins)
    return float(np.sum(cnt * np.abs(mp - obs)) / cnt.sum())


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
