"""Per-class temperature scaling of the seed ensemble (Guo et al. 2017), fit on val_calib."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ecgtrust.models.ensemble import ensemble_proba

_EPS = 1e-7


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """logits (M, N, C), y (N, C) → T (C,), minimising the NLL of the ensemble probability."""
    y = np.asarray(y, dtype=float)
    temps = np.ones(logits.shape[-1])
    for c in range(logits.shape[-1]):
        lc = logits[..., c : c + 1]

        def obj(log_t, lc=lc, c=c):
            return _nll(y[:, c], ensemble_proba(lc, np.array([np.exp(log_t)]))[:, 0])

        res = minimize_scalar(obj, bounds=(np.log(0.05), np.log(20)), method="bounded")
        temps[c] = float(np.exp(res.x))
    return temps
