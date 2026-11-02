"""Label-conditional (Mondrian) split conformal abstention, independently per class.

For class c the candidate values are {present, absent}. Non-conformity scores are
1 - p for positives and p for negatives; each gets its own finite-sample quantile, so
P(true value in set | Y = y) >= 1 - alpha for y in {0, 1} under exchangeability
(Angelopoulos & Bates 2021). A singleton set is a decision; an empty or two-element set
is "refer". If a calibration group is too small, its quantile is +inf: that value can then
never be excluded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ConformalThresholds:
    q_present: np.ndarray
    q_absent: np.ndarray
    alpha: float

    def to_json(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    @classmethod
    def from_json(cls, d: dict) -> ConformalThresholds:
        return cls(np.asarray(d["q_present"], float), np.asarray(d["q_absent"], float), d["alpha"])


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if 1 <= k <= n else float("inf")


def fit_conformal(p: np.ndarray, y: np.ndarray, alpha: float) -> ConformalThresholds:
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    qp = [_quantile(1 - p[y[:, c], c], alpha) for c in range(p.shape[1])]
    qa = [_quantile(p[~y[:, c], c], alpha) for c in range(p.shape[1])]
    return ConformalThresholds(np.array(qp), np.array(qa), alpha)


def _sets(p: np.ndarray, thr: ConformalThresholds) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, float)
    return (1 - p) <= thr.q_present[None, :], p <= thr.q_absent[None, :]


def decide(p: np.ndarray, thr: ConformalThresholds) -> np.ndarray:
    has_p, has_a = _sets(p, thr)
    out = np.full(has_p.shape, "refer", dtype="<U7")
    out[has_p & ~has_a] = "present"
    out[has_a & ~has_p] = "absent"
    return out


def coverage(y: np.ndarray, p: np.ndarray, thr: ConformalThresholds) -> dict[str, np.ndarray]:
    y = np.asarray(y).astype(bool)
    has_p, has_a = _sets(p, thr)
    d = decide(p, thr)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_pos = (has_p & y).sum(0) / y.sum(0)
        cov_neg = (has_a & ~y).sum(0) / (~y).sum(0)
        decided = d != "refer"
        correct = ((d == "present") & y) | ((d == "absent") & ~y)
        acc = correct.sum(0) / decided.sum(0)
    return {
        "cov_pos": cov_pos,
        "cov_neg": cov_neg,
        "refer_rate": 1 - decided.mean(0),
        "acc_non_referred": acc,
    }
