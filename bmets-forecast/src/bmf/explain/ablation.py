"""Leave-one-sequence-out ablation: score_fn(dropped_sequence_or_None) -> metric."""
from __future__ import annotations

import pandas as pd


def sequence_ablation(score_fn, X_by_seq: dict) -> pd.DataFrame:
    full = float(score_fn(None))
    rows = [("none", full, 0.0)]
    for s in X_by_seq:
        v = float(score_fn(s))
        rows.append((s, v, v - full))
    return pd.DataFrame(rows, columns=["dropped", "score", "delta"])
