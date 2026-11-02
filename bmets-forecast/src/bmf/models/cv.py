"""Patient-level cross-validation producing out-of-fold (OOF) predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tabular import make_model


def _build(name, task, **kw):
    if name in ("P3", "P4"):
        from .temporal import TemporalModel
        return TemporalModel(task=task, **kw)
    return make_model(name, task)


def run_cv(model_name: str, X: pd.DataFrame, y: pd.Series, splits: dict, task: str = "binary", **kw) -> pd.DataFrame:
    """X, y indexed by (case_id, t). Fold k trains on all patients with splits[case] != k."""
    cases = X.index.get_level_values(0)
    folds = cases.map(splits).to_numpy()
    if np.any(pd.isna(folds)):
        raise KeyError("some patients are missing from splits")
    out = []
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        if te.sum() == 0:
            continue
        m = _build(model_name, task, **kw)
        m.fit(X[tr], y[tr].to_numpy())
        p = m.predict_proba(X[te])[:, 1] if task == "binary" else m.predict(X[te])
        out.append(pd.DataFrame({"case_id": cases[te], "t": X.index.get_level_values(1)[te], "y": y[te].to_numpy(),
                                 "pred": p, "fold": k, "model": model_name}))
    return pd.concat(out, ignore_index=True)


def fit_full(model_name: str, X: pd.DataFrame, y: pd.Series, task: str = "binary", **kw):
    """Final model on all development data (used once for the locked external evaluation)."""
    return _build(model_name, task, **kw).fit(X, y.to_numpy())
