"""P5: stacked late fusion of OOF predictions (nested: the stacker for fold k never sees fold k labels)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def stack_oof(oofs: list[pd.DataFrame], name: str = "P5") -> pd.DataFrame:
    key = ["case_id", "t"]
    base = oofs[0][key + ["y", "fold"]].copy()
    for i, o in enumerate(oofs):
        z = np.clip(o.pred.to_numpy(), 1e-6, 1 - 1e-6)
        base = base.merge(o[key].assign(**{f"m{i}": np.log(z / (1 - z))}), on=key)
    cols = [f"m{i}" for i in range(len(oofs))]
    out = []
    for k in sorted(base.fold.unique()):
        tr, te = base.fold != k, base.fold == k
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(base.loc[tr, cols], base.loc[tr, "y"])
        out.append(base.loc[te, key + ["y", "fold"]].assign(pred=lr.predict_proba(base.loc[te, cols])[:, 1], model=name))
    return pd.concat(out, ignore_index=True)
