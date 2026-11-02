"""SHAP for tabular models: TreeExplainer (LightGBM) / LinearExplainer (scaled logistic pipeline).
Values are in log-odds (margin) space so that base + sum(values) == model margin."""
from __future__ import annotations

import numpy as np
import pandas as pd


def shap_values(model, X: pd.DataFrame):
    import shap
    if hasattr(model, "steps"):                                   # sklearn Pipeline: imputer -> scaler -> linear
        pre, est = model[:-1], model[-1]
        Xt = pd.DataFrame(pre.transform(X), columns=X.columns, index=X.index)
        if hasattr(est, "coef_"):
            coef, b = np.ravel(est.coef_), float(np.ravel(est.intercept_)[0])
            vals = (Xt.to_numpy() - Xt.to_numpy().mean(0)) * coef   # exact linear SHAP (interventional, independent)
            base = b + float(Xt.to_numpy().mean(0) @ coef)
            return vals, base, Xt
        model, X = est, Xt
    ex = shap.TreeExplainer(model)
    vals = ex.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        vals = vals[..., 1]
    base = ex.expected_value
    base = float(np.ravel(base)[-1])
    return vals, base, X
