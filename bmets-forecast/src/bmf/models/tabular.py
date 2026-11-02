"""Tabular models. P0 = clinical/lesion-burden logistic; P1 = LightGBM on radiomics+clinical;
P2 = linear probe (L2 logistic, C tuned by inner CV) on frozen foundation-model embeddings."""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, task: str = "binary", seed: int = 2026):
    if task not in ("binary", "regression"):
        raise ValueError(task)
    if name == "P0":
        est = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000) if task == "binary" else Ridge(1.0)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    if name == "P1":
        import lightgbm as lgb
        kw = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=seed, verbose=-1)
        return lgb.LGBMClassifier(class_weight="balanced", **kw) if task == "binary" else lgb.LGBMRegressor(**kw)
    if name == "P2":
        est = (LogisticRegressionCV(Cs=np.logspace(-4, 1, 12), cv=3, class_weight="balanced", max_iter=3000, scoring="roc_auc")
               if task == "binary" else RidgeCV(alphas=np.logspace(-2, 4, 12)))
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
    raise KeyError(name)
