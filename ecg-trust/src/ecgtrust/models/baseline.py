"""Gradient-boosted one-vs-rest baseline on PTB-XL+ ECG measurements (spec §4.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ecgtrust import CLASSES


@dataclass
class BaselineModel:
    models: list[HistGradientBoostingClassifier]
    feature_names: list[str]

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x = x[self.feature_names] if isinstance(x, pd.DataFrame) else x
        return np.column_stack([m.predict_proba(x)[:, 1] for m in self.models])


def fit_baseline(x: pd.DataFrame, y: pd.DataFrame, seed: int) -> BaselineModel:
    models = []
    for c in CLASSES:
        m = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, early_stopping=True, random_state=seed
        )
        m.fit(x, np.asarray(y[c]).astype(int))
        models.append(m)
    return BaselineModel(models, list(x.columns))
