import numpy as np
import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.evaluate.metrics import macro_auroc
from ecgtrust.models.baseline import fit_baseline


def synthetic_tabular(n=600, informative=3, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 10))
    logits = x[:, :informative] @ rng.normal(size=(informative, 5)) * 5
    y = pd.DataFrame(rng.random((n, 5)) < 1 / (1 + np.exp(-logits)), columns=list(CLASSES))
    x[rng.random(x.shape) < 0.05] = np.nan  # PTB-XL+ features contain missing values
    return pd.DataFrame(x, columns=[f"f{i}" for i in range(10)]), y


def test_baseline_learns_separable_signal():
    x, y = synthetic_tabular()
    m = fit_baseline(x[:400], y[:400], seed=0)
    assert macro_auroc(y[400:].values, m.predict_proba(x[400:])) > 0.9


def test_baseline_output_shape_and_range():
    x, y = synthetic_tabular(n=200)
    p = fit_baseline(x, y, 0).predict_proba(x)
    assert p.shape == (len(x), 5) and ((p >= 0) & (p <= 1)).all()


def test_baseline_is_seeded():
    x, y = synthetic_tabular(n=300)
    a = fit_baseline(x, y, 1).predict_proba(x)
    b = fit_baseline(x, y, 1).predict_proba(x)
    assert np.array_equal(a, b)
