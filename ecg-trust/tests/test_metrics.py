import numpy as np

from ecgtrust.evaluate.metrics import (
    bootstrap_samples,
    macro_auroc,
    patient_bootstrap,
    per_class_auprc,
    per_class_auroc,
)

rng = np.random.default_rng(0)
Y = (rng.random((300, 5)) < 0.3).astype(int)
P = np.clip(Y * 0.6 + rng.random((300, 5)) * 0.5, 0, 1)
GROUPS = np.repeat(np.arange(150), 2)


def accuracy(y, p):
    return float(((p > 0.5) == y).mean())


def test_perfect_auroc():
    y = np.eye(5)[np.arange(50) % 5]
    assert macro_auroc(y, y.astype(float)) == 1.0


def test_single_value_class_is_nan_and_ignored_in_macro():
    y = np.eye(5)[np.arange(50) % 4]  # class 4 never positive
    auc = per_class_auroc(y, y.astype(float))
    assert np.isnan(auc[4]) and macro_auroc(y, y.astype(float)) == 1.0


def test_auprc_shape():
    assert per_class_auprc(Y, P).shape == (5,)


def test_bootstrap_ci_brackets_point():
    pt, lo, hi = patient_bootstrap(Y, P, GROUPS, macro_auroc, n=200)
    assert lo <= pt <= hi and hi - lo < 0.2


def test_bootstrap_resamples_whole_patients():
    # patient 0 all correct, patient 1 all wrong → any resample is a mix of whole patients
    y2 = np.array([1, 1, 1, 1, 1, 1])
    p2 = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    vals = bootstrap_samples(y2, p2, np.array([0, 0, 0, 1, 1, 1]), accuracy, n=100)
    assert set(np.round(vals, 6)) <= {0.0, 0.5, 1.0}


def test_bootstrap_is_seeded():
    a = patient_bootstrap(Y, P, GROUPS, macro_auroc, n=50, seed=3)
    b = patient_bootstrap(Y, P, GROUPS, macro_auroc, n=50, seed=3)
    assert a == b
