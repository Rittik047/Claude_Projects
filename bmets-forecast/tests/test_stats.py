import numpy as np
import pandas as pd
import pytest

from bmf.eval.stats import (auroc_ci, calibration, cindex_ci, cluster_bootstrap_indices, decision_curve,
                            delta_auc, holm, subgroup_table, td_auc)


def test_auroc_ci_perfect():
    y = np.array([0, 0, 1, 1] * 10); p = y * 0.9 + 0.05
    est, lo, hi = auroc_ci(y, p, groups=np.arange(40) // 2, n_boot=200)
    assert est == 1.0 and lo == 1.0


def test_holm_known():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_calibration_perfect_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 20000); y = rng.random(20000) < p
    c = calibration(y, p)
    assert abs(c["slope"] - 1) < 0.1 and abs(c["intercept"]) < 0.1 and c["ece"] < 0.03


def test_decision_curve_treat_all():
    y = np.array([1] * 30 + [0] * 70); p = np.linspace(0, 1, 100)
    dc = decision_curve(y, p, [0.1, 0.2])
    assert np.allclose(dc.nb_all, 0.3 - 0.7 * np.array([0.1, 0.2]) / np.array([0.9, 0.8]))
    assert np.allclose(dc.nb_none, 0)


def test_cluster_bootstrap_resamples_patients():
    groups = np.repeat(np.arange(10), 3)
    for idx in cluster_bootstrap_indices(groups, n_boot=20, seed=0):
        counts = np.bincount(groups[idx], minlength=10)
        assert np.all(counts % 3 == 0)


def test_delta_auc_detects_better_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400); good = y + rng.normal(0, .5, 400); bad = y + rng.normal(0, 3, 400)
    d, lo, hi, p = delta_auc(y, good, bad, groups=np.arange(400), n_boot=300)
    assert d > 0 and lo > 0 and p < 0.05


def test_survival_metrics():
    rng = np.random.default_rng(0)
    risk = rng.normal(size=300); t = rng.exponential(np.exp(-risk)); c = rng.exponential(2, 300)
    time, ev = np.minimum(t, c), (t <= c).astype(int)
    est, lo, hi = cindex_ci(time, ev, risk, np.arange(300), n_boot=100)
    assert est > 0.65 and lo < est < hi
    auc = td_auc(time, ev, risk, horizons=(0.5, 1.0))
    assert all(v > 0.65 for v in auc.values())


def test_subgroup_table():
    rng = np.random.default_rng(0)
    oof = pd.DataFrame({"case_id": [f"P{i}" for i in range(200)], "t": 1, "y": rng.integers(0, 2, 200)})
    oof["pred"] = oof.y + rng.normal(0, 1, 200)
    meta = pd.DataFrame({"case_id": oof.case_id, "sex": rng.choice(["m", "f"], 200)})
    st = subgroup_table(oof, meta, by=("sex",), n_boot=50)
    assert set(st.level) == {"m", "f"} and {"n", "n_pos", "auroc", "lo", "hi"} <= set(st.columns)
