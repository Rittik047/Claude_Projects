import numpy as np

from ecgtrust.calibrate.conformal import ConformalThresholds, coverage, decide, fit_conformal


def run_trial(seed, alpha, n_cal, n_test):
    rng = np.random.default_rng(seed)
    p = rng.random((n_cal + n_test, 5))
    y = (rng.random(p.shape) < p).astype(int)
    thr = fit_conformal(p[:n_cal], y[:n_cal], alpha)
    c = coverage(y[n_cal:], p[n_cal:], thr)
    return {"cov_pos": c["cov_pos"].mean(), "cov_neg": c["cov_neg"].mean()}


def test_label_conditional_coverage_holds():
    covs = [run_trial(s, alpha=0.1, n_cal=500, n_test=2000) for s in range(200)]
    assert np.mean([c["cov_pos"] for c in covs]) >= 0.89
    assert np.mean([c["cov_neg"] for c in covs]) >= 0.89


def test_quantile_index_small_n():
    p = np.linspace(0.1, 0.9, 9)[:, None].repeat(5, axis=1)
    y = np.ones_like(p, dtype=int)  # n_pos = 9 → ceil(10 * 0.9) = 9 → the max score
    thr = fit_conformal(p, y, alpha=0.1)
    assert np.isclose(thr.q_present[0], (1 - p[:, 0]).max())


def test_too_few_calibration_points_gives_inf_and_refer():  # Review Focus 5
    p = np.full((10, 5), 0.95)
    y = np.zeros((10, 5), dtype=int)
    y[:3] = 1  # 3 positives at alpha=0.05 → ceil(4 * 0.95) = 4 > 3 → inf
    thr = fit_conformal(p, y, alpha=0.05)
    assert np.isinf(thr.q_present[0]) and (decide(p, thr)[:, 0] != "present").all()


def test_decision_mapping():
    thr = ConformalThresholds(np.full(5, 0.3), np.full(5, 0.3), 0.1)
    d = decide(np.array([[0.9] * 5, [0.1] * 5, [0.5] * 5]), thr)
    assert list(d[:, 0]) == ["present", "absent", "refer"]


def test_coverage_report_keys():
    rng = np.random.default_rng(0)
    p = rng.random((200, 5))
    y = (rng.random(p.shape) < p).astype(int)
    c = coverage(y, p, fit_conformal(p, y, 0.1))
    assert set(c) == {"cov_pos", "cov_neg", "refer_rate", "acc_non_referred"}
    assert all(v.shape == (5,) for v in c.values())
