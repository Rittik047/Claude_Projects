import numpy as np

from ecgtrust.calibrate.metrics import brier, ece, reliability_curve
from ecgtrust.calibrate.temperature import fit_temperature

rng = np.random.default_rng(0)


def test_recovers_known_temperature():
    z = rng.normal(0, 2, (1, 20000, 5))
    y = rng.random(z.shape[1:]) < 1 / (1 + np.exp(-z[0]))
    assert np.allclose(fit_temperature(z * 3, y), 3, rtol=0.1)


def test_ece_near_zero_when_calibrated():
    p = rng.random(50000)
    y = rng.random(50000) < p
    assert ece(y, p) < 0.02


def test_ece_known_value():
    assert np.isclose(ece(np.zeros(1000), np.full(1000, 0.3)), 0.3)


def test_brier_and_reliability():
    assert np.isclose(brier(np.array([1, 0]), np.array([1.0, 0.0])), 0.0)
    mp, obs, cnt = reliability_curve(np.zeros(100), np.linspace(0, 1, 100), n_bins=10)
    assert len(mp) == len(obs) == len(cnt) == 10 and cnt.sum() == 100 and (obs == 0).all()
