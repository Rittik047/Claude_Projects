import numpy as np
import pytest
import wfdb

from ecgtrust import LEADS
from ecgtrust.data.signals import fit_lead_stats, load_record, normalize

T = np.arange(1000) / 100.0
LEAD_I_SIGNAL = np.sin(2 * np.pi * 1.0 * T)


def write_fixture(tmp_path, order=LEADS, name="rec"):
    sig = np.stack([LEAD_I_SIGNAL * (1 + LEADS.index(lead)) for lead in order], axis=1)
    wfdb.wrsamp(
        name,
        fs=100,
        units=["mV"] * len(order),
        sig_name=list(order),
        p_signal=sig,
        fmt=["16"] * len(order),
        write_dir=str(tmp_path),
    )
    return tmp_path / name


def test_load_record_reorders_leads(tmp_path):
    x = load_record(write_fixture(tmp_path, order=LEADS[::-1]))
    assert x.shape == (12, 1000) and x.dtype == np.float32
    assert np.allclose(x[0], LEAD_I_SIGNAL, atol=1e-3)
    assert np.allclose(x[6], LEAD_I_SIGNAL * 7, atol=1e-2)  # V1 is canonical index 6


def test_missing_lead_raises(tmp_path):
    with pytest.raises(ValueError, match="missing"):
        load_record(write_fixture(tmp_path, order=LEADS[:11]))


def test_lead_stats_from_train_only():
    m, s = fit_lead_stats(np.ones((4, 12, 10)) * np.arange(12)[None, :, None])
    assert m.shape == (12,) and np.allclose(m, np.arange(12)) and (s > 0).all()


def test_normalize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, (50, 12, 100)).astype(np.float32)
    m, s = fit_lead_stats(x)
    z = normalize(x, m, s)
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-4) and np.allclose(
        z.std(axis=(0, 2)), 1, atol=1e-3
    )
