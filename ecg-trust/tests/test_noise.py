import numpy as np
import pytest

from ecgtrust.robustness.noise import (
    add_noise_at_snr,
    baseline_wander,
    electrode_motion,
    emg_noise,
    lead_dropout,
)

rng = np.random.default_rng(0)


def ecg_like(t=1000):
    tt = np.arange(t) / 100
    beat = np.exp(-(((tt % 0.8) - 0.4) ** 2) / 0.0008)  # spiky QRS-like train, 75 bpm
    return np.tile(beat, (12, 1)) * np.linspace(0.5, 1.5, 12)[:, None]


def band_power_fraction(x, fs, lo, hi):
    f = np.fft.rfftfreq(x.shape[-1], 1 / fs)
    psd = np.abs(np.fft.rfft(x - x.mean(-1, keepdims=True), axis=-1)) ** 2
    return psd[..., (f >= lo) & (f <= hi)].sum() / psd.sum()


@pytest.mark.parametrize("snr", [24, 6, 0, -6])
@pytest.mark.parametrize("gen", [emg_noise, baseline_wander, electrode_motion])
def test_snr_is_exact(snr, gen):
    x = ecg_like()
    y = add_noise_at_snr(x, gen(x.shape, 100, rng), snr)
    measured = 10 * np.log10(x.var(axis=1) / (y - x).var(axis=1))
    assert np.allclose(measured, snr, atol=0.1)


def test_emg_band():
    assert band_power_fraction(emg_noise((12, 100_000), 100, rng), 100, 20, 45) > 0.9


def test_baseline_wander_band():
    assert band_power_fraction(baseline_wander((12, 100_000), 100, rng), 100, 0, 0.5) > 0.9


def test_lead_dropout_zeroes_k_leads():
    y = lead_dropout(ecg_like(), 2, rng)
    assert (np.abs(y).sum(axis=1) == 0).sum() == 2


def test_robustness_curve_rows_and_degradation():
    from ecgtrust.calibrate.conformal import fit_conformal
    from ecgtrust.robustness.run import robustness_curve

    r = np.random.default_rng(1)
    y = np.repeat((r.random((120, 1)) < 0.5).astype(np.float32), 5, axis=1)
    x = np.stack([ecg_like() * (1 + 0.5 * yi[0]) for yi in y]).astype(np.float32)

    def predict_logits(xb):  # lead-0 peak amplitude encodes every label; noise corrupts it
        a = xb[:, 0].max(axis=1, keepdims=True)
        return np.repeat((a - a.mean()) * 20, 5, axis=1)[None]

    p_clean = 1 / (1 + np.exp(-predict_logits(x)[0]))
    thr = fit_conformal(p_clean, y, 0.1)
    df = robustness_curve(
        x,
        y,
        predict_logits,
        np.ones(5),
        thr,
        snr_db=[24, -6],
        noise_types=("emg",),
        dropout_k=(1,),
        seed=0,
    )
    assert list(df.columns) == ["noise", "level", "macro_auroc", "macro_ece", "mean_refer_rate"]
    assert df.iloc[0].noise == "clean" and len(df) == 1 + 2 + 1
    emg = df[df.noise == "emg"].set_index("level")
    assert emg.loc["24", "macro_auroc"] > emg.loc["-6", "macro_auroc"]
