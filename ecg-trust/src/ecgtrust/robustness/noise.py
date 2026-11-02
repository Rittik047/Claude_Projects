"""Physically parameterised synthetic ECG noise at an exact per-lead SNR (spec §5.5).

Real NSTDB noise records are not on the AWS mirror; these generators mimic its three
types: baseline wander (respiration / movement, < 0.5 Hz), EMG-like muscle noise
(band-limited 20-45 Hz at fs = 100 Hz) and electrode-motion artefacts (decaying steps and
short bursts). SNR is defined per lead as 10 log10(var(signal) / var(noise)).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def baseline_wander(shape: tuple[int, int], fs: int, rng: np.random.Generator) -> np.ndarray:
    leads, n = shape
    t = np.arange(n) / fs
    out = np.zeros(shape)
    for _ in range(3):
        f = rng.uniform(0.05, 0.5, (leads, 1))
        out += rng.uniform(0.5, 1.0, (leads, 1)) * np.sin(
            2 * np.pi * f * t + rng.uniform(0, 2 * np.pi, (leads, 1))
        )
    walk = np.cumsum(rng.normal(0, 1, shape), axis=1) / np.sqrt(fs)
    sos = butter(2, 0.5, btype="low", fs=fs, output="sos")
    walk = sosfiltfilt(sos, walk - walk.mean(axis=1, keepdims=True), axis=1)
    return out + 0.2 * walk / (walk.std(axis=1, keepdims=True) + 1e-12)


def emg_noise(shape: tuple[int, int], fs: int, rng: np.random.Generator) -> np.ndarray:
    sos = butter(4, [20, min(45, 0.45 * fs)], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, rng.normal(0, 1, shape), axis=1)


def electrode_motion(shape: tuple[int, int], fs: int, rng: np.random.Generator) -> np.ndarray:
    leads, n = shape
    t = np.arange(n) / fs
    out = np.zeros(shape)
    sos = butter(2, [5, 15], btype="band", fs=fs, output="sos")
    for ld in range(leads):
        for _ in range(rng.integers(1, 4)):
            t0, tau = rng.uniform(0, t[-1]), rng.uniform(0.1, 0.5)
            out[ld] += rng.normal(0, 1) * np.where(t >= t0, np.exp(-(t - t0) / tau), 0.0)
        width = max(int(0.2 * fs), 1)
        start = rng.integers(0, max(n - width, 1))
        burst = sosfiltfilt(sos, rng.normal(0, 1, n))
        out[ld, start : start + width] += 3 * burst[start : start + width]
    return out


def add_noise_at_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` per lead so that var(x)/var(scaled noise) = 10**(snr_db/10)."""
    ps = x.var(axis=-1, keepdims=True)
    pn = noise.var(axis=-1, keepdims=True)
    scale = np.sqrt(np.divide(ps, pn * 10 ** (snr_db / 10), out=np.zeros_like(ps), where=pn > 0))
    return (x + noise * scale).astype(x.dtype)


def lead_dropout(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    y = x.copy()
    y[rng.choice(x.shape[0], size=k, replace=False)] = 0.0
    return y


GENERATORS = {"bw": baseline_wander, "emg": emg_noise, "em": electrode_motion}
