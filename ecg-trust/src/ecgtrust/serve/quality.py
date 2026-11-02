"""Signal-quality flags for an input ECG (12, 1000) in mV at 100 Hz."""

from __future__ import annotations

import numpy as np

from ecgtrust import FS, LEADS


def quality_flags(x: np.ndarray) -> list[str]:
    flags = []
    std = x.std(axis=1)
    flat = std < 1e-4
    flags += [f"flat_lead:{LEADS[i]}" for i in np.flatnonzero(flat)]
    # QRS amplitudes are ~0.5-3 mV; a lead-median 99th-percentile |x| above 20 suggests µV input
    if np.median(np.percentile(np.abs(x), 99, axis=1)) > 20:
        flags.append("possible_unit_error")
    f = np.fft.rfftfreq(x.shape[1], 1 / FS)
    psd = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    total = psd[~flat].sum()
    if total > 0 and psd[~flat][:, f > 40].sum() / total > 0.30:
        flags.append("high_frequency_noise")
    live = x[~flat]
    if live.size:
        at_rail = (live == live.max(axis=1, keepdims=True)) | (
            live == live.min(axis=1, keepdims=True)
        )
        if at_rail.mean() > 0.01:
            flags.append("saturation")
    return flags
