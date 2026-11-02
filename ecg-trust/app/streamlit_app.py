"""ecg-trust demo — run with `streamlit run app/streamlit_app.py`.

Research prototype: not a medical device, not for diagnostic use.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "app" / "examples"
NON_CLAIMS = (
    "Research prototype — **not a medical device and not for diagnostic use**. Trained on "
    "PTB-XL, whose labels are a noisy reference standard. Conformal guarantees are marginal and "
    "assume new ECGs resemble the calibration data. Subgroup results are descriptive."
)
BADGE = {"present": "🟥 present", "absent": "🟩 absent", "refer": "🟨 refer to cardiologist"}


def plot_12_lead(x: np.ndarray, fs: int = 100):
    import matplotlib.pyplot as plt

    from ecgtrust import LEADS

    t = np.arange(x.shape[1]) / fs
    fig, axes = plt.subplots(4, 3, figsize=(11, 6), sharex=True)
    for i, ax in enumerate(axes.T.ravel()):
        ax.plot(t, x[i], lw=0.8, color="#0072B2")
        ax.set_title(LEADS[i], loc="left", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.supxlabel("time (s)", fontsize=8)
    fig.supylabel("mV", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    import streamlit as st

    from ecgtrust.robustness.noise import add_noise_at_snr, emg_noise
    from ecgtrust.serve.predictor import InputError, Predictor

    st.set_page_config(page_title="ecg-trust", layout="wide")
    st.title("ecg-trust — when can an ECG classifier be trusted?")
    st.caption(NON_CLAIMS)

    @st.cache_resource
    def predictor():
        return Predictor.from_release(ROOT / "release")

    index = (
        json.loads((EXAMPLES / "index.json").read_text())
        if (EXAMPLES / "index.json").exists()
        else []
    )
    src = st.sidebar.radio("Input", ["Example ECG", "Upload .npy"])
    fs = 100
    if src == "Example ECG" and index:
        names = [f"{e['ecg_id']} — reference: {', '.join(e['labels']) or 'none'}" for e in index]
        e = index[names.index(st.sidebar.selectbox("Example (PTB-XL test fold)", names))]
        x = np.load(EXAMPLES / e["file"])
    else:
        up = st.sidebar.file_uploader("12 × T array in mV (.npy)", type=["npy"])
        fs = st.sidebar.number_input("sampling rate (Hz)", min_value=100, value=500)
        if up is None:
            st.info("Upload a .npy file to score it.")
            return
        x = np.load(up, allow_pickle=False)
    snr = st.sidebar.slider("Add muscle noise: SNR (dB, 30 = clean)", -6, 30, 30)
    if snr < 30:
        x = add_noise_at_snr(x, emg_noise(x.shape, fs, np.random.default_rng(0)), snr)
    try:
        r = predictor().predict(x, fs)
    except InputError as err:
        st.error(str(err))
        return
    left, right = st.columns([3, 2])
    left.pyplot(plot_12_lead(x, fs))
    right.subheader("Calibrated probability and conformal decision (α = 0.10)")
    for c, p in r.probabilities.items():
        right.progress(p, text=f"{c}: {p:.2f} — {BADGE[r.decisions[c]]}")
    if r.quality:
        right.warning("Signal-quality warnings: " + ", ".join(r.quality))
    right.caption(f"model version {r.model_version}")


if __name__ == "__main__":
    main()
