"""Release-bundle predictor: calibrated probabilities, conformal decisions, quality flags."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ecgtrust import CLASSES, FS, N_SAMPLES
from ecgtrust.calibrate.conformal import ConformalThresholds, decide
from ecgtrust.data.external import resample_to_100
from ecgtrust.models.ensemble import ensemble_proba
from ecgtrust.serve.quality import quality_flags

Decision = Literal["present", "absent", "refer"]
MIN_SECONDS = 2.5


class PredictionResult(BaseModel):
    probabilities: dict[str, float]
    decisions: dict[str, Decision]
    quality: list[str]
    model_version: str


class InputError(ValueError):
    """The input cannot be scored; message is shown to the caller."""


def prepare(x: np.ndarray, fs: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or 12 not in x.shape:
        raise InputError(f"expected a (12, T) array of 12 leads; got shape {x.shape}")
    if x.shape[0] != 12:
        raise InputError(f"expected (12, T) but got {x.shape}: transpose your input (leads first)")
    if not np.isfinite(x).all():
        raise InputError("signal contains NaN or infinite values")
    if fs < FS:
        raise InputError(f"sampling rate {fs} Hz is below the supported minimum of {FS} Hz")
    if x.shape[1] / fs < MIN_SECONDS:
        raise InputError(f"record is {x.shape[1] / fs:.1f} s; at least {MIN_SECONDS} s is required")
    x = resample_to_100(x, fs)
    n = x.shape[1]
    if n >= N_SAMPLES:  # centre-crop
        s = (n - N_SAMPLES) // 2
        return x[:, s : s + N_SAMPLES]
    pad = N_SAMPLES - n
    return np.pad(x, ((0, 0), (pad // 2, pad - pad // 2)))


class Predictor:
    def __init__(
        self, checkpoints: list[Path], temps: np.ndarray, thr: ConformalThresholds, version: str
    ):
        from ecgtrust.predict import load_checkpoint

        self.members = [load_checkpoint(p) for p in checkpoints]
        self.temps, self.thr, self.version = temps, thr, version

    @classmethod
    def from_release(cls, path: Path = Path("release"), alpha: float = 0.10) -> Predictor:
        path = Path(path)
        ckpts = sorted((path / "models").glob("resnet_seed*.pt"))
        thr = ConformalThresholds.from_json(
            json.loads((path / f"conformal_{alpha}.json").read_text())
        )
        freeze = json.loads((path / "freeze.json").read_text())
        return cls(
            ckpts, np.load(path / "T.npy"), thr, str(freeze.get("git_commit", "unknown"))[:12]
        )

    def predict(self, x: np.ndarray, fs: int) -> PredictionResult:
        from ecgtrust.models.train import predict_logits_array

        xp = prepare(x, fs)
        lg = np.stack(
            [
                predict_logits_array(m, xp[None], ck["lead_mean"], ck["lead_std"])
                for m, ck in self.members
            ]
        )
        p = ensemble_proba(lg, self.temps)
        d = decide(p, self.thr)[0]
        return PredictionResult(
            probabilities={c: float(p[0, i]) for i, c in enumerate(CLASSES)},
            decisions={c: str(d[i]) for i, c in enumerate(CLASSES)},
            quality=quality_flags(xp),
            model_version=self.version,
        )
