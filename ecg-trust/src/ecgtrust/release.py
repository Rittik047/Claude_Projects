"""`ecgtrust release`: copy the frozen serving bundle (ResNet ensemble) into `release/`.

The bundle is what `Predictor.from_release` loads: the seed checkpoints, the per-class
temperatures, the conformal thresholds for each α, and `freeze.json` for provenance.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ecgtrust.config import Config
from ecgtrust.models.ensemble import checkpoint_paths

MAX_BYTES = 20 * 1024**2
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "release"


def build_release(cfg: Config, out: Path = DEFAULT_OUT) -> Path:
    freeze = cfg.artifacts / "freeze.json"
    if not freeze.exists():
        raise FileNotFoundError("no freeze.json: run `ecgtrust freeze` before releasing")
    cal = cfg.artifacts / "calibration"
    files = {p: out / "models" / p.name for p in checkpoint_paths(cfg)}
    files[cal / "T.npy"] = out / "T.npy"
    files.update({cal / f"conformal_{a}.json": out / f"conformal_{a}.json" for a in cfg.alphas})
    files[freeze] = out / "freeze.json"
    total = sum(src.stat().st_size for src in files)
    if total > MAX_BYTES:
        raise ValueError(f"release is {total / 1024**2:.1f} MB, above the 20 MB cap")
    if out.exists():
        shutil.rmtree(out)
    for src, dst in files.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return out


def register(sub) -> None:
    p = sub.add_parser("release", help="copy the frozen serving bundle into release/")

    def run(args):
        from ecgtrust.config import load_config

        print(build_release(load_config(args.config)))

    p.set_defaults(func=run)
