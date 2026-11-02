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
EXAMPLES_OUT = Path(__file__).resolve().parents[2] / "app" / "examples"


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


def build_examples(cfg: Config, out: Path = EXAMPLES_OUT) -> list[dict]:
    """Demo ECGs from the test fold, picked by metadata only (never by model output).

    One single-label record per superclass (lowest ecg_id) plus one record flagged by
    PTB-XL technicians for static noise. Saved as float32 (12, 1000) mV at 100 Hz.
    """
    import json

    import numpy as np

    from ecgtrust import CLASSES
    from ecgtrust.train_cli import load_labelled

    meta, y, splits = load_labelled(cfg)
    x = np.load(cfg.artifacts / "ptbxl_x100.npy", mmap_mode="r")
    test = (splits == "test").to_numpy()
    single = (y.sum(axis=1) == 1).to_numpy()
    picks = []
    for c in CLASSES:
        rows = np.flatnonzero(test & single & y[c].to_numpy().astype(bool))
        if len(rows):
            picks.append((rows[0], f"{c} (single label)"))
    noisy = np.flatnonzero(test & meta["static_noise"].notna().to_numpy())
    if len(noisy):
        picks.append((noisy[0], "noisy recording (static noise annotated)"))
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for row, title in picks:
        eid = int(meta.index[row])
        f = f"ecg_{eid}.npy"
        np.save(out / f, np.asarray(x[row], dtype=np.float32))
        labels = [c for c in CLASSES if y.iloc[row][c]]
        index.append({"file": f, "ecg_id": eid, "title": title, "labels": labels, "fs": 100})
    (out / "index.json").write_text(json.dumps(index, indent=2))
    return index


def register(sub) -> None:
    p = sub.add_parser("release", help="copy the frozen serving bundle into release/")

    def run(args):
        from ecgtrust.config import load_config

        cfg = load_config(args.config)
        print(build_release(cfg))
        print(f"{len(build_examples(cfg))} demo ECGs in {EXAMPLES_OUT}")

    p.set_defaults(func=run)
