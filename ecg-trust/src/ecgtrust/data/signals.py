"""WFDB signal loading in canonical lead order, an on-disk cache, and per-lead scaling."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecgtrust import LEADS, N_SAMPLES

_UNIT_SCALE = {"mv": 1.0, "uv": 1e-3, "µv": 1e-3, "v": 1e3}


def load_record(path_no_ext: Path, leads: tuple[str, ...] = LEADS) -> np.ndarray:
    """Return a (12, n_samples) float32 array in mV, channels reordered to `leads`."""
    sig, fields = wfdb.rdsamp(str(path_no_ext))
    names = [n.strip().upper() for n in fields["sig_name"]]
    missing = [ld for ld in leads if ld.upper() not in names]
    if missing:
        raise ValueError(f"{path_no_ext}: missing leads {missing}")
    idx = [names.index(ld.upper()) for ld in leads]
    scale = np.array([_UNIT_SCALE.get(str(fields["units"][i]).lower(), 1.0) for i in idx])
    return (sig[:, idx] * scale).T.astype(np.float32)


def build_cache(
    meta: pd.DataFrame, root: Path, out: Path, workers: int = 8, column: str = "filename_lr"
) -> np.ndarray:
    """Write all records to `out` (.npy, shape (N, 12, 1000)) in `meta.index` order.

    Also writes `<out stem>_index.parquet` mapping row → ecg_id. NaNs are set to 0 and counted.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(len(meta), 12, N_SAMPLES)
    )
    paths = [Path(root) / f for f in meta[column]]
    n_nan = 0

    def _one(i: int) -> int:
        r = load_record(paths[i])
        bad = int(np.isnan(r).sum())
        x[i] = np.nan_to_num(r[:, :N_SAMPLES], nan=0.0)
        return bad

    with ThreadPoolExecutor(workers) as ex:
        n_nan = sum(ex.map(_one, range(len(meta))))
    x.flush()
    idx_path = out.with_name(out.stem.replace("_x100", "") + "_index.parquet")
    pd.DataFrame({"ecg_id": meta.index.to_numpy()}).to_parquet(idx_path)
    print(f"cache {out}: {x.shape}, NaN samples set to 0: {n_nan}")
    return x


def fit_lead_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean and std over records and time; compute on the train split only."""
    mean = x.mean(axis=(0, 2), dtype=np.float64)
    std = x.std(axis=(0, 2), dtype=np.float64)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
