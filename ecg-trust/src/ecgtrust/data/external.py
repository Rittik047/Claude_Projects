"""External validation set: PhysioNet `ecg-arrhythmia` 1.0.0 (Chapman/Shaoxing/Ningbo).

Records are 12-lead, 500 Hz, 10 s, with `Age`, `Sex`, `Dx` (SNOMED CT) header comments.
Diagnoses are harmonised to PTB-XL superclasses through the reviewed table
`configs/external_label_map.csv` (spec §4.3). Column `superclass` is the strict, PTB-XL-faithful
primary map (isolated ST/T and voltage-only findings are PTB-XL *form* statements, so unmapped);
`superclass_liberal` is the sensitivity map. NORM is a *proxy*: only sinus-family rhythm codes
(SR, SB, ST, SA). As in PTB-XL, records with no superclass are excluded from evaluation
(`has_label` False); signal-quality exclusions are kept separately in `excluded_reason`.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from ecgtrust import CLASSES, FS, LEADS, N_SAMPLES
from ecgtrust.data.ptbxl import age_band

SINUS_RHYTHM = 426783006
# sinus rhythm, sinus bradycardia, sinus tachycardia, sinus arrhythmia (irregularity)
SINUS_FAMILY = frozenset({426783006, 426177001, 427084000, 427393009})
EXPECTED_SAMPLES = 5000  # 10 s at 500 Hz


class ExternalLabelError(Exception):
    """The label map assigns one SNOMED CT code to two superclasses."""


def parse_header_comments(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().lower()] = v.strip()
    try:
        age = float(fields.get("age", ""))
        age = None if np.isnan(age) else age
    except ValueError:
        age = None
    sex = fields.get("sex", "").lower()
    sex = sex if sex in ("male", "female") else "unknown"
    dx = [int(c) for c in fields.get("dx", "").replace(" ", "").split(",") if c.isdigit()]
    return {"age": age, "sex": sex, "dx": dx}


def load_label_map(path: Path, column: str = "superclass") -> dict[int, str | None]:
    out: dict[int, str | None] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code, sc = int(row["snomed_ct"]), (row[column].strip() or None)
            if code in out and out[code] != sc:
                raise ExternalLabelError(f"code {code} mapped to {out[code]} and {sc}")
            out[code] = sc
    return out


def external_labels(dx: list[int], label_map: dict[int, str | None]) -> dict[str, bool]:
    mapped = {label_map.get(c) for c in dx} - {None}
    y = {c: c in mapped for c in CLASSES}
    y["NORM"] = bool(dx) and set(dx) <= SINUS_FAMILY
    return y


def label_external_meta(meta: pd.DataFrame, label_map_path: Path) -> pd.DataFrame:
    """Add strict labels (CLASSES, has_label) and liberal ones (<C>_liberal, has_label_liberal)
    computed from the comma-separated `dx` column."""
    out = meta.copy()
    dx = [[int(c) for c in str(d).split(",") if c.strip().isdigit()] for d in out["dx"]]
    for column, suffix in (("superclass", ""), ("superclass_liberal", "_liberal")):
        m = load_label_map(label_map_path, column)
        y = pd.DataFrame([external_labels(d, m) for d in dx], index=out.index)
        for c in CLASSES:
            out[c + suffix] = y[c].astype(bool)
        out["has_label" + suffix] = y[list(CLASSES)].any(axis=1)
    return out


def resample_to_100(x: np.ndarray, fs: int) -> np.ndarray:
    """Polyphase resampling (with anti-alias filtering) of (12, T) from `fs` to 100 Hz."""
    if fs < FS:
        raise ValueError(f"sampling rate {fs} Hz is below the model's {FS} Hz")
    if fs == FS:
        return np.asarray(x, dtype=np.float32)
    g = gcd(int(fs), FS)
    return resample_poly(x, FS // g, int(fs) // g, axis=-1).astype(np.float32)


def flat_leads(x: np.ndarray, tol: float = 1e-4) -> list[int]:
    return [i for i in range(x.shape[0]) if float(np.nanstd(x[i])) < tol]


def _read(path_no_ext: Path) -> tuple[np.ndarray | None, dict]:
    rec = {"record": path_no_ext.name, "excluded_reason": ""}
    try:
        sig, fields = wfdb.rdsamp(str(path_no_ext))
    except Exception as e:  # noqa: BLE001 - unreadable record is excluded, reason kept
        rec.update(age=None, sex="unknown", dx="", excluded_reason=f"read_error:{type(e).__name__}")
        return None, rec
    h = parse_header_comments("\n".join(fields.get("comments", [])))
    rec.update(age=h["age"], sex=h["sex"], dx=",".join(map(str, h["dx"])))
    names = [n.strip().upper() for n in fields["sig_name"]]
    if any(ld.upper() not in names for ld in LEADS):
        rec["excluded_reason"] = "missing_lead"
        return None, rec
    x = sig[:, [names.index(ld.upper()) for ld in LEADS]].T
    if x.shape[1] != EXPECTED_SAMPLES or fields["fs"] != 500:
        rec["excluded_reason"] = f"length_or_fs:{x.shape[1]}@{fields['fs']}"
    elif np.isnan(x).any():
        rec["excluded_reason"] = "nan"
    elif flat := flat_leads(x):
        rec["excluded_reason"] = "flat_lead:" + "|".join(LEADS[i] for i in flat)
    if not h["dx"]:
        rec["excluded_reason"] = rec["excluded_reason"] or "no_dx"
    if rec["excluded_reason"]:
        return None, rec
    return resample_to_100(x, 500)[:, :N_SAMPLES], rec


def build_external_cache(root: Path, label_map_path: Path, out_dir: Path, workers: int = 8):
    """Write `external_x100.npy` (included records only) and `external_meta.parquet`."""
    paths = sorted(p.with_suffix("") for p in Path(root, "WFDBRecords").rglob("*.hea"))
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(_read, paths))
    meta = label_external_meta(pd.DataFrame([r for _, r in results]), label_map_path)
    meta["age_band"] = meta["age"].apply(lambda a: age_band(float("nan") if a is None else a))
    keep = [x for x, _ in results if x is not None]
    meta["cache_row"] = -1
    meta.loc[meta.excluded_reason == "", "cache_row"] = np.arange(len(keep))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "external_x100.npy", np.stack(keep).astype(np.float32))
    meta.to_parquet(out_dir / "external_meta.parquet")
    print(f"external: {len(meta)} records, {len(keep)} included")
    print(
        meta.loc[meta.excluded_reason != "", "excluded_reason"].str.split(":").str[0].value_counts()
    )
    return meta
