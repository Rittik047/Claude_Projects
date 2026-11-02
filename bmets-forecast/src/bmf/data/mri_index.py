"""Index of TCIA NIfTI files and study loaders.

Layout (TCIA usage notes): <root>/BMETSXXXX_YY/BMETSXXXX_YY-<sequence>.nii.gz
Sequence identifiers are matched case-insensitively (t1postFSE vs t1postfse).
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

SEQS_CORE = ("t1c", "t1n", "t2f", "t2w")
SEQS_OPT = ("t1postFSE", "dwi", "adc", "swan", "asl")
# TCIA writes t2 for T2-weighted in file names; metadata calls it t2w.
_ALIASES = {"t2w": ("t2w", "t2"), "swan": ("swan", "swi", "gre"), "t1postFSE": ("t1postfse",)}


def _match(files: list[Path], study_id: str, seq: str) -> Path | None:
    for alias in _ALIASES.get(seq, (seq,)):
        want = f"{study_id}-{alias}.nii.gz".lower()
        for f in files:
            if f.name.lower() == want:
                return f
    return None


def build_index(root: Path | str, tp: pd.DataFrame, seg_glob: str = "*seg*.nii.gz") -> pd.DataFrame:
    """Rows (study_id, case_id, seq, path, exists) for every core/optional sequence and 'seg'."""
    root = Path(root)
    rows = []
    for sid, cid in zip(tp["study_id"], tp["case_id"]):
        d = root / sid
        files = sorted(d.glob("*.nii*")) if d.exists() else []
        for seq in SEQS_CORE + SEQS_OPT:
            f = _match(files, sid, seq)
            rows.append((sid, cid, seq, str(f) if f else "", f is not None))
        segs = [f for f in files if fnmatch.fnmatch(f.name.lower(), seg_glob.lower())]
        rows.append((sid, cid, "seg", str(segs[0]) if segs else "", bool(segs)))
    return pd.DataFrame(rows, columns=["study_id", "case_id", "seq", "path", "exists"])


def _path(index: pd.DataFrame, study_id: str, seq: str) -> str | None:
    r = index[(index.study_id == study_id) & (index.seq == seq) & index.exists]
    return r.path.iloc[0] if len(r) else None


def load_study(index: pd.DataFrame, study_id: str, seqs=SEQS_CORE):
    """Return (vol[C,X,Y,Z] float32, mask[C] uint8, affine). Missing sequences -> zeros, mask 0.

    All sequences must share the reference grid (true after preprocessing; TCIA
    images are co-registered per study). The reference is the first present sequence.
    """
    imgs, ref = {}, None
    for s in seqs:
        p = _path(index, study_id, s)
        if p:
            imgs[s] = nib.load(p)
            ref = ref or imgs[s]
    if ref is None:
        raise FileNotFoundError(f"no sequences found for {study_id}")
    shape = ref.shape[:3]
    vol = np.zeros((len(seqs), *shape), np.float32)
    mask = np.zeros(len(seqs), np.uint8)
    for i, s in enumerate(seqs):
        if s in imgs:
            a = np.asarray(imgs[s].dataobj, dtype=np.float32)
            if a.shape[:3] != shape:
                raise ValueError(f"{study_id}:{s} shape {a.shape} != reference {shape}; run preprocess first")
            vol[i], mask[i] = a.reshape(shape), 1
    return vol, mask, ref.affine


def load_seg(index: pd.DataFrame, study_id: str) -> np.ndarray:
    p = _path(index, study_id, "seg")
    if p is None:
        raise FileNotFoundError(f"no segmentation for {study_id}")
    return np.rint(np.asarray(nib.load(p).dataobj)).astype(np.int16)


def seg_spacing(index: pd.DataFrame, study_id: str) -> tuple[float, float, float]:
    p = _path(index, study_id, "seg")
    return tuple(float(z) for z in nib.load(p).header.get_zooms()[:3])
