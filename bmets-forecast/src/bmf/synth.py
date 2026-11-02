"""Synthetic longitudinal brain-metastasis cohort in the exact TCIA layout.

Used by tests and demos so that every pipeline stage runs without real data.
Images are simple (ellipsoidal "brain", spherical lesions with an oedema rim) but
the files, folder names, TSV columns and label values match the real collection.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from .data.clinical import _RENAME

_CLIN_COLS = [
    "Case ID", "Sex at Birth", "Ethnicity", "Race", "Diagnosis (primary malignancy)",
    "Diagnosis (Biopsy/Resction/Imaging/CSF)", "Age at Death", "MRI without metastases",
    "Age at imaging (MRI without metastases)", "Resection ", "Radiation before first MRI",
    "Age at Imaging (first MRI after diagnosis)", "Radiation treatment between first and second MRI",
    "Age at Imaging (second MRI after diagnosis)", "Radiation treatment between second and third MRI",
    "Age at Imaging (third MRI after diagnosis)", "Radiation treatment between third and fourth MRI",
    "Age at Imaging (fourth MRI after diagnosis)",
]
assert len(_CLIN_COLS) == len(_RENAME)
_META_COLS = ["Study ID", "Manufacturer", "Model", "Field Strength", "t1c", "t1n", "t2f", "t2w",
              "t1postfse", "dwi", "adc", "swan", "asl"]
_FILE_SEQ = {"t1c": "t1c", "t1n": "t1n", "t2f": "t2f", "t2w": "t2", "adc": "adc"}


def _ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def _render(shape, lesions, rng):
    brain = _ball(shape, [s / 2 for s in shape], min(shape) * 0.45)
    seg = np.zeros(shape, np.int16)
    for c, r in lesions:
        seg[_ball(shape, c, r + 2) & (seg == 0) & brain] = 2           # SNFH (oedema)
    for c, r in lesions:
        seg[_ball(shape, c, r) & brain] = 3                            # ET
        if r >= 4:
            seg[_ball(shape, c, r - 2) & brain] = 1                    # NETC core
    base = brain * 1.0
    imgs = {
        "t1n": base * 1.0 + (seg == 1) * -0.3,
        "t1c": base * 1.0 + (seg == 3) * 1.5,
        "t2f": base * 0.8 + (seg == 2) * 1.2 + (seg == 1) * 0.6,
        "t2w": base * 0.7 + (seg >= 1) * 0.8,
        "adc": base * 1.0 - (seg == 3) * 0.3,
    }
    return {k: (v + rng.normal(0, 0.05, shape) * brain).astype(np.float32) for k, v in imgs.items()}, seg


def make_synthetic_cohort(root: Path | str, n_patients: int = 6, shape=(48, 48, 48), seed: int = 0) -> dict:
    """Write images, segmentations and the two TSVs. Returns paths and the ground-truth lesion table."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    img_root = root / "UCSD-BMETS-Longitudinal"
    sites = [np.array(shape) * f for f in ([.35, .35, .5], [.65, .35, .5], [.35, .65, .5], [.65, .65, .5], [.5, .5, .3], [.5, .5, .7])]
    clin_rows, meta_rows, truth, n_cr = [], [], [], 0
    primaries = ["Lung", "Breast", "Melanoma", "Lung", "Breast", "Other"]
    for i in range(n_patients):
        cid = f"BMETS{8001 + i:04d}"
        n_post = [4, 4, 3, 2, 4, 3][i % 6]
        has_normal = i == 2
        rt = [bool(rng.random() < .4) for _ in range(3)]
        age1 = 50.0 + 3 * i
        ages = [age1 + 0.2 * k for k in range(n_post)]
        if i == 1:
            ages[1] = ages[0]                                     # a zero interval (same 0.1-y age)
        # lesion tracks: id -> (center, radius per post timepoint or 0 if absent)
        k_init = 1 + i % 3
        tracks = {j: [sites[j], [float(rng.integers(3, 6))] + [0.0] * (n_post - 1)] for j in range(k_init)}
        for j, (c, rs) in tracks.items():
            for t in range(1, n_post):
                rs[t] = min(6.0, max(0.0, rs[t - 1] + float(rng.choice([-1, 0, 1, 2]))))
        if i == 0:                                                # guaranteed new lesion at post 2
            tracks[5] = [sites[5], [0.0, 3.0] + [4.0] * (n_post - 2)]
        if i == 1:                                                # guaranteed complete response at post 3
            tracks[0][1] = [5.0, 4.0, 0.0, 0.0][:n_post]
        folder = 0
        if has_normal:
            folder += 1
            sid = f"{cid}_{folder:02d}"
            imgs, seg = _render(shape, [], rng)
            _write(img_root, sid, imgs, seg)
            meta_rows.append(_meta_row(sid, i))
        for t in range(n_post):
            folder += 1
            sid = f"{cid}_{folder:02d}"
            les = [(c, rs[t]) for c, rs in tracks.values() if rs[t] > 0]
            imgs, seg = _render(shape, les, rng)
            _write(img_root, sid, imgs, seg)
            meta_rows.append(_meta_row(sid, i))
            for j, (c, rs) in tracks.items():
                if rs[t] > 0:
                    vol = float(_ball(shape, c, rs[t]).sum()) / 1000.0
                    truth.append((cid, t + 1, j, vol, t > 0 and rs[t - 1] == 0 and all(x == 0 for x in rs[:t])))
                elif t > 0 and rs[t - 1] > 0:
                    n_cr += 1
        dead = i % 2 == 0
        row = [cid, ["Female", "Male"][i % 2], "Not Hispanic, Latino(a), or Spanish origin\xa0", "White\xa0",
               primaries[i % 6], "Imaging", f"{ages[-1] + 0.4:.1f}" if dead else "",
               "Yes" if has_normal else "No", f"{age1 - 1:.1f}" if has_normal else "", "No", "No"]
        for t in range(4):
            row.append(f"{ages[t]:.1f}" if t < n_post else "")
            if t < 3:
                row.append(("Yes" if rt[t] else "No") if t + 1 < n_post else "")
        clin_rows.append(row)
    clin_path, meta_path = root / "clinical.tsv", root / "mri_meta.tsv"
    with open(clin_path, "wb") as f:
        f.write(("\t".join(_CLIN_COLS) + "\n").encode("cp1252"))
        for r in clin_rows:
            f.write(("\t".join(r) + "\n").encode("cp1252"))
    pd.DataFrame(meta_rows, columns=_META_COLS).to_csv(meta_path, sep="\t", index=False)
    tdf = pd.DataFrame(truth, columns=["case_id", "t", "lesion_id", "volume_cc", "is_new"])
    return {"root": img_root, "clinical": clin_path, "meta": meta_path, "truth": tdf, "n_complete_response": n_cr}


def _meta_row(sid, i):
    return [sid, "GE Medical Systems", "Signa HDxt", "3" if i % 2 else "1.5", "Yes", "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes", "Yes", "No"]


def _write(img_root: Path, sid: str, imgs: dict, seg: np.ndarray):
    d = img_root / sid
    d.mkdir(parents=True, exist_ok=True)
    aff = np.eye(4)
    for k, v in imgs.items():
        nib.save(nib.Nifti1Image(v, aff), str(d / f"{sid}-{_FILE_SEQ[k]}.nii.gz"))
    nib.save(nib.Nifti1Image(seg, aff), str(d / f"{sid}-seg.nii.gz"))
