"""Pre-flight checks on the real data before any long run (`bmf check`).

check_labels  - does the segmentation follow BraTS-METS 2025 {1 NETC, 2 SNFH, 3 ET, 4 RC}?  Value sets alone
                cannot tell a permuted scheme apart, so each label is also tested against MRI contrast physics:
                enhancing tumour is the brightest compartment on post-contrast T1 (gadolinium leaks through the
                disrupted blood-brain barrier), and vasogenic oedema is hyperintense on FLAIR (excess free water).
check_weights - is each foundation-model checkpoint present, loadable into the right architecture, and does it
                give a finite embedding of the expected size?
"""
from __future__ import annotations

import logging

import nibabel as nib
import numpy as np
import pandas as pd

log = logging.getLogger("bmf")
_ALLOWED = {0, 1, 2, 3, 4}


def _z(a: np.ndarray, brain: np.ndarray) -> np.ndarray:
    v = a[brain]
    return (a - v.mean()) / (v.std() + 1e-6)


def check_labels(index: pd.DataFrame, labels: dict, n: int = 20, seed: int = 2026) -> dict:
    """Sample `n` segmented studies; per label, median z-scored intensity on t1c and t2f inside the brain."""
    ok_paths = index[index.exists.astype(str) == "True"]
    segs = ok_paths[ok_paths.seq == "seg"]
    if segs.empty:
        return {"ok": False, "problems": ["no segmentation files found (check data_root and seg_glob)"],
                "values": [], "per_label": {}, "n_studies": 0}
    sids = segs.study_id.drop_duplicates().sample(n=min(n, segs.study_id.nunique()), random_state=seed)
    name = {v: k for k, v in labels.items()}
    values, rows, notes = set(), [], []
    for sid in sorted(sids):
        p = ok_paths[ok_paths.study_id == sid].set_index("seq").path
        if not {"t1c", "t2f"} <= set(p.index):
            notes.append(f"{sid}: t1c or t2f missing, skipped"); continue
        seg = np.rint(np.asarray(nib.load(p["seg"]).dataobj)).astype(int)
        t1c = np.asarray(nib.load(p["t1c"]).dataobj, np.float32)
        t2f = np.asarray(nib.load(p["t2f"]).dataobj, np.float32)
        if not (seg.shape == t1c.shape == t2f.shape):
            notes.append(f"{sid}: seg/t1c/t2f grids differ, skipped"); continue
        brain = (t1c > 0) | (seg > 0)
        zc, zf = _z(t1c, brain), _z(t2f, brain)
        vs = set(np.unique(seg).tolist()); values |= vs
        for v in sorted(vs - {0}):
            m = seg == v
            rows.append((sid, v, int(m.sum()), float(np.median(zc[m])), float(np.median(zf[m]))))
    df = pd.DataFrame(rows, columns=["study_id", "value", "voxels", "t1c_z", "t2f_z"])
    per = {}
    for v, g in df.groupby("value"):
        per[name.get(v, f"label {v}")] = {"value": int(v), "n_studies": int(len(g)), "voxels_median": float(g.voxels.median()),
                                          "t1c_z": float(g.t1c_z.median()), "t2f_z": float(g.t2f_z.median())}
    problems = []
    if values - _ALLOWED:
        problems.append(f"unexpected label values {sorted(values - _ALLOWED)} (expected subset of {sorted(_ALLOWED)})")
    t1 = {d["value"]: d["t1c_z"] for d in per.values()}
    et, ed, nc, rc = labels["ET"], labels["SNFH"], labels["NETC"], labels.get("RC", 4)
    if rc in t1 and t1[rc] == max(t1.values()) and t1.get(et, -np.inf) < t1[rc]:
        problems.append(f"label 4 is the most enhancing label on t1c: this looks like the old BraTS scheme "
                        f"(1 NCR, 2 ED, 4 ET); remap 4->3 or set labels.ET: 4 in the config")
    if et not in t1:
        problems.append(f"no ET (label {et}) voxels in {len(sids)} studies; metastases almost always enhance")
    else:
        rivals = {k: t1[v] for k, v in (("NETC", nc), ("SNFH", ed)) if v in t1}
        worse = [k for k, z in rivals.items() if z >= t1[et]]
        if worse:
            problems.append(f"ET (label {et}) is not the most enhancing label on t1c (median z {t1[et]:.2f} vs "
                            + ", ".join(f"{k} {rivals[k]:.2f}" for k in worse) + "); labels may be permuted")
    sn = per.get("SNFH")
    if sn and sn["t2f_z"] <= 0:
        problems.append(f"SNFH (label {ed}) is not FLAIR-hyperintense (median z {sn['t2f_z']:.2f}); labels may be permuted")
    return {"ok": not problems, "problems": problems, "values": sorted(int(v) for v in values), "per_label": per,
            "n_studies": int(df.study_id.nunique()) if len(df) else 0, "notes": notes}


def check_weights(cfg: dict) -> dict:
    """For every encoder named in fm.weights: missing / ok (with matched fraction and embedding size) / error."""
    from pathlib import Path

    import torch

    from .fm.registry import ENCODERS, load_encoder
    fm = cfg.get("fm", {})
    wdir = Path(fm.get("weights_dir", "cache/weights"))
    out = {}
    for enc, fname in (fm.get("weights") or {}).items():
        path = wdir / fname
        spec = ENCODERS.get(enc)
        if spec is None:
            out[enc] = {"status": "error", "message": f"unknown encoder {enc}"}; continue
        if not path.exists():
            out[enc] = {"status": "missing", "path": str(path), "download": spec.url}; continue
        try:
            model = load_encoder(enc, path)
            with torch.no_grad():
                e = model(torch.randn(1, spec.in_channels, spec.patch, spec.patch, spec.patch))
            good = bool(torch.isfinite(e).all()) and e.shape[-1] == spec.out_dim
            out[enc] = {"status": "ok" if good else "error", "path": str(path),
                        "matched_fraction": round(float(getattr(model, "load_fraction", 1.0)), 3),
                        "embedding_dim": int(e.shape[-1]), "expected_dim": spec.out_dim}
        except Exception as ex:  # noqa: BLE001
            out[enc] = {"status": "error", "path": str(path), "message": f"{type(ex).__name__}: {ex}"[:400]}
    return out
