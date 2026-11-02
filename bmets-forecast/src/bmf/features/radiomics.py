"""Study-level radiomics on the whole lesion burden (TC and WT masks).

backend='pyradiomics' uses PyRadiomics (optional install, IBSI-compliant, used for the paper);
backend='simple' is a dependency-light fallback (first-order, shape, 2D GLCM) used in tests.
Feature keys are identical whether or not lesions are present (NaN when absent).
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats

REG = {"tc": (1, 3), "wt": (1, 2, 3)}
_FO = ("mean", "std", "p10", "p90", "skew", "kurt")
_GLCM = ("contrast", "homogeneity", "energy", "correlation")


def load_pp_study(pp_dir: Path | str, seqs) -> tuple[dict, np.ndarray]:
    d = Path(pp_dir)
    vols = {s: np.asarray(nib.load(str(d / f"{s}.nii.gz")).dataobj, np.float32) for s in seqs if (d / f"{s}.nii.gz").exists()}
    return vols, np.rint(np.asarray(nib.load(str(d / "seg.nii.gz")).dataobj)).astype(np.int16)


def _first_order(v):
    if v.size < 3:
        return dict.fromkeys(_FO, np.nan)
    return {"mean": float(v.mean()), "std": float(v.std()), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "skew": float(stats.skew(v)), "kurt": float(stats.kurtosis(v))}


def _glcm(img, mask, levels=32):
    from skimage.feature import graycomatrix, graycoprops
    if mask.sum() < 10:
        return dict.fromkeys(_GLCM, np.nan)
    v = img[mask]; lo, hi = np.percentile(v, [1, 99])
    q = np.clip((img - lo) / (hi - lo + 1e-8) * (levels - 1), 0, levels - 1).astype(np.uint8)
    acc = {k: [] for k in _GLCM}
    for z in np.where(mask.any(axis=(0, 1)))[0]:
        m = mask[:, :, z]
        if m.sum() < 4:
            continue
        sl = np.where(m, q[:, :, z], 0)
        P = graycomatrix(sl, [1], [0, np.pi / 2], levels=levels, symmetric=True, normed=True)
        for k in _GLCM:
            acc[k].append(float(np.nanmean(graycoprops(P, k))))
    return {k: float(np.mean(v)) if v else np.nan for k, v in acc.items()}


def radiomics_features(vols: dict, seg: np.ndarray, seqs=("t1c", "t2f", "adc"), backend: str = "simple",
                       spacing=(1.0, 1.0, 1.0)) -> dict:
    if backend == "pyradiomics":
        return _pyradiomics(vols, seg, seqs, spacing)
    out = {}
    vox_cc = float(np.prod(spacing)) / 1000
    for r, labs in REG.items():
        m = np.isin(seg, labs)
        out[f"shape_{r}_volume_cc"] = float(m.sum() * vox_cc)
        if m.any():
            from skimage.measure import marching_cubes, mesh_surface_area
            try:
                vs, fs, _, _ = marching_cubes(m.astype(np.float32), 0.5, spacing=tuple(spacing))
                area = float(mesh_surface_area(vs, fs))
            except (RuntimeError, ValueError):
                area = np.nan
            vol_mm3 = m.sum() * np.prod(spacing)
            out[f"shape_{r}_sphericity"] = float(np.pi ** (1 / 3) * (6 * vol_mm3) ** (2 / 3) / area) if area and area > 0 else np.nan
        else:
            out[f"shape_{r}_sphericity"] = np.nan
        for s in seqs:
            v = vols[s][m] if s in vols else np.array([])
            out.update({f"{s}_{r}_{k}": val for k, val in _first_order(v).items()})
    for s in seqs:
        g = _glcm(vols[s], np.isin(seg, REG["tc"])) if s in vols else dict.fromkeys(_GLCM, np.nan)
        out.update({f"{s}_tc_glcm_{k}": val for k, val in g.items()})
    return out


def _pyradiomics(vols, seg, seqs, spacing):  # pragma: no cover - optional dependency, run on Juno/Colab
    import SimpleITK as sitk
    from radiomics import featureextractor
    ext = featureextractor.RadiomicsFeatureExtractor(binWidth=0.25, normalize=False)
    ext.disableAllFeatures(); ext.enableFeatureClassByName("firstorder"); ext.enableFeatureClassByName("shape")
    ext.enableFeatureClassByName("glcm"); ext.enableFeatureClassByName("glszm")
    out = {}
    for r, labs in REG.items():
        m = np.isin(seg, labs).astype(np.uint8)
        if m.sum() < 10:
            continue
        mi = sitk.GetImageFromArray(m); mi.SetSpacing(spacing)
        for s in seqs:
            if s not in vols:
                continue
            ii = sitk.GetImageFromArray(vols[s]); ii.SetSpacing(spacing)
            for k, v in ext.execute(ii, mi).items():
                if not k.startswith("diagnostics"):
                    out[f"{s}_{r}_{k}"] = float(v)
    return out


def radiomics_or_empty(vols, seg, seqs, backend="pyradiomics", study="?", **kw) -> dict:
    """Per-study guard for the real run: one failing mask/image logs a warning and yields NaNs, not a crash."""
    import logging
    try:
        if backend not in ("simple", "pyradiomics"):
            raise ValueError(f"unknown radiomics backend {backend}")
        return radiomics_features(vols, seg, seqs, backend=backend, **kw)
    except Exception as e:  # noqa: BLE001
        logging.getLogger("bmf").warning("radiomics failed for %s: %s", study, e)
        return {}
