"""Preprocessing: 1 mm isotropic resampling, intra-study and longitudinal rigid+affine
registration (to the patient's baseline t1c), optional N4, z-score in the brain mask.

Images are already skull-stripped (HD-BET) by TCIA, so the brain mask is intensity > 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..data.mri_index import SEQS_CORE, SEQS_OPT


def resample_iso(img: sitk.Image, spacing: float = 1.0, is_label: bool = False, ref: sitk.Image | None = None) -> sitk.Image:
    if ref is None:
        sp = np.array(img.GetSpacing()); sz = np.array(img.GetSize())
        new_sz = [int(round(s)) for s in sz * sp / spacing]
        ref = sitk.Image(new_sz, img.GetPixelID())
        ref.SetSpacing([spacing] * 3); ref.SetOrigin(img.GetOrigin()); ref.SetDirection(img.GetDirection())
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline
    return sitk.Resample(img, ref, sitk.Transform(), interp, 0.0, img.GetPixelID())


def n4_correct(img: sitk.Image, mask: sitk.Image, shrink: int = 4) -> sitk.Image:
    im = sitk.Cast(img, sitk.sitkFloat32)
    small = sitk.Shrink(im, [shrink] * 3); msmall = sitk.Shrink(mask, [shrink] * 3)
    corr = sitk.N4BiasFieldCorrectionImageFilter()
    corr.Execute(small, msmall)
    log_bias = corr.GetLogBiasFieldAsImage(im)
    return im / sitk.Exp(log_bias)


def zscore(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    v = arr[mask]
    if v.size:
        out[mask] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def register_rigid_affine(fixed: sitk.Image, moving: sitk.Image, backend: str = "sitk") -> sitk.Transform:
    """Rigid then affine registration (Mattes MI). backend='ants' uses antspyx when installed."""
    if backend == "ants":
        import ants  # optional dependency
        f = ants.from_numpy(sitk.GetArrayFromImage(fixed).T, spacing=fixed.GetSpacing())
        m = ants.from_numpy(sitk.GetArrayFromImage(moving).T, spacing=moving.GetSpacing())
        r = ants.registration(f, m, type_of_transform="Affine")
        return sitk.ReadTransform(r["fwdtransforms"][0])
    f = sitk.Cast(fixed, sitk.sitkFloat32); m = sitk.Cast(moving, sitk.sitkFloat32)
    tx = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    out = None
    for stage in (tx, sitk.AffineTransform(3)):
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM); reg.SetMetricSamplingPercentage(0.25, seed=0)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, 200, relaxationFactor=0.6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([2, 1]); reg.SetSmoothingSigmasPerLevel([1, 0])
        if out is None:
            reg.SetInitialTransform(stage, inPlace=False)
            out = reg.Execute(f, m)
        else:
            reg.SetMovingInitialTransform(out)
            reg.SetInitialTransform(stage, inPlace=False)
            aff = reg.Execute(f, m)
            comp = sitk.CompositeTransform(out)
            comp.AddTransform(aff)
            comp.FlattenTransform()
            out = comp
    return out


def _read(index: pd.DataFrame, sid: str, seq: str) -> sitk.Image | None:
    r = index[(index.study_id == sid) & (index.seq == seq) & index.exists]
    return sitk.ReadImage(r.path.iloc[0]) if len(r) else None


def preprocess_study(index: pd.DataFrame, study_id: str, ref_study_id: str | None, out_dir: Path | str,
                     spacing: float = 1.0, n4: bool = True, backend: str = "sitk", affine_longitudinal: bool = True) -> Path:
    """Write <out>/<study_id>/{<seq>,seg,brainmask}.nii.gz on the baseline grid of the patient.

    ref_study_id=None -> this study is the patient's baseline (grid = its own t1c at `spacing`).
    Otherwise t1c is registered to <out>/<ref_study_id>/t1c.nii.gz and the transform applied
    to all sequences (B-spline) and to the segmentation (nearest neighbour).
    """
    out_dir = Path(out_dir); d = out_dir / study_id; d.mkdir(parents=True, exist_ok=True)
    t1c = _read(index, study_id, "t1c")
    if t1c is None:
        raise FileNotFoundError(f"{study_id}: t1c missing")
    if ref_study_id is None:
        ref = resample_iso(sitk.Cast(t1c, sitk.sitkFloat32), spacing)
        tx = sitk.Transform()
    else:
        ref = sitk.ReadImage(str(out_dir / ref_study_id / "t1c_raw.nii.gz"))
        tx = register_rigid_affine(ref, sitk.Cast(t1c, sitk.sitkFloat32), backend) if affine_longitudinal else sitk.Transform()
    brain = None
    for seq in SEQS_CORE + SEQS_OPT:
        im = _read(index, study_id, seq)
        if im is None:
            continue
        im = sitk.Cast(im, sitk.sitkFloat32)
        if seq != "t1c" and im.GetSize() != t1c.GetSize():            # intra-study alignment if grids differ
            im = sitk.Resample(im, t1c, register_rigid_affine(sitk.Cast(t1c, sitk.sitkFloat32), im, backend), sitk.sitkLinear, 0.0)
        r = sitk.Resample(im, ref, tx, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
        a = sitk.GetArrayFromImage(r)
        if seq == "t1c":
            sitk.WriteImage(r, str(d / "t1c_raw.nii.gz"))
            brain = a > (a.max() * 1e-3)
            mimg = sitk.GetImageFromArray(brain.astype(np.uint8)); mimg.CopyInformation(r)
            sitk.WriteImage(mimg, str(d / "brainmask.nii.gz"))
        if n4 and seq in ("t1c", "t1n", "t2f", "t2w"):
            mimg = sitk.GetImageFromArray(brain.astype(np.uint8)); mimg.CopyInformation(r)
            a = sitk.GetArrayFromImage(n4_correct(r, mimg))
        z = sitk.GetImageFromArray(zscore(a, brain)); z.CopyInformation(r)
        sitk.WriteImage(z, str(d / f"{seq}.nii.gz"))
    seg = _read(index, study_id, "seg")
    if seg is not None:
        s = sitk.Resample(seg, ref, tx, sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
        sitk.WriteImage(s, str(d / "seg.nii.gz"))
    sitk.WriteTransform(tx, str(d / "to_baseline.h5"))
    return d
