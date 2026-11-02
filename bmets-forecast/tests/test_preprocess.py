import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from bmf.data.clinical import load_clinical, timepoints
from bmf.data.mri_index import build_index
from bmf.preprocess.pipeline import preprocess_study, register_rigid_affine, resample_iso, zscore
from bmf.synth import make_synthetic_cohort


def _img(a, spacing=(1.0, 1.0, 1.0)):
    im = sitk.GetImageFromArray(a)
    im.SetSpacing(spacing)
    return im


def test_resample_label_nearest():
    a = np.zeros((10, 12, 14), np.int16)
    a[3:6, 4:8, 5:9] = 3
    a[6:8, 4:8, 5:9] = 1
    out = sitk.GetArrayFromImage(resample_iso(_img(a, (0.7, 1.3, 2.0)), 1.0, is_label=True))
    assert set(np.unique(out)) <= {0, 1, 3}


def test_zscore_inside_mask():
    rng = np.random.default_rng(0)
    a = rng.normal(5, 3, (20, 20, 20)).astype(np.float32)
    m = np.zeros_like(a, bool); m[5:15, 5:15, 5:15] = True
    z = zscore(a, m)
    assert abs(z[m].mean()) < 1e-4 and abs(z[m].std() - 1) < 1e-3
    assert np.all(z[~m] == 0)


def test_register_recovers_shift():
    g = np.zeros((40, 40, 40), np.float32)
    g[10:30, 12:28, 14:26] = 1.0
    g[18:22, 18:22, 18:22] = 2.0
    g = ndi.gaussian_filter(g, 1.5)
    moved = ndi.shift(g, (0, 3, 0), order=1)
    fixed, moving = _img(g), _img(moved)
    tx = register_rigid_affine(fixed, moving, backend="sitk")
    res = sitk.GetArrayFromImage(sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0))
    com_f = np.array(ndi.center_of_mass(g)); com_r = np.array(ndi.center_of_mass(res))
    assert np.abs(com_f - com_r).max() < 1.0


def test_preprocess_study_writes_outputs(tmp_path):
    out = make_synthetic_cohort(tmp_path / "s", n_patients=1, shape=(32, 32, 32), seed=1)
    tp = timepoints(load_clinical(out["clinical"]))
    idx = build_index(out["root"], tp)
    sids = list(tp.study_id)
    d0 = preprocess_study(idx, sids[0], None, tmp_path / "pp", n4=False)
    d1 = preprocess_study(idx, sids[1], sids[0], tmp_path / "pp", n4=False)
    for d in (d0, d1):
        assert (d / "t1c.nii.gz").exists() and (d / "seg.nii.gz").exists() and (d / "brainmask.nii.gz").exists()
