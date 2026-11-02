import json

import numpy as np
import pandas as pd

from bmf.data.clinical import load_clinical, timepoints
from bmf.data.mri_index import build_index
from bmf.data.splits import make_splits
from bmf.eval.seg_metrics import lesionwise_metrics, sensitivity_by_size
from bmf.preprocess.pipeline import preprocess_study
from bmf.seg.nnunet_export import export_nnunet, nnunet_cmds
from bmf.synth import make_synthetic_cohort


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def _pp(tmp_path):
    out = make_synthetic_cohort(tmp_path / "s", n_patients=3, shape=(32, 32, 32), seed=2)
    clin = load_clinical(out["clinical"]); tp = timepoints(clin)
    idx = build_index(out["root"], tp)
    for cid, g in tp.groupby("case_id"):
        base = None
        for sid in g.sort_values("folder_idx").study_id:
            preprocess_study(idx, sid, base, tmp_path / "pp", n4=False, affine_longitudinal=False)
            base = base or sid
    return clin, tp


def test_export_layout(tmp_path):
    clin, tp = _pp(tmp_path)
    sp = make_splits(clin, n_folds=2, seed=0)
    for prior, nch in ((False, 4), (True, 6)):
        ds = export_nnunet(tmp_path / "pp", tp, sp, tmp_path / f"raw{int(prior)}", dataset_id=501 + prior, prior=prior)
        dj = json.loads((ds / "dataset.json").read_text())
        assert len(dj["channel_names"]) == nch
        n_cases = len(list((ds / "labelsTr").glob("*.nii.gz")))
        assert n_cases == len(tp)
        assert len(list((ds / "imagesTr").glob("*.nii.gz"))) == n_cases * nch


def test_splits_mirrored(tmp_path):
    clin, tp = _pp(tmp_path)
    sp = make_splits(clin, n_folds=2, seed=0)
    ds = export_nnunet(tmp_path / "pp", tp, sp, tmp_path / "raw", dataset_id=501)
    folds = json.loads((ds / "splits_final.json").read_text())
    for k, f in enumerate(folds):
        assert {s.rsplit("_", 1)[0] for s in f["val"]} == {c for c, v in sp.items() if v == k}
    cmds = nnunet_cmds(501, [0, 1])
    assert any("nnUNetv2_plan_and_preprocess" in c for c in cmds) and sum("nnUNetv2_train" in c for c in cmds) == 2


def test_lesionwise_perfect():
    gt = np.zeros((40, 40, 40), np.int16); gt[ball(gt.shape, (12, 12, 20), 4)] = 3; gt[ball(gt.shape, (28, 28, 20), 3)] = 3
    m = lesionwise_metrics(gt.copy(), gt, (1, 1, 1))
    assert m["ET"]["lesion_dice"] == 1.0 and m["ET"]["f1"] == 1.0


def test_lesionwise_missed_lesion():
    gt = np.zeros((40, 40, 40), np.int16); gt[ball(gt.shape, (12, 12, 20), 5)] = 3; gt[ball(gt.shape, (30, 30, 20), 1)] = 3
    pred = gt.copy(); pred[ball(gt.shape, (30, 30, 20), 1)] = 0
    m = lesionwise_metrics(pred, gt, (1, 1, 1))
    assert m["ET"]["f1"] < 1.0 and m["ET"]["fn"] == 1
    s = sensitivity_by_size(pred, gt, (1, 1, 1))
    assert s.set_index("bin").loc["<=3mm", "sensitivity"] == 0.0
    assert s.set_index("bin").loc[">10mm", "sensitivity"] == 1.0
