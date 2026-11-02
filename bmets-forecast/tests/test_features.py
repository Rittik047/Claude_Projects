import numpy as np
import pandas as pd

from bmf.data.clinical import intervals, load_clinical
from bmf.features.radiomics import radiomics_features
from bmf.features.tables import clinical_design, lesion_summary
from bmf.lesions.tracking import track_patient

LES = lambda rows: pd.DataFrame(rows, columns=["lesion_id", "cx", "cy", "cz", "et_cc", "tc_cc", "wt_cc", "max_diam_mm"])


def _setup():
    clin = load_clinical("tests/fixtures/clinical_min.tsv")
    iv = intervals(clin)
    per, tracks = {}, []
    for cid, g in iv.groupby("case_id"):
        n = int(g.t_next.max())
        seq = [LES([(1, 10.0, 10.0, 10.0, 0.5 + 0.1 * k, 0.6, 1.2, 9.0)]) for k in range(n)]
        for k, d in enumerate(seq):
            per[(cid, k + 1)] = d
        tracks.append(track_patient(seq).assign(case_id=cid))
    return clin, iv, per, pd.concat(tracks)


def test_clinical_design_columns():
    clin, iv, _, _ = _setup()
    X = clinical_design(clin, iv)
    assert X.index.names == ["case_id", "t"] and len(X) == len(iv)
    assert {"age", "sex_male", "rt_between", "dt_years", "zero_interval", "n_prior_tp", "primary_lung"} <= set(X.columns)


def test_design_no_future_leak():
    clin, iv, per, tracks = _setup()
    a = lesion_summary(per, tracks, clin, iv)
    per2 = dict(per)
    for (cid, p), d in per.items():
        if p >= 2:
            per2[(cid, p)] = LES([(1, 10.0, 10.0, 10.0, 9.9, 9.9, 9.9, 30.0), (2, 40.0, 40.0, 40.0, 3.0, 3.0, 3.0, 10.0)])
    b = lesion_summary(per2, tracks, clin, iv)
    t1 = a.xs(1, level="t").columns
    pd.testing.assert_frame_equal(a.xs(1, level="t")[t1], b.xs(1, level="t")[t1])
    assert {"n_lesions", "sum_et", "d_sum_et", "edema_ratio", "new_per_year"} <= set(a.columns)


def test_simple_radiomics_keys(tmp_path):
    rng = np.random.default_rng(0)
    vols = {s: rng.normal(0, 1, (24, 24, 24)).astype(np.float32) for s in ("t1c", "t2f")}
    seg = np.zeros((24, 24, 24), np.int16); seg[8:16, 8:16, 8:16] = 3; seg[6:18, 6:18, 6:18][seg[6:18, 6:18, 6:18] == 0] = 2
    f1 = radiomics_features(vols, seg, seqs=("t1c", "t2f", "adc"), backend="simple")
    f2 = radiomics_features(vols, np.zeros_like(seg), seqs=("t1c", "t2f", "adc"), backend="simple")
    assert set(f1) == set(f2) and len(f1) > 20
    assert f1["shape_tc_volume_cc"] > 0 and np.isnan(f2["t1c_tc_mean"])
