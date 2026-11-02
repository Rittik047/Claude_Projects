import numpy as np
import pandas as pd
import pytest

from bmf.data.clinical import intervals, load_clinical, timepoints
from bmf.data.mri_index import build_index, load_seg
from bmf.lesions.components import lesions
from bmf.lesions.labels import interval_labels, lesion_trajectories
from bmf.lesions.tracking import match, track_patient
from bmf.synth import make_synthetic_cohort

LAB = {"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def ball(shape, c, r):
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return sum((gi - ci) ** 2 for gi, ci in zip(g, c)) <= r * r


def seg_with(shape, specs):
    s = np.zeros(shape, np.int16)
    for c, r in specs:
        s[ball(shape, c, r)] = 3
    return s


def test_components_volume():
    s = seg_with((40, 40, 40), [((20, 20, 20), 5)])
    s[ball((40, 40, 40), (5, 5, 5), 2)] = 4                          # resection cavity ignored
    df = lesions(s, (1, 1, 1), LAB)
    assert len(df) == 1
    assert df.et_cc.iloc[0] == pytest.approx(4 / 3 * np.pi * 125 / 1000, rel=0.05)
    assert {"lesion_id", "cx", "cy", "cz", "et_cc", "tc_cc", "wt_cc", "max_diam_mm"} <= set(df.columns)


def test_match_gate():
    prev = pd.DataFrame({"lesion_id": [1], "cx": [0.0], "cy": [0.0], "cz": [0.0]})
    curr = pd.DataFrame({"lesion_id": [7], "cx": [15.0], "cy": [0.0], "cz": [0.0]})
    pairs = match(prev, curr, gate_mm=10.0)
    assert (1, None) in pairs and (None, 7) in pairs


def _per_tp(shape, timeline):
    return [lesions(seg_with(shape, specs), (1, 1, 1), LAB) for specs in timeline]


def _labels(per_tp, dts=None):
    n = len(per_tp)
    dts = dts or [0.2] * (n - 1)
    iv = pd.DataFrame({"case_id": "C", "t": range(1, n), "t_next": range(2, n + 1), "dt_years": dts,
                       "rt_between": False, "zero_interval": [d == 0 for d in dts]})
    tr = track_patient(per_tp)
    return tr, interval_labels(tr.assign(case_id="C"), {("C", k + 1): d for k, d in enumerate(per_tp)}, iv), iv


def test_new_lesion_flag():
    sh = (48, 48, 48)
    _, lab, _ = _labels(_per_tp(sh, [[((15, 15, 24), 4)], [((15, 15, 24), 4), ((34, 34, 24), 3)]]))
    assert lab.newmet_t2b.iloc[0] == 1 and lab.prog_t2a.iloc[0] == 1


def test_prog_threshold():
    sh = (48, 48, 48)
    _, up80, _ = _labels(_per_tp(sh, [[((24, 24, 24), 5)], [((24, 24, 24), 6.1)]]))      # (6.1/5)^3 = +82%
    _, up60, _ = _labels(_per_tp(sh, [[((24, 24, 24), 5)], [((24, 24, 24), 5.85)]]))     # +60%
    assert up80.prog_t2a.iloc[0] == 1
    assert up60.prog_t2a.iloc[0] == 0 and up60.newmet_t2b.iloc[0] == 0


def test_new_lesion_volume_floor():
    sh = (48, 48, 48)
    _, tiny, _ = _labels(_per_tp(sh, [[((15, 15, 24), 4)], [((15, 15, 24), 4), ((34, 34, 24), 1.8)]]))  # ~0.03 cc
    _, small, _ = _labels(_per_tp(sh, [[((15, 15, 24), 4)], [((15, 15, 24), 4), ((34, 34, 24), 2.5)]]))  # ~0.06 cc
    assert tiny.newmet_t2b.iloc[0] == 0 and tiny.prog_t2a.iloc[0] == 0
    assert small.newmet_t2b.iloc[0] == 1 and small.prog_t2a.iloc[0] == 1


def test_empty_timepoint_tracks():
    sh = (32, 32, 32)
    assert len(lesions(np.zeros(sh, np.int16), (1, 1, 1), LAB)) == 0
    tr, lab, iv = _labels(_per_tp(sh, [[], [((16, 16, 16), 3)], []]))
    assert lab.sum_et_t.iloc[0] == 0 and lab.newmet_t2b.iloc[0] == 1
    assert lab.sum_et_next.iloc[1] == 0


def test_zero_interval_growth_rate_finite():
    sh = (40, 40, 40)
    per = _per_tp(sh, [[((20, 20, 20), 4)], [((20, 20, 20), 5)]])
    tr, _, iv = _labels(per, dts=[0.0])
    traj = lesion_trajectories(tr.assign(case_id="C"), {("C", 1): per[0], ("C", 2): per[1]}, iv)
    assert np.isfinite(traj.log_ratio.iloc[0]) and np.isnan(traj.growth_rate_per_yr.iloc[0])


def test_rano_classes_per_lesion():
    sh = (48, 48, 48)
    per = _per_tp(sh, [[((15, 15, 24), 5), ((34, 34, 24), 5)], [((15, 15, 24), 3), ((34, 34, 24), 6.1)], [((34, 34, 24), 6.1)]])
    tr, _, iv = _labels(per)
    traj = lesion_trajectories(tr.assign(case_id="C"), {("C", k + 1): d for k, d in enumerate(per)}, iv)
    first = traj[traj.t == 1].sort_values("cx")
    assert list(first.rano_class) == ["PR", "PD"]
    assert "CR" in set(traj[traj.t == 2].rano_class)


def test_synth_truth_recovered(tmp_path):
    out = make_synthetic_cohort(tmp_path, n_patients=6, seed=0)
    clin = load_clinical(out["clinical"])
    tp = timepoints(clin)
    idx = build_index(out["root"], tp)
    found = set()
    for cid, g in tp[~tp.is_normal].groupby("case_id"):
        per = [lesions(load_seg(idx, s), (1, 1, 1), LAB) for s in g.sort_values("post_idx").study_id]
        tr = track_patient(per)
        for p in tr[tr.status == "new"].post_idx:
            found.add((cid, int(p)))
    truth = out["truth"]
    want = {(c, int(t)) for c, t in truth[truth.is_new][["case_id", "t"]].itertuples(index=False)}
    assert found == want


def test_match_gate_before_assignment():
    # review finding: ungated Hungarian picked far pairs (49 < 51) and then rejected both
    prev = pd.DataFrame({"lesion_id": [1, 2], "cx": [0.0, 20.0], "cy": [0.0, 0.0], "cz": [0.0, 0.0]})
    curr = pd.DataFrame({"lesion_id": [7, 8], "cx": [1.0, -30.0], "cy": [0.0, 0.0], "cz": [0.0, 0.0]})
    pairs = match(prev, curr, gate_mm=10.0)
    assert (1, 7) in pairs and (None, 8) in pairs and (2, None) in pairs
