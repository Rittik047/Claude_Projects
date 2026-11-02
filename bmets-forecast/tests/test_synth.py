from bmf.data.clinical import load_clinical, load_mri_meta, timepoints
from bmf.data.mri_index import build_index, load_seg
from bmf.synth import make_synthetic_cohort


def test_synth_roundtrip(tmp_path):
    out = make_synthetic_cohort(tmp_path, n_patients=6, shape=(48, 48, 48), seed=0)
    clin = load_clinical(out["clinical"])
    meta = load_mri_meta(out["meta"])
    tp = timepoints(clin)
    assert len(clin) == 6 and set(tp.study_id) == set(meta.study_id)
    idx = build_index(out["root"], tp)
    assert idx[idx.seq == "seg"].exists.all()
    seg = load_seg(idx, tp.study_id.iloc[-1])
    assert seg.shape == (48, 48, 48)
    truth = out["truth"]
    assert truth.is_new.any()
    # at least one complete response: a track present at t and absent at t+1
    assert out["n_complete_response"] >= 1
    assert clin.has_normal_mri.any()
