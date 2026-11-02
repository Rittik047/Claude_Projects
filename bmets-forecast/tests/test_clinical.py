from pathlib import Path

import numpy as np
import pytest

from bmf.data.clinical import intervals, load_clinical, load_mri_meta, survival_table, timepoints
from bmf.data.audit import audit

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def clin():
    return load_clinical(FIX / "clinical_min.tsv")


def test_load_clinical_shapes(clin):
    assert len(clin) == 5
    row = clin.set_index("case_id").loc["BMETS9002"]
    assert np.isnan(row["age_t3"])
    assert row["age_t2"] == pytest.approx(50.2)


def test_dirty_values_normalized(clin):
    c = clin.set_index("case_id")
    assert c.loc["BMETS9002", "dx_method"] == "resection"
    assert c.loc["BMETS9003", "dx_method"] == "resection"
    assert c.loc["BMETS9003", "rt_12"] is False or c.loc["BMETS9003", "rt_12"] == False  # "No " -> False
    assert c.loc["BMETS9001", "race"] == "white"
    assert c.loc["BMETS9002", "resection"] == True


def test_timepoint_remap_with_normal_mri(clin):
    tp = timepoints(clin).set_index("study_id")
    assert tp.loc["BMETS9003_01", "is_normal"] == True
    assert tp.loc["BMETS9003_01", "post_idx"] == 0
    assert tp.loc["BMETS9003_02", "post_idx"] == 1
    assert tp.loc["BMETS9003_04", "post_idx"] == 3
    assert tp.loc["BMETS9001_01", "post_idx"] == 1
    assert len(tp) == 16


def test_intervals(clin):
    iv = intervals(clin)
    r = iv[(iv.case_id == "BMETS9001") & (iv.t == 2)].iloc[0]
    assert r.t_next == 3
    assert r.dt_years == pytest.approx(0.2, abs=1e-6)
    assert r.rt_between == True
    z = iv[(iv.case_id == "BMETS9003") & (iv.t == 1)].iloc[0]
    assert z.zero_interval == True
    assert len(iv) == 3 + 1 + 2 + 3 + 1


def test_survival_table(clin):
    s = survival_table(clin).set_index("case_id")
    assert s.loc["BMETS9001", "event"] == 1
    assert s.loc["BMETS9001", "time_years"] == pytest.approx(2.0, abs=1e-6)
    assert s.loc["BMETS9004", "event"] == 0
    assert s.loc["BMETS9004", "time_years"] == pytest.approx(0.5, abs=1e-6)


def test_audit_counts(clin, tmp_path):
    meta = load_mri_meta(FIX / "mri_meta_min.tsv")
    a = audit(clin, meta, out=tmp_path / "audit.json")
    assert a["n_patients"] == 5
    assert a["n_timepoints"] == 16
    assert a["n_intervals"] == 10
    assert a["n_deaths"] == 3
    assert a["n_rt_intervals"] == 4
    assert (tmp_path / "audit.json").exists()
    assert meta["asl"].dtype == bool
