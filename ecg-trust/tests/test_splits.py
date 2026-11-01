import numpy as np
import pandas as pd
import pytest

from ecgtrust.data.splits import LeakageError, assert_no_patient_leakage, make_splits


@pytest.fixture()
def meta_synthetic():
    rng = np.random.default_rng(1)
    n = 400
    patient = rng.integers(0, 250, n)
    fold = (patient % 10) + 1  # one fold per patient, as in PTB-XL
    return pd.DataFrame(
        {"patient_id": patient, "strat_fold": fold, "has_label": rng.random(n) > 0.05},
        index=pd.Index(np.arange(1, n + 1), name="ecg_id"),
    )


def test_folds_map_to_splits(meta_synthetic):
    s = make_splits(meta_synthetic)
    assert set(s[meta_synthetic.strat_fold <= 8].unique()) <= {"train", "excluded"}
    assert set(s[meta_synthetic.strat_fold == 10].unique()) <= {"test", "excluded"}
    assert set(s[meta_synthetic.strat_fold == 9].unique()) <= {
        "val_select",
        "val_calib",
        "excluded",
    }


def test_unlabelled_records_excluded(meta_synthetic):
    s = make_splits(meta_synthetic)
    assert (s[~meta_synthetic.has_label] == "excluded").all()
    assert (s[meta_synthetic.has_label] != "excluded").all()


def test_fold9_split_by_patient_half(meta_synthetic):
    s = make_splits(meta_synthetic)
    f9 = meta_synthetic[meta_synthetic.strat_fold == 9]
    pats = f9.groupby(s[f9.index]).patient_id.nunique()
    assert abs(pats["val_select"] - pats["val_calib"]) <= 1


def test_leakage_detected():
    meta = pd.DataFrame({"patient_id": [1, 1, 2]}, index=pd.Index([10, 11, 12], name="ecg_id"))
    splits = pd.Series(["train", "test", "test"], index=meta.index)
    with pytest.raises(LeakageError):
        assert_no_patient_leakage(meta, splits)


def test_no_leakage_passes(meta_synthetic):
    assert_no_patient_leakage(meta_synthetic, make_splits(meta_synthetic))


def test_split_is_deterministic(meta_synthetic):
    assert make_splits(meta_synthetic, 0).equals(make_splits(meta_synthetic, 0))
