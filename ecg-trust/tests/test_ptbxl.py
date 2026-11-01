from pathlib import Path

import pytest

from ecgtrust import CLASSES
from ecgtrust.data.ptbxl import age_band, load_metadata, load_scp, superclass_labels

FIXTURE = Path(__file__).parent / "fixtures" / "ptbxl"
ID_MI_AND_CD, ID_RHYTHM_ONLY, ID_LIKELIHOOD_ZERO = 2, 3, 4


@pytest.fixture()
def meta():
    return load_metadata(FIXTURE)


@pytest.fixture()
def scp():
    return load_scp(FIXTURE)


def test_age_300_maps_to_over_89():
    assert age_band(300) == ">89" and age_band(89) == "80-89" and age_band(39) == "<40"
    assert (
        age_band(40) == "40-59" and age_band(79) == "60-79" and age_band(float("nan")) == "unknown"
    )


def test_multilabel_mapping(meta, scp):
    y = superclass_labels(meta, scp)
    assert list(y.columns) == list(CLASSES)
    assert y.loc[ID_MI_AND_CD, ["MI", "CD"]].all() and not y.loc[ID_MI_AND_CD, "NORM"]


def test_likelihood_is_ignored(meta, scp):
    # Benchmark convention (Strodthoff et al. 2021): any listed diagnostic code counts.
    assert superclass_labels(meta, scp).loc[ID_LIKELIHOOD_ZERO, "STTC"]


def test_record_with_only_rhythm_codes_has_no_label(meta):
    assert not meta.loc[ID_RHYTHM_ONLY, "has_label"] and meta.loc[ID_MI_AND_CD, "has_label"]


def test_metadata_columns(meta):
    assert meta.loc[2, "age_band"] == ">89" and meta.loc[1, "sex_label"] == "female"
    assert meta.loc[2, "sex_label"] == "male" and meta.index.name == "ecg_id"
