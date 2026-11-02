from pathlib import Path

import numpy as np
import pytest

from ecgtrust.data.external import (
    ExternalLabelError,
    external_labels,
    flat_leads,
    label_external_meta,
    load_label_map,
    parse_header_comments,
    resample_to_100,
)

ROOT = Path(__file__).resolve().parents[1]
CFG_MAP = ROOT / "configs" / "external_label_map.csv"
HDR = "#Age: 85\n#Sex: Male\n#Dx: 164889003,59118001,164934002\n#Rx: Unknown\n"


def good_signal():
    t = np.arange(1000) / 100
    return np.tile(np.sin(2 * np.pi * 1.2 * t), (12, 1)).astype(np.float32)


def test_parse_header():
    assert parse_header_comments(HDR) == {
        "age": 85.0,
        "sex": "male",
        "dx": [164889003, 59118001, 164934002],
    }


def test_unknown_demographics():  # Review Focus 2
    h = parse_header_comments("#Age: Unknown\n#Sex: Unknown\n#Dx: 426783006\n")
    assert h["age"] is None and h["sex"] == "unknown"


def test_missing_dx_line():
    assert parse_header_comments("#Age: 40\n#Sex: Female\n")["dx"] == []


def test_label_harmonisation():
    m = load_label_map(CFG_MAP)
    y = external_labels([164889003, 59118001, 164934002], m)  # AFIB, RBBB, TWC
    # strict (PTB-XL-faithful): isolated T-wave change is a form statement, not STTC
    assert y == {"NORM": False, "MI": False, "STTC": False, "CD": True, "HYP": False}
    lib = load_label_map(CFG_MAP, column="superclass_liberal")
    assert external_labels([164889003, 59118001, 164934002], lib)["STTC"]


def test_shared_codes_map_once():
    m = load_label_map(CFG_MAP)
    assert m[698252002] == "CD" and m[164909002] == "CD" and m[164865005] == "MI"


def test_conflicting_code_mapping_rejected(tmp_path):
    bad = tmp_path / "m.csv"
    bad.write_text("acronym,full_name,snomed_ct,superclass\nA,a,1,CD\nB,b,1,MI\n")
    with pytest.raises(ExternalLabelError):
        load_label_map(bad)


def test_norm_proxy_only_when_sinus_rhythm_alone():
    m = load_label_map(CFG_MAP)
    assert external_labels([426783006], m)["NORM"]
    assert not external_labels([426783006, 164873001], m)["NORM"]  # SR + LVH


def test_resample_preserves_sine():
    t = np.arange(5000) / 500
    x = np.tile(np.sin(2 * np.pi * 5 * t), (12, 1))
    y = resample_to_100(x, 500)
    assert y.shape == (12, 1000) and abs(np.abs(y[:, 100:900]).max() - 1) < 0.02


def test_resample_rejects_low_fs():
    with pytest.raises(ValueError):
        resample_to_100(np.zeros((12, 500)), 50)


def test_flatline_detected():  # Review Focus 2
    x = good_signal()
    x[3] = 0.0
    assert flat_leads(x) == [3] and flat_leads(good_signal()) == []


def test_codes_missing_from_condition_list_are_mapped():
    m = load_label_map(CFG_MAP)
    assert (
        m[713427006] == "CD" and m[445118002] == "CD" and m[733534002] == "CD"
    )  # CRBBB LAFB CLBBB
    assert m[6374002] == "CD" and m[713426002] == "CD"  # BBB, IRBBB
    assert m[54329005] == "MI" and m[57054005] == "MI"  # anterior MI, acute MI
    assert m[55930002] == "STTC" and m[446813000] == "HYP"  # ST changes, LA hypertrophy


def test_voltage_only_codes_are_hyp_only_in_liberal_map():
    strict, lib = load_label_map(CFG_MAP), load_label_map(CFG_MAP, column="superclass_liberal")
    assert strict[55827005] is None and lib[55827005] == "HYP"  # LV high voltage


def test_norm_proxy_accepts_sinus_family_only():
    m = load_label_map(CFG_MAP)
    assert external_labels([426177001], m)["NORM"]  # sinus bradycardia alone
    assert external_labels([427084000, 427393009], m)["NORM"]  # sinus tachy + arrhythmia
    assert not external_labels([426177001, 164889003], m)["NORM"]  # + AFIB


def test_label_external_meta_flags_unlabelled_records():
    import pandas as pd

    meta = pd.DataFrame({"dx": ["426177001", "164889003", "59118001,164934002", ""]})
    out = label_external_meta(meta, CFG_MAP)
    assert out.has_label.tolist() == [True, False, True, False]  # SB→NORM, AF-only→none
    assert out.loc[2, "CD"] and not out.loc[2, "STTC"] and out.loc[2, "STTC_liberal"]
    assert out.has_label_liberal.tolist() == [True, False, True, False]
