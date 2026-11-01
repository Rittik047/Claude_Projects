import json

import numpy as np
import pandas as pd
import pytest

from ecgtrust.evaluate.report import evaluate_split
from ecgtrust.evaluate.subgroups import subgroup_table
from ecgtrust.freeze import TestSetLockedError


def test_small_subgroup_flagged_insufficient():
    rng = np.random.default_rng(0)
    n = 400
    meta = pd.DataFrame({"device": ["big"] * 390 + ["tiny"] * 10})
    y = (rng.random((n, 5)) < 0.4).astype(int)
    p = np.clip(y * 0.5 + rng.random((n, 5)) * 0.5, 0, 1)
    t = subgroup_table(meta, y, p, ["device"], np.arange(n), min_pos=30, n_boot=50)
    assert (t.loc[t.group == "tiny", "status"] == "insufficient").all()
    assert (t.loc[t.group == "big", "status"] == "ok").all()
    assert t.loc[t.group == "tiny", "auroc"].isna().all()


def test_noise_columns_binarised():
    meta = pd.DataFrame({"static_noise": [None, "x", None, "y"] * 50})
    y = np.tile([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], (100, 1))
    t = subgroup_table(
        meta, y, y.astype(float), ["static_noise"], np.arange(200), min_pos=5, n_boot=10
    )
    assert set(t.group) == {"yes", "no"}


def test_metrics_json_has_no_nan(tmp_cfg_val):  # Review Focus 5
    evaluate_split(tmp_cfg_val, "val_select")
    txt = (tmp_cfg_val.reports / "val_select" / "metrics.json").read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    m = json.loads(txt)
    assert m["calibration"]["macro_ece_post"] < m["calibration"]["macro_ece_pre"]
    assert set(m["conformal"]) == {"0.05", "0.1"}


def test_evaluate_test_requires_freeze(tmp_cfg_unfrozen):
    with pytest.raises(TestSetLockedError):
        evaluate_split(tmp_cfg_unfrozen, "test")


def test_calibrators_fit_on_val_calib_only(tmp_cfg_val, monkeypatch):
    import ecgtrust.evaluate.report as rep
    from ecgtrust.models.ensemble import load_preds

    seen = []
    real = rep.fit_temperature
    monkeypatch.setattr(
        rep, "fit_temperature", lambda lg, y: seen.append(lg.shape[1]) or real(lg, y)
    )
    evaluate_split(tmp_cfg_val, "val_select")
    n_calib = len(load_preds(tmp_cfg_val.artifacts / "preds" / "resnet_val_calib.npz")["y"])
    assert seen == [n_calib]
