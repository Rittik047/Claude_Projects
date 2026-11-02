from pathlib import Path

import numpy as np

from ecgtrust.evaluate.comparator import (
    comparator_table,
    load_12sl_decisions,
    operating_point,
    sens_at_spec,
)

FIX = Path(__file__).parent / "fixtures" / "ptbxl_plus"
ID_NML_AND_LVH = 2


def test_operating_point():
    assert operating_point(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])) == (0.5, 0.5)


def test_sens_at_spec_perfect_model():
    assert sens_at_spec(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 1.0) == 1.0


def test_sens_at_spec_respects_specificity_floor():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.85, 0.4, 0.6, 0.8, 0.9])
    assert sens_at_spec(y, p, 1.0) == 0.25 and sens_at_spec(y, p, 0.75) == 1.0


def test_norm_requires_no_other_superclass():
    d = load_12sl_decisions(FIX, FIX / "12sl_map.csv")
    assert d.loc[ID_NML_AND_LVH, "HYP"] and not d.loc[ID_NML_AND_LVH, "NORM"]
    assert d.loc[1, "NORM"] and d.loc[3, "CD"] and d.loc[4, ["MI", "STTC"]].all()


def test_comparator_table_columns():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 5)) < 0.4).astype(int)
    p = np.clip(y * 0.6 + rng.random((300, 5)) * 0.5, 0, 1)
    d = (p > 0.5).astype(int)
    t = comparator_table(y, p, d, np.arange(300), n_boot=50)
    assert {
        "class",
        "12sl_sens",
        "12sl_spec",
        "model_sens_at_12sl_spec",
        "diff",
        "lo",
        "hi",
    } <= set(t.columns)
    assert len(t) == 5
