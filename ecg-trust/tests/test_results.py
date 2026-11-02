import json

import pandas as pd
import pytest

from ecgtrust import CLASSES
from ecgtrust.results import build_results_report


def _metrics(split, model, auc):
    per = [
        {"class": c, "n_pos": 100, "auroc": auc, "lo": auc - 0.02, "hi": auc + 0.02}
        for c in CLASSES
    ]
    return {
        "split": split,
        "model": model,
        "n": 2000,
        "temperature": [1.0] * 5,
        "discrimination": {
            "macro_auroc": {"point": auc, "lo": auc - 0.01, "hi": auc + 0.01},
            "per_class": per,
            "auprc": [0.8] * 5,
        },
        "calibration": {"macro_ece_pre": 0.05, "macro_ece_post": 0.03},
        "conformal": {a: {"cov_pos": [0.9] * 5, "refer_rate": [0.1] * 5} for a in ("0.05", "0.1")},
    }


@pytest.fixture()
def reports_cfg(tmp_cfg_val):
    rep = tmp_cfg_val.reports
    for name, split, model, auc in [
        ("test", "test", "resnet", 0.93),
        ("test_fusion", "test", "fusion", 0.94),
        ("external", "external", "resnet", 0.85),
    ]:
        d = rep / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(_metrics(split, model, auc)))
        rows = [
            ("sex_label", "female", "MI", 0.90, 0.88, 0.92),
            ("sex_label", "male", "MI", 0.80, 0.78, 0.82),
            ("sex_label", "female", "CD", 0.90, 0.85, 0.95),
            ("sex_label", "male", "CD", 0.89, 0.84, 0.94),
        ]
        pd.DataFrame(rows, columns=["column", "group", "class", "auroc", "lo", "hi"]).assign(
            status="ok"
        ).to_csv(d / "subgroups.csv", index=False)
    pd.DataFrame(
        [{"noise": n, "level": lv, "macro_auroc": 0.9} for n in ("bw", "emg") for lv in ("6", "0")]
    ).to_csv(rep / "test" / "robustness.csv", index=False)
    (rep / "test" / "comparator.md").write_text("# GE 12SL comparator — test\n\n| a |\n")
    (rep / "test_runs.log").write_text("2026 split=test model=resnet freeze=abc\n")
    tmp_cfg_val.artifacts.joinpath("freeze.json").write_text(json.dumps({"git_commit": "c0ffee"}))
    return tmp_cfg_val


def test_results_report_built_from_saved_files_only(reports_cfg, monkeypatch):
    import ecgtrust.predict

    monkeypatch.setattr(
        ecgtrust.predict, "predict_logits", lambda *a, **k: pytest.fail("recomputed")
    )
    txt = build_results_report(reports_cfg).read_text()
    assert "freeze" in txt and "External" in txt and "refer rate" in txt
    assert "c0ffee" in txt and "split=test model=resnet" in txt
    assert "reference, not re-run" in txt


def test_subgroup_highlight_only_for_non_overlapping_cis(reports_cfg):
    txt = build_results_report(reports_cfg).read_text()
    section = txt.split("## Subgroups")[1].split("##")[0]
    assert "MI" in section and "CD" not in section


def test_missing_split_is_marked_pending(reports_cfg):
    import shutil

    shutil.rmtree(reports_cfg.reports / "external")
    assert "pending" in build_results_report(reports_cfg).read_text()
