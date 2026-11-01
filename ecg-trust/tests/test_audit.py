import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ecgtrust.audit import run_audit
from ecgtrust.config import load_config

FIX = Path(__file__).parent / "fixtures"
CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture()
def fixture_cfg(tmp_path):
    cfg = load_config(
        CFG,
        data_root=str(tmp_path / "data"),
        artifacts=str(tmp_path / "artifacts"),
        reports=str(tmp_path / "reports"),
    )
    shutil.copytree(FIX / "ptbxl", cfg.data_root / "ptbxl")
    for name in ("ptbxl", "ptbxl_plus", "external"):
        d = cfg.data_root / name
        d.mkdir(parents=True, exist_ok=True)
        log = {"dataset": name, "n_selected": 2, "n_ok": 2, "n_failed": 0, "n_bytes": 10}
        (d / "download_log.json").write_text(json.dumps(log))
    lab = cfg.data_root / "ptbxl_plus" / "labels"
    lab.mkdir(parents=True)
    pd.DataFrame(
        {"ecg_id": [1, 2], "statements": ["['NSR', 'NML']", "['SR', 'IMI', 'LAFB']"]}
    ).to_csv(lab / "12sl_statements.csv", index=False)
    cfg.artifacts.mkdir(parents=True)
    pd.DataFrame(
        {
            "record": ["JS1", "JS2", "JS3"],
            "age": [50.0, None, 70.0],
            "sex": ["male", "unknown", "female"],
            "age_band": ["40-59", "unknown", "60-79"],
            "excluded_reason": ["", "flat_lead:V1", ""],
            "NORM": [True, False, False],
            "MI": [False, False, False],
            "STTC": [False, True, True],
            "CD": [False, False, True],
            "HYP": [False, False, False],
            "has_label": [True, True, False],
            "STTC_liberal": [False, True, True],
            "has_label_liberal": [True, True, True],
            **{f"{c}_liberal": [False] * 3 for c in ("NORM", "MI", "CD", "HYP")},
        }
    ).to_parquet(cfg.artifacts / "external_meta.parquet")
    return cfg


def test_audit_contains_required_sections(fixture_cfg):
    txt = run_audit(fixture_cfg).read_text()
    for h in [
        "## Downloads",
        "## PTB-XL",
        "## External",
        "## 12SL statement vocabulary",
        "## Baseline (val_select)",
    ]:
        assert h in txt
    assert "age coded 300" in txt and "insufficient" in txt


def test_audit_counts(fixture_cfg):
    txt = run_audit(fixture_cfg).read_text()
    assert "| records | 6 |" in txt and "| patients | 5 |" in txt
    assert "| no superclass label (excluded) | 1 |" in txt
    assert "flat_lead" in txt and "IMI" in txt


def test_likelihood_zero_counts_diagnostic_codes_only(fixture_cfg):
    # fixture: NDT:0 is the only diagnostic code with likelihood 0 (SR:0, LVOLT:0 are not)
    txt = run_audit(fixture_cfg).read_text()
    assert (
        "| diagnostic SCP codes with likelihood 0 (still counted, benchmark convention) | 1 |"
        in txt
    )


def test_audit_reports_external_label_availability(fixture_cfg):
    txt = run_audit(fixture_cfg).read_text()
    assert "| signal OK but no superclass (excluded, as in PTB-XL) | 1 |" in txt
    assert "| evaluated (signal OK and labelled) | 1 |" in txt
    assert "### Sensitivity: liberal label map" in txt


def test_audit_writes_machine_readable_summary(fixture_cfg):
    run_audit(fixture_cfg)
    s = json.loads((fixture_cfg.reports / "data_summary.json").read_text())
    assert s["ptbxl"]["records"] == 6 and s["ptbxl"]["patients"] == 5
    assert set(s["ptbxl"]["prevalence"]) == {"train", "val_select", "val_calib", "test"}
    assert s["external"]["evaluated"] == 1 and "NORM" in s["external"]["prevalence_strict"]
