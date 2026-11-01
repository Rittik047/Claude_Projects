import json
import re
import shutil
from pathlib import Path

import pytest

from ecgtrust.config import load_config

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node + playwright")


def _metrics(point):
    per = [
        {"class": c, "n_pos": 50, "auroc": point, "lo": point - 0.02, "hi": point + 0.02}
        for c in ("NORM", "MI", "STTC", "CD", "HYP")
    ]
    return {
        "split": "x",
        "n": 500,
        "temperature": [1.2] * 5,
        "discrimination": {
            "macro_auroc": {"point": point, "lo": point - 0.01, "hi": point + 0.01},
            "per_class": per,
            "auprc": [0.8] * 5,
        },
        "calibration": {
            "macro_ece_pre": 0.05,
            "macro_ece_post": 0.02,
            "ece_pre": [0.05] * 5,
            "ece_post": [0.02] * 5,
        },
        "conformal": {
            "0.1": {
                "refer_rate": [0.2] * 5,
                "cov_pos": [0.9] * 5,
                "cov_neg": [0.91] * 5,
                "acc_non_referred": [0.97] * 5,
            },
            "0.05": {
                "refer_rate": [0.3] * 5,
                "cov_pos": [0.95] * 5,
                "cov_neg": [0.95] * 5,
                "acc_non_referred": [0.98] * 5,
            },
        },
    }


@pytest.fixture()
def cfg(tmp_path):
    c = load_config(CFG, reports=str(tmp_path / "rep"), artifacts=str(tmp_path / "art"))
    for split, pt in (("test", 0.9312), ("external", 0.8123)):
        (c.reports / split).mkdir(parents=True)
        (c.reports / split / "metrics.json").write_text(json.dumps(_metrics(pt)))
    prev = {c_: 0.2 for c_ in ("NORM", "MI", "STTC", "CD", "HYP")}
    (c.reports / "data_summary.json").write_text(
        json.dumps(
            {
                "ptbxl": {
                    "records": 21799,
                    "patients": 18869,
                    "unlabelled": 411,
                    "split_records": {
                        "train": 17084,
                        "val_select": 1072,
                        "val_calib": 1074,
                        "test": 2158,
                    },
                    "prevalence": {s: prev for s in ("train", "val_select", "val_calib", "test")},
                    "validated_by_human": {str(f): 0.67 if f < 9 else 1.0 for f in range(1, 11)},
                },
                "external": {
                    "records": 45152,
                    "signal_ok": 43559,
                    "evaluated": 28646,
                    "prevalence_strict": prev,
                    "evaluated_liberal": 37331,
                    "prevalence_liberal": prev,
                },
            }
        )
    )
    return c


def test_guide_builds_from_saved_reports(cfg):
    from ecgtrust.guide.build import build_guide

    pdf = build_guide(cfg)
    assert pdf.exists() and pdf.suffix == ".pdf"
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))
    assert pages >= 8
    html = (pdf.parent / "guide.html").read_text()
    assert "0.931" in html and "0.812" in html and "28,646" in html
    for src in re.findall(r'<img src="([^"]+)"', html):
        assert (pdf.parent / src).exists(), src
