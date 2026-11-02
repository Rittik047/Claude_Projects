import json

import pytest

from ecgtrust.readme_tables import render_headline_table


def _metrics(point, ece_pre, ece_post, refer):
    return {
        "n": 100,
        "discrimination": {"macro_auroc": {"point": point, "lo": point - 0.01, "hi": point + 0.01}},
        "calibration": {"macro_ece_pre": ece_pre, "macro_ece_post": ece_post},
        "conformal": {
            "0.1": {"refer_rate": [refer] * 5, "cov_pos": [0.9] * 5, "cov_neg": [0.91] * 5}
        },
    }


@pytest.fixture()
def fixture_reports(tmp_path):
    for split, m in {
        "test": _metrics(0.9312, 0.05, 0.02, 0.2),
        "external": _metrics(0.8123, 0.12, 0.09, 0.35),
    }.items():
        (tmp_path / split).mkdir()
        (tmp_path / split / "metrics.json").write_text(json.dumps(m))
    return tmp_path


def test_table_matches_json(fixture_reports):
    t = render_headline_table(fixture_reports)
    point = json.loads((fixture_reports / "test/metrics.json").read_text())
    assert f"{point['discrimination']['macro_auroc']['point']:.3f}" in t
    assert "0.812" in t and "| External" in t and "0.050 → 0.020" in t


def test_missing_split_is_marked(tmp_path):
    assert "not yet evaluated" in render_headline_table(tmp_path)
