import importlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "app" / "examples"


def test_app_module_imports():
    importlib.import_module("app.streamlit_app")


@pytest.mark.skipif(not (EX / "index.json").exists(), reason="examples built by `ecgtrust release`")
def test_examples_valid():
    idx = json.loads((EX / "index.json").read_text())
    assert idx and all(np.load(EX / e["file"]).shape == (12, 1000) for e in idx)


@pytest.mark.skipif(
    not (ROOT / "release" / "freeze.json").exists(), reason="release built by `ecgtrust release`"
)
def test_release_scores_examples():
    from ecgtrust.serve.predictor import Predictor

    p = Predictor.from_release(ROOT / "release")
    for e in json.loads((EX / "index.json").read_text()):
        r = p.predict(np.load(EX / e["file"]), 100)
        assert set(r.decisions.values()) <= {"present", "absent", "refer"}
