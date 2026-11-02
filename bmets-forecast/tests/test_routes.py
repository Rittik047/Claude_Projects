import json
import re
import subprocess
from pathlib import Path

import pytest

from bmf.pipeline import STAGES

ROOT = Path(__file__).resolve().parents[1]
VALID = set(STAGES) | {"to-layout"}


def _bmf_stages(text):
    return re.findall(r"(?m)(?:^\s*|\bdo\s+|&&\s*|!)bmf\s+([a-z\-]+)", text)


def test_juno_scripts_reference_cli():
    files = sorted((ROOT / "juno").glob("*.sh")) + sorted((ROOT / "juno").glob("*.sbatch"))
    assert len(files) >= 9
    used = set()
    for f in files:
        t = f.read_text()
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f
        for s in _bmf_stages(t):
            assert s in VALID, (f.name, s)
            used.add(s)
        if f.suffix == ".sbatch":
            assert "#SBATCH" in t and "00_env.sh" in t
    assert {"preprocess", "seg-export", "embed", "train", "evaluate", "figures", "report", "lock", "external"} <= used


def test_colab_notebooks_valid():
    nbformat = pytest.importorskip("nbformat")
    nbs = sorted((ROOT / "colab").glob("*.ipynb"))
    assert len(nbs) == 5
    used = set()
    for p in nbs:
        nb = nbformat.read(str(p), as_version=4)
        nbformat.validate(nb)
        src = "\n".join(c.source for c in nb.cells)
        assert "drive.mount" in src and "BMF_DATA" in src
        for s in _bmf_stages(src):
            assert s in VALID, (p.name, s)
            used.add(s)
    assert {"audit", "preprocess", "embed", "train", "figures"} <= used
