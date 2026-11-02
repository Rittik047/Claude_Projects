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


def test_juno_shared_files_written_once():
    # review finding: 32 array tasks rewrote index.csv/splits.json concurrently
    arr = (ROOT / "juno" / "02_preprocess.sbatch").read_text()
    assert "bmf index" not in arr and "bmf splits" not in arr and "bmf audit" not in arr
    once = (ROOT / "juno" / "01b_index.sbatch").read_text()
    assert all(f"bmf {s}" in once for s in ("audit", "index", "splits"))
    runall = (ROOT / "juno" / "run_all.sh").read_text()
    assert "01b_index.sbatch" in runall and "afterok:$J0" in runall


def test_no_source_file_is_gitignored():
    """CI imports from a fresh clone: every module under src/ must be trackable."""
    import subprocess
    root = Path(__file__).parents[1]
    files = [str(p.relative_to(root)) for p in (root / "src").rglob("*.py")]
    r = subprocess.run(["git", "check-ignore", "--no-index", *files], cwd=root,
                       capture_output=True, text=True)
    assert r.stdout.strip() == "", f"gitignored source files:\n{r.stdout}"
