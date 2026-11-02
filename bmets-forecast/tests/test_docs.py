import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_guide_builds(tmp_path):
    out = tmp_path / "guide.pdf"
    r = subprocess.run([sys.executable, str(ROOT / "docs/guide/build_guide.py"), "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    from pypdf import PdfReader
    assert len(PdfReader(str(out)).pages) >= 15


def test_figures_referenced_exist():
    src = (ROOT / "docs/guide/build_guide.py").read_text() + (ROOT / "docs/manuscript/manuscript.md").read_text()
    refs = set(re.findall(r"figures/(?:data|examples)/[a-zA-Z0-9_]+\.png", src))
    assert len(refs) >= 25
    missing = [r for r in refs if not (ROOT / "docs" / r).exists()]
    assert not missing, missing


def test_manuscript_fill(tmp_path):
    sys.path.insert(0, str(ROOT / "docs/manuscript"))
    from fill_manuscript import fill
    res = {"cohort": {"n_patients": 300}, "T2a": {"models": {"P3": {"auroc": 0.7312, "auroc_ci": [0.68, 0.78]}}}}
    txt = fill("n={{cohort.n_patients}}, AUROC {{T2a.models.P3.auroc|.2f}} [{{T2a.models.P3.auroc_ci.0|.2f}}–{{T2a.models.P3.auroc_ci.1|.2f}}] {{T4.O1cox.c_index|.2f}}", res)
    assert txt == "n=300, AUROC 0.73 [0.68–0.78] [pending: T4.O1cox.c_index]"


def test_analysis_plan_frozen():
    plan = (ROOT / "docs/analysis_plan.md").read_bytes()
    assert hashlib.sha256(plan).hexdigest() == (ROOT / "docs/analysis_plan.sha256").read_text().split()[0]


def test_guide_contains_preflight_instructions(tmp_path):
    out = tmp_path / "guide.pdf"
    subprocess.run([sys.executable, str(ROOT / "docs/guide/build_guide.py"), "--out", str(out)], check=True, cwd=ROOT)
    from pypdf import PdfReader
    text = " ".join(p.extract_text() for p in PdfReader(str(out)).pages).replace("\n", " ")
    for needle in ["Pre-flight checklist", "get_weights.sh", "sam_med3d_turbo.pth", "SAMMED3D_REPO",
                   "brainsegfounder_swinunetr_brats.pt", "bmf check", "check.json", "old BraTS",
                   "remove_flair_t1ce.py", "FLAIR, T1ce, T1, T2", "00b_smoke.sbatch", "00_smoke_test", "SMOKE OK",
                   "drive.google.com"]:
        assert needle in text, needle
