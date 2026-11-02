"""Generate ecg_trust_colab.ipynb (run: python colab/make_notebook.py). Edit cells here, not
in the .ipynb, so the notebook stays reviewable."""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# ecg-trust on Google Colab (GPU)

Runs the whole pipeline end to end on a Colab GPU:

1. data from Google Drive;
2. caches;
3. feature baseline;
4. 3-seed ResNet ensemble;
5. fusion;
6. calibration and conformal abstention;
7. freeze;
8. single-use internal and external tests;
9. figures, model release and the illustrated PDF guide.

Every report, figure and model file is copied to your Drive as it is produced.

**Before you start**
- Put the data on Drive as described in [`colab/README.md`](https://github.com/Rittik047/Claude_Projects/blob/claude/beautiful-carson-3col06/ecg-trust/colab/README.md). Alternatively, leave `DOWNLOAD_IF_MISSING = True` and the notebook fetches the data once and saves it to Drive.
- Choose **Runtime → Change runtime type → G4** (any GPU works).
- Then choose **Runtime → Run all**.

Typical time on a G4 with the data already on Drive is about 25–40 min. Training takes about 5–10 min; the bootstrap CIs and robustness analyses take most of the rest. A first session that also downloads the data adds about 20–40 min.

> Research software, not a medical device.""",
    ),
    (
        "code",
        """# ===================== Settings =====================
DRIVE_ROOT = "/content/drive/MyDrive/ecg-trust"   # data/, cache/ and outputs/ live here
REPO = "https://github.com/Rittik047/Claude_Projects.git"
BRANCH = "claude/beautiful-carson-3col06"         # use "main" once the branch is merged
DOWNLOAD_IF_MISSING = True   # fetch from the PhysioNet AWS mirror if Drive has no data
SAVE_CACHE_TO_DRIVE = True   # keep the 100 Hz caches on Drive: later runs skip unzip + prepare
RUN_EXTERNAL = True          # external validation on 45k Chinese ECGs (needs ~5.4 GB of data)
BUILD_GUIDE = True           # render the illustrated PDF guide (installs headless Chromium)
N_BOOT = 2000                # bootstrap resamples for CIs (2000 = published setting)""",
    ),
    ("md", "## 1 · Drive and GPU"),
    (
        "code",
        """from google.colab import drive
drive.mount("/content/drive")

import torch
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
assert torch.cuda.is_available(), "No GPU: Runtime → Change runtime type → G4 (or any GPU)"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0))""",
    ),
    (
        "md",
        "## 2 · Code\nA private repository needs a GitHub token stored as the Colab secret `GITHUB_TOKEN` (🔑 icon in the left bar).",
    ),
    (
        "code",
        """import os
from pathlib import Path

token = None
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    pass
url = REPO.replace("https://", f"https://{token}@") if token else REPO
if not Path("/content/Claude_Projects").exists():
    !git clone -q --branch {BRANCH} {url} /content/Claude_Projects
%cd /content/Claude_Projects/ecg-trust
!git log --oneline -1
!pip install -q -e ".[dev]"   # keeps Colab's CUDA build of PyTorch""",
    ),
    ("md", "## 3 · Configuration and helpers"),
    (
        "code",
        """import datetime as dt
import shutil
import yaml

DATA, ART, REP = Path("/content/data"), Path("/content/artifacts"), Path("/content/reports")
DDATA, DCACHE = Path(DRIVE_ROOT) / "data", Path(DRIVE_ROOT) / "cache"
RUN = dt.datetime.now().strftime("%Y%m%d-%H%M")
OUT = Path(DRIVE_ROOT) / "outputs" / RUN

cfg = yaml.safe_load(open("configs/default.yaml"))
cfg.update(data_root=str(DATA), artifacts=str(ART), reports=str(REP), n_boot=N_BOOT)
CFG = "/content/ecgtrust_colab.yaml"
yaml.safe_dump(cfg, open(CFG, "w"))


def ecg(args: str) -> None:
    \"\"\"Run one ecgtrust command with live output; stop the notebook if it fails.\"\"\"
    ip = get_ipython()
    ip.system(f"ecgtrust --config {CFG} {args}")
    if ip.user_ns.get("_exit_code", 0):
        raise RuntimeError(f"ecgtrust {args} failed")


def sync(note: str = "") -> None:
    \"\"\"Copy reports and the model artefacts to Drive (safe to call repeatedly).\"\"\"
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    if REP.exists():
        shutil.copytree(REP, OUT / "reports", dirs_exist_ok=True)
    for sub in ("models", "preds", "calibration"):
        if (ART / sub).exists():
            shutil.copytree(ART / sub, OUT / "artifacts" / sub, dirs_exist_ok=True)
    for f in ("freeze.json", "splits.parquet"):
        if (ART / f).exists():
            shutil.copy2(ART / f, OUT / "artifacts" / f)
    print(f"synced to {OUT} {note}")


print("outputs will be written to", OUT)""",
    ),
    (
        "md",
        """## 4 · Data from Drive
For each dataset the notebook uses, in order:
1. the 100 Hz cache in `ecg-trust/cache/`;
2. a zip in `ecg-trust/data/`;
3. an unzipped folder in `ecg-trust/data/`;
4. a fresh download from the mirror, which is then zipped back to Drive.""",
    ),
    (
        "code",
        """MARKERS = {
    "ptbxl": "ptbxl_database.csv",
    "ptbxl_plus": "features/12sl_features.csv",
    "external": "WFDBRecords",
}
CACHE_FILES = {
    "ptbxl": ["ptbxl_x100.npy", "ptbxl_index.parquet"],
    "external": ["external_x100.npy", "external_meta.parquet"],
}
ART.mkdir(parents=True, exist_ok=True)


def find_root(base: Path, marker: str) -> Path | None:
    \"\"\"The folder (up to two levels deep) that contains `marker`; PhysioNet zips nest it.\"\"\"
    levels = [base, *base.glob("*"), *base.glob("*/*")]
    return next((d for d in levels if d.is_dir() and (d / marker).exists()), None)


def restore_cache(name: str) -> bool:
    files = CACHE_FILES.get(name, [])
    if files and all((DCACHE / f).exists() for f in files):
        for f in files:
            if not (ART / f).exists():
                shutil.copy2(DCACHE / f, ART / f)
        return True
    return False


def get_dataset(name: str) -> None:
    dest, marker = DATA / name, MARKERS[name]
    if (dest / marker).exists():
        return print(f"{name}: already on local disk")
    zf = DDATA / f"{name}.zip"
    if zf.exists():
        print(f"{name}: unzipping {zf} ...")
        tmp = Path(f"/content/unzip_{name}")
        shutil.copy2(zf, "/content/tmp.zip")
        get_ipython().system(f"unzip -q -o /content/tmp.zip -d {tmp}")
        os.remove("/content/tmp.zip")
        root = find_root(tmp, marker)
        assert root, f"{zf} does not contain {marker}; see colab/README.md"
        shutil.move(str(root), dest)
        return print(f"{name}: ready")
    root = find_root(DDATA / name, marker) if (DDATA / name).exists() else None
    if root:
        print(f"{name}: copying unzipped folder from Drive (slow for many small files) ...")
        shutil.copytree(root, dest)
        return
    if not DOWNLOAD_IF_MISSING:
        raise FileNotFoundError(f"{name}: put {name}.zip in {DDATA} (see colab/README.md)")
    print(f"{name}: downloading from the PhysioNet AWS mirror (SHA-256 verified) ...")
    ecg(f"download --dataset {name} --workers 32")
    DDATA.mkdir(parents=True, exist_ok=True)
    get_ipython().system(f"cd {DATA} && zip -q -r -1 /content/{name}.zip {name}")
    shutil.copy2(f"/content/{name}.zip", zf)
    print(f"{name}: saved {zf} for next time")


have_ptbxl_cache = restore_cache("ptbxl")
have_ext_cache = RUN_EXTERNAL and restore_cache("external")
get_dataset("ptbxl")        # metadata is always needed (labels, splits, subgroups)
get_dataset("ptbxl_plus")   # GE 12SL measurements and statements
if RUN_EXTERNAL and not have_ext_cache:
    get_dataset("external")
!du -sh /content/data/* 2>/dev/null""",
    ),
    ("md", "## 5 · Splits and 100 Hz caches"),
    (
        "code",
        """ecg("prepare --workers 16" + ("" if RUN_EXTERNAL else " --skip-external"))
if SAVE_CACHE_TO_DRIVE:
    DCACHE.mkdir(parents=True, exist_ok=True)
    for f in sum(CACHE_FILES.values(), []):
        if (ART / f).exists() and not (DCACHE / f).exists():
            shutil.copy2(ART / f, DCACHE / f)
    print("caches saved to", DCACHE)
ecg("audit")
sync("(audit)")""",
    ),
    (
        "md",
        "## 6 · Models\nGradient-boosted baselines on PTB-XL+ measurements, then the 3-seed ResNet ensemble on the GPU, then the logit fusion (weight chosen on `val_select`).",
    ),
    (
        "code",
        """ecg("train --model baseline --features 12sl")
ecg("train --model baseline --features ecgdeli")""",
    ),
    (
        "code",
        """%%time
ecg("train --model resnet --seeds 0 1 2")
sync("(models)")""",
    ),
    (
        "code",
        """ecg("train --model fusion")
!cat /content/reports/val/resnet_summary.json /content/reports/val/fusion_summary.json""",
    ),
    (
        "md",
        "## 7 · Validation evaluation, then freeze\nEverything up to here uses validation data only. `freeze` hashes the checkpoints, configuration and label maps; the test sets unlock only after it.",
    ),
    (
        "code",
        """ecg("evaluate --split val_select")
ecg("evaluate --split val_select --model fusion")
ecg("freeze")
sync("(validation + freeze)")""",
    ),
    (
        "md",
        "## 8 · Single-use test runs\nEach test set is evaluated once per model; a repeat is refused unless `--rerun-reason` is given and logged.",
    ),
    (
        "code",
        """ecg("evaluate --split test")
ecg("evaluate --split test --model fusion")
if RUN_EXTERNAL:
    ecg("evaluate --split external")
ecg("results")
sync("(test results)")
from IPython.display import Markdown, display
display(Markdown(open(REP / "results.md").read()))""",
    ),
    ("md", "## 9 · Figures, model release and guide"),
    (
        "code",
        """ecg("figures")
ecg("release")
!du -sh release
shutil.copytree("release", OUT / "release", dirs_exist_ok=True)
sync("(figures + release)")""",
    ),
    (
        "code",
        """if BUILD_GUIDE:
    !npm install --silent --prefix /content/node playwright@1.48.2 > /dev/null
    !cd /content/node && npx playwright install --with-deps chromium > /dev/null
    os.environ["NODE_PATH"] = "/content/node/node_modules"
    ecg("guide")
    sync("(guide)")
    print("guide:", OUT / "reports" / "guide" / "ecg_trust_guide.pdf")""",
    ),
    ("md", "## 10 · Look at the results"),
    (
        "code",
        """from IPython.display import Image, display

for rel in [
    "03_performance/auroc_forest_all_runs.png",
    "04_calibration/reliability_test.png",
    "05_abstention/coverage_vs_target.png",
    "06_robustness/noise_robustness_panels.png",
    "07_explainability/lead_importance_integrated_gradients.png",
    "08_comparator/vs_12sl_test_fusion.png",
]:
    f = REP / "figures" / rel
    if f.exists():
        print(rel)
        display(Image(filename=str(f), width=900))
print("All figures:", OUT / "reports" / "figures")""",
    ),
    (
        "md",
        """## Notes
- **Reproducibility.** The splits and caches are bit-identical to the original CPU run.
  - GPU training is seeded, but some cuDNN kernels are non-deterministic, so checkpoints and metrics can differ slightly from the CPU results in the repository (typically by about 0.002 AUROC). Report the run you actually performed.
  - Set `ECGTRUST_DEVICE=cpu` to force CPU.
- **New freeze.** This notebook makes its own freeze and test log in `outputs/<run>/`. The repository's `freeze-v1` results stay the reference.
- **Disconnects.** If Colab disconnects, re-run from the top. Cached data and caches on Drive make the restart quick, and finished stages are already in `outputs/<run>/`.""",
    ),
]


def build() -> dict:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        lines = src.splitlines(keepends=True)
        cell = {
            "id": f"cell-{i:02d}",
            "cell_type": "markdown" if kind == "md" else "code",
            "metadata": {},
            "source": lines,
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "G4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("ecg_trust_colab.ipynb")
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(out)
