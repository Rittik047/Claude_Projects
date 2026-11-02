import json

import pandas as pd
import pytest
import yaml

from bmf.cli import main
from bmf.synth import make_synthetic_cohort

STAGES = ["audit", "index", "splits", "preprocess", "track", "labels", "features"]


def _cfg(tmp, out, name):
    c = {"paths": {"data_root": str(out["root"]), "clinical": str(out["clinical"]), "mri_meta": str(out["meta"]),
                   "cache": str(tmp / name / "cache"), "reports": str(tmp / name / "reports")},
         "n_folds": 3, "preprocess": {"n4": False, "affine_longitudinal": False}, "bootstrap": {"n": 30},
         "radiomics_backend": "simple", "lock_dir": str(tmp / "locked")}
    p = tmp / f"{name}.yaml"; p.write_text(yaml.safe_dump(c)); return str(p)


def test_lock_then_external(tmp_path):
    dev = _cfg(tmp_path, make_synthetic_cohort(tmp_path / "dev_raw", n_patients=12, shape=(32, 32, 32), seed=4), "dev")
    ext = _cfg(tmp_path, make_synthetic_cohort(tmp_path / "ext_raw", n_patients=8, shape=(32, 32, 32), seed=9), "ext")
    for s in STAGES:
        assert main([s, "--config", dev]) == 0 and main([s, "--config", ext]) == 0
    assert main(["lock", "--config", dev, "--task", "T2a"]) == 0
    lock = json.loads((tmp_path / "locked" / "lock.json").read_text())
    assert {"P0", "P1"} <= set(lock["models"]) and len(lock["sha256"]) == 64
    assert main(["external", "--config", ext, "--task", "T2a"]) == 0
    r = json.loads((tmp_path / "ext" / "reports" / "external_T2a.json").read_text())
    assert r["lock_sha256"] == lock["sha256"] and {"P0", "P1"} <= set(r["models"])
    with pytest.raises(SystemExit):
        main(["external", "--config", ext, "--task", "T2a"])        # external set may be evaluated only once


def test_to_layout(tmp_path):
    src = make_synthetic_cohort(tmp_path / "s", n_patients=2, shape=(16, 16, 16), seed=0)["root"]
    rows = [(p.parent.name.replace("BMETS", "EXT"), p.name.split("-")[-1].replace(".nii.gz", ""), str(p)) for p in sorted(src.rglob("*.nii.gz"))]
    m = tmp_path / "map.csv"; pd.DataFrame(rows, columns=["study_id", "seq", "path"]).to_csv(m, index=False)
    assert main(["to-layout", "--map", str(m), "--out", str(tmp_path / "lay")]) == 0
    sid = rows[0][0]
    assert (tmp_path / "lay" / sid / f"{sid}-t1c.nii.gz").exists()
