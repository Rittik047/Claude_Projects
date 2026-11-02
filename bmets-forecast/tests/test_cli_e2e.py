import json

import yaml

from bmf.cli import main
from bmf.synth import make_synthetic_cohort


def test_e2e_synthetic(tmp_path):
    pytest = __import__("pytest"); pytest.importorskip("torch")
    out = make_synthetic_cohort(tmp_path / "raw", n_patients=12, shape=(40, 40, 40), seed=3)
    cfg = {"paths": {"data_root": str(out["root"]), "clinical": str(out["clinical"]), "mri_meta": str(out["meta"]),
                     "cache": str(tmp_path / "cache"), "reports": str(tmp_path / "reports"), "figures": str(tmp_path / "reports/figures")},
           "n_folds": 3, "preprocess": {"n4": False, "affine_longitudinal": False},
           "train": {"epochs": 5, "encoder": "random3d"}, "bootstrap": {"n": 50}, "radiomics_backend": "simple"}
    cp = tmp_path / "cfg.yaml"; cp.write_text(yaml.safe_dump(cfg))
    for stage in ["audit", "index", "splits", "preprocess", "track", "labels", "features", "embed"]:
        assert main([stage, "--config", str(cp)]) == 0, stage
    for m in ["P0", "P1", "P2", "P3", "P5"]:
        assert main(["train", "--config", str(cp), "--model", m, "--task", "T2a"]) == 0, m
    assert main(["train", "--config", str(cp), "--model", "O1", "--task", "T4"]) == 0
    assert main(["train", "--config", str(cp), "--model", "P1", "--task", "T3"]) == 0
    for stage in ["evaluate", "explain", "figures", "report"]:
        assert main([stage, "--config", str(cp)]) == 0, stage
    res = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert {"P0", "P1", "P3"} <= set(res["T2a"]["models"])
    assert "auroc" in res["T2a"]["models"]["P1"]
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) >= 15
    log = (tmp_path / "reports" / "figures_log.txt").read_text().splitlines()
    assert len(log) == 1, log            # no figure failed
    seg_cmds = main(["seg-export", "--config", str(cp)])
    assert seg_cmds == 0 and (tmp_path / "cache" / "nnunet_raw").exists()
