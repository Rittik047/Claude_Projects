"""Pre-flight checks on real data: segmentation label scheme and foundation-model weights."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from bmf.checks import check_labels
from bmf.data.clinical import load_clinical, timepoints
from bmf.data.mri_index import build_index
from bmf.fm.registry import load_checked
from bmf.synth import make_synthetic_cohort

LAB = {"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    out = make_synthetic_cohort(tmp_path_factory.mktemp("raw"), n_patients=4, shape=(40, 40, 40), seed=5)
    tp = timepoints(load_clinical(out["clinical"]))
    return out, tp


def _index(out, tp):
    return build_index(out["root"], tp)


def _relabel(ix, mapping):
    for p in ix.query("seq == 'seg' and exists").path:
        im = nib.load(p); a = np.asarray(im.dataobj).astype(np.int16); b = a.copy()
        for src, dst in mapping.items():
            b[a == src] = dst
        nib.save(nib.Nifti1Image(b, im.affine), p)


def test_load_checked_strips_module_prefix_and_reports_fraction():
    m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    sd = {f"module.{k}": v.clone() for k, v in m.state_dict().items()}
    assert load_checked(m, {"state_dict": sd}, "toy") == pytest.approx(1.0)


def test_load_checked_raises_when_checkpoint_does_not_match():
    m = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    with pytest.raises(ValueError, match="toy.*0% of"):
        load_checked(m, {"encoder.w": torch.zeros(1)}, "toy")


def test_check_labels_passes_on_brats_mets_scheme(cohort):
    out, tp = cohort
    r = check_labels(_index(out, tp), LAB, n=6, seed=0)
    assert r["ok"], r["problems"]
    assert set(r["values"]) <= {0, 1, 2, 3}
    assert r["per_label"]["ET"]["t1c_z"] > r["per_label"]["SNFH"]["t1c_z"]


def test_check_labels_flags_swapped_et_and_oedema(cohort, tmp_path):
    out = make_synthetic_cohort(tmp_path / "raw", n_patients=3, shape=(40, 40, 40), seed=6)
    tp = timepoints(load_clinical(out["clinical"])); ix = _index(out, tp)
    _relabel(ix, {2: 3, 3: 2})
    r = check_labels(ix, LAB, n=6, seed=0)
    assert not r["ok"] and any("ET" in p for p in r["problems"])


def test_check_labels_flags_old_brats_scheme_with_et_as_4(cohort, tmp_path):
    out = make_synthetic_cohort(tmp_path / "raw", n_patients=3, shape=(40, 40, 40), seed=7)
    tp = timepoints(load_clinical(out["clinical"])); ix = _index(out, tp)
    _relabel(ix, {3: 4})
    r = check_labels(ix, LAB, n=6, seed=0)
    assert not r["ok"] and any("label 4" in p for p in r["problems"])


def test_brainsegfounder_loader_rejects_wrong_checkpoint(tmp_path):
    monai = pytest.importorskip("monai")
    from monai.networks.nets import SwinUNETR
    from bmf.fm.registry import load_encoder
    net = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_v2=False)
    good = tmp_path / "good.pt"; torch.save({"state_dict": {f"module.{k}": v for k, v in net.state_dict().items()}}, good)
    enc = load_encoder("brainsegfounder", good)
    assert enc(torch.zeros(1, 4, 64, 64, 64)).shape == (1, 768)
    bad = tmp_path / "bad.pt"; torch.save({"state_dict": {"foo.weight": torch.zeros(2)}}, bad)
    with pytest.raises(ValueError, match="brainsegfounder"):
        load_encoder("brainsegfounder", bad)


def test_cli_check_writes_report(cohort, tmp_path):
    import yaml
    from bmf.cli import main
    out, _ = cohort
    cfg = {"paths": {"data_root": str(out["root"]), "clinical": str(out["clinical"]), "mri_meta": str(out["meta"]),
                     "cache": str(tmp_path / "cache"), "reports": str(tmp_path / "reports")},
           "fm": {"weights_dir": str(tmp_path / "w"), "weights": {"brainsegfounder": "missing.pt"}}}
    cp = tmp_path / "c.yaml"; cp.write_text(yaml.safe_dump(cfg))
    for st in ["audit", "index"]:
        assert main([st, "--config", str(cp)]) == 0
    assert main(["check", "--config", str(cp)]) == 0
    r = json.loads((tmp_path / "reports" / "check.json").read_text())
    assert r["labels"]["ok"] is True
    assert r["weights"]["brainsegfounder"]["status"] == "missing"
