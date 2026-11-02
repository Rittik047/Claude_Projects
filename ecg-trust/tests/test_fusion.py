import json

import numpy as np
import pytest

from ecgtrust.evaluate.report import evaluate_split
from ecgtrust.freeze import TestSetLockedError, log_test_run
from ecgtrust.models.ensemble import ensemble_proba, load_preds
from ecgtrust.models.fusion import build_fusion, fuse, logit, select_weight

rng = np.random.default_rng(0)
Y = (rng.random((400, 5)) < 0.3).astype(np.float32)


def _write_baseline_preds(cfg, quality: float):
    """Synthetic 12SL-baseline probabilities aligned with the stored ResNet predictions."""
    r = np.random.default_rng(2)
    for s in ("val_select", "val_calib"):
        d = load_preds(cfg.artifacts / "preds" / f"resnet_{s}.npz")
        p = 1 / (1 + np.exp(-((d["y"] * 2 - 1) * quality + r.normal(0, 1, d["y"].shape))))
        np.savez(
            cfg.artifacts / "preds" / f"baseline_12sl_{s}.npz",
            p=p,
            y=d["y"],
            ecg_id=d["ecg_id"],
            patient_id=d["patient_id"],
        )


def test_fuse_endpoints_recover_each_model():
    lg = rng.normal(0, 2, (3, 400, 5))
    pb = rng.random((400, 5))
    f1, f0 = fuse(lg, pb, 1.0), fuse(lg, pb, 0.0)
    assert f1.shape == (1, 400, 5)
    assert np.allclose(ensemble_proba(f1), ensemble_proba(lg), atol=1e-5)
    assert np.allclose(ensemble_proba(f0), np.clip(pb, 1e-6, 1 - 1e-6), atol=1e-6)


def test_logit_is_finite_at_bounds():
    assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


def test_select_weight_prefers_better_model():
    lg = rng.normal(0, 1, (2, 400, 5))  # uninformative ResNet
    pb = np.clip(Y * 0.8 + rng.random(Y.shape) * 0.2, 0, 1)  # near-perfect baseline
    w, scores = select_weight(lg, pb, Y)
    assert w <= 0.2 and len(scores) == 11 and max(scores.values()) == scores[w]


def test_build_fusion_writes_preds_and_weight(tmp_cfg_val):
    _write_baseline_preds(tmp_cfg_val, quality=2.0)
    out = build_fusion(tmp_cfg_val)
    w = json.loads((tmp_cfg_val.artifacts / "models" / "fusion.json").read_text())["w"]
    assert out["w"] == w and 0.0 <= w <= 1.0
    d = load_preds(tmp_cfg_val.artifacts / "preds" / "fusion_val_calib.npz")
    assert d["logits"].shape[0] == 1
    assert (tmp_cfg_val.reports / "val" / "fusion_summary.json").exists()


def test_build_fusion_rejects_misaligned_ids(tmp_cfg_val):
    _write_baseline_preds(tmp_cfg_val, quality=2.0)
    f = tmp_cfg_val.artifacts / "preds" / "baseline_12sl_val_select.npz"
    d = dict(np.load(f))
    d["ecg_id"] = d["ecg_id"][::-1]
    np.savez(f, **d)
    with pytest.raises(ValueError, match="align"):
        build_fusion(tmp_cfg_val)


def test_fusion_evaluation_uses_its_own_calibration(tmp_cfg_val):
    _write_baseline_preds(tmp_cfg_val, quality=2.0)
    build_fusion(tmp_cfg_val)
    r = evaluate_split(tmp_cfg_val, "val_select", model="fusion")
    assert (tmp_cfg_val.reports / "val_select_fusion" / "metrics.json").exists()
    assert (tmp_cfg_val.artifacts / "calibration" / "fusion" / "T.npy").exists()
    assert r["model"] == "fusion"
    assert r["robustness"] == {"skipped": "signal-level analysis is ResNet-only"}


def test_fusion_refused_on_external(tmp_cfg_val):
    with pytest.raises(ValueError, match="12SL"):
        evaluate_split(tmp_cfg_val, "external", model="fusion")


def test_test_log_is_per_model(tmp_cfg_val):
    log_test_run(tmp_cfg_val, "test", "h")
    log_test_run(tmp_cfg_val, "test", "h", model="fusion")
    with pytest.raises(TestSetLockedError):
        log_test_run(tmp_cfg_val, "test", "h", model="fusion")
