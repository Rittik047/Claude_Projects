import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from ecgtrust import CLASSES
from ecgtrust.models.train import build_model
from ecgtrust.serve.api import create_app
from ecgtrust.serve.predictor import Predictor

T = np.arange(1000) / 100
ECG_MV = (np.tile(np.exp(-(((T % 0.8) - 0.4) ** 2) / 0.0008), (12, 1)) * 1.2).astype(np.float32)


def ecg_at(fs, seconds=10):
    t = np.arange(int(fs * seconds)) / fs
    return np.tile(np.exp(-(((t % 0.8) - 0.4) ** 2) / 0.0008), (12, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    d = tmp_path_factory.mktemp("release")
    cfg = {"width": 4, "blocks": [1]}
    torch.manual_seed(0)
    for s in range(2):
        (d / "models").mkdir(exist_ok=True)
        torch.save(
            {
                "state_dict": build_model(cfg).state_dict(),
                "config": cfg,
                "lead_mean": np.zeros(12, np.float32),
                "lead_std": np.ones(12, np.float32),
                "seed": s,
            },
            d / "models" / f"resnet_seed{s}.pt",
        )
    np.save(d / "T.npy", np.ones(5))
    for a in ("0.05", "0.1"):
        (d / f"conformal_{a}.json").write_text(
            json.dumps({"q_present": [0.6] * 5, "q_absent": [0.6] * 5, "alpha": float(a)})
        )
    (d / "freeze.json").write_text(json.dumps({"git_commit": "abc"}))
    return d


@pytest.fixture(scope="module")
def predictor(release):
    return Predictor.from_release(release)


@pytest.fixture(scope="module")
def client(predictor):
    return TestClient(create_app(predictor))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_version"]


def test_predict_returns_valid_decisions(client):
    r = client.post("/predict", json={"signal": ECG_MV.tolist(), "fs": 100})
    assert r.status_code == 200
    body = r.json()
    assert set(body["decisions"].values()) <= {"present", "absent", "refer"}
    assert set(body["probabilities"]) == set(CLASSES)


def test_transposed_input_rejected(client):  # Review Focus 3
    r = client.post("/predict", json={"signal": ECG_MV.T.tolist(), "fs": 100})
    assert r.status_code == 422 and "transpose" in r.json()["detail"].lower()


def test_microvolt_input_flagged(client):  # Review Focus 3
    r = client.post("/predict", json={"signal": (ECG_MV * 1000).tolist(), "fs": 100})
    assert "possible_unit_error" in r.json()["quality"]


@pytest.mark.parametrize("fs", [250, 360, 500])  # Review Focus 4
def test_other_sampling_rates_resampled(predictor, fs):
    assert set(predictor.predict(ecg_at(fs), fs).probabilities) == set(CLASSES)


def test_short_record_padded(predictor):
    assert set(predictor.predict(ecg_at(100, seconds=5), 100).probabilities) == set(CLASSES)


def test_low_fs_nan_and_too_short_rejected(client):
    assert client.post("/predict", json={"signal": ECG_MV.tolist(), "fs": 50}).status_code == 422
    bad = ECG_MV.copy()
    bad[0, 5] = np.nan
    r = client.post(
        "/predict",
        json={
            "signal": [[None if np.isnan(v) else float(v) for v in row] for row in bad],
            "fs": 100,
        },
    )
    assert r.status_code == 422
    assert (
        client.post("/predict", json={"signal": ECG_MV[:, :200].tolist(), "fs": 100}).status_code
        == 422
    )


def test_flat_lead_flag(predictor):
    x = ECG_MV.copy()
    x[6] = 0
    assert "flat_lead:V1" in predictor.predict(x, 100).quality


def test_npy_upload(client):
    import io

    buf = io.BytesIO()
    np.save(buf, ECG_MV)
    r = client.post("/predict/npy", files={"file": ("x.npy", buf.getvalue())}, data={"fs": "100"})
    assert r.status_code == 200 and set(r.json()["decisions"]) == set(CLASSES)
