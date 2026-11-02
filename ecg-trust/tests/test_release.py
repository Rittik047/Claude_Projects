import json

import numpy as np
import pytest

from ecgtrust.release import MAX_BYTES, build_release


@pytest.fixture()
def frozen_cfg(tmp_cfg_val):
    art = tmp_cfg_val.artifacts
    for s in tmp_cfg_val.seeds:
        d = art / "models" / f"seed{s}"
        d.mkdir(parents=True)
        (d / f"resnet_seed{s}.pt").write_bytes(b"w" * 10)
    cal = art / "calibration"
    cal.mkdir()
    np.save(cal / "T.npy", np.ones(5))
    for a in tmp_cfg_val.alphas:
        (cal / f"conformal_{a}.json").write_text("{}")
    (art / "freeze.json").write_text(json.dumps({"git_commit": "abc", "files": {}}))
    return tmp_cfg_val


def test_release_copies_bundle(frozen_cfg, tmp_path):
    out = build_release(frozen_cfg, tmp_path / "release")
    names = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert names == sorted(
        ["T.npy", "conformal_0.05.json", "conformal_0.1.json", "freeze.json"]
        + [f"resnet_seed{s}.pt" for s in frozen_cfg.seeds]
    )


def test_release_requires_freeze(frozen_cfg, tmp_path):
    (frozen_cfg.artifacts / "freeze.json").unlink()
    with pytest.raises(FileNotFoundError, match="freeze"):
        build_release(frozen_cfg, tmp_path / "release")


def test_release_size_cap(frozen_cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("ecgtrust.release.MAX_BYTES", 5)
    with pytest.raises(ValueError, match="MB"):
        build_release(frozen_cfg, tmp_path / "release")
    assert MAX_BYTES == 20 * 1024**2
