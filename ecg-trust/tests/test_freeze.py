from pathlib import Path

import pytest

from ecgtrust.config import load_config
from ecgtrust.freeze import TestSetLockedError, log_test_run, require_freeze, write_freeze

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


@pytest.fixture()
def tmp_cfg(tmp_path):
    return load_config(CFG, artifacts=str(tmp_path / "art"), reports=str(tmp_path / "rep"))


@pytest.fixture()
def frozen(tmp_cfg, tmp_path):
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"weights")
    write_freeze(tmp_cfg, [ck])

    class F:
        checkpoint = ck

    return F


def test_test_split_locked_without_freeze(tmp_cfg):
    with pytest.raises(TestSetLockedError):
        require_freeze(tmp_cfg, "test")


def test_frozen_manifest_unlocks(tmp_cfg, frozen):
    m = require_freeze(tmp_cfg, "external")
    assert str(frozen.checkpoint) in m["files"] and m["alphas"] == tmp_cfg.alphas


def test_hash_mismatch_locks(tmp_cfg, frozen):
    frozen.checkpoint.write_bytes(b"tampered")
    with pytest.raises(TestSetLockedError):
        require_freeze(tmp_cfg, "external")


def test_val_split_never_locked(tmp_cfg):
    assert require_freeze(tmp_cfg, "val_select") == {}


def test_second_test_run_refused(tmp_cfg, frozen):
    log_test_run(tmp_cfg, "test", "h")
    with pytest.raises(TestSetLockedError):
        log_test_run(tmp_cfg, "test", "h")
    log_test_run(tmp_cfg, "test", "h", rerun_reason="bug fix in metric code")
    assert "bug fix in metric code" in (tmp_cfg.reports / "test_runs.log").read_text()
