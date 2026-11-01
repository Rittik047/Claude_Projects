"""Freeze manifest and single-use test lock (spec §3, §7).

`test` and `external` predictions are allowed only when `artifacts/freeze.json` exists and
every frozen file (checkpoints, config, label maps) still matches its recorded SHA-256.
Each test run is appended to `reports/test_runs.log`; a second run of the same split is
refused unless a reason is given, and that reason is logged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from ecgtrust.config import Config
from ecgtrust.data.download import sha256_file

LOCKED_SPLITS = ("test", "external")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
CONFIG_FILES = ("default.yaml", "external_label_map.csv", "12sl_label_map.csv")


class TestSetLockedError(Exception):
    """Test data requested without a valid freeze, or a repeated test run."""

    __test__ = False  # not a pytest test class


def _manifest_path(cfg: Config) -> Path:
    return cfg.artifacts / "freeze.json"


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=CONFIG_DIR, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_freeze(cfg: Config, paths: list[Path]) -> Path:
    files = [Path(p) for p in paths] + [
        CONFIG_DIR / f for f in CONFIG_FILES if (CONFIG_DIR / f).exists()
    ]
    manifest = {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "alphas": list(cfg.alphas),
        "files": {str(p): sha256_file(p) for p in files},
    }
    out = _manifest_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return out


def freeze_hash(cfg: Config) -> str:
    return hashlib.sha256(_manifest_path(cfg).read_bytes()).hexdigest()


def require_freeze(cfg: Config, split: str) -> dict:
    if split not in LOCKED_SPLITS:
        return {}
    path = _manifest_path(cfg)
    if not path.exists():
        raise TestSetLockedError(f"'{split}' is locked: run `ecgtrust freeze` first")
    manifest = json.loads(path.read_text())
    for f, h in manifest["files"].items():
        if not Path(f).exists() or sha256_file(Path(f)) != h:
            raise TestSetLockedError(f"'{split}' is locked: frozen file changed: {f}")
    return manifest


def log_test_run(cfg: Config, split: str, freeze_hash: str, rerun_reason: str | None = None):
    log = cfg.reports / "test_runs.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    previous = (
        [ln for ln in log.read_text().splitlines() if f"split={split} " in ln]
        if log.exists()
        else []
    )
    if previous and not rerun_reason:
        raise TestSetLockedError(f"'{split}' was already evaluated; pass a rerun reason to repeat")
    stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    line = f"{stamp} split={split} freeze={freeze_hash[:12]}"
    if rerun_reason:
        line += f" RERUN reason={rerun_reason!r}"
    with log.open("a") as f:
        f.write(line + "\n")


def register(sub) -> None:
    p = sub.add_parser("freeze", help="hash checkpoints, config and label maps into freeze.json")

    def run(args):
        from ecgtrust.config import load_config
        from ecgtrust.models.ensemble import checkpoint_paths

        cfg = load_config(args.config)
        print(write_freeze(cfg, checkpoint_paths(cfg)), freeze_hash(cfg)[:12])

    p.set_defaults(func=run)
