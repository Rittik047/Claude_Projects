"""Config loading: base.yaml merged with an optional route file (juno.yaml / colab.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | str | None = None) -> dict:
    """Load configs/base.yaml, then overlay `path` (if given and not base itself)."""
    base_path = Path(__file__).resolve().parents[2] / "configs" / "base.yaml"
    cfg = yaml.safe_load(base_path.read_text())
    if path is not None and Path(path).resolve() != base_path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    return cfg
