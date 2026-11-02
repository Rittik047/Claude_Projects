"""Typed access to configs/default.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path("configs/default.yaml")


@dataclass(frozen=True)
class Config:
    data_root: Path
    artifacts: Path
    reports: Path
    mirror: str
    versions: dict[str, str]
    seeds: list[int]
    alphas: list[float]
    n_boot: int
    ece_bins: int
    snr_db: list[float]
    age_bands: list[str]
    external_min_pos: int
    train: dict = field(default_factory=dict)

    def dataset_dir(self, name: str) -> Path:
        return self.data_root / name


def load_config(path: str | Path = DEFAULT_CONFIG, **overrides) -> Config:
    """Load the YAML config; relative paths resolve against the config file's parent dir."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    raw.update(overrides)
    base = path.resolve().parent.parent
    for key in ("data_root", "artifacts", "reports"):
        p = Path(raw[key])
        raw[key] = p if p.is_absolute() else base / p
    return Config(**raw)
