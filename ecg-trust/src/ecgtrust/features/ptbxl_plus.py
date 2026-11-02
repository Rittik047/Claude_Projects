"""PTB-XL+ ECG measurements (Strodthoff et al. 2023): GE 12SL and open-source ECGDeli."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

FILES = {"12sl": "12sl_features.csv", "ecgdeli": "ecgdeli_features.csv"}


def load_features(root: Path, source: Literal["12sl", "ecgdeli"]) -> pd.DataFrame:
    """Numeric feature table indexed by ecg_id; missing measurements stay NaN."""
    df = pd.read_csv(Path(root) / "features" / FILES[source])
    df = df.set_index("ecg_id")
    return df.select_dtypes("number").astype("float32")
