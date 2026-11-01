"""PTB-XL metadata and diagnostic-superclass labels.

Labelling follows the Strodthoff et al. (2021) benchmark: a superclass is positive if *any*
listed SCP code with `diagnostic == 1` maps to it, regardless of the code's likelihood.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd

from ecgtrust import CLASSES

AGE_DEID_CODE = 300  # PTB-XL codes ages > 89 as 300 for de-identification


def age_band(age: float) -> str:
    if age is None or (isinstance(age, float) and math.isnan(age)):
        return "unknown"
    if age >= AGE_DEID_CODE or age > 89:
        return ">89"
    if age >= 80:
        return "80-89"
    if age >= 60:
        return "60-79"
    if age >= 40:
        return "40-59"
    return "<40"


def load_scp(root: Path) -> pd.DataFrame:
    """SCP statement table indexed by code."""
    return pd.read_csv(Path(root) / "scp_statements.csv", index_col=0)


def _codes(meta: pd.DataFrame) -> pd.Series:
    col = meta["scp_codes"]
    return col if isinstance(col.iloc[0], dict) else col.apply(ast.literal_eval)


def superclass_labels(meta: pd.DataFrame, scp: pd.DataFrame) -> pd.DataFrame:
    """Boolean (n_records, 5) frame with columns CLASSES, same index as `meta`."""
    diag = scp[scp["diagnostic"] == 1]["diagnostic_class"].to_dict()
    codes = _codes(meta)
    rows = [{c: any(diag.get(k) == c for k in d) for c in CLASSES} for d in codes]
    return pd.DataFrame(rows, index=meta.index, columns=list(CLASSES)).astype(bool)


def load_metadata(root: Path) -> pd.DataFrame:
    """`ptbxl_database.csv` indexed by ecg_id, plus age_band, sex_label and has_label."""
    root = Path(root)
    meta = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    meta["patient_id"] = meta["patient_id"].astype(int)
    meta["age_band"] = meta["age"].apply(age_band)
    meta["sex_label"] = meta["sex"].map({0: "male", 1: "female"}).fillna("unknown")
    meta["has_label"] = superclass_labels(meta, load_scp(root)).any(axis=1)
    return meta
