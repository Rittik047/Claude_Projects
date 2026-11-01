"""Patient-level splits (spec §3) and the leakage guard.

train = folds 1-8; fold 9 is split 50/50 *by patient* into val_select (model selection)
and val_calib (temperature scaling + conformal calibration); test = fold 10.
Records with no diagnostic-superclass label are "excluded".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val_select", "val_calib", "test")


class LeakageError(Exception):
    """A patient appears in more than one split."""


def make_splits(meta: pd.DataFrame, seed: int = 0) -> pd.Series:
    fold = meta["strat_fold"]
    s = pd.Series("excluded", index=meta.index, name="split", dtype=object)
    s[fold <= 8] = "train"
    s[fold == 10] = "test"
    f9_patients = np.sort(meta.loc[fold == 9, "patient_id"].unique())
    perm = np.random.default_rng(seed).permutation(f9_patients)
    select = set(perm[: len(perm) // 2 + len(perm) % 2].tolist())
    in9 = fold == 9
    s[in9] = np.where(meta.loc[in9, "patient_id"].isin(select), "val_select", "val_calib")
    s[~meta["has_label"].astype(bool)] = "excluded"
    return s


def assert_no_patient_leakage(meta: pd.DataFrame, splits: pd.Series) -> None:
    used = splits[splits != "excluded"]
    n_splits = meta.loc[used.index, "patient_id"].groupby(used.values).unique()
    seen: dict[int, str] = {}
    for split, patients in n_splits.items():
        for p in patients:
            if p in seen and seen[p] != split:
                raise LeakageError(f"patient {p} in both {seen[p]} and {split}")
            seen[p] = split
