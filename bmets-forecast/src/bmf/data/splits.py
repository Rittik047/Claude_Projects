"""Patient-level stratified K-fold splits (stratum = primary tumour x number of post-dx MRIs)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def make_splits(clin: pd.DataFrame, n_folds: int = 5, seed: int = 2026) -> dict[str, int]:
    prim = clin["primary"].fillna("other").astype(str)
    counts = prim.value_counts()
    prim = prim.where(prim.map(counts) >= n_folds, "other")
    strata = prim + "|" + clin["n_post"].astype(str)
    sc = strata.value_counts()
    strata = strata.where(strata.map(sc) >= n_folds, prim)          # collapse rare strata
    sc = strata.value_counts()
    strata = strata.where(strata.map(sc) >= n_folds, "all")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out: dict[str, int] = {}
    ids = clin["case_id"].astype(str).to_numpy()
    for k, (_, te) in enumerate(skf.split(ids, strata)):
        for i in te:
            out[ids[i]] = k
    return out


def save_splits(splits: dict[str, int], path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(sorted(splits.items())), indent=1))


def load_splits(path: Path | str) -> dict[str, int]:
    return {k: int(v) for k, v in json.loads(Path(path).read_text()).items()}
