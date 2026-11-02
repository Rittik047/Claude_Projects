"""Parse the UCSD-BMETS-Longitudinal clinical and MRI-metadata TSVs.

The TCIA files are Windows-1252 encoded and contain trailing spaces, non-breaking
spaces and inconsistent capitalisation; everything is normalised here so that the
rest of the pipeline sees clean, lower-case categorical values and booleans.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_RENAME = {
    "case id": "case_id",
    "sex at birth": "sex",
    "ethnicity": "ethnicity",
    "race": "race",
    "diagnosis (primary malignancy)": "primary",
    "diagnosis (biopsy/resction/imaging/csf)": "dx_method",
    "age at death": "age_death",
    "mri without metastases": "has_normal_mri",
    "age at imaging (mri without metastases)": "age_normal",
    "resection": "resection",
    "radiation before first mri": "rt_before",
    "age at imaging (first mri after diagnosis)": "age_t1",
    "radiation treatment between first and second mri": "rt_12",
    "age at imaging (second mri after diagnosis)": "age_t2",
    "radiation treatment between second and third mri": "rt_23",
    "age at imaging (third mri after diagnosis)": "age_t3",
    "radiation treatment between third and fourth mri": "rt_34",
    "age at imaging (fourth mri after diagnosis)": "age_t4",
}
AGE_COLS = ["age_t1", "age_t2", "age_t3", "age_t4"]
RT_COLS = ["rt_12", "rt_23", "rt_34"]
_BOOL_COLS = ["has_normal_mri", "resection", "rt_before", *RT_COLS]
_NUM_COLS = ["age_death", "age_normal", *AGE_COLS]


def _clean(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.replace("\xa0", " ", regex=False).str.replace("�", "", regex=False)
    s = s.str.strip().str.strip('"').str.strip()
    return s.mask(s == "", pd.NA)


def _to_bool(s: pd.Series) -> pd.Series:
    m = s.str.lower().map({"yes": True, "no": False})
    return m.astype(object).where(m.notna(), None)


def load_clinical(path: Path | str) -> pd.DataFrame:
    """One row per patient with normalised columns (see _RENAME values)."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="cp1252", keep_default_na=False)
    df.columns = [c.replace("\xa0", " ").strip().lower() for c in df.columns]
    df = df.rename(columns=_RENAME)
    df = df[list(_RENAME.values())]
    df = df.apply(_clean)
    df = df[df["case_id"].notna()].reset_index(drop=True)
    for c in ["sex", "ethnicity", "race", "primary", "dx_method"]:
        df[c] = df[c].str.lower().str.replace(r"\s*,\s*", ", ", regex=True)
    for c in _NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in _BOOL_COLS:
        df[c] = _to_bool(df[c])
    df["has_normal_mri"] = df["has_normal_mri"].fillna(False).astype(bool)
    df["n_post"] = df[AGE_COLS].notna().sum(axis=1).astype(int)
    df["case_id"] = df["case_id"].astype(str)
    return df


def timepoints(clin: pd.DataFrame) -> pd.DataFrame:
    """One row per TCIA folder BMETSxxxx_YY.

    Folders are numbered consecutively; when a pre-diagnosis normal MRI exists it is
    folder _01 (post_idx 0) and the post-diagnosis scans follow (post_idx 1..4).
    """
    rows = []
    for r in clin.itertuples(index=False):
        folder = 0
        if r.has_normal_mri:
            folder += 1
            rows.append((f"{r.case_id}_{folder:02d}", r.case_id, folder, 0, r.age_normal, True))
        for k in range(1, r.n_post + 1):
            folder += 1
            rows.append((f"{r.case_id}_{folder:02d}", r.case_id, folder, k, getattr(r, f"age_t{k}"), False))
    return pd.DataFrame(rows, columns=["study_id", "case_id", "folder_idx", "post_idx", "age", "is_normal"])


def intervals(clin: pd.DataFrame) -> pd.DataFrame:
    """One row per consecutive post-diagnosis pair (t -> t+1)."""
    rows = []
    for r in clin.itertuples(index=False):
        for t in range(1, r.n_post):
            a, b = getattr(r, f"age_t{t}"), getattr(r, f"age_t{t + 1}")
            dt = round(float(b - a), 6)
            rt = getattr(r, RT_COLS[t - 1])
            rows.append((r.case_id, t, t + 1, dt, bool(rt) if rt is not None else False, dt <= 0.0))
    return pd.DataFrame(rows, columns=["case_id", "t", "t_next", "dt_years", "rt_between", "zero_interval"])


def survival_table(clin: pd.DataFrame) -> pd.DataFrame:
    """Overall survival from first post-diagnosis MRI; censored at last MRI when alive."""
    last = clin[AGE_COLS].max(axis=1)
    dead = clin["age_death"].notna()
    end = np.where(dead, clin["age_death"], last)
    return pd.DataFrame({
        "case_id": clin["case_id"],
        "time_years": np.round(end - clin["age_t1"].to_numpy(), 6),
        "event": dead.astype(int).to_numpy(),
    })


SEQ_META_COLS = ["t1c", "t1n", "t2f", "t2w", "t1postfse", "dwi", "adc", "swan", "asl"]


def load_mri_meta(path: Path | str) -> pd.DataFrame:
    """MRI acquisition metadata: one row per study; sequence availability as bool."""
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="cp1252", keep_default_na=False)
    df.columns = [c.replace("\xa0", " ").strip().lower() for c in df.columns]
    df = df.apply(_clean)
    df = df.rename(columns={"study id": "study_id", "field strength": "field_strength"})
    df = df[df["study_id"].notna()].reset_index(drop=True)
    df["field_strength"] = pd.to_numeric(df["field_strength"], errors="coerce")
    for c in SEQ_META_COLS:
        if c in df:
            df[c] = df[c].str.lower().eq("yes").fillna(False).astype(bool)
    df["case_id"] = df["study_id"].str.rsplit("_", n=1).str[0]
    return df
