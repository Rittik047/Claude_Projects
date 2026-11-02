"""Leak-free design matrices indexed by (case_id, t): row t only uses information up to time point t
(plus radiation planned for the interval t -> t+1 and the interval length, both known at decision time)."""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARIES = ("lung", "breast", "melanoma", "gastrointestinal", "renal", "prostate", "other")


def clinical_design(clin: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    c = clin.set_index("case_id")
    rows = []
    for r in intervals.itertuples(index=False):
        p = c.loc[r.case_id]
        prim = str(p.primary) if isinstance(p.primary, str) else "other"
        prim = prim if prim in PRIMARIES else "other"
        row = {
            "case_id": r.case_id, "t": r.t,
            "age": float(p[f"age_t{r.t}"]), "sex_male": float(p.sex == "male"),
            "resection": float(bool(p.resection)), "has_normal_mri": float(bool(p.has_normal_mri)),
            "rt_between": float(r.rt_between), "dt_years": float(r.dt_years),
            "zero_interval": float(r.zero_interval), "n_prior_tp": float(r.t - 1),
            "years_since_dx": float(p[f"age_t{r.t}"] - p.age_t1),
        }
        row.update({f"primary_{k}": float(prim == k) for k in PRIMARIES})
        rows.append(row)
    return pd.DataFrame(rows).set_index(["case_id", "t"])


def _tp_stats(d: pd.DataFrame) -> dict:
    tc, wt = float(d.tc_cc.sum()), float(d.wt_cc.drop_duplicates().sum()) if len(d) else 0.0
    return {"n_lesions": float(len(d)), "sum_et": float(d.et_cc.sum()), "sum_tc": tc, "sum_wt": wt,
            "max_diam": float(d.max_diam_mm.max()) if len(d) else 0.0,
            "edema_ratio": float((wt - tc) / tc) if tc > 0 else 0.0,
            "log_sum_et": float(np.log1p(d.et_cc.sum() * 1000))}


def lesion_summary(per_tp: dict, tracks: pd.DataFrame, clin: pd.DataFrame, intervals: pd.DataFrame) -> pd.DataFrame:
    """Lesion burden at t, change vs t-1, and brain-metastasis velocity (new lesions per year so far)."""
    c = clin.set_index("case_id")
    empty = pd.DataFrame(columns=["et_cc", "tc_cc", "wt_cc", "max_diam_mm"])
    rows = []
    for r in intervals.itertuples(index=False):
        cur = _tp_stats(per_tp.get((r.case_id, r.t), empty))
        row = {"case_id": r.case_id, "t": r.t, **cur}
        if r.t > 1:
            prev = _tp_stats(per_tp.get((r.case_id, r.t - 1), empty))
            row.update({f"d_{k}": cur[k] - prev[k] for k in ("n_lesions", "sum_et", "sum_tc", "sum_wt", "log_sum_et")})
        else:
            row.update({f"d_{k}": 0.0 for k in ("n_lesions", "sum_et", "sum_tc", "sum_wt", "log_sum_et")})
        tr = tracks[(tracks.case_id == r.case_id) & (tracks.post_idx <= r.t)]
        n_new = float((tr.status == "new").sum())
        yrs = float(c.loc[r.case_id, f"age_t{r.t}"] - c.loc[r.case_id, "age_t1"])
        row["n_new_so_far"] = n_new
        row["new_per_year"] = n_new / yrs if yrs > 0 else 0.0
        row["n_resolved_so_far"] = float((tr.status == "resolved").sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index(["case_id", "t"])


def add_deltas(R: pd.DataFrame, prefix: str = "rad_") -> pd.DataFrame:
    """Append change-over-time columns rad_delta_<name> = value(t) - value(t-1) within patient (0 at t = 1)."""
    R = R.sort_index()
    D = R.groupby(level="case_id").diff().fillna(0.0)
    D.columns = [c.replace(prefix, f"{prefix}delta_", 1) for c in R.columns]
    return R.join(D)
