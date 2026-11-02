"""Interval-level (T2) and lesion-level (T3) labels: volumetric RANO-BM adaptation.

Thresholds (spec): progression >= +73 % vs nadir or new lesion >= 0.05 cc;
partial response <= -65 % vs baseline; complete response = enhancing volume 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROG, PR, NEW_MIN = 0.73, -0.65, 0.05
EPS_CC = 0.001


def _et(per_tp: dict, case, post) -> pd.DataFrame:
    return per_tp.get((case, post), pd.DataFrame(columns=["lesion_id", "et_cc"]))


def interval_labels(tracks: pd.DataFrame, per_tp: dict, intervals: pd.DataFrame,
                    prog: float = PROG, new_min_cc: float = NEW_MIN) -> pd.DataFrame:
    """One row per interval: prog_t2a, newmet_t2b, sum_et_t, sum_et_next, nadir_et."""
    out = []
    for r in intervals.itertuples(index=False):
        sums = [float(_et(per_tp, r.case_id, p).et_cc.sum()) for p in range(1, r.t_next + 1)]
        s_t, s_n, nadir = sums[r.t - 1], sums[r.t_next - 1], min(sums[: r.t])
        tr = tracks[(tracks.case_id == r.case_id) & (tracks.post_idx == r.t_next) & (tracks.status == "new")]
        les = _et(per_tp, r.case_id, r.t_next).set_index("lesion_id")
        new_cc = [float(les.loc[l, "et_cc"]) for l in tr.lesion_id if l in les.index]
        newmet = int(any(v >= new_min_cc for v in new_cc))
        grow = (s_n - nadir) / nadir >= prog if nadir > 0 else s_n >= new_min_cc
        out.append((r.case_id, r.t, r.t_next, int(bool(grow) or bool(newmet)), newmet, s_t, s_n, nadir))
    return pd.DataFrame(out, columns=["case_id", "t", "t_next", "prog_t2a", "newmet_t2b", "sum_et_t", "sum_et_next", "nadir_et"])


def lesion_trajectories(tracks: pd.DataFrame, per_tp: dict, intervals: pd.DataFrame,
                        prog: float = PROG, pr: float = PR) -> pd.DataFrame:
    """One row per (track, interval) where the lesion exists at t: log_ratio, rano_class, growth rate."""
    out = []
    for r in intervals.itertuples(index=False):
        tc = tracks[tracks.case_id == r.case_id]
        for tid, g in tc.groupby("track_id"):
            vols = {}
            for p in range(1, r.t_next + 1):
                row = g[g.post_idx == p]
                lid = row.lesion_id.iloc[0] if len(row) else np.nan
                les = _et(per_tp, r.case_id, p)
                vols[p] = float(les.loc[les.lesion_id == lid, "et_cc"].sum()) if not pd.isna(lid) else 0.0
            v_t, v_n = vols[r.t], vols[r.t_next]
            if v_t <= 0:
                continue
            first = min(p for p in range(1, r.t + 1) if vols[p] > 0)        # first appearance
            base = vols[first]
            nadir = min(vols[p] for p in range(first, r.t + 1))
            if v_n == 0:
                cls = "CR"
            elif nadir > 0 and (v_n - nadir) / nadir >= prog:
                cls = "PD"
            elif (v_n - base) / base <= pr:
                cls = "PR"
            else:
                cls = "SD"
            lr = float(np.log((v_n + EPS_CC) / (v_t + EPS_CC)))
            gr = lr / r.dt_years if r.dt_years > 0 else np.nan
            row = g[g.post_idx == r.t].iloc[0]
            les = _et(per_tp, r.case_id, r.t)
            cx = float(les.loc[les.lesion_id == row.lesion_id, "cx"].iloc[0]) if "cx" in les else np.nan
            out.append((r.case_id, tid, r.t, r.t_next, v_t, v_n, lr, cls, gr, cx))
    return pd.DataFrame(out, columns=["case_id", "track_id", "t", "t_next", "et_t", "et_next", "log_ratio",
                                      "rano_class", "growth_rate_per_yr", "cx"])
