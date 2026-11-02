"""Lesion correspondence across registered time points (Hungarian on centroid distance)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def match(prev: pd.DataFrame, curr: pd.DataFrame, gate_mm: float = 10.0) -> list[tuple]:
    """Pairs (prev_id|None, curr_id|None); assignments farther than gate_mm are rejected."""
    pairs, used_p, used_c = [], set(), set()
    if len(prev) and len(curr):
        P = prev[["cx", "cy", "cz"]].to_numpy(float)
        C = curr[["cx", "cy", "cz"]].to_numpy(float)
        D = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=-1)
        BIG = 1e6                                      # gate BEFORE assignment, so far pairs never displace near ones
        for i, j in zip(*linear_sum_assignment(np.where(D <= gate_mm, D, BIG))):
            if D[i, j] <= gate_mm:
                pairs.append((prev.lesion_id.iloc[i], curr.lesion_id.iloc[j]))
                used_p.add(prev.lesion_id.iloc[i]); used_c.add(curr.lesion_id.iloc[j])
    pairs += [(p, None) for p in prev.lesion_id if p not in used_p]
    pairs += [(None, c) for c in curr.lesion_id if c not in used_c]
    return pairs


def track_patient(per_tp: list[pd.DataFrame], gate_mm: float = 10.0) -> pd.DataFrame:
    """per_tp[k] = lesions at post-diagnosis time point k+1.

    status: baseline (first time point) | stable (matched, present before) |
    new (no prior track within gate) | recurred (matched to a track absent at t-1) |
    resolved (track present at t-1, absent now; lesion_id = NaN).
    Matching uses each track's last known centroid, so a lesion that disappears and
    re-grows at the same site is 'recurred', not 'new'.
    """
    rows, last = [], {}            # track_id -> (cx,cy,cz)
    active, next_id = set(), 0
    for k, df in enumerate(per_tp):
        post = k + 1
        if last:
            prev = pd.DataFrame([(tid, *c) for tid, c in last.items()], columns=["lesion_id", "cx", "cy", "cz"])
        else:
            prev = pd.DataFrame(columns=["lesion_id", "cx", "cy", "cz"])
        pairs = match(prev, df, gate_mm)
        now = set()
        for tid, lid in pairs:
            if lid is None:
                if tid in active:
                    rows.append((tid, post, np.nan, "resolved"))
                continue
            if tid is None:
                tid, next_id = next_id, next_id + 1
                status = "baseline" if k == 0 else "new"
            else:
                status = "stable" if tid in active else "recurred"
            r = df[df.lesion_id == lid].iloc[0]
            last[tid] = (r.cx, r.cy, r.cz)
            now.add(tid)
            rows.append((tid, post, lid, status))
        active = now
    return pd.DataFrame(rows, columns=["track_id", "post_idx", "lesion_id", "status"])
