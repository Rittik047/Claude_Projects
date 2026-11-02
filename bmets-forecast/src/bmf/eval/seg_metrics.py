"""Lesion-wise segmentation metrics in the style of the BraTS-METS challenge.

Regions: ET = {3}, TC = {1,3}, WT = {1,2,3}. GT lesions are 26-connected components
(GT mask dilated by `dilate` voxels only to group components that touch); a prediction
component counts as a TP for a GT lesion if it overlaps that lesion's dilated mask.
Lesion-wise Dice = sum of per-GT-lesion Dice / (n_GT + n_FP); unmatched GT -> Dice 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REGIONS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}
_S26 = np.ones((3, 3, 3), bool)


def _dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def lesionwise_metrics(pred: np.ndarray, gt: np.ndarray, spacing, labels=("ET", "TC", "WT"), dilate: int = 2) -> dict:
    out = {}
    for reg in labels:
        g, p = np.isin(gt, REGIONS[reg]), np.isin(pred, REGIONS[reg])
        glab, ng = ndi.label(g, _S26)
        plab, npred = ndi.label(p, _S26)
        matched_pred, dices, tp = set(), [], 0
        for i in range(1, ng + 1):
            gi = glab == i
            gd = ndi.binary_dilation(gi, _S26, iterations=dilate)
            hits = set(np.unique(plab[gd])) - {0}
            if hits:
                tp += 1
                matched_pred |= hits
                dices.append(_dice(gi, np.isin(plab, list(hits))))
            else:
                dices.append(0.0)
        fp = npred - len(matched_pred)
        fn = ng - tp
        denom = ng + fp
        out[reg] = {
            "lesion_dice": float(sum(dices) / denom) if denom else 1.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 1.0,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "global_dice": float(_dice(g, p)),
        }
    return out


def sensitivity_by_size(pred: np.ndarray, gt: np.ndarray, spacing, bins_mm=(3, 10), region: str = "TC") -> pd.DataFrame:
    """Detection sensitivity per GT lesion size bin (size = largest bounding-box extent in mm)."""
    sp = np.asarray(spacing, float)
    g = np.isin(gt, REGIONS[region]); p = np.isin(pred, REGIONS[region])
    glab, ng = ndi.label(g, _S26)
    names = [f"<={bins_mm[0]}mm"] + [f"{a}-{b}mm" for a, b in zip(bins_mm[:-1], bins_mm[1:])] + [f">{bins_mm[-1]}mm"]
    det = {n: [] for n in names}
    for i, sl in enumerate(ndi.find_objects(glab), start=1):
        ext = max((s.stop - s.start) * d for s, d in zip(sl, sp))
        k = int(np.searchsorted(bins_mm, ext, side="left"))
        det[names[k]].append(bool(p[glab == i].any()))
    return pd.DataFrame([(n, len(v), float(np.mean(v)) if v else np.nan) for n, v in det.items()],
                        columns=["bin", "n", "sensitivity"])
