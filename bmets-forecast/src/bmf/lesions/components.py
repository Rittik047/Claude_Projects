"""Per-lesion measurements from a BraTS-METS-style label map."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

_COLS = ["lesion_id", "cx", "cy", "cz", "et_cc", "tc_cc", "wt_cc", "max_diam_mm"]
_S26 = np.ones((3, 3, 3), bool)


def lesions(seg: np.ndarray, spacing, labels_cfg: dict) -> pd.DataFrame:
    """Lesions = 26-connected components of ET ∪ NETC (resection cavity excluded).

    Volumes in cc; centroid in mm (voxel index × spacing); wt_cc = volume of the
    WT (ET ∪ NETC ∪ SNFH) component containing the lesion; max_diam_mm = largest
    bounding-box extent in mm.
    """
    sp = np.asarray(spacing, float)
    vox_cc = float(np.prod(sp)) / 1000.0
    et, netc, snfh = labels_cfg["ET"], labels_cfg["NETC"], labels_cfg["SNFH"]
    tc_mask = np.isin(seg, (et, netc))
    lab, n = ndi.label(tc_mask, structure=_S26)
    if n == 0:
        return pd.DataFrame(columns=_COLS).astype({c: float for c in _COLS[1:]})
    wt_lab, _ = ndi.label(np.isin(seg, (et, netc, snfh)), structure=_S26)
    wt_sizes = np.bincount(wt_lab.ravel())
    idx = np.arange(1, n + 1)
    tc_v = ndi.sum(np.ones_like(lab), lab, idx)
    et_v = ndi.sum(seg == et, lab, idx)
    cents = ndi.center_of_mass(tc_mask, lab, idx)
    slices = ndi.find_objects(lab)
    rows = []
    for k, (i, c, sl) in enumerate(zip(idx, cents, slices)):
        wt_id = np.bincount(wt_lab[sl][lab[sl] == i]).argmax()
        ext = max((s.stop - s.start) * d for s, d in zip(sl, sp))
        rows.append((int(i), *(np.asarray(c) * sp), et_v[k] * vox_cc, tc_v[k] * vox_cc, wt_sizes[wt_id] * vox_cc, float(ext)))
    return pd.DataFrame(rows, columns=_COLS)
