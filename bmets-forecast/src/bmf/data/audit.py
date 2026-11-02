"""Cohort audit: reproduces the numbers in the design spec (section 2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clinical import SEQ_META_COLS, intervals, load_clinical, load_mri_meta, survival_table, timepoints


def audit(clin: pd.DataFrame, meta: pd.DataFrame, out: Path | str | None = None) -> dict:
    tp, iv, sv = timepoints(clin), intervals(clin), survival_table(clin)
    dt = iv["dt_years"].to_numpy()
    deaths = sv[sv.event == 1]
    a = {
        "n_patients": int(len(clin)),
        "n_timepoints": int(len(tp)),
        "n_normal_mri": int(clin["has_normal_mri"].sum()),
        "post_mris_per_patient": {int(k): int(v) for k, v in clin["n_post"].value_counts().sort_index().items()},
        "n_intervals": int(len(iv)),
        "interval_years_median": float(np.median(dt)) if len(dt) else None,
        "interval_years_iqr": [float(x) for x in np.percentile(dt, [25, 75])] if len(dt) else None,
        "n_zero_intervals": int(iv["zero_interval"].sum()),
        "n_rt_intervals": int(iv["rt_between"].sum()),
        "rt_by_interval": {f"{t}->{t + 1}": f"{int(g.rt_between.sum())}/{len(g)}" for t, g in iv.groupby("t")},
        "n_resection_before": int(pd.Series(clin["resection"]).fillna(False).astype(bool).sum()),
        "n_deaths": int(len(deaths)),
        "median_years_first_mri_to_death": float(deaths.time_years.median()) if len(deaths) else None,
        "primary": {str(k): int(v) for k, v in clin["primary"].value_counts().items()},
        "sex": {str(k): int(v) for k, v in clin["sex"].value_counts().items()},
        "age_first_mri_mean_sd": [float(clin.age_t1.mean()), float(clin.age_t1.std())],
        "n_studies_meta": int(len(meta)),
        "manufacturer": {str(k): int(v) for k, v in meta["manufacturer"].value_counts().items()},
        "field_strength": {str(k): int(v) for k, v in meta["field_strength"].value_counts().items()},
        "sequence_availability": {c: float(meta[c].mean()) for c in SEQ_META_COLS if c in meta},
    }
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(a, indent=2))
    return a


def main(argv=None):
    p = argparse.ArgumentParser(description="Cohort audit of UCSD-BMETS-Longitudinal TSVs")
    p.add_argument("--clinical", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--out", default="reports/audit.json")
    a = p.parse_args(argv)
    res = audit(load_clinical(a.clinical), load_mri_meta(a.meta), a.out)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
