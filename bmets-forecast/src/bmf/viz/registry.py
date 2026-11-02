"""FIGURES registry: name -> (function, builder(bundle) -> kwargs).

`demo_bundle()` fabricates schema-correct inputs so every figure can be rendered without data
(tests; labelled 'synthetic example' in the guide). The CLI builds the real bundle from results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data_figs as D, pred_figs as P, seg_figs as S, surv_figs as V, traj_figs as T


def demo_bundle(seed: int = 0, n: int = 60) -> dict:
    rng = np.random.default_rng(seed)
    prim = rng.choice(["lung", "breast", "melanoma", "renal", "other"], n, p=[.4, .3, .1, .1, .1])
    n_post = rng.choice([2, 3, 4], n, p=[.2, .1, .7])
    age1 = rng.normal(62, 12, n).round(1)
    clin = pd.DataFrame({"case_id": [f"BMETS{i:04d}" for i in range(n)], "primary": prim, "n_post": n_post,
                         "has_normal_mri": rng.random(n) < .06, "age_t1": age1, "sex": rng.choice(["female", "male"], n)})
    for k in range(2, 5):
        clin[f"age_t{k}"] = np.where(n_post >= k, age1 + 0.2 * (k - 1), np.nan)
    for a, b in ((1, 2), (2, 3), (3, 4)):
        clin[f"rt_{a}{b}"] = np.where(n_post > a, rng.random(n) < .35, None)
    last = clin[[f"age_t{k}" for k in range(1, 5)]].max(1)
    clin["age_death"] = np.where(rng.random(n) < .5, last + rng.exponential(.4, n), np.nan)
    iv = pd.DataFrame([(c, t, t + 1, 0.2 if rng.random() > .1 else 0.0, bool(rng.random() < .33)) for c, k in zip(clin.case_id, n_post)
                       for t in range(1, k)], columns=["case_id", "t", "t_next", "dt_years", "rt_between"])
    iv["zero_interval"] = iv.dt_years == 0
    surv = pd.DataFrame({"case_id": clin.case_id, "time_years": np.where(clin.age_death.notna(), clin.age_death - age1, last - age1).clip(0.05),
                         "event": clin.age_death.notna().astype(int)})
    studies = [f"{c}_{k:02d}" for c, m in zip(clin.case_id, n_post) for k in range(1, m + 1)]
    meta = pd.DataFrame({"study_id": studies, "manufacturer": rng.choice(["GE Medical Systems", "Siemens"], len(studies), p=[.95, .05]),
                         "model": rng.choice(["Signa HDxt", "DISCOVERY MR750w"], len(studies)), "field_strength": rng.choice([1.5, 3.0], len(studies))})
    for s in ["t1c", "t1n", "t2f", "t2w", "t1postfse", "dwi", "adc", "swan", "asl"]:
        meta[s] = rng.random(len(studies)) < (1 if s in ("t1c", "t1n", "t2f", "t2w") else .9)
    y = rng.integers(0, 2, len(iv))
    oofs = {m: pd.DataFrame({"case_id": iv.case_id, "t": iv.t, "y": y, "fold": rng.integers(0, 5, len(iv)),
                             "pred": 1 / (1 + np.exp(-(s * (y - .5) * 2 + rng.normal(0, 1, len(iv)))))})
            for m, s in (("P0", .4), ("P1", .8), ("P3", 1.0))}
    X = pd.DataFrame(rng.normal(size=(len(iv), 8)), columns=["sum_et", "d_sum_et", "n_lesions", "new_per_year", "rt_between",
                                                               "dt_years", "age", "edema_ratio"])
    shap_vals = X.to_numpy() * np.array([.8, 1.2, .5, .9, -.6, .3, .1, .2]) + rng.normal(0, .1, X.shape)
    vol = {s: rng.normal(0, 1, (48, 48, 24)).astype(np.float32) for s in ("t1c", "t1n", "t2f", "t2w")}
    seg = np.zeros((48, 48, 24), np.int16); seg[16:30, 16:30, 8:16] = 2; seg[20:26, 20:26, 10:14] = 3; seg[22:24, 22:24, 11:13] = 1
    lesion_tab = pd.DataFrame({"study_id": rng.choice(studies, 300), "et_cc": rng.lognormal(-1, 1.2, 300)})
    tracks = pd.DataFrame({"post_idx": rng.integers(1, 5, 400), "status": rng.choice(["baseline", "stable", "new", "resolved", "recurred"], 400)})
    size_tab = pd.DataFrame([(m, b, 20, s) for m, off in (("S0", 0), ("S2", .08)) for b, s in (("<=3mm", .45 + off), ("3-10mm", .85 + off / 2), (">10mm", .98))],
                            columns=["method", "bin", "n", "sensitivity"])
    per_case = pd.DataFrame({"method": rng.choice(["S0", "S1", "S2"], 200), "lesion_dice": rng.beta(6, 2, 200), "field_strength": rng.choice([1.5, 3.0], 200)})
    yrs = np.tile([0, .2, .4, .6], 120)
    traj_long = pd.DataFrame({"case_id": np.repeat(np.arange(120), 4), "track_id": 0, "years": yrs,
                              "et_cc": np.exp(np.repeat(rng.normal(0, .5, 120), 4) * yrs * 3) * .5, "primary": np.repeat(rng.choice(["lung", "breast", "melanoma"], 120), 4)})
    curves = np.cumsum(rng.normal(0, 20, (90, 6)), 1); labels = np.repeat([0, 1, 2], 30); curves[labels == 1] += 60; curves[labels == 2] -= 60
    traj = pd.DataFrame({"primary": rng.choice(["lung", "breast", "melanoma"], 300), "rano_class": rng.choice(["CR", "PR", "SD", "PD"], 300, p=[.1, .2, .5, .2])})
    obs = rng.normal(0, 1, 200)
    st = pd.DataFrame([("sex", "male", 150, 40, .74, .65, .82), ("sex", "female", 180, 50, .71, .63, .79), ("primary", "lung", 120, 35, .76, .66, .85)],
                      columns=["variable", "level", "n", "n_pos", "auroc", "lo", "hi"])
    return dict(clin=clin, iv=iv, surv=surv, meta=meta, oofs=oofs, X=X, shap=shap_vals, vol=vol, seg=seg, lesion_tab=lesion_tab,
                tracks=tracks, size_tab=size_tab, per_case=per_case, traj_long=traj_long, curves=curves, labels=labels, traj=traj,
                obs=obs, pred=obs * .7 + rng.normal(0, .5, 200), subgroups=st,
                delta=pd.DataFrame({"model": ["P1", "P3", "P5"], "delta": [.04, .07, .09], "lo": [-.01, .02, .03], "hi": [.09, .12, .15], "p_holm": [.2, .02, .01]}),
                ablation=pd.DataFrame({"dropped": ["none", "t1c", "t2f", "t1n", "t2w"], "score": [.75, .66, .72, .74, .745], "delta": [0, -.09, -.03, -.01, -.005]}),
                attention=pd.DataFrame([[.1, .2, .7], [.3, .3, .4]], index=["progressed", "stable"], columns=[1, 2, 3]),
                emb=rng.normal(size=(150, 32)) + np.repeat(np.eye(3, 32) * 3, 50, 0), emb_labels=pd.Series(np.repeat(["lung", "breast", "melanoma"], 50)),
                boot={m: rng.normal(.65 + .05 * i, .03, 500) for i, m in enumerate(("P0", "P1", "P3"))},
                inter=pd.DataFrame(np.abs(rng.normal(0, .1, (6, 6))), index=list(X.columns[:6]), columns=list(X.columns[:6])),
                risk=pd.Series(rng.normal(size=n), index=range(n)), td={"O1-Cox": {0.5: .72, 1.0: .70}, "O1-DeepSurv": {0.5: .74, 1.0: .71}},
                fn_crops=[(rng.normal(size=(24, 24)), np.pad(np.ones((4, 4)), 10), "1.8 mm · t1c")] * 3)


FIGURES = {
    # data
    "cohort_flow": (D.fig_cohort_flow, lambda b: dict(clin=b["clin"], iv=b["iv"])),
    "swimmer": (D.fig_swimmer, lambda b: dict(clin=b["clin"])),
    "interval_hist": (D.fig_interval_hist, lambda b: dict(iv=b["iv"])),
    "rt_by_interval": (D.fig_rt_by_interval, lambda b: dict(iv=b["iv"])),
    "primary_bars": (D.fig_primary_bars, lambda b: dict(clin=b["clin"])),
    "age_by_primary": (D.fig_age_by_primary, lambda b: dict(clin=b["clin"])),
    "scanner_mosaic": (D.fig_scanner_mosaic, lambda b: dict(meta=b["meta"])),
    "sequence_availability": (D.fig_sequence_availability, lambda b: dict(meta=b["meta"])),
    "km_by_primary": (D.fig_km_by_primary, lambda b: dict(clin=b["clin"], surv=b["surv"])),
    "timepoints_per_patient": (D.fig_timepoints_per_patient, lambda b: dict(clin=b["clin"])),
    "followup_to_death": (D.fig_followup_to_death, lambda b: dict(surv=b["surv"])),
    # imaging / T1
    "montage": (S.fig_montage, lambda b: dict(vols=b["vol"], seg=b["seg"])),
    "lesion_count_volume": (S.fig_lesion_count_volume, lambda b: dict(lesion_tab=b["lesion_tab"])),
    "seg_dice_by_size": (S.fig_seg_dice_by_size, lambda b: dict(size_tab=b["size_tab"])),
    "seg_method_compare": (S.fig_seg_method_compare, lambda b: dict(per_case=b["per_case"])),
    "tracking_flow": (S.fig_tracking_flow, lambda b: dict(tracks=b["tracks"])),
    "seg_by_field_strength": (S.fig_seg_by_field_strength, lambda b: dict(per_case=b["per_case"])),
    "fn_gallery": (S.fig_fn_gallery, lambda b: dict(crops=b["fn_crops"])),
    # T2
    "roc": (P.fig_roc, lambda b: dict(oofs=b["oofs"])),
    "pr": (P.fig_pr, lambda b: dict(oofs=b["oofs"])),
    "delta_forest": (P.fig_delta_forest, lambda b: dict(delta_tab=b["delta"])),
    "calibration": (P.fig_calibration, lambda b: dict(oofs=b["oofs"])),
    "decision_curve": (P.fig_decision_curve, lambda b: dict(oofs=b["oofs"])),
    "shap_beeswarm": (P.fig_shap_beeswarm, lambda b: dict(vals=b["shap"], X=b["X"])),
    "shap_bar": (P.fig_shap_bar, lambda b: dict(vals=b["shap"], X=b["X"])),
    "shap_dependence": (P.fig_shap_dependence, lambda b: dict(vals=b["shap"], X=b["X"])),
    "shap_interaction": (P.fig_shap_interaction, lambda b: dict(inter=b["inter"])),
    "saliency_overlay": (P.fig_saliency_overlay, lambda b: dict(img2d=b["vol"]["t1c"][:, :, 12], sal2d=np.abs(b["vol"]["t2f"][:, :, 12]), mask2d=(b["seg"][:, :, 12] == 3))),
    "attention_time": (P.fig_attention_time, lambda b: dict(att=b["attention"])),
    "embedding_umap": (P.fig_embedding_umap, lambda b: dict(emb=b["emb"], labels=b["emb_labels"])),
    "sequence_ablation": (P.fig_sequence_ablation, lambda b: dict(abl=b["ablation"])),
    "subgroup_forest": (P.fig_subgroup_forest, lambda b: dict(st=b["subgroups"])),
    "bootstrap_dist": (P.fig_bootstrap_dist, lambda b: dict(boot=b["boot"])),
    "risk_distribution": (P.fig_risk_distribution, lambda b: dict(oof=b["oofs"]["P3"])),
    # T3
    "spaghetti": (T.fig_spaghetti, lambda b: dict(traj_long=b["traj_long"])),
    "trajectory_clusters": (T.fig_trajectory_clusters, lambda b: dict(curves=b["curves"], labels=b["labels"], grid=np.linspace(0, 1, 6))),
    "pred_vs_obs": (T.fig_pred_vs_obs, lambda b: dict(obs=b["obs"], pred=b["pred"])),
    "waterfall": (T.fig_waterfall, lambda b: dict(best_change=pd.Series(b["obs"] * .8))),
    "rano_class_bars": (T.fig_rano_class_bars, lambda b: dict(traj=b["traj"])),
    # T4
    "km_risk_tertiles": (V.fig_km_risk_tertiles, lambda b: dict(surv=b["surv"], risk=b["risk"])),
    "td_auc": (V.fig_td_auc, lambda b: dict(td=b["td"])),
}

DATA_FIGURES = ["cohort_flow", "swimmer", "interval_hist", "rt_by_interval", "primary_bars", "age_by_primary", "scanner_mosaic",
                "sequence_availability", "km_by_primary", "timepoints_per_patient", "followup_to_death"]


def render(name: str, bundle: dict, out_dir: Path | str) -> Path:
    fn, build = FIGURES[name]
    return fn(**build(bundle), out=Path(out_dir) / name)


def data_bundle(clinical_tsv, meta_tsv) -> dict:
    """Real bundle for the data-description figures (public TSVs only)."""
    from ..data.clinical import intervals, load_clinical, load_mri_meta, survival_table
    clin = load_clinical(clinical_tsv)
    return dict(clin=clin, iv=intervals(clin), surv=survival_table(clin), meta=load_mri_meta(meta_tsv))
