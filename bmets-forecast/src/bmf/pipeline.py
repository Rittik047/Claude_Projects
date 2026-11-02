"""Pipeline stages behind the `bmf` CLI. Each stage reads/writes `paths.cache`; reports are aggregate-only.

Stage order: audit -> index -> splits -> preprocess -> [seg-export / nnU-Net on GPU] -> track -> labels
-> features -> embed -> train (per model/task) -> evaluate -> explain -> figures -> report
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .data.audit import audit as _audit
from .data.clinical import intervals, load_clinical, load_mri_meta, survival_table, timepoints
from .data.mri_index import SEQS_CORE, build_index
from .data.splits import load_splits, make_splits, save_splits

log = logging.getLogger("bmf")
LAB = None


def _git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class Ctx:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        p = cfg["paths"]
        self.cache, self.reports = Path(p["cache"]), Path(p["reports"])
        self.figs = Path(p.get("figures", self.reports / "figures"))
        if (cfg.get("subset") or {}).get("n_patients"):   # smoke runs never touch the full run's outputs
            rep = self.reports
            self.cache = self.cache.with_name(self.cache.name + "_smoke")
            self.reports = rep.with_name(rep.name + "_smoke")
            self.figs = self.reports / self.figs.relative_to(rep) if self.figs.is_relative_to(rep) else self.reports / "figures"
        self.cache.mkdir(parents=True, exist_ok=True); self.reports.mkdir(parents=True, exist_ok=True)
        self.labels = cfg.get("labels", {"NETC": 1, "SNFH": 2, "ET": 3, "RC": 4})

    def clin(self):
        c = load_clinical(self.cfg["paths"]["clinical"])
        sub = self.cfg.get("subset") or {}
        n = sub.get("n_patients")
        if n:   # smoke tests: seeded sample of patients with enough scans to form intervals
            n_tp = c.filter(like="age_t").notna().sum(axis=1)
            pool = c[n_tp >= sub.get("min_timepoints", 3)]
            c = pool.sample(n=min(int(n), len(pool)), random_state=self.cfg.get("seed", 2026))
            c = c.sort_values("case_id").reset_index(drop=True)
        return c

    def tp(self):
        return pd.read_csv(self.cache / "timepoints.csv")

    def index(self):
        return pd.read_csv(self.cache / "index.csv", keep_default_na=False).assign(exists=lambda d: d.exists.astype(str) == "True")

    def splits(self):
        return load_splits(self.cache / "splits.json")

    def pp(self, sid):
        return self.cache / "pp" / sid

    def train_cfg(self, k, default=None):
        return self.cfg.get("train", {}).get(k, default)


# ---------------------------------------------------------------- data stages
def stage_audit(c: Ctx, **_):
    a = _audit(c.clin(), load_mri_meta(c.cfg["paths"]["mri_meta"]), c.reports / "audit.json")
    log.info("audit: %s patients, %s time points, %s intervals", a["n_patients"], a["n_timepoints"], a["n_intervals"])


def stage_index(c: Ctx, **_):
    tp = timepoints(c.clin())
    tp.to_csv(c.cache / "timepoints.csv", index=False)
    idx = build_index(c.cfg["paths"]["data_root"], tp, c.cfg.get("seg_glob", "*seg*.nii.gz"))
    idx.to_csv(c.cache / "index.csv", index=False)
    miss = idx[(idx.seq.isin(SEQS_CORE + ("seg",))) & ~idx.exists]
    log.info("index: %d studies, %d missing core/seg files", tp.shape[0], len(miss))


def stage_splits(c: Ctx, **_):
    save_splits(make_splits(c.clin(), c.cfg.get("n_folds", 5), c.cfg.get("seed", 2026)), c.cache / "splits.json")


def stage_preprocess(c: Ctx, shard: str | None = None, force: bool = False, **_):
    """Per patient, the first folder (normal MRI if present, else first post-dx scan) is the baseline grid."""
    from .preprocess.pipeline import preprocess_study
    tp, idx = c.tp(), c.index()
    pc = c.cfg.get("preprocess", {})
    cases = sorted(tp.case_id.unique())
    if shard:
        i, n = map(int, shard.split("/"))
        cases = cases[i::n]
    for cid in cases:
        g = tp[tp.case_id == cid].sort_values("folder_idx")
        base = None
        for sid in g.study_id:
            if (c.pp(sid) / "seg.nii.gz").exists() and not force:
                base = base or sid; continue
            preprocess_study(idx, sid, base, c.cache / "pp", spacing=pc.get("spacing_mm", 1.0), n4=pc.get("n4", True),
                             backend=pc.get("register_backend", "sitk"), affine_longitudinal=pc.get("affine_longitudinal", True))
            base = base or sid
    log.info("preprocess: %d patients", len(cases))


def stage_seg_export(c: Ctx, **_):
    from .seg.nnunet_export import export_nnunet, nnunet_cmds
    out = c.cache / "nnunet_raw"
    s0 = export_nnunet(c.cache / "pp", c.tp(), c.splits(), out, 501, prior=False)
    s2 = export_nnunet(c.cache / "pp", c.tp(), c.splits(), out, 502, prior=True)
    k = c.cfg.get("n_folds", 5)
    (c.cache / "nnunet_cmds.sh").write_text("\n".join(nnunet_cmds(501, range(k)) + nnunet_cmds(502, range(k))) + "\n")
    log.info("nnU-Net datasets: %s, %s", s0, s2)


def stage_seg_eval(c: Ctx, pred_dir: str | None = None, method: str = "S0", **_):
    """Evaluate OOF nnU-Net predictions (<pred_dir>/<study_id>.nii.gz) against expert labels."""
    import nibabel as nib
    from .eval.seg_metrics import lesionwise_metrics, sensitivity_by_size
    meta = load_mri_meta(c.cfg["paths"]["mri_meta"]).set_index("study_id")
    rows, sizes = [], []
    for p in sorted(Path(pred_dir).glob("*.nii.gz")):
        sid = p.name.replace(".nii.gz", "")
        gt = np.rint(np.asarray(nib.load(str(c.pp(sid) / "seg.nii.gz")).dataobj)).astype(int)
        pr = np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(int)
        m = lesionwise_metrics(pr, gt, (1, 1, 1))
        rows.append({"method": method, "study_id": sid, "field_strength": meta.field_strength.get(sid, np.nan),
                     **{f"{r}_{k}": v for r, d in m.items() for k, v in d.items()}})
        sizes.append(sensitivity_by_size(pr, gt, (1, 1, 1)).assign(method=method, study_id=sid))
    (c.reports / "tables").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(c.reports / "tables" / f"seg_{method}_per_case.csv", index=False)
    s = pd.concat(sizes)
    s["hits"] = s.sensitivity * s.n
    agg = s.groupby(["method", "bin"], sort=False).agg(n=("n", "sum"), hits=("hits", "sum")).reset_index()
    agg["sensitivity"] = agg.hits / agg.n.replace(0, np.nan)
    agg.drop(columns="hits").to_csv(c.reports / "tables" / f"seg_{method}_by_size.csv", index=False)


def _load_seg(c: Ctx, sid: str, seg_source: str = "expert"):
    import nibabel as nib
    p = c.pp(sid) / "seg.nii.gz" if seg_source == "expert" else c.cache / "seg_pred" / seg_source / f"{sid}.nii.gz"
    img = nib.load(str(p))
    return np.rint(np.asarray(img.dataobj)).astype(np.int16), tuple(float(z) for z in img.header.get_zooms()[:3])


def stage_track(c: Ctx, seg_source: str = "expert", **_):
    from .lesions.components import lesions
    from .lesions.tracking import track_patient
    tp = c.tp()
    gate = c.cfg.get("tracking", {}).get("gate_mm", 10.0)
    L, T = [], []
    for cid, g in tp[~tp.is_normal].groupby("case_id"):
        per = []
        for r in g.sort_values("post_idx").itertuples():
            seg, sp = _load_seg(c, r.study_id, seg_source)
            d = lesions(seg, sp, c.labels).assign(case_id=cid, post_idx=r.post_idx, study_id=r.study_id)
            per.append(d); L.append(d)
        T.append(track_patient(per, gate).assign(case_id=cid))
    pd.concat(L).to_csv(c.cache / "lesions.csv", index=False)
    pd.concat(T).to_csv(c.cache / "tracks.csv", index=False)


def _per_tp(c: Ctx) -> dict:
    les = pd.read_csv(c.cache / "lesions.csv")
    return {(cid, int(p)): g.reset_index(drop=True) for (cid, p), g in les.groupby(["case_id", "post_idx"])}


def stage_labels(c: Ctx, **_):
    from .lesions.labels import interval_labels, lesion_trajectories
    r = c.cfg.get("rano", {})
    iv = intervals(c.clin())
    tracks, per = pd.read_csv(c.cache / "tracks.csv"), _per_tp(c)
    lab = interval_labels(tracks, per, iv, r.get("prog_increase", 0.73), r.get("new_lesion_min_cc", 0.05))
    lab.merge(iv, on=["case_id", "t", "t_next"]).to_csv(c.cache / "labels_t2.csv", index=False)
    lesion_trajectories(tracks, per, iv, r.get("prog_increase", 0.73), r.get("pr_decrease", -0.65)).to_csv(c.cache / "traj_t3.csv", index=False)
    survival_table(c.clin()).to_csv(c.cache / "survival.csv", index=False)
    log.info("labels: %d intervals, T2a+ %d, T2b+ %d", len(lab), lab.prog_t2a.sum(), lab.newmet_t2b.sum())


def stage_features(c: Ctx, **_):
    from .features.radiomics import load_pp_study, radiomics_or_empty
    from .features.tables import add_deltas, clinical_design, lesion_summary
    clin, tp = c.clin(), c.tp()
    iv = intervals(clin)
    X = clinical_design(clin, iv).join(lesion_summary(_per_tp(c), pd.read_csv(c.cache / "tracks.csv"), clin, iv))
    backend = c.cfg.get("radiomics_backend", "pyradiomics")
    sid = tp[~tp.is_normal].set_index(["case_id", "post_idx"]).study_id
    rad = {}
    for (cid, t) in X.index:                         # radiomics at t; deltas vs t-1 (t-1 is always an earlier interval start)
        vols, seg = load_pp_study(c.pp(sid[(cid, t)]), ("t1c", "t2f", "adc"))
        rad[(cid, t)] = radiomics_or_empty(vols, seg, ("t1c", "t2f", "adc"), backend=backend, study=sid[(cid, t)])
    R = pd.DataFrame.from_dict(rad, orient="index"); R.index = pd.MultiIndex.from_tuples(R.index, names=["case_id", "t"])
    R.columns = [f"rad_{k}" for k in R.columns]
    X.join(add_deltas(R)).to_csv(c.cache / "features_t2.csv")


def stage_embed(c: Ctx, encoder: str | None = None, device: str = "cpu", **_):
    from .data.mri_index import load_study
    from .fm.embed import embed_study, save_embeddings
    from .fm.registry import ENCODERS, load_encoder
    enc = encoder or c.train_cfg("encoder", c.cfg.get("fm", {}).get("default", "brainsegfounder"))
    wdir = Path(c.cfg.get("fm", {}).get("weights_dir", "cache/weights"))
    wfile = c.cfg.get("fm", {}).get("weights", {}).get(enc)
    model = load_encoder(enc, (wdir / wfile) if wfile else None)
    tp = c.tp()
    pidx = pd.DataFrame([(s, cid, q, str(c.pp(s) / f"{q}.nii.gz"), (c.pp(s) / f"{q}.nii.gz").exists())
                         for s, cid in zip(tp.study_id, tp.case_id) for q in SEQS_CORE + ("seg",)],
                        columns=["study_id", "case_id", "seq", "path", "exists"])
    for sid in tp[~tp.is_normal].study_id:
        out = c.cache / "emb" / enc / f"{sid}.npz"
        if out.exists():
            continue
        vol, _, _ = load_study(pidx, sid, SEQS_CORE)
        seg, _ = _load_seg(c, sid)
        save_embeddings(out, embed_study(ENCODERS[enc], model, vol, seg, device=device))


# ---------------------------------------------------------------- modelling
P0_COLS = ["age", "sex_male", "resection", "rt_between", "dt_years", "zero_interval", "n_prior_tp", "years_since_dx",
           "n_lesions", "log_sum_et", "d_log_sum_et", "max_diam", "new_per_year", "n_new_so_far"]


def _emb_table(c: Ctx, enc: str) -> pd.DataFrame:
    """(case_id, post_idx) -> [brain ⊕ mean-lesion] embedding."""
    from .fm.embed import load_embeddings
    tp = c.tp(); tp = tp[~tp.is_normal]
    rows = {}
    for r in tp.itertuples():
        p = c.cache / "emb" / enc / f"{r.study_id}.npz"
        if not p.exists():
            continue
        e = load_embeddings(p)
        les = np.mean(list(e["lesions"].values()), 0) if e["lesions"] else np.zeros_like(e["brain"])
        rows[(r.case_id, r.post_idx)] = np.concatenate([e["brain"], les])
    return rows


def _design(c: Ctx, model: str, task: str):
    lab = pd.read_csv(c.cache / "labels_t2.csv").set_index(["case_id", "t"])
    F = pd.read_csv(c.cache / "features_t2.csv").set_index(["case_id", "t"])
    ycol = {"T2a": "prog_t2a", "T2b": "newmet_t2b"}[task]
    y = lab.loc[F.index, ycol]
    prim = [k for k in F.columns if k.startswith("primary_")]
    base = F[P0_COLS + prim]
    if model == "P0":
        return base, y
    if model == "P1":
        return F, y
    enc = c.train_cfg("encoder", "brainsegfounder")
    E = _emb_table(c, enc)
    if model == "P2":
        keep = [k for k in F.index if k in E]
        Em = pd.DataFrame([E[k] for k in keep], index=pd.MultiIndex.from_tuples(keep, names=["case_id", "t"])).add_prefix("emb_")
        return base.loc[keep].join(Em), y.loc[keep]
    if model == "P3":
        clin = c.clin().set_index("case_id")
        rows, keep = [], []
        for (cid, t) in F.index:
            if not all((cid, k) in E for k in range(1, t + 1)):
                continue
            ages = np.array([clin.loc[cid, f"age_t{k}"] for k in range(1, t + 1)], np.float32)
            rows.append({"seq": np.stack([E[(cid, k)] for k in range(1, t + 1)]).astype(np.float32),
                         "dt": ages - ages[0], **base.loc[(cid, t)].to_dict()})
            keep.append((cid, t))
        return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keep, names=["case_id", "t"])), y.loc[keep]
    raise KeyError(model)


def stage_train(c: Ctx, model: str = "P0", task: str = "T2a", device: str = "cpu", **_):
    from .models.cv import run_cv
    sp = c.splits()
    (c.cache / "oof").mkdir(exist_ok=True)
    out = c.cache / "oof" / f"{task}_{model}.csv"
    if task in ("T2a", "T2b"):
        if model == "P5":
            from .models.fusion import stack_oof
            parts = [pd.read_csv(c.cache / "oof" / f"{task}_{m}.csv") for m in ("P0", "P1", "P3") if (c.cache / "oof" / f"{task}_{m}.csv").exists()]
            stack_oof(parts).to_csv(out, index=False); return
        X, y = _design(c, model, task)
        kw = dict(epochs=c.train_cfg("epochs", 60), device=device) if model == "P3" else {}
        run_cv(model, X, y, sp, "binary", **kw).to_csv(out, index=False)
    elif task == "T3":
        tr = pd.read_csv(c.cache / "traj_t3.csv")
        F = pd.read_csv(c.cache / "features_t2.csv").set_index(["case_id", "t"])
        X = tr.set_index(["case_id", "t"]).join(F[P0_COLS], how="left")
        X["log_et_t"] = np.log(X.et_t + 1e-3)
        X = X.reset_index().set_index(["case_id", "t"])
        cols = ["log_et_t"] + P0_COLS
        oof = run_cv(model, X[cols], X["log_ratio"], sp, "regression")
        oof.to_csv(out, index=False)
    elif task == "T4":
        from .models.survival import cox_cv, deepsurv_cv
        s = pd.read_csv(c.cache / "survival.csv").set_index("case_id")
        F = pd.read_csv(c.cache / "features_t2.csv").set_index(["case_id", "t"])
        X = F.xs(1, level="t")[["age", "sex_male", "resection", "n_lesions", "log_sum_et", "max_diam"] +
                                [k for k in F.columns if k.startswith("primary_") and F[k].sum() >= 5]]
        X = X.loc[X.index.intersection(s.index)]
        sp2 = {k: sp[k] for k in X.index}
        cox_cv(X, s, sp2).to_csv(c.cache / "oof" / "T4_O1cox.csv")
        deepsurv_cv(X, s, sp2, epochs=c.train_cfg("surv_epochs", 300)).to_csv(c.cache / "oof" / "T4_O1deepsurv.csv")
    else:
        raise KeyError(task)


# ---------------------------------------------------------------- evaluation & reporting
def stage_evaluate(c: Ctx, **_):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score
    from .eval.stats import auroc_ci, calibration, cindex_ci, delta_auc, holm, subgroup_table, td_auc
    nb = c.cfg.get("bootstrap", {}).get("n", 2000)
    res = {"git": _git_hash()}
    meta = load_mri_meta(c.cfg["paths"]["mri_meta"])
    tp = c.tp()
    fs = tp.merge(meta[["study_id", "field_strength"]], on="study_id")[["case_id", "post_idx", "field_strength"]].rename(columns={"post_idx": "t"})
    clin = c.clin()
    lab = pd.read_csv(c.cache / "labels_t2.csv")
    sub = lab[["case_id", "t", "rt_between"]].merge(clin[["case_id", "primary", "sex"]], on="case_id").merge(fs, on=["case_id", "t"], how="left")
    (c.reports / "tables").mkdir(exist_ok=True)
    for task in ("T2a", "T2b"):
        oofs = {p.stem.split("_", 1)[1]: pd.read_csv(p) for p in sorted((c.cache / "oof").glob(f"{task}_P*.csv"))}
        if not oofs:
            continue
        r = {"n_intervals": int(len(next(iter(oofs.values())))), "n_events": int(next(iter(oofs.values())).y.sum()), "models": {}}
        for m, o in oofs.items():
            if o.y.nunique() < 2:
                continue
            est, lo, hi = auroc_ci(o.y, o.pred, o.case_id, nb)
            cal = calibration(o.y, o.pred)
            r["models"][m] = {"auroc": est, "auroc_ci": [lo, hi], "auprc": float(average_precision_score(o.y, o.pred)),
                              "ece": cal["ece"], "cal_slope": cal["slope"], "cal_intercept": cal["intercept"], "n": int(len(o))}
        if "P0" in oofs:
            ref = oofs["P0"].set_index(["case_id", "t"])
            deltas = []
            for m, o in oofs.items():
                if m == "P0":
                    continue
                j = o.set_index(["case_id", "t"]).join(ref[["pred"]], rsuffix="_ref", how="inner")
                d, lo, hi, p = delta_auc(j.y, j.pred, j.pred_ref, j.index.get_level_values(0), nb)
                deltas.append({"model": m, "delta": d, "lo": lo, "hi": hi, "p": p})
            if deltas:
                D = pd.DataFrame(deltas); D["p_holm"] = holm(D.p)
                D.to_csv(c.reports / "tables" / f"{task}_delta_vs_P0.csv", index=False)
                r["delta_vs_P0"] = D.to_dict("records")
        best = max(r["models"], key=lambda m: r["models"][m]["auroc"]) if r["models"] else None
        if best:
            st = subgroup_table(oofs[best], sub, by=("primary", "sex", "rt_between", "field_strength"), n_boot=nb)
            st = st[(st.n >= 10)]
            st.to_csv(c.reports / "tables" / f"{task}_subgroups_{best}.csv", index=False)
            r["best_model"] = best
        res[task] = r
    p3 = c.cache / "oof" / "T3_P1.csv"
    if p3.exists():
        o = pd.read_csv(p3)
        res["T3"] = {"P1": {"mae": float(np.mean(np.abs(o.y - o.pred))), "spearman": float(spearmanr(o.y, o.pred).statistic), "n": int(len(o))}}
    s = pd.read_csv(c.cache / "survival.csv").set_index("case_id")
    for name in ("O1cox", "O1deepsurv"):
        p = c.cache / "oof" / f"T4_{name}.csv"
        if p.exists():
            o = pd.read_csv(p, index_col=0)
            ss = s.loc[o.index]
            est, lo, hi = cindex_ci(ss.time_years, ss.event, o.risk, o.index, nb)
            res.setdefault("T4", {})[name] = {"c_index": est, "c_index_ci": [lo, hi], "td_auc": {str(k): v for k, v in td_auc(ss.time_years, ss.event, o.risk).items()}}
    (c.cache / "results_raw.json").write_text(json.dumps(res, indent=2, default=float))


def stage_explain(c: Ctx, task: str = "T2a", **_):
    """SHAP for P1 refit on all development data (for explanation only; performance comes from OOF)."""
    from .explain.shap_tab import shap_values
    from .models.cv import fit_full
    X, y = _design(c, "P1", task)
    m = fit_full("P1", X, y)
    vals, base, Xs = shap_values(m, X)
    (c.cache / "explain").mkdir(exist_ok=True)
    np.savez_compressed(c.cache / "explain" / f"shap_{task}_P1.npz", vals=vals, base=base, cols=np.array(X.columns), X=X.to_numpy(float))
    try:
        import shap
        inter = shap.TreeExplainer(m).shap_interaction_values(X.iloc[:500])
        inter = np.asarray(inter[1] if isinstance(inter, list) else inter)
        if inter.ndim == 4:
            inter = inter[..., 1]
        np.save(c.cache / "explain" / f"shap_inter_{task}_P1.npy", np.abs(inter).mean(0))
    except Exception as e:  # noqa: BLE001
        log.warning("interaction values skipped: %s", e)
    imp = pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values(ascending=False)
    imp.head(30).to_csv(c.reports / "tables" / f"shap_top_{task}_P1.csv", header=["mean_abs_shap"])


def stage_figures(c: Ctx, **_):
    from .viz import data_figs as D, pred_figs as P, seg_figs as S, surv_figs as V, traj_figs as T
    figs = c.figs
    clin, meta = c.clin(), load_mri_meta(c.cfg["paths"]["mri_meta"])
    iv, surv = intervals(clin), survival_table(clin)
    jobs = [
        ("cohort_flow", lambda: D.fig_cohort_flow(clin, iv, figs / "cohort_flow")),
        ("swimmer", lambda: D.fig_swimmer(clin, figs / "swimmer")),
        ("interval_hist", lambda: D.fig_interval_hist(iv, figs / "interval_hist")),
        ("rt_by_interval", lambda: D.fig_rt_by_interval(iv, figs / "rt_by_interval")),
        ("primary_bars", lambda: D.fig_primary_bars(clin, figs / "primary_bars")),
        ("scanner_mosaic", lambda: D.fig_scanner_mosaic(meta, figs / "scanner_mosaic")),
        ("sequence_availability", lambda: D.fig_sequence_availability(meta, figs / "sequence_availability")),
        ("km_by_primary", lambda: D.fig_km_by_primary(clin, surv, figs / "km_by_primary")),
    ]
    les = pd.read_csv(c.cache / "lesions.csv")
    tracks = pd.read_csv(c.cache / "tracks.csv")
    jobs += [("lesion_count_volume", lambda: S.fig_lesion_count_volume(les, figs / "lesion_count_volume")),
             ("tracking_flow", lambda: S.fig_tracking_flow(tracks, figs / "tracking_flow"))]
    tp = c.tp(); ex = tp[~tp.is_normal].study_id.iloc[0]
    from .features.radiomics import load_pp_study
    vols, seg = load_pp_study(c.pp(ex), ("t1c", "t1n", "t2f", "t2w"))
    jobs.append(("montage", lambda: S.fig_montage(vols, seg, figs / "montage", title=f"{ex} (preprocessed)")))
    for task in ("T2a", "T2b"):
        oofs = {p.stem.split("_", 1)[1]: pd.read_csv(p) for p in sorted((c.cache / "oof").glob(f"{task}_P*.csv"))}
        oofs = {k: v for k, v in oofs.items() if v.y.nunique() == 2}
        if not oofs:
            continue
        jobs += [(f"{task}_roc", lambda o=oofs, t=task: P.fig_roc(o, figs / f"{t}_roc", title=f"{t}: ROC")),
                 (f"{task}_pr", lambda o=oofs, t=task: P.fig_pr(o, figs / f"{t}_pr", title=f"{t}: precision–recall")),
                 (f"{task}_calibration", lambda o=oofs, t=task: P.fig_calibration(o, figs / f"{t}_calibration")),
                 (f"{task}_decision_curve", lambda o=oofs, t=task: P.fig_decision_curve(o, figs / f"{t}_decision_curve"))]
        best = max(oofs, key=lambda m: __import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(oofs[m].y, oofs[m].pred))
        jobs.append((f"{task}_risk_distribution", lambda o=oofs[best], t=task, b=best: P.fig_risk_distribution(o, figs / f"{t}_risk_distribution", model=b)))
        dp = c.reports / "tables" / f"{task}_delta_vs_P0.csv"
        if dp.exists():
            jobs.append((f"{task}_delta_forest", lambda d=pd.read_csv(dp), t=task: P.fig_delta_forest(d, figs / f"{t}_delta_forest")))
        for sp in sorted((c.reports / "tables").glob(f"{task}_subgroups_*.csv")):
            jobs.append((f"{task}_subgroups", lambda d=pd.read_csv(sp), t=task: P.fig_subgroup_forest(d, figs / f"{t}_subgroup_forest")))
    for sh in sorted((c.cache / "explain").glob("shap_T2*_P1.npz")):
        z = np.load(sh, allow_pickle=True); Xs = pd.DataFrame(z["X"], columns=list(z["cols"])); t = sh.stem.split("_")[1]
        jobs += [(f"{t}_shap_beeswarm", lambda v=z["vals"], X=Xs, t=t: P.fig_shap_beeswarm(v, X, figs / f"{t}_shap_beeswarm", title=f"{t}: SHAP (P1, log-odds)")),
                 (f"{t}_shap_bar", lambda v=z["vals"], X=Xs, t=t: P.fig_shap_bar(v, X, figs / f"{t}_shap_bar")),
                 (f"{t}_shap_dependence", lambda v=z["vals"], X=Xs, t=t: P.fig_shap_dependence(v, X, figs / f"{t}_shap_dependence"))]
        ip = c.cache / "explain" / f"shap_inter_{t}_P1.npy"
        if ip.exists():
            M = np.load(ip); top = np.argsort(np.abs(z["vals"]).mean(0))[::-1][:8]
            jobs.append((f"{t}_shap_interaction", lambda M=M, top=top, X=Xs, t=t: P.fig_shap_interaction(
                pd.DataFrame(M[np.ix_(top, top)], index=X.columns[top], columns=X.columns[top]), figs / f"{t}_shap_interaction")))
    enc = c.train_cfg("encoder", "brainsegfounder"); E = _emb_table(c, enc)
    if len(E) >= 10:
        keys = list(E); prim = clin.set_index("case_id").primary.fillna("other")
        jobs.append(("embedding_umap", lambda: P.fig_embedding_umap(np.stack([E[k] for k in keys]), pd.Series([prim[k[0]] for k in keys]),
                                                                    figs / "embedding_umap", title=f"{enc} embeddings by primary")))
    trp = c.cache / "traj_t3.csv"
    if trp.exists():
        tr = pd.read_csv(trp).merge(clin[["case_id", "primary"]], on="case_id")
        if len(tr):
            jobs.append(("rano_class_bars", lambda: T.fig_rano_class_bars(tr, figs / "rano_class_bars")))
            ages = clin.set_index("case_id")
            les2 = les.merge(tracks[["case_id", "post_idx", "lesion_id", "track_id"]].dropna(), on=["case_id", "post_idx", "lesion_id"])
            les2["years"] = [ages.loc[cc, f"age_t{p}"] - ages.loc[cc, "age_t1"] for cc, p in zip(les2.case_id, les2.post_idx)]
            les2 = les2.merge(clin[["case_id", "primary"]], on="case_id")
            jobs.append(("spaghetti", lambda: T.fig_spaghetti(les2, figs / "spaghetti")))
            best = tr.groupby(["case_id", "track_id"]).apply(lambda g: (g.et_next / g.et_t.iloc[0] - 1).min(), include_groups=False)
            jobs.append(("waterfall", lambda: T.fig_waterfall(best, figs / "waterfall")))
    t3 = c.cache / "oof" / "T3_P1.csv"
    if t3.exists():
        o3 = pd.read_csv(t3); jobs.append(("T3_pred_vs_obs", lambda o=o3: T.fig_pred_vs_obs(o.y, o.pred, figs / "T3_pred_vs_obs")))
    cox = c.cache / "oof" / "T4_O1cox.csv"
    if cox.exists():
        o4 = pd.read_csv(cox, index_col=0); s4 = surv.set_index("case_id").loc[o4.index]
        jobs.append(("T4_km_risk_tertiles", lambda s=s4, o=o4: V.fig_km_risk_tertiles(s, o.risk, figs / "T4_km_risk_tertiles")))
        rr = json.loads((c.cache / "results_raw.json").read_text()) if (c.cache / "results_raw.json").exists() else {}
        td = {k: {float(h): v for h, v in d["td_auc"].items()} for k, d in rr.get("T4", {}).items()}
        if td:
            jobs.append(("T4_td_auc", lambda: V.fig_td_auc(td, figs / "T4_td_auc")))
    ok, fail = 0, []
    for name, job in jobs:
        try:
            job(); ok += 1
        except Exception as e:  # noqa: BLE001  - one bad figure must not stop the rest
            fail.append(f"{name}: {e}")
    (c.reports / "figures_log.txt").write_text(f"rendered {ok}\n" + "\n".join(fail))
    log.info("figures: %d rendered, %d failed", ok, len(fail))


def stage_report(c: Ctx, **_):
    """Aggregate-only results for the manuscript (safe to commit)."""
    raw = json.loads((c.cache / "results_raw.json").read_text())
    audit = json.loads((c.reports / "audit.json").read_text()) if (c.reports / "audit.json").exists() else {}
    out = {"git": raw.get("git"), "cohort": {k: audit.get(k) for k in ("n_patients", "n_timepoints", "n_intervals", "n_deaths", "n_rt_intervals")}}
    out.update({k: v for k, v in raw.items() if k != "git"})
    (c.reports / "results.json").write_text(json.dumps(out, indent=2, default=float))
    rows = [(t, m, d.get("auroc"), *(d.get("auroc_ci") or [None, None]), d.get("auprc"), d.get("ece"), d.get("cal_slope"))
            for t in ("T2a", "T2b") for m, d in raw.get(t, {}).get("models", {}).items()]
    pd.DataFrame(rows, columns=["task", "model", "auroc", "lo", "hi", "auprc", "ece", "cal_slope"]).to_csv(c.reports / "tables" / "T2_summary.csv", index=False)


STAGES = {
    "audit": stage_audit, "index": stage_index, "splits": stage_splits, "preprocess": stage_preprocess,
    "seg-export": stage_seg_export, "seg-eval": stage_seg_eval, "track": stage_track, "labels": stage_labels,
    "features": stage_features, "embed": stage_embed, "train": stage_train, "evaluate": stage_evaluate,
    "explain": stage_explain, "figures": stage_figures, "report": stage_report,
}


# ---------------------------------------------------------------- locked external validation
def _lock_dir(c: Ctx) -> Path:
    if (c.cfg.get("subset") or {}).get("n_patients"):
        return c.cache / "locked"
    return Path(c.cfg.get("lock_dir", c.cache / "locked"))


def stage_lock(c: Ctx, task: str = "T2a", **_):
    """Fit P0/P1 on ALL development intervals and freeze them (hash recorded) before any external run."""
    import hashlib
    import time

    import joblib
    from .models.cv import fit_full
    d = _lock_dir(c); d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    models = {}
    for m in ("P0", "P1"):
        X, y = _design(c, m, task)
        f = d / f"{task}_{m}.joblib"
        joblib.dump({"model": fit_full(m, X, y), "columns": list(X.columns)}, f)
        h.update(f.read_bytes()); models[m] = f.name
    (d / "lock.json").write_text(json.dumps({"task": task, "models": models, "sha256": h.hexdigest(), "git": _git_hash(),
                                             "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))


def stage_external(c: Ctx, task: str = "T2a", **_):
    """Single, locked evaluation on an external cohort prepared with the same stages (audit..features)."""
    import joblib
    from sklearn.metrics import average_precision_score
    from .eval.stats import auroc_ci, calibration
    out = c.reports / f"external_{task}.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the external set is evaluated once (delete deliberately to re-run)")
    lock = json.loads((_lock_dir(c) / "lock.json").read_text())
    lab = pd.read_csv(c.cache / "labels_t2.csv").set_index(["case_id", "t"])
    F = pd.read_csv(c.cache / "features_t2.csv").set_index(["case_id", "t"])
    y = lab.loc[F.index, {"T2a": "prog_t2a", "T2b": "newmet_t2b"}[task]]
    res = {"lock_sha256": lock["sha256"], "n": int(len(y)), "n_events": int(y.sum()), "models": {}}
    for m, fn in lock["models"].items():
        b = joblib.load(_lock_dir(c) / fn)
        X = F.reindex(columns=b["columns"])
        p = b["model"].predict_proba(X)[:, 1]
        if y.nunique() == 2:
            est, lo, hi = auroc_ci(y, p, F.index.get_level_values(0), c.cfg.get("bootstrap", {}).get("n", 2000))
            cal = calibration(y, p)
            res["models"][m] = {"auroc": est, "auroc_ci": [lo, hi], "auprc": float(average_precision_score(y, p)),
                                "ece": cal["ece"], "cal_slope": cal["slope"]}
        else:
            res["models"][m] = {"auroc": None, "note": "single outcome class in external set"}
    out.write_text(json.dumps(res, indent=2, default=float))


def to_layout(map_csv: str, out_dir: str) -> Path:
    """Arrange an external cohort into the TCIA layout <out>/<study_id>/<study_id>-<seq>.nii.gz via symlinks.
    map_csv columns: study_id (CASE_YY), seq (t1c|t1n|t2f|t2|seg|...), path."""
    m = pd.read_csv(map_csv)
    out = Path(out_dir)
    for r in m.itertuples(index=False):
        d = out / r.study_id; d.mkdir(parents=True, exist_ok=True)
        dst = d / f"{r.study_id}-{r.seq}.nii.gz"
        if not dst.exists():
            dst.symlink_to(Path(r.path).resolve())
    return out


STAGES.update({"lock": stage_lock, "external": stage_external})
