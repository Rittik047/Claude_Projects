"""Statistics for the analysis plan: patient-clustered bootstrap CIs, paired ΔAUROC, Holm,
calibration (ECE, slope, intercept), decision curves, C-index, IPCW time-dependent AUC, subgroups."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def cluster_bootstrap_indices(groups, n_boot: int = 2000, seed: int = 0):
    """Yield row indices of bootstrap samples that resample whole patients (clusters) with replacement."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([rows[g] for g in pick])


def _boot_stat(fn, y, cols, groups, n_boot, seed):
    vals = []
    for idx in cluster_bootstrap_indices(groups, n_boot, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(fn(yy, *[c[idx] for c in cols]))
    return np.asarray(vals)


def auroc_ci(y, p, groups, n_boot: int = 2000, seed: int = 0, metric: str = "auroc"):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    f = roc_auc_score if metric == "auroc" else average_precision_score
    b = _boot_stat(lambda yy, pp: f(yy, pp), y, [p], groups, n_boot, seed)
    return float(f(y, p)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def delta_auc(y, p_a, p_b, groups, n_boot: int = 2000, seed: int = 0):
    """AUROC(a) - AUROC(b) with paired cluster bootstrap CI and two-sided bootstrap p-value."""
    y = np.asarray(y).astype(int); a, b = np.asarray(p_a, float), np.asarray(p_b, float)
    d = _boot_stat(lambda yy, aa, bb: roc_auc_score(yy, aa) - roc_auc_score(yy, bb), y, [a, b], groups, n_boot, seed)
    est = roc_auc_score(y, a) - roc_auc_score(y, b)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(est), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1.0, max(p, 1 / len(d))))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(y, p, n_bins: int = 10) -> dict:
    """ECE (equal-frequency bins), calibration slope (logistic of y on logit p) and
    calibration-in-the-large intercept (logistic with logit p as offset)."""
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    lp = _logit(p)
    slope = float(LogisticRegression(C=1e6, max_iter=1000).fit(lp[:, None], y).coef_[0, 0])
    a = 0.0
    for _ in range(50):                                   # Newton for intercept with offset
        mu = 1 / (1 + np.exp(-(a + lp)))
        g, h = (y - mu).sum(), (mu * (1 - mu)).sum()
        a += g / max(h, 1e-9)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tab = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs=("y", "mean"))
    ece = float((tab.n * (tab.mean_pred - tab.obs).abs()).sum() / len(y))
    return {"ece": ece, "slope": slope, "intercept": float(a), "table": tab.reset_index()}


def decision_curve(y, p, thresholds) -> pd.DataFrame:
    y = np.asarray(y).astype(int); p = np.asarray(p, float); n = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        pos = p >= pt
        tp, fp = (pos & (y == 1)).sum(), (pos & (y == 0)).sum()
        w = pt / (1 - pt)
        rows.append((pt, tp / n - fp / n * w, prev - (1 - prev) * w, 0.0))
    return pd.DataFrame(rows, columns=["threshold", "nb_model", "nb_all", "nb_none"])


def cindex_ci(time, event, risk, groups, n_boot: int = 2000, seed: int = 0):
    from lifelines.utils import concordance_index
    time, event, risk = map(np.asarray, (time, event, risk))
    f = lambda e, t, r: concordance_index(t, -r, e)
    vals = []
    for idx in cluster_bootstrap_indices(groups, n_boot, seed):
        if event[idx].sum() == 0:
            continue
        try:
            vals.append(f(event[idx], time[idx], risk[idx]))
        except ZeroDivisionError:            # no admissible pairs in this resample
            continue
    if not vals:
        return float(f(event, time, risk)), np.nan, np.nan
    return float(f(event, time, risk)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def td_auc(time, event, risk, horizons=(0.5, 1.0)) -> dict:
    """Cumulative/dynamic AUC at each horizon with IPCW (Uno 2007): cases T<=h & event (weighted 1/G(T-)),
    controls T>h (weighted 1/G(h)); G = Kaplan-Meier of the censoring distribution."""
    from lifelines import KaplanMeierFitter
    time, event, risk = (np.asarray(x, float) for x in (time, event, risk))
    km = KaplanMeierFitter().fit(time, 1 - event)
    G = lambda t: np.clip(km.survival_function_at_times(np.maximum(t - 1e-9, 0)).to_numpy(), 1e-3, None)
    out = {}
    for h in horizons:
        case = (time <= h) & (event == 1); ctrl = time > h
        if case.sum() == 0 or ctrl.sum() == 0:
            out[h] = np.nan; continue
        wc = 1 / G(time[case]); wn = np.full(ctrl.sum(), 1 / G(np.array([h]))[0])
        rc, rn = risk[case], risk[ctrl]
        conc = (rc[:, None] > rn[None, :]) + 0.5 * (rc[:, None] == rn[None, :])
        out[h] = float((wc[:, None] * wn[None, :] * conc).sum() / (wc.sum() * wn.sum()))
    return out


def subgroup_table(oof: pd.DataFrame, meta: pd.DataFrame, by=("primary", "sex", "rt_between", "field_strength"),
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """AUROC per level of each subgroup variable. `meta` is keyed by case_id (or case_id+t if it has 't')."""
    key = ["case_id", "t"] if "t" in meta.columns else ["case_id"]
    df = oof.merge(meta, on=key, how="left")
    rows = []
    for var in by:
        if var not in df:
            continue
        for lev, g in df.groupby(var, dropna=False):
            n, npos = len(g), int(g.y.sum())
            if 0 < npos < n and n >= 10:
                est, lo, hi = auroc_ci(g.y, g.pred, g.case_id, n_boot, seed)
            else:
                est = lo = hi = np.nan
            rows.append((var, lev, n, npos, est, lo, hi))
    return pd.DataFrame(rows, columns=["variable", "level", "n", "n_pos", "auroc", "lo", "hi"])
