"""T3 lesion-trajectory figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import CAT, INK2, MUTED, PRIMARY_COLORS, RANO_COLORS, bar_kw, fig, save


def fig_spaghetti(traj_long: pd.DataFrame, out, facet="primary"):
    """traj_long: case_id, track_id, years, et_cc, plus facet column (e.g. primary / rt)."""
    levels = [l for l in ["lung", "breast", "melanoma"] if l in set(traj_long[facet])] or list(dict.fromkeys(traj_long[facet]))[:3]
    f, axs = fig(3.0 * len(levels), 3.0, ncols=len(levels), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, lev in zip(axs, levels):
        d = traj_long[traj_long[facet] == lev]
        for _, g in d.groupby(["case_id", "track_id"]):
            g = g.sort_values("years")
            v0 = g.et_cc.iloc[0]
            if v0 > 0:
                ax.plot(g.years, 100 * (g.et_cc / v0 - 1), color=PRIMARY_COLORS.get(lev, CAT[0]), lw=0.8, alpha=0.35)
        ax.axhline(73, color=CAT[7], lw=1, ls="--"); ax.axhline(-65, color=CAT[2], lw=1, ls="--")
        ax.set_title(f"{lev} (lesions={d.groupby(['case_id', 'track_id']).ngroups})"); ax.set_xlabel("years")
        ax.set_ylim(-110, 400)
    axs[0].set_ylabel("% change in enhancing volume")
    f.suptitle("Per-lesion trajectories (dashed: +73 % / −65 % RANO-BM volumetric)", fontsize=10, fontweight="bold")
    return save(f, out)


def fig_trajectory_clusters(curves: np.ndarray, labels: np.ndarray, out, grid=None):
    """curves: [n_lesions, n_grid] % change on a common time grid (NaN allowed); labels: cluster id."""
    grid = np.arange(curves.shape[1]) if grid is None else grid
    ks = sorted(set(labels))
    f, ax = fig(4.8, 3.4)
    for i, k in enumerate(ks[:8]):
        c = curves[labels == k]
        ax.plot(grid, np.nanmedian(c, 0), color=CAT[i], lw=2.5, label=f"cluster {k} (n={len(c)})")
        ax.fill_between(grid, np.nanpercentile(c, 25, 0), np.nanpercentile(c, 75, 0), color=CAT[i], alpha=0.15, lw=0)
    ax.axhline(0, color=MUTED, lw=1); ax.set_xlabel("years since first appearance"); ax.set_ylabel("% change (median, IQR)")
    ax.legend(); ax.set_title("Lesion growth-trajectory clusters")
    return save(f, out)


def fig_pred_vs_obs(obs, pred, out, lo=None, hi=None, title="Lesion log-volume ratio (T3)"):
    obs, pred = np.asarray(obs), np.asarray(pred)
    f, ax = fig(4.0, 4.0)
    if lo is not None:
        ax.vlines(obs, lo, hi, color=CAT[0], alpha=0.2, lw=1)
    ax.scatter(obs, pred, s=8, color=CAT[0], lw=0)
    m = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(m, m, color=MUTED, ls="--", lw=1)
    from scipy.stats import spearmanr
    ax.set_title(f"{title}\nSpearman ρ = {spearmanr(obs, pred).statistic:.2f}, MAE = {np.mean(np.abs(obs - pred)):.2f}")
    ax.set_xlabel("observed"); ax.set_ylabel("predicted")
    return save(f, out)


def fig_waterfall(best_change: pd.Series, out, title="Best volumetric response per lesion"):
    v = (best_change.sort_values(ascending=False) * 100).clip(upper=300)
    cls = np.where(v <= -99.9, "CR", np.where(v <= -65, "PR", np.where(v >= 73, "PD", "SD")))
    f, ax = fig(6.0, 3.2)
    ax.bar(np.arange(len(v)), v, color=[RANO_COLORS[c] for c in cls], width=1.0, linewidth=0)
    ax.axhline(73, color=INK2, ls="--", lw=0.8); ax.axhline(-65, color=INK2, ls="--", lw=0.8)
    for c in ["PD", "SD", "PR", "CR"]:
        ax.bar([], [], color=RANO_COLORS[c], label=f"{c} ({int((cls == c).sum())})")
    ax.set_xticks([]); ax.set_ylabel("% change from baseline (capped 300)"); ax.legend(ncol=4); ax.set_title(title)
    return save(f, out)


def fig_rano_class_bars(traj: pd.DataFrame, out, by="primary"):
    t = pd.crosstab(traj[by], traj.rano_class, normalize="index").reindex(columns=["CR", "PR", "SD", "PD"]).fillna(0) * 100
    f, ax = fig(5.0, 0.4 * len(t) + 1.4)
    left = np.zeros(len(t))
    for c in t.columns:
        ax.barh(t.index.astype(str), t[c], left=left, color=RANO_COLORS[c], label=c, **bar_kw()); left += t[c].to_numpy()
    ax.set_xlabel("% of lesion-intervals"); ax.legend(ncol=4, loc="lower right"); ax.grid(axis="y", visible=False)
    ax.set_title(f"Next-scan lesion response by {by}")
    return save(f, out)
