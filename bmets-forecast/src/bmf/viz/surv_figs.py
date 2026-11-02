"""T4 survival figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import CAT, INK2, MUTED, MODEL_COLORS, fig, save


def fig_km_risk_tertiles(surv: pd.DataFrame, risk: pd.Series, out, title="Overall survival by predicted-risk tertile"):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    s = surv.copy(); s["risk"] = risk.reindex(s.index)
    s["tert"] = pd.qcut(s.risk, 3, labels=["low", "mid", "high"])
    f, ax = fig(4.8, 3.6)
    for i, (lab, c) in enumerate(zip(["low", "mid", "high"], [CAT[0], CAT[3], CAT[7]])):
        d = s[s.tert == lab]
        KaplanMeierFitter().fit(d.time_years, d.event, label=f"{lab} risk (n={len(d)})").plot_survival_function(ax=ax, ci_show=True, color=c, lw=2, ci_alpha=0.12)
    p = multivariate_logrank_test(s.time_years, s.tert, s.event).p_value
    ax.text(0.98, 0.95, f"log-rank p = {p:.2g}", transform=ax.transAxes, ha="right", color=INK2, fontsize=8)
    ax.set_xlabel("years since first post-dx MRI"); ax.set_ylabel("overall survival"); ax.set_ylim(0, 1.02); ax.set_title(title)
    return save(f, out)


def fig_td_auc(td: dict, out):
    """td: model -> {horizon: auc}."""
    f, ax = fig(4.4, 3.2)
    for i, (m, d) in enumerate(td.items()):
        h = sorted(d); ax.plot(h, [d[k] for k in h], marker="o", ms=5, color=MODEL_COLORS.get(m, CAT[i]), label=m)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1); ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("horizon (years)"); ax.set_ylabel("time-dependent AUC (IPCW)"); ax.legend(); ax.set_title("Survival discrimination over time")
    return save(f, out)
