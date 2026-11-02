"""Cohort / data-description figures (computed from the public TSVs)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .style import CAT, GRID, INK, INK2, MUTED, PRIMARY_COLORS, PRIMARY_ORDER, SEQ, bar_kw, fig, save


def _prim(s):
    return s.where(s.isin(PRIMARY_ORDER), "other")


def fig_cohort_flow(clin, iv, out):
    f, ax = fig(5.6, 3.6); ax.axis("off"); ax.grid(False)
    n_int = len(iv)
    boxes = [
        (0.5, 0.9, f"UCSD-BMETS-Longitudinal\n{len(clin)} patients · {int(clin.n_post.sum() + clin.has_normal_mri.sum())} MRI time points"),
        (0.5, 0.62, f"{int(clin.n_post.sum())} post-diagnosis MRIs\n({int(clin.has_normal_mri.sum())} pre-diagnosis normal MRIs excluded from T2–T4)"),
        (0.25, 0.3, f"T2/T3: {n_int} intervals\n{int(iv.rt_between.sum())} with radiation"),
        (0.75, 0.3, f"T4: {len(clin)} patients\n{int(clin.age_death.notna().sum())} deaths"),
    ]
    for x, y, t in boxes:
        ax.text(x, y, t, ha="center", va="center", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.5", fc="#f0efec", ec="#c3c2b7", lw=1))
    for x0, y0, x1, y1 in [(0.5, 0.82, 0.5, 0.7), (0.5, 0.54, 0.25, 0.38), (0.5, 0.54, 0.75, 0.38)]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax.set_title("Cohort and analysis units")
    return save(f, out)


def fig_swimmer(clin, out, max_patients=300):
    c = clin.copy().head(max_patients)
    c["end"] = c[["age_t1", "age_t2", "age_t3", "age_t4"]].max(axis=1)
    c["dur"] = np.where(c.age_death.notna(), c.age_death - c.age_t1, c.end - c.age_t1)
    c = c.sort_values("dur").reset_index(drop=True)
    f, ax = fig(5.6, 6.0)
    for i, r in c.iterrows():
        ax.plot([0, r.dur], [i, i], color=GRID, lw=1.2, solid_capstyle="round", zorder=1)
        for k in range(1, 5):
            a = r[f"age_t{k}"]
            if pd.notna(a):
                rt = k > 1 and r.get(f"rt_{k - 1}{k}") is True
                ax.scatter(a - r.age_t1, i, s=6, color=CAT[1] if rt else CAT[0], zorder=2, lw=0)
        if pd.notna(r.age_death):
            ax.scatter(r.age_death - r.age_t1, i, marker="x", s=8, color=INK, lw=0.8, zorder=3)
    ax.scatter([], [], color=CAT[0], s=12, label="MRI")
    ax.scatter([], [], color=CAT[1], s=12, label="MRI after interval with radiation")
    ax.scatter([], [], marker="x", color=INK, s=12, label="death")
    ax.set_xlabel("years since first post-diagnosis MRI"); ax.set_ylabel("patients (sorted by follow-up)")
    ax.set_yticks([]); ax.legend(loc="lower right"); ax.set_title("Imaging timeline per patient (swimmer plot)")
    return save(f, out)


def fig_interval_hist(iv, out):
    f, ax = fig()
    mo = iv.dt_years * 12
    xmax = float(np.ceil(np.percentile(mo, 99) + 1.2))
    ax.hist(mo.clip(upper=xmax), bins=np.arange(-0.6, xmax + 1.2, 1.2), color=CAT[0], **bar_kw())
    ax.set_xlim(-1, xmax + 0.6)
    ax.axvline(np.median(iv.dt_years) * 12, color=INK2, lw=1, ls="--")
    ax.text(np.median(iv.dt_years) * 12 + 0.5, ax.get_ylim()[1] * 0.9, f"median {np.median(iv.dt_years) * 12:.1f} mo", color=INK2, fontsize=8)
    ax.set_xlabel("interval between consecutive MRIs (months; ages recorded to 0.1 y)"); ax.set_ylabel("intervals")
    ax.set_title(f"Scan intervals (n = {len(iv)}; {int(iv.zero_interval.sum())} recorded as 0.0 y)")
    return save(f, out)


def fig_rt_by_interval(iv, out):
    g = iv.groupby("t").rt_between.agg(["sum", "size"])
    f, ax = fig(4.2, 3.2)
    x = np.arange(len(g)); lab = [f"{t}→{t + 1}" for t in g.index]
    ax.bar(x, g["size"] - g["sum"], color="#b9b8b0", label="no radiation", **bar_kw())
    ax.bar(x, g["sum"], bottom=g["size"] - g["sum"], color=CAT[1], label="radiation", **bar_kw())
    ax.set_xticks(x, lab); ax.set_ylabel("intervals"); ax.legend(); ax.set_title("Radiation between consecutive MRIs")
    return save(f, out)


def fig_primary_bars(clin, out):
    s = _prim(clin.primary.fillna("other")).value_counts().reindex(PRIMARY_ORDER).fillna(0)
    f, ax = fig(4.8, 3.0)
    ax.barh(s.index[::-1], s.values[::-1], color=[PRIMARY_COLORS[p] for p in s.index[::-1]], **bar_kw())
    for i, v in enumerate(s.values[::-1]):
        ax.text(v + 1, i, f"{int(v)}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("patients"); ax.set_title("Primary malignancy"); ax.grid(axis="y", visible=False)
    return save(f, out)


def fig_age_by_primary(clin, out):
    c = clin.assign(p=_prim(clin.primary.fillna("other")))
    f, ax = fig(5.2, 3.2)
    data = [c.loc[c.p == p, "age_t1"].dropna() for p in PRIMARY_ORDER]
    bp = ax.boxplot(data, tick_labels=PRIMARY_ORDER, patch_artist=True, widths=0.55, medianprops=dict(color=INK))
    for b, p in zip(bp["boxes"], PRIMARY_ORDER):
        b.set(facecolor=PRIMARY_COLORS[p], alpha=0.85, edgecolor=PRIMARY_COLORS[p])
    ax.set_ylabel("age at first post-dx MRI (y)"); ax.tick_params(axis="x", rotation=30); ax.set_title("Age by primary malignancy")
    return save(f, out)


def fig_scanner_mosaic(meta, out):
    t = pd.crosstab(meta.manufacturer.fillna("?") + " · " + meta.model.fillna("?"), meta.field_strength)
    t = t.loc[t.sum(1).sort_values().index]
    f, ax = fig(5.4, 3.2)
    left = np.zeros(len(t))
    for i, fs in enumerate(sorted(t.columns)):
        ax.barh(t.index, t[fs], left=left, color=CAT[i], label=f"{fs:g} T", **bar_kw()); left += t[fs].to_numpy()
    ax.set_xlabel("MRI studies"); ax.legend(); ax.set_title("Scanner model and field strength"); ax.grid(axis="y", visible=False)
    return save(f, out)


def fig_sequence_availability(meta, out):
    seqs = ["t1c", "t1n", "t2f", "t2w", "t1postfse", "dwi", "adc", "swan", "asl"]
    tp_idx = meta.study_id.str[-2:].astype(int)
    M = pd.DataFrame({s: meta.groupby(tp_idx)[s].mean() for s in seqs}).T * 100
    f, ax = fig(5.0, 3.4)
    im = ax.imshow(M.to_numpy(), cmap=SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_yticks(range(len(seqs)), seqs); ax.set_xticks(range(M.shape[1]), [f"_{c:02d}" for c in M.columns])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="#ffffff" if v > 60 else INK)
    ax.grid(False); ax.set_xlabel("folder time-point index"); f.colorbar(im, ax=ax, label="% studies with sequence")
    ax.set_title("Sequence availability")
    return save(f, out)


def fig_km_by_primary(clin, surv, out):
    from lifelines import KaplanMeierFitter
    s = surv.merge(clin[["case_id", "primary"]], on="case_id")
    s["p"] = _prim(s.primary.fillna("other"))
    top = ["lung", "breast", "melanoma"]
    s["grp"] = s.p.where(s.p.isin(top), "other")
    f, ax = fig(5.0, 3.4)
    colors = {"lung": CAT[0], "breast": CAT[1], "melanoma": CAT[2], "other": "#898781"}
    for g in top + ["other"]:
        d = s[s.grp == g]
        if len(d) < 3:
            continue
        KaplanMeierFitter().fit(d.time_years, d.event, label=f"{g} (n={len(d)})").plot_survival_function(
            ax=ax, ci_show=False, color=colors[g], lw=2)
    ax.set_xlabel("years since first post-dx MRI"); ax.set_ylabel("overall survival"); ax.set_ylim(0, 1.02)
    ax.set_title("Overall survival by primary (censored at last MRI)")
    return save(f, out)


def fig_timepoints_per_patient(clin, out):
    s = clin.n_post.value_counts().sort_index()
    f, ax = fig(3.8, 3.0)
    ax.bar(s.index.astype(str), s.values, color=CAT[0], **bar_kw())
    for x, v in zip(range(len(s)), s.values):
        ax.text(x, v + 2, str(v), ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("post-diagnosis MRIs per patient"); ax.set_ylabel("patients"); ax.set_title("Longitudinal depth")
    return save(f, out)


def fig_followup_to_death(surv, out):
    d = surv[surv.event == 1].time_years
    f, ax = fig(4.4, 3.0)
    ax.hist(d, bins=np.arange(0, d.max() + 0.25, 0.25), color=CAT[7], **bar_kw())
    ax.axvline(d.median(), color=INK2, ls="--", lw=1)
    ax.set_xlabel("years from first post-dx MRI to death"); ax.set_ylabel("patients")
    ax.set_title(f"Deaths (n = {len(d)}; median {d.median():.1f} y)")
    return save(f, out)
