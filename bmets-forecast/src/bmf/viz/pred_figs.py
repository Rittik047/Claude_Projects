"""T2 prediction figures: discrimination, calibration, clinical utility, explainability."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from ..eval.stats import calibration, decision_curve
from .style import CAT, DIV, INK, INK2, MODEL_COLORS, MUTED, SEQ, bar_kw, fig, save


def _mc(m, i):
    return MODEL_COLORS.get(m, CAT[i % len(CAT)])


def fig_roc(oofs: dict, out, title="Next-scan progression (T2a)"):
    from sklearn.metrics import roc_auc_score
    f, ax = fig(4.2, 4.0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for i, (m, o) in enumerate(oofs.items()):
        fpr, tpr, _ = roc_curve(o.y, o.pred)
        ax.plot(fpr, tpr, color=_mc(m, i), label=f"{m}  AUROC {roc_auc_score(o.y, o.pred):.2f}")
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.legend(loc="lower right"); ax.set_title(title)
    return save(f, out)


def fig_pr(oofs: dict, out, title="Precision–recall (T2a)"):
    from sklearn.metrics import average_precision_score
    f, ax = fig(4.2, 4.0)
    prev = next(iter(oofs.values())).y.mean()
    ax.axhline(prev, color=MUTED, lw=1, ls="--")
    for i, (m, o) in enumerate(oofs.items()):
        p, r, _ = precision_recall_curve(o.y, o.pred)
        ax.plot(r, p, color=_mc(m, i), label=f"{m}  AP {average_precision_score(o.y, o.pred):.2f}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.legend(); ax.set_title(title)
    return save(f, out)


def fig_delta_forest(delta_tab: pd.DataFrame, out, ref="P0"):
    """delta_tab: model, delta, lo, hi, p_holm."""
    d = delta_tab.reset_index(drop=True)
    f, ax = fig(4.8, 0.5 * len(d) + 1.4)
    ax.axvline(0, color=MUTED, lw=1)
    for i, r in d.iterrows():
        ax.plot([r.lo, r.hi], [i, i], color=_mc(r.model, i), lw=2, solid_capstyle="round")
        ax.scatter(r.delta, i, color=_mc(r.model, i), s=40, zorder=3, edgecolor="#fcfcfb", linewidth=2)
        ax.text(max(d.hi.max(), 0) + 0.01, i, f"{r.delta:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  p={r.p_holm:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(d)), d.model); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"ΔAUROC vs {ref} (95% CI, patient bootstrap; Holm)"); ax.set_title("Incremental discrimination")
    return save(f, out)


def fig_calibration(oofs: dict, out, n_bins=8):
    f, ax = fig(4.2, 4.0)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for i, (m, o) in enumerate(oofs.items()):
        c = calibration(o.y, o.pred, n_bins)
        t = c["table"]
        ax.plot(t.mean_pred, t.obs, marker="o", ms=4, color=_mc(m, i), label=f"{m}  slope {c['slope']:.2f}, ECE {c['ece']:.3f}")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency"); ax.legend(loc="upper left"); ax.set_title("Calibration")
    return save(f, out)


def fig_decision_curve(oofs: dict, out, thresholds=np.linspace(0.05, 0.6, 23)):
    f, ax = fig(4.8, 3.4)
    first = next(iter(oofs.values()))
    dc0 = decision_curve(first.y, first.pred, thresholds)
    ax.plot(dc0.threshold, dc0.nb_all, color=MUTED, lw=1, ls="--", label="treat all")
    ax.plot(dc0.threshold, dc0.nb_none, color=INK2, lw=1, label="treat none")
    for i, (m, o) in enumerate(oofs.items()):
        dc = decision_curve(o.y, o.pred, thresholds)
        ax.plot(dc.threshold, dc.nb_model, color=_mc(m, i), label=m)
    ax.set_ylim(min(-0.05, dc0.nb_all.min() * 0 - 0.05), max(0.05, first.y.mean() * 1.1))
    ax.set_xlabel("threshold probability (earlier MRI / escalation)"); ax.set_ylabel("net benefit"); ax.legend(ncol=2)
    ax.set_title("Decision curve")
    return save(f, out)


def fig_shap_beeswarm(vals: np.ndarray, X: pd.DataFrame, out, top=12, title="SHAP (log-odds)"):
    imp = np.abs(vals).mean(0); order = np.argsort(imp)[::-1][:top]
    f, ax = fig(5.6, 0.32 * len(order) + 1.2)
    rng = np.random.default_rng(0)
    for row, j in enumerate(order[::-1]):
        v = vals[:, j]; x = X.iloc[:, j].to_numpy(float)
        lo, hi = np.nanpercentile(x, [5, 95]); c = np.clip((x - lo) / (hi - lo + 1e-12), 0, 1)
        ax.scatter(v, row + rng.uniform(-0.3, 0.3, len(v)), c=c, cmap=DIV, s=6, lw=0, vmin=0, vmax=1)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(order)), [X.columns[j] for j in order[::-1]]); ax.grid(axis="y", visible=False)
    sm = __import__("matplotlib").cm.ScalarMappable(cmap=DIV); sm.set_array([])
    cb = f.colorbar(sm, ax=ax, ticks=[0, 1], aspect=30); cb.ax.set_yticklabels(["low", "high"]); cb.set_label("feature value")
    ax.set_xlabel("SHAP value (impact on log-odds)"); ax.set_title(title)
    return save(f, out)


def fig_shap_bar(vals: np.ndarray, X: pd.DataFrame, out, top=15):
    imp = pd.Series(np.abs(vals).mean(0), index=X.columns).sort_values().tail(top)
    f, ax = fig(5.0, 0.3 * len(imp) + 1.0)
    ax.barh(imp.index, imp.values, color=CAT[0], **bar_kw()); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP|"); ax.set_title("Global feature importance")
    return save(f, out)


def fig_shap_dependence(vals: np.ndarray, X: pd.DataFrame, out, features=None, color_by=None):
    imp = np.abs(vals).mean(0)
    features = features or [X.columns[j] for j in np.argsort(imp)[::-1][:4]]
    f, axs = fig(3.0 * len(features), 2.8, ncols=len(features))
    axs = np.atleast_1d(axs)
    for ax, ft in zip(axs, features):
        j = list(X.columns).index(ft)
        cvals = X[color_by].to_numpy(float) if color_by else None
        ax.scatter(X[ft], vals[:, j], s=7, lw=0, c=cvals if cvals is not None else CAT[0], cmap=SEQ if cvals is not None else None)
        ax.axhline(0, color=MUTED, lw=1); ax.set_xlabel(ft); ax.set_ylabel("SHAP")
    f.suptitle("SHAP dependence", fontsize=10, fontweight="bold")
    return save(f, out)


def fig_shap_interaction(inter: pd.DataFrame, out):
    """inter: square DataFrame of mean |SHAP interaction| between top features."""
    f, ax = fig(4.8, 4.2)
    im = ax.imshow(inter.to_numpy(), cmap=SEQ); ax.grid(False)
    ax.set_xticks(range(len(inter)), inter.columns, rotation=60, ha="right"); ax.set_yticks(range(len(inter)), inter.index)
    f.colorbar(im, ax=ax, label="mean |interaction|"); ax.set_title("SHAP interaction strength")
    return save(f, out)


def fig_saliency_overlay(img2d: np.ndarray, sal2d: np.ndarray, out, mask2d=None, title="Integrated gradients (t1c)"):
    f, axs = fig(5.2, 2.8, ncols=2)
    axs[0].imshow(img2d.T, cmap="gray", origin="lower"); axs[0].set_title("input"); axs[0].axis("off")
    axs[1].imshow(img2d.T, cmap="gray", origin="lower")
    axs[1].imshow(np.ma.masked_less(sal2d.T, np.percentile(sal2d, 90)), cmap=SEQ, origin="lower", alpha=0.8)
    if mask2d is not None:
        axs[1].contour(mask2d.T, levels=[0.5], colors=[CAT[7]], linewidths=0.8)
    axs[1].set_title("attribution (top decile)"); axs[1].axis("off")
    f.suptitle(title, fontsize=10, fontweight="bold")
    return save(f, out)


def fig_attention_time(att: pd.DataFrame, out):
    """att: rows = examples (or outcome groups), columns = time point -> mean attention."""
    f, ax = fig(4.6, 3.0)
    for i, (k, r) in enumerate(att.iterrows()):
        ax.plot(range(1, att.shape[1] + 1), r.values, marker="o", ms=4, color=CAT[i], label=str(k))
    ax.set_xticks(range(1, att.shape[1] + 1)); ax.set_xlabel("time point (oldest → current)"); ax.set_ylabel("attention weight")
    ax.legend(); ax.set_title("Temporal attention")
    return save(f, out)


def fig_embedding_umap(emb: np.ndarray, labels: pd.Series, out, title="Foundation-model embeddings"):
    try:
        import umap
        Z = umap.UMAP(random_state=0).fit_transform(emb); meth = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        Z = TSNE(random_state=0, perplexity=min(30, max(5, len(emb) // 4))).fit_transform(emb); meth = "t-SNE"
    cats = list(dict.fromkeys(labels.astype(str)))[:3]
    lab = labels.astype(str).where(labels.astype(str).isin(cats), "other")
    f, ax = fig(4.4, 4.0)
    for i, c in enumerate(cats + (["other"] if (lab == "other").any() else [])):
        m = (lab == c).to_numpy()
        ax.scatter(Z[m, 0], Z[m, 1], s=10, color=CAT[i] if c != "other" else MUTED, label=c, edgecolor="#fcfcfb", linewidth=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(f"{meth} 1"); ax.set_ylabel(f"{meth} 2"); ax.legend(); ax.set_title(title)
    return save(f, out)


def fig_sequence_ablation(abl: pd.DataFrame, out):
    d = abl[abl.dropped != "none"]
    f, ax = fig(4.4, 3.0)
    ax.bar(d.dropped, d.delta, color=[CAT[7] if v < 0 else CAT[0] for v in d.delta], **bar_kw())
    ax.axhline(0, color=MUTED, lw=1); ax.set_ylabel("Δ AUROC when sequence removed"); ax.set_title("Sequence ablation")
    return save(f, out)


def fig_subgroup_forest(st: pd.DataFrame, out):
    d = st.dropna(subset=["auroc"]).reset_index(drop=True)
    f, ax = fig(5.4, 0.32 * len(d) + 1.2)
    for i, r in d.iterrows():
        ax.plot([r.lo, r.hi], [i, i], color=CAT[0], lw=2, solid_capstyle="round")
        ax.scatter(r.auroc, i, color=CAT[0], s=30, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    ax.axvline(0.5, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(d)), [f"{r.variable}: {r.level} (n={r.n}, +{r.n_pos})" for r in d.itertuples()])
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel("AUROC (95% CI)"); ax.set_title("Subgroup performance")
    return save(f, out)


def fig_bootstrap_dist(boot: dict, out):
    f, ax = fig(4.6, 3.0)
    for i, (m, v) in enumerate(boot.items()):
        ax.hist(v, bins=40, alpha=0.6, color=_mc(m, i), label=m, density=True)
    ax.set_xlabel("AUROC (bootstrap replicates)"); ax.set_ylabel("density"); ax.legend(); ax.set_title("Bootstrap distributions")
    return save(f, out)


def fig_risk_distribution(oof: pd.DataFrame, out, model="model"):
    f, ax = fig(4.6, 3.0)
    for i, (lab, name) in enumerate([(0, "no event"), (1, "event")]):
        ax.hist(oof.pred[oof.y == lab], bins=np.linspace(0, 1, 26), alpha=0.7, color=CAT[0] if lab == 0 else CAT[7],
                label=f"{name} (n={int((oof.y == lab).sum())})", density=True)
    ax.set_xlabel(f"predicted risk ({model})"); ax.set_ylabel("density"); ax.legend(); ax.set_title("Risk distribution by outcome")
    return save(f, out)
