"""07_explainability (val_select only): integrated gradients, lead importance, embeddings and
12SL feature importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ecgtrust import CLASSES, LEADS
from ecgtrust.figures.data import plot_ecg
from ecgtrust.figures.style import CLASS_COLORS, INK, MUTED, plt, save


def integrated_gradients(members, x: np.ndarray, target: int, steps: int = 24) -> np.ndarray:
    """Ensemble-mean integrated gradients (Sundararajan et al. 2017) for one class logit.

    x: (N, 12, T) in mV. Baseline: each member's training lead mean (a flat line), i.e. zero
    after normalisation. Returns (N, 12, T) attributions in normalised input units.
    """
    alphas = torch.linspace(1 / steps, 1, steps)
    total = np.zeros(x.shape, dtype=np.float32)
    for model, ck in members:
        mean = np.asarray(ck["lead_mean"])[None, :, None]
        std = np.asarray(ck["lead_std"])[None, :, None]
        z = torch.from_numpy(((x - mean) / std).astype(np.float32))
        grads = torch.zeros_like(z)
        for a in alphas:
            zi = (a * z).requires_grad_(True)
            model(zi)[:, target].sum().backward()
            grads += zi.grad
        total += (z * grads / steps).numpy()
    return total / len(members)


def _members(cfg):
    from ecgtrust.models.ensemble import checkpoint_paths
    from ecgtrust.predict import load_checkpoint

    ckpts = [p for p in checkpoint_paths(cfg) if p.exists()]
    return [load_checkpoint(p) for p in ckpts]


def _smooth(a: np.ndarray, k: int = 9) -> np.ndarray:
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), -1, a)


def saliency_examples(members, x, y, p, ids, out: Path) -> list[Path]:
    files = []
    for c, name in enumerate(CLASSES):
        tp = np.flatnonzero(y[:, c] == 1)
        if not len(tp):
            continue
        i = tp[np.argmax(p[tp, c])]
        att = integrated_gradients(members, x[i : i + 1], c)[0]
        shade = _smooth(np.clip(att, 0, None))
        shade = shade / (shade.max() + 1e-9)
        fig, axes = plt.subplots(12, 1, figsize=(8, 9), sharex=True)
        plot_ecg(axes, x[i], INK, shade)
        others = ", ".join(CLASSES[j] for j in np.flatnonzero(y[i]) if j != c) or "none"
        fig.suptitle(
            f"{name}: integrated gradients on the most confident true positive\n"
            f"(val_select ecg_id {ids[i]}, p={p[i, c]:.2f}; other labels: {others}; "
            "red = evidence for the class)",
            fontweight="bold",
            fontsize=9,
        )
        files.append(save(fig, out / "saliency" / f"integrated_gradients_{name}.png"))
    return files


def lead_importance(members, x, y, p, out: Path, n: int = 40) -> Path:
    imp = np.zeros((5, 12))
    for c in range(5):
        tp = np.flatnonzero((y[:, c] == 1) & (p[:, c] > 0.5))
        tp = tp[np.argsort(-p[tp, c])][:n]
        att = np.abs(integrated_gradients(members, x[tp], c)).sum(-1)  # (n, 12)
        imp[c] = (att / att.sum(1, keepdims=True)).mean(0)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    im = ax.imshow(imp * 100, cmap="magma_r", aspect="auto")
    for i in range(5):
        for j in range(12):
            ax.text(
                j,
                i,
                f"{imp[i, j] * 100:.0f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if imp[i, j] > imp.max() * 0.6 else INK,
            )
    ax.set_xticks(range(12), LEADS)
    ax.set_yticks(range(5), CLASSES)
    fig.colorbar(im, ax=ax, label="share of |attribution| (%)")
    ax.set_title(f"Lead importance from integrated gradients (top {n} true positives per class)")
    return save(fig, out / "lead_importance_integrated_gradients.png")


@torch.no_grad()
def _embed(model, ck, x: np.ndarray, bs: int = 256) -> np.ndarray:
    mean = np.asarray(ck["lead_mean"])[None, :, None]
    std = np.asarray(ck["lead_std"])[None, :, None]
    out = []
    for s in range(0, len(x), bs):
        z = torch.from_numpy(((x[s : s + bs] - mean) / std).astype(np.float32))
        out.append(model.layers(model.stem(z)).mean(-1).numpy())
    return np.concatenate(out)


def embedding(members, x, y, out: Path) -> Path:
    from sklearn.manifold import TSNE

    emb = np.concatenate([_embed(m, ck, x) for m, ck in members], axis=1)
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
    single = np.where(y.sum(1) == 1, y.argmax(1), -1)
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.scatter(*xy[single == -1].T, s=5, color=MUTED, alpha=0.4, label="multi-label")
    for c, name in enumerate(CLASSES):
        ax.scatter(*xy[single == c].T, s=7, color=CLASS_COLORS[name], label=name, alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE of the ensemble's learned ECG representation (val_select)")
    ax.legend(frameon=False, fontsize=7, markerscale=2)
    return save(fig, out / "embedding_tsne.png")


def permutation_importance_12sl(cfg, out: Path, top: int = 12) -> Path | None:
    import joblib
    from sklearn.inspection import permutation_importance

    from ecgtrust.features.ptbxl_plus import load_features
    from ecgtrust.train_cli import load_labelled

    f = cfg.artifacts / "models" / "baseline_12sl.joblib"
    if not f.exists():
        return None
    model = joblib.load(f)
    meta, y, splits = load_labelled(cfg)
    m = (splits == "val_select").to_numpy()
    feats = load_features(cfg.dataset_dir("ptbxl_plus"), "12sl").reindex(meta.index[m])
    feats = feats[model.feature_names]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.8))
    for c, (ax, clf) in enumerate(zip(axes, model.models, strict=True)):
        r = permutation_importance(
            clf,
            feats,
            y[CLASSES[c]].to_numpy()[m].astype(int),
            scoring="roc_auc",
            n_repeats=3,
            random_state=0,
            n_jobs=1,
        )
        order = np.argsort(-r.importances_mean)[:top][::-1]
        ax.barh(
            np.array(model.feature_names)[order],
            r.importances_mean[order],
            xerr=r.importances_std[order],
            color=CLASS_COLORS[CLASSES[c]],
        )
        ax.set_title(CLASSES[c])
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel("AUROC drop")
    fig.suptitle(
        "12SL baseline: permutation importance on val_select (top measurements)", fontweight="bold"
    )
    return save(fig, out / "12sl_permutation_importance.png")


def make(cfg, out: Path) -> list[Path]:
    from ecgtrust.models.ensemble import ensemble_proba, load_preds

    files = []
    pf = cfg.artifacts / "preds" / "resnet_val_select.npz"
    members = _members(cfg)
    if members and pf.exists():
        torch.set_num_threads(4)
        d = load_preds(pf)
        from ecgtrust.train_cli import load_labelled

        meta, _, _ = load_labelled(cfg)
        rows = meta.index.get_indexer(d["ecg_id"])
        xall = np.load(cfg.artifacts / "ptbxl_x100.npy", mmap_mode="r")
        x = np.asarray(xall[np.sort(rows)])[np.argsort(np.argsort(rows))]
        y, p = d["y"], ensemble_proba(d["logits"])
        files += saliency_examples(members, x, y, p, d["ecg_id"], out)
        files.append(lead_importance(members, x, y, p, out))
        files.append(embedding(members, x, y, out))
    f = permutation_importance_12sl(cfg, out)
    return files + ([f] if f else [])
