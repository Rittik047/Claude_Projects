"""Imaging and segmentation (T1) figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from .style import CAT, INK, INK2, MODEL_COLORS, bar_kw, fig, save

SEG_CMAP = ListedColormap([(0, 0, 0, 0), CAT[3], CAT[0], CAT[7], CAT[6]])   # NETC, SNFH, ET, RC


def fig_montage(vols: dict, seg: np.ndarray, out, z: int | None = None, title="Example time point"):
    z = int(np.argmax((seg > 0).sum((0, 1)))) if z is None else z
    seqs = [s for s in ("t1c", "t1n", "t2f", "t2w") if s in vols]
    f, axs = fig(2.0 * (len(seqs) + 1), 2.4, ncols=len(seqs) + 1)
    for ax, s in zip(axs, seqs):
        ax.imshow(vols[s][:, :, z].T, cmap="gray", origin="lower"); ax.set_title(s); ax.axis("off")
    axs[-1].imshow(vols[seqs[0]][:, :, z].T, cmap="gray", origin="lower")
    axs[-1].imshow(seg[:, :, z].T, cmap=SEG_CMAP, vmin=0, vmax=4, origin="lower", alpha=0.6, interpolation="nearest")
    axs[-1].set_title("labels: NETC/SNFH/ET"); axs[-1].axis("off")
    f.suptitle(title, fontsize=10, fontweight="bold")
    return save(f, out)


def fig_lesion_count_volume(lesion_tab: pd.DataFrame, out):
    """lesion_tab: one row per (study, lesion) with et_cc; plus study-level counts."""
    cnt = lesion_tab.groupby("study_id").size()
    f, (a, b) = fig(8.0, 3.0, ncols=2)
    a.hist(cnt, bins=np.arange(0.5, cnt.max() + 1.5), color=CAT[0], **bar_kw()); a.set_xlabel("lesions per time point"); a.set_ylabel("time points")
    b.hist(np.log10(lesion_tab.et_cc.clip(lower=1e-3)), bins=30, color=CAT[0], **bar_kw())
    b.set_xlabel("log10 enhancing volume per lesion (cc)"); b.set_ylabel("lesions")
    a.set_title("Lesion count"); b.set_title("Lesion size")
    return save(f, out)


def fig_seg_dice_by_size(size_tab: pd.DataFrame, out):
    """size_tab: method, bin, n, sensitivity."""
    bins = list(dict.fromkeys(size_tab.bin)); methods = list(dict.fromkeys(size_tab.method))
    f, ax = fig(5.0, 3.2)
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        d = size_tab[size_tab.method == m].set_index("bin").reindex(bins)
        ax.bar(np.arange(len(bins)) + i * w, d.sensitivity, w, color=CAT[i], label=m, **bar_kw())
    ax.set_xticks(np.arange(len(bins)) + w * (len(methods) - 1) / 2, bins); ax.set_ylim(0, 1.05)
    ax.set_ylabel("lesion detection sensitivity"); ax.legend(); ax.set_title("Detection by lesion size")
    return save(f, out)


def fig_seg_method_compare(per_case: pd.DataFrame, out, metric="lesion_dice"):
    methods = list(dict.fromkeys(per_case.method))
    f, ax = fig(4.6, 3.2)
    bp = ax.boxplot([per_case.loc[per_case.method == m, metric] for m in methods], tick_labels=methods, patch_artist=True,
                    widths=0.5, medianprops=dict(color=INK), showfliers=False)
    for i, b in enumerate(bp["boxes"]):
        b.set(facecolor=CAT[i], alpha=0.85, edgecolor=CAT[i])
    ax.set_ylabel(metric.replace("_", " ")); ax.set_ylim(0, 1.02); ax.set_title("Segmentation: lesion-wise Dice (ET)")
    return save(f, out)


def fig_tracking_flow(tracks: pd.DataFrame, out):
    order = ["baseline", "stable", "recurred", "new", "resolved"]
    colors = {"baseline": CAT[0], "stable": "#86b6ef", "recurred": CAT[3], "new": CAT[7], "resolved": CAT[2]}
    t = pd.crosstab(tracks.post_idx, tracks.status).reindex(columns=order).fillna(0)
    f, ax = fig(4.8, 3.2)
    bottom = np.zeros(len(t))
    for s in order:
        ax.bar(t.index.astype(str), t[s], bottom=bottom, color=colors[s], label=s, **bar_kw()); bottom += t[s].to_numpy()
    ax.set_xlabel("post-diagnosis time point"); ax.set_ylabel("lesion tracks"); ax.legend(ncol=2)
    ax.set_title("Lesion tracking: status per time point")
    return save(f, out)


def fig_seg_by_field_strength(per_case: pd.DataFrame, out, metric="lesion_dice"):
    f, ax = fig(4.4, 3.2)
    for i, (fs, g) in enumerate(per_case.groupby("field_strength")):
        ax.hist(g[metric], bins=np.linspace(0, 1, 21), alpha=0.75, color=CAT[i], label=f"{fs:g} T (n={len(g)})", **bar_kw())
    ax.set_xlabel(metric.replace("_", " ")); ax.set_ylabel("time points"); ax.legend(); ax.set_title("Segmentation quality by field strength")
    return save(f, out)


def fig_fn_gallery(crops: list[tuple[np.ndarray, np.ndarray, str]], out):
    """crops: (image_2d, gt_mask_2d, caption) for missed lesions."""
    n = max(1, len(crops))
    f, axs = fig(1.8 * n, 2.2, ncols=n)
    axs = np.atleast_1d(axs)
    for ax, (im, m, cap) in zip(axs, crops):
        ax.imshow(im.T, cmap="gray", origin="lower"); ax.contour(m.T, levels=[0.5], colors=[CAT[7]], linewidths=1)
        ax.set_title(cap, fontsize=8, color=INK2); ax.axis("off")
    f.suptitle("Missed lesions (false negatives)", fontsize=10, fontweight="bold")
    return save(f, out)
