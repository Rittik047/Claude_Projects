"""Lead occlusion: AUROC drop when one lead is zeroed (light, descriptive explainability)."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ecgtrust import CLASSES, LEADS  # noqa: E402
from ecgtrust.evaluate.metrics import per_class_auroc  # noqa: E402


def lead_occlusion(
    predict_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    base = per_class_auroc(y, predict_fn(x))
    rows = []
    for i, lead in enumerate(LEADS):
        xo = np.array(x, copy=True)
        xo[:, i] = 0.0
        auc = per_class_auroc(y, predict_fn(xo))
        rows += [
            {"lead": lead, "class": c, "auroc_drop": float(base[k] - auc[k])}
            for k, c in enumerate(CLASSES)
        ]
    return pd.DataFrame(rows)


def occlusion_heatmap(df: pd.DataFrame, path) -> None:
    m = df.pivot(index="class", columns="lead", values="auroc_drop").loc[list(CLASSES), list(LEADS)]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(m.to_numpy(), cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(12), LEADS)
    ax.set_yticks(range(5), CLASSES)
    for (r, c), v in np.ndenumerate(m.to_numpy()):
        ax.text(
            c,
            r,
            f"{v:.3f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="white" if v > m.to_numpy().max() * 0.6 else "black",
        )
    fig.colorbar(im, ax=ax, label="AUROC drop")
    ax.set_title("Lead occlusion: AUROC drop when a lead is zeroed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analysis(cfg, split: str, ctx: dict) -> dict | None:
    if split not in ("val_select", "test"):
        return None
    from ecgtrust import predict
    from ecgtrust.models.ensemble import checkpoint_paths, ensemble_proba
    from ecgtrust.robustness.run import _split_signals

    ckpts = checkpoint_paths(cfg)
    df = lead_occlusion(
        lambda xb: ensemble_proba(predict.predict_logits(ckpts, xb), ctx["temps"]),
        _split_signals(cfg, ctx),
        ctx["y"],
    )
    out = ctx.get("out_dir", cfg.reports / split)
    df.to_csv(out / "occlusion.csv", index=False)
    occlusion_heatmap(df, out / "occlusion.png")
    return df.to_dict(orient="records")


ANALYSIS = ("occlusion", analysis)
