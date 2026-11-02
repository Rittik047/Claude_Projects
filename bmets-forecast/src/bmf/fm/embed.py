"""Whole-brain and per-lesion embeddings from a frozen encoder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .registry import EncoderSpec


def _adapt_channels(vol: np.ndarray, c: int) -> np.ndarray:
    if vol.shape[0] >= c:
        return vol[:c]
    return np.concatenate([vol, np.repeat(vol[:1], c - vol.shape[0], 0)], 0)


def _crop(vol: np.ndarray, center, size: int) -> np.ndarray:
    out = np.zeros((vol.shape[0], size, size, size), np.float32)
    lo = [int(round(c)) - size // 2 for c in center]
    src = tuple(slice(max(0, l), min(s, l + size)) for l, s in zip(lo, vol.shape[1:]))
    dst = tuple(slice(sl.start - l, sl.stop - l) for sl, l in zip(src, lo))
    out[(slice(None),) + dst] = vol[(slice(None),) + src]
    return out


@torch.no_grad()
def _run(model, patches: list[np.ndarray], device, bs=4) -> np.ndarray:
    outs = []
    for i in range(0, len(patches), bs):
        x = torch.from_numpy(np.stack(patches[i:i + bs])).to(device)
        outs.append(model(x).float().cpu().numpy())
    return np.concatenate(outs, 0)


def embed_study(spec: EncoderSpec, model, vol: np.ndarray, seg: np.ndarray, device="cpu",
                min_brain_frac: float = 0.1, lesion_labels=(1, 3)) -> dict:
    """brain = mean embedding of non-overlapping patches with >= min_brain_frac brain voxels;
    lesions = embedding of a patch centred on each ET∪NETC component (26-connected)."""
    model = model.to(device).eval()
    v = _adapt_channels(vol.astype(np.float32), spec.in_channels)
    p = spec.patch
    brain = np.abs(vol).sum(0) > 0
    pads = [(0, (-s) % p) for s in v.shape[1:]]
    vp = np.pad(v, [(0, 0)] + pads); bp = np.pad(brain, pads)
    patches = []
    for i in range(0, vp.shape[1], p):
        for j in range(0, vp.shape[2], p):
            for k in range(0, vp.shape[3], p):
                if bp[i:i + p, j:j + p, k:k + p].mean() >= min_brain_frac:
                    patches.append(vp[:, i:i + p, j:j + p, k:k + p])
    if not patches:
        patches = [_crop(v, [s / 2 for s in v.shape[1:]], p)]
    brain_emb = _run(model, patches, device).mean(0)
    lab, n = ndi.label(np.isin(seg, lesion_labels), np.ones((3, 3, 3), bool))
    les = {}
    if n:
        cents = ndi.center_of_mass(lab > 0, lab, range(1, n + 1))
        embs = _run(model, [_crop(v, c, p) for c in cents], device)
        les = {int(i + 1): embs[i] for i in range(n)}
    return {"brain": brain_emb, "lesions": les}


def save_embeddings(path: Path | str, e: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, brain=e["brain"], **{f"lesion_{k}": v for k, v in e["lesions"].items()})


def load_embeddings(path: Path | str) -> dict:
    z = np.load(path)
    return {"brain": z["brain"], "lesions": {int(k.split("_")[1]): z[k] for k in z.files if k.startswith("lesion_")}}
