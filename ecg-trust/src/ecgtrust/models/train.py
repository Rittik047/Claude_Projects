"""Seeded training loop; keeps the epoch with the best val_select macro-AUROC.

Runs on CUDA when available (set ECGTRUST_DEVICE=cpu to force CPU). GPU results match CPU
results closely but not bit-for-bit, because some cuDNN kernels are non-deterministic."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecgtrust.config import Config
from ecgtrust.data.signals import fit_lead_stats
from ecgtrust.evaluate.metrics import macro_auroc
from ecgtrust.models.resnet1d import ResNet1d


def device() -> torch.device:
    want = os.environ.get("ECGTRUST_DEVICE", "")
    if want:
        return torch.device(want)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(4)


def build_model(train_cfg: dict) -> ResNet1d:
    return ResNet1d(width=train_cfg["width"], blocks=tuple(train_cfg["blocks"]))


def _batches(n: int, bs: int, gen: torch.Generator | None):
    idx = torch.randperm(n, generator=gen) if gen is not None else torch.arange(n)
    for i in range(0, n, bs):
        yield idx[i : i + bs].numpy()


def _to_tensor(x: np.ndarray, idx, mean, std) -> torch.Tensor:
    order = np.argsort(idx)  # read memmap rows in ascending order, then restore batch order
    xb = np.empty((len(idx),) + x.shape[1:], dtype=np.float32)
    xb[order] = x[idx[order]]
    xb = (xb - mean[None, :, None]) / std[None, :, None]
    return torch.from_numpy(xb.astype(np.float32))


@torch.no_grad()
def predict_logits_array(model: nn.Module, x: np.ndarray, mean, std, bs: int = 256) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = [
        model(_to_tensor(x, idx, mean, std).to(dev)).cpu().numpy()
        for idx in _batches(len(x), bs, None)
    ]
    return np.concatenate(out)


def train_model(
    cfg: Config,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    out_dir: Path,
    max_epochs: int | None = None,
) -> Path:
    tc = cfg.train
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    mean, std = fit_lead_stats(np.asarray(x_tr))
    dev = device()
    model = build_model(tc).to(dev)
    epochs = max_epochs or tc["max_epochs"]
    steps = int(np.ceil(len(x_tr) / tc["batch_size"]))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=epochs * steps)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    y_tr_t = torch.from_numpy(np.asarray(y_tr, dtype=np.float32)).to(dev)
    best, best_epoch, best_state, bad = -np.inf, 0, None, 0
    log = (out_dir / "train_log.jsonl").open("w")
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for idx in _batches(len(x_tr), tc["batch_size"], gen):
            opt.zero_grad()
            xb = _to_tensor(x_tr, idx, mean, std).to(dev, non_blocking=True)
            loss = loss_fn(model(xb), y_tr_t[torch.from_numpy(idx).to(dev)])
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(idx), n + len(idx)
        val = macro_auroc(np.asarray(y_va), predict_logits_array(model, x_va, mean, std))
        log.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": total / n,
                    "val_select_macro_auroc": val,
                    "seconds": round(time.time() - t0, 1),
                }
            )
            + "\n"
        )
        log.flush()
        score = val if np.isfinite(val) else -1.0  # degenerate val split: keep training state
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    log.close()
    ckpt = out_dir / f"resnet_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": dict(tc),
            "lead_mean": mean,
            "lead_std": std,
            "seed": seed,
            "best_epoch": best_epoch,
            "val_select_macro_auroc": float(best),
        },
        ckpt,
    )
    return ckpt
