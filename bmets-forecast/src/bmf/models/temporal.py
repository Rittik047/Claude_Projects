"""P3/P4: transformer over the sequence of per-time-point embeddings (+ clinical token)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch import nn


def _time_enc(dt: torch.Tensor, d: int) -> torch.Tensor:
    """Sinusoidal encoding of cumulative years since the first post-diagnosis MRI (period 1 month .. 10 y)."""
    freqs = torch.exp(torch.linspace(math.log(2 * math.pi / 10.0), math.log(2 * math.pi / (1 / 12)), d // 2, device=dt.device))
    ang = dt[..., None] * freqs
    return torch.cat([ang.sin(), ang.cos()], -1)


class TemporalTransformer(nn.Module):
    def __init__(self, d_in: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, n_clin: int = 0, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(d_in, d_model)          # inputs are standardised per feature by TemporalModel
        self.clin = nn.Linear(n_clin, d_model) if n_clin else None
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.d_model = d_model

    def forward(self, seq, dt, clin=None, mask=None):
        B, T, _ = seq.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=seq.device) if mask is None else mask
        seq = torch.where(mask[..., None], seq, torch.zeros_like(seq))
        h = self.inp(seq) + _time_enc(dt, self.d_model)
        tok = self.cls.expand(B, 1, -1)
        if self.clin is not None and clin is not None:
            tok = tok + self.clin(clin)[:, None]
        h = torch.cat([tok, h], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), ~mask], 1)
        out = self.enc(h, src_key_padding_mask=pad)
        return self.head(out[:, 0]).squeeze(-1)


def _batch(X: pd.DataFrame, clin_cols, Tmax=None):
    seqs = list(X["seq"]); dts = list(X["dt"])
    Tmax = Tmax or max(len(s) for s in seqs)
    D = seqs[0].shape[1]
    S = np.zeros((len(seqs), Tmax, D), np.float32); Dt = np.zeros((len(seqs), Tmax), np.float32)
    M = np.zeros((len(seqs), Tmax), bool)
    for i, (s, d) in enumerate(zip(seqs, dts)):
        n = min(len(s), Tmax)
        S[i, :n] = s[-n:]; Dt[i, :n] = d[-n:]; M[i, :n] = True
    C = X[clin_cols].to_numpy(np.float32) if clin_cols else np.zeros((len(seqs), 0), np.float32)
    return torch.from_numpy(S), torch.from_numpy(Dt), torch.from_numpy(np.nan_to_num(C)), torch.from_numpy(M)


class TemporalModel:
    """sklearn-like wrapper. X columns: 'seq' (T x D array), 'dt' (T,), plus numeric clinical columns."""

    def __init__(self, task="binary", d_model=128, n_layers=2, n_heads=4, epochs=60, lr=3e-4, wd=1e-2, seed=2026,
                 device="cpu", encoder: nn.Module | None = None):
        self.kw = dict(d_model=d_model, n_layers=n_layers, n_heads=n_heads)
        self.task, self.epochs, self.lr, self.wd, self.seed, self.device = task, epochs, lr, wd, seed, device

    def fit(self, X: pd.DataFrame, y):
        torch.manual_seed(self.seed)
        self.clin_cols = [c for c in X.columns if c not in ("seq", "dt")]
        self.mu = X[self.clin_cols].mean().fillna(0) if self.clin_cols else None
        self.sd = X[self.clin_cols].std().replace(0, 1).fillna(1) if self.clin_cols else None
        toks = np.concatenate(list(X["seq"]), 0)
        self.smu, self.ssd = toks.mean(0), toks.std(0) + 1e-6
        S, Dt, C, M = _batch(self._norm(X), self.clin_cols)
        self.Tmax = S.shape[1]
        self.model = TemporalTransformer(S.shape[-1], n_clin=C.shape[1], **self.kw).to(self.device)
        yt = torch.tensor(np.asarray(y, np.float32))
        if self.task == "binary":
            pw = ((1 - yt.mean()) / yt.mean().clamp(min=1e-3)).detach()
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            loss_fn = nn.MSELoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        n = len(yt); g = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(n, generator=g).split(32):
                opt.zero_grad()
                out = self.model(S[idx].to(self.device), Dt[idx].to(self.device), C[idx].to(self.device), M[idx].to(self.device))
                loss_fn(out, yt[idx].to(self.device)).backward()
                opt.step()
        return self

    def _norm(self, X):
        X = X.copy()
        X["seq"] = [((s - self.smu) / self.ssd).astype(np.float32) for s in X["seq"]]
        if self.clin_cols:
            X[self.clin_cols] = (X[self.clin_cols] - self.mu) / self.sd
        return X

    @torch.no_grad()
    def _raw(self, X):
        self.model.eval()
        S, Dt, C, M = _batch(self._norm(X), self.clin_cols, self.Tmax)
        return self.model(S.to(self.device), Dt.to(self.device), C.to(self.device), M.to(self.device)).cpu().numpy()

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self._raw(X)))
        return np.stack([1 - p, p], 1)

    def predict(self, X):
        return self._raw(X)
