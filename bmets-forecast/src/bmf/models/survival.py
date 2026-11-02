"""T4 overall survival: penalised Cox (lifelines) and DeepSurv (torch MLP, Cox partial likelihood)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def cox_cv(X: pd.DataFrame, surv: pd.DataFrame, splits: dict, penalizer: float = 0.1) -> pd.DataFrame:
    """X indexed by case_id; surv has time_years, event. Returns OOF risk (higher = worse)."""
    from lifelines import CoxPHFitter
    out = []
    folds = X.index.map(splits).to_numpy()
    for k in sorted(set(folds)):
        tr, te = folds != k, folds == k
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        Z = ((X - mu) / sd).fillna(0)
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(Z[tr].join(surv[["time_years", "event"]]), "time_years", "event")
        out.append(pd.DataFrame({"risk": cph.predict_partial_hazard(Z[te]).to_numpy().ravel(), "fold": k}, index=X.index[te]))
    return pd.concat(out)


def _cox_loss(risk, time, event):
    import torch
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, 0)
    return -((r - log_cum) * e).sum() / e.sum().clamp(min=1)


def deepsurv_cv(X: pd.DataFrame, surv: pd.DataFrame, splits: dict, hidden: int = 32, epochs: int = 300,
                lr: float = 1e-3, wd: float = 1e-3, seed: int = 2026) -> pd.DataFrame:
    import torch
    from torch import nn
    out = []
    folds = X.index.map(splits).to_numpy()
    for k in sorted(set(folds)):
        torch.manual_seed(seed)
        tr, te = folds != k, folds == k
        mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
        Z = torch.tensor(((X - mu) / sd).fillna(0).to_numpy(np.float32))
        t = torch.tensor(surv.loc[X.index, "time_years"].to_numpy(np.float32))
        e = torch.tensor(surv.loc[X.index, "event"].to_numpy(np.float32))
        net = nn.Sequential(nn.Linear(Z.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        trm = torch.tensor(tr)
        for _ in range(epochs):
            opt.zero_grad()
            _cox_loss(net(Z[trm]).squeeze(-1), t[trm], e[trm]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            out.append(pd.DataFrame({"risk": net(Z[torch.tensor(te)]).squeeze(-1).numpy(), "fold": k}, index=X.index[te]))
    return pd.concat(out)
