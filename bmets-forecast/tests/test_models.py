import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from bmf.models.cv import run_cv
from bmf.models.fusion import stack_oof
from bmf.models.survival import cox_cv, deepsurv_cv


def _tab(n_pat=80, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pat):
        for t in range(1, rng.integers(2, 5)):
            x = rng.normal(size=5)
            rows.append({"case_id": f"P{i:03d}", "t": t, "d_sum_et": x[0], **{f"f{k}": x[k] for k in range(1, 5)},
                         "y": int(x[0] + 0.3 * rng.normal() > 0.5)})
    df = pd.DataFrame(rows).set_index(["case_id", "t"])
    splits = {f"P{i:03d}": i % 5 for i in range(n_pat)}
    return df.drop(columns="y"), df["y"], splits


def test_run_cv_no_patient_leak():
    X, y, sp = _tab()
    oof = run_cv("P0", X, y, sp)
    assert len(oof) == len(X) and set(oof.columns) >= {"case_id", "t", "y", "pred", "fold"}
    assert (oof.groupby("case_id").fold.nunique() == 1).all()
    assert (oof.fold == oof.case_id.map(sp)).all()


def test_p1_learns_signal():
    X, y, sp = _tab()
    oof = run_cv("P1", X, y, sp)
    assert roc_auc_score(oof.y, oof.pred) > 0.8


def test_temporal_forward_mask():
    torch = pytest.importorskip("torch")
    from bmf.models.temporal import TemporalTransformer
    m = TemporalTransformer(d_in=16, d_model=32, n_layers=2, n_heads=4, n_clin=3).eval()
    seq = torch.randn(2, 4, 16); dt = torch.tensor([[0, .2, .4, .6]] * 2); clin = torch.randn(2, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    a = m(seq, dt, clin, mask)
    seq2 = seq.clone(); seq2[0, 2:] = 99.0
    b = m(seq2, dt, clin, mask)
    assert a.shape == (2,) and torch.allclose(a[0], b[0], atol=1e-5)


def test_temporal_model_cv_runs():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        T = int(rng.integers(1, 4)); sig = rng.normal()
        rows.append({"case_id": f"P{i:03d}", "t": T, "seq": rng.normal(size=(T, 8)).astype(np.float32) + sig,
                     "dt": np.linspace(0, .2 * (T - 1), T).astype(np.float32), "age": rng.normal(), "y": int(sig > 0)})
    df = pd.DataFrame(rows).set_index(["case_id", "t"])
    sp = {f"P{i:03d}": i % 4 for i in range(40)}
    oof = run_cv("P3", df.drop(columns="y"), df["y"], sp, epochs=40)
    assert roc_auc_score(oof.y, oof.pred) > 0.75


def test_stack_uses_only_train_oof():
    X, y, sp = _tab()
    a = run_cv("P0", X, y, sp); b = run_cv("P1", X, y, sp)
    s1 = stack_oof([a, b])
    a2, b2 = a.copy(), b.copy()
    flip = a2.fold == 0
    a2.loc[flip, "y"] = 1 - a2.loc[flip, "y"]; b2.loc[flip, "y"] = a2.loc[flip, "y"]
    s2 = stack_oof([a2, b2])
    assert np.allclose(s1[s1.fold == 0].pred, s2[s2.fold == 0].pred)


def _surv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"), index=[f"P{i:03d}" for i in range(n)])
    risk = 1.2 * X.a - 0.8 * X.b
    T = rng.exponential(np.exp(-risk)); C = rng.exponential(2.0, n)
    s = pd.DataFrame({"time_years": np.minimum(T, C), "event": (T <= C).astype(int)}, index=X.index)
    return X, s, {k: i % 5 for i, k in enumerate(X.index)}


def test_cox_cindex_signal():
    from lifelines.utils import concordance_index
    X, s, sp = _surv()
    oof = cox_cv(X, s, sp)
    assert concordance_index(s.loc[oof.index, "time_years"], -oof.risk, s.loc[oof.index, "event"]) > 0.7


def test_deepsurv_cindex_signal():
    pytest.importorskip("torch")
    from lifelines.utils import concordance_index
    X, s, sp = _surv()
    oof = deepsurv_cv(X, s, sp, epochs=200)
    assert concordance_index(s.loc[oof.index, "time_years"], -oof.risk, s.loc[oof.index, "event"]) > 0.7
