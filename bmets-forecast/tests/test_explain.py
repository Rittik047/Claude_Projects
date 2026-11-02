import numpy as np
import pandas as pd
import pytest

from bmf.explain.ablation import sequence_ablation
from bmf.explain.shap_tab import shap_values
from bmf.models.tabular import make_model


def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = (X.a - X.b + rng.normal(0, .5, n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("name", ["P0", "P1"])
def test_shap_additivity(name):
    X, y = _xy()
    m = make_model(name).fit(X, y)
    vals, base, Xs = shap_values(m, X)
    if name == "P1":
        margin = m.predict(X, raw_score=True)
    else:
        margin = m.decision_function(X)
    assert vals.shape == X.shape
    assert np.allclose(base + vals.sum(1), margin, atol=1e-3)
    assert np.abs(vals[:, 0]).mean() > np.abs(vals[:, 2]).mean()


def test_ig_completeness():
    torch = pytest.importorskip("torch")
    from bmf.explain.saliency import integrated_gradients
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Conv3d(2, 4, 3, padding=1), torch.nn.Tanh(), torch.nn.Flatten(), torch.nn.LazyLinear(1))
    x = torch.randn(2, 6, 6, 6)
    net(x[None])
    ig = integrated_gradients(lambda v: net(v)[:, 0], x, steps=64)
    gap = (net(x[None]) - net(torch.zeros_like(x)[None])).item()
    assert ig.shape == x.shape
    assert abs(ig.sum().item() - gap) <= 0.05 * abs(gap) + 1e-4


def test_gradcam_shape():
    torch = pytest.importorskip("torch")
    from bmf.explain.saliency import gradcam3d
    conv = torch.nn.Conv3d(1, 3, 3, padding=1)
    net = torch.nn.Sequential(conv, torch.nn.ReLU(), torch.nn.AdaptiveAvgPool3d(1), torch.nn.Flatten(), torch.nn.Linear(3, 1))
    cam = gradcam3d(lambda v: net(v)[:, 0], torch.randn(1, 8, 8, 8), conv)
    assert cam.shape == (8, 8, 8) and cam.min() >= 0 and cam.max() <= 1


def test_ablation_rows():
    seqs = {"t1c": ["f1"], "t2f": ["f2"], "adc": ["f3"]}
    score = lambda drop: 0.8 - (0.1 if drop == "t1c" else 0.02 if drop else 0.0)
    df = sequence_ablation(score, seqs)
    assert list(df.dropped) == ["none", "t1c", "t2f", "adc"]
    assert df.set_index("dropped").loc["t1c", "delta"] == pytest.approx(-0.1)
