import numpy as np
import torch

from ecgtrust.figures.build import build_figures
from ecgtrust.figures.explain import integrated_gradients
from ecgtrust.models.resnet1d import ResNet1d


def test_integrated_gradients_completeness():
    """IG axiom: attributions sum to f(x) - f(baseline) (Sundararajan et al. 2017)."""
    torch.manual_seed(0)
    model = ResNet1d(width=8, blocks=(1, 1, 1, 1)).eval()
    ck = {"lead_mean": np.zeros(12), "lead_std": np.ones(12)}
    x = np.random.default_rng(0).normal(0, 1, (2, 12, 1000)).astype(np.float32)
    att = integrated_gradients([(model, ck)], x, target=1, steps=64)
    with torch.no_grad():
        fx = model(torch.from_numpy(x))[:, 1].numpy()
        f0 = model(torch.zeros(1, 12, 1000))[:, 1].numpy()
    assert np.allclose(att.sum(axis=(1, 2)), fx - f0, atol=0.05 * np.abs(fx - f0).max() + 1e-3)


def test_figures_skip_missing_inputs_and_write_index(tmp_cfg_val):
    made = build_figures(tmp_cfg_val, ["01_data", "02_training", "03_performance"])
    assert len(made["01_data"]) >= 5  # metadata-only plots need no signals or features
    assert made["03_performance"] == []  # no frozen test predictions in the fixture
    assert (tmp_cfg_val.reports / "figures" / "README.md").exists()
