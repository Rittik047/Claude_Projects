import numpy as np

from ecgtrust.robustness.occlusion import lead_occlusion

rng = np.random.default_rng(0)
Y = (rng.random((200, 5)) < 0.5).astype(int)
X_WITH_SIGNAL_IN_LEAD_I = rng.normal(0, 0.1, (200, 12, 50)) + Y[:, :1, None] * 1.0


def test_only_used_lead_matters():
    def fn(x):
        return np.repeat(1 / (1 + np.exp(-x[:, 0].mean(axis=1, keepdims=True))), 5, axis=1)

    df = lead_occlusion(fn, X_WITH_SIGNAL_IN_LEAD_I, np.repeat(Y[:, :1], 5, axis=1))
    assert set(df.columns) >= {"lead", "class", "auroc_drop"} and len(df) == 12 * 5
    assert (df[df.lead == "I"].auroc_drop > 0.1).all()
    assert (df[df.lead != "I"].auroc_drop.abs() < 1e-9).all()
