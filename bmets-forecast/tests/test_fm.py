import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bmf.fm.embed import embed_study, load_embeddings, save_embeddings
from bmf.fm.lora import apply_lora
from bmf.fm.registry import ENCODERS, load_encoder


def test_random3d_embed_shapes(tmp_path):
    spec = ENCODERS["random3d"]
    model = load_encoder("random3d")
    vol = np.random.default_rng(0).normal(size=(4, 48, 48, 48)).astype(np.float32)
    vol[:, :4] = 0
    seg = np.zeros((48, 48, 48), np.int16); seg[20:26, 20:26, 20:26] = 3; seg[36:40, 36:40, 36:40] = 3
    e = embed_study(spec, model, vol, seg)
    assert e["brain"].shape == (spec.out_dim,)
    assert len(e["lesions"]) == 2 and all(v.shape == (spec.out_dim,) for v in e["lesions"].values())
    save_embeddings(tmp_path / "x.npz", e)
    e2 = load_embeddings(tmp_path / "x.npz")
    assert np.allclose(e2["brain"], e["brain"]) and set(e2["lesions"]) == set(e["lesions"])


def test_random3d_deterministic():
    a, b = load_encoder("random3d"), load_encoder("random3d")
    x = torch.ones(1, 4, 32, 32, 32)
    assert torch.allclose(a(x), b(x))


def test_lora_trainable_params():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(8, 24); self.proj = torch.nn.Linear(8, 8); self.mlp = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.mlp(self.proj(self.qkv(x)[..., :8]))
    m = Block()
    x = torch.randn(2, 8); y0 = m(x).detach()
    apply_lora(m, r=4, alpha=8, targets=("qkv", "proj"))
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)
    assert torch.allclose(m(x), y0, atol=1e-6)      # B initialised to zero -> identical output


def test_registry_missing_weights_message(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        load_encoder("brainsegfounder", weights_path=tmp_path / "nope.pt")
    assert "nope.pt" in str(e.value) and "github.com" in str(e.value)
