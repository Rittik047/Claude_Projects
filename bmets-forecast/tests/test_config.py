from bmf.config import load_config


def test_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "rpatra")
    p = tmp_path / "r.yaml"
    p.write_text("paths:\n  cache: /scratch/${USER}/bmets/cache\nn_folds: 3\n")
    c = load_config(p)
    assert c["paths"]["cache"] == "/scratch/rpatra/bmets/cache"
    assert c["n_folds"] == 3 and c["seed"] == 2026          # base values kept
