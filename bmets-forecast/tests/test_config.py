from pathlib import Path

from bmf.config import load_config


def test_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "rpatra")
    p = tmp_path / "r.yaml"
    p.write_text("paths:\n  cache: /scratch/${USER}/bmets/cache\nn_folds: 3\n")
    c = load_config(p)
    assert c["paths"]["cache"] == "/scratch/rpatra/bmets/cache"
    assert c["n_folds"] == 3 and c["seed"] == 2026          # base values kept


def test_subset_keeps_n_longitudinal_patients_deterministically(tmp_path):
    from bmf.pipeline import Ctx
    fix = Path(__file__).parent / "fixtures"
    cfg = {"paths": {"clinical": str(fix / "clinical_min.tsv"), "cache": str(tmp_path / "c"),
                     "reports": str(tmp_path / "r")}, "seed": 2026, "subset": {"n_patients": 2, "min_timepoints": 3}}
    a, b = Ctx(cfg).clin(), Ctx(cfg).clin()
    assert len(a) == 2 and list(a.case_id) == list(b.case_id)
    tp_counts = a.filter(like="age_t").notna().sum(axis=1)
    assert (tp_counts >= 3).all()


def test_cli_subset_flag_sets_config(tmp_path, monkeypatch):
    from bmf import cli
    seen = {}
    monkeypatch.setitem(cli.STAGES, "audit", lambda ctx, **kw: seen.update(ctx.cfg.get("subset", {}), cfg=ctx.cfg))
    fix = Path(__file__).parent / "fixtures"
    y = tmp_path / "c.yaml"
    y.write_text(f"paths: {{clinical: {fix/'clinical_min.tsv'}, cache: {tmp_path/'c'}, reports: {tmp_path/'r'}}}\n")
    cli.main(["audit", "--config", str(y), "--subset", "3"])
    assert seen["n_patients"] == 3
    c = seen["cfg"]                       # smoke defaults: small folds/bootstrap/epochs, no pyradiomics
    assert c["n_folds"] == 3 and c["bootstrap"]["n"] == 200 and c["train"]["epochs"] == 5
    assert c["radiomics_backend"] == "simple"


def test_subset_runs_use_separate_cache_and_reports(tmp_path):
    from bmf.pipeline import Ctx
    cfg = {"paths": {"clinical": "x", "cache": str(tmp_path / "cache"), "reports": str(tmp_path / "reports")},
           "subset": {"n_patients": 4}}
    c = Ctx(cfg)
    assert c.cache.name == "cache_smoke" and c.reports.name == "reports_smoke"
    assert c.figs.parent.name == "reports_smoke"


def test_subset_never_writes_real_lock_dir(tmp_path):
    from bmf.pipeline import Ctx, _lock_dir
    cfg = {"paths": {"clinical": "x", "cache": str(tmp_path / "cache"), "reports": str(tmp_path / "r")},
           "lock_dir": str(tmp_path / "cache" / "locked"), "subset": {"n_patients": 4}}
    assert "cache_smoke" in str(_lock_dir(Ctx(cfg)))
