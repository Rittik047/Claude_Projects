import pytest

from bmf.viz.registry import FIGURES, demo_bundle, render

BUNDLE = demo_bundle()


def test_registry_size():
    assert len(FIGURES) >= 35


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_all_figures_render(name, tmp_path):
    p = render(name, BUNDLE, tmp_path)
    assert p.exists() and p.with_suffix(".pdf").exists() and p.stat().st_size > 5000
