import pytest

from wdbc_pipeline.config import ConfigError, config_hash, load_config

from .conftest import ROOT


def test_default_and_quick_load():
    full = load_config(ROOT / "configs" / "default.yaml")
    quick = load_config(ROOT / "configs" / "quick.yaml")
    assert full.cv.outer_splits == 10 and full.cv.outer_repeats == 5
    assert quick.cv.outer_splits == 3  # inherited file overrides
    assert quick.models.param_grids == full.models.param_grids  # inherited untouched
    assert len(full.models.include) == 10


def test_notebook_replica_restores_original_choices():
    rep = load_config(ROOT / "configs" / "notebook_replica.yaml")
    assert rep.data.source == "url" and rep.data.url
    assert rep.models.class_weight_with_oversampling is True
    assert rep.evaluation.dev_sensitivity_mode == "bootstrap"


def test_hash_ignores_runtime_but_tracks_analysis():
    base = load_config(ROOT / "configs" / "default.yaml")
    same = load_config(
        ROOT / "configs" / "default.yaml", ["runtime.n_jobs=2", "project.run_name=x"]
    )
    diff = load_config(ROOT / "configs" / "default.yaml", ["cv.outer_repeats=2"])
    assert config_hash(base) == config_hash(same)
    assert config_hash(base) != config_hash(diff)


@pytest.mark.parametrize(
    "override",
    [
        "cv.typo=1",
        "data.source=excel",
        "cv.inner_splits=1",
        "evaluation.dev_sensitivity_mode=jackknife",
    ],
)
def test_invalid_config_is_rejected(override):
    with pytest.raises(ConfigError):
        load_config(ROOT / "configs" / "default.yaml", [override])
