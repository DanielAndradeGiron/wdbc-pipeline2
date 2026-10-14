import numpy as np
import pytest
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from wdbc_pipeline.models import (
    REGISTRY,
    audit_stacking_architecture,
    build_model,
    param_grid,
    resolve_models,
    strip_prefix,
)
from wdbc_pipeline.sampling import RandomOverSampledClassifier, random_oversample_indices


def test_oversampling_balances_classes():
    y = np.array([0] * 30 + [1] * 10)
    idx = random_oversample_indices(y, 0)
    assert np.bincount(y[idx]).tolist() == [30, 30]
    assert set(range(40)) <= set(idx)  # originals kept


def test_oversampler_is_fit_only_and_sklearn_compatible(toy_xy):
    X, y = toy_xy
    est = RandomOverSampledClassifier(LogisticRegression(), random_state=0)
    clone(est).fit(X, y)
    est.fit(X, y)
    assert est.n_resampled_ == 2 * np.bincount(y).max()
    assert est.predict_proba(X).shape == (len(X), 2)  # no resampling at predict time
    cal = CalibratedClassifierCV(est, cv=StratifiedKFold(3, shuffle=True, random_state=0)).fit(X, y)
    assert cal.predict_proba(X).shape == (len(X), 2)


def test_param_grid_prefix_matches_pipeline(quick_cfg, toy_xy):
    X, _ = toy_xy
    model = build_model("RandomForest", X, 0, quick_cfg)
    grid = param_grid("RandomForest", quick_cfg)
    model.set_params(**{k: v[0] for k, v in grid.items()})  # raises if any key is invalid
    assert strip_prefix({"clf__estimator__max_depth": 3}, quick_cfg) == {"max_depth": 3}
    assert param_grid("Stacking", quick_cfg) == {}


def test_no_double_balancing_by_default(quick_cfg, toy_xy):
    X, _ = toy_xy
    rf = build_model("RandomForest", X, 0, quick_cfg).named_steps["clf"].estimator
    assert rf.class_weight is None


def test_stacking_architecture_gate(quick_cfg, toy_xy):
    X, _ = toy_xy
    assert audit_stacking_architecture(build_model("Stacking", X, 0, quick_cfg), quick_cfg)


def test_registry_and_resolution(quick_cfg):
    assert list(REGISTRY)[:3] == ["AdaBoost", "GradientBoosting", "RandomForest"]
    usable, skipped = resolve_models(quick_cfg)
    assert set(usable) | set(skipped) == set(quick_cfg.models.include)


@pytest.mark.parametrize(
    "name", ["AdaBoost", "GradientBoosting", "ExtraTrees", "BaggingExtraTree", "VotingSoft"]
)
def test_sklearn_models_fit(name, quick_cfg, toy_xy):
    X, y = toy_xy
    p = build_model(name, X, 0, quick_cfg).fit(X, y).predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
