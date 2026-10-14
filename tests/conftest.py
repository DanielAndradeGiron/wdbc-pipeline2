from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wdbc_pipeline.artifacts import RunContext
from wdbc_pipeline.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def quick_cfg():
    return load_config(ROOT / "configs" / "quick.yaml")


@pytest.fixture
def tiny_cfg():
    """Smallest configuration that still exercises every stage."""
    return load_config(
        ROOT / "configs" / "quick.yaml",
        [
            "models.include=[AdaBoost, ExtraTrees, Stacking]",
            "models.param_grids={AdaBoost: {n_estimators: [30, 60]}, ExtraTrees: {n_estimators: [30]}}",
            "cv.outer_splits=2",
            "cv.inner_splits=2",
            "cv.search_n_iter=1",
            "cv.calibration_splits=2",
            "cv.stacking_splits=2",
            "threshold.outer_splits=2",
            "evaluation.holdout_bootstrap=20",
            "evaluation.roc_bootstrap=10",
            "evaluation.dca_bootstrap=10",
            "evaluation.dev_sensitivity_repetitions=2",
            "explain.permutation_repeats=2",
            "explain.shap=false",
            "explain.lime=false",
            "runtime.n_jobs=1",
            "runtime.excel_export=false",
        ],
    )


@pytest.fixture
def ctx(tiny_cfg, tmp_path):
    return RunContext.create(tiny_cfg, tmp_path / "run")


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = (X["a"] + 0.5 * rng.normal(size=120) > 0.6).astype(int).to_numpy()
    return X, y
