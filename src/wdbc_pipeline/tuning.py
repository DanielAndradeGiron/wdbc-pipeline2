"""Inner hyperparameter search and fold-local probability calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from .config import Config
from .models import build_model, param_grid


@dataclass
class TunedModel:
    calibrated: Any  # CalibratedClassifierCV fitted on (X, y)
    tuned: Any  # un-calibrated best estimator (refit on X, y)
    best_params: dict[str, Any]
    inner_score: float


def make_calibrator(estimator: Any, cfg: Config, seed: int) -> CalibratedClassifierCV:
    cv = StratifiedKFold(cfg.cv.calibration_splits, shuffle=True, random_state=seed)
    return CalibratedClassifierCV(
        estimator=estimator, method=cfg.cv.calibration_method, cv=cv, n_jobs=1, ensemble=True
    )


def tune(
    name: str, X: pd.DataFrame, y: np.ndarray, seed: int, cfg: Config
) -> tuple[Any, dict, float]:
    model = build_model(name, X, seed, cfg)
    grid = param_grid(name, cfg)
    if not grid:
        return model.fit(X, y), {}, float("nan")
    n_combos = math.prod(len(v) for v in grid.values())
    search = RandomizedSearchCV(
        model,
        grid,
        n_iter=min(cfg.cv.search_n_iter, n_combos),
        scoring=cfg.cv.scoring,
        cv=StratifiedKFold(cfg.cv.inner_splits, shuffle=True, random_state=seed + 1),
        random_state=seed + 2,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search.best_estimator_, dict(search.best_params_), float(search.best_score_)


def tune_and_calibrate(
    name: str, X: pd.DataFrame, y: np.ndarray, seed: int, cfg: Config
) -> TunedModel:
    tuned, best, inner = tune(name, X, y, seed, cfg)
    calibrated = make_calibrator(clone(tuned), cfg, seed + 101).fit(X, y)
    return TunedModel(calibrated, tuned, best, inner)
