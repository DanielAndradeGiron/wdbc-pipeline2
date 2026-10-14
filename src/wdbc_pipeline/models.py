"""Candidate model registry, preprocessing and leakage-safe architectures.

Every candidate is ``Pipeline([prep, clf])`` where ``clf`` optionally wraps the
learner in :class:`RandomOverSampledClassifier`. Preprocessing is fitted on the
original training fold, and resampling happens only on that fold's training
rows. Ensembles (VotingSoft, Stacking) receive raw features and each base
learner owns its own preprocessing and resampling, so stacking meta-features
are produced without leakage.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import ExtraTreeClassifier

from .config import Config
from .logging_utils import get_logger
from .sampling import RandomOverSampledClassifier

log = get_logger("models")

ENSEMBLES = ("VotingSoft", "Stacking")


def preprocessor(X_ref: pd.DataFrame) -> ColumnTransformer:
    num = X_ref.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X_ref.columns if c not in num]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- base learners
def _cw(cfg: Config) -> str | None:
    """class_weight for RF/ET. Avoid double balancing unless replicating the notebook."""
    if not cfg.models.oversample or cfg.models.class_weight_with_oversampling:
        return "balanced"
    return None


def _rf(seed: int, cfg: Config) -> Any:
    return RandomForestClassifier(
        n_estimators=300, class_weight=_cw(cfg), random_state=seed, n_jobs=1
    )


def _et(seed: int, cfg: Config) -> Any:
    return ExtraTreesClassifier(
        n_estimators=300, class_weight=_cw(cfg), random_state=seed, n_jobs=1
    )


def _gb(seed: int, cfg: Config) -> Any:
    return GradientBoostingClassifier(random_state=seed)


def _xgb(seed: int, cfg: Config) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
    )


def _lgbm(seed: int, cfg: Config) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=250, learning_rate=0.05, num_leaves=15, random_state=seed, n_jobs=1, verbose=-1
    )


def _cat(seed: int, cfg: Config) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=250,
        depth=5,
        learning_rate=0.05,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int, Config], Any] | None  # None for ensembles
    requires: tuple[str, ...] = ()


REGISTRY: dict[str, ModelSpec] = {
    "AdaBoost": ModelSpec("AdaBoost", lambda s, c: AdaBoostClassifier(random_state=s)),
    "GradientBoosting": ModelSpec("GradientBoosting", _gb),
    "RandomForest": ModelSpec("RandomForest", _rf),
    "ExtraTrees": ModelSpec("ExtraTrees", _et),
    "BaggingExtraTree": ModelSpec(
        "BaggingExtraTree",
        lambda s, c: BaggingClassifier(
            estimator=ExtraTreeClassifier(random_state=s),
            n_estimators=200,
            random_state=s,
            n_jobs=1,
        ),
    ),
    "XGBoost": ModelSpec("XGBoost", _xgb, ("xgboost",)),
    "LightGBM": ModelSpec("LightGBM", _lgbm, ("lightgbm",)),
    "CatBoost": ModelSpec("CatBoost", _cat, ("catboost",)),
    "VotingSoft": ModelSpec("VotingSoft", None),
    "Stacking": ModelSpec("Stacking", None),
}


def missing_dependencies(name: str) -> list[str]:
    return [m for m in REGISTRY[name].requires if importlib.util.find_spec(m) is None]


def resolve_models(cfg: Config) -> tuple[list[str], dict[str, list[str]]]:
    """Return (usable model names, {skipped: missing packages})."""
    unknown = [m for m in cfg.models.include if m not in REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; available: {list(REGISTRY)}")
    usable, skipped = [], {}
    for name in cfg.models.include:
        miss = missing_dependencies(name)
        if miss:
            skipped[name] = miss
        else:
            usable.append(name)
    if skipped:
        msg = ", ".join(f"{k} (needs {', '.join(v)})" for k, v in skipped.items())
        if cfg.models.on_missing_dependency == "error":
            raise ImportError(
                f"Missing dependencies for: {msg}. Install them or set "
                "models.on_missing_dependency=skip."
            )
        log.warning("Skipping models with missing dependencies: %s", msg)
    if not usable:
        raise RuntimeError("No candidate model can be built.")
    return usable, skipped


# --------------------------------------------------------------------------- architectures
def sampled(clf: Any, X_ref: pd.DataFrame, seed: int, cfg: Config) -> Pipeline:
    step = RandomOverSampledClassifier(clf, random_state=seed) if cfg.models.oversample else clf
    return Pipeline([("prep", preprocessor(X_ref)), ("clf", step)])


def build_model(name: str, X_ref: pd.DataFrame, seed: int, cfg: Config) -> Any:
    if name == "VotingSoft":
        est = [
            ("rf", sampled(_rf(seed, cfg), X_ref, seed + 11, cfg)),
            ("et", sampled(_et(seed, cfg), X_ref, seed + 12, cfg)),
            ("gb", sampled(_gb(seed, cfg), X_ref, seed + 13, cfg)),
        ]
        return VotingClassifier(est, voting="soft", n_jobs=1)
    if name == "Stacking":
        est = [
            ("rf", sampled(_rf(seed, cfg), X_ref, seed + 21, cfg)),
            ("et", sampled(_et(seed, cfg), X_ref, seed + 22, cfg)),
            ("gb", sampled(_gb(seed, cfg), X_ref, seed + 23, cfg)),
        ]
        return StackingClassifier(
            est,
            final_estimator=LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            ),
            cv=StratifiedKFold(cfg.cv.stacking_splits, shuffle=True, random_state=seed + 24),
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=1,
        )
    spec = REGISTRY[name]
    assert spec.factory is not None
    return sampled(spec.factory(seed, cfg), X_ref, seed, cfg)


def param_prefix(cfg: Config) -> str:
    return "clf__estimator__" if cfg.models.oversample else "clf__"


def param_grid(name: str, cfg: Config) -> dict[str, list[Any]]:
    """Grid from config with the correct pipeline prefix (empty for ensembles)."""
    if name in ENSEMBLES:
        return {}
    grid = cfg.models.param_grids.get(name, {}) or {}
    prefix = param_prefix(cfg)
    return {f"{prefix}{k}": list(v) for k, v in grid.items()}


def strip_prefix(params: dict[str, Any], cfg: Config) -> dict[str, Any]:
    prefix = param_prefix(cfg)
    return {k.removeprefix(prefix): v for k, v in params.items()}


def audit_stacking_architecture(model: Any, cfg: Config) -> bool:
    """Machine-verifiable gate: every stacking base learner owns prep + sampler."""
    assert isinstance(model, StackingClassifier)
    for name, est in model.estimators:
        assert isinstance(est, Pipeline), f"{name} is not a Pipeline"
        assert list(est.named_steps) == ["prep", "clf"], f"Unexpected steps in {name}"
        if cfg.models.oversample:
            assert isinstance(est.named_steps["clf"], RandomOverSampledClassifier), name
    assert isinstance(model.cv, StratifiedKFold) and model.cv.shuffle
    return True
