"""Typed, immutable configuration loaded from YAML.

* ``extends:`` lets a YAML file inherit from another one (deep merge).
* ``--set section.key=value`` overrides are applied after loading.
* Unknown keys raise immediately (typos never pass silently).
* :func:`config_hash` fingerprints every setting that can change results, so
  checkpoints and run directories are bound to the exact analysis plan.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "wdbc-pipeline"
    seed: int = 42
    output_dir: str = "outputs/runs"
    run_name: str = "run"


@dataclass(frozen=True)
class DataConfig:
    source: str = "sklearn"
    url: str | None = None
    csv_path: str | None = "data/raw/wdbc.csv"
    expected_sha256: str | None = None
    target: str = "diagnosis"
    positive_label: str = "M"
    id_like_columns: tuple[str, ...] = (
        "id",
        "index",
        "unnamed_0",
        "unnamed_32",
        "patient_id",
        "record_id",
        "name",
        "email",
    )
    drop_duplicates: bool = True
    test_size: float = 0.20


@dataclass(frozen=True)
class CVConfig:
    outer_splits: int = 10
    outer_repeats: int = 5
    inner_splits: int = 3
    search_n_iter: int = 15
    scoring: str = "f1_weighted"
    calibration_method: str = "sigmoid"
    calibration_splits: int = 3
    stacking_splits: int = 5


@dataclass(frozen=True)
class ModelsConfig:
    include: tuple[str, ...] = (
        "AdaBoost",
        "GradientBoosting",
        "RandomForest",
        "ExtraTrees",
        "BaggingExtraTree",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "VotingSoft",
        "Stacking",
    )
    on_missing_dependency: str = "error"
    oversample: bool = True
    class_weight_with_oversampling: bool = False
    param_grids: dict[str, dict[str, list[Any]]] = field(default_factory=dict)


@dataclass(frozen=True)
class Tolerance:
    metric: str
    tolerance: float
    direction: str = "max"


@dataclass(frozen=True)
class SelectionConfig:
    tolerances: tuple[Tolerance, ...] = (
        Tolerance("f1_weighted", 0.005, "max"),
        Tolerance("mcc", 0.010, "max"),
        Tolerance("roc_auc", 0.005, "max"),
        Tolerance("brier", 0.005, "min"),
    )


@dataclass(frozen=True)
class StatsConfig:
    alpha: float = 0.05
    metric: str = "f1_weighted"


@dataclass(frozen=True)
class ThresholdConfig:
    outer_splits: int = 5
    min_sensitivity: float = 0.95
    grid_start: float = 0.05
    grid_stop: float = 0.95
    grid_step: float = 0.01


@dataclass(frozen=True)
class EvaluationConfig:
    primary_threshold: float = 0.50
    ece_bins: int = 10
    ece_strategy: str = "quantile"
    holdout_bootstrap: int = 2000
    roc_bootstrap: int = 1000
    dca_bootstrap: int = 1000
    dca_grid: tuple[float, float, float] = (0.05, 0.75, 0.01)
    dev_sensitivity_repetitions: int = 100
    dev_sensitivity_mode: str = "subsample"
    dev_subsample_fraction: float = 0.80


@dataclass(frozen=True)
class ExplainConfig:
    enabled: bool = True
    permutation_repeats: int = 30
    shap: bool = True
    shap_background: int = 80
    shap_explained: int = 50
    lime: bool = True
    lime_min_fidelity: float = 0.50
    lime_num_samples: int = 20000


@dataclass(frozen=True)
class RuntimeConfig:
    n_jobs: int = -1
    checkpoint: bool = True
    excel_export: bool = True
    figure_formats: tuple[str, ...] = ("png", "pdf", "svg")
    dpi: int = 600


@dataclass(frozen=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Sections that never change numerical results (excluded from the hash).
_NON_RESULT_KEYS = {"runtime": None, "project": {"output_dir", "run_name", "name"}}

_SECTION_TYPES: dict[str, type] = {
    "project": ProjectConfig,
    "data": DataConfig,
    "cv": CVConfig,
    "models": ModelsConfig,
    "selection": SelectionConfig,
    "stats": StatsConfig,
    "threshold": ThresholdConfig,
    "evaluation": EvaluationConfig,
    "explain": ExplainConfig,
    "runtime": RuntimeConfig,
}


class ConfigError(ValueError):
    """Raised for invalid or inconsistent configuration values."""


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict) and key != "param_grids":
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ConfigError(f"Circular 'extends' chain at {path}")
    seen.add(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    parent = raw.pop("extends", None)
    if parent:
        return _deep_merge(_read_yaml(path.parent / parent, seen), raw)
    return raw


def _coerce(section_cls: type, values: Mapping[str, Any], section: str) -> Any:
    known = {f.name: f for f in dataclasses.fields(section_cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in section '{section}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if section_cls is SelectionConfig and key == "tolerances":
            value = tuple(Tolerance(**t) for t in value)
        elif isinstance(value, list) and key != "param_grids":
            value = tuple(value)
        kwargs[key] = value
    return section_cls(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML)."""
    out = copy.deepcopy(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        keys = dotted.strip().split(".")
        node = out
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(value)
    return out


def config_from_dict(raw: Mapping[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTION_TYPES)
    if unknown:
        raise ConfigError(f"Unknown configuration section(s): {sorted(unknown)}")
    sections = {
        name: _coerce(cls, raw.get(name, {}) or {}, name) for name, cls in _SECTION_TYPES.items()
    }
    cfg = Config(**sections)
    validate(cfg)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    raw = apply_overrides(_read_yaml(Path(path)), overrides)
    return config_from_dict(raw)


def validate(cfg: Config) -> None:
    errors: list[str] = []
    if cfg.data.source not in {"sklearn", "csv", "url"}:
        errors.append("data.source must be 'sklearn', 'csv' or 'url'")
    if cfg.data.source == "url" and not cfg.data.url:
        errors.append("data.url is required when data.source='url'")
    if not 0 < cfg.data.test_size < 1:
        errors.append("data.test_size must be in (0, 1)")
    for name in ("outer_splits", "inner_splits", "calibration_splits", "stacking_splits"):
        if getattr(cfg.cv, name) < 2:
            errors.append(f"cv.{name} must be >= 2")
    if cfg.cv.outer_repeats < 1 or cfg.cv.search_n_iter < 1:
        errors.append("cv.outer_repeats and cv.search_n_iter must be >= 1")
    if cfg.cv.calibration_method not in {"sigmoid", "isotonic"}:
        errors.append("cv.calibration_method must be 'sigmoid' or 'isotonic'")
    if cfg.models.on_missing_dependency not in {"error", "skip"}:
        errors.append("models.on_missing_dependency must be 'error' or 'skip'")
    if not cfg.models.include:
        errors.append("models.include cannot be empty")
    for tol in cfg.selection.tolerances:
        if tol.direction not in {"max", "min"} or tol.tolerance < 0:
            errors.append(f"invalid tolerance entry: {tol}")
    if cfg.threshold.outer_splits < 2 or not 0 < cfg.threshold.min_sensitivity <= 1:
        errors.append("threshold.outer_splits >= 2 and 0 < threshold.min_sensitivity <= 1")
    if cfg.evaluation.ece_strategy not in {"quantile", "uniform"}:
        errors.append("evaluation.ece_strategy must be 'quantile' or 'uniform'")
    if cfg.evaluation.dev_sensitivity_mode not in {"subsample", "bootstrap"}:
        errors.append("evaluation.dev_sensitivity_mode must be 'subsample' or 'bootstrap'")
    if not 0 < cfg.evaluation.dev_subsample_fraction < 1:
        errors.append("evaluation.dev_subsample_fraction must be in (0, 1)")
    if len(cfg.evaluation.dca_grid) != 3:
        errors.append("evaluation.dca_grid must be [start, stop, step]")
    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


def result_relevant_dict(cfg: Config) -> dict[str, Any]:
    d = cfg.to_dict()
    for section, keys in _NON_RESULT_KEYS.items():
        if keys is None:
            d.pop(section, None)
        else:
            for k in keys:
                d[section].pop(k, None)
    return d


def config_hash(cfg: Config, length: int = 12) -> str:
    payload = json.dumps(result_relevant_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
