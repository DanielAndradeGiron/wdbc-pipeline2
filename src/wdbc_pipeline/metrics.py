"""Classification, calibration and decision metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "f1_weighted",
    "mcc",
    "kappa",
    "roc_auc",
    "average_precision",
    "brier",
    "ece",
)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, bins: int = 10, strategy: str = "quantile"
) -> float:
    """Frequency-weighted ECE; empty bins are omitted."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:  # all predictions identical -> one bin (the notebook returned 0 here)
            return float(abs(y.mean() - p.mean()))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, bins + 1)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    group = np.digitize(p, edges[1:-1], right=True)
    ece = 0.0
    for j in range(len(edges) - 1):
        m = group == j
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def _safe_auc(fn, y: np.ndarray, p: np.ndarray) -> float:
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")


def classification_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
    ece_bins: int = 10,
    ece_strategy: str = "quantile",
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    z = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, z),
        "balanced_accuracy": balanced_accuracy_score(y, z),
        "precision": precision_score(y, z, zero_division=0),
        "sensitivity": recall_score(y, z, zero_division=0),
        "specificity": recall_score(y, z, pos_label=0, zero_division=0),
        "f1": f1_score(y, z, zero_division=0),
        "f1_weighted": f1_score(y, z, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, z),
        "kappa": cohen_kappa_score(y, z),
        "roc_auc": _safe_auc(roc_auc_score, y, p),
        "average_precision": _safe_auc(average_precision_score, y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p, ece_bins, ece_strategy),
    }


def confusion_counts(y: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(
        y, (np.asarray(p) >= threshold).astype(int), labels=[0, 1]
    ).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else beta_dist.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = [np.flatnonzero(y == c) for c in (0, 1)]
    return np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])


def net_benefit(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    n, prev = len(y), y.mean()
    rows = []
    for t in grid:
        tn, fp, fn, tp = confusion_counts(y, p, t)
        w = t / (1 - t)
        rows.append(
            {
                "threshold": float(t),
                "model": tp / n - fp / n * w,
                "treat_all": prev - (1 - prev) * w,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)
