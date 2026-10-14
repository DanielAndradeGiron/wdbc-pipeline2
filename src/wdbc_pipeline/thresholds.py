"""Fully nested out-of-fold operating threshold for the frozen algorithm.

The algorithm chosen by the benchmark stays fixed. Tuning and calibration are
repeated inside every threshold fold without seeing that fold's validation
rows, so every development record receives exactly one honest OOF probability.
Rule: maximise specificity subject to sensitivity ≥ ``min_sensitivity``
(ties → higher F1, then lower threshold); fallback: maximise sensitivity.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .artifacts import RunContext
from .data import load_development
from .logging_utils import get_logger
from .metrics import confusion_counts
from .tuning import tune_and_calibrate

log = get_logger("thresholds")


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    return np.round(np.arange(start, stop + 1e-12, step), 10)


def threshold_table(y: np.ndarray, p: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in grid:
        tn, fp, fn, tp = confusion_counts(y, p, t)
        rows.append(
            {
                "threshold": float(t),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "PPV": tp / (tp + fp) if tp + fp else np.nan,
                "NPV": tn / (tn + fn) if tn + fn else np.nan,
                "F1": f1_score(y, (p >= t).astype(int), zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(table: pd.DataFrame, min_sensitivity: float) -> tuple[float, str]:
    eligible = table[table["sensitivity"] >= min_sensitivity]
    if len(eligible):
        row = eligible.sort_values(
            ["specificity", "F1", "threshold"], ascending=[False, False, True]
        ).iloc[0]
        return float(row["threshold"]), "min_sensitivity_then_specificity"
    row = table.sort_values(
        ["sensitivity", "specificity", "threshold"], ascending=[False, False, True]
    ).iloc[0]
    return float(row["threshold"]), "fallback_max_sensitivity"


def run_threshold(ctx: RunContext) -> float:
    cfg = ctx.cfg
    dev = load_development(ctx)
    name = ctx.load_json("selection_rule")["selected_model"]
    splitter = StratifiedKFold(
        cfg.threshold.outer_splits, shuffle=True, random_state=cfg.project.seed + 700
    )
    folds = list(splitter.split(dev.X, dev.y))

    def work(f: int, tr: np.ndarray, va: np.ndarray):
        tm = tune_and_calibrate(name, dev.X.iloc[tr], dev.y[tr], cfg.project.seed + 8000 * f, cfg)
        return (
            f,
            va,
            tm.calibrated.predict_proba(dev.X.iloc[va])[:, 1],
            tm.best_params,
            tm.inner_score,
            len(tr),
        )

    oof = np.full(len(dev), np.nan)
    logs = []
    for f, va, p, bp, inner, ntr in Parallel(n_jobs=cfg.runtime.n_jobs)(
        delayed(work)(f, tr, va) for f, (tr, va) in enumerate(folds, start=1)
    ):
        oof[va] = p
        logs.append(
            {
                "threshold_outer_fold": f,
                "fixed_algorithm": name,
                "best_params": json.dumps(bp, sort_keys=True, default=str),
                "inner_score": inner,
                "train_n": ntr,
                "validation_n": len(va),
            }
        )
    assert np.isfinite(oof).all(), "Incomplete OOF predictions"
    ctx.save_table(
        pd.DataFrame(logs).sort_values("threshold_outer_fold"),
        "threshold_crossfit_fixed_algorithm_log",
    )
    oof_df = pd.DataFrame({"source_row_id": dev.ids, "observed": dev.y, "probability_oof": oof})
    assert oof_df["source_row_id"].is_unique and oof_df["probability_oof"].between(0, 1).all()
    ctx.save_table(oof_df, "threshold_oof_predictions")

    grid = threshold_grid(
        cfg.threshold.grid_start, cfg.threshold.grid_stop, cfg.threshold.grid_step
    )
    table = threshold_table(dev.y, oof, grid)
    ctx.save_table(table, "fully_nested_oof_thresholds_fixed_algorithm")
    thr, status = select_threshold(table, cfg.threshold.min_sensitivity)
    ctx.save_json(
        {
            "algorithm": name,
            "threshold": thr,
            "rule": status,
            "min_sensitivity": cfg.threshold.min_sensitivity,
        },
        "threshold_selection",
    )
    log.info("Frozen OOF threshold for %s: %.2f (%s)", name, thr, status)
    return thr
