"""Single opening of the locked holdout and all post-holdout analyses."""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import plotting
from .artifacts import RunContext
from .data import HoldoutGuard, Split, load_development
from .final import FINAL_TUNED, load_final
from .logging_utils import get_logger
from .metrics import (
    METRIC_NAMES,
    classification_metrics,
    clopper_pearson,
    confusion_counts,
    net_benefit,
    stratified_bootstrap_indices,
)
from .tuning import make_calibrator

log = get_logger("evaluation")

PRED_TABLE = "final_internal_holdout_predictions"


def _operating_points(frozen: dict) -> list[tuple[str, float]]:
    return [
        ("primary", float(frozen["primary_threshold"])),
        ("secondary_OOF", float(frozen["secondary_oof_threshold"])),
    ]


def _metrics(ctx: RunContext, y: np.ndarray, p: np.ndarray, t: float) -> dict[str, float]:
    e = ctx.cfg.evaluation
    return classification_metrics(y, p, t, e.ece_bins, e.ece_strategy)


# --------------------------------------------------------------------------- holdout
def run_holdout(ctx: RunContext, force: bool = False) -> None:
    model, frozen = load_final(ctx)
    hold = HoldoutGuard(ctx).open(force=force)
    if list(hold.X.columns) != frozen["feature_names"]:
        raise RuntimeError("Holdout feature names differ from the frozen model's features.")
    t0 = time.perf_counter()
    p = model.predict_proba(hold.X)[:, 1]
    inference_s = time.perf_counter() - t0

    rows, cms = [], []
    for label, t in _operating_points(frozen):
        rows.append(
            {
                "evaluation": label,
                "threshold": t,
                "inference_s": inference_s,
                **_metrics(ctx, hold.y, p, t),
            }
        )
        tn, fp, fn, tp = confusion_counts(hold.y, p, t)
        title = {"primary": "Primary threshold", "secondary_OOF": "Secondary nested OOF"}[label]
        cms.append((f"{title}: {t:.2f}", np.array([[tn, fp], [fn, tp]])))
    results = pd.DataFrame(rows)
    ctx.save_table(results, "table_05_final_internal_holdout_metrics")
    ctx.save_table(
        pd.DataFrame({"source_row_id": hold.ids, "observed": hold.y, "probability": p}), PRED_TABLE
    )
    ctx.save_figure(plotting.confusion_matrices(cms), "figure_03_holdout_confusion_matrices")
    for r in rows:
        log.info(
            "Holdout @%.2f: acc=%.4f sens=%.4f spec=%.4f MCC=%.4f AUC=%.4f",
            r["threshold"],
            r["accuracy"],
            r["sensitivity"],
            r["specificity"],
            r["mcc"],
            r["roc_auc"],
        )


def _holdout_predictions(ctx: RunContext) -> tuple[Split, np.ndarray, dict]:
    hold = HoldoutGuard(ctx).read_after_opening()
    preds = ctx.load_table(PRED_TABLE)
    if not np.array_equal(preds["source_row_id"].to_numpy(), hold.ids):
        raise RuntimeError("Stored holdout predictions do not match the holdout rows.")
    frozen = ctx.load_json("frozen_configuration_before_holdout")
    return hold, preds["probability"].to_numpy(dtype=float), frozen


# --------------------------------------------------------------------------- uncertainty
def run_uncertainty(ctx: RunContext) -> None:
    cfg = ctx.cfg
    hold, p, frozen = _holdout_predictions(ctx)
    y = hold.y
    alpha = cfg.stats.alpha
    q_lo, q_hi = alpha / 2, 1 - alpha / 2
    ci_rows, exact_rows = [], []
    for i, (label, t) in enumerate(_operating_points(frozen)):
        rng = np.random.default_rng(cfg.project.seed + 14000 + i)
        boot = pd.DataFrame(
            [
                _metrics(ctx, y[idx], p[idx], t)
                for idx in (
                    stratified_bootstrap_indices(y, rng)
                    for _ in range(cfg.evaluation.holdout_bootstrap)
                )
            ]
        )
        est = _metrics(ctx, y, p, t)
        for m in METRIC_NAMES:
            ci_rows.append(
                {
                    "evaluation": label,
                    "threshold": t,
                    "metric": m,
                    "estimate": est[m],
                    "ci_low": boot[m].quantile(q_lo),
                    "ci_high": boot[m].quantile(q_hi),
                    "confidence": 1 - alpha,
                    "B": len(boot),
                    "method": "stratified percentile bootstrap; conditional on frozen model",
                }
            )
        tn, fp, fn, tp = confusion_counts(y, p, t)
        for m, k, n in [
            ("sensitivity", tp, tp + fn),
            ("specificity", tn, tn + fp),
            ("precision", tp, tp + fp),
            ("NPV", tn, tn + fn),
        ]:
            lo, hi = clopper_pearson(k, n, alpha)
            exact_rows.append(
                {
                    "evaluation": label,
                    "threshold": t,
                    "metric": m,
                    "successes": k,
                    "denominator": n,
                    "estimate": k / n if n else np.nan,
                    "ci_low": lo,
                    "ci_high": hi,
                    "method": "Clopper-Pearson exact",
                }
            )
    ctx.save_table(pd.DataFrame(ci_rows), "table_06_holdout_bootstrap_ci")
    ctx.save_table(pd.DataFrame(exact_rows), "table_07_holdout_exact_binomial_ci")
    _dev_sensitivity(ctx, hold, frozen)


def _dev_sensitivity(ctx: RunContext, hold: Split, frozen: dict) -> None:
    """Training-sample perturbation with frozen algorithm, hyperparameters and threshold.

    ``subsample`` (default) draws a stratified fraction *without* replacement so
    no record can appear in both the fitting and the calibration folds of the
    internal CalibratedClassifierCV. ``bootstrap`` replicates the original
    notebook (with replacement).
    """
    cfg = ctx.cfg
    ev = cfg.evaluation
    dev = load_development(ctx)
    tuned = joblib.load(ctx.models / FINAL_TUNED)
    rng = np.random.default_rng(cfg.project.seed + 15000)
    plans = []
    for _ in range(ev.dev_sensitivity_repetitions):
        parts = []
        for c in (0, 1):
            pool = np.flatnonzero(dev.y == c)
            if ev.dev_sensitivity_mode == "bootstrap":
                parts.append(rng.choice(pool, len(pool), replace=True))
            else:
                parts.append(
                    rng.choice(pool, round(ev.dev_subsample_fraction * len(pool)), replace=False)
                )
        plans.append(np.concatenate(parts))

    def work(b: int, idx: np.ndarray) -> np.ndarray:
        m = make_calibrator(clone(tuned), cfg, cfg.project.seed + 17000 + b).fit(
            dev.X.iloc[idx], dev.y[idx]
        )
        return m.predict_proba(hold.X)[:, 1]

    probs = Parallel(n_jobs=cfg.runtime.n_jobs)(
        delayed(work)(b, idx) for b, idx in enumerate(plans)
    )
    rows = []
    for b, p in enumerate(probs, start=1):
        for label, t in _operating_points(frozen):
            rows.append(
                {
                    "development_resample": b,
                    "mode": ev.dev_sensitivity_mode,
                    "evaluation": label,
                    "threshold": t,
                    "algorithm": frozen["final_model"],
                    **_metrics(ctx, hold.y, p, t),
                }
            )
    dev_unc = pd.DataFrame(rows)
    ctx.save_table(dev_unc, "development_pipeline_sensitivity")
    summ = []
    for label, g in dev_unc.groupby("evaluation"):
        for m in METRIC_NAMES:
            summ.append(
                {
                    "evaluation": label,
                    "metric": m,
                    "median": g[m].median(),
                    "p2_5": g[m].quantile(0.025),
                    "p97_5": g[m].quantile(0.975),
                    "interpretation": "training-sample perturbation conditional on frozen "
                    "algorithm/hyperparameters/threshold; not external validation",
                }
            )
    ctx.save_table(pd.DataFrame(summ), "development_uncertainty_summary")
    log.info("Development-sample sensitivity: %d %s resamples", len(probs), ev.dev_sensitivity_mode)


# --------------------------------------------------------------------------- curves
def run_curves(ctx: RunContext) -> None:
    cfg = ctx.cfg
    ev = cfg.evaluation
    hold, p, frozen = _holdout_predictions(ctx)
    y = hold.y
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)

    fpr_grid = np.linspace(0, 1, 201)
    rng = np.random.default_rng(cfg.project.seed + 17500)
    tprs = []
    for _ in range(ev.roc_bootstrap):
        idx = stratified_bootstrap_indices(y, rng)
        bf, bt, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(fpr_grid, bf, bt))
    roc_lo, roc_hi = np.quantile(np.asarray(tprs), [0.025, 0.975], axis=0)
    ctx.save_figure(
        plotting.roc_pr(fpr, tpr, fpr_grid, roc_lo, roc_hi, auc, recall, precision, ap, y.mean()),
        "figure_04_holdout_roc_pr",
    )

    frac, meanp = calibration_curve(y, p, n_bins=ev.ece_bins, strategy=ev.ece_strategy)
    m = _metrics(ctx, y, p, 0.5)
    ctx.save_table(
        pd.DataFrame({"mean_predicted": meanp, "observed_frequency": frac}),
        "holdout_calibration_curve",
    )
    ctx.save_figure(
        plotting.calibration(meanp, frac, m["brier"], m["ece"], ev.ece_bins),
        "figure_05_holdout_calibration",
    )

    start, stop, step = ev.dca_grid
    grid = np.round(np.arange(start, stop + 1e-12, step), 10)
    point = net_benefit(y, p, grid)
    rng = np.random.default_rng(cfg.project.seed + 18000)
    boots = np.vstack(
        [
            net_benefit(y[idx], p[idx], grid)["model"].to_numpy()
            for idx in (stratified_bootstrap_indices(y, rng) for _ in range(ev.dca_bootstrap))
        ]
    )
    point["ci95_low"], point["ci95_high"] = np.quantile(boots, [0.025, 0.975], axis=0)
    ctx.save_table(point, "table_08_holdout_dca_bootstrap")
    ctx.save_figure(
        plotting.dca(point, frozen["primary_threshold"], frozen["secondary_oof_threshold"]),
        "figure_06_holdout_dca",
    )
    log.info("Curves: AUC=%.4f AP=%.4f Brier=%.4f ECE=%.4f", auc, ap, m["brier"], m["ece"])
