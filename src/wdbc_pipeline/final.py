"""Final fit on the full development set and freezing of all decisions."""

from __future__ import annotations

import joblib

from .artifacts import RunContext, sha256_file
from .config import config_hash
from .data import load_development
from .logging_utils import get_logger
from .models import strip_prefix
from .tuning import tune_and_calibrate

log = get_logger("final")

FINAL_MODEL = "final_calibrated_model.joblib"
FINAL_TUNED = "final_tuned_uncalibrated.joblib"


def run_final(ctx: RunContext) -> None:
    cfg = ctx.cfg
    dev = load_development(ctx)
    selected = ctx.load_json("selection_rule")["selected_model"]
    thr = ctx.load_json("threshold_selection")
    if thr["algorithm"] != selected:
        raise RuntimeError("Threshold was derived for a different algorithm than the selected one.")

    tm = tune_and_calibrate(selected, dev.X, dev.y, cfg.project.seed + 12000, cfg)
    model_path, tuned_path = ctx.models / FINAL_MODEL, ctx.models / FINAL_TUNED
    joblib.dump(tm.calibrated, model_path)
    joblib.dump(tm.tuned, tuned_path)

    frozen = {
        "final_model": selected,
        "benchmark_selected_model": selected,
        "selection_consistency": True,
        "best_params": strip_prefix(tm.best_params, cfg),
        "final_inner_score": tm.inner_score,
        "calibration": {
            "method": cfg.cv.calibration_method,
            "cv": cfg.cv.calibration_splits,
            "ensemble": True,
        },
        "primary_threshold": cfg.evaluation.primary_threshold,
        "secondary_oof_threshold": thr["threshold"],
        "threshold_rule": thr["rule"],
        "minimum_sensitivity": cfg.threshold.min_sensitivity,
        "feature_names": list(dev.X.columns),
        "model_sha256": sha256_file(model_path),
        "config_hash": config_hash(cfg),
    }
    ctx.save_json(frozen, "frozen_configuration_before_holdout")
    log.info("Final %s frozen with params %s", selected, frozen["best_params"])


def load_final(ctx: RunContext):
    frozen = ctx.load_json("frozen_configuration_before_holdout")
    path = ctx.models / FINAL_MODEL
    if sha256_file(path) != frozen["model_sha256"]:
        raise RuntimeError("Final model file does not match the frozen checksum.")
    return joblib.load(path), frozen
