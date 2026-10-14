"""Repeated nested cross-validation benchmark (parallel, checkpointed, resumable).

For every outer block (repeat × fold) and every candidate:
inner RandomizedSearchCV → refit → fold-local sigmoid calibration → predict the
untouched outer validation fold. Outer blocks are independent, so they run in
parallel (``runtime.n_jobs``) while every inner estimator uses ``n_jobs=1`` to
avoid oversubscription. Each finished block is checkpointed; an interrupted run
resumes from the last completed block.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from .artifacts import RunContext
from .config import Config
from .data import Split, load_development
from .logging_utils import get_logger
from .metrics import classification_metrics
from .models import REGISTRY, resolve_models
from .tuning import tune_and_calibrate

log = get_logger("benchmark")

_MODEL_INDEX = {
    name: j for j, name in enumerate(REGISTRY)
}  # stable seeds even if models are skipped


def model_seed(base: int, block_index: int, model: str) -> int:
    return base + 1000 * block_index + _MODEL_INDEX[model]


def outer_splitter(cfg: Config) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(
        n_splits=cfg.cv.outer_splits,
        n_repeats=cfg.cv.outer_repeats,
        random_state=cfg.project.seed + 100,
    )


def _run_block(
    k: int, tr: np.ndarray, va: np.ndarray, dev: Split, models: list[str], cfg: Config
) -> dict[str, Any]:
    repeat, fold = k // cfg.cv.outer_splits + 1, k % cfg.cv.outer_splits + 1
    Xtr, Xva, ytr, yva = dev.X.iloc[tr], dev.X.iloc[va], dev.y[tr], dev.y[va]
    rows: list[dict[str, Any]] = []
    params: list[dict[str, Any]] = []
    preds: list[dict[str, Any]] = []
    key = {"block": k + 1, "repeat": repeat, "fold": fold}
    for name in models:
        t0 = time.perf_counter()
        tm = tune_and_calibrate(name, Xtr, ytr, model_seed(cfg.project.seed, k, name), cfg)
        fit_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        p = tm.calibrated.predict_proba(Xva)[:, 1]
        pred_s = time.perf_counter() - t1
        rows.append(
            {
                **key,
                "model": name,
                "fit_s": fit_s,
                "predict_s": pred_s,
                "calibrated": True,
                **classification_metrics(
                    yva, p, 0.5, cfg.evaluation.ece_bins, cfg.evaluation.ece_strategy
                ),
            }
        )
        params.append(
            {
                **key,
                "model": name,
                "inner_score": tm.inner_score,
                "best_params": json.dumps(tm.best_params, sort_keys=True, default=str),
                "calibration": cfg.cv.calibration_method,
            }
        )
        preds.extend(
            {
                **key,
                "model": name,
                "source_row_id": int(dev.ids[i]),
                "observed": int(dev.y[i]),
                "probability": float(q),
            }
            for i, q in zip(va, p)
        )
    return {"rows": rows, "params": params, "preds": preds, "models": models}


def run_benchmark(ctx: RunContext) -> pd.DataFrame:
    cfg = ctx.cfg
    dev = load_development(ctx)
    models, skipped = resolve_models(cfg)
    ctx.save_json({"usable": models, "skipped": skipped}, "benchmark_models")

    splits = list(outer_splitter(cfg).split(dev.X, dev.y))
    manifest = []
    for k, (tr, va) in enumerate(splits):
        repeat, fold = k // cfg.cv.outer_splits + 1, k % cfg.cv.outer_splits + 1
        manifest += [
            {
                "block": k + 1,
                "repeat": repeat,
                "fold": fold,
                "source_row_id": int(dev.ids[i]),
                "role": "outer_train",
            }
            for i in tr
        ]
        manifest += [
            {
                "block": k + 1,
                "repeat": repeat,
                "fold": fold,
                "source_row_id": int(dev.ids[i]),
                "role": "outer_validation",
            }
            for i in va
        ]
    ctx.save_table(pd.DataFrame(manifest), "nested_repeated_cv_fold_manifest")

    ckpt_dir = ctx.checkpoints / "benchmark"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def ckpt(k: int) -> Path:
        return ckpt_dir / f"block_{k + 1:03d}.joblib"

    results: dict[int, dict] = {}
    if cfg.runtime.checkpoint:
        for k in range(len(splits)):
            if ckpt(k).exists():
                res = joblib.load(ckpt(k))
                if res.get("models") == models:
                    results[k] = res
        if results:
            log.info(
                "Resuming: %d/%d outer blocks loaded from checkpoints", len(results), len(splits)
            )
    todo = [k for k in range(len(splits)) if k not in results]
    log.info(
        "Nested CV: %d blocks × %d models (%d pending, n_jobs=%s)",
        len(splits),
        len(models),
        len(todo),
        cfg.runtime.n_jobs,
    )

    def work(k: int) -> tuple[int, dict]:
        res = _run_block(k, splits[k][0], splits[k][1], dev, models, cfg)
        if cfg.runtime.checkpoint:
            joblib.dump(res, ckpt(k))
        return k, res

    done = 0
    for k, res in Parallel(n_jobs=cfg.runtime.n_jobs, return_as="generator_unordered")(
        delayed(work)(k) for k in todo
    ):
        results[k] = res
        done += 1
        log.info(
            "  outer block %d/%d done (%d/%d pending finished)", k + 1, len(splits), done, len(todo)
        )

    order = sorted(results)
    fold_df = pd.DataFrame([r for k in order for r in results[k]["rows"]])
    param_df = pd.DataFrame([r for k in order for r in results[k]["params"]])
    oof = pd.DataFrame([r for k in order for r in results[k]["preds"]])

    # Integrity gates
    counts = oof.groupby(["block", "model"]).size().unstack()
    assert counts.nunique(axis=1).eq(1).all(), "Models did not share validation rows within a block"
    assert oof.groupby("model").size().eq(len(dev) * cfg.cv.outer_repeats).all(), (
        "Incomplete OOF predictions"
    )
    assert oof["probability"].between(0, 1).all(), "Probabilities outside [0, 1]"

    ctx.save_table(fold_df, "nested_repeated_cv_by_fold")
    ctx.save_table(param_df, "nested_repeated_cv_params")
    ctx.save_table(oof, "nested_repeated_cv_predictions")
    return fold_df
