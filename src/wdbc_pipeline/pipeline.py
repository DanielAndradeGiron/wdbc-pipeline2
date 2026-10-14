"""Stage orchestration. Each stage reads its inputs from the run directory, so
stages can be (re)run individually, e.g. ``--stages curves explain report``."""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from pathlib import Path

from .artifacts import RunContext
from .config import Config, config_hash
from .logging_utils import get_logger, setup_logging, timed

STAGES: tuple[str, ...] = (
    "data",
    "benchmark",
    "select",
    "stats",
    "threshold",
    "final",
    "holdout",
    "uncertainty",
    "curves",
    "explain",
    "report",
)


def _set_determinism(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def run_pipeline(
    cfg: Config,
    stages: list[str] | None = None,
    run_dir: Path | None = None,
    force_holdout: bool = False,
    project_root: Path | None = None,
    verbose: bool = False,
    make_zip: bool = True,
) -> RunContext:
    from . import benchmark, data, evaluation, explain, final, report, selection, stats, thresholds
    from .plotting import journal_style

    _set_determinism(cfg.project.seed)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    journal_style()

    ctx = RunContext.create(cfg, run_dir)
    log = setup_logging(ctx.logs / "pipeline.log", verbose=verbose)
    selected = list(STAGES) if not stages or stages == ["all"] else stages
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s) {unknown}; valid: {', '.join(STAGES)}")
    selected = [s for s in STAGES if s in selected]  # always canonical order

    log.info("Run directory: %s (config hash %s)", ctx.run_dir, config_hash(cfg))
    if cfg.cv.outer_repeats == 1 and cfg.cv.outer_splits <= 3:
        log.warning("QUICK/STRUCTURAL configuration — do not report these results.")

    actions: dict[str, Callable[[], object]] = {
        "data": lambda: data.run_data_stage(ctx, project_root),
        "benchmark": lambda: benchmark.run_benchmark(ctx),
        "select": lambda: selection.run_selection(ctx),
        "stats": lambda: stats.run_stats(ctx),
        "threshold": lambda: thresholds.run_threshold(ctx),
        "final": lambda: final.run_final(ctx),
        "holdout": lambda: evaluation.run_holdout(ctx, force=force_holdout),
        "uncertainty": lambda: evaluation.run_uncertainty(ctx),
        "curves": lambda: evaluation.run_curves(ctx),
        "explain": lambda: explain.run_explain(ctx),
        "report": lambda: report.run_report(ctx, make_zip=make_zip),
    }
    for stage in selected:
        with timed(f"stage '{stage}'", log):
            actions[stage]()
    get_logger().info("Pipeline finished: %s", ", ".join(selected))
    return ctx
