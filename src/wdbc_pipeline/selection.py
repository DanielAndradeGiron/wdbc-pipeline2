"""Model ranking and practical-tolerance selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import plotting
from .artifacts import RunContext
from .config import Tolerance
from .logging_utils import get_logger

log = get_logger("selection")

_SUMMARY_METRICS = (
    "accuracy",
    "f1_weighted",
    "mcc",
    "roc_auc",
    "average_precision",
    "brier",
    "ece",
    "sensitivity",
    "specificity",
)


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    agg: dict[str, tuple[str, str]] = {}
    for m in _SUMMARY_METRICS:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_sd"] = (m, "std")
    agg["fit_time_mean"] = ("fit_s", "mean")
    agg["n_blocks"] = ("block", "nunique")
    return fold_df.groupby("model").agg(**agg).reset_index()


def practical_select(
    summary: pd.DataFrame, tolerances: tuple[Tolerance, ...]
) -> tuple[str, pd.DataFrame]:
    """Hierarchical tolerance cascade; tie-break: lowest mean fit time, then name."""
    cand = summary.copy()
    audit = []
    for tol in tolerances:
        col = f"{tol.metric}_mean"
        best = cand[col].max() if tol.direction == "max" else cand[col].min()
        keep = (
            cand[col] >= best - tol.tolerance
            if tol.direction == "max"
            else cand[col] <= best + tol.tolerance
        )
        audit.append(
            {
                "criterion": col,
                "direction": tol.direction,
                "best": best,
                "tolerance": tol.tolerance,
                "before": len(cand),
                "after": int(keep.sum()),
                "retained": "|".join(cand.loc[keep, "model"]),
            }
        )
        cand = cand.loc[keep]
        if len(cand) == 1:
            break
    tie = cand.sort_values(["fit_time_mean", "model"], kind="mergesort")
    chosen = str(tie.iloc[0]["model"])
    if len(cand) > 1:
        audit.append(
            {
                "criterion": "tie_break: fit_time_mean, model",
                "direction": "min",
                "best": float(tie.iloc[0]["fit_time_mean"]),
                "tolerance": np.nan,
                "before": len(cand),
                "after": 1,
                "retained": chosen,
            }
        )
    return chosen, pd.DataFrame(audit)


def manuscript_table(summary: pd.DataFrame) -> pd.DataFrame:
    def ms(r: pd.Series, m: str) -> str:
        return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_sd']:.4f}"

    out = pd.DataFrame(
        {
            "descriptive_rank": summary["descriptive_rank"],
            "model": summary["model"],
            "Accuracy": summary.apply(ms, axis=1, m="accuracy"),
            "F1_weighted": summary.apply(ms, axis=1, m="f1_weighted"),
            "MCC": summary.apply(ms, axis=1, m="mcc"),
            "ROC_AUC": summary.apply(ms, axis=1, m="roc_auc"),
            "Average_precision": summary.apply(ms, axis=1, m="average_precision"),
            "Brier": summary["brier_mean"].round(4),
            "ECE": summary["ece_mean"].round(4),
            "Fit_time_s": summary["fit_time_mean"].round(2),
            "selected": summary["selected"],
        }
    )
    return out


def run_selection(ctx: RunContext) -> str:
    cfg = ctx.cfg
    fold_df = ctx.load_table("nested_repeated_cv_by_fold")
    summary = summarize(fold_df)
    chosen, audit = practical_select(summary, cfg.selection.tolerances)
    summary["selected"] = summary["model"].eq(chosen)
    summary = summary.sort_values("f1_weighted_mean", ascending=False).reset_index(drop=True)
    summary.insert(0, "descriptive_rank", np.arange(1, len(summary) + 1))

    ctx.save_table(summary, "table_01_repeated_nested_cv_model_summary")
    ctx.save_table(audit, "table_02_practical_tolerance_audit")
    ctx.save_table(manuscript_table(summary), "manuscript_table_model_comparison")

    top_f1 = summary.iloc[0]["model"]
    note = None
    if top_f1 != chosen:
        note = (
            f"{chosen} was selected over {top_f1} (highest mean weighted F1) because both fell "
            "within every practical tolerance and the tie-break favours lower computational cost. "
            "Report this explicitly; tolerances are prespecified, not post hoc."
        )
        log.warning(note)
    ctx.save_json(
        {
            "selected_model": chosen,
            "highest_mean_f1_weighted": top_f1,
            "tolerances": [t.__dict__ for t in cfg.selection.tolerances],
            "tie_break": "lowest mean search-and-fit time, then model name",
            "note": note,
        },
        "selection_rule",
    )

    f1_tol = next((t.tolerance for t in cfg.selection.tolerances if t.metric == "f1_weighted"), 0.0)
    ctx.save_figure(
        plotting.benchmark_bar(summary, chosen, f1_tol), "figure_01_nested_cv_weighted_f1"
    )
    log.info("Selected model: %s (retained by cascade: %s)", chosen, audit.iloc[-1]["retained"])
    return chosen
