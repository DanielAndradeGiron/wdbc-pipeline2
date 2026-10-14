"""Dependence-aware statistical comparison of candidates.

* Friedman test across repeated-CV blocks — *exploratory*: blocks share
  training data, so the independence assumption is violated.
* Nadeau–Bengio corrected resampled paired t-test (selected vs each
  comparator) with variance inflation ``1/J + n_test/n_train``.
* Holm step-down adjustment for multiplicity.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import t as student_t

from . import plotting
from .artifacts import RunContext
from .logging_utils import get_logger

log = get_logger("stats")


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def corrected_paired_ttest(
    diff: np.ndarray, test_ratio: float, alpha: float = 0.05
) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test on paired differences."""
    d = np.asarray(diff, dtype=float)
    n = len(d)
    mean, var = d.mean(), d.var(ddof=1)
    se = math.sqrt((1 / n + test_ratio) * var)
    tval = mean / se if se > 0 else np.nan
    p = 2 * student_t.sf(abs(tval), df=n - 1) if np.isfinite(tval) else 1.0
    q = student_t.ppf(1 - alpha / 2, n - 1)
    return {
        "mean_difference": mean,
        "corrected_se": se,
        "ci95_low": mean - q * se,
        "ci95_high": mean + q * se,
        "t_value": tval,
        "p_raw": p,
    }


def friedman(pivot: pd.DataFrame) -> tuple[float, float]:
    stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    return float(stat), float(p)


def run_stats(ctx: RunContext) -> None:
    cfg = ctx.cfg
    metric = cfg.stats.metric
    fold_df = ctx.load_table("nested_repeated_cv_by_fold")
    selected = ctx.load_json("selection_rule")["selected_model"]
    pivot = fold_df.pivot(index="block", columns="model", values=metric).dropna(axis=1)

    if pivot.shape[1] >= 3:
        stat, p_global = friedman(pivot)
    else:
        stat, p_global = float("nan"), float("nan")
        log.warning("Friedman test needs ≥3 models; skipped.")
    ctx.save_table(
        pd.DataFrame(
            [
                {
                    "metric": metric,
                    "chi_square": stat,
                    "p_value": p_global,
                    "blocks": len(pivot),
                    "models": pivot.shape[1],
                    "interpretation": "Exploratory: repeated-CV blocks are not independent; "
                    "non-significance does not establish equivalence.",
                }
            ]
        ),
        "table_03_friedman_repeated_cv",
    )

    test_ratio = 1 / (cfg.cv.outer_splits - 1)
    rows = []
    for name in pivot.columns:
        if name == selected:
            continue
        res = corrected_paired_ttest(
            (pivot[selected] - pivot[name]).to_numpy(), test_ratio, cfg.stats.alpha
        )
        rows.append({"selected_model": selected, "comparator": name, **res})
    paired = pd.DataFrame(rows)
    if len(paired):
        paired["p_holm"] = holm_adjust(paired["p_raw"].to_numpy())
        paired["reject_holm"] = paired["p_holm"] < cfg.stats.alpha
        f1_tol = next((t.tolerance for t in cfg.selection.tolerances if t.metric == metric), 0.0)
        ctx.save_figure(
            plotting.paired_differences(paired, selected, f1_tol),
            "figure_02_corrected_paired_differences",
        )
    ctx.save_table(paired, "table_04_nadeau_bengio_holm")
    ctx.save_json(
        {
            "friedman_chi_square": stat,
            "friedman_p": p_global,
            "n_significant_holm": int(paired["reject_holm"].sum()) if len(paired) else 0,
        },
        "stats_summary",
    )
    log.info(
        "Friedman χ²=%.3f p=%.4g; Holm-significant comparisons: %s",
        stat,
        p_global,
        int(paired["reject_holm"].sum()) if len(paired) else 0,
    )
