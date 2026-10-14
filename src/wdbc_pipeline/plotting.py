"""Publication-style figures. All functions return a Matplotlib figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless & deterministic (works in VS Code terminals and CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

PALETTE = {
    "navy": "#17365D",
    "blue": "#2F75B5",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "red": "#C44E52",
    "gray": "#7A7A7A",
    "light": "#D9E6F2",
    "muted": "#B8C2CC",
}


def journal_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#202020",
            "text.color": "#202020",
            "legend.frameon": False,
        }
    )


def benchmark_bar(summary: pd.DataFrame, selected: str, tol: float) -> Figure:
    df = summary.sort_values("f1_weighted_mean")
    best = summary["f1_weighted_mean"].max()
    colors = [
        PALETTE["blue"]
        if m == selected
        else (PALETTE["teal"] if v >= best - tol else PALETTE["muted"])
        for m, v in zip(df["model"], df["f1_weighted_mean"])
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(
        df["model"],
        df["f1_weighted_mean"],
        xerr=df["f1_weighted_sd"],
        color=colors,
        edgecolor="white",
        capsize=2,
    )
    ax.axvline(
        best - tol,
        color=PALETTE["gold"],
        ls="--",
        lw=1.5,
        label=f"Practical tolerance boundary (Δ={tol:.3f})",
    )
    ax.set(
        xlabel="Weighted F1 (mean ± SD)",
        ylabel="",
        title="Repeated nested cross-validation performance",
    )
    ax.set_xlim(max(0, df["f1_weighted_mean"].min() - 0.06), 1.005)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def paired_differences(paired: pd.DataFrame, selected: str, tol: float) -> Figure:
    q = paired.sort_values("mean_difference")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(
        q["mean_difference"],
        q["comparator"],
        xerr=[q["mean_difference"] - q["ci95_low"], q["ci95_high"] - q["mean_difference"]],
        fmt="o",
        color=PALETTE["blue"],
        ecolor=PALETTE["gray"],
        capsize=3,
    )
    ax.axvline(0, color="black", lw=1)
    ax.axvspan(
        -tol, tol, color=PALETTE["gold"], alpha=0.22, label="Weighted-F1 practical tolerance"
    )
    ax.set(
        xlabel=f"Mean paired difference ({selected} − comparator)",
        ylabel="",
        title="Corrected paired differences in weighted F1",
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def confusion_matrices(cms: list[tuple[str, np.ndarray]]) -> Figure:
    fig, axes = plt.subplots(1, len(cms), figsize=(4.1 * len(cms), 3.7))
    axes = np.atleast_1d(axes)
    for ax, (label, cm) in zip(axes, cms):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            square=True,
            ax=ax,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )
        ax.set(title=label, xlabel="Predicted class", ylabel="Observed class")
    fig.suptitle("Internal holdout confusion matrices", fontweight="bold")
    fig.tight_layout()
    return fig


def roc_pr(fpr, tpr, fpr_grid, roc_lo, roc_hi, auc, recall, precision, ap, prevalence) -> Figure:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax[0].plot(fpr, tpr, color=PALETTE["blue"], lw=2, label=f"ROC-AUC = {auc:.3f}")
    ax[0].fill_between(
        fpr_grid, roc_lo, roc_hi, color=PALETTE["blue"], alpha=0.16, label="95% bootstrap band"
    )
    ax[0].plot([0, 1], [0, 1], "--", color=PALETTE["gray"], lw=1)
    ax[0].set(xlabel="1 − Specificity", ylabel="Sensitivity", title="ROC curve")
    ax[0].legend(loc="lower right")
    ax[1].plot(
        recall, precision, color=PALETTE["teal"], lw=2, label=f"Average precision = {ap:.3f}"
    )
    ax[1].axhline(
        prevalence, ls="--", color=PALETTE["gray"], lw=1, label=f"Prevalence = {prevalence:.3f}"
    )
    ax[1].set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision–recall curve")
    ax[1].legend(loc="lower left")
    fig.suptitle("Discrimination on the locked internal holdout", fontweight="bold")
    fig.tight_layout()
    return fig


def calibration(meanp, frac, brier: float, ece: float, n_bins: int) -> Figure:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(meanp, frac, "o-", color=PALETTE["blue"], lw=2, label=f"Frozen model ({n_bins} bins)")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["gray"], label="Ideal calibration")
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed event frequency",
        title=f"Calibration on the internal holdout\nBrier = {brier:.3f}; ECE = {ece:.3f}",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def dca(dca_df: pd.DataFrame, primary: float, secondary: float) -> Figure:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(dca_df["threshold"], dca_df["model"], label="Frozen model", lw=2, color=PALETTE["blue"])
    ax.fill_between(
        dca_df["threshold"],
        dca_df["ci95_low"],
        dca_df["ci95_high"],
        color=PALETTE["blue"],
        alpha=0.16,
        label="95% bootstrap band",
    )
    ax.plot(
        dca_df["threshold"],
        dca_df["treat_all"],
        "--",
        color=PALETTE["red"],
        label="Intervene in all",
    )
    ax.plot(
        dca_df["threshold"], dca_df["treat_none"], ":", color="black", label="Intervene in none"
    )
    ax.axvline(
        primary, color=PALETTE["gray"], alpha=0.7, label=f"Primary threshold ({primary:.2f})"
    )
    ax.axvline(
        secondary, color=PALETTE["teal"], alpha=0.9, label=f"Secondary threshold ({secondary:.2f})"
    )
    lo = min(dca_df["model"].min(), dca_df["ci95_low"].min(), 0) - 0.02
    ax.set_ylim(lo, max(dca_df["treat_all"].max(), dca_df["ci95_high"].max()) + 0.03)
    ax.set(
        xlabel="Threshold probability",
        ylabel="Net benefit",
        title="Exploratory decision-curve analysis",
    )
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def importance_bar(
    df: pd.DataFrame, value: str, err: str | None, title: str, xlabel: str, top: int = 20
) -> Figure:
    q = df.nlargest(top, value).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 0.3 * len(q) + 1.6))
    ax.barh(
        q["feature"],
        q[value],
        xerr=None if err is None else q[err],
        color=PALETTE["blue"],
        ecolor=PALETTE["gray"],
        capsize=2,
    )
    ax.set(xlabel=xlabel, ylabel="", title=title)
    fig.tight_layout()
    return fig


def lime_bar(lime_df: pd.DataFrame, fidelity: float) -> Figure:
    q = lime_df.sort_values("weight")
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.barh(
        q["condition"],
        q["weight"],
        color=[PALETTE["teal"] if x > 0 else PALETTE["red"] for x in q["weight"]],
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Local surrogate weight",
        ylabel="",
        title=f"LIME explanation of rule-based post hoc error case (R²={fidelity:.3f})",
    )
    fig.tight_layout()
    return fig
