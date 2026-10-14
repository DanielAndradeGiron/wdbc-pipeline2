"""Post hoc explainability: permutation importance (always), SHAP and
fidelity-gated LIME (optional extras: ``pip install -e .[explain]``)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import plotting
from .artifacts import RunContext
from .data import load_development
from .evaluation import _holdout_predictions
from .final import load_final
from .logging_utils import get_logger

log = get_logger("explain")


def _label(c: str) -> str:
    return c.replace("_", " ")


def run_explain(ctx: RunContext) -> dict[str, str]:
    cfg = ctx.cfg.explain
    seed = ctx.cfg.project.seed
    status = {"permutation": "not_run", "SHAP": "not_run", "LIME": "not_run"}
    if not cfg.enabled:
        ctx.save_json(status, "explainability_status")
        return status
    model, _ = load_final(ctx)
    dev = load_development(ctx)
    hold, p_hold, _ = _holdout_predictions(ctx)

    # 1) Model-agnostic permutation importance (no extra dependency).
    perm = permutation_importance(
        model,
        hold.X,
        hold.y,
        scoring="roc_auc",
        n_repeats=cfg.permutation_repeats,
        random_state=seed,
        n_jobs=1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": [_label(c) for c in hold.X.columns],
            "importance_mean": perm.importances_mean,
            "importance_sd": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    ctx.save_table(perm_df, "table_09a_permutation_importance_auc")
    ctx.save_figure(
        plotting.importance_bar(
            perm_df,
            "importance_mean",
            "importance_sd",
            "Permutation importance on the internal holdout",
            "Mean decrease in ROC-AUC",
        ),
        "figure_07a_permutation_importance",
    )
    status["permutation"] = "valid_posthoc_global_explanation"

    def predict_pos(z: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(z, columns=dev.X.columns))[:, 1]

    # 2) SHAP (permutation explainer, model-agnostic).
    if cfg.shap and importlib.util.find_spec("shap") is not None:
        try:
            import matplotlib.pyplot as plt
            import shap

            background = dev.X.sample(min(cfg.shap_background, len(dev.X)), random_state=seed)
            explained = hold.X.sample(min(cfg.shap_explained, len(hold.X)), random_state=seed + 1)
            explainer = shap.Explainer(predict_pos, background, algorithm="permutation", seed=seed)
            sv = explainer(explained)
            values = np.asarray(sv.values)
            labels = [_label(c) for c in explained.columns]
            shap.summary_plot(values, explained, feature_names=labels, show=False, max_display=20)
            fig = plt.gcf()
            fig.set_size_inches(8.2, 7.0)
            plt.title(
                "Global SHAP summary for malignant-class probability", fontweight="bold", pad=12
            )
            ctx.save_figure(fig, "figure_07_shap_global_summary")
            shap_df = pd.DataFrame(
                {"feature": labels, "mean_abs_shap": np.abs(values).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False)
            ctx.save_table(shap_df, "table_09_shap_global_importance")
            np.save(ctx.metadata / "shap_values.npy", values)
            status["SHAP"] = "valid_posthoc_global_explanation"
        except Exception as exc:
            status["SHAP"] = f"failed: {exc!r}"
            log.warning("SHAP failed: %r", exc)
    elif cfg.shap:
        status["SHAP"] = "skipped_package_not_installed"

    # 3) LIME on a rule-chosen error case, reported only if local fidelity is adequate.
    if cfg.lime and importlib.util.find_spec("lime") is not None:
        try:
            from lime.lime_tabular import LimeTabularExplainer

            if not all(pd.api.types.is_numeric_dtype(dev.X[c]) for c in dev.X):
                status["LIME"] = "omitted_categorical_predictors_require_separate_protocol"
            else:
                malignant = np.flatnonzero(hold.y == 1)
                fneg = malignant[p_hold[malignant] < 0.5]
                idx = int(fneg[0]) if len(fneg) else int(malignant[np.argmin(p_hold[malignant])])
                explainer = LimeTabularExplainer(
                    dev.X.to_numpy(),
                    feature_names=[_label(c) for c in dev.X.columns],
                    class_names=["Benign", "Malignant"],
                    mode="classification",
                    random_state=seed,
                    discretize_continuous=True,
                )
                exp = explainer.explain_instance(
                    hold.X.iloc[idx].to_numpy(),
                    lambda z: model.predict_proba(pd.DataFrame(z, columns=dev.X.columns)),
                    num_features=min(15, dev.X.shape[1]),
                    num_samples=cfg.lime_num_samples,
                )
                fid = float(exp.score)
                lime_df = pd.DataFrame(exp.as_list(label=1), columns=["condition", "weight"])
                lime_df["source_row_id"] = int(hold.ids[idx])
                lime_df["observed_class"] = int(hold.y[idx])
                lime_df["predicted_probability_malignant"] = float(p_hold[idx])
                lime_df["local_fidelity_R2"] = fid
                ok = fid >= cfg.lime_min_fidelity
                lime_df["interpretability_status"] = (
                    "retained" if ok else "not_reliable_for_interpretation"
                )
                ctx.save_table(lime_df, "table_10_lime_rule_based_posthoc_error_case")
                if ok:
                    ctx.save_figure(
                        plotting.lime_bar(lime_df, fid), "figure_08_lime_local_explanation"
                    )
                    status["LIME"] = "retained_fidelity_threshold_met"
                else:
                    status["LIME"] = f"not_interpretable_low_fidelity_R2_{fid:.4f}"
                    log.info("LIME figure suppressed: R²=%.4f < %.2f", fid, cfg.lime_min_fidelity)
        except Exception as exc:
            status["LIME"] = f"failed: {exc!r}"
            log.warning("LIME failed: %r", exc)
    elif cfg.lime:
        status["LIME"] = "skipped_package_not_installed"

    ctx.save_json(status, "explainability_status")
    log.info("Explainability: %s", status)
    return status
