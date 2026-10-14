import numpy as np
import pandas as pd

from wdbc_pipeline.config import Tolerance
from wdbc_pipeline.selection import practical_select
from wdbc_pipeline.stats import corrected_paired_ttest, holm_adjust


def test_holm_matches_reference():
    # reference values from statsmodels.multipletests(method="holm")
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm_adjust(p), [0.03, 0.06, 0.06, 0.02])


def test_holm_monotone_and_capped():
    adj = holm_adjust(np.array([0.5, 0.9, 0.7]))
    assert (adj <= 1).all() and adj.max() == 1.0


def test_corrected_ttest_is_more_conservative():
    rng = np.random.default_rng(1)
    d = rng.normal(0.01, 0.02, size=50)
    naive = corrected_paired_ttest(d, test_ratio=0.0)
    nb = corrected_paired_ttest(d, test_ratio=1 / 9)
    assert nb["corrected_se"] > naive["corrected_se"]
    assert nb["p_raw"] > naive["p_raw"]


def test_practical_selection_cascade_and_tiebreak():
    summary = pd.DataFrame(
        {
            "model": ["A", "B", "C"],
            "f1_weighted_mean": [0.960, 0.958, 0.900],
            "mcc_mean": [0.92, 0.92, 0.80],
            "roc_auc_mean": [0.99, 0.99, 0.95],
            "brier_mean": [0.03, 0.03, 0.08],
            "fit_time_mean": [5.0, 1.0, 0.1],
        }
    )
    tols = (
        Tolerance("f1_weighted", 0.005),
        Tolerance("mcc", 0.01),
        Tolerance("roc_auc", 0.005),
        Tolerance("brier", 0.005, "min"),
    )
    chosen, audit = practical_select(summary, tols)
    assert chosen == "B"  # C excluded by tolerance, B faster than A
    assert audit.iloc[0]["retained"] == "A|B"
    assert audit.iloc[-1]["criterion"].startswith("tie_break")


def test_practical_selection_stops_when_unique():
    summary = pd.DataFrame(
        {"model": ["A", "B"], "f1_weighted_mean": [0.99, 0.90], "fit_time_mean": [9.0, 1.0]}
    )
    chosen, audit = practical_select(summary, (Tolerance("f1_weighted", 0.005),))
    assert chosen == "A" and len(audit) == 1


def test_holm_empty():
    assert holm_adjust(np.array([])).size == 0
