import numpy as np
import pytest

from wdbc_pipeline.metrics import (
    METRIC_NAMES,
    classification_metrics,
    clopper_pearson,
    expected_calibration_error,
    net_benefit,
)


def test_ece_is_zero_for_perfect_bins():
    p = np.repeat([0.2, 0.8], 50)
    y = np.r_[np.r_[np.ones(10), np.zeros(40)], np.r_[np.ones(40), np.zeros(10)]]
    assert expected_calibration_error(y, p, bins=2, strategy="uniform") == pytest.approx(0.0)


def test_ece_detects_miscalibration():
    y = np.r_[np.zeros(50), np.ones(50)]
    p = np.full(100, 0.9)
    assert expected_calibration_error(y, p, bins=10, strategy="quantile") == pytest.approx(0.4)


def test_classification_metrics_keys_and_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])
    m = classification_metrics(y, p, 0.5)
    assert set(m) == set(METRIC_NAMES)
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5 and m["roc_auc"] == 0.75


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.30850, abs=1e-4)
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.69150, abs=1e-4)


def test_net_benefit_treat_all_formula():
    y = np.array([1, 0, 0, 0])
    nb = net_benefit(y, np.ones(4), np.array([0.2]))
    assert nb.loc[0, "model"] == pytest.approx(0.25 - 0.75 * 0.25)
    assert nb.loc[0, "model"] == pytest.approx(nb.loc[0, "treat_all"])
