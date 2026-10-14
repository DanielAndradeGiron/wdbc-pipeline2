import numpy as np

from wdbc_pipeline.thresholds import select_threshold, threshold_grid, threshold_table


def test_grid_is_inclusive_and_clean():
    g = threshold_grid(0.05, 0.95, 0.01)
    assert g[0] == 0.05 and g[-1] == 0.95 and len(g) == 91


def test_rule_maximises_specificity_under_sensitivity_constraint():
    y = np.array([0] * 6 + [1] * 4)
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.4, 0.7, 0.8, 0.9])
    table = threshold_table(y, p, threshold_grid(0.05, 0.95, 0.05))
    thr, rule = select_threshold(table, min_sensitivity=1.0)
    assert rule == "min_sensitivity_then_specificity"
    row = table.set_index("threshold").loc[thr]
    assert row["sensitivity"] == 1.0
    assert row["specificity"] == table[table.sensitivity >= 1.0]["specificity"].max()


def test_fallback_when_constraint_unreachable():
    y = np.array([0, 1])
    p = np.array([0.9, 0.01])
    table = threshold_table(y, p, threshold_grid(0.05, 0.95, 0.05))
    _, rule = select_threshold(table, min_sensitivity=1.0)
    assert rule == "fallback_max_sensitivity"
