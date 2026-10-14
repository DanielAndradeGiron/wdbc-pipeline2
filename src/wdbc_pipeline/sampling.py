"""Fold-local random oversampling as a scikit-learn meta-estimator.

``RandomOverSampledClassifier`` reproduces ``imblearn``'s
``RandomOverSampler(sampling_strategy="auto")`` followed by a classifier:
minority classes are resampled *with replacement* up to the majority count,
but **only inside ``fit``**. Predictions are never resampled, so it is safe to
nest inside ``Pipeline``, ``RandomizedSearchCV``, ``CalibratedClassifierCV``,
``VotingClassifier`` and ``StackingClassifier``: every training fold is
resampled independently and validation folds are untouched.

Implementing it natively removes the ``imbalanced-learn`` dependency, whose
releases frequently lag behind scikit-learn and break installs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted


def random_oversample_indices(y: np.ndarray, random_state: Any = None) -> np.ndarray:
    """Indices that balance every class to the majority count (with replacement)."""
    rng = check_random_state(random_state)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = [np.arange(len(y))]
    for cls, n in zip(classes, counts):
        if n < target:
            pool = np.flatnonzero(y == cls)
            idx.append(rng.choice(pool, size=target - n, replace=True))
    return np.concatenate(idx)


def _take(X: Any, idx: np.ndarray) -> Any:
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]


class RandomOverSampledClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` on a randomly oversampled copy of the training data."""

    def __init__(self, estimator: Any, random_state: Any = None) -> None:
        self.estimator = estimator
        self.random_state = random_state

    def fit(self, X: Any, y: Any, **fit_params: Any) -> RandomOverSampledClassifier:
        y = np.asarray(y)
        idx = random_oversample_indices(y, self.random_state)
        self.estimator_ = clone(self.estimator).fit(_take(X, idx), y[idx], **fit_params)
        self.classes_ = self.estimator_.classes_
        self.n_resampled_ = len(idx)
        if hasattr(self.estimator_, "n_features_in_"):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "estimator_")
        return self.estimator_.feature_importances_
