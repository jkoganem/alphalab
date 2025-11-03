"""Purged K-Fold cross-validation for time series.

Implements the purged K-fold cross-validation method from López de Prado's
"Advances in Financial Machine Learning" to prevent temporal leakage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PurgedKFold:
    """Purged K-Fold cross-validation with embargo.

    Creates K-fold splits for time series data with:
    1. Purging: Removes observations from train set that are too close to test set
    2. Embargo: Additional buffer period after test set to prevent leakage

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds
    embargo_days : int, default 5
        Number of days to embargo after each test fold
    purge_days : int, default 2
        Number of days before test fold to purge from training

    Examples
    --------
    >>> pkf = PurgedKFold(n_splits=5, embargo_days=5)
    >>> for train_idx, test_idx in pkf.split(data):
    ...     # train on train_idx, test on test_idx
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_days: int = 5,
        purge_days: int = 2,
    ) -> None:
        """Initialize Purged K-Fold."""
        if n_splits < 2:
            msg = f"n_splits must be >= 2, got {n_splits}"
            raise ValueError(msg)

        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.purge_days = purge_days

        logger.info(
            f"Initialized PurgedKFold with n_splits={n_splits}, "
            f"embargo={embargo_days}, purge={purge_days}"
        )

    def split(
        self, data: pd.DataFrame, groups: pd.Series | None = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits.

        Parameters
        ----------
        data : pd.DataFrame
            Data with DatetimeIndex or MultiIndex with date level
        groups : pd.Series | None, optional
            Not used, for sklearn compatibility

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples

        Yields
        ------
        tuple[np.ndarray, np.ndarray]
            Train and test indices for each fold
        """
        # Get dates
        if isinstance(data.index, pd.MultiIndex):
            dates = data.index.get_level_values("date")
        elif isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            msg = "Data must have DatetimeIndex or MultiIndex with 'date' level"
            raise ValueError(msg)

        unique_dates = dates.unique().sort_values()
        n_dates = len(unique_dates)

        # Calculate fold size
        fold_size = n_dates // self.n_splits

        logger.info(
            f"Splitting {n_dates} dates into {self.n_splits} folds "
            f"of ~{fold_size} days each"
        )

        splits = []

        for fold in range(self.n_splits):
            # Define test set boundaries
            test_start_idx = fold * fold_size
            test_end_idx = min((fold + 1) * fold_size, n_dates)

            # Get test dates
            test_dates = unique_dates[test_start_idx:test_end_idx]

            # Define purge boundaries
            purge_start = test_dates.min() - pd.Timedelta(days=self.purge_days)
            purge_end = test_dates.max() + pd.Timedelta(days=self.embargo_days)

            # Training set: all dates except test and embargo/purge windows
            train_mask = (dates < purge_start) | (dates > purge_end)
            test_mask = dates.isin(test_dates)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            logger.info(
                f"Fold {fold + 1}/{self.n_splits}: "
                f"Train={len(train_indices)}, Test={len(test_indices)}, "
                f"Test period: {test_dates.min()} to {test_dates.max()}"
            )

            splits.append((train_indices, test_indices))

        return splits


def purged_cv_score(
    estimator: object,
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedKFold | int = 5,
    scoring: str = "accuracy",
) -> dict[str, object]:
    """Perform purged cross-validation and return scores.

    Parameters
    ----------
    estimator : object
        Scikit-learn compatible estimator
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    cv : PurgedKFold | int, default 5
        Cross-validation splitter or number of folds
    scoring : str, default "accuracy"
        Scoring metric

    Returns
    -------
    dict[str, object]
        Cross-validation results including fold scores and mean
    """
    if isinstance(cv, int):
        cv = PurgedKFold(n_splits=cv)

    logger.info("Running purged cross-validation")

    fold_scores = []

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Remove NaN
        train_valid = ~(X_train.isna().any(axis=1) | y_train.isna())
        test_valid = ~(X_test.isna().any(axis=1) | y_test.isna())

        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        if len(X_train) == 0 or len(X_test) == 0:
            logger.warning(f"Fold {fold_num + 1}: Empty train or test set, skipping")
            continue

        # Train
        estimator.fit(X_train, y_train)  # type: ignore[attr-defined]

        # Score
        if scoring == "accuracy":
            score = estimator.score(X_test, y_test)  # type: ignore[attr-defined]
        elif scoring == "mse":
            from sklearn.metrics import mean_squared_error

            preds = estimator.predict(X_test)  # type: ignore[attr-defined]
            score = -mean_squared_error(y_test, preds)  # Negative for consistency
        else:
            msg = f"Unknown scoring: {scoring}"
            raise ValueError(msg)

        fold_scores.append(score)
        logger.info(f"Fold {fold_num + 1} score: {score:.4f}")

    results = {
        "fold_scores": fold_scores,
        "mean_score": np.mean(fold_scores) if fold_scores else 0.0,
        "std_score": np.std(fold_scores) if fold_scores else 0.0,
        "n_folds": len(fold_scores),
    }

    logger.info(
        f"Purged CV complete: Mean={results['mean_score']:.4f}, "
        f"Std={results['std_score']:.4f}"
    )

    return results


def check_leakage(
    data: pd.DataFrame, train_indices: np.ndarray, test_indices: np.ndarray
) -> dict[str, bool]:
    """Check for temporal leakage between train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Data with datetime index
    train_indices : np.ndarray
        Training indices
    test_indices : np.ndarray
        Test indices

    Returns
    -------
    dict[str, bool]
        Leakage check results
    """
    # Get dates
    if isinstance(data.index, pd.MultiIndex):
        dates = data.index.get_level_values("date")
    else:
        dates = data.index

    train_dates = dates[train_indices]
    test_dates = dates[test_indices]

    train_min, train_max = train_dates.min(), train_dates.max()
    test_min, test_max = test_dates.min(), test_dates.max()

    # Check for overlaps
    overlap = (train_max >= test_min) and (train_min <= test_max)
    train_after_test = train_min > test_max  # OK
    train_before_test = train_max < test_min  # OK

    results = {
        "has_overlap": overlap,
        "train_dates_ok": train_before_test or train_after_test,
        "train_min": str(train_min),
        "train_max": str(train_max),
        "test_min": str(test_min),
        "test_max": str(test_max),
    }

    if overlap:
        logger.warning(
            "LEAKAGE DETECTED: Train and test sets have overlapping dates!"
        )

    return results
