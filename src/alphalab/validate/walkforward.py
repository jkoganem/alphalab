"""Walk-forward validation for backtesting.

Implements walk-forward testing with rolling windows for realistic
out-of-sample performance evaluation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class WalkForward:
    """Walk-forward validation with rolling windows.

    Divides time series into sequential train/test windows with model
    retraining at each step.

    Parameters
    ----------
    n_folds : int, default 6
        Number of walk-forward folds
    train_period_days : int | None, default None
        Fixed training window (if None, uses expanding window)
    test_period_days : int, default 60
        Test period length in days
    embargo_days : int, default 5
        Embargo period between train and test
    step_days : int | None, default None
        Step size between folds (if None, uses test_period_days)

    Examples
    --------
    >>> wf = WalkForward(n_folds=6, train_period_days=252, test_period_days=60)
    >>> for train_idx, test_idx in wf.split(data):
    ...     # train and test
    ...     pass
    """

    def __init__(
        self,
        n_folds: int = 6,
        train_period_days: int | None = None,
        test_period_days: int = 60,
        embargo_days: int = 5,
        step_days: int | None = None,
    ) -> None:
        """Initialize Walk-Forward validator."""
        if n_folds < 2:
            msg = f"n_folds must be >= 2, got {n_folds}"
            raise ValueError(msg)

        self.n_folds = n_folds
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.embargo_days = embargo_days
        self.step_days = step_days if step_days is not None else test_period_days

        self.window_type = "rolling" if train_period_days is not None else "expanding"

        logger.info(
            f"Initialized WalkForward with n_folds={n_folds}, "
            f"window_type={self.window_type}, train_days={train_period_days}, "
            f"test_days={test_period_days}"
        )

    def split(
        self, data: pd.DataFrame, min_train_days: int = 126
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test splits.

        Parameters
        ----------
        data : pd.DataFrame
            Data with DatetimeIndex or MultiIndex with date level
        min_train_days : int, default 126
            Minimum training period (for expanding window)

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray]]
            List of (train_indices, test_indices) tuples
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

        logger.info(f"Creating {self.n_folds} walk-forward windows from {n_dates} dates")

        splits = []

        # Starting point for first test window
        if self.window_type == "expanding":
            # Start with minimum training period
            current_date_idx = min_train_days
        else:
            # Start with fixed training window
            current_date_idx = self.train_period_days if self.train_period_days else min_train_days

        fold = 0
        while fold < self.n_folds and current_date_idx < n_dates:
            # Define test window
            test_start_idx = current_date_idx + self.embargo_days
            test_end_idx = min(test_start_idx + self.test_period_days, n_dates)

            if test_end_idx - test_start_idx < 10:  # Need minimum test size
                break

            # Define training window
            if self.window_type == "rolling":
                # Fixed size rolling window
                train_start_idx = max(
                    0, test_start_idx - self.embargo_days - (self.train_period_days or 0)
                )
            else:
                # Expanding window (use all data up to embargo)
                train_start_idx = 0

            train_end_idx = current_date_idx

            # Get date ranges
            train_dates = unique_dates[train_start_idx:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            # Convert to indices
            train_mask = dates.isin(train_dates)
            test_mask = dates.isin(test_dates)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                logger.info(
                    f"Fold {fold + 1}/{self.n_folds}: "
                    f"Train={len(train_indices)} ({train_dates.min()} to {train_dates.max()}), "
                    f"Test={len(test_indices)} ({test_dates.min()} to {test_dates.max()})"
                )

                splits.append((train_indices, test_indices))
                fold += 1

            # Move to next fold
            current_date_idx += self.step_days

        logger.info(f"Created {len(splits)} walk-forward folds")

        return splits


def walk_forward_backtest(
    backtest_fn: Callable[[pd.DataFrame, pd.DataFrame], dict[str, object]],
    data: pd.DataFrame,
    wf: WalkForward,
) -> dict[str, object]:
    """Run walk-forward backtest.

    Parameters
    ----------
    backtest_fn : Callable
        Function that takes (train_data, test_data) and returns backtest results
    data : pd.DataFrame
        Full dataset
    wf : WalkForward
        Walk-forward validator

    Returns
    -------
    dict[str, object]
        Aggregated results across all folds
    """
    logger.info("Running walk-forward backtest")

    fold_results = []

    for fold_num, (train_idx, test_idx) in enumerate(wf.split(data)):
        logger.info(f"Processing fold {fold_num + 1}")

        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]

        # Run backtest on this fold
        try:
            results = backtest_fn(train_data, test_data)
            fold_results.append(results)

        except Exception as e:
            logger.error(f"Error in fold {fold_num + 1}: {e}")
            continue

    # Aggregate results
    if not fold_results:
        logger.warning("No successful fold results")
        return {"fold_results": [], "n_folds": 0}

    # Combine equity curves
    all_equity_curves = []
    for results in fold_results:
        if "equity_curve" in results:
            all_equity_curves.append(results["equity_curve"])

    if all_equity_curves:
        combined_equity = pd.concat(all_equity_curves).sort_index()
    else:
        combined_equity = pd.Series(dtype=float)

    # Aggregate metrics
    aggregated = {
        "fold_results": fold_results,
        "n_folds": len(fold_results),
        "combined_equity_curve": combined_equity,
    }

    # Calculate aggregate statistics
    if fold_results and "returns" in fold_results[0]:
        all_returns = pd.concat([r["returns"] for r in fold_results if "returns" in r])
        aggregated["combined_returns"] = all_returns

        # Calculate overall metrics
        total_return = (1 + all_returns).prod() - 1
        sharpe = all_returns.mean() / all_returns.std() * np.sqrt(252) if all_returns.std() > 0 else 0

        aggregated["total_return"] = total_return
        aggregated["sharpe_ratio"] = sharpe

        logger.info(
            f"Walk-forward complete: Total return={total_return:.2%}, "
            f"Sharpe={sharpe:.2f}"
        )

    return aggregated


def analyze_walk_forward_stability(results: dict[str, object]) -> pd.DataFrame:
    """Analyze stability of performance across walk-forward folds.

    Parameters
    ----------
    results : dict[str, object]
        Results from walk_forward_backtest

    Returns
    -------
    pd.DataFrame
        Stability analysis showing metrics by fold
    """
    fold_results = results.get("fold_results", [])

    if not fold_results:
        return pd.DataFrame()

    stability_data = []

    for i, fold in enumerate(fold_results):
        fold_metrics = {
            "fold": i + 1,
        }

        # Extract key metrics
        if "returns" in fold:
            returns = fold["returns"]
            fold_metrics["return"] = (1 + returns).prod() - 1
            fold_metrics["volatility"] = returns.std() * np.sqrt(252)
            fold_metrics["sharpe"] = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )

        if "equity_curve" in fold:
            equity = fold["equity_curve"]
            # Drawdown
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            fold_metrics["max_drawdown"] = drawdown.min()

        stability_data.append(fold_metrics)

    stability_df = pd.DataFrame(stability_data)

    logger.info("\nWalk-Forward Stability Analysis:")
    logger.info(f"\n{stability_df.to_string()}")

    # Calculate stability metrics
    if "sharpe" in stability_df.columns:
        sharpe_std = stability_df["sharpe"].std()
        logger.info(f"\nSharpe std across folds: {sharpe_std:.3f}")

    return stability_df
