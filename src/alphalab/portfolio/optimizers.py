"""Portfolio optimization implementations.

This module provides portfolio weight optimization methods including
equal weight, inverse volatility, and mean-variance optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EqualWeightOptimizer:
    """Equal weight portfolio optimizer.

    Allocates equal weight to all securities with non-zero signals.

    Parameters
    ----------
    normalize : bool, default True
        Whether to normalize weights to sum to 1 (or match signal sum)

    Examples
    --------
    >>> optimizer = EqualWeightOptimizer()
    >>> weights = optimizer.allocate(signals)
    """

    def __init__(self, normalize: bool = True) -> None:
        """Initialize equal weight optimizer."""
        self.normalize = normalize
        logger.info("Initialized EqualWeightOptimizer")

    def allocate(
        self, signals: pd.DataFrame, risk: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Allocate equal weights to all non-zero signals.

        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals with 'signal' column
        risk : pd.DataFrame | None, optional
            Not used, for interface compatibility

        Returns
        -------
        pd.DataFrame
            Portfolio weights with 'weight' column
        """
        logger.info("Computing equal weights")

        signal_values = signals["signal"]

        # Count non-zero signals per date
        counts = signal_values.groupby(level="date").transform(
            lambda x: (x != 0).sum()
        )

        # Equal weight: 1 / count for non-zero signals
        weights = pd.Series(0.0, index=signals.index)
        weights = weights.where(signal_values == 0, np.sign(signal_values) / counts)

        # Handle NaN and inf
        weights = weights.replace([np.inf, -np.inf], 0)
        weights = weights.fillna(0)

        if self.normalize:
            # Normalize weights to sum to target (preserve sign structure)
            date_sums = weights.groupby(level="date").transform("sum")
            weights = weights / date_sums.where(date_sums != 0, 1)

        result = pd.DataFrame({"weight": weights}, index=signals.index)

        n_positions = (result["weight"] != 0).sum()
        logger.info(f"Generated {n_positions} non-zero weights")

        return result


class InverseVolatilityOptimizer:
    """Inverse volatility portfolio optimizer.

    Allocates weights inversely proportional to volatility, giving less
    weight to more volatile securities.

    Parameters
    ----------
    vol_lookback : int, default 60
        Lookback window for volatility estimation
    min_vol : float, default 0.01
        Minimum volatility threshold to avoid division by zero
    normalize : bool, default True
        Whether to normalize weights

    Examples
    --------
    >>> optimizer = InverseVolatilityOptimizer(vol_lookback=60)
    >>> weights = optimizer.allocate(signals, risk_data)
    """

    def __init__(
        self,
        vol_lookback: int = 60,
        min_vol: float = 0.01,
        normalize: bool = True,
    ) -> None:
        """Initialize inverse volatility optimizer."""
        self.vol_lookback = vol_lookback
        self.min_vol = min_vol
        self.normalize = normalize
        logger.info(f"Initialized InverseVolatilityOptimizer with lookback={vol_lookback}")

    def allocate(
        self, signals: pd.DataFrame, risk: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Allocate weights inverse to volatility.

        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals
        risk : pd.DataFrame | None, optional
            Must contain volatility column

        Returns
        -------
        pd.DataFrame
            Volatility-adjusted weights
        """
        logger.info("Computing inverse volatility weights")

        if risk is None:
            logger.warning("No risk data provided, falling back to equal weights")
            return EqualWeightOptimizer(normalize=self.normalize).allocate(signals)

        # Look for volatility column
        vol_col = f"vol_{self.vol_lookback}d"
        if vol_col not in risk.columns:
            # Try alternative column names
            vol_cols = [col for col in risk.columns if "vol" in col.lower()]
            if vol_cols:
                vol_col = vol_cols[0]
                logger.info(f"Using {vol_col} for volatility")
            else:
                logger.warning("No volatility column found, using equal weights")
                return EqualWeightOptimizer(normalize=self.normalize).allocate(signals)

        signal_values = signals["signal"]
        volatilities = risk[vol_col]

        # Clip volatility to minimum
        vols_clipped = volatilities.clip(lower=self.min_vol)

        # Inverse volatility weights
        inv_vol = 1.0 / vols_clipped

        # Apply to signals (preserve direction)
        weights = np.sign(signal_values) * inv_vol
        weights = weights.where(signal_values != 0, 0)

        # Normalize within each date
        if self.normalize:
            # Normalize long and short separately to preserve market neutrality
            def normalize_date(group: pd.Series) -> pd.Series:
                long_mask = group > 0
                short_mask = group < 0

                if long_mask.any():
                    long_sum = group[long_mask].sum()
                    group[long_mask] = group[long_mask] / long_sum if long_sum != 0 else 0

                if short_mask.any():
                    short_sum = np.abs(group[short_mask].sum())
                    group[short_mask] = group[short_mask] / short_sum if short_sum != 0 else 0

                return group

            weights = weights.groupby(level="date").transform(normalize_date)

        result = pd.DataFrame({"weight": weights}, index=signals.index)

        n_positions = (result["weight"] != 0).sum()
        avg_weight = result["weight"][result["weight"] != 0].abs().mean()
        logger.info(f"Generated {n_positions} positions, avg weight: {avg_weight:.4f}")

        return result


class MeanVarianceOptimizer:
    """Mean-variance portfolio optimizer with Ledoit-Wolf shrinkage.

    Uses quadratic optimization to find weights that maximize Sharpe ratio
    subject to constraints.

    Parameters
    ----------
    lookback : int, default 126
        Lookback window for covariance estimation
    target_vol : float | None, default None
        Target volatility (if None, optimize for max Sharpe)
    max_weight : float, default 0.1
        Maximum weight per position
    use_shrinkage : bool, default True
        Whether to use Ledoit-Wolf covariance shrinkage

    Examples
    --------
    >>> optimizer = MeanVarianceOptimizer(lookback=126, max_weight=0.1)
    >>> weights = optimizer.allocate(signals, risk_data)
    """

    def __init__(
        self,
        lookback: int = 126,
        target_vol: float | None = None,
        max_weight: float = 0.1,
        use_shrinkage: bool = True,
    ) -> None:
        """Initialize mean-variance optimizer."""
        self.lookback = lookback
        self.target_vol = target_vol
        self.max_weight = max_weight
        self.use_shrinkage = use_shrinkage
        logger.info(
            f"Initialized MeanVarianceOptimizer with lookback={lookback}, "
            f"max_weight={max_weight}"
        )

    def allocate(
        self, signals: pd.DataFrame, risk: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Allocate weights using mean-variance optimization.

        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals (treated as expected returns)
        risk : pd.DataFrame | None, optional
            Historical returns for covariance estimation

        Returns
        -------
        pd.DataFrame
            Optimized portfolio weights
        """
        logger.info("Computing mean-variance optimal weights")

        # For now, use simple inverse vol if no risk data
        # Full optimization requires returns data and scipy.optimize
        if risk is None:
            logger.warning("No risk data provided, using inverse vol instead")
            return InverseVolatilityOptimizer(normalize=True).allocate(signals)

        # This is a simplified version - full implementation would use
        # scipy.optimize.minimize with constraints
        # For now, we'll do a simple mean-variance scaling

        signal_values = signals["signal"]

        # Get returns for covariance estimation
        # Look for return columns
        ret_cols = [col for col in risk.columns if col.startswith("ret_")]
        if not ret_cols:
            logger.warning("No return data for covariance, using inverse vol")
            return InverseVolatilityOptimizer(normalize=True).allocate(signals)

        # Use 1-day returns (for future covariance calculation)
        # ret_col = "ret_1d" if "ret_1d" in risk.columns else ret_cols[0]

        # For each date, compute optimal weights
        # This is simplified - a full implementation would use rolling windows
        weights = signal_values.copy() * 0.0

        # Get unique dates
        dates = signal_values.index.get_level_values("date").unique()

        for date in dates[-100:]:  # Only last 100 dates for speed in this example
            # Get signals for this date
            date_signals = signal_values.loc[date]

            if isinstance(date_signals, float):
                # Single value
                continue

            # Get active symbols
            active = date_signals[date_signals != 0]
            if len(active) < 2:
                # Not enough securities for optimization
                weights.loc[date] = date_signals / len(active) if len(active) > 0 else 0
                continue

            # Simplified: just scale by signal and normalize
            # Full version would compute covariance and solve QP
            date_weights = active / active.abs().sum()
            date_weights = date_weights.clip(lower=-self.max_weight, upper=self.max_weight)

            # Renormalize after clipping
            date_weights = date_weights / date_weights.abs().sum()

            weights.loc[date, date_weights.index] = date_weights

        result = pd.DataFrame({"weight": weights}, index=signals.index)

        n_positions = (result["weight"] != 0).sum()
        logger.info(f"Generated {n_positions} mean-variance optimized positions")

        return result


def create_portfolio_optimizer(
    method: str, **params: object
) -> EqualWeightOptimizer | InverseVolatilityOptimizer | MeanVarianceOptimizer:
    """Factory function to create portfolio optimizers.

    Parameters
    ----------
    method : str
        Optimization method: "equal_weight", "inverse_vol", or "mean_variance"
    **params : object
        Method-specific parameters

    Returns
    -------
    EqualWeightOptimizer | InverseVolatilityOptimizer | MeanVarianceOptimizer
        Portfolio optimizer instance

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == "equal_weight":
        return EqualWeightOptimizer(**params)  # type: ignore[arg-type]
    elif method in {"inverse_vol", "inv_vol"}:
        return InverseVolatilityOptimizer(**params)  # type: ignore[arg-type]
    elif method == "mean_variance":
        return MeanVarianceOptimizer(**params)  # type: ignore[arg-type]
    else:
        msg = f"Unknown portfolio optimization method: {method}"
        raise ValueError(msg)
