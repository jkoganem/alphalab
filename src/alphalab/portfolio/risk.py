"""Risk management and constraint enforcement.

This module provides risk targeting, exposure limits, and constraint
enforcement for portfolio construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RiskManager:
    """Portfolio risk manager.

    Enforces risk constraints and targets on portfolio weights.

    Parameters
    ----------
    volatility_target : float | None, default None
        Target annualized volatility (e.g., 0.15 for 15%)
    max_gross_exposure : float, default 2.0
        Maximum gross exposure (long + |short|)
    max_net_exposure : float, default 1.0
        Maximum net exposure (long - |short|)
    max_position_size : float, default 0.1
        Maximum weight per position
    max_turnover_pct : float | None, default None
        Maximum turnover as percentage of portfolio value
    max_leverage : float, default 1.0
        Maximum leverage (gross / net)

    Examples
    --------
    >>> rm = RiskManager(volatility_target=0.15, max_gross_exposure=2.0)
    >>> adjusted_weights = rm.apply_constraints(weights, current_weights, vol_forecast)
    """

    def __init__(
        self,
        volatility_target: float | None = None,
        max_gross_exposure: float = 2.0,
        max_net_exposure: float = 1.0,
        max_position_size: float = 0.1,
        max_turnover_pct: float | None = None,
        max_leverage: float = 1.0,
    ) -> None:
        """Initialize risk manager."""
        self.volatility_target = volatility_target
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_position_size = max_position_size
        self.max_turnover_pct = max_turnover_pct
        self.max_leverage = max_leverage

        logger.info(
            f"Initialized RiskManager with vol_target={volatility_target}, "
            f"max_gross={max_gross_exposure}, max_net={max_net_exposure}"
        )

    def apply_constraints(
        self,
        weights: pd.DataFrame,
        current_weights: pd.DataFrame | None = None,
        vol_forecast: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Apply risk constraints to portfolio weights.

        Parameters
        ----------
        weights : pd.DataFrame
            Target weights with 'weight' column
        current_weights : pd.DataFrame | None, optional
            Current portfolio weights (for turnover constraint)
        vol_forecast : pd.Series | None, optional
            Volatility forecast (for vol targeting)

        Returns
        -------
        pd.DataFrame
            Adjusted weights satisfying constraints
        """
        logger.info("Applying risk constraints")

        adjusted = weights.copy()

        # Apply constraints within each date
        dates = weights.index.get_level_values("date").unique()

        for date in dates:
            date_weights = adjusted.loc[date, "weight"]

            if isinstance(date_weights, float):
                # Single value, skip
                continue

            # 1. Clip individual positions
            date_weights = date_weights.clip(
                lower=-self.max_position_size, upper=self.max_position_size
            )

            # 2. Check and enforce gross exposure
            gross_exposure = date_weights.abs().sum()
            if gross_exposure > self.max_gross_exposure:
                scale_factor = self.max_gross_exposure / gross_exposure
                date_weights = date_weights * scale_factor
                logger.debug(
                    f"{date}: Scaled weights by {scale_factor:.3f} for gross exposure"
                )

            # 3. Check and enforce net exposure
            net_exposure = date_weights.sum()
            if abs(net_exposure) > self.max_net_exposure:
                # Scale down proportionally
                scale_factor = self.max_net_exposure / abs(net_exposure)
                date_weights = date_weights * scale_factor
                logger.debug(
                    f"{date}: Scaled weights by {scale_factor:.3f} for net exposure"
                )

            # 4. Enforce leverage constraint
            gross_exp = date_weights.abs().sum()
            net_exp = abs(date_weights.sum())
            if net_exp > 0:
                leverage = gross_exp / net_exp
                if leverage > self.max_leverage:
                    scale_factor = self.max_leverage * net_exp / gross_exp
                    date_weights = date_weights * scale_factor
                    logger.debug(
                        f"{date}: Scaled weights by {scale_factor:.3f} for leverage"
                    )

            # 5. Turnover constraint (if current weights provided)
            if current_weights is not None and self.max_turnover_pct is not None:
                if date in current_weights.index.get_level_values("date"):
                    curr_weights = current_weights.loc[date, "weight"]
                    if not isinstance(curr_weights, float):
                        # Calculate turnover
                        weight_change = (date_weights - curr_weights.reindex(date_weights.index, fill_value=0)).abs()
                        turnover = weight_change.sum()

                        if turnover > self.max_turnover_pct:
                            # Scale back changes to respect turnover limit
                            scale_factor = self.max_turnover_pct / turnover
                            date_weights = (
                                curr_weights.reindex(date_weights.index, fill_value=0)
                                + (date_weights - curr_weights.reindex(date_weights.index, fill_value=0)) * scale_factor
                            )
                            logger.debug(
                                f"{date}: Reduced turnover from {turnover:.2%} "
                                f"to {self.max_turnover_pct:.2%}"
                            )

            adjusted.loc[date, "weight"] = date_weights

        return adjusted

    def target_volatility(
        self,
        weights: pd.DataFrame,
        vol_forecast: pd.Series,
        periods_per_year: int = 252,
    ) -> pd.DataFrame:
        """Scale weights to target volatility.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights
        vol_forecast : pd.Series
            Forecasted volatility for each security
        periods_per_year : int, default 252
            Periods per year for annualization

        Returns
        -------
        pd.DataFrame
            Scaled weights
        """
        if self.volatility_target is None:
            return weights

        logger.info(f"Targeting {self.volatility_target:.1%} annualized volatility")

        scaled = weights.copy()
        dates = weights.index.get_level_values("date").unique()

        for date in dates:
            date_weights = scaled.loc[date, "weight"]

            if isinstance(date_weights, float):
                continue

            # Get volatilities for this date
            if date in vol_forecast.index.get_level_values("date"):
                date_vols = vol_forecast.loc[date]

                # Estimate portfolio volatility (simplified: ignore correlations)
                # More accurate: would use covariance matrix
                portfolio_var = (date_weights**2 * date_vols**2).sum()
                portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(periods_per_year)

                if portfolio_vol > 0:
                    scale_factor = self.volatility_target / portfolio_vol
                    # Cap scaling to avoid extreme leverage
                    scale_factor = np.clip(scale_factor, 0.1, 3.0)

                    scaled.loc[date, "weight"] = date_weights * scale_factor
                    logger.debug(
                        f"{date}: Scaled by {scale_factor:.3f} "
                        f"(vol: {portfolio_vol:.2%} -> {portfolio_vol * scale_factor:.2%})"
                    )

        return scaled


def calculate_portfolio_risk(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int = 60,
) -> dict[str, pd.Series]:
    """Calculate portfolio risk metrics.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights with MultiIndex (date, symbol)
    returns : pd.DataFrame
        Asset returns with MultiIndex (date, symbol)
    lookback : int, default 60
        Lookback window for calculations

    Returns
    -------
    dict[str, pd.Series]
        Risk metrics including volatility, beta, exposures
    """
    logger.info("Calculating portfolio risk metrics")

    # Pivot to wide format
    weights_wide = weights["weight"].unstack("symbol", fill_value=0)
    returns_wide = returns.unstack("symbol")

    # Portfolio returns
    # Align weights and returns (weights at t applied to returns at t+1)
    aligned_weights = weights_wide.shift(1)
    portfolio_returns = (aligned_weights * returns_wide).sum(axis=1)

    # Rolling volatility
    portfolio_vol = portfolio_returns.rolling(window=lookback).std() * np.sqrt(252)

    # Exposures
    gross_exposure = weights_wide.abs().sum(axis=1)
    net_exposure = weights_wide.sum(axis=1)
    long_exposure = weights_wide.clip(lower=0).sum(axis=1)
    short_exposure = weights_wide.clip(upper=0).abs().sum(axis=1)

    # Turnover (weight changes)
    weight_changes = weights_wide.diff().abs()
    turnover = weight_changes.sum(axis=1)

    metrics = {
        "portfolio_returns": portfolio_returns,
        "portfolio_volatility": portfolio_vol,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "turnover": turnover,
    }

    logger.info(
        f"Average gross exposure: {gross_exposure.mean():.2f}, "
        f"Average turnover: {turnover.mean():.2%}"
    )

    return metrics


def check_constraints(
    weights: pd.DataFrame,
    max_gross: float = 2.0,
    max_net: float = 1.0,
    max_position: float = 0.1,
) -> pd.DataFrame:
    """Check if weights violate constraints.

    Parameters
    ----------
    weights : pd.DataFrame
        Portfolio weights
    max_gross : float, default 2.0
        Max gross exposure
    max_net : float, default 1.0
        Max net exposure
    max_position : float, default 0.1
        Max position size

    Returns
    -------
    pd.DataFrame
        Constraint violations by date
    """
    dates = weights.index.get_level_values("date").unique()

    violations = []

    for date in dates:
        date_weights = weights.loc[date, "weight"]

        if isinstance(date_weights, float):
            continue

        gross = date_weights.abs().sum()
        net = date_weights.sum()
        max_pos = date_weights.abs().max()

        violations.append(
            {
                "date": date,
                "gross_violation": max(0, gross - max_gross),
                "net_violation": max(0, abs(net) - max_net),
                "position_violation": max(0, max_pos - max_position),
                "has_violation": (
                    gross > max_gross or abs(net) > max_net or max_pos > max_position
                ),
            }
        )

    violations_df = pd.DataFrame(violations).set_index("date")

    n_violations = violations_df["has_violation"].sum()
    if n_violations > 0:
        logger.warning(f"Found {n_violations} dates with constraint violations")

    return violations_df
