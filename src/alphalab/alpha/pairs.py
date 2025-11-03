"""Pairs trading / statistical arbitrage alpha models.

This module implements pairs trading strategies based on cointegration
and mean reversion of spread between two securities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from alphalab.features.core import compute_zscore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PairsTradingAlpha:
    """Statistical arbitrage via pairs trading.

    Identifies cointegrated pairs and trades the mean-reverting spread.

    Parameters
    ----------
    formation_window : int, default 126
        Lookback window for cointegration testing (formation period)
    trading_window : int, default 20
        Rolling window for spread z-score calculation
    entry_threshold : float, default 2.0
        Z-score threshold for entry (e.g., 2.0 = 2 std devs)
    exit_threshold : float, default 0.5
        Z-score threshold for exit
    stop_loss_threshold : float, default 4.0
        Z-score threshold for stop loss
    min_half_life : int, default 1
        Minimum half-life of mean reversion (in days)
    max_half_life : int, default 60
        Maximum half-life of mean reversion (in days)
    significance_level : float, default 0.05
        P-value threshold for cointegration test

    Examples
    --------
    >>> pairs = PairsTradingAlpha(formation_window=126, entry_threshold=2.0)
    >>> alpha = pairs.score(features, prices)
    """

    def __init__(
        self,
        formation_window: int = 126,
        trading_window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 4.0,
        min_half_life: int = 1,
        max_half_life: int = 60,
        significance_level: float = 0.05,
    ) -> None:
        """Initialize pairs trading alpha."""
        self.formation_window = formation_window
        self.trading_window = trading_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.significance_level = significance_level

        self.pairs: list[tuple[str, str, float]] = []  # (symbol1, symbol2, hedge_ratio)

        logger.info(
            f"Initialized PairsTradingAlpha with formation={formation_window}, "
            f"trading={trading_window}, entry={entry_threshold}"
        )

    def identify_pairs(self, prices: pd.DataFrame) -> list[tuple[str, str, float]]:
        """Identify cointegrated pairs from price data.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data with columns as symbols

        Returns
        -------
        list[tuple[str, str, float]]
            List of (symbol1, symbol2, hedge_ratio) tuples
        """
        logger.info("Identifying cointegrated pairs...")

        symbols = prices.columns.tolist()
        pairs = []

        # Test all possible pairs
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                # Get price series
                p1 = prices[sym1].dropna()
                p2 = prices[sym2].dropna()

                # Align series
                aligned = pd.DataFrame({"p1": p1, "p2": p2}).dropna()

                if len(aligned) < self.formation_window:
                    continue

                # Test for cointegration
                try:
                    _, pvalue, _ = coint(aligned["p1"], aligned["p2"])

                    if pvalue < self.significance_level:
                        # Calculate hedge ratio via OLS
                        # p1 = beta * p2 + alpha
                        # spread = p1 - beta * p2
                        hedge_ratio = np.cov(aligned["p1"], aligned["p2"])[0, 1] / np.var(
                            aligned["p2"]
                        )

                        # Calculate spread
                        spread = aligned["p1"] - hedge_ratio * aligned["p2"]

                        # Test for stationarity of spread
                        adf_result = adfuller(spread, maxlag=1)
                        adf_pvalue = adf_result[1]

                        if adf_pvalue < self.significance_level:
                            # Estimate half-life of mean reversion
                            half_life = self._estimate_half_life(spread)

                            if self.min_half_life <= half_life <= self.max_half_life:
                                pairs.append((sym1, sym2, hedge_ratio))
                                logger.info(
                                    f"Found pair: {sym1}/{sym2}, "
                                    f"hedge_ratio={hedge_ratio:.4f}, "
                                    f"half_life={half_life:.1f} days"
                                )

                except Exception as e:
                    logger.debug(f"Error testing {sym1}/{sym2}: {e}")
                    continue

        logger.info(f"Identified {len(pairs)} cointegrated pairs")
        return pairs

    def _estimate_half_life(self, spread: pd.Series) -> float:
        """Estimate half-life of mean reversion using AR(1) model.

        Parameters
        ----------
        spread : pd.Series
            Spread time series

        Returns
        -------
        float
            Half-life in periods
        """
        spread_lag = spread.shift(1).dropna()
        spread_curr = spread[1:]

        # Align
        aligned = pd.DataFrame({"curr": spread_curr, "lag": spread_lag}).dropna()

        if len(aligned) < 2:
            return np.inf

        # AR(1): spread(t) = alpha + beta * spread(t-1) + error
        # Half-life = -log(2) / log(beta)
        try:
            beta = np.corrcoef(aligned["curr"], aligned["lag"])[0, 1]

            if beta >= 1 or beta <= 0:
                return np.inf

            half_life = -np.log(2) / np.log(beta)
            return half_life if half_life > 0 else np.inf

        except Exception:
            return np.inf

    def score(
        self, features: pd.DataFrame, prices: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate pairs trading alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        prices : pd.DataFrame | None, optional
            Price data for pair identification

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol)
        """
        logger.info("Generating pairs trading alpha scores")

        # If no prices provided, try to extract from features
        if prices is None:
            # Look for price column in features
            if "close" in features.columns:
                prices_series = features["close"]
            else:
                msg = "prices parameter required or 'close' must be in features"
                raise ValueError(msg)
        else:
            # Assume prices has the right format
            if isinstance(prices, pd.DataFrame):
                if prices.index.nlevels == 2:
                    # MultiIndex - extract close or first column
                    col = "close" if "close" in prices.columns else prices.columns[0]
                    prices_series = prices[col]
                else:
                    msg = "prices must have MultiIndex (date, symbol)"
                    raise ValueError(msg)
            else:
                prices_series = prices

        # Pivot to wide format for pair analysis
        prices_wide = prices_series.unstack("symbol")

        # Identify pairs on formation window
        if not self.pairs:
            formation_prices = prices_wide.iloc[-self.formation_window :]
            self.pairs = self.identify_pairs(formation_prices)

        if not self.pairs:
            logger.warning("No cointegrated pairs found, returning zero alpha")
            return pd.DataFrame({"alpha": 0.0}, index=features.index)

        # Calculate signals for each pair
        alpha_scores = pd.Series(0.0, index=features.index)

        for sym1, sym2, hedge_ratio in self.pairs:
            # Get price series
            if sym1 not in prices_wide.columns or sym2 not in prices_wide.columns:
                continue

            p1 = prices_wide[sym1]
            p2 = prices_wide[sym2]

            # Calculate spread
            spread = p1 - hedge_ratio * p2

            # Calculate z-score of spread
            spread_zscore = compute_zscore(spread, window=self.trading_window)

            # Generate signals based on z-score thresholds
            # When spread is high (z > entry_threshold): short spread (short sym1, long sym2)
            # When spread is low (z < -entry_threshold): long spread (long sym1, short sym2)

            # For sym1: trade opposite to z-score
            sym1_signal = pd.Series(0.0, index=spread.index)
            sym1_signal = sym1_signal.where(
                spread_zscore.abs() <= self.entry_threshold, -np.sign(spread_zscore)
            )

            # Exit when z-score near zero
            sym1_signal = sym1_signal.where(
                spread_zscore.abs() >= self.exit_threshold, 0.0
            )

            # Stop loss when z-score extreme
            sym1_signal = sym1_signal.where(
                spread_zscore.abs() <= self.stop_loss_threshold, 0.0
            )

            # For sym2: trade with hedge ratio adjustment
            sym2_signal = -sym1_signal * hedge_ratio

            # Add to alpha scores (stack back to MultiIndex)
            for date in spread.index:
                if date in alpha_scores.index.get_level_values("date"):
                    if (date, sym1) in alpha_scores.index:
                        alpha_scores.loc[(date, sym1)] += sym1_signal.loc[date]
                    if (date, sym2) in alpha_scores.index:
                        alpha_scores.loc[(date, sym2)] += sym2_signal.loc[date]

        result = pd.DataFrame({"alpha": alpha_scores}, index=features.index)

        n_signals = (result["alpha"] != 0).sum()
        logger.info(f"Generated {n_signals} pairs trading signals across {len(self.pairs)} pairs")

        return result


class SimplePairsAlpha:
    """Simplified pairs trading using correlation.

    Simpler alternative that uses correlation instead of cointegration.
    Faster but less statistically rigorous.

    Parameters
    ----------
    lookback_window : int, default 60
        Lookback for correlation calculation
    min_correlation : float, default 0.7
        Minimum correlation threshold
    entry_threshold : float, default 1.5
        Z-score entry threshold

    Examples
    --------
    >>> pairs = SimplePairsAlpha(min_correlation=0.8)
    >>> alpha = pairs.score(features, prices)
    """

    def __init__(
        self,
        lookback_window: int = 60,
        min_correlation: float = 0.7,
        entry_threshold: float = 1.5,
    ) -> None:
        """Initialize simple pairs trading."""
        self.lookback_window = lookback_window
        self.min_correlation = min_correlation
        self.entry_threshold = entry_threshold

        logger.info(
            f"Initialized SimplePairsAlpha with lookback={lookback_window}, "
            f"min_corr={min_correlation}"
        )

    def score(
        self, features: pd.DataFrame, prices: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Generate simple pairs alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Features
        prices : pd.DataFrame | None, optional
            Prices

        Returns
        -------
        pd.DataFrame
            Alpha scores
        """
        logger.info("Generating simple pairs alpha (correlation-based)")

        # Extract prices
        if prices is None:
            if "close" in features.columns:
                prices_series = features["close"]
            else:
                msg = "Need prices data"
                raise ValueError(msg)
        else:
            if isinstance(prices, pd.DataFrame) and prices.index.nlevels == 2:
                col = "close" if "close" in prices.columns else prices.columns[0]
                prices_series = prices[col]
            else:
                prices_series = prices

        prices_wide = prices_series.unstack("symbol")

        # Calculate rolling correlations and find highly correlated pairs
        # This is simplified - just use static correlation for now
        returns = prices_wide.pct_change()
        corr_matrix = returns.corr()

        # Find pairs with high correlation
        alpha_scores = pd.Series(0.0, index=features.index)

        symbols = corr_matrix.columns.tolist()
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                if corr_matrix.loc[sym1, sym2] > self.min_correlation:
                    # Calculate spread (price difference)
                    spread = prices_wide[sym1] - prices_wide[sym2]
                    spread_z = compute_zscore(spread, window=self.lookback_window)

                    # Simple mean reversion signal
                    signal = -np.sign(spread_z).where(
                        spread_z.abs() > self.entry_threshold, 0
                    )

                    # Add to alpha
                    for date in signal.index:
                        if date in alpha_scores.index.get_level_values("date"):
                            if (date, sym1) in alpha_scores.index:
                                alpha_scores.loc[(date, sym1)] += signal.loc[date]
                            if (date, sym2) in alpha_scores.index:
                                alpha_scores.loc[(date, sym2)] -= signal.loc[date]

        result = pd.DataFrame({"alpha": alpha_scores}, index=features.index)

        n_signals = (result["alpha"] != 0).sum()
        logger.info(f"Generated {n_signals} simple pairs signals")

        return result


def create_pairs_alpha(
    alpha_type: str = "cointegration", **kwargs: object
) -> PairsTradingAlpha | SimplePairsAlpha:
    """Factory function to create pairs trading alphas.

    Parameters
    ----------
    alpha_type : str, default "cointegration"
        Type: "cointegration" or "correlation"
    **kwargs : object
        Alpha-specific parameters

    Returns
    -------
    PairsTradingAlpha | SimplePairsAlpha
        Pairs alpha instance
    """
    if alpha_type == "cointegration":
        return PairsTradingAlpha(**kwargs)  # type: ignore[arg-type]
    elif alpha_type == "correlation":
        return SimplePairsAlpha(**kwargs)  # type: ignore[arg-type]
    else:
        msg = f"Unknown pairs alpha type: {alpha_type}"
        raise ValueError(msg)
