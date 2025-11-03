"""Feature pipeline implementations for transforming OHLCV to features.

This module provides FeaturePipeline implementations that create features
suitable for alpha models while preventing look-ahead bias.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import pandas as pd

# Suppress harmless all-NaN slice warnings from rolling window calculations
warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)

from alphalab.features.core import (
    compute_beta,
    compute_hl_volatility,
    compute_kurtosis,
    compute_parkinson_volatility,
    compute_returns,
    compute_rsi,
    compute_skewness,
    compute_volatility,
    compute_zscore,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StandardFeaturePipeline:
    """Standard feature pipeline for alpha models.

    Generates a comprehensive set of technical and statistical features
    from OHLCV data including returns, volatility, momentum, and higher moments.

    Optionally merges macro economic indicators (FRED) and fundamental metrics (FMP).

    Parameters
    ----------
    lookback_windows : list[int], default [5, 10, 20, 60, 126]
        Lookback windows for rolling features
    return_method : str, default "log"
        Return calculation method: "log" or "simple"
    include_rsi : bool, default True
        Whether to include RSI features
    include_beta : bool, default True
        Whether to include beta features (requires benchmark)
    benchmark_symbol : str | None, default "SPY"
        Benchmark symbol for beta calculation
    include_macro : bool, default False
        Whether to include macro economic indicators (FRED)
    include_fundamentals : bool, default False
        Whether to include fundamental metrics (FMP)
    """

    def __init__(
        self,
        lookback_windows: list[int] | None = None,
        return_method: str = "log",
        include_rsi: bool = True,
        include_beta: bool = True,
        benchmark_symbol: str | None = "SPY",
        include_macro: bool = False,
        include_fundamentals: bool = False,
    ) -> None:
        """Initialize feature pipeline."""
        # Extended lookback windows to include longer momentum signals (Tier-A)
        self.lookback_windows = lookback_windows or [1, 5, 10, 20, 21, 60, 126, 252]
        self.return_method = return_method
        self.include_rsi = include_rsi
        self.include_beta = include_beta
        self.benchmark_symbol = benchmark_symbol
        self.include_macro = include_macro
        self.include_fundamentals = include_fundamentals

    def transform(self, ohlcv: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Transform OHLCV data to features.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            OHLCV data with MultiIndex (date, symbol)
        **kwargs : object
            Additional parameters:
            - macro_data: pd.DataFrame with date index and macro indicator columns (optional)
            - fundamental_data: dict[str, pd.DataFrame] mapping symbols to fundamental DataFrames (optional)

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        """
        logger.info("Generating features from OHLCV data")

        # Extract optional data from kwargs
        macro_data = kwargs.get("macro_data", None)
        fundamental_data = kwargs.get("fundamental_data", None)

        features = pd.DataFrame(index=ohlcv.index)

        # Use adjusted close for returns if available, otherwise close
        price_col = "adj_close" if "adj_close" in ohlcv.columns else "close"
        prices = ohlcv[price_col].unstack("symbol")

        # 1. Returns at multiple horizons
        for window in self.lookback_windows:
            returns = compute_returns(prices, periods=window, method=self.return_method)
            features[f"ret_{window}d"] = returns.stack("symbol")

        # 2. Volatility at multiple horizons
        returns_1d = compute_returns(prices, periods=1, method=self.return_method)
        for window in [10, 20, 60]:
            vol = compute_volatility(returns_1d, window=window, annualize=False)
            features[f"vol_{window}d"] = vol.stack("symbol")

        # 2b. Tier-A: Advanced volatility estimators (using high-low range)
        if "high" in ohlcv.columns and "low" in ohlcv.columns:
            high_prices = ohlcv["high"].unstack("symbol")
            low_prices = ohlcv["low"].unstack("symbol")

            # Parkinson volatility (more efficient than close-to-close)
            parkinson_vol = compute_parkinson_volatility(high_prices, low_prices, window=20, annualize=False)
            features["parkinson_20d"] = parkinson_vol.stack("symbol")

            # High-low volatility (simple intraday range measure)
            hl_vol = compute_hl_volatility(high_prices, low_prices, window=20)
            features["hl_vol_20d"] = hl_vol.stack("symbol")

        # 2c. Tier-A: Liquidity measure (average dollar volume)
        if "volume" in ohlcv.columns:
            volume = ohlcv["volume"].unstack("symbol")
            # Dollar volume = price * volume
            dollar_volume = prices * volume
            # 20-day average dollar volume
            adv = dollar_volume.rolling(window=20, min_periods=10).mean()
            features["adv_20d"] = adv.stack("symbol")

        # 3. Z-scores of recent returns
        for window in [5, 20]:
            zscore = compute_zscore(returns_1d, window=window)
            features[f"zscore_{window}d"] = zscore.stack("symbol")

        # 4. Higher moments
        for window in [20, 60]:
            skew = compute_skewness(returns_1d, window=window)
            kurt = compute_kurtosis(returns_1d, window=window)
            features[f"skew_{window}d"] = skew.stack("symbol")
            features[f"kurt_{window}d"] = kurt.stack("symbol")

        # 5. RSI
        if self.include_rsi:
            for window in [14, 28]:
                rsi = prices.apply(lambda col: compute_rsi(col, window=window))
                features[f"rsi_{window}d"] = rsi.stack("symbol")

        # 6. Volume features
        if "volume" in ohlcv.columns:
            volume = ohlcv["volume"].unstack("symbol")

            # Volume z-score
            for window in [20, 60]:
                vol_zscore = compute_zscore(volume, window=window)
                features[f"volume_zscore_{window}d"] = vol_zscore.stack("symbol")

            # Turnover (volume / rolling average)
            for window in [20]:
                vol_ma = volume.rolling(window=window).mean()
                turnover = volume / vol_ma
                features[f"turnover_{window}d"] = turnover.stack("symbol")

        # 7. Gap features (open vs previous close) - Tier-A enhanced
        if "open" in ohlcv.columns:
            open_prices = ohlcv["open"].unstack("symbol")
            close_prices = ohlcv["close"].unstack("symbol")

            # Daily gap (existing)
            gap = (open_prices - close_prices.shift(1)) / close_prices.shift(1)
            features["gap"] = gap.stack("symbol")

            # Tier-A: 1-day gap (same as gap, for consistency with naming)
            features["gap_1d"] = gap.stack("symbol")

            # Tier-A: 20-day gap z-score (normalized gap relative to recent gaps)
            gap_zscore = compute_zscore(gap, window=20)
            features["gap_z20"] = gap_zscore.stack("symbol")

        # 8. Beta (if benchmark available)
        if self.include_beta and self.benchmark_symbol:
            if self.benchmark_symbol in prices.columns:
                benchmark_returns = returns_1d[self.benchmark_symbol]
                for window in [60, 126]:
                    # Calculate beta for each symbol
                    beta_data = {}
                    for symbol in prices.columns:
                        if symbol != self.benchmark_symbol:
                            beta = compute_beta(
                                returns_1d[symbol],
                                benchmark_returns,
                                window=window,
                            )
                            beta_data[symbol] = beta

                    beta_df = pd.DataFrame(beta_data)
                    # Ensure columns are named properly before stacking
                    stacked_beta = beta_df.stack()
                    stacked_beta.index.names = ["date", "symbol"]
                    features[f"beta_{window}d"] = stacked_beta

        # 9. Seasonality features
        dates = features.index.get_level_values("date")
        features["day_of_week"] = dates.dayofweek
        features["day_of_month"] = dates.day
        features["month"] = dates.month

        # 10. Macro economic indicators (broadcast to all symbols)
        if self.include_macro and macro_data is not None:
            logger.info(f"Merging {len(macro_data.columns)} macro indicators")
            # Get unique dates from features index
            feature_dates = features.index.get_level_values("date")

            # Convert macro data index to UTC if needed (stock data is UTC-aware)
            if macro_data.index.tz is None and feature_dates.tz is not None:
                macro_data.index = macro_data.index.tz_localize("UTC")

            for col in macro_data.columns:
                # Forward fill macro data first (to fill sparse monthly/quarterly data to daily)
                macro_filled = macro_data[col].ffill()
                # Then reindex to match feature dates and forward fill again for any missing dates
                macro_aligned = macro_filled.reindex(feature_dates).ffill()
                # Assign directly - pandas will broadcast the date-aligned series
                features[f"macro_{col}"] = macro_aligned.values

        # 11. Fundamental metrics (symbol-specific)
        if self.include_fundamentals and fundamental_data is not None:
            logger.info(f"Merging fundamental data for {len(fundamental_data)} symbols")
            for symbol, fund_df in fundamental_data.items():
                if symbol not in features.index.get_level_values("symbol"):
                    continue

                # Forward fill fundamentals (quarterly/annual reports persist until next report)
                symbol_dates = features.loc[pd.IndexSlice[:, symbol], :].index.get_level_values("date")

                for col in fund_df.columns:
                    fund_series = fund_df[col].reindex(symbol_dates).ffill()
                    features.loc[pd.IndexSlice[:, symbol], f"fund_{col}"] = fund_series.values

        logger.info(f"Generated {len(features.columns)} features")

        return features


class MinimalFeaturePipeline:
    """Minimal feature pipeline with only basic returns and volatility.

    Useful for simple alpha models or when computational efficiency is critical.

    Parameters
    ----------
    lookback_windows : list[int], default [20, 60, 126]
        Lookback windows for returns
    return_method : str, default "log"
        Return calculation method
    """

    def __init__(
        self,
        lookback_windows: list[int] | None = None,
        return_method: str = "log",
    ) -> None:
        """Initialize minimal feature pipeline."""
        self.lookback_windows = lookback_windows or [20, 60, 126]
        self.return_method = return_method

    def transform(self, ohlcv: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Transform OHLCV data to minimal feature set.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            OHLCV data with MultiIndex (date, symbol)
        **kwargs : object
            Additional parameters

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        """
        logger.info("Generating minimal features from OHLCV data")

        features = pd.DataFrame(index=ohlcv.index)

        # Use adjusted close for returns if available
        price_col = "adj_close" if "adj_close" in ohlcv.columns else "close"
        prices = ohlcv[price_col].unstack("symbol")

        # Returns at multiple horizons
        for window in self.lookback_windows:
            returns = compute_returns(prices, periods=window, method=self.return_method)
            features[f"ret_{window}d"] = returns.stack("symbol")

        # Recent volatility
        returns_1d = compute_returns(prices, periods=1, method=self.return_method)
        vol = compute_volatility(returns_1d, window=20, annualize=False)
        features["vol_20d"] = vol.stack("symbol")

        logger.info(f"Generated {len(features.columns)} features")

        return features


def create_feature_pipeline(
    pipeline_type: str = "standard", **kwargs: object
) -> StandardFeaturePipeline | MinimalFeaturePipeline:
    """Factory function to create feature pipelines.

    Parameters
    ----------
    pipeline_type : str, default "standard"
        Type of pipeline: "standard" or "minimal"
    **kwargs : object
        Pipeline-specific parameters

    Returns
    -------
    StandardFeaturePipeline | MinimalFeaturePipeline
        Feature pipeline instance

    Raises
    ------
    ValueError
        If pipeline_type is not recognized
    """
    if pipeline_type == "standard":
        return StandardFeaturePipeline(**kwargs)  # type: ignore[arg-type]
    elif pipeline_type == "minimal":
        return MinimalFeaturePipeline(**kwargs)  # type: ignore[arg-type]
    else:
        msg = f"Unknown pipeline type: {pipeline_type}"
        raise ValueError(msg)
