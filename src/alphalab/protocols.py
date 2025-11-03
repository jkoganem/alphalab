"""Core protocols and data contracts for Alpha Backtest Lab.

This module defines typed interfaces for all major components using Protocol classes
and dataclasses for immutable records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """Immutable OHLCV bar record.

    Parameters
    ----------
    ts : pd.Timestamp
        Timestamp in UTC timezone-aware format
    open : float
        Opening price
    high : float
        High price
    low : float
        Low price
    close : float
        Closing price
    volume : float
        Trading volume
    """

    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataSource(Protocol):
    """Protocol for data sources that fetch market data.

    Implementations should support fetching OHLCV data for multiple symbols
    over a specified date range.
    """

    def fetch(
        self,
        symbols: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch OHLCV data for given symbols and date range.

        Parameters
        ----------
        symbols : list[str]
            List of ticker symbols to fetch
        start : pd.Timestamp
            Start date (inclusive), timezone-aware
        end : pd.Timestamp
            End date (inclusive), timezone-aware
        interval : str, default "1d"
            Data interval (e.g., "1d", "1h")

        Returns
        -------
        pd.DataFrame
            MultiIndex DataFrame with (date, symbol) index and OHLCV columns
        """
        ...


class FeaturePipeline(Protocol):
    """Protocol for feature engineering pipelines.

    Transforms raw OHLCV data into features suitable for alpha models.
    """

    def transform(self, ohlcv: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        """Transform OHLCV data into features.

        Parameters
        ----------
        ohlcv : pd.DataFrame
            Raw OHLCV data with MultiIndex (date, symbol)
        **kwargs : object
            Additional transformation parameters

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        """
        ...


class AlphaModel(Protocol):
    """Protocol for alpha signal generation models.

    Alpha models score securities based on features, producing raw alpha scores.
    """

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha scores from features.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol)
        """
        ...


class SignalModel(Protocol):
    """Protocol for converting alpha scores to trading signals.

    Signal models convert continuous alpha scores into discrete signals
    (e.g., +1/0/-1) or normalized weights.
    """

    def to_signal(self, alpha: pd.DataFrame) -> pd.DataFrame:
        """Convert alpha scores to trading signals.

        Parameters
        ----------
        alpha : pd.DataFrame
            Alpha scores with MultiIndex (date, symbol)

        Returns
        -------
        pd.DataFrame
            Trading signals with MultiIndex (date, symbol)
        """
        ...


class PortfolioOptimizer(Protocol):
    """Protocol for portfolio weight optimization.

    Optimizers convert signals and risk estimates into target portfolio weights.
    """

    def allocate(
        self, signals: pd.DataFrame, risk: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Allocate portfolio weights based on signals and risk.

        Parameters
        ----------
        signals : pd.DataFrame
            Trading signals with MultiIndex (date, symbol)
        risk : pd.DataFrame | None, optional
            Risk estimates (e.g., covariance matrix)

        Returns
        -------
        pd.DataFrame
            Target portfolio weights with MultiIndex (date, symbol)
        """
        ...


class ExecutionModel(Protocol):
    """Protocol for execution simulation.

    Simulates the execution of target weights into actual fills/trades,
    accounting for market impact, liquidity constraints, etc.
    """

    def simulate(self, target_w: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        """Simulate execution of target weights.

        Parameters
        ----------
        target_w : pd.DataFrame
            Target weights with MultiIndex (date, symbol)
        prices : pd.DataFrame
            Price data with MultiIndex (date, symbol)

        Returns
        -------
        pd.DataFrame
            Executed fills/trades with MultiIndex (date, symbol)
        """
        ...


class Backtester(Protocol):
    """Protocol for backtesting engine.

    Runs a complete backtest given weights, prices, and cost configuration.
    """

    def run(
        self, weights: pd.DataFrame, prices: pd.DataFrame, costs_cfg: dict[str, object]
    ) -> dict[str, object]:
        """Run backtest simulation.

        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights with MultiIndex (date, symbol)
        prices : pd.DataFrame
            Price data with MultiIndex (date, symbol)
        costs_cfg : dict[str, object]
            Cost model configuration

        Returns
        -------
        dict[str, object]
            Backtest results including equity curve and statistics
        """
        ...


class CostModel(Protocol):
    """Protocol for transaction cost estimation.

    Estimates transaction costs (fees, slippage, borrow costs) for orders.
    """

    def estimate(self, orders: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
        """Estimate transaction costs for orders.

        Parameters
        ----------
        orders : pd.DataFrame
            Order data with MultiIndex (date, symbol)
        prices : pd.DataFrame
            Price data with MultiIndex (date, symbol)

        Returns
        -------
        pd.Series
            Cost estimates indexed by date
        """
        ...


class Metric(Protocol):
    """Protocol for performance metrics.

    Computes performance metrics from equity curves and returns.
    """

    def compute(
        self, equity_curve: pd.Series, returns: pd.Series, **kwargs: object
    ) -> float | dict[str, float]:
        """Compute performance metric(s).

        Parameters
        ----------
        equity_curve : pd.Series
            Equity curve indexed by date
        returns : pd.Series
            Returns series indexed by date
        **kwargs : object
            Additional metric parameters

        Returns
        -------
        float | dict[str, float]
            Metric value(s)
        """
        ...
