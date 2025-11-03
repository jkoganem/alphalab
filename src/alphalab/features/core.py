"""Core feature engineering functions.

This module provides building blocks for creating technical and statistical
features from OHLCV data, with careful handling of missing data and look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.Series | pd.DataFrame,
    periods: int = 1,
    method: str = "log",
) -> pd.Series | pd.DataFrame:
    """Compute returns from prices.

    Parameters
    ----------
    prices : pd.Series | pd.DataFrame
        Price series or DataFrame
    periods : int, default 1
        Number of periods for return calculation
    method : str, default "log"
        Return type: "log" or "simple"

    Returns
    -------
    pd.Series | pd.DataFrame
        Returns series or DataFrame

    Raises
    ------
    ValueError
        If method is not "log" or "simple"
    """
    if method == "log":
        return np.log(prices / prices.shift(periods))
    elif method == "simple":
        return prices.pct_change(periods)
    else:
        msg = f"Invalid return method: {method}. Use 'log' or 'simple'"
        raise ValueError(msg)


def compute_volatility(
    returns: pd.Series | pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """Compute rolling volatility.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame
        Returns series or DataFrame
    window : int, default 20
        Rolling window size
    annualize : bool, default True
        Whether to annualize volatility
    periods_per_year : int, default 252
        Trading periods per year for annualization

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling volatility
    """
    vol = returns.rolling(window=window, min_periods=window // 2).std()

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def compute_zscore(
    series: pd.Series | pd.DataFrame,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute rolling z-score.

    Parameters
    ----------
    series : pd.Series | pd.DataFrame
        Input series or DataFrame
    window : int, default 20
        Rolling window size
    min_periods : int | None, optional
        Minimum periods for calculation

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling z-score
    """
    if min_periods is None:
        min_periods = window // 2

    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()

    return (series - mean) / std


def compute_beta(
    returns: pd.Series | pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute rolling beta relative to benchmark.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame
        Asset returns
    benchmark_returns : pd.Series
        Benchmark returns
    window : int, default 60
        Rolling window size
    min_periods : int | None, optional
        Minimum periods for calculation

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling beta
    """
    if min_periods is None:
        min_periods = window // 2

    # Align benchmark with returns
    benchmark_aligned = benchmark_returns.reindex(returns.index, method="ffill")

    # Compute covariance and variance
    cov = returns.rolling(window=window, min_periods=min_periods).cov(benchmark_aligned)
    var = benchmark_aligned.rolling(window=window, min_periods=min_periods).var()

    return cov / var


def winsorize(
    data: pd.Series | pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series | pd.DataFrame:
    """Winsorize data to clip extreme values.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Input data
    lower : float, default 0.01
        Lower percentile threshold
    upper : float, default 0.99
        Upper percentile threshold

    Returns
    -------
    pd.Series | pd.DataFrame
        Winsorized data
    """
    if isinstance(data, pd.Series):
        lower_bound = data.quantile(lower)
        upper_bound = data.quantile(upper)
        return data.clip(lower=lower_bound, upper=upper_bound)
    else:
        return data.apply(lambda col: col.clip(lower=col.quantile(lower), upper=col.quantile(upper)))


def compute_rsi(
    prices: pd.Series,
    window: int = 14,
) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    Parameters
    ----------
    prices : pd.Series
        Price series
    window : int, default 14
        RSI window size

    Returns
    -------
    pd.Series
        RSI values (0-100)
    """
    delta = prices.diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_skewness(
    returns: pd.Series | pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute rolling skewness.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame
        Returns series or DataFrame
    window : int, default 60
        Rolling window size
    min_periods : int | None, optional
        Minimum periods for calculation

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling skewness
    """
    if min_periods is None:
        min_periods = window // 2

    return returns.rolling(window=window, min_periods=min_periods).skew()


def compute_kurtosis(
    returns: pd.Series | pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute rolling kurtosis.

    Parameters
    ----------
    returns : pd.Series | pd.DataFrame
        Returns series or DataFrame
    window : int, default 60
        Rolling window size
    min_periods : int | None, optional
        Minimum periods for calculation

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling kurtosis
    """
    if min_periods is None:
        min_periods = window // 2

    return returns.rolling(window=window, min_periods=min_periods).kurt()


def compute_rolling_correlation(
    x: pd.Series | pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    window: int = 60,
    min_periods: int | None = None,
) -> pd.Series | pd.DataFrame:
    """Compute rolling correlation between two series.

    Parameters
    ----------
    x : pd.Series | pd.DataFrame
        First series or DataFrame
    y : pd.Series | pd.DataFrame
        Second series or DataFrame
    window : int, default 60
        Rolling window size
    min_periods : int | None, optional
        Minimum periods for calculation

    Returns
    -------
    pd.Series | pd.DataFrame
        Rolling correlation
    """
    if min_periods is None:
        min_periods = window // 2

    return x.rolling(window=window, min_periods=min_periods).corr(y)


def rank_normalize(
    data: pd.DataFrame,
    by_date: bool = True,
    method: str = "average",
) -> pd.DataFrame:
    """Rank normalize data cross-sectionally or time-series.

    Parameters
    ----------
    data : pd.DataFrame
        Data with MultiIndex (date, symbol)
    by_date : bool, default True
        If True, rank within each date (cross-sectional)
        If False, rank within each symbol (time-series)
    method : str, default "average"
        Ranking method: "average", "min", "max", "first", "dense"

    Returns
    -------
    pd.DataFrame
        Rank-normalized data (0 to 1 scale)
    """
    if by_date:
        # Cross-sectional ranking
        result = data.groupby(level="date", group_keys=False).rank(method=method, pct=True)
    else:
        # Time-series ranking
        result = data.groupby(level="symbol", group_keys=False).rank(method=method, pct=True)

    return result


def zscore_normalize(
    data: pd.DataFrame,
    by_date: bool = True,
) -> pd.DataFrame:
    """Z-score normalize data cross-sectionally or time-series.

    Parameters
    ----------
    data : pd.DataFrame
        Data with MultiIndex (date, symbol)
    by_date : bool, default True
        If True, normalize within each date (cross-sectional)
        If False, normalize within each symbol (time-series)

    Returns
    -------
    pd.DataFrame
        Z-score normalized data
    """
    if by_date:
        # Cross-sectional normalization
        result = data.groupby(level="date", group_keys=False).transform(
            lambda x: (x - x.mean()) / x.std()
        )
    else:
        # Time-series normalization
        result = data.groupby(level="symbol", group_keys=False).transform(
            lambda x: (x - x.mean()) / x.std()
        )

    return result


def compute_parkinson_volatility(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    window: int = 20,
    annualize: bool = False,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """Compute Parkinson volatility estimator using high-low range.

    Parkinson volatility uses the high-low range to estimate volatility,
    which is more efficient than close-to-close volatility for the same
    number of observations.

    Formula: sqrt((1/(4*ln(2))) * (ln(H/L))^2)

    Parameters
    ----------
    high : pd.Series | pd.DataFrame
        High prices
    low : pd.Series | pd.DataFrame
        Low prices
    window : int, default 20
        Rolling window size
    annualize : bool, default False
        Whether to annualize volatility
    periods_per_year : int, default 252
        Trading periods per year for annualization

    Returns
    -------
    pd.Series | pd.DataFrame
        Parkinson volatility estimate
    """
    # Parkinson formula
    hl_ratio = np.log(high / low)
    parkinson_var = hl_ratio ** 2

    # Rolling average and convert to volatility
    vol = np.sqrt(
        parkinson_var.rolling(window=window, min_periods=window // 2).mean() / (4 * np.log(2))
    )

    if annualize:
        vol = vol * np.sqrt(periods_per_year)

    return vol


def compute_hl_volatility(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    window: int = 20,
) -> pd.Series | pd.DataFrame:
    """Compute high-low volatility as percentage of price.

    Simple measure of intraday volatility: (high - low) / close

    Parameters
    ----------
    high : pd.Series | pd.DataFrame
        High prices
    low : pd.Series | pd.DataFrame
        Low prices
    window : int, default 20
        Rolling window size for averaging

    Returns
    -------
    pd.Series | pd.DataFrame
        Average high-low range as percentage
    """
    # Simple high-low range
    hl_range = (high - low) / low

    # Rolling average
    hl_vol = hl_range.rolling(window=window, min_periods=window // 2).mean()

    return hl_vol


def sector_neutralize(
    data: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Neutralize data by sector (demean within each sector).

    Parameters
    ----------
    data : pd.DataFrame
        Data with MultiIndex (date, symbol)
    sector_map : dict[str, str]
        Mapping from symbol to sector

    Returns
    -------
    pd.DataFrame
        Sector-neutralized data
    """
    # Add sector information
    symbols = data.index.get_level_values("symbol")
    sectors = symbols.map(sector_map)

    # Group by date and sector, demean
    result = data.copy()
    for col in data.columns:
        result[col] = data.groupby([data.index.get_level_values("date"), sectors])[col].transform(
            lambda x: x - x.mean()
        )

    return result
