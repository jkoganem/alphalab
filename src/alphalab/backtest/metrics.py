"""Performance metrics for backtesting.

This module provides functions to calculate various performance and risk metrics
from backtest results including returns, volatility, Sharpe ratio, and drawdowns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def calculate_returns_metrics(
    returns: pd.Series, periods_per_year: int = 252
) -> dict[str, float]:
    """Calculate return-based metrics.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int, default 252
        Number of periods per year for annualization

    Returns
    -------
    dict[str, float]
        Dictionary of return metrics
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    # Annualized return (geometric)
    if years > 0:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = 0.0

    # CAGR (same as annual_return)
    cagr = annual_return

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "cagr": cagr,
        "periods": n_periods,
        "years": years,
    }


def calculate_risk_metrics(
    returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Calculate risk-based metrics.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    periods_per_year : int, default 252
        Periods per year
    risk_free_rate : float, default 0.0
        Annualized risk-free rate

    Returns
    -------
    dict[str, float]
        Dictionary of risk metrics
    """
    # Volatility
    vol_daily = returns.std()
    vol_annual = vol_daily * np.sqrt(periods_per_year)

    # Mean return
    mean_daily = returns.mean()
    mean_annual = mean_daily * periods_per_year

    # Sharpe ratio
    excess_return = mean_annual - risk_free_rate
    sharpe = excess_return / vol_annual if vol_annual > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    sortino = excess_return / downside_std if downside_std > 0 else 0.0

    # Skewness and kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()

    return {
        "volatility_daily": vol_daily,
        "volatility_annual": vol_annual,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "skewness": skew,
        "kurtosis": kurt,
    }


def calculate_drawdown(equity_curve: pd.Series) -> pd.DataFrame:
    """Calculate drawdown series.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity curve over time

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: peak, drawdown, drawdown_pct
    """
    # Running maximum (peak)
    running_max = equity_curve.expanding().max()

    # Drawdown (absolute)
    drawdown = equity_curve - running_max

    # Drawdown percentage
    drawdown_pct = drawdown / running_max

    return pd.DataFrame(
        {"peak": running_max, "drawdown": drawdown, "drawdown_pct": drawdown_pct},
        index=equity_curve.index,
    )


def calculate_drawdown_metrics(equity_curve: pd.Series) -> dict[str, float]:
    """Calculate drawdown metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity curve

    Returns
    -------
    dict[str, float]
        Drawdown metrics including max drawdown and Calmar ratio
    """
    dd_df = calculate_drawdown(equity_curve)

    # Max drawdown
    max_dd = dd_df["drawdown_pct"].min()

    # Average drawdown
    avg_dd = dd_df["drawdown_pct"][dd_df["drawdown_pct"] < 0].mean()

    # Max drawdown duration (days in drawdown)
    in_drawdown = dd_df["drawdown_pct"] < 0
    if in_drawdown.any():
        # Find longest consecutive period
        dd_periods = (~in_drawdown).cumsum()[in_drawdown]
        max_dd_duration = dd_periods.value_counts().max() if len(dd_periods) > 0 else 0
    else:
        max_dd_duration = 0

    # Calmar ratio (annual return / max drawdown)
    returns = equity_curve.pct_change().dropna()
    if len(returns) > 0:
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        calmar = -annual_return / max_dd if max_dd < 0 else 0.0
    else:
        calmar = 0.0

    return {
        "max_drawdown": max_dd,
        "avg_drawdown": avg_dd if not np.isnan(avg_dd) else 0.0,
        "max_dd_duration": int(max_dd_duration),
        "calmar_ratio": calmar,
    }


def calculate_trade_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """Calculate trade-based metrics.

    Parameters
    ----------
    trades : pd.DataFrame
        Trades DataFrame with 'notional' column

    Returns
    -------
    dict[str, float]
        Trade metrics
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "avg_trade_size": 0.0,
            "total_traded": 0.0,
        }

    n_trades = len(trades)
    avg_trade_size = trades["notional"].abs().mean()
    total_traded = trades["notional"].abs().sum()

    return {
        "n_trades": n_trades,
        "avg_trade_size": avg_trade_size,
        "total_traded": total_traded,
    }


def calculate_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Calculate comprehensive performance metrics.

    Parameters
    ----------
    equity_curve : pd.Series
        Equity curve
    returns : pd.Series
        Returns series
    trades : pd.DataFrame | None, optional
        Trades data
    periods_per_year : int, default 252
        Periods per year
    risk_free_rate : float, default 0.0
        Risk-free rate

    Returns
    -------
    dict[str, float]
        All calculated metrics
    """
    logger.info("Calculating performance metrics")

    metrics = {}

    # Return metrics
    metrics.update(calculate_returns_metrics(returns, periods_per_year))

    # Risk metrics
    metrics.update(calculate_risk_metrics(returns, periods_per_year, risk_free_rate))

    # Drawdown metrics
    metrics.update(calculate_drawdown_metrics(equity_curve))

    # Trade metrics
    if trades is not None and not trades.empty:
        metrics.update(calculate_trade_metrics(trades))

    # Additional metrics required by evaluator
    # Win rate - percentage of positive return days
    if len(returns) > 0:
        metrics["win_rate"] = (returns > 0).sum() / len(returns)
    else:
        metrics["win_rate"] = 0.0

    # Rolling Sharpe consistency - std of 60-day rolling Sharpe
    if len(returns) >= 60:
        rolling_window = 60
        rolling_returns = returns.rolling(rolling_window)
        rolling_mean = rolling_returns.mean() * periods_per_year
        rolling_std = rolling_returns.std() * np.sqrt(periods_per_year)
        rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std.replace(0, np.nan)
        rolling_sharpe = rolling_sharpe.dropna()
        metrics["rolling_sharpe_std"] = rolling_sharpe.std() if len(rolling_sharpe) > 0 else 1.0
    else:
        metrics["rolling_sharpe_std"] = 1.0

    # Tail risk - 5th percentile of daily returns
    if len(returns) > 0:
        metrics["return_quantile_05"] = returns.quantile(0.05)
    else:
        metrics["return_quantile_05"] = 0.0

    # Total trades count (for hard filter)
    metrics["total_trades"] = metrics.get("n_trades", 0)

    # Average daily turnover (for hard filter)
    if trades is not None and not trades.empty and len(equity_curve) > 0:
        # Turnover = total traded value / average portfolio value / number of days
        avg_portfolio_value = equity_curve.mean()
        total_traded = trades["notional"].abs().sum()
        n_days = len(equity_curve)
        metrics["avg_daily_turnover"] = (total_traded / avg_portfolio_value) / n_days if avg_portfolio_value > 0 and n_days > 0 else 0.0
    else:
        metrics["avg_daily_turnover"] = 0.0

    logger.info(
        f"Metrics - Return: {metrics['annual_return']:.2%}, "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
        f"Max DD: {metrics['max_drawdown']:.2%}"
    )

    return metrics


def print_metrics_summary(metrics: dict[str, float]) -> None:
    """Print formatted metrics summary.

    Parameters
    ----------
    metrics : dict[str, float]
        Metrics dictionary
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\nReturns:")
    print(f"  Total Return:        {metrics.get('total_return', 0):.2%}")
    print(f"  Annual Return:       {metrics.get('annual_return', 0):.2%}")
    print(f"  CAGR:                {metrics.get('cagr', 0):.2%}")

    print("\nRisk:")
    print(f"  Annual Volatility:   {metrics.get('volatility_annual', 0):.2%}")
    print(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio:        {metrics.get('calmar_ratio', 0):.2f}")

    print("\nDrawdown:")
    print(f"  Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Avg Drawdown:        {metrics.get('avg_drawdown', 0):.2%}")
    print(f"  Max DD Duration:     {metrics.get('max_dd_duration', 0):.0f} days")

    print("\nDistribution:")
    print(f"  Skewness:            {metrics.get('skewness', 0):.2f}")
    print(f"  Kurtosis:            {metrics.get('kurtosis', 0):.2f}")

    if "n_trades" in metrics:
        print("\nTrading:")
        print(f"  Number of Trades:    {metrics.get('n_trades', 0):.0f}")
        print(f"  Avg Trade Size:      ${metrics.get('avg_trade_size', 0):,.0f}")
        print(f"  Total Traded:        ${metrics.get('total_traded', 0):,.0f}")

    print("=" * 60 + "\n")
