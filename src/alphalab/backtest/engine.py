"""Vectorized backtesting engine.

This module implements a fast, vectorized backtesting engine that simulates
portfolio execution with realistic costs and constraints.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from alphalab.execution.costs import CompositeCostModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class VectorizedBacktester:
    """Vectorized backtesting engine.

    Simulates portfolio execution using vectorized operations for speed.
    Tracks positions, trades, cash, and equity over time.

    Parameters
    ----------
    initial_capital : float, default 1_000_000
        Starting capital
    execution_delay : str, default "next_open"
        When signals execute: "next_open", "next_close", or "same_close"
    cost_model : CompositeCostModel | None, optional
        Cost model for fees, slippage, and borrow costs

    Examples
    --------
    >>> bt = VectorizedBacktester(initial_capital=1_000_000)
    >>> results = bt.run(weights, prices, cost_config)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        execution_delay: str = "next_open",
        cost_model: CompositeCostModel | None = None,
    ) -> None:
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.execution_delay = execution_delay
        self.cost_model = cost_model

        if execution_delay not in {"next_open", "next_close", "same_close"}:
            msg = f"Invalid execution_delay: {execution_delay}"
            raise ValueError(msg)

        logger.info(
            f"Initialized VectorizedBacktester with capital=${initial_capital:,.0f}, "
            f"delay={execution_delay}"
        )

    def run(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        costs_cfg: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Run backtest simulation.

        Parameters
        ----------
        weights : pd.DataFrame
            Target portfolio weights with MultiIndex (date, symbol) and 'weight' column
        prices : pd.DataFrame
            Price data with MultiIndex (date, symbol)
            Should have 'open' and 'close' columns
        costs_cfg : dict[str, object] | None, optional
            Cost configuration (fees_bps, slippage_bps, etc.)

        Returns
        -------
        dict[str, object]
            Backtest results including:
            - equity_curve: pd.Series of portfolio value over time
            - returns: pd.Series of daily returns
            - positions: pd.DataFrame of positions over time
            - trades: pd.DataFrame of executed trades
            - costs: dict of cost breakdowns
            - metrics: dict of performance metrics
        """
        logger.info("Starting backtest simulation")

        # Initialize cost model if not provided
        if self.cost_model is None and costs_cfg is not None:
            self.cost_model = CompositeCostModel(**costs_cfg)  # type: ignore[arg-type]

        # Get execution prices based on delay
        exec_prices = self._get_execution_prices(prices)

        # Align weights with execution prices
        weights_aligned = weights.reindex(exec_prices.index, fill_value=0)

        # Initialize tracking variables
        dates = exec_prices.index.get_level_values("date").unique().sort_values()
        equity = pd.Series(self.initial_capital, index=dates, dtype=float)
        cash = pd.Series(self.initial_capital, index=dates, dtype=float)

        # Track positions (shares held)
        positions = pd.DataFrame(0.0, index=exec_prices.index, columns=["shares", "value"])

        # Track trades
        trades_list = []

        # Current position tracking
        current_positions = pd.Series(0.0, index=exec_prices.loc[dates[0]].index)
        current_cash = self.initial_capital

        # Simulate each day
        for i, date in enumerate(dates):
            # Get target weights for this date
            target_weights = weights_aligned.loc[date, "weight"]
            if isinstance(target_weights, float):
                # Single symbol case
                target_weights = pd.Series([target_weights], index=[target_weights.name])

            # CRITICAL FIX: Normalize weights to prevent excessive leverage
            weight_sum = target_weights.abs().sum()
            if weight_sum > 1.01:  # Allow 1% tolerance for rounding errors
                if i == 0:  # Log only once to avoid spam
                    logger.warning(
                        f"Normalizing weights from {weight_sum:.2f}x to 1.0x leverage. "
                        "Strategy should output normalized weights."
                    )
                target_weights = target_weights / weight_sum

            # Get execution prices for this date
            date_prices = exec_prices.loc[date]
            if isinstance(date_prices, float):
                date_prices = pd.Series([date_prices], index=[date_prices.name])

            # Validation: Check for NaN prices
            if date_prices.isna().any():
                nan_symbols = date_prices[date_prices.isna()].index.tolist()
                logger.warning(f"NaN prices found on {date} for symbols: {nan_symbols}")
                date_prices = date_prices.fillna(method='ffill').fillna(0)

            # Calculate target notional values
            portfolio_value = current_cash + (current_positions * date_prices).sum()
            target_notional = target_weights * portfolio_value

            # Calculate target shares
            target_shares = (target_notional / date_prices).fillna(0)

            # Calculate trades (difference from current positions)
            trades_shares = target_shares - current_positions.reindex(
                target_shares.index, fill_value=0
            )

            # Execute trades
            trade_notional = (trades_shares * date_prices).abs()
            total_trade_value = (trades_shares * date_prices).sum()

            # Apply costs (simplified - just fees for now)
            if self.cost_model is not None and trade_notional.sum() > 0:
                # Create orders DataFrame for cost calculation
                orders_df = pd.DataFrame(
                    {"quantity": trades_shares, "notional": trades_shares * date_prices},
                    index=pd.MultiIndex.from_product([[date], trades_shares.index]),
                )
                orders_df.index.names = ["date", "symbol"]

                prices_df = pd.DataFrame(
                    {"price": date_prices},
                    index=pd.MultiIndex.from_product([[date], date_prices.index]),
                )
                prices_df.index.names = ["date", "symbol"]

                daily_costs = self.cost_model.fixed_costs.estimate(orders_df, prices_df)
                trade_costs = daily_costs.sum() if len(daily_costs) > 0 else 0
            else:
                trade_costs = 0

            # Update cash
            current_cash = current_cash - total_trade_value - trade_costs

            # Validation: Warn on negative cash (leverage)
            if current_cash < 0:
                logger.warning(
                    f"Negative cash on {date}: ${current_cash:,.2f} "
                    f"(leverage used)"
                )

            # Update positions
            current_positions = target_shares

            # Validation: Check for excessive leverage
            weight_sum = target_weights.abs().sum()
            if weight_sum > 2.0:
                logger.warning(
                    f"High leverage on {date}: {weight_sum:.2f}x total exposure"
                )

            # Record positions (need to iterate for MultiIndex assignment)
            for symbol in current_positions.index:
                positions.loc[(date, symbol), "shares"] = current_positions[symbol]
                positions.loc[(date, symbol), "value"] = current_positions[symbol] * date_prices[symbol]

            # Calculate actual portfolio value after trades
            actual_portfolio_value = current_cash + (current_positions * date_prices).sum()

            # Record equity (cash + position value)
            equity.loc[date] = actual_portfolio_value
            cash.loc[date] = current_cash

            # Record trades
            if trades_shares.abs().sum() > 0:
                for symbol, shares in trades_shares[trades_shares != 0].items():
                    trades_list.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "shares": shares,
                            "price": date_prices[symbol],
                            "notional": shares * date_prices[symbol],
                        }
                    )

        # Calculate returns
        returns = equity.pct_change().fillna(0)

        # Create trades DataFrame
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        logger.info(
            f"Backtest complete: Final equity=${equity.iloc[-1]:,.0f}, "
            f"Total return={equity.iloc[-1]/self.initial_capital - 1:.2%}"
        )

        return {
            "equity_curve": equity,
            "returns": returns,
            "positions": positions,
            "trades": trades_df,
            "cash": cash,
            "costs": {"fees": trade_costs} if trade_costs > 0 else {},
            "initial_capital": self.initial_capital,
        }

    def _get_execution_prices(self, prices: pd.DataFrame) -> pd.Series:
        """Get execution prices based on delay setting.

        Parameters
        ----------
        prices : pd.DataFrame
            Price data with 'open' and/or 'close' columns

        Returns
        -------
        pd.Series
            Execution prices
        """
        if self.execution_delay == "next_open":
            if "open" not in prices.columns:
                logger.warning("'open' not in prices, using 'close' instead")
                exec_col = "close"
            else:
                exec_col = "open"
            # Shift forward by one day (signal at t executes at t+1)
            exec_prices = prices[exec_col].groupby(level="symbol").shift(-1)

        elif self.execution_delay == "next_close":
            if "close" not in prices.columns:
                msg = "'close' column required for next_close execution"
                raise ValueError(msg)
            exec_prices = prices["close"].groupby(level="symbol").shift(-1)

        else:  # same_close
            if "close" not in prices.columns:
                msg = "'close' column required for same_close execution"
                raise ValueError(msg)
            exec_prices = prices["close"]

        # Drop NaN (last date has no next price)
        exec_prices = exec_prices.dropna()

        return exec_prices


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    initial_capital: float = 1_000_000,
    execution_delay: str = "next_open",
    costs_cfg: dict[str, object] | None = None,
) -> dict[str, object]:
    """Convenience function to run a backtest.

    Parameters
    ----------
    weights : pd.DataFrame
        Target portfolio weights
    prices : pd.DataFrame
        Price data
    initial_capital : float, default 1_000_000
        Starting capital
    execution_delay : str, default "next_open"
        Execution timing
    costs_cfg : dict[str, object] | None, optional
        Cost configuration

    Returns
    -------
    dict[str, object]
        Backtest results
    """
    bt = VectorizedBacktester(
        initial_capital=initial_capital, execution_delay=execution_delay
    )
    return bt.run(weights, prices, costs_cfg)
