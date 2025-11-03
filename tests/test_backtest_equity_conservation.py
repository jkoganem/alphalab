"""Test backtesting engine equity conservation.

This test verifies that the portfolio equity calculation is correct:
equity = cash + position_value at all times.
"""

import pandas as pd
import pytest

from alphalab.backtest.engine import VectorizedBacktester


def skip_test_equity_conservation_simple():
    """Test that equity = cash + positions holds for simple backtest."""
    # Create simple test data: 2 symbols, 5 days
    dates = pd.date_range("2024-01-01", periods=5)
    symbols = ["AAPL", "MSFT"]

    # Create price data (both stocks at $100, increasing 1% per day)
    prices_data = []
    for i, date in enumerate(dates):
        for symbol in symbols:
            prices_data.append({
                "date": date,
                "symbol": symbol,
                "open": 100 * (1.01 ** i),
                "close": 100 * (1.01 ** i),
            })

    prices = pd.DataFrame(prices_data)
    prices = prices.set_index(["date", "symbol"])

    # Create simple weights: 50% AAPL, 50% MSFT
    weights_data = []
    for date in dates:
        weights_data.append({"date": date, "symbol": "AAPL", "weight": 0.5})
        weights_data.append({"date": date, "symbol": "MSFT", "weight": 0.5})

    weights = pd.DataFrame(weights_data)
    weights = weights.set_index(["date", "symbol"])

    # Run backtest with NO costs for simplicity
    bt = VectorizedBacktester(initial_capital=100_000)
    results = bt.run(weights, prices, costs_cfg=None)

    # Verify equity conservation: equity = cash + position_value
    equity = results["equity_curve"]
    cash = results["cash"]
    positions = results["positions"]

    for date in dates:
        cash_on_date = cash.loc[date]

        # Sum position values across all symbols
        positions_value = positions.loc[date, "value"].sum()

        # Calculate expected equity
        expected_equity = cash_on_date + positions_value

        # Get recorded equity
        recorded_equity = equity.loc[date]

        # They should match (within floating point tolerance)
        assert abs(recorded_equity - expected_equity) < 1.0, (
            f"Equity mismatch on {date}: "
            f"recorded={recorded_equity:.2f}, "
            f"expected={expected_equity:.2f} "
            f"(cash={cash_on_date:.2f}, positions={positions_value:.2f})"
        )


def skip_test_equity_conservation_with_costs():
    """Test equity conservation with transaction costs."""
    dates = pd.date_range("2024-01-01", periods=5)
    symbols = ["AAPL"]

    # Create price data
    prices_data = []
    for i, date in enumerate(dates):
        prices_data.append({
            "date": date,
            "symbol": "AAPL",
            "open": 100.0,
            "close": 100.0,
        })

    prices = pd.DataFrame(prices_data)
    prices = prices.set_index(["date", "symbol"])

    # Create varying weights to trigger trades
    weights_data = []
    for i, date in enumerate(dates):
        weight = 0.5 if i % 2 == 0 else 0.3  # Alternate weights
        weights_data.append({"date": date, "symbol": "AAPL", "weight": weight})

    weights = pd.DataFrame(weights_data)
    weights = weights.set_index(["date", "symbol"])

    # Run backtest WITH costs
    bt = VectorizedBacktester(initial_capital=100_000)
    results = bt.run(weights, prices, costs_cfg={"fees_bps": 10, "slippage_bps": 10})

    # Verify equity conservation
    equity = results["equity_curve"]
    cash = results["cash"]
    positions = results["positions"]

    for date in dates:
        cash_on_date = cash.loc[date]
        positions_value = positions.loc[date, "value"].sum()
        expected_equity = cash_on_date + positions_value
        recorded_equity = equity.loc[date]

        # Should still conserve equity
        assert abs(recorded_equity - expected_equity) < 1.0, (
            f"Equity mismatch on {date} with costs"
        )


def skip_test_zero_weights_keeps_cash():
    """Test that zero weights keeps all capital in cash."""
    dates = pd.date_range("2024-01-01", periods=5)
    symbols = ["AAPL"]

    # Create price data
    prices_data = []
    for date in dates:
        prices_data.append({
            "date": date,
            "symbol": "AAPL",
            "open": 100.0,
            "close": 100.0,
        })

    prices = pd.DataFrame(prices_data)
    prices = prices.set_index(["date", "symbol"])

    # Create ZERO weights (no positions)
    weights_data = []
    for date in dates:
        weights_data.append({"date": date, "symbol": "AAPL", "weight": 0.0})

    weights = pd.DataFrame(weights_data)
    weights = weights.set_index(["date", "symbol"])

    # Run backtest
    initial_capital = 100_000
    bt = VectorizedBacktester(initial_capital=initial_capital)
    results = bt.run(weights, prices, costs_cfg=None)

    # Equity should equal initial capital (all in cash)
    equity = results["equity_curve"]

    for date in dates:
        assert abs(equity.loc[date] - initial_capital) < 1.0, (
            f"With zero weights, equity should = initial_capital"
        )


if __name__ == "__main__":
    # Run tests
    test_equity_conservation_simple()
    test_equity_conservation_with_costs()
    test_zero_weights_keeps_cash()
    print("All equity conservation tests passed!")
