"""Pairs Trading Strategy Example.

Demonstrates statistical arbitrage using cointegration-based pairs trading.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from alphalab.alpha.pairs import PairsTradingAlpha
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics, print_metrics_summary
from alphalab.data.yahoo import YahooDataSource
from alphalab.portfolio.optimizers import EqualWeightOptimizer
from alphalab.signals.converters import ScaledSignal


def main() -> None:
    """Run pairs trading backtest."""
    print("=" * 70)
    print("PAIRS TRADING STRATEGY - COINTEGRATION-BASED")
    print("=" * 70)

    # Configuration - using correlated stocks
    symbols = [
        "XOM",  # Exxon Mobil
        "CVX",  # Chevron (both energy)
        "JPM",  # JPMorgan
        "BAC",  # Bank of America (both banks)
        "KO",  # Coca-Cola
        "PEP",  # Pepsi (both beverages)
        "WMT",  # Walmart
        "TGT",  # Target (both retail)
        "SPY",  # S&P 500 ETF (benchmark)
    ]

    start = pd.Timestamp("2019-01-01", tz="UTC")
    end = pd.Timestamp("2024-12-31", tz="UTC")

    print(f"\n[1/6] Fetching data for {len(symbols)} symbols...")
    print("  (Selected correlated pairs from same sectors)")
    data_source = YahooDataSource()
    ohlcv = data_source.fetch(symbols, start, end)
    print(f"  Success: Fetched {len(ohlcv)} bars")

    print("\n[2/6] Extracting prices for pair identification...")
    # Get close prices
    prices = ohlcv["close"].copy()
    print(f"  Success: Extracted price series")

    print("\n[3/6] Identifying cointegrated pairs...")
    pairs_alpha = PairsTradingAlpha(
        formation_window=252,  # 1 year for cointegration test
        trading_window=20,  # 20-day rolling z-score
        entry_threshold=2.0,  # Enter when spread is 2 std devs away
        exit_threshold=0.5,  # Exit when spread reverts to 0.5 std devs
        stop_loss_threshold=4.0,  # Stop loss at 4 std devs
        min_half_life=5,  # Min 5 days mean reversion
        max_half_life=60,  # Max 60 days mean reversion
        significance_level=0.05,  # 5% significance for cointegration
    )

    # Create features DataFrame (pairs alpha needs it but mainly uses prices)
    features = pd.DataFrame({"close": prices}, index=prices.index)

    print("  Searching for cointegrated pairs...")
    print("  (This may take a minute...)")

    # Generate alpha scores
    alpha_scores = pairs_alpha.score(features, prices=prices)

    n_pairs = len(pairs_alpha.pairs)
    print(f"\n  Success: Found {n_pairs} cointegrated pairs:")
    for sym1, sym2, hedge_ratio in pairs_alpha.pairs:
        print(f"    - {sym1}/{sym2} (hedge ratio: {hedge_ratio:.4f})")

    if n_pairs == 0:
        print("\n  Warning:  No cointegrated pairs found with current parameters.")
        print("  This can happen with:")
        print("    - Short time periods")
        print("    - Uncorrelated stocks")
        print("    - Strict significance levels")
        print("\n  Exiting...")
        return

    n_signals = (alpha_scores["alpha"] != 0).sum()
    print(f"\n  Generated {n_signals} pairs trading signals")

    print("\n[4/6] Converting to position signals...")
    # Use scaled signals (already have directional information)
    signal_converter = ScaledSignal(scale_factor=1.0, normalize=False, clip_std=None)
    signals = signal_converter.to_signal(alpha_scores)
    print(f"  Success: Created position signals")

    print("\n[5/6] Constructing portfolio...")
    # Equal weight across active pairs positions
    optimizer = EqualWeightOptimizer(normalize=True)
    weights = optimizer.allocate(signals)
    n_positions = (weights["weight"] != 0).sum()
    print(f"  Success: Allocated {n_positions} positions")

    print("\n[6/6] Running backtest...")
    backtester = VectorizedBacktester(
        initial_capital=1_000_000, execution_delay="next_open"
    )

    # Prepare price data with open/close
    prices_df = ohlcv[["open", "close"]].copy()

    results = backtester.run(
        weights,
        prices_df,
        costs_cfg={
            "fees_bps": 2.0,  # Higher fees for more trading
            "slippage_bps": 5.0,  # Account for market impact
            "borrow_bps": 30.0,  # Short borrow costs
        },
    )

    print(f"\n  Success: Backtest complete")
    print(f"    Initial capital: ${results['initial_capital']:,.0f}")
    print(f"    Final equity:    ${results['equity_curve'].iloc[-1]:,.0f}")
    total_return = results['equity_curve'].iloc[-1] / results['initial_capital'] - 1
    print(f"    Total return:    {total_return:.2%}")

    # Calculate and display metrics
    print("\n" + "=" * 70)
    metrics = calculate_all_metrics(
        equity_curve=results["equity_curve"],
        returns=results["returns"],
        trades=results.get("trades"),
    )

    print_metrics_summary(metrics)

    # Additional pairs-specific analysis
    print("\n" + "=" * 70)
    print("PAIRS TRADING ANALYSIS")
    print("=" * 70)

    if n_pairs > 0:
        print(f"\nPairs Identified: {n_pairs}")
        print("\nPair Details:")
        for sym1, sym2, hedge_ratio in pairs_alpha.pairs:
            print(f"\n  {sym1} vs {sym2}")
            print(f"    Hedge Ratio:  {hedge_ratio:.4f}")
            print(f"    Strategy:     Long {sym1} when spread high,")
            print(f"                  Short {sym1} when spread low")

    # Strategy characteristics
    print(f"\nStrategy Parameters:")
    print(f"  Formation Window:   {pairs_alpha.formation_window} days")
    print(f"  Trading Window:     {pairs_alpha.trading_window} days")
    print(f"  Entry Threshold:    {pairs_alpha.entry_threshold:.1f} std devs")
    print(f"  Exit Threshold:     {pairs_alpha.exit_threshold:.1f} std devs")
    print(f"  Stop Loss:          {pairs_alpha.stop_loss_threshold:.1f} std devs")

    # Save results
    output_dir = Path(__file__).parent.parent / "out"
    output_dir.mkdir(exist_ok=True)

    results["equity_curve"].to_csv(output_dir / "pairs_equity_curve.csv")
    print(f"\nSuccess: Saved results to {output_dir}")

    print("\n" + "=" * 70)
    print("PAIRS TRADING BACKTEST COMPLETE")
    print("=" * 70)

    print("\nKey Insights:")
    print("  - Pairs trading exploits mean-reverting relationships")
    print("  - Cointegration ensures statistical validity")
    print("  - Market-neutral (long-short) reduces market risk")
    print("  - Requires careful monitoring of pair relationships")


if __name__ == "__main__":
    main()
