"""Simple end-to-end backtest example.

This script demonstrates how to use the Alpha Backtest Lab components together
to run a complete momentum strategy backtest.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from alphalab.alpha.momentum import CrossSectionalMomentum
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics, print_metrics_summary
from alphalab.data.yahoo import YahooDataSource
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.portfolio.optimizers import EqualWeightOptimizer
from alphalab.signals.converters import RankLongShort


def main() -> None:
    """Run simple backtest example."""
    print("=" * 70)
    print("ALPHA BACKTEST LAB - SIMPLE MOMENTUM STRATEGY EXAMPLE")
    print("=" * 70)

    # 1. Fetch data
    print("\n[1/6] Fetching market data...")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "SPY"]
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2024-12-31", tz="UTC")

    data_source = YahooDataSource()
    ohlcv = data_source.fetch(symbols, start, end, interval="1d")
    print(f"  Success: Fetched {len(ohlcv)} bars for {len(symbols)} symbols")

    # 2. Generate features
    print("\n[2/6] Generating features...")
    feature_pipeline = StandardFeaturePipeline(
        lookback_windows=[5, 20, 60, 126], benchmark_symbol="SPY"
    )
    features = feature_pipeline.transform(ohlcv)
    print(f"  Success: Generated {len(features.columns)} features")

    # 3. Generate alpha scores
    print("\n[3/6] Generating alpha scores...")
    alpha_model = CrossSectionalMomentum(
        lookback_days=126, neutralization="zscore", winsor_pct=0.01
    )
    alpha_scores = alpha_model.score(features)
    print(f"  Success: Generated {alpha_scores['alpha'].notna().sum()} alpha scores")

    # 4. Convert to signals
    print("\n[4/6] Converting to trading signals...")
    signal_converter = RankLongShort(
        long_pct=0.3, short_pct=0.3, equal_weight=True, neutralize_beta_to=None
    )
    signals = signal_converter.to_signal(alpha_scores, features)
    n_long = (signals["signal"] > 0).sum()
    n_short = (signals["signal"] < 0).sum()
    print(f"  Success: Generated {n_long} long and {n_short} short signals")

    # 5. Optimize portfolio
    print("\n[5/6] Optimizing portfolio weights...")
    optimizer = EqualWeightOptimizer(normalize=True)
    weights = optimizer.allocate(signals, risk=features)
    n_positions = (weights["weight"] != 0).sum()
    print(f"  Success: Allocated {n_positions} positions")

    # 6. Run backtest
    print("\n[6/6] Running backtest...")
    backtester = VectorizedBacktester(
        initial_capital=1_000_000, execution_delay="next_open"
    )

    # Prepare prices for backtest
    prices = ohlcv[["open", "close"]].copy()

    # Run backtest
    results = backtester.run(
        weights,
        prices,
        costs_cfg={"fees_bps": 1.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    print(f"  Success: Backtest complete")
    print(f"    Initial capital: ${results['initial_capital']:,.0f}")
    print(f"    Final equity:    ${results['equity_curve'].iloc[-1]:,.0f}")
    print(f"    Total return:    {(results['equity_curve'].iloc[-1] / results['initial_capital'] - 1):.2%}")

    # Calculate and print metrics
    print("\n" + "=" * 70)
    metrics = calculate_all_metrics(
        equity_curve=results["equity_curve"],
        returns=results["returns"],
        trades=results.get("trades"),
        periods_per_year=252,
    )

    print_metrics_summary(metrics)

    # Optional: Save results
    output_dir = Path(__file__).parent.parent / "out"
    output_dir.mkdir(exist_ok=True)

    results["equity_curve"].to_csv(output_dir / "equity_curve.csv")
    print(f"Success: Saved equity curve to {output_dir / 'equity_curve.csv'}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
