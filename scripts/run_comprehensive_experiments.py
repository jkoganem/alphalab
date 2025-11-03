"""Comprehensive experimental analysis of momentum and mean reversion alpha strategies.

This script runs backtests across different alpha models, compares their performance,
and generates detailed visualizations for academic reporting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from alphalab.alpha.mean_reversion import MeanReversion
from alphalab.alpha.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.portfolio.optimizers import (
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from alphalab.signals.converters import RankLongShort, ScaledSignal, ThresholdSignal

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Global configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "stocks_7y_2025.parquet"
START_DATE = pd.Timestamp("2019-01-01", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL_CAPITAL = 1_000_000


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and prepare data for all experiments."""
    print("\n" + "=" * 80)
    print("FETCHING DATA")
    print("=" * 80)

    print(f"\nLoading data from {DATA_FILE}...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_FILE}. "
            "Run: python scripts/download_data.py"
        )

    ohlcv = pd.read_parquet(DATA_FILE)
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv.loc[START_DATE:END_DATE]

    symbols = ohlcv.index.get_level_values("symbol").unique().tolist()
    print(f"PASS Loaded {len(ohlcv)} bars for {len(symbols)} symbols")

    print("\nGenerating features...")
    pipeline = StandardFeaturePipeline()
    features = pipeline.transform(ohlcv)
    print(f"PASS Generated {len(features.columns)} features")

    return ohlcv, features


def run_time_series_momentum(
    ohlcv: pd.DataFrame, features: pd.DataFrame
) -> dict[str, any]:
    """Run time-series momentum strategy."""
    print("\n" + "=" * 80)
    print("STRATEGY 1: TIME-SERIES MOMENTUM")
    print("=" * 80)

    alpha = TimeSeriesMomentum(lookback_days=126).score(features)
    signals = ThresholdSignal(long_threshold=0.5, short_threshold=-0.5).to_signal(alpha)
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)

    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 1.0, "slippage_bps": 3.0, "borrow_bps": 0.0},
    )

    metrics = calculate_all_metrics(
        results["equity_curve"], results["returns"], results.get("trades")
    )

    print(f"PASS Total Return: {metrics['total_return']:.2%}")
    print(f"PASS Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"PASS Max Drawdown: {metrics['max_drawdown']:.2%}")

    return {"results": results, "metrics": metrics, "weights": weights}


def run_cross_sectional_momentum(
    ohlcv: pd.DataFrame, features: pd.DataFrame
) -> dict[str, any]:
    """Run cross-sectional momentum strategy."""
    print("\n" + "=" * 80)
    print("STRATEGY 2: CROSS-SECTIONAL MOMENTUM (LONG-SHORT)")
    print("=" * 80)

    alpha = CrossSectionalMomentum(
        lookback_days=126, neutralization="zscore"
    ).score(features)

    signals = RankLongShort(
        long_pct=0.3, short_pct=0.3, neutralize_beta_to=None
    ).to_signal(alpha, features)

    weights = InverseVolatilityOptimizer(vol_lookback=60).allocate(signals, features)

    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    metrics = calculate_all_metrics(
        results["equity_curve"], results["returns"], results.get("trades")
    )

    print(f"PASS Total Return: {metrics['total_return']:.2%}")
    print(f"PASS Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"PASS Max Drawdown: {metrics['max_drawdown']:.2%}")

    return {"results": results, "metrics": metrics, "weights": weights}


def run_mean_reversion(
    ohlcv: pd.DataFrame, features: pd.DataFrame
) -> dict[str, any]:
    """Run mean reversion strategy."""
    print("\n" + "=" * 80)
    print("STRATEGY 3: MEAN REVERSION")
    print("=" * 80)

    alpha = MeanReversion(lookback_days=20, entry_threshold=2.0).score(features)
    signals = ScaledSignal(scale_factor=1.0, normalize=False).to_signal(alpha)
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)

    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    metrics = calculate_all_metrics(
        results["equity_curve"], results["returns"], results.get("trades")
    )

    print(f"PASS Total Return: {metrics['total_return']:.2%}")
    print(f"PASS Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"PASS Max Drawdown: {metrics['max_drawdown']:.2%}")

    return {"results": results, "metrics": metrics, "weights": weights}


def generate_summary_table(all_results: dict[str, dict]) -> pd.DataFrame:
    """Generate summary statistics table."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS TABLE")
    print("=" * 80)

    summary_data = []
    for strategy_name, data in all_results.items():
        metrics = data["metrics"]
        summary_data.append(
            {
                "Strategy": strategy_name,
                "Total Return": f"{metrics['total_return']:.2%}",
                "Annual Return": f"{metrics['annual_return']:.2%}",
                "Volatility": f"{metrics['volatility_annual']:.2%}",
                "Sharpe Ratio": f"{metrics['sharpe_ratio']:.3f}",
                "Sortino Ratio": f"{metrics['sortino_ratio']:.3f}",
                "Calmar Ratio": f"{metrics['calmar_ratio']:.3f}",
                "Max Drawdown": f"{metrics['max_drawdown']:.2%}",
                "Win Rate": f"{metrics.get('win_rate', 0):.2%}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))

    # Save to CSV
    output_dir = Path(__file__).parent.parent / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    print(f"\nPASS Saved summary table to {output_dir / 'summary_statistics.csv'}")

    return summary_df


def main() -> None:
    """Run all experiments and generate visualizations."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ALPHA STRATEGY EXPERIMENTS")
    print("=" * 80)
    print(f"\nInitial Capital: ${INITIAL_CAPITAL:,.0f}")
    print(f"Date Range: {START_DATE.date()} to {END_DATE.date()}")

    # Fetch data
    ohlcv, features = fetch_data()

    # Get symbol count from data
    n_symbols = len(ohlcv.index.get_level_values("symbol").unique())
    print(f"Universe: {n_symbols} symbols")

    # Run all strategies
    all_results = {}

    all_results["Time-Series Momentum"] = run_time_series_momentum(ohlcv, features)
    all_results["Cross-Sectional Momentum"] = run_cross_sectional_momentum(
        ohlcv, features
    )
    all_results["Mean Reversion"] = run_mean_reversion(ohlcv, features)

    # Generate summary table
    summary_df = generate_summary_table(all_results)

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {Path(__file__).parent.parent / 'outputs' / 'experiments'}")
    print("\nKey Findings:")

    best_sharpe = max(all_results.items(), key=lambda x: x[1]["metrics"]["sharpe_ratio"])
    best_return = max(all_results.items(), key=lambda x: x[1]["metrics"]["total_return"])
    lowest_dd = min(all_results.items(), key=lambda x: abs(x[1]["metrics"]["max_drawdown"]))

    print(f"  • Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.3f})")
    print(f"  • Best Total Return: {best_return[0]} ({best_return[1]['metrics']['total_return']:.2%})")
    print(f"  • Lowest Max Drawdown: {lowest_dd[0]} ({lowest_dd[1]['metrics']['max_drawdown']:.2%})")


if __name__ == "__main__":
    main()
