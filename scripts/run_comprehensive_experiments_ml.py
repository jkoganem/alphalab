"""Comprehensive experimental analysis of multiple alpha strategies.

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
from alphalab.alpha.ml import MLAlpha
from alphalab.alpha.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from alphalab.alpha.pairs import PairsTradingAlpha
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics
from alphalab.data.yahoo import YahooDataSource
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.portfolio.optimizers import (
    EqualWeightOptimizer,
    InverseVolatilityOptimizer,
)
from alphalab.portfolio.risk import RiskManager
from alphalab.report.json_export import (
    export_backtest_results,
    export_comparison_results,
)
from alphalab.signals.converters import RankLongShort, ScaledSignal, ThresholdSignal

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# Global configuration
# Load symbols from the same data source used by LLM experiments (149 symbols)
DATA_FILE = Path(__file__).parent.parent / "data" / "stocks_7y_2025.parquet"
START_DATE = pd.Timestamp("2019-01-01", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL_CAPITAL = 1_000_000


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and prepare data for all experiments."""
    print("\n" + "=" * 80)
    print("FETCHING DATA")
    print("=" * 80)

    # Load data from parquet file (same as LLM experiments use)
    print(f"\nLoading data from {DATA_FILE}...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Data file not found at {DATA_FILE}. "
            "Run: python scripts/download_data.py"
        )

    ohlcv = pd.read_parquet(DATA_FILE)
    ohlcv = ohlcv.sort_index()

    # Filter to date range
    ohlcv = ohlcv.loc[START_DATE:END_DATE]

    symbols = ohlcv.index.get_level_values("symbol").unique().tolist()
    print(f"PASS Loaded {len(ohlcv)} bars for {len(symbols)} symbols")

    # Use existing data instead of re-fetching from Yahoo
    # data_source = YahooDataSource()
    # ohlcv = data_source.fetch(symbols, START_DATE, END_DATE)
    # print(f"PASS Fetched {len(ohlcv)} bars")

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

    # Generate alpha
    alpha = TimeSeriesMomentum(lookback_days=126).score(features)

    # Convert to signals
    signals = ThresholdSignal(long_threshold=0.5, short_threshold=-0.5).to_signal(alpha)

    # Equal weight portfolio
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)

    # Backtest
    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 1.0, "slippage_bps": 3.0, "borrow_bps": 0.0},
    )

    # Calculate metrics
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

    # Generate alpha
    alpha = CrossSectionalMomentum(
        lookback_days=126, neutralization="zscore"
    ).score(features)

    # Convert to long-short signals
    signals = RankLongShort(
        long_pct=0.3, short_pct=0.3, neutralize_beta_to=None
    ).to_signal(alpha, features)

    # Risk-weighted portfolio
    weights = InverseVolatilityOptimizer(vol_lookback=60).allocate(signals, features)

    # Backtest
    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    # Calculate metrics
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

    # Generate alpha
    alpha = MeanReversion(lookback_days=20, entry_threshold=2.0).score(features)

    # Convert to signals
    signals = ScaledSignal(scale_factor=1.0, normalize=False).to_signal(alpha)

    # Equal weight portfolio
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)

    # Backtest
    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    # Calculate metrics
    metrics = calculate_all_metrics(
        results["equity_curve"], results["returns"], results.get("trades")
    )

    print(f"PASS Total Return: {metrics['total_return']:.2%}")
    print(f"PASS Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"PASS Max Drawdown: {metrics['max_drawdown']:.2%}")

    return {"results": results, "metrics": metrics, "weights": weights}


def run_ml_alpha_strategy(
    ohlcv: pd.DataFrame, features: pd.DataFrame
) -> dict[str, any]:
    """Run machine learning alpha strategy."""
    print("\n" + "=" * 80)
    print("STRATEGY 4: MACHINE LEARNING ALPHA")
    print("=" * 80)

    print("Training ML model with rolling window...")
    # Generate ML alpha
    ml_alpha = MLAlpha(
        model_type="regression",  # Use regression instead of classification for better signals
        estimator="gbm",
        forward_horizon=1,  # Predict 1-day ahead (more reliable than 5-day)
        train_window=252,
    )
    alpha = ml_alpha.score(features)

    # Convert to long-short signals
    signals = RankLongShort(long_pct=0.2, short_pct=0.2).to_signal(alpha, features)

    # Risk-weighted with constraints
    weights = InverseVolatilityOptimizer(vol_lookback=60).allocate(signals, features)

    # Apply risk management
    risk_mgr = RiskManager(
        volatility_target=0.15, max_gross_exposure=2.0, max_position_size=0.1
    )
    weights = risk_mgr.apply_constraints(weights, features)

    # Backtest
    backtester = VectorizedBacktester(
        initial_capital=INITIAL_CAPITAL, execution_delay="next_open"
    )
    results = backtester.run(
        weights,
        ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0},
    )

    # Calculate metrics
    metrics = calculate_all_metrics(
        results["equity_curve"], results["returns"], results.get("trades")
    )

    print(f"PASS Total Return: {metrics['total_return']:.2%}")
    print(f"PASS Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"PASS Max Drawdown: {metrics['max_drawdown']:.2%}")

    return {"results": results, "metrics": metrics, "weights": weights}


def generate_comparison_plots(all_results: dict[str, dict]) -> None:
    """Generate comprehensive comparison plots."""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / "outputs" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Equity Curves Comparison
    print("\n[1/7] Plotting equity curves...")
    plt.figure(figsize=(14, 7))
    for strategy_name, data in all_results.items():
        equity = data["results"]["equity_curve"]
        plt.plot(equity.index, equity.values, label=strategy_name, linewidth=2)

    plt.title("Equity Curves Comparison (2019-2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves_comparison.png", dpi=300)
    plt.close()
    print(f"  PASS Saved equity_curves_comparison.png")

    # 2. Normalized Returns Comparison
    print("[2/7] Plotting normalized returns...")
    plt.figure(figsize=(14, 7))
    for strategy_name, data in all_results.items():
        equity = data["results"]["equity_curve"]
        normalized = (equity / equity.iloc[0]) * 100
        plt.plot(normalized.index, normalized.values, label=strategy_name, linewidth=2)

    plt.title(
        "Normalized Returns Comparison (Base = 100)", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_returns_comparison.png", dpi=300)
    plt.close()
    print(f"  PASS Saved normalized_returns_comparison.png")

    # 3. Drawdown Comparison
    print("[3/7] Plotting drawdowns...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (strategy_name, data) in enumerate(all_results.items()):
        equity = data["results"]["equity_curve"]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        axes[idx].fill_between(
            drawdown.index, 0, drawdown.values * 100, alpha=0.5, color="red"
        )
        axes[idx].set_title(f"{strategy_name} Drawdown", fontweight="bold")
        axes[idx].set_xlabel("Date")
        axes[idx].set_ylabel("Drawdown (%)")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "drawdowns_comparison.png", dpi=300)
    plt.close()
    print(f"  PASS Saved drawdowns_comparison.png")

    # 4. Performance Metrics Bar Chart
    print("[4/7] Plotting performance metrics...")
    metrics_to_plot = ["total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio"]
    metric_labels = ["Total Return", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        values = [data["metrics"][metric] for data in all_results.values()]
        strategies = list(all_results.keys())

        bars = axes[idx].bar(strategies, values, alpha=0.7, edgecolor="black")

        # Color bars by value
        colors = ["green" if v > 0 else "red" for v in values]
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        axes[idx].set_title(label, fontweight="bold")
        axes[idx].set_ylabel(label)
        axes[idx].grid(True, alpha=0.3, axis="y")
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].axhline(y=0, color="black", linestyle="-", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "performance_metrics.png", dpi=300)
    plt.close()
    print(f"  PASS Saved performance_metrics.png")

    # 5. Risk Metrics Comparison
    print("[5/7] Plotting risk metrics...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Max Drawdown
    strategies = list(all_results.keys())
    max_dd = [abs(data["metrics"]["max_drawdown"]) * 100 for data in all_results.values()]

    axes[0].bar(strategies, max_dd, alpha=0.7, color="orange", edgecolor="black")
    axes[0].set_title("Maximum Drawdown (%)", fontweight="bold")
    axes[0].set_ylabel("Drawdown (%)")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=45)

    # Volatility
    volatility = [
        data["metrics"]["volatility_annual"] * 100 for data in all_results.values()
    ]
    axes[1].bar(strategies, volatility, alpha=0.7, color="purple", edgecolor="black")
    axes[1].set_title("Annual Volatility (%)", fontweight="bold")
    axes[1].set_ylabel("Volatility (%)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "risk_metrics.png", dpi=300)
    plt.close()
    print(f"  PASS Saved risk_metrics.png")

    # 6. Rolling Sharpe Ratio
    print("[6/7] Plotting rolling Sharpe ratios...")
    plt.figure(figsize=(14, 7))

    for strategy_name, data in all_results.items():
        returns = data["results"]["returns"]
        rolling_sharpe = (
            returns.rolling(window=252).mean()
            / returns.rolling(window=252).std()
            * np.sqrt(252)
        )
        plt.plot(
            rolling_sharpe.index, rolling_sharpe.values, label=strategy_name, linewidth=2
        )

    plt.title(
        "Rolling 1-Year Sharpe Ratio Comparison", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_sharpe.png", dpi=300)
    plt.close()
    print(f"  PASS Saved rolling_sharpe.png")

    # 7. Monthly Returns Heatmap (for best strategy)
    print("[7/7] Plotting monthly returns heatmap...")
    best_strategy = max(
        all_results.items(), key=lambda x: x[1]["metrics"]["sharpe_ratio"]
    )
    strategy_name = best_strategy[0]
    returns = best_strategy[1]["results"]["returns"]

    # Create monthly returns pivot
    monthly_returns = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    monthly_pivot = monthly_returns.to_frame("returns")
    monthly_pivot["year"] = monthly_pivot.index.year
    monthly_pivot["month"] = monthly_pivot.index.month
    pivot_table = monthly_pivot.pivot(
        index="year", columns="month", values="returns"
    )
    pivot_table = pivot_table * 100  # Convert to percentage

    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Monthly Return (%)"},
        linewidths=0.5,
    )
    plt.title(
        f"Monthly Returns Heatmap: {strategy_name}", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.savefig(output_dir / "monthly_returns_heatmap.png", dpi=300)
    plt.close()
    print(f"  PASS Saved monthly_returns_heatmap.png")

    print(f"\nPASS All visualizations saved to {output_dir}")


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
                "Volatility": f"{metrics['annual_volatility']:.2%}",
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
    all_results["ML Alpha"] = run_ml_alpha_strategy(ohlcv, features)

    # Generate comparison plots
    generate_comparison_plots(all_results)

    # Generate summary table
    summary_df = generate_summary_table(all_results)

    # Export to JSON
    print("\n" + "=" * 80)
    print("EXPORTING RESULTS TO JSON")
    print("=" * 80)

    output_dir = Path(__file__).parent.parent / "outputs" / "experiments"

    for strategy_name, data in all_results.items():
        filename = strategy_name.lower().replace(" ", "_").replace("-", "_")
        export_backtest_results(
            results=data["results"],
            metrics=data["metrics"],
            output_path=output_dir / f"{filename}_results.json",
            metadata={"strategy": strategy_name, "symbols": SYMBOLS},
        )

    # Export comparison
    comparison_data = {
        name: {"metrics": data["metrics"], "equity_curve": data["results"]["equity_curve"]}
        for name, data in all_results.items()
    }
    export_comparison_results(
        comparison_data, output_dir / "strategy_comparison.json"
    )

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {output_dir}")
    print("\nKey Findings:")

    best_sharpe = max(all_results.items(), key=lambda x: x[1]["metrics"]["sharpe_ratio"])
    best_return = max(all_results.items(), key=lambda x: x[1]["metrics"]["total_return"])
    lowest_dd = min(all_results.items(), key=lambda x: abs(x[1]["metrics"]["max_drawdown"]))

    print(f"  • Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['metrics']['sharpe_ratio']:.3f})")
    print(f"  • Best Total Return: {best_return[0]} ({best_return[1]['metrics']['total_return']:.2%})")
    print(f"  • Lowest Max Drawdown: {lowest_dd[0]} ({lowest_dd[1]['metrics']['max_drawdown']:.2%})")


if __name__ == "__main__":
    main()
