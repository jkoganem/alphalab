"""Generate publication-quality visualizations for comprehensive alpha strategy experiments.

This script creates visualizations comparing the performance of different alpha strategies
for inclusion in documentation and research presentations.
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

# Set publication-quality style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

# Configuration
DATA_FILE = Path(__file__).parent.parent / "data" / "stocks_7y_2025.parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "experiments"
START_DATE = pd.Timestamp("2019-01-01", tz="UTC")
END_DATE = pd.Timestamp("2024-12-31", tz="UTC")
INITIAL_CAPITAL = 1_000_000


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and prepare data."""
    print("Loading data...")
    ohlcv = pd.read_parquet(DATA_FILE)
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv.loc[START_DATE:END_DATE]

    pipeline = StandardFeaturePipeline()
    features = pipeline.transform(ohlcv)

    return ohlcv, features


def run_strategies(ohlcv: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Run all strategies and collect results."""
    print("Running backtests...")

    results = {}

    # 1. Time-Series Momentum
    print("  - Time-Series Momentum...")
    alpha = TimeSeriesMomentum(lookback_days=126).score(features)
    signals = ThresholdSignal(long_threshold=0.5, short_threshold=-0.5).to_signal(alpha)
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)
    backtester = VectorizedBacktester(initial_capital=INITIAL_CAPITAL, execution_delay="next_open")
    backtest_results = backtester.run(
        weights, ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 1.0, "slippage_bps": 3.0, "borrow_bps": 0.0}
    )
    metrics = calculate_all_metrics(backtest_results["equity_curve"], backtest_results["returns"])
    results["Time-Series Momentum"] = {"results": backtest_results, "metrics": metrics, "weights": weights}

    # 2. Cross-Sectional Momentum
    print("  - Cross-Sectional Momentum...")
    alpha = CrossSectionalMomentum(lookback_days=126, neutralization="zscore").score(features)
    signals = RankLongShort(long_pct=0.3, short_pct=0.3, neutralize_beta_to=None).to_signal(alpha, features)
    weights = InverseVolatilityOptimizer(vol_lookback=60).allocate(signals, features)
    backtester = VectorizedBacktester(initial_capital=INITIAL_CAPITAL, execution_delay="next_open")
    backtest_results = backtester.run(
        weights, ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0}
    )
    metrics = calculate_all_metrics(backtest_results["equity_curve"], backtest_results["returns"])
    results["Cross-Sectional Momentum"] = {"results": backtest_results, "metrics": metrics, "weights": weights}

    # 3. Mean Reversion
    print("  - Mean Reversion...")
    alpha = MeanReversion(lookback_days=20, entry_threshold=2.0).score(features)
    signals = ScaledSignal(scale_factor=1.0, normalize=False).to_signal(alpha)
    weights = EqualWeightOptimizer(normalize=True).allocate(signals)
    backtester = VectorizedBacktester(initial_capital=INITIAL_CAPITAL, execution_delay="next_open")
    backtest_results = backtester.run(
        weights, ohlcv[["open", "close"]],
        costs_cfg={"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0}
    )
    metrics = calculate_all_metrics(backtest_results["equity_curve"], backtest_results["returns"])
    results["Mean Reversion"] = {"results": backtest_results, "metrics": metrics, "weights": weights}

    return results


def create_equity_curve_plot(results: dict, output_path: Path):
    """Create equity curve comparison plot."""
    print("Creating equity curve plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for (name, data), color in zip(results.items(), colors):
        equity = data["results"]["equity_curve"]
        equity_normalized = equity / INITIAL_CAPITAL
        ax.plot(equity.index, equity_normalized, label=name, linewidth=2, color=color, alpha=0.9)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Normalized Equity (Base = 1.0)", fontweight="bold")
    ax.set_title("Strategy Performance Comparison (2019-2024)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def create_drawdown_plot(results: dict, output_path: Path):
    """Create drawdown comparison plot."""
    print("Creating drawdown plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for (name, data), color in zip(results.items(), colors):
        equity = data["results"]["equity_curve"]
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        ax.fill_between(drawdown.index, 0, drawdown * 100, label=name, alpha=0.6, color=color)

    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("Drawdown (%)", fontweight="bold")
    ax.set_title("Strategy Drawdown Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([ax.get_ylim()[0], 5])

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def create_metrics_comparison_bar(results: dict, output_path: Path):
    """Create bar chart comparing key metrics."""
    print("Creating metrics comparison bar chart...")

    metrics_data = []
    for name, data in results.items():
        m = data["metrics"]
        metrics_data.append({
            "Strategy": name,
            "Sharpe Ratio": m["sharpe_ratio"],
            "Sortino Ratio": m["sortino_ratio"],
            "Calmar Ratio": m["calmar_ratio"]
        })

    df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.25

    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    ax.bar(x - width, df["Sharpe Ratio"], width, label="Sharpe Ratio", color=colors[0], alpha=0.8)
    ax.bar(x, df["Sortino Ratio"], width, label="Sortino Ratio", color=colors[1], alpha=0.8)
    ax.bar(x + width, df["Calmar Ratio"], width, label="Calmar Ratio", color=colors[2], alpha=0.8)

    ax.set_xlabel("Strategy", fontweight="bold")
    ax.set_ylabel("Ratio Value", fontweight="bold")
    ax.set_title("Risk-Adjusted Performance Metrics", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Strategy"], rotation=15, ha="right")
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def create_returns_distribution(results: dict, output_path: Path):
    """Create returns distribution violin plot."""
    print("Creating returns distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    returns_data = []
    for name, data in results.items():
        returns = data["results"]["returns"] * 100  # Convert to %
        for r in returns:
            returns_data.append({"Strategy": name, "Daily Return (%)": r})

    df = pd.DataFrame(returns_data)

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    palette = dict(zip(results.keys(), colors))

    sns.violinplot(data=df, x="Strategy", y="Daily Return (%)", palette=palette, ax=ax, inner="box")

    ax.set_xlabel("Strategy", fontweight="bold")
    ax.set_ylabel("Daily Return (%)", fontweight="bold")
    ax.set_title("Distribution of Daily Returns", fontsize=14, fontweight="bold", pad=20)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def create_rolling_sharpe(results: dict, output_path: Path):
    """Create rolling Sharpe ratio plot."""
    print("Creating rolling Sharpe plot...")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    window = 60  # 60-day rolling window

    for (name, data), color in zip(results.items(), colors):
        returns = data["results"]["returns"]
        rolling_sharpe = (
            returns.rolling(window).mean() * 252 /
            (returns.rolling(window).std() * np.sqrt(252))
        )
        ax.plot(rolling_sharpe.index, rolling_sharpe, label=name, linewidth=2, color=color, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Date", fontweight="bold")
    ax.set_ylabel("60-Day Rolling Sharpe Ratio", fontweight="bold")
    ax.set_title("Rolling Sharpe Ratio Over Time", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="best", frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def create_summary_table_image(results: dict, output_path: Path):
    """Create summary statistics table as image."""
    print("Creating summary table image...")

    summary_data = []
    for name, data in results.items():
        m = data["metrics"]
        summary_data.append({
            "Strategy": name,
            "Total Return": f"{m['total_return']:.2%}",
            "Annual Return": f"{m['annual_return']:.2%}",
            "Sharpe": f"{m['sharpe_ratio']:.3f}",
            "Sortino": f"{m['sortino_ratio']:.3f}",
            "Max DD": f"{m['max_drawdown']:.2%}",
            "Win Rate": f"{m['win_rate']:.2%}"
        })

    df = pd.DataFrame(summary_data)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        if i % 2 == 0:
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  Saved to: {output_path}")


def main():
    """Generate all visualizations."""
    print("\n" + "=" * 80)
    print("GENERATING EXPERIMENT VISUALIZATIONS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch data and run strategies
    ohlcv, features = fetch_data()
    results = run_strategies(ohlcv, features)

    print("\nGenerating visualizations...")

    # Create all plots
    create_equity_curve_plot(results, OUTPUT_DIR / "equity_curves.png")
    create_drawdown_plot(results, OUTPUT_DIR / "drawdowns.png")
    create_metrics_comparison_bar(results, OUTPUT_DIR / "metrics_comparison.png")
    create_returns_distribution(results, OUTPUT_DIR / "returns_distribution.png")
    create_rolling_sharpe(results, OUTPUT_DIR / "rolling_sharpe.png")
    create_summary_table_image(results, OUTPUT_DIR / "summary_table.png")

    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  - equity_curves.png: Normalized equity curve comparison")
    print("  - drawdowns.png: Drawdown comparison over time")
    print("  - metrics_comparison.png: Risk-adjusted performance metrics")
    print("  - returns_distribution.png: Daily returns distribution")
    print("  - rolling_sharpe.png: Rolling Sharpe ratio over time")
    print("  - summary_table.png: Summary statistics table")


if __name__ == "__main__":
    main()
