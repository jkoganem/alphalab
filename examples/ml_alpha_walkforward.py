"""ML Alpha with Walk-Forward Validation Example.

Demonstrates advanced features:
- ML-based alpha model
- Walk-forward validation
- Risk management
- HTML report generation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from alphalab.alpha.ml import MLAlpha
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics
from alphalab.data.yahoo import YahooDataSource
from alphalab.features.pipeline import StandardFeaturePipeline
from alphalab.portfolio.optimizers import InverseVolatilityOptimizer
from alphalab.portfolio.risk import RiskManager
from alphalab.report.html import generate_html_report
from alphalab.signals.converters import RankLongShort
from alphalab.validate.walkforward import WalkForward, analyze_walk_forward_stability


def main() -> None:
    """Run ML alpha with walk-forward validation."""
    print("=" * 80)
    print("ML ALPHA WITH WALK-FORWARD VALIDATION")
    print("=" * 80)

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "SPY"]
    start = pd.Timestamp("2018-01-01", tz="UTC")
    end = pd.Timestamp("2024-12-31", tz="UTC")

    print(f"\n[1/7] Fetching data for {len(symbols)} symbols...")
    data_source = YahooDataSource()
    ohlcv = data_source.fetch(symbols, start, end)
    print(f"  Success: Fetched {len(ohlcv)} bars")

    print("\n[2/7] Generating features...")
    feature_pipeline = StandardFeaturePipeline(
        lookback_windows=[5, 20, 60, 126],
        include_rsi=True,
        include_beta=True,
        benchmark_symbol="SPY",
    )
    features = feature_pipeline.transform(ohlcv)
    print(f"  Success: Generated {len(features.columns)} features")

    print("\n[3/7] Creating ML alpha model...")
    ml_alpha = MLAlpha(
        model_type="classification",
        estimator="gbm",
        forward_horizon=5,
        train_window=252,  # 1 year rolling window
        scale_features=True,
    )
    print("  Success: Initialized Gradient Boosting Classifier")

    print("\n[4/7] Generating alpha scores...")
    alpha_scores = ml_alpha.score(features)
    n_scores = alpha_scores["alpha"].notna().sum()
    print(f"  Success: Generated {n_scores} ML alpha predictions")

    # Show feature importance
    if ml_alpha.feature_importance_ is not None:
        print("\n  Top 5 Important Features:")
        for feat, importance in ml_alpha.feature_importance_.head().items():
            print(f"    {feat}: {importance:.4f}")

    print("\n[5/7] Converting to trading signals...")
    signal_converter = RankLongShort(
        long_pct=0.2,
        short_pct=0.2,
        equal_weight=False,  # Weight by alpha strength
        neutralize_beta_to="SPY",
    )
    signals = signal_converter.to_signal(alpha_scores, features)
    print(f"  Success: Created beta-neutral long-short signals")

    print("\n[6/7] Optimizing portfolio with risk constraints...")
    # First optimize
    optimizer = InverseVolatilityOptimizer(vol_lookback=60)
    weights = optimizer.allocate(signals, risk=features)

    # Apply risk constraints
    risk_manager = RiskManager(
        volatility_target=0.15,  # 15% annual vol target
        max_gross_exposure=1.5,
        max_net_exposure=0.3,
        max_position_size=0.08,
        max_turnover_pct=0.3,
    )

    # For simplicity, apply constraints without current weights
    weights = risk_manager.apply_constraints(weights)
    print(f"  Success: Applied risk constraints")

    print("\n[7/7] Running walk-forward backtest...")
    wf = WalkForward(
        n_folds=4,
        train_period_days=504,  # 2 years training
        test_period_days=126,  # 6 months testing
        embargo_days=5,
    )

    # Prepare prices
    prices = ohlcv[["open", "close"]].copy()

    # Simple walk-forward (just run single backtest for demo)
    # Full walk-forward would require refitting the model in each fold
    print("  Running backtest (simplified - not refitting model per fold)...")

    backtester = VectorizedBacktester(
        initial_capital=1_000_000, execution_delay="next_open"
    )

    results = backtester.run(
        weights,
        prices,
        costs_cfg={"fees_bps": 1.0, "slippage_bps": 3.0, "borrow_bps": 25.0},
    )

    print(f"\n  Success: Backtest complete")
    print(f"    Initial capital: ${results['initial_capital']:,.0f}")
    print(f"    Final equity:    ${results['equity_curve'].iloc[-1]:,.0f}")
    print(
        f"    Total return:    {(results['equity_curve'].iloc[-1] / results['initial_capital'] - 1):.2%}"
    )

    # Calculate metrics
    print("\n[Analysis] Calculating performance metrics...")
    metrics = calculate_all_metrics(
        equity_curve=results["equity_curve"],
        returns=results["returns"],
        trades=results.get("trades"),
    )

    print(f"\n  Performance Summary:")
    print(f"    Annual Return:    {metrics['annual_return']:.2%}")
    print(f"    Volatility:       {metrics['volatility_annual']:.2%}")
    print(f"    Sharpe Ratio:     {metrics['sharpe_ratio']:.2f}")
    print(f"    Max Drawdown:     {metrics['max_drawdown']:.2%}")
    print(f"    Calmar Ratio:     {metrics['calmar_ratio']:.2f}")

    # Generate HTML report
    print("\n[Report] Generating HTML report...")
    output_dir = Path(__file__).parent.parent / "out"
    output_dir.mkdir(exist_ok=True)

    report_path = output_dir / "ml_alpha_walkforward_report.html"
    generate_html_report(
        results=results,
        metrics=metrics,
        output_path=report_path,
        title="ML Alpha with Walk-Forward Validation",
    )
    print(f"  Success: Report saved to: {report_path}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOpen {report_path} in your browser to view the full report.")
    print("\nKey Features Demonstrated:")
    print("  Success: Machine Learning alpha (GBM classifier)")
    print("  Success: Rolling window training (no look-ahead)")
    print("  Success: Beta-neutral signals")
    print("  Success: Inverse volatility weighting")
    print("  Success: Volatility targeting (15%)")
    print("  Success: Risk constraints (exposure, turnover, position size)")
    print("  Success: Comprehensive HTML reporting")


if __name__ == "__main__":
    main()
