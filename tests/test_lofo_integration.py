"""Quick test to verify LOFO integration is working."""

import pandas as pd
import numpy as np
from src.alphalab.ai.ablation import LOFOAnalyzer
from src.alphalab.ai.spec_compiler import FactorSpecCompiler

# Create a simple spec
spec = {
    "family": "cross_sectional_rank",
    "factors": [
        {"name": "mom20", "expr": "ret_20d", "transforms": ["divide:vol_20d+0.01", "cs_robust_zscore", "winsor:-3,3"]},
        {"name": "rev5", "expr": "-ret_5d", "transforms": ["divide:vol_20d+0.01", "cs_robust_zscore", "winsor:-3,3"]},
    ],
    "combine": {"method": "weighted_mean", "weights": [0.6, 0.4]},
    "post": ["cs_rank_pct_centered", "clip:-2,2"],
    "rationale": "Test spec for LOFO",
}

# Create fake data
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=100, freq="D")
symbols = ["AAPL", "MSFT", "GOOGL"]
index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

features = pd.DataFrame({
    "ret_20d": np.random.randn(len(index)) * 0.02,
    "ret_5d": np.random.randn(len(index)) * 0.01,
    "vol_20d": np.abs(np.random.randn(len(index)) * 0.01) + 0.02,
    "close": 100 + np.random.randn(len(index)) * 5,
}, index=index)

data = pd.DataFrame({
    "open": features["close"] - np.random.randn(len(index)),
    "high": features["close"] + abs(np.random.randn(len(index))),
    "low": features["close"] - abs(np.random.randn(len(index))),
    "close": features["close"],
    "volume": np.abs(np.random.randn(len(index)) * 1000000),
}, index=index)

# Create backtest function (simplified)
def backtest_fn(spec_to_test, feats, dat):
    """Simplified backtest function."""
    compiler = FactorSpecCompiler()
    strategy_class = compiler.compile_to_class(spec_to_test, class_name="TestStrategy")
    strategy = strategy_class()

    # Generate alpha scores
    alpha_scores = strategy.score(feats)

    # Simple metrics
    return {
        "sharpe_ratio": np.random.uniform(0.3, 0.7),  # Fake Sharpe for testing
        "total_return": np.random.uniform(0.1, 0.3),
        "max_drawdown": np.random.uniform(-0.3, -0.1),
        "win_rate": np.random.uniform(0.45, 0.55),
    }

# Test LOFO
print("Testing LOFO integration...")
analyzer = LOFOAnalyzer()

try:
    results = analyzer.analyze_spec(spec, features, data, backtest_fn)
    print("\n[SUCCESS] LOFO analysis completed successfully!")
    print(f"\nFull metrics: Sharpe={results['full_metrics']['sharpe_ratio']:.2f}")
    print(f"\nAblations:")
    for factor_name, factor_results in results['ablations'].items():
        if 'deltas' in factor_results:
            delta_sharpe = factor_results['deltas'].get('sharpe_ratio_delta', 0.0)
            print(f"  {factor_name}: Delta_Sharpe = {delta_sharpe:+.3f}")

    # Test formatting
    print("\n" + "="*80)
    print("FORMATTED FEEDBACK:")
    print("="*80)
    feedback = analyzer.format_ablation_feedback(results)
    print(feedback)

except Exception as e:
    print(f"\n[FAIL] LOFO analysis failed: {e}")
    import traceback
    traceback.print_exc()
