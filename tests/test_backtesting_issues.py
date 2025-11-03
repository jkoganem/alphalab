#!/usr/bin/env python3
"""Test script to identify critical issues in the backtesting engine."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create simple test data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Create price data (MultiIndex: date, symbol)
price_data = []
for date in dates:
    for symbol in symbols:
        price_data.append({
            'date': date,
            'symbol': symbol,
            'open': 100 + np.random.randn() * 5,
            'close': 100 + np.random.randn() * 5,
            'high': 105 + np.random.randn() * 5,
            'low': 95 + np.random.randn() * 5,
            'volume': 1000000
        })

prices_df = pd.DataFrame(price_data)
prices_df = prices_df.set_index(['date', 'symbol'])

print("=" * 80)
print("TEST 1: EXCESSIVE LEVERAGE DUE TO UNNORMALIZED WEIGHTS")
print("=" * 80)

# Create weights that sum to much more than 1 (like the premium seeds)
# This mimics what cs_rank_pct_centered + clip:-2,2 produces
weight_data = []
for date in dates:
    for i, symbol in enumerate(symbols):
        # Create weights that range from -2 to 2 (like clipped centered ranks)
        weight = (i - 2) * 0.8  # Will create weights: -1.6, -0.8, 0, 0.8, 1.6
        weight_data.append({
            'date': date,
            'symbol': symbol,
            'weight': weight
        })

weights_df = pd.DataFrame(weight_data)
weights_df = weights_df.set_index(['date', 'symbol'])

print(f"\nSample weights for first date:")
first_date_weights = weights_df.loc[dates[0]]
print(first_date_weights)
print(f"\nSum of absolute weights: {first_date_weights['weight'].abs().sum():.2f}")
print(f"Sum of weights: {first_date_weights['weight'].sum():.2f}")

# Now run backtest with these weights
from alphalab.backtest.engine import VectorizedBacktester

backtester = VectorizedBacktester(initial_capital=100000)
results = backtester.run(weights_df, prices_df)

print(f"\nBacktest Results:")
print(f"Initial capital: ${backtester.initial_capital:,.0f}")
print(f"Final equity: ${results['equity_curve'].iloc[-1]:,.0f}")
print(f"Total return: {(results['equity_curve'].iloc[-1] / backtester.initial_capital - 1) * 100:.1f}%")

# Check for negative cash (leverage indicator)
cash_series = results['cash']
min_cash = cash_series.min()
print(f"Minimum cash during backtest: ${min_cash:,.0f}")
if min_cash < 0:
    print("WARNING: Negative cash indicates leverage is being used!")
    max_negative_cash = abs(cash_series[cash_series < 0].min())
    print(f"   Maximum leverage: ${max_negative_cash:,.0f}")

print("\n" + "=" * 80)
print("TEST 2: NORMALIZED WEIGHTS (WHAT IT SHOULD BE)")
print("=" * 80)

# Create properly normalized weights
normalized_weight_data = []
for date in dates:
    raw_weights = []
    for i, symbol in enumerate(symbols):
        weight = (i - 2) * 0.8  # Same as before
        raw_weights.append(weight)

    # Normalize to sum to 1 (long-short portfolio)
    weight_sum = sum(abs(w) for w in raw_weights)
    normalized_weights = [w / weight_sum for w in raw_weights]

    for symbol, weight in zip(symbols, normalized_weights):
        normalized_weight_data.append({
            'date': date,
            'symbol': symbol,
            'weight': weight
        })

normalized_weights_df = pd.DataFrame(normalized_weight_data)
normalized_weights_df = normalized_weights_df.set_index(['date', 'symbol'])

print(f"\nSample normalized weights for first date:")
first_date_norm_weights = normalized_weights_df.loc[dates[0]]
print(first_date_norm_weights)
print(f"\nSum of absolute weights: {first_date_norm_weights['weight'].abs().sum():.2f}")
print(f"Sum of weights: {first_date_norm_weights['weight'].sum():.2f}")

# Run backtest with normalized weights
results_norm = backtester.run(normalized_weights_df, prices_df)

print(f"\nBacktest Results with Normalized Weights:")
print(f"Initial capital: ${backtester.initial_capital:,.0f}")
print(f"Final equity: ${results_norm['equity_curve'].iloc[-1]:,.0f}")
print(f"Total return: {(results_norm['equity_curve'].iloc[-1] / backtester.initial_capital - 1) * 100:.1f}%")

# Check cash
cash_series_norm = results_norm['cash']
min_cash_norm = cash_series_norm.min()
print(f"Minimum cash during backtest: ${min_cash_norm:,.0f}")
if min_cash_norm < 0:
    print("WARNING: Still using leverage even with normalized weights")
else:
    print("PASS: No leverage with normalized weights")

print("\n" + "=" * 80)
print("TEST 3: EVALUATOR SCORING BUG")
print("=" * 80)

from alphalab.backtest.metrics import calculate_all_metrics
from alphalab.ai.evaluator import StrategyEvaluator

# Create metrics for a catastrophic strategy (like the premium seeds showed)
catastrophic_metrics = {
    'total_return': -1.0,  # -100% return
    'sharpe_ratio': -5.0,
    'sortino_ratio': -5.0,
    'max_drawdown': -1.0,  # -100% drawdown
    'calmar_ratio': -1.0,
    'win_rate': 0.1,
    'total_trades': 200,
    'avg_daily_turnover': 1.5,
    'rolling_sharpe_std': 2.0,
    'return_quantile_05': -0.10
}

evaluator = StrategyEvaluator()
eval_result = evaluator.evaluate(catastrophic_metrics)

print("\nEvaluating catastrophic strategy (-100% return):")
print(f"Score: {eval_result['score']:.1f}/100")
print(f"Tier 1 (hard filters) pass: {eval_result['tier1_pass']}")
print(f"Tier 2 score: {eval_result['tier2_score']:.1f}")
print(f"Recommendation: {eval_result['recommendation']}")
print(f"\nScore breakdown:")
for key, value in eval_result['breakdown'].items():
    print(f"  {key}: {value:.2f}")

# Now test with realistic bad metrics
bad_metrics = {
    'total_return': 0.10,  # 10% return
    'sharpe_ratio': 0.3,   # Below minimum
    'sortino_ratio': 0.4,
    'max_drawdown': -0.35,  # -35% drawdown
    'calmar_ratio': 0.3,
    'win_rate': 0.45,
    'total_trades': 200,
    'avg_daily_turnover': 1.5,
    'rolling_sharpe_std': 0.5,
    'return_quantile_05': -0.03
}

eval_result_bad = evaluator.evaluate(bad_metrics)

print("\n\nEvaluating bad strategy (0.3 Sharpe):")
print(f"Score: {eval_result_bad['score']:.1f}/100")
print(f"Tier 1 (hard filters) pass: {eval_result_bad['tier1_pass']}")
print(f"Tier 2 score: {eval_result_bad['tier2_score']:.1f}")
print(f"Recommendation: {eval_result_bad['recommendation']}")

print("\n" + "=" * 80)
print("SUMMARY OF CRITICAL ISSUES FOUND")
print("=" * 80)

print("\n1. BACKTESTING ENGINE (engine.py):")
print("   - Lines 138-140: Weights are NOT normalized before calculating target notional")
print("   - This causes strategies with weights summing to 37.25 to use 37x leverage")
print("   - Only warns about leverage (lines 188-192) instead of preventing it")
print("   - Fix: Normalize weights to sum(abs(weights)) = 1.0 or user-specified max")

print("\n2. EVALUATOR SCORING (evaluator.py):")
print("   - Line 167: Drawdown scoring formula breaks with extreme drawdowns")
print("   - Returns 0 contribution instead of heavily penalizing > -50% drawdowns")
print("   - Catastrophic strategies can still get high scores if other metrics are NaN")

print("\n3. STRATEGY GENERATION:")
print("   - cs_rank_pct_centered + clip:-2,2 produces weights that sum to large values")
print("   - Across 149 stocks, this can create 37x leverage")
print("   - Strategies aren't validating weight normalization")

print("\n" + "=" * 80)
print("RECOMMENDED FIXES")
print("=" * 80)

print("\n1. In engine.py, add weight normalization:")
print("   # After line 122: target_weights = weights_aligned.loc[date, 'weight']")
print("   # Add normalization:")
print("   weight_sum = target_weights.abs().sum()")
print("   if weight_sum > 1.01:  # Allow 1% tolerance")
print("       target_weights = target_weights / weight_sum")

print("\n2. In evaluator.py, fix drawdown scoring:")
print("   # Replace line 167:")
print("   # scores['max_drawdown'] = max(0, (1 - dd / 0.5)) * 100")
print("   # With:")
print("   if dd > 0.5:  # Drawdown worse than -50%")
print("       scores['max_drawdown'] = 0  # Minimum score for extreme drawdowns")
print("   else:")
print("       scores['max_drawdown'] = (1 - dd / 0.5) * 100")

print("\n3. In strategy generation, always normalize after post-processing:")
print("   # After any cs_rank_pct_centered or clip operations")
print("   # Add: normalize:sum_abs")