#!/usr/bin/env python3
"""Validate, backtest, and add seed strategies to the database."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from alphalab.ai.spec_compiler import FactorSpecCompiler
from alphalab.ai.strategy_database import StrategyDatabase
from alphalab.ai.evaluator import StrategyEvaluator
from alphalab.backtest.engine import run_backtest
from alphalab.features.pipeline import StandardFeaturePipeline


def main():
    print("=" * 80)
    print("SEED STRATEGY VALIDATION AND BACKTESTING")
    print("=" * 80)

    # Load seed strategies
    seed_file = Path("data/seed_strategies.json")
    print(f"\n[LOAD] Reading seed strategies from {seed_file}...")
    with open(seed_file) as f:
        specs = json.load(f)
    print(f"  [OK] Loaded {len(specs)} seed strategies")

    # Load market data
    print("\n[DATA] Loading market data...")
    data_path = Path("data/stocks_7y_2025.parquet")
    data = pd.read_parquet(data_path)
    print(f"  [OK] Loaded {len(data)} rows for {data.index.get_level_values('symbol').nunique()} symbols")

    # Load macro data
    print("\n[MACRO] Loading macro economic data from cache...")
    from alphalab.data.market_data_db import MarketDataDatabase
    db_path = ".alphalab/market_data.db"
    market_db = MarketDataDatabase(db_path=db_path)
    series_ids = ['gdp', 'inflation', 'unemployment', 'fed_funds', '10y_treasury', '2y_treasury', 'vix', 'credit_spread', 'consumer_sentiment', 'retail_sales']
    macro_data = market_db.get_all_macro_series(series_ids)
    print(f"  [OK] Loaded {len(macro_data.columns)} macro indicators")

    # Generate features
    print("\n[FEATURES] Generating features...")
    pipeline = StandardFeaturePipeline(include_macro=True)
    features = pipeline.transform(data, macro_data=macro_data)
    print(f"  [OK] Generated {len(features.columns)} features")

    # Initialize components
    compiler = FactorSpecCompiler()
    db = StrategyDatabase()
    # Conservative profile: Sharpe >= 0.7, Max DD <= 30%
    evaluator = StrategyEvaluator(
        min_sharpe=0.7,
        max_drawdown=0.30,
        min_trades=100,
        max_turnover=5.0
    )

    # Process each seed strategy
    print("\n" + "=" * 80)
    print("PROCESSING SEED STRATEGIES")
    print("=" * 80)

    results = []

    for i, spec in enumerate(specs, 1):
        # Create strategy name
        strategy_name = f"Seed_{i:02d}_{spec['factors'][0]['name'][:30]}"

        print(f"\n[{i}/{len(specs)}] {strategy_name}")
        print("-" * 80)

        # Validate spec
        valid, issues = compiler.validate_spec(spec)
        if not valid:
            print(f"  [FAIL] Validation errors:")
            for issue in issues:
                print(f"    - {issue}")
            results.append({
                "name": strategy_name,
                "valid": False,
                "issues": issues
            })
            continue

        print("  [OK] Spec validated")

        # Compile strategy
        try:
            StrategyClass = compiler.compile_to_class(spec, class_name=f"Seed{i}")
            strategy = StrategyClass()
            print("  [OK] Strategy compiled")
        except Exception as e:
            print(f"  [FAIL] Compilation error: {e}")
            results.append({
                "name": strategy_name,
                "valid": False,
                "issues": [str(e)]
            })
            continue

        # Backtest
        try:
            print("  [BACKTEST] Running backtest...")
            bt_result = run_backtest(
                strategy=strategy,
                features=features,
                data=data
            )

            sharpe = bt_result["sharpe_ratio"]
            total_return = bt_result["total_return"] * 100
            max_dd = bt_result["max_drawdown"] * 100

            print(f"    Return: {total_return:+.1f}%")
            print(f"    Sharpe: {sharpe:.3f}")
            print(f"    Max DD: {max_dd:.1f}%")

        except Exception as e:
            print(f"  [FAIL] Backtest error: {e}")
            results.append({
                "name": strategy_name,
                "valid": True,
                "backtest_error": str(e)
            })
            continue

        # Evaluate
        try:
            eval_result = evaluator.evaluate(bt_result)
            score = eval_result["score"]
            passed = eval_result["passed"]

            print(f"    Score: {score:.1f}/100")
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {eval_result.get('recommendation', 'Evaluation complete')}")

        except Exception as e:
            print(f"  [FAIL] Evaluation error: {e}")
            score = 0.0
            passed = False

        # Add to database
        try:
            db.add_strategy(
                name=strategy_name,
                spec=spec,
                sharpe=sharpe,
                returns=total_return,
                evaluation_score=score,
                validation_passed=passed,
                validation_errors=[],
                model="chatgpt-pro-seed",
                metadata={
                    "seed_index": i,
                    "rationale": spec["rationale"]
                }
            )
            print(f"  [DB] Added to database")

        except Exception as e:
            print(f"  [WARN] Database insert failed: {e}")

        results.append({
            "name": strategy_name,
            "valid": True,
            "sharpe": sharpe,
            "return": total_return,
            "max_dd": max_dd,
            "score": score,
            "passed": passed
        })

    # Summary
    print("\n" + "=" * 80)
    print("SEED STRATEGY SUMMARY")
    print("=" * 80)

    valid_count = sum(1 for r in results if r.get("valid", False))
    passed_count = sum(1 for r in results if r.get("passed", False))

    print(f"\nTotal strategies: {len(specs)}")
    print(f"Valid: {valid_count}/{len(specs)}")
    print(f"Passed (score >= 50): {passed_count}/{valid_count if valid_count > 0 else len(specs)}")

    # Show top performers
    valid_results = [r for r in results if r.get("valid") and "sharpe" in r]
    if valid_results:
        print("\n" + "-" * 80)
        print("TOP 5 PERFORMERS BY SHARPE RATIO")
        print("-" * 80)

        sorted_results = sorted(valid_results, key=lambda x: x["sharpe"], reverse=True)
        for i, r in enumerate(sorted_results[:5], 1):
            status = "[OK]" if r["passed"] else "[FAIL]"
            print(f"{i}. [{status}] {r['name']}")
            print(f"   Sharpe: {r['sharpe']:.3f} | Return: {r['return']:+.1f}% | Score: {r['score']:.1f}/100")

    # Database stats
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)

    stats = db.get_statistics()
    print(f"\nTotal strategies in database: {stats['total_strategies']}")
    print(f"Passed strategies: {stats['passed_strategies']}")
    print(f"Pass rate: {stats['pass_rate']:.1%}")
    if stats['best_sharpe']:
        print(f"Best Sharpe: {stats['best_sharpe']:.3f}")
        print(f"Best Score: {stats['best_score']:.1f}/100")

    print("\n" + "=" * 80)
    print("[FINISHED] SEED STRATEGIES ADDED TO DATABASE")
    print("=" * 80)
    print("\nReady to run experiments with warm-started database!")


if __name__ == "__main__":
    main()
