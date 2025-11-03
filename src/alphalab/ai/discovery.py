"""End-to-end strategy discovery pipeline."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from alphalab.ai.ablation import CanaryBacktester, LOFOAnalyzer
from alphalab.ai.evaluator import StrategyEvaluator
from alphalab.ai.generator import StrategyGenerator
from alphalab.ai.prompt_loader import PromptLoader, load_refinement_prompt
from alphalab.ai.spec_compiler import FactorSpecCompiler
from alphalab.ai.strategy_database import StrategyDatabase, ExampleSelector
from alphalab.backtest.engine import VectorizedBacktester
from alphalab.backtest.metrics import calculate_all_metrics
from alphalab.utils.config import load_dotenv

# Load .env file automatically
load_dotenv()


def _build_column_descriptions(columns: list[str]) -> list[dict[str, str]]:
    """Build column descriptions for LLM prompts.

    Args:
        columns: List of column names

    Returns:
        List of dicts with 'name' and 'description' keys
    """
    # Comprehensive column descriptions
    descriptions = {
        # OHLCV
        "open": "Opening price",
        "high": "Highest price",
        "low": "Lowest price",
        "close": "Closing price",
        "volume": "Trading volume",
        "adj_close": "Adjusted closing price",

        # Returns
        "ret_1d": "1-day price return",
        "ret_5d": "5-day price return",
        "ret_10d": "10-day price return",
        "ret_20d": "20-day price return (monthly)",
        "ret_60d": "60-day price return (quarterly)",
        "ret_126d": "126-day price return (6-month)",

        # Volatility
        "vol_5d": "5-day rolling volatility",
        "vol_10d": "10-day rolling volatility",
        "vol_20d": "20-day rolling volatility (monthly)",
        "vol_60d": "60-day rolling volatility (quarterly)",

        # Z-scores
        "zscore_5d": "5-day z-score of returns",
        "zscore_20d": "20-day z-score of returns",

        # Higher moments
        "skew_20d": "20-day return skewness",
        "skew_60d": "60-day return skewness",
        "kurt_20d": "20-day return kurtosis",
        "kurt_60d": "60-day return kurtosis",

        # RSI
        "rsi_14d": "14-day Relative Strength Index (0-100, overbought >70, oversold <30)",
        "rsi_28d": "28-day Relative Strength Index",

        # Volume
        "volume_zscore_20d": "20-day z-score of volume (unusual volume detection)",
        "volume_zscore_60d": "60-day z-score of volume",
        "turnover_20d": "Volume relative to 20-day average (liquidity measure)",

        # Gap
        "gap": "Overnight gap (open vs previous close)",

        # Beta
        "beta_60d": "60-day beta vs market benchmark (systematic risk)",
        "beta_126d": "126-day beta vs market benchmark",

        # Seasonality
        "day_of_week": "Day of week (0=Monday, 6=Sunday)",
        "day_of_month": "Day of month (1-31)",
        "month": "Month (1-12)",

        # Macro indicators (FRED) - common series
        "macro_GDP": "GDP - Gross Domestic Product (broad economic health, FRED)",
        "macro_UNRATE": "UNRATE - Unemployment Rate (labor market strength, FRED)",
        "macro_CPIAUCSL": "CPIAUCSL - Consumer Price Index (inflation measure, FRED)",
        "macro_FEDFUNDS": "FEDFUNDS - Federal Funds Rate (monetary policy stance, FRED)",
        "macro_DGS10": "DGS10 - 10-Year Treasury Yield (risk-free rate benchmark, FRED)",
        "macro_DGS2": "DGS2 - 2-Year Treasury Yield (FRED)",
        "macro_VIXCLS": "VIXCLS - VIX Volatility Index (market fear gauge, FRED)",
        "macro_BAA10Y": "BAA10Y - Corporate Bond Spread (credit risk premium, FRED)",

        # Fundamental metrics (FMP) - common metrics
        "fund_peRatio": "P/E Ratio - Price-to-Earnings (valuation multiple, FMP)",
        "fund_priceToBookRatio": "P/B Ratio - Price-to-Book (asset valuation, FMP)",
        "fund_returnOnEquity": "ROE - Return on Equity (profitability, FMP)",
        "fund_returnOnAssets": "ROA - Return on Assets (efficiency, FMP)",
        "fund_debtToEquity": "Debt-to-Equity Ratio (financial leverage, FMP)",
        "fund_currentRatio": "Current Ratio (short-term liquidity, FMP)",
        "fund_grossProfitMargin": "Gross Profit Margin (pricing power, FMP)",
        "fund_operatingProfitMargin": "Operating Profit Margin (operational efficiency, FMP)",
        "fund_netProfitMargin": "Net Profit Margin (bottom-line profitability, FMP)",
        "fund_revenuePerShare": "Revenue per Share (sales efficiency, FMP)",
        "fund_netIncomePerShare": "Net Income per Share (earnings, FMP)",
        "fund_operatingCashFlowPerShare": "Operating Cash Flow per Share (cash generation, FMP)",
    }

    result = []
    for col in columns:
        # Get description if available, otherwise use column name
        desc = descriptions.get(col, col)
        result.append({"name": col, "description": desc})

    return result


class StrategyDiscovery:
    """
    Complete pipeline for LLM-powered strategy discovery.

    Workflow:
    1. Generate strategy ideas using LLM
    2. Convert ideas to Python code
    3. Validate and import code safely
    4. Backtest each strategy
    5. Evaluate and rank results
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        initial_capital: float = 1_000_000,
        costs: dict | None = None,
        use_spec_compiler: bool = True,  # NEW: Enable spec compiler by default
    ):
        """
        Initialize strategy discovery pipeline.

        Args:
            llm_provider: 'openai' or 'anthropic'
            api_key: LLM API key
            model: Model name (or use defaults)
            initial_capital: Starting capital for backtests
            costs: Transaction costs dict (fees_bps, slippage_bps, borrow_bps)
            use_spec_compiler: If True, use JSON spec -> compiler pipeline (recommended)
        """
        # Strategy database for few-shot learning
        self.database = StrategyDatabase()
        self.example_selector = ExampleSelector(self.database)

        # Prompt loader for modular prompts
        self.prompt_loader = PromptLoader()

        self.generator = StrategyGenerator(
            provider=llm_provider,
            model=model,
        )
        self.evaluator = StrategyEvaluator()
        self.initial_capital = initial_capital
        self.costs = costs or {"fees_bps": 2.0, "slippage_bps": 5.0, "borrow_bps": 30.0}
        self.use_spec_compiler = use_spec_compiler

        # LOFO ablation and canary testing (Recommendation #2)
        self.lofo_analyzer = LOFOAnalyzer() if use_spec_compiler else None
        self.canary_tester = CanaryBacktester() if use_spec_compiler else None
        self.spec_compiler = FactorSpecCompiler() if use_spec_compiler else None

        self.generated_strategies = []
        self.backtest_results = {}

    def discover_with_refinement(
        self,
        initial_strategies: int,
        data: pd.DataFrame,
        features: pd.DataFrame,
        market_context: str | None = None,
        target_score: float = 70.0,
        max_iterations: int = 3,
        min_score: float = 40.0,
        preferences: Any | None = None,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.5,
    ) -> DiscoveryReport:
        """
        Automatically iterate and refine strategies until target score is reached.

        Args:
            initial_strategies: Number of strategies in first batch
            data: OHLCV price data
            features: Feature matrix
            market_context: Market description
            target_score: Stop when a strategy reaches this score
            max_iterations: Maximum refinement iterations
            min_score: Minimum score to keep a strategy
            preferences: StrategyPreferences object with user requirements
            early_stopping_patience: Stop if no improvement for N iterations (default: 10)
            early_stopping_min_delta: Minimum score improvement to reset patience (default: 0.5)

        Returns:
            DiscoveryReport with best strategies found across all iterations
        """
        print(f"\n{'='*80}")
        print(f"AUTOMATED STRATEGY REFINEMENT")
        print(f"{'='*80}")
        print(f"Target Score: {target_score}/100")
        print(f"Max Iterations: {max_iterations}")
        print(f"Early Stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        print(f"{'='*80}\n")

        all_strategies = []
        all_results = {}
        all_evaluations = {}
        best_overall = None

        # Early stopping tracking
        best_score_so_far = float('-inf')
        iterations_without_improvement = 0
        early_stop_reason = None

        for iteration in range(max_iterations):
            print(f"\n{'='*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*80}\n")

            if iteration == 0:
                # Initial generation
                report = self.discover_strategies(
                    n_strategies=initial_strategies,
                    data=data,
                    features=features,
                    market_context=market_context,
                    min_score=min_score,
                    preferences=preferences,
                )
            else:
                # Refinement based on previous results
                refinement_context = self._build_refinement_prompt(
                    best_overall, all_evaluations, iteration
                )

                n_refined = max(3, initial_strategies // 2)
                print(f"Generating {n_refined} refined strategies...")
                report = self.discover_strategies(
                    n_strategies=n_refined,
                    data=data,
                    features=features,
                    market_context=refinement_context,
                    min_score=min_score,
                    preferences=preferences,
                )

            # Accumulate results
            all_strategies.extend(report.strategies)
            all_results.update(report.backtest_results)
            all_evaluations.update(report.evaluations)

            # Find current best
            current_best = report.get_top_strategy()
            if current_best and (
                not best_overall
                or current_best["evaluation"]["score"] > best_overall["evaluation"]["score"]
            ):
                best_overall = current_best
                # Attach features and data for LOFO analysis
                best_overall["features"] = features
                best_overall["data"] = data
                print(f"\n[TARGET] NEW BEST: {best_overall['name']}")
                print(f"   Score: {best_overall['evaluation']['score']:.1f}/100")
                print(f"   {best_overall['evaluation']['recommendation']}")

            # Early stopping check (only applies AFTER target is reached)
            current_score = best_overall["evaluation"]["score"] if best_overall else float('-inf')
            improvement = current_score - best_score_so_far

            if improvement >= early_stopping_min_delta:
                # Significant improvement found - reset patience
                best_score_so_far = current_score
                iterations_without_improvement = 0
                print(f"\n[EARLY STOP] Improvement: +{improvement:.2f} (patience reset to 0)")
            else:
                # No significant improvement
                iterations_without_improvement += 1

            # Check if target reached
            if best_overall and best_overall["evaluation"]["score"] >= target_score:
                print(f"\n[SUCCESS] TARGET SCORE REACHED!")
                print(f"   {best_overall['name']}: {best_overall['evaluation']['score']:.1f}/100")

                # Only apply early stopping AFTER target is reached
                if iterations_without_improvement >= early_stopping_patience:
                    print(f"\n[EARLY STOP] Target reached + {iterations_without_improvement} iterations without improvement >= {early_stopping_min_delta}")
                    print(f"   Stopping to avoid wasted compute.")
                    early_stop_reason = "target_reached_no_improvement"
                    break
                else:
                    print(f"   Continue searching for improvements ({iterations_without_improvement}/{early_stopping_patience} iterations without improvement)")
            else:
                # Target not yet reached - keep going regardless of patience
                if iterations_without_improvement > 0:
                    print(f"\n[PATIENCE] No improvement for {iterations_without_improvement}/{early_stopping_patience} iterations (target not reached - continuing)")

        # Re-rank all accumulated strategies
        ranked = sorted(
            all_evaluations.items(), key=lambda x: x[1]["score"], reverse=True
        )

        print(f"\n{'='*80}")
        print(f"REFINEMENT COMPLETE")
        print(f"{'='*80}")
        print(f"Total strategies generated: {len(all_strategies)}")
        print(f"Best score achieved: {best_overall['evaluation']['score']:.1f}/100" if best_overall else "No strategies passed")
        print(f"Iterations used: {iteration + 1}/{max_iterations}")
        if early_stop_reason:
            if early_stop_reason == "target_reached_no_improvement":
                print(f"Stop reason: Target reached + no further improvements ({iterations_without_improvement} iterations without >={early_stopping_min_delta} improvement)")
        else:
            print(f"Stop reason: Max iterations reached")

        return DiscoveryReport(
            ideas=[s["idea"] for s in all_strategies],
            strategies=all_strategies,
            backtest_results=all_results,
            evaluations=all_evaluations,
            ranked=ranked,
        )

    def _run_lofo_analysis(
        self, strategy_spec: dict, features: pd.DataFrame, data: pd.DataFrame
    ) -> dict | None:
        """Run LOFO ablation analysis on a strategy spec."""
        if not self.use_spec_compiler or not self.lofo_analyzer:
            return None

        try:
            # Create backtest function for LOFO
            def backtest_fn(spec, feats, dat):
                # Compile spec to class
                from alphalab.ai.spec_compiler import FactorSpecCompiler
                compiler = FactorSpecCompiler()
                strategy_class = compiler.compile_to_class(spec)

                # Run backtest
                from alphalab.backtest.engine import VectorizedBacktester
                from alphalab.execution.costs import CompositeCostModel
                from alphalab.backtest.metrics import calculate_all_metrics

                # Instantiate strategy
                strategy = strategy_class()

                # Generate alpha scores
                alpha_scores = strategy.score(feats)

                # Convert to weights
                signals_binary = pd.DataFrame(index=alpha_scores.index)
                signals_binary["signal"] = alpha_scores["alpha"].apply(
                    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
                )

                weights = signals_binary.copy()
                weights.columns = ["weight"]

                # Normalize weights by date
                def normalize_weights(group):
                    total = group["weight"].abs().sum()
                    if total > 0:
                        return group["weight"] / total
                    return group["weight"]

                weights["weight"] = weights.groupby("date", group_keys=False).apply(
                    normalize_weights
                )

                # Run backtest
                backtester = VectorizedBacktester(initial_capital=self.initial_capital)
                results = backtester.run(weights, dat, costs_cfg=self.costs)

                # Calculate metrics
                metrics = calculate_all_metrics(
                    equity_curve=results["equity_curve"],
                    returns=results["returns"],
                    trades=results.get("trades"),
                    risk_free_rate=0.02,
                )
                return metrics

            # Run LOFO analysis
            ablation_results = self.lofo_analyzer.analyze_spec(
                strategy_spec, features, data, backtest_fn
            )
            return ablation_results
        except Exception as e:
            print(f"  [WARNING] LOFO analysis failed: {e}")
            return None

    def _build_refinement_prompt(
        self, best_strategy: dict, all_evaluations: dict, iteration: int
    ) -> str:
        """Build prompt for refinement iteration based on previous results."""
        if not best_strategy:
            return "Generate novel strategies with different approaches."

        # Try to get LOFO feedback if strategy has a spec
        lofo_feedback = ""
        if self.use_spec_compiler and "spec" in best_strategy:
            spec = best_strategy.get("spec")
            if spec is not None:
                print("  [LOFO] Running ablation analysis on best strategy...")
                ablation_results = self._run_lofo_analysis(
                    spec,
                    best_strategy.get("features"),
                    best_strategy.get("data")
                )
                if ablation_results and self.lofo_analyzer:
                    lofo_feedback = "\n" + self.lofo_analyzer.format_ablation_feedback(ablation_results) + "\n"
                    print(f"  [LOFO] Ablation feedback generated")
                else:
                    print(f"  [LOFO] Ablation analysis failed or returned None")
            else:
                print(f"  [LOFO] Spec is None, skipping ablation")
        else:
            # Debug: Why isn't LOFO running?
            if not self.use_spec_compiler:
                print(f"  [LOFO] Skipped: use_spec_compiler={self.use_spec_compiler}")
            elif "spec" not in best_strategy:
                print(f"  [LOFO] Skipped: 'spec' not in best_strategy. Keys: {list(best_strategy.keys())}")

        # Get available columns for the prompt
        columns_data = _build_column_descriptions(list(self.spec_compiler.available_columns)) if self.spec_compiler else []

        # Find the worst strategy from all_evaluations for negative example
        worst_strategy = None
        if all_evaluations:
            # Find strategy with lowest score
            worst_name = min(all_evaluations.keys(), key=lambda k: all_evaluations[k].get("score", 0.0))
            # Need to find the full strategy dict - search in the report
            # For now, we'll construct a minimal worst_strategy dict from all_evaluations
            worst_eval = all_evaluations[worst_name]
            worst_strategy = {
                "name": worst_name,
                "evaluation": worst_eval,
                "metrics": worst_eval.get("metrics", {}),
                "spec": worst_eval.get("spec", {}),
            }

        # Use modular prompt loader
        return load_refinement_prompt(
            loader=self.prompt_loader,
            iteration=iteration,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,  # Default to generating 3 refined strategies
            columns_data=columns_data,
            lofo_feedback=lofo_feedback,
            worst_strategy=worst_strategy,
        )

    def discover_strategies(
        self,
        n_strategies: int,
        data: pd.DataFrame,
        features: pd.DataFrame,
        market_context: str | None = None,
        min_score: float = 40.0,
        preferences: Any | None = None,
    ) -> DiscoveryReport:
        """
        Generate, backtest, and evaluate strategies end-to-end.

        Args:
            n_strategies: Number of strategies to generate
            data: OHLCV price data (MultiIndex: date, symbol)
            features: Feature matrix for strategy input
            market_context: Description of market conditions
            min_score: Minimum evaluation score to pass
            preferences: StrategyPreferences object with user requirements

        Returns:
            DiscoveryReport with all results
        """
        print(f"\n{'='*80}")
        print(f"STRATEGY DISCOVERY PIPELINE")
        print(f"{'='*80}\n")

        # Update generator's n_ideas to match requested n_strategies
        self.generator.n_ideas = n_strategies

        # Step 1: Generate ideas
        print(f"[1/5] Generating {n_strategies} strategy ideas...")
        data_schema = {
            "columns": _build_column_descriptions(list(features.columns)),
            "n_symbols": len(features.index.get_level_values("symbol").unique()),
            "date_range": f"{features.index.get_level_values('date').min()} to {features.index.get_level_values('date').max()}",
        }

        # Pass example_selector to enable few-shot learning from database
        ideas = self.generator.generate_ideas(
            market_context=market_context,
            data_description=data_schema,
            example_selector=self.example_selector,
        )
        print(f"  [OK] Generated {len(ideas)} ideas")

        # Step 2: Convert to code (use batch generation for efficiency)
        print(f"\n[2/5] Converting ideas to Python code...")
        strategies = []

        # Use batch generation if available
        if hasattr(self.generator, 'generate_strategies_batch') and self.use_spec_compiler:
            try:
                batch_results = self.generator.generate_strategies_batch(ideas, data_schema)

                for i, (idea, code, spec) in enumerate(batch_results, 1):
                    try:
                        strategy_obj = self._validate_and_import(code, idea["name"])
                        if strategy_obj:
                            strategies.append({
                                "name": idea["name"],
                                "code": code,
                                "obj": strategy_obj,
                                "idea": idea,
                                "spec": spec,
                            })
                            print(f"  [OK] [{i}/{len(ideas)}] {idea['name']}")
                        else:
                            print(f"  [FAIL] [{i}/{len(ideas)}] {idea['name']} - validation failed")
                    except Exception as e:
                        print(f"  [FAIL] [{i}/{len(ideas)}] {idea['name']} - {e}")

            except Exception as e:
                print(f"  [BATCH FAILED] Falling back to individual generation: {e}")
                # Fall back to individual generation
                for i, idea in enumerate(ideas, 1):
                    try:
                        # Validate idea has required 'name' field
                        idea_name = idea.get("name", f"Idea_{i}")
                        code = self.generator.generate_code(idea, data_schema)
                        strategy_obj = self._validate_and_import(code, idea_name)
                        if strategy_obj:
                            spec = self.generator.get_last_spec() if self.use_spec_compiler else None
                            strategies.append({
                                "name": idea_name,
                                "code": code,
                                "obj": strategy_obj,
                                "idea": idea,
                                "spec": spec,
                            })
                            print(f"  [OK] [{i}/{len(ideas)}] {idea_name}")
                        else:
                            print(f"  [FAIL] [{i}/{len(ideas)}] {idea_name} - validation failed")
                    except Exception as e:
                        # Safely get idea name with fallback
                        idea_name = idea.get("name", f"Idea_{i}") if isinstance(idea, dict) else f"Idea_{i}"
                        print(f"  [FAIL] [{i}/{len(ideas)}] {idea_name} - {e}")
        else:
            # Legacy: individual generation
            for i, idea in enumerate(ideas, 1):
                try:
                    # Validate idea has required 'name' field
                    idea_name = idea.get("name", f"Idea_{i}")
                    code = self.generator.generate_code(idea, data_schema)
                    strategy_obj = self._validate_and_import(code, idea_name)
                    if strategy_obj:
                        spec = self.generator.get_last_spec() if self.use_spec_compiler else None
                        strategies.append({
                            "name": idea_name,
                            "code": code,
                            "obj": strategy_obj,
                            "idea": idea,
                            "spec": spec,
                        })
                        print(f"  [OK] [{i}/{len(ideas)}] {idea_name}")
                    else:
                        print(f"  [FAIL] [{i}/{len(ideas)}] {idea_name} - validation failed")
                except Exception as e:
                    # Safely get idea name with fallback
                    idea_name = idea.get("name", f"Idea_{i}") if isinstance(idea, dict) else f"Idea_{i}"
                    print(f"  [FAIL] [{i}/{len(ideas)}] {idea_name} - {e}")

        print(f"\n  Validated: {len(strategies)}/{len(ideas)} strategies")
        self.generated_strategies = strategies

        # Step 3: Backtest all strategies
        print(f"\n[3/5] Backtesting {len(strategies)} strategies...")
        results = {}
        for i, strategy in enumerate(strategies, 1):
            try:
                result = self._backtest_strategy(strategy["obj"], data, features)
                results[strategy["name"]] = result
                sharpe = result["metrics"]["sharpe_ratio"]
                ret = result["metrics"]["total_return"]
                print(f"  [OK] [{i}/{len(strategies)}] {strategy['name']}: {ret:.1%} return, {sharpe:.2f} Sharpe")
            except Exception as e:
                print(f"  [FAIL] [{i}/{len(strategies)}] {strategy['name']}: {e}")

        self.backtest_results = results

        # Step 4: Evaluate and rank
        print(f"\n[4/5] Evaluating strategies...")
        evaluations = {}
        for name, result in results.items():
            eval_result = self.evaluator.evaluate(result["metrics"])
            evaluations[name] = eval_result
            status = "[OK]" if eval_result["passed"] else "[FAIL]"
            print(f"  {status} {name}: Score {eval_result['score']:.1f}/100 - {eval_result['recommendation']}")

            # Store in database for few-shot learning
            if self.database and self.use_spec_compiler:
                from datetime import datetime, timezone
                from alphalab.ai.strategy_database import StrategyRecord

                # Find corresponding strategy for spec
                strategy_with_spec = next((s for s in strategies if s["name"] == name and "spec" in s), None)

                if strategy_with_spec:
                    record = StrategyRecord(
                        name=name,
                        spec=strategy_with_spec["spec"],
                        sharpe=result["metrics"].get("sharpe_ratio"),
                        returns=result["metrics"].get("total_return"),
                        evaluation_score=eval_result["score"],
                        validation_passed=True,  # Made it to backtest
                        validation_errors=[],
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model=self.generator.model,  # Store model for comparison
                        metadata={},
                    )
                    self.database.store_strategy(record)

        # Step 5: Filter and rank
        print(f"\n[5/5] Ranking results...")
        passed = {
            name: eval_result
            for name, eval_result in evaluations.items()
            if eval_result["score"] >= min_score
        }

        ranked = sorted(passed.items(), key=lambda x: x[1]["score"], reverse=True)

        print(f"\n{'='*80}")
        print(f"DISCOVERY COMPLETE")
        print(f"{'='*80}")
        print(f"  Generated: {len(ideas)} ideas")
        print(f"  Implemented: {len(strategies)} strategies")
        print(f"  Backtested: {len(results)} strategies")
        print(f"  Passed filters: {len(passed)} strategies")

        if ranked:
            print(f"\n  [TOP] Top Strategy: {ranked[0][0]}")
            print(f"     Score: {ranked[0][1]['score']:.1f}/100")
            print(f"     {ranked[0][1]['recommendation']}")

        return DiscoveryReport(
            ideas=ideas,
            strategies=strategies,
            backtest_results=results,
            evaluations=evaluations,
            ranked=ranked,
        )

    def _backtest_strategy(
        self, strategy_class: type, data: pd.DataFrame, features: pd.DataFrame
    ) -> dict[str, Any]:
        """Run backtest for a single strategy."""
        # Instantiate strategy
        strategy = strategy_class()

        # Generate alpha scores
        alpha_scores = strategy.score(features)

        # For now, use simple threshold conversion
        # In production, you'd use proper signal converters
        signals_binary = pd.DataFrame(index=alpha_scores.index)
        signals_binary["signal"] = alpha_scores["alpha"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )

        # Convert to weights (equal weight for simplicity)
        weights = signals_binary.copy()
        weights.columns = ["weight"]

        # Normalize weights by date
        def normalize_weights(group):
            total = group["weight"].abs().sum()
            if total > 0:
                return group["weight"] / total
            return group["weight"]

        weights["weight"] = weights.groupby("date", group_keys=False).apply(
            normalize_weights
        )

        # Run backtest
        backtester = VectorizedBacktester(initial_capital=self.initial_capital)
        results = backtester.run(weights, data, costs_cfg=self.costs)

        # Calculate metrics
        metrics = calculate_all_metrics(
            equity_curve=results["equity_curve"],
            returns=results["returns"],
            trades=results.get("trades"),
            risk_free_rate=0.02,
        )

        return {"results": results, "metrics": metrics}

    def _validate_and_import(
        self, code: str, strategy_name: str
    ) -> type | None:
        """
        Validate and dynamically import generated strategy code.

        Args:
            code: Python code string
            strategy_name: Name for error messages

        Returns:
            Strategy class if valid, None otherwise
        """
        # Basic safety checks
        if any(
            forbidden in code
            for forbidden in ["import os", "subprocess", "eval", "exec", "__import__"]
        ):
            print(f"    Security check failed for {strategy_name}")
            return None

        try:
            # Write to temp file and import
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(code)
                temp_path = f.name

            spec = importlib.util.spec_from_file_location("temp_strategy", temp_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["temp_strategy"] = module
            spec.loader.exec_module(module)

            # Find the strategy class (should have a score method)
            strategy_class = None
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "score")
                    and name != "AlphaModel"
                ):
                    strategy_class = obj
                    break

            Path(temp_path).unlink()  # Clean up

            return strategy_class

        except Exception as e:
            print(f"    Import failed for {strategy_name}: {e}")
            return None


class DiscoveryReport:
    """Results from strategy discovery pipeline."""

    def __init__(
        self,
        ideas: list[dict],
        strategies: list[dict],
        backtest_results: dict,
        evaluations: dict,
        ranked: list[tuple[str, dict]],
    ):
        """Initialize report with all discovery results."""
        self.ideas = ideas
        self.strategies = strategies
        self.backtest_results = backtest_results
        self.evaluations = evaluations
        self.ranked = ranked

    def summary(self) -> str:
        """Generate text summary of discovery."""
        lines = [
            "\n" + "=" * 80,
            "STRATEGY DISCOVERY SUMMARY",
            "=" * 80,
            f"\nGenerated: {len(self.ideas)} ideas",
            f"Implemented: {len(self.strategies)} strategies",
            f"Backtested: {len(self.backtest_results)} strategies",
            f"Passed filters: {len(self.ranked)} strategies",
        ]

        if self.ranked:
            lines.append(f"\nTop 3 Strategies:")
            for i, (name, eval_result) in enumerate(self.ranked[:3], 1):
                metrics = self.backtest_results[name]["metrics"]
                lines.append(
                    f"\n{i}. {name}"
                    f"\n   Score: {eval_result['score']:.1f}/100"
                    f"\n   Sharpe: {metrics['sharpe_ratio']:.2f}"
                    f"\n   Return: {metrics['total_return']:.1%}"
                    f"\n   Max DD: {metrics['max_drawdown']:.1%}"
                    f"\n   {eval_result['recommendation']}"
                )

        return "\n".join(lines)

    def get_top_strategy(self) -> dict | None:
        """Get the best strategy."""
        if not self.ranked:
            return None

        top_name = self.ranked[0][0]
        strategy_dict = next(s for s in self.strategies if s["name"] == top_name)

        return {
            "name": top_name,
            "code": strategy_dict["code"],
            "idea": strategy_dict["idea"],
            "spec": strategy_dict.get("spec"),  # Include spec for LOFO analysis
            "evaluation": self.ranked[0][1],
            "metrics": self.backtest_results[top_name]["metrics"],
        }

    def save(self, filepath: str) -> None:
        """Save report to markdown file."""
        with open(filepath, "w") as f:
            f.write(self.summary())
            f.write("\n\n" + "=" * 80)
            f.write("\n\nDETAILED RESULTS\n\n")

            for name, eval_result in self.ranked:
                f.write(f"\n## {name}\n\n")
                f.write(f"**Score:** {eval_result['score']:.1f}/100\n\n")
                f.write(f"**Recommendation:** {eval_result['recommendation']}\n\n")

                metrics = self.backtest_results[name]["metrics"]
                f.write(f"**Performance:**\n")
                f.write(f"- Total Return: {metrics['total_return']:.1%}\n")
                f.write(f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
                f.write(f"- Max Drawdown: {metrics['max_drawdown']:.1%}\n")
                f.write(f"- Win Rate: {metrics.get('win_rate', 0):.1%}\n\n")

                # Find the strategy code
                strategy_dict = next(s for s in self.strategies if s["name"] == name)
                f.write(f"**Code:**\n```python\n{strategy_dict['code']}\n```\n\n")
