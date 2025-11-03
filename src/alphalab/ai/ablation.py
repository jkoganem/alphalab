"""LOFO (Leave-One-Factor-Out) ablation analysis for multi-factor strategies.

ChatGPT Pro Recommendation #2: Provide factor-level feedback to LLM.
Instead of just "Sharpe is too low", tell it "Removing factor X improves Sharpe by Y".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class LOFOAnalyzer:
    """Leave-One-Factor-Out ablation analysis for understanding factor contributions."""

    def __init__(self):
        """Initialize LOFO analyzer."""
        pass

    def analyze_spec(
        self,
        spec: dict[str, Any],
        features: pd.DataFrame,
        data: pd.DataFrame,
        backtest_fn,
    ) -> dict[str, Any]:
        """
        Perform LOFO ablation analysis on a factor spec.

        For each factor in the spec:
        1. Remove that factor
        2. Re-compile and backtest
        3. Calculate delta metrics (Delta_Sharpe, Delta_Drawdown, etc.)

        Args:
            spec: Factor specification dict
            features: Feature matrix for backtesting
            data: OHLCV data for backtesting
            backtest_fn: Function that takes (spec, features, data) and returns metrics dict

        Returns:
            Dict with ablation results:
            {
                "full_metrics": {...},  # Metrics with all factors
                "ablations": {
                    "factor_name": {
                        "metrics": {...},
                        "deltas": {
                            "sharpe_delta": -0.15,  # Negative = factor helps
                            "drawdown_delta": 0.05,  # Positive = factor increases DD
                            ...
                        }
                    },
                    ...
                },
                "summary": {
                    "best_factor": "mom20",  # Removing this hurts most
                    "worst_factor": "rev5",  # Removing this helps most
                    "factor_rankings": [...]  # Sorted by contribution
                }
            }
        """
        factors = spec.get("factors", [])

        if len(factors) < 2:
            return {
                "full_metrics": {},
                "ablations": {},
                "summary": {"error": "Need at least 2 factors for LOFO analysis"},
            }

        # 1. Baseline: Run with all factors
        full_metrics = backtest_fn(spec, features, data)

        # 2. Run ablations: Remove each factor one at a time
        ablations = {}
        for i, factor in enumerate(factors):
            factor_name = factor["name"]

            # Create ablated spec (remove this factor)
            ablated_spec = self._remove_factor(spec, i)

            # Backtest ablated version
            try:
                ablated_metrics = backtest_fn(ablated_spec, features, data)

                # Calculate deltas
                deltas = self._calculate_deltas(full_metrics, ablated_metrics)

                ablations[factor_name] = {
                    "metrics": ablated_metrics,
                    "deltas": deltas,
                }
            except Exception as e:
                ablations[factor_name] = {
                    "error": str(e),
                    "deltas": {},
                }

        # 3. Summarize: Which factors help/hurt most?
        summary = self._summarize_ablations(ablations)

        return {
            "full_metrics": full_metrics,
            "ablations": ablations,
            "summary": summary,
        }

    def _remove_factor(self, spec: dict[str, Any], factor_idx: int) -> dict[str, Any]:
        """Create new spec with one factor removed."""
        import copy

        ablated = copy.deepcopy(spec)

        # Remove factor
        ablated["factors"] = [
            f for i, f in enumerate(ablated["factors"]) if i != factor_idx
        ]

        # Adjust weights (re-normalize to sum to 1.0)
        old_weights = ablated["combine"]["weights"]
        new_weights = [w for i, w in enumerate(old_weights) if i != factor_idx]

        if sum(new_weights) > 0:
            # Normalize
            total = sum(new_weights)
            new_weights = [w / total for w in new_weights]
        else:
            # Equal weights as fallback
            new_weights = [1.0 / len(new_weights)] * len(new_weights)

        ablated["combine"]["weights"] = new_weights

        return ablated

    def _calculate_deltas(
        self, full_metrics: dict, ablated_metrics: dict
    ) -> dict[str, float]:
        """Calculate delta metrics (ablated - full).

        Negative delta = factor helps (removing it hurts)
        Positive delta = factor hurts (removing it helps)
        """
        key_metrics = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "total_return",
        ]

        deltas = {}
        for metric in key_metrics:
            full_val = full_metrics.get(metric, 0.0)
            ablated_val = ablated_metrics.get(metric, 0.0)
            deltas[f"{metric}_delta"] = ablated_val - full_val

        return deltas

    def _summarize_ablations(self, ablations: dict) -> dict[str, Any]:
        """Summarize which factors help/hurt most."""
        if not ablations:
            return {}

        # Sort factors by Sharpe delta (most negative = most helpful)
        factor_deltas = []
        for factor_name, result in ablations.items():
            if "deltas" in result and "sharpe_ratio_delta" in result["deltas"]:
                sharpe_delta = result["deltas"]["sharpe_ratio_delta"]
                factor_deltas.append((factor_name, sharpe_delta))

        if not factor_deltas:
            return {}

        # Sort by delta (most negative first = most helpful)
        factor_deltas.sort(key=lambda x: x[1])

        best_factor, best_delta = factor_deltas[0]
        worst_factor, worst_delta = factor_deltas[-1]

        return {
            "best_factor": best_factor,  # Removing this hurts most
            "best_factor_delta": best_delta,
            "worst_factor": worst_factor,  # Removing this helps most
            "worst_factor_delta": worst_delta,
            "factor_rankings": [
                {"factor": name, "sharpe_delta": delta} for name, delta in factor_deltas
            ],
        }

    def format_ablation_feedback(self, ablation_results: dict) -> str:
        """
        Format ablation results into human-readable feedback for LLM.

        Returns a string that can be inserted into refinement prompts.
        """
        if "summary" not in ablation_results or "error" in ablation_results["summary"]:
            return "LOFO ablation analysis not available."

        summary = ablation_results["summary"]
        ablations = ablation_results["ablations"]

        lines = ["FACTOR ABLATION ANALYSIS (Leave-One-Factor-Out):", ""]

        # Show impact of each factor
        lines.append("Impact when each factor is REMOVED:")
        for ranking in summary.get("factor_rankings", []):
            factor = ranking["factor"]
            sharpe_delta = ranking["sharpe_delta"]

            if factor in ablations:
                deltas = ablations[factor].get("deltas", {})
                dd_delta = deltas.get("max_drawdown_delta", 0.0)
                wr_delta = deltas.get("win_rate_delta", 0.0)

                # Interpret
                if sharpe_delta < -0.05:
                    verdict = "HELPS (removing it hurts)"
                    action = "KEEP or INCREASE weight"
                elif sharpe_delta > 0.05:
                    verdict = "HURTS (removing it helps)"
                    action = "DROP or REDUCE weight"
                else:
                    verdict = "NEUTRAL"
                    action = "KEEP at current weight"

                lines.append(
                    f"  - {factor}: Delta_Sharpe={sharpe_delta:+.3f}, "
                    f"Delta_DD={dd_delta:+.3f}, Delta_WinRate={wr_delta:+.3f} => {verdict}"
                )
                lines.append(f"    Action: {action}")
        lines.append("")

        # Key recommendations
        best_factor = summary.get("best_factor")
        worst_factor = summary.get("worst_factor")
        worst_delta = summary.get("worst_factor_delta", 0.0)

        lines.append("KEY RECOMMENDATIONS:")
        if best_factor:
            lines.append(
                f"  - '{best_factor}' is your STRONGEST factor - keep high weight"
            )
        if worst_factor and worst_delta > 0.02:
            lines.append(
                f"  - '{worst_factor}' is DRAGGING DOWN performance - reduce weight or drop"
            )
        lines.append("")

        return "\n".join(lines)


class CanaryBacktester:
    """Fast canary backtest on subset of data for quick iteration.

    ChatGPT Pro recommendation: Use 2019-2020 (2 years) on 15 tickers subset
    for initial screening before full 5-year backtest.
    """

    def __init__(
        self,
        canary_date_range: tuple[str, str] = ("2019-01-01", "2020-12-31"),
        canary_tickers: list[str] | None = None,
        min_sharpe: float = 0.30,  # Lower bar for canary
    ):
        """
        Initialize canary backtester.

        Args:
            canary_date_range: (start_date, end_date) for canary test
            canary_tickers: Subset of tickers (default: 15 liquid tickers)
            min_sharpe: Minimum Sharpe to pass canary (lower than full test)
        """
        self.canary_date_range = canary_date_range
        self.min_sharpe = min_sharpe

        # Default to 15 liquid tickers if not specified
        if canary_tickers is None:
            # Top 15 by market cap / liquidity from typical S&P universe
            self.canary_tickers = [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "NVDA",
                "META",
                "TSLA",
                "BRK.B",
                "JPM",
                "JNJ",
                "V",
                "WMT",
                "PG",
                "MA",
                "HD",
            ]
        else:
            self.canary_tickers = canary_tickers

    def prepare_canary_data(
        self, data: pd.DataFrame, features: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filter data and features to canary subset.

        Args:
            data: Full OHLCV data
            features: Full feature matrix

        Returns:
            (canary_data, canary_features) - Filtered to canary date range and tickers
        """
        start_date, end_date = self.canary_date_range

        # Filter by date
        canary_data = data[
            (data.index.get_level_values("date") >= start_date)
            & (data.index.get_level_values("date") <= end_date)
        ]
        canary_features = features[
            (features.index.get_level_values("date") >= start_date)
            & (features.index.get_level_values("date") <= end_date)
        ]

        # Filter by tickers (only if tickers exist in data)
        available_tickers = [
            t for t in self.canary_tickers if t in data.index.get_level_values("symbol")
        ]

        if available_tickers:
            canary_data = canary_data[
                canary_data.index.get_level_values("symbol").isin(available_tickers)
            ]
            canary_features = canary_features[
                canary_features.index.get_level_values("symbol").isin(available_tickers)
            ]

        return canary_data, canary_features

    def passes_canary(self, metrics: dict) -> tuple[bool, str]:
        """
        Check if strategy passes canary test.

        Lower bar than full test - just check for positive signal.

        Args:
            metrics: Backtest metrics dict

        Returns:
            (passed, message)
        """
        sharpe = metrics.get("sharpe_ratio", 0.0)
        drawdown = metrics.get("max_drawdown", 1.0)
        win_rate = metrics.get("win_rate", 0.0)

        issues = []

        # Relaxed criteria for canary
        if sharpe < self.min_sharpe:
            issues.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe}")

        if drawdown > 0.50:  # More lenient than full test (0.40)
            issues.append(f"Drawdown {drawdown:.1%} > 50%")

        if win_rate < 0.45:  # More lenient than full test (0.50)
            issues.append(f"Win rate {win_rate:.1%} < 45%")

        if issues:
            return False, "Canary FAIL: " + ", ".join(issues)

        return True, "Canary PASS"
