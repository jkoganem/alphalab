"""Strategy evaluation with multi-tier filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class StrategyEvaluator:
    """Evaluate strategies across multiple dimensions with tiered filtering."""

    def __init__(
        self,
        max_drawdown: float = 0.40,
        min_sharpe: float = 0.50,  # Entry-level professional standard (was 0.30)
        min_trades: int = 100,
        max_turnover: float = 5.0,
    ):
        """
        Initialize evaluator with hard filter thresholds.

        Args:
            max_drawdown: Maximum acceptable drawdown (e.g., 0.40 = 40%)
            min_sharpe: Minimum Sharpe ratio (0.50 = entry-level professional)
            min_trades: Minimum number of trades for statistical significance
            max_turnover: Maximum average daily turnover
        """
        self.hard_filters = {
            "max_drawdown": max_drawdown,
            "min_sharpe": min_sharpe,
            "min_trades": min_trades,
            "max_turnover": max_turnover,
        }

        self.score_weights = {
            "sharpe_ratio": 0.35,
            "sortino_ratio": 0.15,
            "max_drawdown": 0.20,
            "calmar_ratio": 0.10,
            "win_rate": 0.05,
            "consistency": 0.10,
            "tail_risk": 0.05,
        }

    def evaluate(self, strategy_results: dict[str, Any]) -> dict[str, Any]:
        """
        Full evaluation pipeline with multi-tier filtering.

        Args:
            strategy_results: Backtest results with metrics

        Returns:
            Evaluation dict with:
                - passed: bool
                - score: float (0-100)
                - tier1_pass: bool
                - tier2_score: float
                - tier3_pass: bool
                - issues: list[str]
                - recommendation: str
                - breakdown: dict
        """
        # TIER 1: Hard filters
        tier1_pass, issues = self._check_hard_filters(strategy_results)

        # TIER 2: Composite scoring (always calculate for feedback)
        tier2_score = self._calculate_composite_score(strategy_results)
        breakdown = self._get_score_breakdown(strategy_results)

        # If Tier 1 fails, return early but show Tier 2 score for feedback
        if not tier1_pass:
            return {
                "passed": False,
                "score": tier2_score,  # Show actual score instead of 0
                "tier1_pass": False,
                "tier2_score": tier2_score,
                "tier3_pass": False,
                "issues": issues,
                "recommendation": f"REJECT - Failed hard filters (Tier 2 score: {tier2_score:.1f}/100)",
                "breakdown": breakdown,
            }

        # TIER 3: Robustness checks
        tier3_pass, robustness_issues = self._check_robustness(strategy_results)

        # Final recommendation
        if tier3_pass and tier2_score >= 70:
            recommendation = "STRONG BUY - Excellent risk-adjusted returns"
        elif tier3_pass and tier2_score >= 50:
            recommendation = "CAUTIOUS BUY - Deploy with reduced capital"
        elif tier3_pass and tier2_score >= 30:
            recommendation = "MONITOR - Paper trade first"
        else:
            recommendation = f"REJECT - Failed robustness: {', '.join(robustness_issues)}"

        return {
            "passed": tier3_pass and tier2_score >= 30,  # Lowered from 40 to 30
            "score": tier2_score,
            "tier1_pass": True,
            "tier2_score": tier2_score,
            "tier3_pass": tier3_pass,
            "issues": issues + robustness_issues,
            "recommendation": recommendation,
            "breakdown": breakdown,  # Already calculated above
        }

    def _check_hard_filters(
        self, results: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Check if strategy meets minimum requirements."""
        issues = []

        # Max drawdown check
        if abs(results.get("max_drawdown", 0)) > self.hard_filters["max_drawdown"]:
            dd = results["max_drawdown"]
            issues.append(
                f"Drawdown {dd:.1%} exceeds limit {self.hard_filters['max_drawdown']:.1%}"
            )

        # Sharpe ratio check
        if results.get("sharpe_ratio", -999) < self.hard_filters["min_sharpe"]:
            sharpe = results.get("sharpe_ratio", 0)
            issues.append(
                f"Sharpe {sharpe:.2f} below minimum {self.hard_filters['min_sharpe']:.2f}"
            )

        # Minimum trades check
        total_trades = results.get("total_trades", 0)
        if total_trades < self.hard_filters["min_trades"]:
            issues.append(
                f"Only {total_trades} trades - need {self.hard_filters['min_trades']} minimum"
            )

        # Turnover check
        turnover = results.get("avg_daily_turnover", 0)
        if turnover > self.hard_filters["max_turnover"]:
            issues.append(
                f"Turnover {turnover:.1f}x exceeds limit {self.hard_filters['max_turnover']:.1f}x"
            )

        # Positive returns check
        if results.get("total_return", -1) <= 0:
            issues.append("Negative or zero total return")

        return (len(issues) == 0, issues)

    def _calculate_composite_score(self, results: dict[str, Any]) -> float:
        """
        Calculate weighted composite score (0-100).

        Each component is normalized to 0-100 range, then weighted.
        """
        scores = {}

        # Sharpe ratio (normalized, cap at 1.5)
        sharpe = results.get("sharpe_ratio", 0)
        scores["sharpe_ratio"] = min(max(sharpe, 0) / 1.5, 1.0) * 100

        # Sortino ratio (cap at 1.8)
        sortino = results.get("sortino_ratio", 0)
        scores["sortino_ratio"] = min(max(sortino, 0) / 1.8, 1.0) * 100

        # Max drawdown (inverted - lower is better)
        # CRITICAL FIX: Properly penalize extreme drawdowns
        dd = abs(results.get("max_drawdown", 1.0))
        if dd > 0.5:  # Drawdown worse than -50% gets 0 score
            scores["max_drawdown"] = 0.0
        else:
            scores["max_drawdown"] = (1 - dd / 0.5) * 100

        # Calmar ratio (return / max_drawdown, cap at 2.0)
        calmar = results.get("calmar_ratio", 0)
        scores["calmar_ratio"] = min(max(calmar, 0) / 2.0, 1.0) * 100

        # Win rate
        win_rate = results.get("win_rate", 0)
        scores["win_rate"] = win_rate * 100

        # Consistency (1 - std of rolling Sharpe)
        rolling_std = results.get("rolling_sharpe_std", 1.0)
        scores["consistency"] = max(0, (1 - rolling_std)) * 100

        # Tail risk (5th percentile daily return, inverted)
        tail_return = abs(results.get("return_quantile_05", -0.05))
        scores["tail_risk"] = max(0, min((tail_return - 0.01) / 0.05, 1.0)) * 100

        # Weighted average
        total_score = sum(
            scores[metric] * weight for metric, weight in self.score_weights.items()
        )

        return max(0, min(100, total_score))

    def _check_robustness(
        self, results: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Check strategy robustness across different tests."""
        issues = []

        # Check if any robustness data is available
        has_robustness_data = any([
            "walk_forward_sharpes" in results and len(results["walk_forward_sharpes"]) >= 2,
            "out_of_sample_sharpe" in results,
            "regime_sharpes" in results and len(results["regime_sharpes"]) >= 3,
            "parameter_sensitivity" in results,
        ])

        # If no robustness data available, pass by default (basic backtest only)
        if not has_robustness_data:
            return (True, ["Robustness checks skipped - data not available"])

        # 1. Walk-forward degradation
        wf_sharpes = results.get("walk_forward_sharpes", [])
        if len(wf_sharpes) >= 2:
            degradation = (wf_sharpes[-1] - wf_sharpes[0]) / max(
                abs(wf_sharpes[0]), 0.01
            )
            if degradation < -0.5:  # More than 50% decline
                issues.append(f"Walk-forward Sharpe declined {degradation:.1%}")

        # 2. Out-of-sample performance
        if "out_of_sample_sharpe" in results and results.get("sharpe_ratio", 0) > 0:
            oos_ratio = results["out_of_sample_sharpe"] / results["sharpe_ratio"]
            if oos_ratio < 0.5:  # OOS less than half of IS
                issues.append(
                    f"Out-of-sample Sharpe only {oos_ratio:.1%} of in-sample"
                )

        # 3. Regime consistency
        regime_sharpes = results.get("regime_sharpes", {})
        if len(regime_sharpes) >= 3:
            positive_regimes = sum(1 for s in regime_sharpes.values() if s > 0)
            if positive_regimes < 2:
                issues.append(
                    f"Only works in {positive_regimes} market regime(s)"
                )

        # 4. Parameter sensitivity
        if "parameter_sensitivity" in results:
            param_std = results["parameter_sensitivity"].get("sharpe_std", 0)
            if param_std > 0.2:
                issues.append(
                    f"High parameter sensitivity (Sharpe std={param_std:.2f})"
                )

        return (len(issues) == 0, issues)

    def _get_score_breakdown(self, results: dict[str, Any]) -> dict[str, float]:
        """Return detailed breakdown of score components."""
        sharpe = min(max(results.get("sharpe_ratio", 0), 0) / 1.5, 1.0)
        sortino = min(max(results.get("sortino_ratio", 0), 0) / 1.8, 1.0)
        dd = abs(results.get("max_drawdown", 1.0))
        calmar = min(max(results.get("calmar_ratio", 0), 0) / 2.0, 1.0)
        win_rate = results.get("win_rate", 0)
        rolling_std = results.get("rolling_sharpe_std", 1.0)

        return {
            "sharpe_contribution": sharpe * 100 * 0.35,
            "sortino_contribution": sortino * 100 * 0.15,
            "drawdown_contribution": max(0, (1 - dd / 0.5)) * 100 * 0.20,
            "calmar_contribution": calmar * 100 * 0.10,
            "winrate_contribution": win_rate * 100 * 0.05,
            "consistency_contribution": max(0, (1 - rolling_std)) * 100 * 0.10,
        }

    def rank_strategies(
        self, all_results: dict[str, dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Rank multiple strategies by composite score.

        Args:
            all_results: Dict mapping strategy name to results

        Returns:
            List of (strategy_name, evaluation) sorted by score (descending)
        """
        evaluations = {
            name: self.evaluate(results) for name, results in all_results.items()
        }

        # Sort by score, descending
        ranked = sorted(
            evaluations.items(), key=lambda x: x[1]["score"], reverse=True
        )

        return ranked
