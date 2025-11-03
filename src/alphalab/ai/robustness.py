"""Walk-forward validation and noise robustness testing.

ChatGPT Pro Recommendation #3: Ensure strategies are reproducible and robust.
Only accept strategies that degrade gracefully under noise and work out-of-sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class WalkForwardValidator:
    """Walk-forward validation with anchored expanding windows.

    ChatGPT Pro recommendation: Test strategies on multiple out-of-sample periods
    to ensure they weren't just lucky on the full sample.
    """

    def __init__(
        self,
        splits: list[tuple[tuple[str, str], tuple[str, str]]] | None = None,
    ):
        """
        Initialize walk-forward validator.

        Args:
            splits: List of (train, test) date range tuples.
                    Each is ((train_start, train_end), (test_start, test_end))
                    If None, uses default splits for 2019-2023 data.
        """
        if splits is None:
            # Default: Anchored expanding window
            self.splits = [
                (("2019-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")),  # Train 3yr, test 1yr
                (("2019-01-01", "2022-12-31"), ("2023-01-01", "2023-12-31")),  # Train 4yr, test 1yr
            ]
        else:
            self.splits = splits

    def validate(
        self,
        strategy_class,
        features: pd.DataFrame,
        data: pd.DataFrame,
        backtest_fn,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            strategy_class: Strategy class to test
            features: Full feature matrix
            data: Full OHLCV data
            backtest_fn: Function that takes (strategy, features, data) and returns metrics

        Returns:
            Dict with:
            {
                "in_sample_sharpe": 0.45,  # Full sample Sharpe
                "oos_sharpes": [0.38, 0.42],  # Out-of-sample Sharpes per split
                "avg_oos_sharpe": 0.40,
                "sharpe_degradation": 0.11,  # (in_sample - avg_oos) / in_sample
                "passed": True,  # If degradation < 50%
                "details": [...]  # Per-split details
            }
        """
        # 1. In-sample (full data) backtest
        strategy = strategy_class()
        in_sample_metrics = backtest_fn(strategy, features, data)
        in_sample_sharpe = in_sample_metrics.get("sharpe_ratio", 0.0)

        # 2. Walk-forward splits
        oos_results = []
        for (train_start, train_end), (test_start, test_end) in self.splits:
            # Filter to train period
            train_features = features[
                (features.index.get_level_values("date") >= train_start)
                & (features.index.get_level_values("date") <= train_end)
            ]
            train_data = data[
                (data.index.get_level_values("date") >= train_start)
                & (data.index.get_level_values("date") <= train_end)
            ]

            # Filter to test period
            test_features = features[
                (features.index.get_level_values("date") >= test_start)
                & (features.index.get_level_values("date") <= test_end)
            ]
            test_data = data[
                (data.index.get_level_values("date") >= test_start)
                & (data.index.get_level_values("date") <= test_end)
            ]

            # Strategy is "trained" on train period (though alphas are stateless)
            # Test on out-of-sample period
            strategy_oos = strategy_class()
            oos_metrics = backtest_fn(strategy_oos, test_features, test_data)
            oos_sharpe = oos_metrics.get("sharpe_ratio", 0.0)

            oos_results.append({
                "train_period": f"{train_start} to {train_end}",
                "test_period": f"{test_start} to {test_end}",
                "oos_sharpe": oos_sharpe,
                "oos_metrics": oos_metrics,
            })

        # 3. Calculate degradation
        oos_sharpes = [r["oos_sharpe"] for r in oos_results]
        avg_oos_sharpe = np.mean(oos_sharpes) if oos_sharpes else 0.0

        if in_sample_sharpe > 0:
            sharpe_degradation = (in_sample_sharpe - avg_oos_sharpe) / in_sample_sharpe
        else:
            sharpe_degradation = 1.0  # Total degradation if in-sample was 0 or negative

        # 4. Pass criteria: <50% degradation, positive in 2+ splits
        positive_splits = sum(1 for s in oos_sharpes if s > 0)
        passed = sharpe_degradation < 0.50 and positive_splits >= len(self.splits) / 2

        return {
            "in_sample_sharpe": in_sample_sharpe,
            "oos_sharpes": oos_sharpes,
            "avg_oos_sharpe": avg_oos_sharpe,
            "sharpe_degradation": sharpe_degradation,
            "positive_splits": positive_splits,
            "total_splits": len(self.splits),
            "passed": passed,
            "details": oos_results,
        }


class NoiseRobustnessTester:
    """Test strategy robustness to noise and perturbations.

    ChatGPT Pro recommendation: Only accept strategies that degrade gracefully
    when we inject realistic noise (date dropout, symbol dropout, feature jitter).
    """

    def __init__(
        self,
        date_dropout_pct: float = 0.05,  # Drop 5% of dates
        symbol_dropout_pct: float = 0.05,  # Drop 5% of symbols
        feature_jitter_pct: float = 0.02,  # Add 2% noise to features
        n_trials: int = 3,  # Run multiple trials for stability
    ):
        """
        Initialize noise robustness tester.

        Args:
            date_dropout_pct: Fraction of dates to randomly drop
            symbol_dropout_pct: Fraction of symbols to randomly drop
            feature_jitter_pct: Fraction of noise to add to features (as std multiplier)
            n_trials: Number of noise trials to run
        """
        self.date_dropout_pct = date_dropout_pct
        self.symbol_dropout_pct = symbol_dropout_pct
        self.feature_jitter_pct = feature_jitter_pct
        self.n_trials = n_trials

    def test(
        self,
        strategy_class,
        features: pd.DataFrame,
        data: pd.DataFrame,
        backtest_fn,
    ) -> dict[str, Any]:
        """
        Test strategy robustness to noise.

        Args:
            strategy_class: Strategy class to test
            features: Feature matrix
            data: OHLCV data
            backtest_fn: Function that takes (strategy, features, data) and returns metrics

        Returns:
            Dict with:
            {
                "baseline_sharpe": 0.45,
                "noise_sharpes": [0.42, 0.43, 0.41],
                "avg_noise_sharpe": 0.42,
                "sharpe_drop": 0.067,  # (baseline - avg_noise) / baseline
                "passed": True,  # If drop < 20%
                "details": [...]  # Per-trial details
            }
        """
        # 1. Baseline (no noise)
        strategy = strategy_class()
        baseline_metrics = backtest_fn(strategy, features, data)
        baseline_sharpe = baseline_metrics.get("sharpe_ratio", 0.0)

        # 2. Run noise trials
        noise_results = []
        for trial in range(self.n_trials):
            # Apply noise
            noisy_features, noisy_data = self._inject_noise(
                features, data, seed=trial
            )

            # Backtest with noise
            strategy_noisy = strategy_class()
            noisy_metrics = backtest_fn(strategy_noisy, noisy_features, noisy_data)
            noisy_sharpe = noisy_metrics.get("sharpe_ratio", 0.0)

            noise_results.append({
                "trial": trial + 1,
                "sharpe": noisy_sharpe,
                "metrics": noisy_metrics,
            })

        # 3. Calculate degradation
        noise_sharpes = [r["sharpe"] for r in noise_results]
        avg_noise_sharpe = np.mean(noise_sharpes) if noise_sharpes else 0.0

        if baseline_sharpe > 0:
            sharpe_drop = (baseline_sharpe - avg_noise_sharpe) / baseline_sharpe
        else:
            sharpe_drop = 1.0

        # 4. Pass criteria: <20% Sharpe drop
        passed = sharpe_drop < 0.20

        return {
            "baseline_sharpe": baseline_sharpe,
            "noise_sharpes": noise_sharpes,
            "avg_noise_sharpe": avg_noise_sharpe,
            "sharpe_drop": sharpe_drop,
            "passed": passed,
            "details": noise_results,
        }

    def _inject_noise(
        self,
        features: pd.DataFrame,
        data: pd.DataFrame,
        seed: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Inject noise into features and data."""
        rng = np.random.RandomState(seed)

        # 1. Date dropout: Remove random dates
        unique_dates = features.index.get_level_values("date").unique()
        n_drop_dates = int(len(unique_dates) * self.date_dropout_pct)
        if n_drop_dates > 0:
            drop_dates = rng.choice(unique_dates, size=n_drop_dates, replace=False)
            features = features[~features.index.get_level_values("date").isin(drop_dates)]
            data = data[~data.index.get_level_values("date").isin(drop_dates)]

        # 2. Symbol dropout: Remove random symbols
        unique_symbols = features.index.get_level_values("symbol").unique()
        n_drop_symbols = int(len(unique_symbols) * self.symbol_dropout_pct)
        if n_drop_symbols > 0:
            drop_symbols = rng.choice(unique_symbols, size=n_drop_symbols, replace=False)
            features = features[~features.index.get_level_values("symbol").isin(drop_symbols)]
            data = data[~data.index.get_level_values("symbol").isin(drop_symbols)]

        # 3. Feature jitter: Add Gaussian noise to numeric features
        features = features.copy()
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in features.columns:
                col_std = features[col].std()
                if col_std > 0:
                    noise = rng.normal(0, col_std * self.feature_jitter_pct, size=len(features))
                    features[col] = features[col] + noise

        return features, data


class RobustnessValidator:
    """Combined walk-forward and noise robustness validation.

    This is the full Recommendation #3 implementation.
    """

    def __init__(
        self,
        enable_walk_forward: bool = True,
        enable_noise_tests: bool = True,
        walk_forward_max_degradation: float = 0.50,  # Max 50% Sharpe drop
        noise_max_degradation: float = 0.20,  # Max 20% Sharpe drop
    ):
        """
        Initialize robustness validator.

        Args:
            enable_walk_forward: Run walk-forward validation
            enable_noise_tests: Run noise robustness tests
            walk_forward_max_degradation: Max allowed Sharpe degradation for walk-forward
            noise_max_degradation: Max allowed Sharpe degradation for noise tests
        """
        self.enable_walk_forward = enable_walk_forward
        self.enable_noise_tests = enable_noise_tests

        self.walk_forward_validator = WalkForwardValidator() if enable_walk_forward else None
        self.noise_tester = NoiseRobustnessTester() if enable_noise_tests else None

        self.walk_forward_max_degradation = walk_forward_max_degradation
        self.noise_max_degradation = noise_max_degradation

    def validate(
        self,
        strategy_class,
        features: pd.DataFrame,
        data: pd.DataFrame,
        backtest_fn,
    ) -> dict[str, Any]:
        """
        Run full robustness validation.

        Args:
            strategy_class: Strategy class to test
            features: Feature matrix
            data: OHLCV data
            backtest_fn: Function that takes (strategy, features, data) and returns metrics

        Returns:
            Dict with:
            {
                "walk_forward_passed": True/False,
                "noise_passed": True/False,
                "overall_passed": True/False,
                "walk_forward_results": {...},
                "noise_results": {...},
                "summary": "..."
            }
        """
        results = {
            "walk_forward_passed": None,
            "noise_passed": None,
            "overall_passed": False,
            "walk_forward_results": None,
            "noise_results": None,
        }

        # 1. Walk-forward validation
        if self.enable_walk_forward:
            wf_results = self.walk_forward_validator.validate(
                strategy_class, features, data, backtest_fn
            )
            results["walk_forward_results"] = wf_results
            results["walk_forward_passed"] = wf_results["passed"]

        # 2. Noise robustness tests
        if self.enable_noise_tests:
            noise_results = self.noise_tester.test(
                strategy_class, features, data, backtest_fn
            )
            results["noise_results"] = noise_results
            results["noise_passed"] = noise_results["passed"]

        # 3. Overall pass: Must pass both (if enabled)
        checks = []
        if self.enable_walk_forward:
            checks.append(results["walk_forward_passed"])
        if self.enable_noise_tests:
            checks.append(results["noise_passed"])

        results["overall_passed"] = all(checks) if checks else True

        # 4. Summary message
        summary_parts = []
        if self.enable_walk_forward:
            wf_deg = results["walk_forward_results"]["sharpe_degradation"]
            summary_parts.append(
                f"Walk-forward: {wf_deg:.1%} degradation "
                f"({'PASS' if results['walk_forward_passed'] else 'FAIL'})"
            )
        if self.enable_noise_tests:
            noise_drop = results["noise_results"]["sharpe_drop"]
            summary_parts.append(
                f"Noise: {noise_drop:.1%} drop "
                f"({'PASS' if results['noise_passed'] else 'FAIL'})"
            )

        results["summary"] = " | ".join(summary_parts) if summary_parts else "No robustness tests run"

        return results
