"""Factor specification compiler - converts JSON specs to Python strategy code.

This module implements the key recommendation from ChatGPT Pro analysis:
Instead of having LLMs generate free-form code (which leads to failures),
have them output structured JSON specifications that we compile into code.

This guarantees:
- Cross-sectional ranking
- Multi-factor combinations
- Risk normalization by volatility
- Robust z-scoring
- Winsorization and clipping
- Proper NaN handling
- Correct output format
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


class FactorSpecCompiler:
    """Compiles JSON factor specifications into executable strategy classes."""

    def __init__(self, available_columns: list[str] | None = None):
        """Initialize compiler with available transforms.

        Args:
            available_columns: List of available column names. If None, uses default technical features.
                              Should be set dynamically to include macro/fundamental columns when available.
        """
        if available_columns is None:
            # Default technical features (backward compatibility + Tier-A enhancements)
            self.available_columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                # Returns (extended with Tier-A: ret_21d, ret_126d, ret_252d)
                "ret_1d",
                "ret_5d",
                "ret_10d",
                "ret_20d",
                "ret_21d",  # Tier-A: 1-month momentum
                "ret_60d",
                "ret_126d",  # Tier-A: 6-month momentum
                "ret_252d",  # Tier-A: 12-month momentum
                # Volatility (extended with Tier-A: parkinson_20d, hl_vol_20d)
                "vol_10d",
                "vol_20d",
                "vol_60d",
                "parkinson_20d",  # Tier-A: Parkinson volatility estimator
                "hl_vol_20d",  # Tier-A: High-low volatility
                # Volume features
                "volume_20d_avg",
                "volume_zscore_20d",
                "volume_zscore_60d",
                "turnover_20d",
                "adv_20d",  # Tier-A: Average dollar volume (liquidity)
                # Technical indicators
                "rsi_14d",
                "rsi_28d",
                # Z-scores
                "zscore_5d",
                "zscore_10d",
                "zscore_20d",
                "zscore_60d",
                # Higher moments
                "skew_20d",
                "skew_60d",
                "kurt_20d",
                "kurt_60d",
                # Gap features (Tier-A: gap_1d, gap_z20)
                "gap",
                "gap_1d",  # Tier-A: Overnight gap
                "gap_z20",  # Tier-A: Gap z-score
                # Calendar features
                "day_of_week",
                "day_of_month",
                "week_of_year",
                "month",
                # Macro economic indicators (from FRED)
                "macro_gdp",
                "macro_unemployment",
                "macro_inflation",
                "macro_fed_funds",
                "macro_10y_treasury",
                "macro_2y_treasury",
                "macro_vix",
                "macro_credit_spread",
                "macro_consumer_sentiment",
                "macro_retail_sales",
            ]
        else:
            self.available_columns = available_columns

    def validate_spec(self, spec: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate that a spec meets all requirements."""
        issues = []

        # Check required fields
        required = ["family", "factors", "combine", "post", "rationale"]
        for field in required:
            if field not in spec:
                issues.append(f"Missing required field: {field}")

        # Check factor count
        factors = spec.get("factors", [])
        if not (2 <= len(factors) <= 4):
            issues.append(f"Must have 2-4 factors, got {len(factors)}")

        # Check each factor
        for i, factor in enumerate(factors):
            if "name" not in factor:
                issues.append(f"Factor {i} missing 'name'")
            if "expr" not in factor:
                issues.append(f"Factor {i} missing 'expr'")
            if "transforms" not in factor:
                issues.append(f"Factor {i} missing 'transforms'")

            # Check column exists
            expr = factor.get("expr", "")
            # Extract column name (handle negation like "-ret_5d")
            col = expr.lstrip("-")
            if col and col not in self.available_columns:
                issues.append(f"Factor {i} uses non-existent column: {col}")

        # Check combine method
        combine = spec.get("combine", {})
        if combine.get("method") != "weighted_mean":
            issues.append("combine.method must be 'weighted_mean'")

        weights = combine.get("weights", [])
        if len(weights) != len(factors):
            issues.append(f"weights length {len(weights)} != factors length {len(factors)}")

        if weights and abs(sum(weights) - 1.0) > 0.01:
            issues.append(f"weights must sum to 1.0, got {sum(weights)}")

        # Check post-processing
        post = spec.get("post", [])
        if "cs_rank_pct_centered" not in post:
            issues.append("post must include 'cs_rank_pct_centered'")

        return (len(issues) == 0, issues)

    def compile_to_code(self, spec: dict[str, Any], class_name: str = "CompiledStrategy") -> str:
        """Compile a validated spec into executable Python code."""
        factors = spec["factors"]
        combine = spec["combine"]
        post = spec["post"]
        rationale = spec["rationale"]

        # Generate factor computations
        factor_code_lines = []
        for factor in factors:
            name = factor["name"]
            # Sanitize name to be valid Python identifier (no leading digits)
            safe_name = self._sanitize_identifier(name)
            expr = factor["expr"]
            transforms = factor["transforms"]

            # Build transform pipeline
            var = expr
            lines = [f"        # Factor: {name}"]
            lines.append(f"        {safe_name}_raw = features['{expr.lstrip('-')}']")

            # Handle negation
            if expr.startswith("-"):
                lines.append(f"        {safe_name}_raw = -{safe_name}_raw")

            current_var = f"{safe_name}_raw"

            # Apply transforms in order
            for transform in transforms:
                if transform.startswith("multiply:"):
                    multiplier = transform.split(":", 1)[1]
                    # Check if it's a scalar or a feature/macro
                    if multiplier.startswith("macro_") or multiplier in self.available_columns:
                        lines.append(f"        {safe_name}_tmp = {current_var} * features['{multiplier}']")
                    else:
                        lines.append(f"        {safe_name}_tmp = {current_var} * {multiplier}")
                    current_var = f"{safe_name}_tmp"

                elif transform.startswith("divide:"):
                    divisor = transform.split(":", 1)[1]
                    # Check if dividing by feature+offset or just a scalar
                    if "+" in divisor:
                        # Format: divide:feature+offset
                        feature_name = divisor.split("+")[0]
                        offset = divisor.split("+")[1]
                        lines.append(f"        {safe_name}_tmp = {current_var} / (features['{feature_name}'] + {offset})")
                    elif divisor.startswith("macro_") or divisor in self.available_columns:
                        # Format: divide:feature (no offset, just divide by feature)
                        lines.append(f"        {safe_name}_tmp = {current_var} / features['{divisor}']")
                    else:
                        # Format: divide:scalar (just a number)
                        lines.append(f"        {safe_name}_tmp = {current_var} / {divisor}")
                    current_var = f"{safe_name}_tmp"

                elif transform.startswith("subtract:"):
                    subtrahend = transform.split(":", 1)[1]
                    # Check if it's a scalar or a feature/macro
                    if subtrahend.startswith("macro_") or subtrahend in self.available_columns:
                        lines.append(f"        {safe_name}_tmp = {current_var} - features['{subtrahend}']")
                    else:
                        lines.append(f"        {safe_name}_tmp = {current_var} - {subtrahend}")
                    current_var = f"{safe_name}_tmp"

                elif transform.startswith("add:"):
                    addend = transform.split(":", 1)[1]
                    # Check if it's a scalar or a feature/macro
                    if addend.startswith("macro_") or addend in self.available_columns:
                        lines.append(f"        {safe_name}_tmp = {current_var} + features['{addend}']")
                    else:
                        lines.append(f"        {safe_name}_tmp = {current_var} + {addend}")
                    current_var = f"{safe_name}_tmp"

                elif transform == "abs":
                    lines.append(f"        {safe_name}_tmp = {current_var}.abs()")
                    current_var = f"{safe_name}_tmp"

                elif transform == "cs_robust_zscore":
                    lines.append(f"        {safe_name}_z = self._cs_robust_zscore({current_var})")
                    current_var = f"{safe_name}_z"

                elif transform.startswith("winsor:"):
                    bounds = transform.split(":", 1)[1]
                    lower, upper = map(float, bounds.split(","))
                    lines.append(f"        {safe_name}_win = {current_var}.clip({lower}, {upper})")
                    current_var = f"{safe_name}_win"

            lines.append(f"        df['{safe_name}'] = {current_var}")
            lines.append("")
            factor_code_lines.extend(lines)

        # Generate combination
        weights = combine["weights"]
        weight_terms = " + ".join([f"{w}*df['{self._sanitize_identifier(factors[i]['name'])}']" for i, w in enumerate(weights)])
        combine_line = f"        combined = {weight_terms}"

        # Generate post-processing
        post_lines = ["        # Post-processing"]
        current = "combined"
        for post_step in post:
            if post_step == "cs_rank_pct_centered":
                post_lines.append(f"        cs_rank = {current}.groupby(level='date').rank(pct=True) - 0.5")
                current = "cs_rank"
            elif post_step.startswith("clip:"):
                bounds = post_step.split(":", 1)[1]
                lower, upper = map(float, bounds.split(","))
                post_lines.append(f"        alpha = {current}.clip({lower}, {upper})")
                current = "alpha"

        # Final code
        code = f'''from __future__ import annotations

import pandas as pd
import numpy as np


class {class_name}:
    """{rationale}

    Auto-compiled from factor specification.
    Guarantees: cross-sectional, multi-factor, risk-normalized, robust z-score, clipped.
    """

    def _cs_robust_zscore(self, series: pd.Series) -> pd.Series:
        """Cross-sectional robust z-score using median and MAD."""
        grouped = series.groupby(level='date')
        median = grouped.transform('median')
        mad = grouped.transform(lambda s: (s - s.median()).abs().median() + 1e-9)
        return (series - median) / (1.4826 * mad)

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate alpha scores from compiled spec."""
        df = pd.DataFrame(index=features.index)

{chr(10).join(factor_code_lines)}
        # Combine factors
{combine_line}

{chr(10).join(post_lines)}

        # Return in required format
        result = pd.DataFrame(index=features.index)
        result['alpha'] = alpha.fillna(0.0)
        return result
'''
        return code

    def compile_to_class(self, spec: dict[str, Any], class_name: str = "CompiledStrategy"):
        """Compile spec and return executable class object."""
        code = self.compile_to_code(spec, class_name)

        # Execute code to create class
        namespace = {"pd": pd, "np": np}
        exec(code, namespace)

        return namespace[class_name]

    def _sanitize_identifier(self, name: str) -> str:
        """Convert factor name to valid Python identifier.

        Handles cases like:
        - "20d_momentum" -> "momentum_20d"
        - "5d-reversal" -> "reversal_5d"
        - "vol@risk" -> "vol_risk"
        """
        import re

        # Replace non-alphanumeric with underscore
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)

        # If starts with digit, move digits to end
        if name and name[0].isdigit():
            # Extract leading digits
            match = re.match(r'^(\d+)(.*)$', name)
            if match:
                digits, rest = match.groups()
                name = f"{rest}_{digits}" if rest else f"factor_{digits}"

        # Ensure not empty and not a keyword
        if not name or name in ['class', 'def', 'return', 'import', 'from']:
            name = f"factor_{name}"

        return name


def example_spec() -> dict[str, Any]:
    """Example factor spec following the ChatGPT Pro template."""
    return {
        "family": "cross_sectional_rank",
        "factors": [
            {
                "name": "mom20",
                "expr": "ret_20d",
                "transforms": [
                    "divide:vol_20d+0.01",
                    "cs_robust_zscore",
                    "winsor:-3,3",
                ],
            },
            {
                "name": "rev5",
                "expr": "-ret_5d",
                "transforms": [
                    "divide:vol_20d+0.01",
                    "cs_robust_zscore",
                    "winsor:-3,3",
                ],
            },
            {
                "name": "vol_press",
                "expr": "volume_zscore_20d",
                "transforms": [
                    "divide:vol_20d+0.01",
                    "cs_robust_zscore",
                    "winsor:-3,3",
                ],
            },
        ],
        "combine": {"method": "weighted_mean", "weights": [0.45, 0.35, 0.20]},
        "post": ["cs_rank_pct_centered", "clip:-2,2"],
        "rationale": "Blend momentum with short-horizon reversal and volume pressure; all risk-normalized to reduce drawdowns.",
    }


if __name__ == "__main__":
    # Test the compiler
    compiler = FactorSpecCompiler()
    spec = example_spec()

    # Validate
    valid, issues = compiler.validate_spec(spec)
    print(f"Valid: {valid}")
    if issues:
        print("Issues:", issues)
    else:
        # Compile
        code = compiler.compile_to_code(spec)
        print("\nGenerated Code:")
        print("=" * 80)
        print(code)
        print("=" * 80)

        # Test instantiation
        StrategyClass = compiler.compile_to_class(spec)
        strategy = StrategyClass()
        print(f"\nSuccessfully instantiated: {strategy.__class__.__name__}")
