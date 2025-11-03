"""Prompt template loader for LLM-powered strategy generation.

This module provides utilities for loading and rendering prompt templates
from markdown files in the configs/prompts/ directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PromptLoader:
    """Load and render prompt templates with variable substitution.

    This class loads prompt templates from markdown files and renders them
    with provided template variables using Python's str.format().

    Example:
        >>> loader = PromptLoader()
        >>> prompt = loader.load("idea_generation",
        ...                      n_ideas=10,
        ...                      market_context="Bull market",
        ...                      columns_text=str(available_columns))
    """

    def __init__(self, prompts_dir: str | Path | None = None):
        """Initialize the prompt loader.

        Args:
            prompts_dir: Path to prompts directory. If None, uses default
                        location at configs/prompts/ relative to project root.
        """
        if prompts_dir is None:
            # Default to configs/prompts/ in project root
            # Assuming this file is in src/alphalab/ai/
            project_root = Path(__file__).parent.parent.parent.parent
            prompts_dir = project_root / "configs" / "prompts"

        self.prompts_dir = Path(prompts_dir)

        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")

    def load(self, prompt_name: str, **kwargs: Any) -> str:
        """Load and render a prompt template.

        Args:
            prompt_name: Name of the prompt file (without .md extension)
            **kwargs: Template variables to substitute in the prompt

        Returns:
            Rendered prompt string with variables substituted

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            KeyError: If required template variable is missing

        Example:
            >>> loader.load("factor_spec_generation",
            ...            idea_name="Momentum Strategy",
            ...            idea_hypothesis="Stocks with positive momentum outperform",
            ...            idea_features="ret_20d, ret_60d",
            ...            available_columns=json.dumps(columns),
            ...            validation_feedback="")
        """
        prompt_file = self.prompts_dir / f"{prompt_name}.md"

        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {prompt_file}\\n"
                f"Available prompts: {self.list_prompts()}"
            )

        # Load template content
        template = prompt_file.read_text()

        # Render with variables
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(
                f"Missing required template variable: {e}\\n"
                f"Required variables can be found in the template file: {prompt_file}"
            )

    def list_prompts(self) -> list[str]:
        """List all available prompt templates.

        Returns:
            List of prompt names (without .md extension)
        """
        return sorted([
            p.stem for p in self.prompts_dir.glob("*.md")
            if p.stem != "README"
        ])

    def get_template_variables(self, prompt_name: str) -> list[str]:
        """Extract template variable names from a prompt.

        Args:
            prompt_name: Name of the prompt file (without .md extension)

        Returns:
            List of variable names used in the template

        Example:
            >>> loader.get_template_variables("idea_generation")
            ['n_ideas', 'columns_text', 'market_context', 'data_desc', 'constraints_section']
        """
        import re

        prompt_file = self.prompts_dir / f"{prompt_name}.md"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_file}")

        template = prompt_file.read_text()

        # Extract variable names from {variable_name} patterns
        # Ignore escaped braces {{}} used for JSON examples
        variables = set()
        for match in re.finditer(r'\{([^{}]+)\}', template):
            var_name = match.group(1)
            # Skip if this is part of a double brace {{var}}
            if not (match.start() > 0 and template[match.start() - 1] == '{'):
                if not (match.end() < len(template) and template[match.end()] == '}'):
                    variables.add(var_name)

        return sorted(list(variables))


# Convenience functions for common prompt loading patterns

def load_idea_generation_prompt(
    loader: PromptLoader,
    n_ideas: int,
    market_context: str | None,
    available_data: dict | None,
    preferences: Any | None = None,
    example_selector=None,
) -> str:
    """Load idea generation prompt with standard variables.

    Args:
        loader: PromptLoader instance
        n_ideas: Number of strategy ideas to generate
        market_context: Description of current market conditions
        available_data: Available data description
        preferences: User preferences object (with to_prompt() method)
        example_selector: ExampleSelector for few-shot learning from database

    Returns:
        Rendered prompt string
    """
    context = market_context or "General market conditions across multiple regimes"
    data_desc = (
        json.dumps(available_data, indent=2)
        if available_data
        else "OHLCV price data, volume, and basic technical indicators"
    )

    # Get few-shot examples from database if available
    examples_text = ""
    if example_selector:
        examples = example_selector.get_few_shot_examples(n_good=3, n_bad=2)
        if examples["good"] or examples["bad"]:
            examples_text = "\n" + example_selector.format_examples_for_prompt(examples) + "\n"

    # Build user preferences section if provided
    user_prefs_section = ""
    if preferences is not None:
        user_prefs_section = f"""
USER PREFERENCES & REQUIREMENTS:
{preferences.to_prompt()}

IMPORTANT: Design strategies that align with the user's preferences above.
This is critical - the strategies MUST respect the constraints and preferences specified.
"""

    # Default constraints if no preferences
    default_constraints = """
CONSTRAINTS:
- Daily rebalancing maximum
- Long-short market-neutral preferred
- Maximum 50% turnover per day
- Must be implementable with available data
- Should work across multiple market regimes
"""

    constraints_section = (
        user_prefs_section if preferences is not None else default_constraints
    )

    # Get available columns
    base_columns = ["open","high","low","close","volume","ret_1d","ret_5d","ret_20d","ret_60d",
                    "vol_10d","vol_20d","vol_60d","volume_20d_avg","volume_zscore_20d",
                    "rsi_14d","rsi_28d","day_of_week","day_of_month","week_of_year",
                    "month","zscore_10d","zscore_20d","zscore_60d"]
    macro_columns = ["macro_vix","macro_gdp","macro_inflation","macro_fed_funds",
                     "macro_10y_treasury","macro_2y_treasury","macro_credit_spread",
                     "macro_consumer_sentiment","macro_retail_sales","macro_unemployment"]
    columns_text = str(base_columns + macro_columns)

    return loader.load(
        "idea_generation",
        n_ideas=n_ideas,
        columns_text=columns_text,
        market_context=context,
        data_desc=data_desc,
        constraints_section=constraints_section,
        examples_text=examples_text,
    )


def load_factor_spec_prompt(
    loader: PromptLoader,
    idea: dict[str, Any],
    data_schema: dict | None,
    validation_errors: list[str] | None = None,
    compiler = None,
) -> str:
    """Load factor spec generation prompt with standard variables.

    Args:
        loader: PromptLoader instance
        idea: Strategy idea dict with 'name', 'hypothesis', 'features' keys
        data_schema: Data schema information
        validation_errors: List of validation error messages from previous attempt
        compiler: FactorSpecCompiler instance (to get available columns)

    Returns:
        Rendered prompt string
    """
    available_columns = compiler.available_columns if compiler else []

    validation_feedback = ""
    if validation_errors:
        validation_feedback = f"""
VALIDATION FEEDBACK (address each item explicitly in the rewrite):
{chr(10).join(f"- {err}" for err in validation_errors)}
"""

    return loader.load(
        "factor_spec_generation",
        idea_name=idea.get('name', 'Unknown Strategy'),
        idea_hypothesis=idea.get('hypothesis', idea.get('description', 'No hypothesis provided')),
        idea_features=', '.join(idea.get('features', [])),
        available_columns=json.dumps(available_columns),
        validation_feedback=validation_feedback,
    )


def load_batch_spec_prompt(
    loader: PromptLoader,
    ideas: list[dict],
    columns_data: list,
    example_selector = None,
) -> str:
    """Load batch spec generation prompt with standard variables.

    Args:
        loader: PromptLoader instance
        ideas: List of strategy idea dicts
        columns_data: List of available columns (with or without descriptions)
        example_selector: Optional FewShotExampleSelector for examples

    Returns:
        Rendered prompt string
    """
    ideas_json = json.dumps(ideas, indent=2)

    # Get few-shot examples if available
    examples_text = ""
    if example_selector:
        examples = example_selector.get_few_shot_examples(n_good=2, n_bad=2)
        if examples["good"] or examples["bad"]:
            examples_text = "\\n" + example_selector.format_examples_for_prompt(examples) + "\\n"

    # Format columns with descriptions (if available)
    if columns_data and isinstance(columns_data[0], dict):
        # New format: show descriptions
        columns_text = "\\n".join([f'  - "{col["name"]}": {col["description"]}'
                                 for col in columns_data])
    else:
        # Old format: just list column names
        columns_text = json.dumps(columns_data, indent=2)

    return loader.load(
        "batch_spec_generation",
        n_ideas=len(ideas),
        examples_text=examples_text,
        columns_text=columns_text,
        ideas_json=ideas_json,
    )


def load_code_generation_prompt(
    loader: PromptLoader,
    idea: dict[str, Any],
    data_schema: dict | None,
) -> str:
    """Load code generation prompt (legacy, deprecated).

    Args:
        loader: PromptLoader instance
        idea: Strategy idea dict
        data_schema: Data schema information

    Returns:
        Rendered prompt string
    """
    schema_desc = (
        json.dumps(data_schema, indent=2)
        if data_schema
        else "Standard OHLCV + technical indicators"
    )

    return loader.load(
        "code_generation",
        idea_name=idea['name'],
        idea_hypothesis=idea['hypothesis'],
        idea_features=', '.join(idea['features']),
        expected_conditions=idea.get('expected_conditions', 'Various'),
        schema_desc=schema_desc,
    )


def load_refinement_prompt(
    loader: PromptLoader,
    iteration: int,
    best_strategy: dict,
    all_evaluations: dict,
    n_ideas: int,
    columns_data: list,
    lofo_feedback: str = "",
    worst_strategy: dict | None = None,
) -> str:
    """Load refinement prompt for iterative strategy improvement.

    Args:
        loader: PromptLoader instance
        iteration: Current iteration number
        best_strategy: Best strategy from previous iteration with metrics and evaluation
        all_evaluations: Dict of all strategy evaluations from previous iteration
        n_ideas: Number of new strategy ideas to generate
        columns_data: List of available columns
        lofo_feedback: Optional LOFO (Leave-One-Factor-Out) analysis feedback
        worst_strategy: Worst strategy to use as negative example (optional)

    Returns:
        Rendered prompt string
    """
    # Extract metrics from best strategy
    metrics = best_strategy.get("metrics", {})
    evaluation = best_strategy.get("evaluation", {})

    best_strategy_name = best_strategy.get("name", "Unknown")
    score = evaluation.get("score", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    total_return = metrics.get("total_return", 0.0)
    max_drawdown = metrics.get("max_drawdown", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    sortino = metrics.get("sortino_ratio", 0.0)

    # Format issues list
    issues = evaluation.get("issues", [])
    issues_list = "\n".join(f"- {issue}" for issue in issues) if issues else "None"

    # Extract worst strategy information if provided
    if worst_strategy:
        worst_metrics = worst_strategy.get("metrics", {})
        worst_evaluation = worst_strategy.get("evaluation", {})

        worst_strategy_name = worst_strategy.get("name", "Unknown")
        worst_score = worst_evaluation.get("score", 0.0)
        worst_sharpe = worst_metrics.get("sharpe_ratio", 0.0)
        worst_total_return = worst_metrics.get("total_return", 0.0)
        worst_max_drawdown = worst_metrics.get("max_drawdown", 0.0)

        worst_issues = worst_evaluation.get("issues", [])
        worst_issues_list = "\n".join(f"- {issue}" for issue in worst_issues) if worst_issues else "No specific issues identified"

        worst_strategy_spec = worst_strategy.get("spec", {})
        worst_strategy_spec_str = json.dumps(worst_strategy_spec, indent=2)
        worst_strategy_spec_str = worst_strategy_spec_str.replace('{', '{{').replace('}', '}}')
    else:
        # Default values if no worst strategy provided
        worst_strategy_name = "N/A"
        worst_score = 0.0
        worst_sharpe = 0.0
        worst_total_return = 0.0
        worst_max_drawdown = 0.0
        worst_issues_list = "No worst strategy available"
        worst_strategy_spec_str = "{{}}"

    # Identify common failure patterns
    common_issues = []
    for eval_result in all_evaluations.values():
        common_issues.extend(eval_result.get("issues", []))

    issue_counts = {}
    for issue in common_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1

    top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    common_issues_list = "\n".join(
        f"- {issue} (seen {count}x)" for issue, count in top_issues
    ) if top_issues else "None"

    # Build prescriptive guidance based on metrics
    prescriptive_guidance = _build_prescriptive_guidance(metrics)

    # Format columns
    if columns_data and isinstance(columns_data[0], dict):
        # New format: show descriptions
        columns_text = "\n".join([f'  - "{col["name"]}": {col["description"]}'
                                 for col in columns_data])
    else:
        # Old format: just list column names
        columns_text = json.dumps(columns_data, indent=2)

    # Extract and format strategy specification
    best_strategy_spec = best_strategy.get("spec", {})
    best_strategy_spec_str = json.dumps(best_strategy_spec, indent=2)
    # Escape curly braces to prevent format() errors
    best_strategy_spec_str = best_strategy_spec_str.replace('{', '{{').replace('}', '}}')

    # Escape curly braces in lofo_feedback to prevent format() errors
    if lofo_feedback:
        lofo_feedback = lofo_feedback.replace('{', '{{').replace('}', '}}')

    return loader.load(
        "refinement_prompt",
        iteration=iteration,
        best_strategy_name=best_strategy_name,
        score=score,
        sharpe=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        sortino=sortino,
        best_strategy_spec=best_strategy_spec_str,
        worst_strategy_name=worst_strategy_name,
        worst_score=worst_score,
        worst_sharpe=worst_sharpe,
        worst_total_return=worst_total_return,
        worst_max_drawdown=worst_max_drawdown,
        worst_issues_list=worst_issues_list,
        worst_strategy_spec=worst_strategy_spec_str,
        issues_list=issues_list,
        n_strategies=len(all_evaluations),
        common_issues_list=common_issues_list,
        lofo_feedback=lofo_feedback,
        prescriptive_guidance=prescriptive_guidance,
        n_ideas=n_ideas,
        columns_text=columns_text,
    )


def _build_prescriptive_guidance(metrics: dict) -> str:
    """Build prescriptive guidance based on strategy metrics.

    Args:
        metrics: Dictionary of backtest metrics

    Returns:
        Formatted prescriptive guidance string
    """
    sharpe = metrics.get('sharpe_ratio', 0.0)
    max_dd = abs(metrics.get('max_drawdown', 0.0))
    win_rate = metrics.get('win_rate', 0.0)
    sortino = metrics.get('sortino_ratio', 0.0)

    target_sharpe = 0.70
    target_dd = 0.30
    target_win_rate = 0.50

    guidance_parts = []

    # Sharpe ratio guidance
    if sharpe < target_sharpe:
        improvement_pct = ((target_sharpe - sharpe) / max(abs(sharpe), 0.01)) * 100
        guidance_parts.append(f"""
**CRITICAL: Sharpe Ratio Too Low ({sharpe:.2f} < {target_sharpe})**

You need {improvement_pct:.0f}% improvement in risk-adjusted returns.

**How to improve:**
1. **Add a supporting factor** to diversify signal sources (e.g., combine momentum with volume confirmation)
2. **Use cross-sectional robust normalization** with `"cs_robust_zscore"` transform
3. **Ensure volatility risk adjustment** with `"divide:vol_20d+0.01"` transform
4. **Consider multi-factor approach** - combine 2-3 complementary factors with weighted mean
""")

    # Drawdown guidance
    if max_dd > target_dd:
        guidance_parts.append(f"""
**Drawdown Too High ({max_dd:.1%} > {target_dd:.0%})**

**How to reduce drawdown:**
1. **Reduce position concentration** - ensure `"cs_rank_pct_centered"` and `"clip:-2,2"` in post-processing
2. **Add mean reversion overlay** if using pure momentum (momentum strategies have higher drawdowns)
3. **Strengthen outlier handling** with `"winsor:-3,3"` transform for each factor
4. **Consider blending momentum and reversal** at different time horizons
""")

    # Win rate guidance
    if win_rate < target_win_rate:
        guidance_parts.append(f"""
**Win Rate Too Low ({win_rate:.1%} < {target_win_rate:.0%})**

**How to improve win rate:**
1. **Add mean reversion component** - short-term reversals have higher win rates
2. **Use RSI for oversold/overbought signals** - `"rsi_14d"` can improve consistency
3. **Smooth signals over time** - consider using longer lookback periods
""")

    # Sortino ratio guidance
    if sortino < 0.90:
        guidance_parts.append(f"""
**Sortino Ratio Low ({sortino:.2f} < 0.90)**

**How to reduce downside volatility:**
1. **Add volume confirmation** to avoid false breakouts - use `"volume_zscore_20d"`
2. **Use volatility scaling** to reduce exposure in high-volatility periods
3. **Consider defensive factors** during market stress
""")

    if guidance_parts:
        return "\n".join(guidance_parts)
    else:
        return "**Performance is acceptable.** Continue refining the approach with variations on the current thesis."
