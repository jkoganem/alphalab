#!/usr/bin/env python3
"""Stress test for LOFO module and refinement prompt integration.

This test creates a realistic scenario where LOFO feedback containing
JSON specs with curly braces is passed to the refinement prompt template.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphalab.ai.ablation import LOFOAnalyzer
from alphalab.ai.prompt_loader import PromptLoader, load_refinement_prompt


def create_mock_lofo_feedback():
    """Create mock LOFO feedback that contains JSON with curly braces."""
    return """FACTOR ABLATION ANALYSIS (Leave-One-Factor-Out):

Impact when each factor is REMOVED:
  - momentum: Delta_Sharpe=-0.150, Delta_DD=+0.050, Delta_WinRate=-0.020 => HELPS (removing it hurts)
    Action: KEEP or INCREASE weight
  - volume_confirmation: Delta_Sharpe=+0.030, Delta_DD=-0.010, Delta_WinRate=+0.005 => NEUTRAL
    Action: KEEP at current weight

KEY RECOMMENDATIONS:
- KEEP 'momentum' (Delta_Sharpe=-0.150) - critical factor
- CONSIDER dropping or reducing 'volume_confirmation' (minor negative impact)

SPEC CAUSING ISSUES (with curly braces):
{
  "family": "cross_sectional_rank",
  "factors": [
    {
      "name": "momentum",
      "expr": "ret_60d",
      "transforms": ["cs_robust_zscore", "winsor:-3,3"]
    },
    {
      "name": "volume_confirmation",
      "expr": "volume_zscore_20d",
      "transforms": ["cs_robust_zscore", "winsor:-3,3"]
    }
  ],
  "combine": {
    "method": "weighted_mean",
    "weights": [0.70, 0.30]
  },
  "post": ["cs_rank_pct_centered", "clip:-2,2"]
}
"""


def create_mock_best_strategy():
    """Create a mock best strategy for testing."""
    return {
        "name": "Test Momentum Strategy",
        "code": "# test code",
        "idea": {"hypothesis": "Test hypothesis"},
        "spec": {
            "family": "cross_sectional_rank",
            "factors": [
                {
                    "name": "momentum",
                    "expr": "ret_60d",
                    "transforms": ["cs_robust_zscore", "winsor:-3,3"]
                }
            ],
            "combine": {"method": "weighted_mean", "weights": [1.0]},
            "post": ["cs_rank_pct_centered", "clip:-2,2"]
        },
        "metrics": {
            "sharpe_ratio": 0.45,
            "total_return": 0.25,
            "max_drawdown": 0.18,
            "win_rate": 0.48,
            "sortino_ratio": 0.55
        },
        "evaluation": {
            "score": 35.2,
            "issues": ["Low Sharpe ratio (0.45 < 0.70)", "Win rate below 50%"]
        }
    }


def test_lofo_with_refinement_prompt():
    """Test LOFO feedback integration with refinement prompt."""
    print("="*80)
    print("LOFO STRESS TEST: Testing refinement prompt with LOFO feedback")
    print("="*80)

    # Create mock data
    lofo_feedback = create_mock_lofo_feedback()
    best_strategy = create_mock_best_strategy()
    all_evaluations = {
        "Test Momentum Strategy": best_strategy["evaluation"],
        "Failed Strategy 1": {"score": 15.0, "issues": ["Low Sharpe"]},
        "Failed Strategy 2": {"score": 20.0, "issues": ["High drawdown"]},
    }

    # Create mock columns data
    columns_data = [
        {"name": "ret_60d", "description": "60-day return"},
        {"name": "vol_20d", "description": "20-day volatility"},
        {"name": "volume_zscore_20d", "description": "Volume z-score"}
    ]

    print("\n[TEST 1] Testing with LOFO feedback containing JSON with curly braces...")
    print(f"LOFO feedback length: {len(lofo_feedback)} chars")
    print(f"Contains curly braces: {'{' in lofo_feedback}")

    try:
        loader = PromptLoader()
        prompt = load_refinement_prompt(
            loader=loader,
            iteration=2,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,
            columns_data=columns_data,
            lofo_feedback=lofo_feedback
        )

        print("[PASS] SUCCESS: Refinement prompt generated without errors")
        print(f"  Prompt length: {len(prompt)} chars")

        # Verify the curly braces are preserved in output
        if '{"family"' in prompt or '"family":' in prompt:
            print("[PASS] JSON structure preserved in prompt")
        else:
            print("⚠ WARNING: JSON structure may have been mangled")

    except KeyError as e:
        print(f"[FAIL] FAILED with KeyError: {e}")
        print("  This is the bug we're testing for!")
        return False
    except Exception as e:
        print(f"[FAIL] FAILED with unexpected error: {type(e).__name__}: {e}")
        return False

    print("\n[TEST 2] Testing without LOFO feedback (baseline)...")
    try:
        prompt_no_lofo = load_refinement_prompt(
            loader=loader,
            iteration=2,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,
            columns_data=columns_data,
            lofo_feedback=""
        )
        print("[PASS] SUCCESS: Prompt generated without LOFO feedback")
        print(f"  Prompt length: {len(prompt_no_lofo)} chars")
        print(f"  Difference with LOFO: {len(prompt) - len(prompt_no_lofo)} chars")

    except Exception as e:
        print(f"[FAIL] FAILED: {type(e).__name__}: {e}")
        return False

    print("\n[TEST 3] Testing with extreme case: nested JSON in LOFO...")
    extreme_lofo = lofo_feedback + """

NESTED SPEC TEST:
{
  "outer": {
    "inner": {
      "deep": {
        "value": "test"
      }
    }
  }
}
"""
    try:
        prompt_extreme = load_refinement_prompt(
            loader=loader,
            iteration=2,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,
            columns_data=columns_data,
            lofo_feedback=extreme_lofo
        )
        print("[PASS] SUCCESS: Handled extreme nested JSON case")
        print(f"  Prompt length: {len(prompt_extreme)} chars")

    except Exception as e:
        print(f"[FAIL] FAILED: {type(e).__name__}: {e}")
        return False

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    return True


if __name__ == "__main__":
    success = test_lofo_with_refinement_prompt()
    sys.exit(0 if success else 1)
