#!/usr/bin/env python3
"""Comprehensive test for template JSON escaping.

This test verifies that all prompt templates properly escape JSON examples
to prevent Python .format() errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from alphalab.ai.prompt_loader import (
    PromptLoader,
    load_refinement_prompt,
    load_batch_spec_prompt,
    load_idea_generation_prompt,
)


def test_batch_spec_template():
    """Test batch_spec_generation template with JSON examples."""
    print("\n" + "="*80)
    print("TEST 1: batch_spec_generation template")
    print("="*80)

    loader = PromptLoader()

    # Mock data
    ideas = [
        {"name": "Test Strategy", "hypothesis": "Test", "features": ["ret_60d"]}
    ]
    columns_data = [
        {"name": "ret_60d", "description": "60-day return"}
    ]

    try:
        prompt = load_batch_spec_prompt(
            loader=loader,
            ideas=ideas,
            columns_data=columns_data,
            example_selector=None
        )
        print("[PASS] SUCCESS: batch_spec_generation loaded without errors")
        print(f"  Prompt length: {len(prompt)} chars")

        # Verify JSON examples are preserved in output (but escaped in template)
        if '"family": "cross_sectional_rank"' in prompt:
            print("[PASS] JSON examples properly rendered in output")
        else:
            print("[FAIL] WARNING: JSON examples may not be rendering correctly")
            return False

    except KeyError as e:
        print(f"[FAIL] FAILED with KeyError: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] FAILED with {type(e).__name__}: {e}")
        return False

    return True


def test_refinement_template():
    """Test refinement_prompt template with JSON examples."""
    print("\n" + "="*80)
    print("TEST 2: refinement_prompt template")
    print("="*80)

    loader = PromptLoader()

    # Mock data
    best_strategy = {
        "name": "Test Strategy",
        "metrics": {
            "sharpe_ratio": 0.45,
            "total_return": 0.25,
            "max_drawdown": 0.18,
            "win_rate": 0.48,
            "sortino_ratio": 0.55
        },
        "evaluation": {
            "score": 35.2,
            "issues": ["Low Sharpe ratio"]
        }
    }

    all_evaluations = {
        "Test Strategy": best_strategy["evaluation"]
    }

    columns_data = [
        {"name": "ret_60d", "description": "60-day return"}
    ]

    try:
        prompt = load_refinement_prompt(
            loader=loader,
            iteration=2,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,
            columns_data=columns_data,
            lofo_feedback=""
        )
        print("[PASS] SUCCESS: refinement_prompt loaded without errors")
        print(f"  Prompt length: {len(prompt)} chars")

        # Verify JSON examples are preserved in output
        if '"family": "cross_sectional_rank"' in prompt:
            print("[PASS] JSON examples properly rendered in output")
        else:
            print("[FAIL] WARNING: JSON examples may not be rendering correctly")
            return False

    except KeyError as e:
        print(f"[FAIL] FAILED with KeyError: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] FAILED with {type(e).__name__}: {e}")
        return False

    return True


def test_refinement_with_lofo():
    """Test refinement_prompt with LOFO feedback containing JSON."""
    print("\n" + "="*80)
    print("TEST 3: refinement_prompt with LOFO feedback containing JSON")
    print("="*80)

    loader = PromptLoader()

    # Mock data with LOFO feedback containing JSON
    lofo_feedback = """
FACTOR ABLATION ANALYSIS:

SPEC WITH CURLY BRACES:
{
  "family": "cross_sectional_rank",
  "factors": [
    {
      "name": "momentum",
      "expr": "ret_60d"
    }
  ],
  "combine": {
    "method": "weighted_mean",
    "weights": [1.0]
  }
}
"""

    best_strategy = {
        "name": "Test Strategy",
        "metrics": {
            "sharpe_ratio": 0.45,
            "total_return": 0.25,
            "max_drawdown": 0.18,
            "win_rate": 0.48,
            "sortino_ratio": 0.55
        },
        "evaluation": {
            "score": 35.2,
            "issues": ["Low Sharpe ratio"]
        }
    }

    all_evaluations = {
        "Test Strategy": best_strategy["evaluation"]
    }

    columns_data = [
        {"name": "ret_60d", "description": "60-day return"}
    ]

    try:
        prompt = load_refinement_prompt(
            loader=loader,
            iteration=2,
            best_strategy=best_strategy,
            all_evaluations=all_evaluations,
            n_ideas=3,
            columns_data=columns_data,
            lofo_feedback=lofo_feedback
        )
        print("[PASS] SUCCESS: refinement_prompt with LOFO loaded without errors")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  LOFO feedback included: {len(lofo_feedback)} chars")

        # Verify both template JSON and LOFO JSON are preserved
        if '"family": "cross_sectional_rank"' in prompt:
            print("[PASS] JSON examples properly rendered in output")

        # The LOFO feedback should be double-escaped ({{ in output becomes { after format)
        if 'SPEC WITH CURLY BRACES:' in prompt:
            print("[PASS] LOFO feedback included in output")
        else:
            print("[FAIL] WARNING: LOFO feedback may have been stripped")
            return False

    except KeyError as e:
        print(f"[FAIL] FAILED with KeyError: {e}")
        print("  This is the exact error we were trying to fix!")
        return False
    except Exception as e:
        print(f"[FAIL] FAILED with {type(e).__name__}: {e}")
        return False

    return True


def test_idea_generation_template():
    """Test idea_generation template (uses double braces already)."""
    print("\n" + "="*80)
    print("TEST 4: idea_generation template")
    print("="*80)

    loader = PromptLoader()

    try:
        prompt = load_idea_generation_prompt(
            loader=loader,
            n_ideas=3,
            market_context="Test market",
            available_data={"test": "data"},
            preferences=None,
            example_selector=None
        )
        print("[PASS] SUCCESS: idea_generation loaded without errors")
        print(f"  Prompt length: {len(prompt)} chars")

        # Verify JSON examples are preserved in output
        if '"name":' in prompt:
            print("[PASS] JSON examples properly rendered in output")
        else:
            print("[FAIL] WARNING: JSON examples may not be rendering correctly")
            return False

    except KeyError as e:
        print(f"[FAIL] FAILED with KeyError: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] FAILED with {type(e).__name__}: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("="*80)
    print("COMPREHENSIVE TEMPLATE ESCAPING TEST")
    print("="*80)
    print("\nTesting all prompt templates for proper JSON escaping...")

    results = []

    # Run all tests
    results.append(("batch_spec_generation", test_batch_spec_template()))
    results.append(("refinement_prompt", test_refinement_template()))
    results.append(("refinement_prompt + LOFO", test_refinement_with_lofo()))
    results.append(("idea_generation", test_idea_generation_template()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = True
    for test_name, passed in results:
        status = "[PASS] PASS" if passed else "[FAIL] FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False

    print("="*80)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("="*80)
        return 0
    else:
        print("SOME TESTS FAILED!")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
