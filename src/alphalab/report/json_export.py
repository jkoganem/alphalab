"""JSON export utilities for backtest results.

This module provides functions to export backtest results, metrics, and
visualizations to JSON format for further analysis or integration with
other tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _serialize_value(value: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable types.

    Args:
        value: The value to serialize

    Returns:
        JSON-serializable version of the value
    """
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.Series):
        return value.to_dict()
    elif isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    else:
        return value


def export_backtest_results(
    results: dict[str, Any],
    metrics: dict[str, Any],
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
    include_timeseries: bool = True,
) -> None:
    """Export backtest results and metrics to JSON file.

    Args:
        results: Dictionary containing backtest results with keys:
            - equity_curve: pd.Series of portfolio values over time
            - returns: pd.Series of portfolio returns
            - positions: pd.DataFrame of positions over time
            - trades: pd.DataFrame of trade records
            - costs: dict of cost breakdowns
        metrics: Dictionary containing performance metrics
        output_path: Path to save JSON file
        metadata: Optional metadata to include (strategy name, params, etc.)
        include_timeseries: Whether to include full timeseries data
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build export dictionary
    export_data: dict[str, Any] = {
        "metadata": metadata or {},
        "metrics": {},
        "summary": {},
    }

    # Serialize metrics
    for key, value in metrics.items():
        export_data["metrics"][key] = _serialize_value(value)

    # Create summary statistics
    if "equity_curve" in results:
        equity = results["equity_curve"]
        export_data["summary"]["initial_capital"] = _serialize_value(
            results.get("initial_capital", equity.iloc[0])
        )
        export_data["summary"]["final_equity"] = _serialize_value(equity.iloc[-1])
        export_data["summary"]["total_return"] = _serialize_value(
            (equity.iloc[-1] / equity.iloc[0]) - 1
        )
        export_data["summary"]["start_date"] = _serialize_value(equity.index[0])
        export_data["summary"]["end_date"] = _serialize_value(equity.index[-1])
        export_data["summary"]["duration_days"] = (
            equity.index[-1] - equity.index[0]
        ).days

    # Include timeseries data if requested
    if include_timeseries:
        export_data["timeseries"] = {}

        if "equity_curve" in results:
            equity = results["equity_curve"]
            export_data["timeseries"]["equity_curve"] = [
                {
                    "date": _serialize_value(date),
                    "equity": _serialize_value(value),
                }
                for date, value in equity.items()
            ]

        if "returns" in results:
            returns = results["returns"]
            export_data["timeseries"]["returns"] = [
                {
                    "date": _serialize_value(date),
                    "return": _serialize_value(value),
                }
                for date, value in returns.items()
            ]

        if "positions" in results:
            positions = results["positions"]
            export_data["timeseries"]["positions"] = _serialize_value(positions)

        if "trades" in results and results["trades"] is not None:
            trades = results["trades"]
            export_data["timeseries"]["trades"] = _serialize_value(trades)

    # Include cost breakdown
    if "costs" in results:
        export_data["costs"] = _serialize_value(results["costs"])

    # Write to file
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    print(f"Exported backtest results to {output_path}")


def export_comparison_results(
    strategy_results: dict[str, dict[str, Any]],
    output_path: str | Path,
) -> None:
    """Export comparison of multiple strategies to JSON.

    Args:
        strategy_results: Dict mapping strategy names to their results/metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_data = {
        "strategies": {},
        "comparison_metrics": {},
    }

    # Export each strategy's results
    for strategy_name, data in strategy_results.items():
        strategy_data = {
            "metrics": _serialize_value(data.get("metrics", {})),
            "summary": {},
        }

        # Add summary stats
        if "equity_curve" in data:
            equity = data["equity_curve"]
            strategy_data["summary"]["final_equity"] = _serialize_value(
                equity.iloc[-1]
            )
            strategy_data["summary"]["total_return"] = _serialize_value(
                (equity.iloc[-1] / equity.iloc[0]) - 1
            )

        comparison_data["strategies"][strategy_name] = strategy_data

    # Create comparison metrics
    metric_names = ["total_return", "sharpe_ratio", "sortino_ratio", "max_drawdown"]
    for metric in metric_names:
        comparison_data["comparison_metrics"][metric] = {}
        for strategy_name, data in strategy_results.items():
            metrics = data.get("metrics", {})
            if metric in metrics:
                comparison_data["comparison_metrics"][metric][
                    strategy_name
                ] = _serialize_value(metrics[metric])

    # Write to file
    with open(output_path, "w") as f:
        json.dump(comparison_data, f, indent=2, default=str)

    print(f"Exported strategy comparison to {output_path}")


def load_backtest_results(input_path: str | Path) -> dict[str, Any]:
    """Load backtest results from JSON file.

    Args:
        input_path: Path to JSON file

    Returns:
        Dictionary containing loaded results
    """
    input_path = Path(input_path)

    with open(input_path, "r") as f:
        data = json.load(f)

    # Convert timeseries back to pandas objects if present
    if "timeseries" in data:
        if "equity_curve" in data["timeseries"]:
            equity_data = data["timeseries"]["equity_curve"]
            dates = [pd.Timestamp(item["date"]) for item in equity_data]
            values = [item["equity"] for item in equity_data]
            data["timeseries"]["equity_curve"] = pd.Series(values, index=dates)

        if "returns" in data["timeseries"]:
            returns_data = data["timeseries"]["returns"]
            dates = [pd.Timestamp(item["date"]) for item in returns_data]
            values = [item["return"] for item in returns_data]
            data["timeseries"]["returns"] = pd.Series(values, index=dates)

    return data
