"""HTML report generation for backtest results.

This module creates comprehensive HTML reports with charts and tables
summarizing backtest performance.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 string.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure

    Returns
    -------
    str
        Base64 encoded PNG image
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64


def plot_equity_curve(equity: pd.Series, title: str = "Equity Curve") -> str:
    """Plot equity curve and return as base64.

    Parameters
    ----------
    equity : pd.Series
        Equity curve
    title : str, default "Equity Curve"
        Plot title

    Returns
    -------
    str
        Base64 encoded image
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity.index, equity.values, linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="y")

    return fig_to_base64(fig)


def plot_drawdown(equity: pd.Series, title: str = "Drawdown") -> str:
    """Plot drawdown chart.

    Parameters
    ----------
    equity : pd.Series
        Equity curve
    title : str
        Plot title

    Returns
    -------
    str
        Base64 encoded image
    """
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color="red")
    ax.plot(drawdown.index, drawdown.values, color="red", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))

    return fig_to_base64(fig)


def plot_monthly_returns(returns: pd.Series, title: str = "Monthly Returns Heatmap") -> str:
    """Plot monthly returns heatmap.

    Parameters
    ----------
    returns : pd.Series
        Daily returns
    title : str
        Plot title

    Returns
    -------
    str
        Base64 encoded image
    """
    # Resample to monthly
    monthly = (1 + returns).resample("M").prod() - 1

    # Create matrix (years x months)
    monthly_df = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )

    pivot = monthly_df.pivot_table(index="year", columns="month", values="return", fill_value=0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)

    # Labels
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.astype(int))
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Return", rotation=270, labelpad=20)

    # Annotations
    for i in range(len(pivot)):
        for j in range(12):
            if not np.isnan(pivot.iloc[i, j]):
                ax.text(
                    j,
                    i,
                    f"{pivot.iloc[i, j]:.1%}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    return fig_to_base64(fig)


def generate_html_report(
    results: dict[str, object],
    metrics: dict[str, float],
    output_path: str | Path,
    title: str = "Backtest Report",
) -> None:
    """Generate comprehensive HTML report.

    Parameters
    ----------
    results : dict[str, object]
        Backtest results with equity_curve, returns, trades, etc.
    metrics : dict[str, float]
        Performance metrics
    output_path : str | Path
        Output HTML file path
    title : str, default "Backtest Report"
        Report title
    """
    logger.info(f"Generating HTML report: {output_path}")

    # Extract data
    equity = results.get("equity_curve", pd.Series(dtype=float))
    returns = results.get("returns", pd.Series(dtype=float))
    # trades = results.get("trades", pd.DataFrame())  # Reserved for future use

    # Generate plots
    equity_plot = plot_equity_curve(equity) if not equity.empty else ""
    drawdown_plot = plot_drawdown(equity) if not equity.empty else ""
    monthly_plot = plot_monthly_returns(returns) if not returns.empty else ""

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        .metric-value.positive {{
            color: #4CAF50;
        }}
        .metric-value.negative {{
            color: #f44336;
        }}
        .chart {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #888;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated on: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Performance Summary</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {_get_color_class(metrics.get('total_return', 0))}">{metrics.get('total_return', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Annual Return</div>
            <div class="metric-value {_get_color_class(metrics.get('annual_return', 0))}">{metrics.get('annual_return', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value {_get_color_class(metrics.get('sharpe_ratio', 0))}">{metrics.get('sharpe_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value {_get_color_class(metrics.get('sortino_ratio', 0))}">{metrics.get('sortino_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Calmar Ratio</div>
            <div class="metric-value {_get_color_class(metrics.get('calmar_ratio', 0))}">{metrics.get('calmar_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Volatility (Annual)</div>
            <div class="metric-value">{metrics.get('volatility_annual', 0):.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Number of Trades</div>
            <div class="metric-value">{metrics.get('n_trades', 0):.0f}</div>
        </div>
    </div>

    <h2>Equity Curve</h2>
    <div class="chart">
        <img src="data:image/png;base64,{equity_plot}" alt="Equity Curve" style="width:100%;">
    </div>

    <h2>Drawdown Analysis</h2>
    <div class="chart">
        <img src="data:image/png;base64,{drawdown_plot}" alt="Drawdown" style="width:100%;">
    </div>

    <h2>Monthly Returns</h2>
    <div class="chart">
        <img src="data:image/png;base64,{monthly_plot}" alt="Monthly Returns" style="width:100%;">
    </div>

    <h2>Detailed Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        {_generate_metrics_table(metrics)}
    </table>

    <div class="footer">
        Generated with Alpha Backtest Lab | <a href="https://claude.com/claude-code">Claude Code</a>
    </div>
</body>
</html>
"""

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        f.write(html)

    logger.info(f"Report saved to: {output_path}")


def _get_color_class(value: float) -> str:
    """Get CSS class for metric value color."""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return ""


def _generate_metrics_table(metrics: dict[str, float]) -> str:
    """Generate HTML table rows for metrics."""
    rows = []
    for key, value in sorted(metrics.items()):
        # Format value
        if isinstance(value, (int, np.integer)):
            formatted = f"{value:,.0f}"
        elif "ratio" in key.lower() or "sharpe" in key.lower() or "sortino" in key.lower() or "calmar" in key.lower():
            formatted = f"{value:.2f}"
        elif "return" in key.lower() or "drawdown" in key.lower() or "volatility" in key.lower():
            formatted = f"{value:.2%}"
        else:
            formatted = f"{value:.4f}"

        rows.append(f"<tr><td>{key.replace('_', ' ').title()}</td><td>{formatted}</td></tr>")

    return "\n".join(rows)
