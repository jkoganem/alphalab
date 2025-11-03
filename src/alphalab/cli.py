"""Command-line interface for Alpha Backtest Lab using Typer.

This module provides a typed CLI with commands for data ingestion, alpha building,
backtesting, validation, and reporting. Flags always override config file settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from alphalab import __version__
from alphalab.config import BacktestConfig

app = typer.Typer(
    name="alphalab",
    help="Leakage-aware alpha generation and statistical backtesting toolkit",
    add_completion=False,
)

# Sub-commands
data_app = typer.Typer(help="Data ingestion and management commands")
alpha_app = typer.Typer(help="Alpha signal generation commands")
backtest_app = typer.Typer(help="Backtesting commands")
validate_app = typer.Typer(help="Validation and cross-validation commands")
report_app = typer.Typer(help="Report generation commands")
utils_app = typer.Typer(help="Utility commands")

app.add_typer(data_app, name="data")
app.add_typer(alpha_app, name="alpha")
app.add_typer(backtest_app, name="backtest")
app.add_typer(validate_app, name="validate")
app.add_typer(report_app, name="report")
app.add_typer(utils_app, name="utils")

console = Console()


@app.command()
def version() -> None:
    """Display version information."""
    console.print(f"[bold blue]Alpha Backtest Lab[/bold blue] version [green]{__version__}[/green]")


# ============================================================================
# Data commands
# ============================================================================


@data_app.command("ingest")
def data_ingest(
    source: Annotated[str, typer.Option(help="Data source (yahoo or local)")] = "yahoo",
    tickers: Annotated[
        Optional[list[str]], typer.Option(help="Ticker symbols (space-separated)")
    ] = None,
    start: Annotated[str, typer.Option(help="Start date (YYYY-MM-DD)")] = "2015-01-01",
    end: Annotated[str, typer.Option(help="End date (YYYY-MM-DD)")] = "2025-01-01",
    interval: Annotated[str, typer.Option(help="Data interval (e.g., 1d, 1h)")] = "1d",
    out: Annotated[str, typer.Option(help="Output path for data file")] = "data/ohlcv.parquet",
) -> None:
    """Ingest market data from specified source.

    Examples:
        alphalab data ingest --source yahoo --tickers AAPL MSFT SPY
        alphalab data ingest --start 2020-01-01 --end 2024-12-31
    """
    console.print(f"[bold]Ingesting data from {source}[/bold]")

    if tickers is None:
        console.print("[red]Error: --tickers is required[/red]")
        raise typer.Exit(1)

    try:
        # Import here to avoid circular dependencies
        from alphalab.data.yahoo import YahooDataSource

        ds = YahooDataSource()

        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(description=f"Fetching {len(tickers)} symbols...", total=None)
            df = ds.fetch(tickers, start_ts, end_ts, interval)

        # Save to parquet
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, compression="snappy")

        console.print(f"[green]Success:[/green] Saved {len(df)} rows to {out}")
        console.print(f"  Symbols: {len(df.index.get_level_values('symbol').unique())}")
        console.print(f"  Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@data_app.command("describe")
def data_describe(
    path: Annotated[str, typer.Argument(help="Path to data file")],
) -> None:
    """Display summary statistics for data file.

    Examples:
        alphalab data describe data/ohlcv.parquet
    """
    try:
        df = pd.read_parquet(path)
        console.print(f"[bold]Data Summary: {path}[/bold]\n")
        console.print(f"Shape: {df.shape}")
        console.print(f"Symbols: {len(df.index.get_level_values('symbol').unique())}")
        console.print(f"Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
        console.print(f"\nColumns: {', '.join(df.columns)}")
        console.print(f"\nMissing values:\n{df.isnull().sum()}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@data_app.command("validate")
def data_validate(
    path: Annotated[str, typer.Argument(help="Path to data file")],
) -> None:
    """Validate data file for quality issues.

    Examples:
        alphalab data validate data/ohlcv.parquet
    """
    console.print(f"[bold]Validating {path}[/bold]")
    console.print("[yellow]Warning:[/yellow] Validation not yet implemented")


# ============================================================================
# Alpha commands
# ============================================================================


@alpha_app.command("build")
def alpha_build(
    recipe: Annotated[Optional[str], typer.Option(help="Alpha recipe YAML file")] = None,
    input_data: Annotated[str, typer.Option("--in", help="Input data file")] = "data/ohlcv.parquet",
    output: Annotated[str, typer.Option("--out", help="Output alpha file")] = "artifacts/alpha/alpha.parquet",
) -> None:
    """Build alpha signals from data.

    Examples:
        alphalab alpha build --recipe configs/alpha/momo.yaml
    """
    console.print("[bold]Building alpha signals[/bold]")
    console.print("[yellow]Warning:[/yellow] Alpha building not yet implemented")


@alpha_app.command("list")
def alpha_list() -> None:
    """List available alpha models."""
    console.print("[bold]Available Alpha Models:[/bold]\n")
    models = [
        ("momentum_xs", "Cross-sectional momentum"),
        ("momentum_ts", "Time-series momentum"),
        ("mean_reversion", "Mean reversion strategy"),
        ("pairs", "Pairs trading / statistical arbitrage"),
        ("ml_alpha", "Machine learning alpha model"),
    ]
    for name, desc in models:
        console.print(f"  - [cyan]{name}[/cyan]: {desc}")


# ============================================================================
# Backtest commands
# ============================================================================


@backtest_app.command("run")
def backtest_run(
    config: Annotated[Optional[str], typer.Option(help="Config YAML file")] = None,
    start: Annotated[Optional[str], typer.Option(help="Start date override")] = None,
    end: Annotated[Optional[str], typer.Option(help="End date override")] = None,
    capital: Annotated[float, typer.Option(help="Initial capital")] = 1_000_000.0,
    report_path: Annotated[Optional[str], typer.Option("--report", help="Report output path")] = None,
) -> None:
    """Run a backtest with specified configuration.

    Examples:
        alphalab backtest run --config configs/backtest/momo.yaml
        alphalab backtest run --config configs/backtest/momo.yaml --capital 5000000
    """
    console.print("[bold]Running backtest[/bold]")

    if config is None:
        console.print("[red]Error: --config is required[/red]")
        raise typer.Exit(1)

    try:
        cfg = BacktestConfig.from_yaml(config)
        console.print(f"  Config: {config}")
        console.print(f"  Alpha: {cfg.alpha.name}")
        console.print(f"  Capital: ${capital:,.0f}")
        console.print("\n[yellow]Warning:[/yellow] Backtest execution not yet implemented")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


# ============================================================================
# Validation commands
# ============================================================================


@validate_app.command("purged-kfold")
def validate_purged_kfold(
    config: Annotated[str, typer.Option(help="Config YAML file")],
    folds: Annotated[int, typer.Option(help="Number of folds")] = 5,
    embargo: Annotated[int, typer.Option(help="Embargo days")] = 5,
    seed: Annotated[int, typer.Option(help="Random seed")] = 42,
) -> None:
    """Run purged k-fold cross-validation.

    Examples:
        alphalab validate purged-kfold --config configs/backtest/momo.yaml --folds 5
    """
    console.print("[bold]Running purged k-fold validation[/bold]")
    console.print("[yellow]Warning:[/yellow] Purged k-fold not yet implemented")


@validate_app.command("walkforward")
def validate_walkforward(
    config: Annotated[str, typer.Option(help="Config YAML file")],
    folds: Annotated[int, typer.Option(help="Number of folds")] = 6,
    embargo: Annotated[int, typer.Option(help="Embargo days")] = 5,
    report_path: Annotated[Optional[str], typer.Option("--report", help="Report path")] = None,
) -> None:
    """Run walk-forward validation.

    Examples:
        alphalab validate walkforward --config configs/validate/wf_momo.yaml
    """
    console.print("[bold]Running walk-forward validation[/bold]")
    console.print("[yellow]Warning:[/yellow] Walk-forward not yet implemented")


# ============================================================================
# Report commands
# ============================================================================


@report_app.command("generate")
def report_generate(
    results: Annotated[str, typer.Argument(help="Path to backtest results")],
    output: Annotated[str, typer.Option("--out", help="Output report path")] = "out/report.html",
    format: Annotated[str, typer.Option(help="Output format: html or json")] = "html",
) -> None:
    """Generate HTML or JSON report from backtest results.

    Examples:
        alphalab report generate artifacts/backtest_results.pkl --out out/report.html
        alphalab report generate artifacts/backtest_results.pkl --out out/report.json --format json
    """
    console.print(f"[bold]Generating {format.upper()} report[/bold]")
    console.print("[yellow]Warning:[/yellow] Report generation not yet implemented")


# ============================================================================
# Utility commands
# ============================================================================


@utils_app.command("verify-config")
def utils_verify_config(
    config: Annotated[str, typer.Argument(help="Path to config YAML")],
) -> None:
    """Verify configuration file is valid.

    Examples:
        alphalab utils verify-config configs/backtest/momo.yaml
    """
    try:
        cfg = BacktestConfig.from_yaml(config)
        console.print(f"[green]Success:[/green] Config is valid: {config}")
        console.print(f"  Alpha: {cfg.alpha.name}")
        console.print(f"  Signal: {cfg.signal.method}")
        console.print(f"  Portfolio: {cfg.portfolio.method}")
    except Exception as e:
        console.print(f"[red]Error:[/red] Config is invalid: {e}")
        raise typer.Exit(1) from e


@utils_app.command("cache-clear")
def utils_cache_clear() -> None:
    """Clear all cached data and artifacts."""
    console.print("[yellow]Clearing cache...[/yellow]")
    console.print("[yellow]Warning:[/yellow] Cache clearing not yet implemented")


if __name__ == "__main__":
    app()
