"""Tests for configuration loading and validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from alphalab.config import (
    AlphaConfig,
    BacktestConfig,
    DataConfig,
    ExecutionConfig,
    PortfolioConfig,
    ReportConfig,
    RiskConfig,
    SignalConfig,
    UniverseConfig,
    ValidationConfig,
)


def test_universe_config_valid():
    """Test UniverseConfig with valid data."""
    config = UniverseConfig(
        tickers=["AAPL", "MSFT"],
        exclude_illiquid=True,
        min_price=5.0,
    )
    assert config.tickers == ["AAPL", "MSFT"]
    assert config.exclude_illiquid is True
    assert config.min_price == 5.0


def test_data_config_defaults():
    """Test DataConfig default values."""
    config = DataConfig(path="data/test.parquet")
    assert config.interval == "1d"
    assert config.returns == "log"


def test_risk_config_validation():
    """Test RiskConfig validation of positive values."""
    # Valid config
    config = RiskConfig(volatility_target=0.15, max_gross_exposure=2.0)
    assert config.volatility_target == 0.15

    # Invalid negative volatility
    with pytest.raises(ValidationError):
        RiskConfig(volatility_target=-0.1)


def test_execution_config_defaults():
    """Test ExecutionConfig default values."""
    config = ExecutionConfig()
    assert config.delay == "next_open"
    assert config.slippage_model == "fixed_bps"
    assert config.fees_bps == 1.0


def test_backtest_config_from_yaml(tmp_path: Path):
    """Test BacktestConfig loading from YAML."""
    yaml_content = """
seed: 42
universe:
  tickers: ["AAPL", "MSFT"]
  exclude_illiquid: true
data:
  path: "data/test.parquet"
  interval: "1d"
  returns: "log"
alpha:
  name: "momentum_xs"
  params:
    lookback_days: 126
signal:
  method: "rank_long_short"
  params:
    long_pct: 0.2
    short_pct: 0.2
portfolio:
  method: "equal_weight"
  params: {}
risk:
  volatility_target: 0.15
  max_gross_exposure: 2.0
execution:
  delay: "next_open"
  fees_bps: 1.0
report:
  path: "out/report.html"
  include: ["summary", "equity_curve"]
"""
    # Write YAML to temp file
    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    # Load config
    config = BacktestConfig.from_yaml(yaml_file)

    # Verify
    assert config.seed == 42
    assert config.universe.tickers == ["AAPL", "MSFT"]
    assert config.alpha.name == "momentum_xs"
    assert config.alpha.params["lookback_days"] == 126
    assert config.signal.method == "rank_long_short"
    assert config.report.path == "out/report.html"


def test_backtest_config_to_yaml(tmp_path: Path):
    """Test BacktestConfig saving to YAML."""
    config = BacktestConfig(
        seed=42,
        universe=UniverseConfig(tickers=["AAPL", "MSFT"]),
        data=DataConfig(path="data/test.parquet"),
        alpha=AlphaConfig(name="momentum_xs", params={"lookback_days": 126}),
        signal=SignalConfig(method="rank_long_short", params={}),
        portfolio=PortfolioConfig(method="equal_weight", params={}),
        report=ReportConfig(path="out/report.html"),
    )

    yaml_file = tmp_path / "output_config.yaml"
    config.to_yaml(yaml_file)

    # Verify file exists and can be loaded
    assert yaml_file.exists()
    loaded_config = BacktestConfig.from_yaml(yaml_file)
    assert loaded_config.seed == config.seed
    assert loaded_config.alpha.name == config.alpha.name


def test_validation_config_defaults():
    """Test ValidationConfig default values."""
    config = ValidationConfig()
    assert config.cv == "purged_kfold"
    assert config.n_splits == 5
    assert config.embargo_days == 5
