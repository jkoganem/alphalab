"""Configuration models using Pydantic for validation and YAML loading.

This module defines all configuration schemas for the backtesting system,
ensuring type safety and validation at runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class UniverseConfig(BaseModel):
    """Universe selection configuration.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to include
    exclude_illiquid : bool, default True
        Whether to exclude illiquid securities
    min_price : float, default 5.0
        Minimum price threshold
    min_market_cap : float | None, default None
        Minimum market cap threshold (if available)
    """

    tickers: list[str]
    exclude_illiquid: bool = True
    min_price: float = 5.0
    min_market_cap: float | None = None


class DataConfig(BaseModel):
    """Data loading configuration.

    Parameters
    ----------
    path : str
        Path to data file (Parquet or CSV)
    interval : str, default "1d"
        Data interval (e.g., "1d", "1h")
    returns : Literal["log", "simple"], default "log"
        Returns calculation method
    """

    path: str
    interval: str = "1d"
    returns: Literal["log", "simple"] = "log"


class AlphaConfig(BaseModel):
    """Alpha model configuration.

    Parameters
    ----------
    name : str
        Alpha model name (e.g., "momentum_xs", "mean_reversion")
    params : dict[str, object]
        Model-specific parameters
    """

    name: str
    params: dict[str, object] = Field(default_factory=dict)


class SignalConfig(BaseModel):
    """Signal generation configuration.

    Parameters
    ----------
    method : str
        Signal conversion method (e.g., "rank_long_short", "threshold")
    params : dict[str, object]
        Method-specific parameters
    """

    method: str
    params: dict[str, object] = Field(default_factory=dict)


class PortfolioConfig(BaseModel):
    """Portfolio construction configuration.

    Parameters
    ----------
    method : str
        Portfolio optimization method (e.g., "equal_weight", "inverse_vol")
    params : dict[str, object]
        Method-specific parameters
    """

    method: str
    params: dict[str, object] = Field(default_factory=dict)


class RiskConfig(BaseModel):
    """Risk management configuration.

    Parameters
    ----------
    volatility_target : float | None, default None
        Target annualized volatility
    max_gross_exposure : float, default 2.0
        Maximum gross exposure (long + short)
    max_net_exposure : float, default 1.0
        Maximum net exposure (long - short)
    max_turnover_pct : float | None, default None
        Maximum turnover percentage per period
    max_position_size : float, default 0.1
        Maximum position size as fraction of portfolio
    """

    volatility_target: float | None = None
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 1.0
    max_turnover_pct: float | None = None
    max_position_size: float = 0.1

    @field_validator("volatility_target", "max_gross_exposure", "max_net_exposure")
    @classmethod
    def positive_values(cls, v: float | None) -> float | None:
        """Validate that values are positive."""
        if v is not None and v <= 0:
            msg = "Value must be positive"
            raise ValueError(msg)
        return v


class ExecutionConfig(BaseModel):
    """Execution simulation configuration.

    Parameters
    ----------
    delay : Literal["next_open", "next_close", "same_close"], default "next_open"
        Execution timing relative to signal generation
    lot_size : int, default 1
        Minimum trading lot size
    slippage_model : Literal["fixed_bps", "sqrt_impact"], default "fixed_bps"
        Slippage model type
    slippage_bps : float, default 5.0
        Slippage in basis points (for fixed model)
    impact_coeff : float, default 0.1
        Impact coefficient (for sqrt model)
    fees_bps : float, default 1.0
        Transaction fees in basis points
    borrow_bps : float, default 30.0
        Annualized borrow cost for shorts in basis points
    """

    delay: Literal["next_open", "next_close", "same_close"] = "next_open"
    lot_size: int = 1
    slippage_model: Literal["fixed_bps", "sqrt_impact"] = "fixed_bps"
    slippage_bps: float = 5.0
    impact_coeff: float = 0.1
    fees_bps: float = 1.0
    borrow_bps: float = 30.0


class ValidationConfig(BaseModel):
    """Validation configuration for cross-validation and walk-forward.

    Parameters
    ----------
    cv : Literal["purged_kfold", "timeseries"], default "purged_kfold"
        Cross-validation method
    n_splits : int, default 5
        Number of CV splits
    embargo_days : int, default 5
        Embargo period in days (for purged k-fold)
    wf_folds : int | None, default None
        Number of walk-forward folds
    wf_train_days : int | None, default None
        Training window size for walk-forward
    wf_test_days : int | None, default None
        Test window size for walk-forward
    """

    cv: Literal["purged_kfold", "timeseries"] = "purged_kfold"
    n_splits: int = 5
    embargo_days: int = 5
    wf_folds: int | None = None
    wf_train_days: int | None = None
    wf_test_days: int | None = None


class ReportConfig(BaseModel):
    """Report generation configuration.

    Parameters
    ----------
    path : str
        Output path for report
    include : list[str]
        Sections to include in report
    format : Literal["html", "pdf"], default "html"
        Report output format
    """

    path: str
    include: list[str] = Field(
        default_factory=lambda: [
            "summary",
            "equity_curve",
            "drawdowns",
            "factor_returns",
            "exposures",
            "significance",
        ]
    )
    format: Literal["html", "pdf"] = "html"


class BacktestConfig(BaseModel):
    """Complete backtesting configuration.

    This is the top-level configuration model that combines all sub-configs.

    Parameters
    ----------
    seed : int, default 42
        Random seed for reproducibility
    universe : UniverseConfig
        Universe selection configuration
    data : DataConfig
        Data loading configuration
    alpha : AlphaConfig
        Alpha model configuration
    signal : SignalConfig
        Signal generation configuration
    portfolio : PortfolioConfig
        Portfolio construction configuration
    risk : RiskConfig
        Risk management configuration
    execution : ExecutionConfig
        Execution simulation configuration
    validate : ValidationConfig | None, default None
        Validation configuration
    report : ReportConfig
        Report generation configuration
    """

    seed: int = 42
    universe: UniverseConfig
    data: DataConfig
    alpha: AlphaConfig
    signal: SignalConfig
    portfolio: PortfolioConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    validate: ValidationConfig | None = None
    report: ReportConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> BacktestConfig:
        """Load configuration from YAML file.

        Parameters
        ----------
        path : str | Path
            Path to YAML configuration file

        Returns
        -------
        BacktestConfig
            Loaded and validated configuration

        Raises
        ------
        FileNotFoundError
            If configuration file does not exist
        ValueError
            If configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            msg = f"Configuration file not found: {path}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Parameters
        ----------
        path : str | Path
            Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
