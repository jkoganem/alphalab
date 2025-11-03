"""Mean reversion alpha models.

This module implements mean reversion strategies that profit from
short-term price reversals and overreactions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphalab.features.core import compute_zscore, winsorize

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MeanReversion:
    """Mean Reversion (MR) alpha model.

    Generates contrarian signals based on short-term price deviations from
    a moving average. Assumes prices will revert to their mean.

    Parameters
    ----------
    lookback_days : int, default 20
        Window for calculating mean and standard deviation
    short_term_days : int, default 5
        Recent period for measuring deviation
    entry_threshold : float, default 2.0
        Z-score threshold for entry (e.g., 2.0 = 2 std devs)
    winsor_pct : float | None, default 0.01
        Percentile for winsorization
    cross_sectional : bool, default False
        If True, normalize z-scores cross-sectionally

    Examples
    --------
    >>> mr = MeanReversion(lookback_days=20, entry_threshold=2.0)
    >>> alpha_scores = mr.score(features)
    """

    def __init__(
        self,
        lookback_days: int = 20,
        short_term_days: int = 5,
        entry_threshold: float = 2.0,
        winsor_pct: float | None = 0.01,
        cross_sectional: bool = False,
        vol_normalize: bool = True,
        vix_scale: bool = False,
    ) -> None:
        """Initialize Mean Reversion alpha model."""
        self.lookback_days = lookback_days
        self.short_term_days = short_term_days
        self.entry_threshold = entry_threshold
        self.winsor_pct = winsor_pct
        self.cross_sectional = cross_sectional
        self.vol_normalize = vol_normalize
        self.vix_scale = vix_scale

        logger.info(
            f"Initialized MR with lookback={lookback_days}, "
            f"short_term={short_term_days}, threshold={entry_threshold}, "
            f"vol_normalize={vol_normalize}, vix_scale={vix_scale}"
        )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
            Should contain z-score columns

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol) and single column 'alpha'
            Negative z-scores -> positive alpha (buy oversold)
            Positive z-scores -> negative alpha (sell overbought)
        """
        # Look for z-score column matching our lookback
        zscore_col = f"zscore_{self.lookback_days}d"

        if zscore_col in features.columns:
            logger.info(f"Using existing {zscore_col} from features")
            zscores = features[zscore_col].copy()
        else:
            # Calculate z-scores from short-term returns
            ret_col = f"ret_{self.short_term_days}d"
            if ret_col not in features.columns:
                msg = f"Required column '{ret_col}' or '{zscore_col}' not found"
                raise ValueError(msg)

            logger.info(f"Computing z-scores from {ret_col}")
            returns = features[ret_col]
            zscores = compute_zscore(returns, window=self.lookback_days)

        # Apply winsorization if specified
        if self.winsor_pct is not None:
            zscores = winsorize(zscores, lower=self.winsor_pct, upper=1 - self.winsor_pct)

        # Mean reversion signal: negative z-score (oversold) -> buy (+1)
        # positive z-score (overbought) -> sell (-1)
        alpha = -zscores  # Flip sign for contrarian

        # Optional: apply threshold (only trade extreme deviations)
        if self.entry_threshold > 0:
            # Only generate signals when |z-score| > threshold
            alpha = alpha.where(np.abs(zscores) > self.entry_threshold, 0)

        # Volatility normalization (CRITICAL for mean reversion)
        if self.vol_normalize:
            if "vol_20d" not in features.columns:
                logger.warning("vol_20d not found, skipping volatility normalization")
            else:
                vol = features["vol_20d"].copy()
                # Scale signals inversely by volatility (lower vol = stronger signal)
                # High volatility stocks have noisier mean reversion
                alpha = alpha / (vol + 0.01)
                logger.info("Applied volatility normalization to mean reversion signals")

        # VIX scaling (academic research: 46% correlation between VIX and reversal strength)
        if self.vix_scale:
            # Look for VIX in features (should be available as macro_VIX from FRED)
            vix_col = None
            for col in features.columns:
                if "vix" in col.lower():
                    vix_col = col
                    break

            if vix_col is not None:
                vix = features[vix_col].copy()
                # Normalize VIX to 0-1 scale (typical range: 10-40)
                vix_normalized = (vix - 10) / 30
                vix_normalized = vix_normalized.clip(0, 1)
                # Scale signals: higher VIX = stronger mean reversion signals
                alpha = alpha * (1 + vix_normalized)
                logger.info(f"Applied VIX scaling using {vix_col}")
            else:
                logger.warning("VIX column not found, skipping VIX scaling")

        # Create result DataFrame
        result = pd.DataFrame({"alpha": alpha}, index=features.index)

        # Optional: cross-sectional normalization
        if self.cross_sectional:
            from alphalab.features.core import zscore_normalize

            result = zscore_normalize(result, by_date=True)

        n_valid = result["alpha"].notna().sum()
        n_signals = (result["alpha"] != 0).sum()
        logger.info(
            f"Generated {n_valid} valid scores, {n_signals} non-zero signals "
            f"({n_signals/n_valid*100:.1f}%)"
        )

        return result


class ResiduaMeanReversion:
    """Residual Mean Reversion using market beta.

    Generates signals based on deviations from expected returns given market beta.
    More sophisticated than simple mean reversion as it accounts for market exposure.

    Parameters
    ----------
    lookback_days : int, default 20
        Window for calculating residuals
    beta_window : int, default 60
        Window for calculating beta
    entry_threshold : float, default 1.5
        Z-score threshold for residual
    benchmark_symbol : str, default "SPY"
        Benchmark for beta calculation

    Examples
    --------
    >>> rmr = ResiduaMeanReversion(lookback_days=20, beta_window=60)
    >>> alpha_scores = rmr.score(features)
    """

    def __init__(
        self,
        lookback_days: int = 20,
        beta_window: int = 60,
        entry_threshold: float = 1.5,
        benchmark_symbol: str = "SPY",
    ) -> None:
        """Initialize Residual Mean Reversion alpha model."""
        self.lookback_days = lookback_days
        self.beta_window = beta_window
        self.entry_threshold = entry_threshold
        self.benchmark_symbol = benchmark_symbol

        logger.info(
            f"Initialized Residual MR with lookback={lookback_days}, "
            f"beta_window={beta_window}, benchmark={benchmark_symbol}"
        )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate residual mean reversion alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
            Should contain beta and return columns

        Returns
        -------
        pd.DataFrame
            Alpha scores based on residuals from market model
        """
        # Get beta column
        beta_col = f"beta_{self.beta_window}d"

        if beta_col not in features.columns:
            msg = f"Required column '{beta_col}' not found. Need beta estimates."
            raise ValueError(msg)

        # Get short-term returns
        ret_col = f"ret_{self.lookback_days}d"
        if ret_col not in features.columns:
            # Try 5-day returns as alternative
            ret_col = "ret_5d"
            if ret_col not in features.columns:
                msg = f"No suitable return column found for lookback={self.lookback_days}"
                raise ValueError(msg)

        logger.info(f"Computing residual MR using {beta_col} and {ret_col}")

        # Extract data
        betas = features[beta_col].unstack("symbol")
        returns = features[ret_col].unstack("symbol")

        # Get benchmark returns (should be in features)
        if self.benchmark_symbol not in returns.columns:
            logger.warning(
                f"Benchmark {self.benchmark_symbol} not found, using simple MR instead"
            )
            # Fall back to simple mean reversion
            zscores = compute_zscore(returns.stack("symbol"), window=self.lookback_days)
            alpha = -zscores
        else:
            # Calculate expected returns based on beta
            benchmark_returns = returns[self.benchmark_symbol]

            # Expected return = beta * benchmark_return
            expected_returns = betas.mul(benchmark_returns, axis=0)

            # Residuals = actual - expected
            residuals = returns - expected_returns

            # Z-score of residuals
            residual_series = residuals.stack("symbol")
            zscores = compute_zscore(residual_series, window=self.lookback_days)

            # Mean reversion on residuals (contrarian)
            alpha = -zscores

            # Apply threshold
            if self.entry_threshold > 0:
                alpha = alpha.where(np.abs(zscores) > self.entry_threshold, 0)

        result = pd.DataFrame({"alpha": alpha}, index=features.index)

        n_valid = result["alpha"].notna().sum()
        n_signals = (result["alpha"] != 0).sum()
        logger.info(f"Generated {n_valid} valid residual MR scores, {n_signals} signals")

        return result


def create_mean_reversion_alpha(
    alpha_type: str = "simple", **kwargs: object
) -> MeanReversion | ResiduaMeanReversion:
    """Factory function to create mean reversion alpha models.

    Parameters
    ----------
    alpha_type : str, default "simple"
        Type of mean reversion: "simple" or "residual"
    **kwargs : object
        Parameters passed to the alpha model constructor

    Returns
    -------
    MeanReversion | ResiduaMeanReversion
        Mean reversion alpha model instance

    Raises
    ------
    ValueError
        If alpha_type is not recognized
    """
    if alpha_type == "simple":
        return MeanReversion(**kwargs)  # type: ignore[arg-type]
    elif alpha_type == "residual":
        return ResiduaMeanReversion(**kwargs)  # type: ignore[arg-type]
    else:
        msg = f"Unknown mean reversion alpha type: {alpha_type}"
        raise ValueError(msg)
