"""Signal conversion utilities.

This module converts continuous alpha scores into discrete trading signals
with various methods including ranking, thresholding, and neutralization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphalab.features.core import zscore_normalize

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RankLongShort:
    """Rank-based long-short signal converter.

    Ranks securities by alpha scores and goes long the top percentile
    and short the bottom percentile.

    Parameters
    ----------
    long_pct : float, default 0.2
        Percentile of top securities to go long (e.g., 0.2 = top 20%)
    short_pct : float, default 0.2
        Percentile of bottom securities to go short
    equal_weight : bool, default True
        If True, equal weight within long/short baskets
        If False, weight proportional to alpha scores
    neutralize_beta_to : str | None, default None
        Benchmark symbol for beta neutralization (e.g., "SPY")

    Examples
    --------
    >>> converter = RankLongShort(long_pct=0.3, short_pct=0.3)
    >>> signals = converter.to_signal(alpha_scores, features)
    """

    def __init__(
        self,
        long_pct: float = 0.2,
        short_pct: float = 0.2,
        equal_weight: bool = True,
        neutralize_beta_to: str | None = None,
    ) -> None:
        """Initialize rank long-short converter."""
        if not 0 < long_pct <= 1:
            msg = f"long_pct must be in (0, 1], got {long_pct}"
            raise ValueError(msg)
        if not 0 < short_pct <= 1:
            msg = f"short_pct must be in (0, 1], got {short_pct}"
            raise ValueError(msg)
        if long_pct + short_pct > 1:
            logger.warning(
                f"long_pct ({long_pct}) + short_pct ({short_pct}) > 1, "
                "signals may overlap"
            )

        self.long_pct = long_pct
        self.short_pct = short_pct
        self.equal_weight = equal_weight
        self.neutralize_beta_to = neutralize_beta_to

        logger.info(
            f"Initialized RankLongShort with long_pct={long_pct}, "
            f"short_pct={short_pct}, equal_weight={equal_weight}"
        )

    def to_signal(
        self, alpha: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Convert alpha scores to long-short signals.

        Parameters
        ----------
        alpha : pd.DataFrame
            Alpha scores with MultiIndex (date, symbol) and 'alpha' column
        features : pd.DataFrame | None, optional
            Features needed for beta neutralization

        Returns
        -------
        pd.DataFrame
            Trading signals with MultiIndex (date, symbol) and 'signal' column
            Positive values = long, negative = short, zero = no position
        """
        logger.info("Converting alpha to rank long-short signals")

        # Get alpha values
        alpha_values = alpha["alpha"].copy()

        # Rank within each date (cross-sectional)
        ranks = alpha_values.groupby(level="date").rank(pct=True, method="average")

        # Initialize signals to zero
        signals = pd.Series(0.0, index=alpha.index)

        # Long top percentile
        long_threshold = 1 - self.long_pct
        signals = signals.where(ranks <= long_threshold, 1.0)

        # Short bottom percentile
        short_threshold = self.short_pct
        signals = signals.where(ranks >= short_threshold, -1.0)

        # Set middle to zero
        signals = signals.where((ranks <= short_threshold) | (ranks >= long_threshold), 0.0)

        # Weight by alpha scores if not equal weight
        if not self.equal_weight:
            # Scale signals by normalized alpha scores
            alpha_normed = zscore_normalize(
                pd.DataFrame({"alpha": alpha_values}), by_date=True
            )["alpha"]
            # Only apply to non-zero signals
            signals = signals * np.sign(signals) * np.abs(alpha_normed).where(signals != 0, 0)

        # Neutralize beta if requested
        if self.neutralize_beta_to is not None:
            if features is None:
                logger.warning("Cannot neutralize beta: features not provided")
            else:
                signals = self._neutralize_beta(signals, features)

        result = pd.DataFrame({"signal": signals}, index=alpha.index)

        # Report statistics
        n_long = (result["signal"] > 0).sum()
        n_short = (result["signal"] < 0).sum()
        n_zero = (result["signal"] == 0).sum()
        logger.info(f"Generated signals: {n_long} long, {n_short} short, {n_zero} neutral")

        return result

    def _neutralize_beta(
        self, signals: pd.Series, features: pd.DataFrame
    ) -> pd.Series:
        """Neutralize portfolio beta to benchmark.

        Parameters
        ----------
        signals : pd.Series
            Raw signals
        features : pd.DataFrame
            Features containing beta estimates

        Returns
        -------
        pd.Series
            Beta-neutralized signals
        """
        # Look for beta column (try multiple lookback windows)
        beta_col = None
        for window in [60, 126, 252]:
            col = f"beta_{window}d"
            if col in features.columns:
                beta_col = col
                break

        if beta_col is None:
            logger.warning("No beta column found in features, skipping neutralization")
            return signals

        logger.info(f"Neutralizing beta using {beta_col}")

        # Get betas
        betas = features[beta_col]

        # For each date, adjust signals to make portfolio beta = 0
        def neutralize_date(group: pd.DataFrame) -> pd.Series:
            date_signals = group["signal"]
            date_betas = group["beta"]

            # Skip if insufficient data
            if date_signals.notna().sum() < 2 or date_betas.notna().sum() < 2:
                return date_signals

            # Portfolio beta = weighted average of individual betas
            # We want: sum(w_i * beta_i) = 0
            # Adjust: w_i' = w_i - beta_i * (sum(w_j * beta_j) / sum(beta_j^2))

            portfolio_beta = (date_signals * date_betas).sum()
            beta_squared_sum = (date_betas**2).sum()

            if beta_squared_sum > 0:
                adjustment = date_betas * (portfolio_beta / beta_squared_sum)
                neutralized = date_signals - adjustment
                return neutralized
            else:
                return date_signals

        # Combine signals and betas
        combined = pd.DataFrame({"signal": signals, "beta": betas})

        # Apply neutralization within each date
        neutralized = combined.groupby(level="date", group_keys=False).apply(neutralize_date)

        return neutralized

class ThresholdSignal:
    """Threshold-based signal converter.

    Converts alpha scores to +1/-1/0 signals based on thresholds.

    Parameters
    ----------
    long_threshold : float, default 0.5
        Alpha threshold for long positions
    short_threshold : float, default -0.5
        Alpha threshold for short positions
    normalize_first : bool, default True
        Whether to z-score normalize alpha before applying thresholds

    Examples
    --------
    >>> converter = ThresholdSignal(long_threshold=1.0, short_threshold=-1.0)
    >>> signals = converter.to_signal(alpha_scores)
    """

    def __init__(
        self,
        long_threshold: float = 0.5,
        short_threshold: float = -0.5,
        normalize_first: bool = True,
    ) -> None:
        """Initialize threshold signal converter."""
        if long_threshold <= short_threshold:
            msg = f"long_threshold ({long_threshold}) must be > short_threshold ({short_threshold})"
            raise ValueError(msg)

        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.normalize_first = normalize_first

        logger.info(
            f"Initialized ThresholdSignal with thresholds: "
            f"long>{long_threshold}, short<{short_threshold}"
        )

    def to_signal(
        self, alpha: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Convert alpha scores to threshold signals.

        Parameters
        ----------
        alpha : pd.DataFrame
            Alpha scores with 'alpha' column
        features : pd.DataFrame | None, optional
            Not used, for interface compatibility

        Returns
        -------
        pd.DataFrame
            Binary signals (+1, 0, -1)
        """
        logger.info("Converting alpha to threshold signals")

        alpha_values = alpha["alpha"].copy()

        # Normalize if requested
        if self.normalize_first:
            alpha_normed = zscore_normalize(
                pd.DataFrame({"alpha": alpha_values}), by_date=True
            )["alpha"]
        else:
            alpha_normed = alpha_values

        # Apply thresholds
        signals = pd.Series(0, index=alpha.index, dtype=float)
        signals = signals.where(alpha_normed <= self.long_threshold, 1.0)
        signals = signals.where(alpha_normed >= self.short_threshold, -1.0)
        signals = signals.where(
            (alpha_normed <= self.short_threshold) | (alpha_normed >= self.long_threshold), 0.0
        )

        result = pd.DataFrame({"signal": signals}, index=alpha.index)

        n_long = (result["signal"] > 0).sum()
        n_short = (result["signal"] < 0).sum()
        logger.info(f"Generated {n_long} long, {n_short} short signals")

        return result


class ScaledSignal:
    """Scaled signal converter.

    Scales alpha scores to target range, optionally with normalization.

    Parameters
    ----------
    scale_factor : float, default 1.0
        Scaling factor for signals
    normalize : bool, default True
        Whether to z-score normalize before scaling
    clip_std : float | None, default 3.0
        Clip signals at N standard deviations

    Examples
    --------
    >>> converter = ScaledSignal(scale_factor=2.0, clip_std=3.0)
    >>> signals = converter.to_signal(alpha_scores)
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        normalize: bool = True,
        clip_std: float | None = 3.0,
    ) -> None:
        """Initialize scaled signal converter."""
        self.scale_factor = scale_factor
        self.normalize = normalize
        self.clip_std = clip_std

        logger.info(
            f"Initialized ScaledSignal with scale={scale_factor}, "
            f"normalize={normalize}, clip_std={clip_std}"
        )

    def to_signal(
        self, alpha: pd.DataFrame, features: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Convert alpha scores to scaled signals.

        Parameters
        ----------
        alpha : pd.DataFrame
            Alpha scores
        features : pd.DataFrame | None, optional
            Not used

        Returns
        -------
        pd.DataFrame
            Scaled signals
        """
        logger.info("Converting alpha to scaled signals")

        alpha_values = alpha["alpha"].copy()

        # Normalize if requested
        if self.normalize:
            signals = zscore_normalize(
                pd.DataFrame({"alpha": alpha_values}), by_date=True
            )["alpha"]
        else:
            signals = alpha_values

        # Clip if requested
        if self.clip_std is not None:
            signals = signals.clip(lower=-self.clip_std, upper=self.clip_std)

        # Scale
        signals = signals * self.scale_factor

        result = pd.DataFrame({"signal": signals}, index=alpha.index)

        logger.info(
            f"Scaled signals - mean: {result['signal'].mean():.3f}, "
            f"std: {result['signal'].std():.3f}"
        )

        return result


def create_signal_converter(
    method: str, **params: object
) -> RankLongShort | ThresholdSignal | ScaledSignal:
    """Factory function to create signal converters.

    Parameters
    ----------
    method : str
        Conversion method: "rank_long_short", "threshold", or "scaled"
    **params : object
        Method-specific parameters

    Returns
    -------
    RankLongShort | ThresholdSignal | ScaledSignal
        Signal converter instance

    Raises
    ------
    ValueError
        If method is not recognized
    """
    if method == "rank_long_short":
        return RankLongShort(**params)  # type: ignore[arg-type]
    elif method == "threshold":
        return ThresholdSignal(**params)  # type: ignore[arg-type]
    elif method == "scaled":
        return ScaledSignal(**params)  # type: ignore[arg-type]
    else:
        msg = f"Unknown signal conversion method: {method}"
        raise ValueError(msg)
