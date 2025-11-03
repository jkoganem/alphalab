"""Momentum-based alpha models.

This module implements time-series and cross-sectional momentum strategies
with careful handling of look-ahead bias and optional neutralization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from alphalab.features.core import rank_normalize, winsorize, zscore_normalize

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TimeSeriesMomentum:
    """Time-Series Momentum (TSMOM) alpha model.

    Generates signals based on the sign and magnitude of past returns for each
    security individually. Positive past returns generate buy signals, negative
    past returns generate sell signals.

    Parameters
    ----------
    lookback_days : int, default 126
        Number of trading days to look back for momentum calculation
    cooldown_days : int, default 0
        Number of recent days to exclude from momentum calculation
        (e.g., skip last 5 days to avoid short-term reversals)
    winsor_pct : float | None, default 0.01
        Percentile for winsorization (e.g., 0.01 = clip at 1st/99th percentile)
        Set to None to disable winsorization
    return_method : str, default "log"
        Return calculation method: "log" or "simple"

    Examples
    --------
    >>> tsmom = TimeSeriesMomentum(lookback_days=126, cooldown_days=5)
    >>> alpha_scores = tsmom.score(features)
    """

    def __init__(
        self,
        lookback_days: int = 126,
        cooldown_days: int = 0,
        winsor_pct: float | None = 0.01,
        return_method: str = "log",
        vol_normalize: bool = True,
    ) -> None:
        """Initialize TSMOM alpha model."""
        self.lookback_days = lookback_days
        self.cooldown_days = cooldown_days
        self.winsor_pct = winsor_pct
        self.return_method = return_method
        self.vol_normalize = vol_normalize

        logger.info(
            f"Initialized TSMOM with lookback={lookback_days}, "
            f"cooldown={cooldown_days}, winsor={winsor_pct}, vol_normalize={vol_normalize}"
        )

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate TSMOM alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
            Must contain column matching pattern f"ret_{lookback_days}d"

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol) and single column 'alpha'

        Raises
        ------
        ValueError
            If required return column is not found
        """
        # Look for the return column
        ret_col = f"ret_{self.lookback_days}d"

        if ret_col not in features.columns:
            msg = f"Required column '{ret_col}' not found in features"
            raise ValueError(msg)

        logger.info(f"Computing TSMOM alpha using {ret_col}")

        # Extract returns
        returns = features[ret_col].copy()

        # Apply cooldown if specified
        if self.cooldown_days > 0:
            # Shift back by cooldown days to exclude recent returns
            returns = returns.groupby(level="symbol").shift(self.cooldown_days)

        # Normalize by volatility if specified (CRITICAL for time-series momentum)
        if self.vol_normalize:
            # Use 20-day volatility for normalization (standard practice)
            if "vol_20d" not in features.columns:
                logger.warning("vol_20d not found, skipping volatility normalization")
                alpha = returns
            else:
                vol = features["vol_20d"].copy()
                # Normalize returns by volatility (with small offset to avoid division by zero)
                alpha = returns / (vol + 0.01)
        else:
            alpha = returns

        # Apply winsorization if specified
        if self.winsor_pct is not None:
            alpha = winsorize(alpha, lower=self.winsor_pct, upper=1 - self.winsor_pct)

        # Create result DataFrame
        result = pd.DataFrame({"alpha": alpha}, index=features.index)

        # Count valid signals
        n_valid = result["alpha"].notna().sum()
        logger.info(f"Generated {n_valid} valid TSMOM alpha scores")

        return result


class CrossSectionalMomentum:
    """Cross-Sectional Momentum (XSMOM) alpha model.

    Ranks securities by past returns and generates relative signals. Unlike
    TSMOM which treats each security independently, XSMOM creates a ranking
    across all securities at each point in time.

    Parameters
    ----------
    lookback_days : int, default 126
        Number of trading days for momentum calculation
    neutralization : str, default "zscore"
        Neutralization method: "zscore", "rank", or "none"
    winsor_pct : float | None, default 0.01
        Percentile for winsorization before neutralization
    sector_neutral : bool, default False
        Whether to make signals sector-neutral (requires sector_map)

    Examples
    --------
    >>> xsmom = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
    >>> alpha_scores = xsmom.score(features)
    """

    def __init__(
        self,
        lookback_days: int = 126,
        neutralization: str = "zscore",
        winsor_pct: float | None = 0.01,
        sector_neutral: bool = False,
    ) -> None:
        """Initialize XSMOM alpha model."""
        self.lookback_days = lookback_days
        self.neutralization = neutralization
        self.winsor_pct = winsor_pct
        self.sector_neutral = sector_neutral

        if neutralization not in {"zscore", "rank", "none"}:
            msg = f"Invalid neutralization: {neutralization}. Use 'zscore', 'rank', or 'none'"
            raise ValueError(msg)

        logger.info(
            f"Initialized XSMOM with lookback={lookback_days}, "
            f"neutralization={neutralization}, sector_neutral={sector_neutral}"
        )

    def score(
        self, features: pd.DataFrame, sector_map: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Generate XSMOM alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        sector_map : dict[str, str] | None, optional
            Mapping from symbol to sector (required if sector_neutral=True)

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol) and single column 'alpha'

        Raises
        ------
        ValueError
            If required return column is not found or sector_map is missing
        """
        # Look for the return column
        ret_col = f"ret_{self.lookback_days}d"

        if ret_col not in features.columns:
            msg = f"Required column '{ret_col}' not found in features"
            raise ValueError(msg)

        if self.sector_neutral and sector_map is None:
            msg = "sector_map is required when sector_neutral=True"
            raise ValueError(msg)

        logger.info(f"Computing XSMOM alpha using {ret_col}")

        # Extract returns
        returns = features[[ret_col]].copy()
        returns.columns = ["alpha"]

        # Apply winsorization if specified
        if self.winsor_pct is not None:
            returns["alpha"] = winsorize(
                returns["alpha"], lower=self.winsor_pct, upper=1 - self.winsor_pct
            )

        # Apply neutralization
        if self.neutralization == "zscore":
            # Cross-sectional z-score (normalize within each date)
            result = zscore_normalize(returns, by_date=True)
        elif self.neutralization == "rank":
            # Cross-sectional ranking (0 to 1 scale)
            result = rank_normalize(returns, by_date=True)
        else:  # none
            result = returns

        # Apply sector neutralization if requested
        if self.sector_neutral and sector_map is not None:
            from alphalab.features.core import sector_neutralize

            result = sector_neutralize(result, sector_map)

        # Count valid signals
        n_valid = result["alpha"].notna().sum()
        logger.info(f"Generated {n_valid} valid XSMOM alpha scores")

        return result


class DualMomentum:
    """Dual Momentum combining time-series and cross-sectional momentum.

    Combines TSMOM (absolute momentum) with XSMOM (relative momentum) by
    taking a weighted average of both signals.

    Parameters
    ----------
    lookback_days : int, default 126
        Lookback period for both TSMOM and XSMOM
    ts_weight : float, default 0.5
        Weight for time-series momentum (1 - ts_weight goes to cross-sectional)
    neutralization : str, default "zscore"
        Neutralization method for cross-sectional component

    Examples
    --------
    >>> dual = DualMomentum(lookback_days=126, ts_weight=0.5)
    >>> alpha_scores = dual.score(features)
    """

    def __init__(
        self,
        lookback_days: int = 126,
        ts_weight: float = 0.5,
        neutralization: str = "zscore",
    ) -> None:
        """Initialize Dual Momentum alpha model."""
        if not 0 <= ts_weight <= 1:
            msg = f"ts_weight must be between 0 and 1, got {ts_weight}"
            raise ValueError(msg)

        self.lookback_days = lookback_days
        self.ts_weight = ts_weight
        self.xs_weight = 1 - ts_weight

        # Initialize component models
        self.tsmom = TimeSeriesMomentum(lookback_days=lookback_days)
        self.xsmom = CrossSectionalMomentum(
            lookback_days=lookback_days, neutralization=neutralization
        )

        logger.info(
            f"Initialized Dual Momentum with ts_weight={ts_weight}, "
            f"xs_weight={self.xs_weight}"
        )

    def score(
        self, features: pd.DataFrame, sector_map: dict[str, str] | None = None
    ) -> pd.DataFrame:
        """Generate Dual Momentum alpha scores.

        Parameters
        ----------
        features : pd.DataFrame
            Feature DataFrame with MultiIndex (date, symbol)
        sector_map : dict[str, str] | None, optional
            Mapping from symbol to sector

        Returns
        -------
        pd.DataFrame
            Alpha scores with MultiIndex (date, symbol) and single column 'alpha'
        """
        logger.info("Computing Dual Momentum alpha")

        # Get TSMOM scores
        ts_scores = self.tsmom.score(features)

        # Get XSMOM scores
        xs_scores = self.xsmom.score(features, sector_map=sector_map)

        # Combine with weights
        combined = pd.DataFrame(index=features.index)
        combined["alpha"] = (
            self.ts_weight * ts_scores["alpha"] + self.xs_weight * xs_scores["alpha"]
        )

        n_valid = combined["alpha"].notna().sum()
        logger.info(f"Generated {n_valid} valid Dual Momentum alpha scores")

        return combined


def create_momentum_alpha(
    alpha_type: str = "xsmom", **kwargs: object
) -> TimeSeriesMomentum | CrossSectionalMomentum | DualMomentum:
    """Factory function to create momentum alpha models.

    Parameters
    ----------
    alpha_type : str, default "xsmom"
        Type of momentum alpha: "tsmom", "xsmom", or "dual"
    **kwargs : object
        Parameters passed to the alpha model constructor

    Returns
    -------
    TimeSeriesMomentum | CrossSectionalMomentum | DualMomentum
        Momentum alpha model instance

    Raises
    ------
    ValueError
        If alpha_type is not recognized
    """
    if alpha_type == "tsmom":
        return TimeSeriesMomentum(**kwargs)  # type: ignore[arg-type]
    elif alpha_type == "xsmom":
        return CrossSectionalMomentum(**kwargs)  # type: ignore[arg-type]
    elif alpha_type == "dual":
        return DualMomentum(**kwargs)  # type: ignore[arg-type]
    else:
        msg = f"Unknown momentum alpha type: {alpha_type}"
        raise ValueError(msg)
