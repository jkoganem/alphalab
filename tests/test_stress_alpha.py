"""Stress tests for alpha models under extreme conditions.

These tests verify alpha models can handle:
- Extreme feature values (outliers, infinities)
- Missing data and NaN values
- Single symbol edge cases
- Perfect correlations
- Zero variance scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.alpha.mean_reversion import MeanReversion
from alphalab.alpha.momentum import CrossSectionalMomentum, TimeSeriesMomentum


@pytest.fixture
def extreme_features() -> pd.DataFrame:
    """Generate features with extreme values."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["NORMAL", "OUTLIER", "ZERO"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    n_dates = len(dates)

    # Normal returns for first symbol
    returns_normal = np.random.normal(0.001, 0.02, n_dates)

    # Extreme returns for outlier
    returns_outlier = np.random.choice([-0.9, 0.9], n_dates)  # 90% swings

    # Zero returns for third symbol
    returns_zero = np.zeros(n_dates)

    # Stack returns
    returns = np.column_stack([returns_normal, returns_outlier, returns_zero]).flatten()

    # Create features
    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": returns * 126,  # Cumulative
            "volatility_20d": np.abs(returns) * 10,
            "zscore_20d": returns / (np.abs(returns).mean() + 1e-8),
        },
        index=index,
    )

    return features


@pytest.fixture
def missing_features() -> pd.DataFrame:
    """Generate features with missing data."""
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(index))

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": returns * 126,
            "volatility_20d": np.abs(returns) * 10,
        },
        index=index,
    )

    # Introduce missing data (30% of rows)
    missing_mask = np.random.random(len(features)) < 0.3
    features.loc[missing_mask, "ret_126d"] = np.nan

    return features


def test_time_series_momentum_extreme_values(extreme_features):
    """Test TS momentum with extreme return values."""
    alpha_model = TimeSeriesMomentum(lookback_days=126)
    alpha = alpha_model.score(extreme_features)

    # Should produce alpha scores without crashing
    assert "alpha" in alpha.columns
    assert len(alpha) > 0

    # Should not produce infinite alpha
    assert not np.any(np.isinf(alpha["alpha"]))


def test_cross_sectional_momentum_extreme_values(extreme_features):
    """Test CS momentum with extreme return values."""
    alpha_model = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
    alpha = alpha_model.score(extreme_features)

    # Should handle outliers via z-score normalization
    assert "alpha" in alpha.columns
    assert not np.any(np.isinf(alpha["alpha"]))

    # Z-scores should be roughly bounded
    alpha_values = alpha["alpha"].dropna()
    assert alpha_values.max() < 10  # Extreme outliers removed
    assert alpha_values.min() > -10


def test_mean_reversion_zero_variance():
    """Test mean reversion with zero variance (flat prices)."""
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D", tz="UTC")
    symbols = ["FLAT"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Flat prices -> zero variance
    features = pd.DataFrame(
        {
            "ret_1d": 0.0,
            "ret_20d": 0.0,
            "volatility_20d": 0.0,
            "zscore_20d": 0.0,
        },
        index=index,
    )

    alpha_model = MeanReversion(lookback_days=20, entry_threshold=1.0)
    alpha = alpha_model.score(features)

    # Should produce zero or NaN alpha (no mean reversion signal)
    assert "alpha" in alpha.columns
    # Either all NaN or all zero
    assert np.all(alpha["alpha"].isna()) or np.all(alpha["alpha"] == 0)


def test_time_series_momentum_missing_data(missing_features):
    """Test TS momentum with missing feature data."""
    alpha_model = TimeSeriesMomentum(lookback_days=126)
    alpha = alpha_model.score(missing_features)

    # Should handle missing data gracefully
    assert "alpha" in alpha.columns

    # Some alpha scores should be valid
    valid_alpha = alpha["alpha"].notna().sum()
    assert valid_alpha > 0


def test_cross_sectional_momentum_single_symbol():
    """Test CS momentum with only one symbol (can't cross-sectionally rank)."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["ONLY_ONE"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": np.cumsum(returns),
            "volatility_20d": np.abs(returns) * 10,
        },
        index=index,
    )

    alpha_model = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
    alpha = alpha_model.score(features)

    # With one symbol, z-score normalization should produce NaN or 0
    assert "alpha" in alpha.columns
    # Either all NaN (no variation to normalize) or all 0 (demean with 1 symbol)
    assert np.all(alpha["alpha"].isna()) or np.allclose(
        alpha["alpha"].fillna(0), 0, atol=1e-10
    )


def test_momentum_perfect_correlation():
    """Test momentum when all symbols move identically."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["CLONE1", "CLONE2", "CLONE3"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    # Same returns for all symbols
    returns_base = np.random.normal(0.001, 0.02, len(dates))
    returns = np.tile(returns_base, len(symbols))

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": returns * 126,
            "volatility_20d": np.abs(returns) * 10,
        },
        index=index,
    )

    alpha_model = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
    alpha = alpha_model.score(features)

    # With perfect correlation, cross-sectional alpha should be near zero
    # (no relative outperformance)
    assert "alpha" in alpha.columns
    alpha_values = alpha["alpha"].dropna()
    if len(alpha_values) > 0:
        assert np.abs(alpha_values.mean()) < 0.01  # Near zero on average


def test_momentum_extreme_negative_returns():
    """Test momentum during market crash (all negative returns)."""
    dates = pd.date_range("2020-03-01", "2020-03-31", freq="D", tz="UTC")
    symbols = ["CRASH1", "CRASH2", "CRASH3"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    # All symbols declining, but at different rates
    returns = np.random.normal(-0.05, 0.02, len(index))  # -5% daily average

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": returns * 126,
            "volatility_20d": 0.5,  # High volatility
        },
        index=index,
    )

    # TS Momentum should produce negative signals
    alpha_ts = TimeSeriesMomentum(lookback_days=126).score(features)
    assert alpha_ts["alpha"].mean() < 0  # Mostly negative

    # CS Momentum should still rank relative performance
    alpha_cs = CrossSectionalMomentum(
        lookback_days=126, neutralization="zscore"
    ).score(features)
    assert "alpha" in alpha_cs.columns
    assert not np.all(alpha_cs["alpha"].isna())  # Should produce some rankings


def test_mean_reversion_extreme_z_scores():
    """Test mean reversion with extreme z-score deviations."""
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D", tz="UTC")
    symbols = ["EXTREME"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Extreme z-scores (10 standard deviations)
    zscores = np.array([10.0] * 15 + [-10.0] * 16)

    features = pd.DataFrame(
        {
            "ret_1d": zscores * 0.01,
            "zscore_20d": zscores,
            "volatility_20d": 0.3,
        },
        index=index,
    )

    alpha_model = MeanReversion(lookback_days=20, entry_threshold=2.0)
    alpha = alpha_model.score(features)

    # Should produce strong mean reversion signals
    assert "alpha" in alpha.columns
    assert alpha["alpha"].abs().max() > 5  # Strong signals


def test_momentum_constant_prices():
    """Test momentum with constant prices (no movement)."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["CONSTANT"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Zero returns throughout
    features = pd.DataFrame(
        {
            "ret_1d": 0.0,
            "ret_126d": 0.0,
            "volatility_20d": 0.0,
        },
        index=index,
    )

    # TS Momentum should produce zero alpha
    alpha_ts = TimeSeriesMomentum(lookback_days=126).score(features)
    assert np.all(alpha_ts["alpha"] == 0) or np.all(alpha_ts["alpha"].isna())


def test_alpha_with_inf_values():
    """Test alpha models handle infinite feature values."""
    dates = pd.date_range("2023-01-01", "2023-01-10", freq="D", tz="UTC")
    symbols = ["INF_STOCK"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    features = pd.DataFrame(
        {
            "ret_1d": [0.1, 0.2, np.inf, 0.3, -np.inf, 0.1, 0.2, 0.1, 0.2, 0.1],
            "ret_126d": [1.0] * 10,
            "volatility_20d": [0.3] * 10,
        },
        index=index,
    )

    alpha_model = TimeSeriesMomentum(lookback_days=126)
    alpha = alpha_model.score(features)

    # Should handle inf by replacing with NaN or clipping
    assert not np.any(np.isinf(alpha["alpha"]))


def test_alpha_large_feature_values():
    """Test alpha models with very large (but not infinite) feature values."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["LARGE"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Very large but valid returns (10000%)
    features = pd.DataFrame(
        {
            "ret_1d": 100.0,
            "ret_126d": 12600.0,
            "volatility_20d": 500.0,
        },
        index=index,
    )

    alpha_model = TimeSeriesMomentum(lookback_days=126)
    alpha = alpha_model.score(features)

    # Should produce valid (though large) alpha
    assert "alpha" in alpha.columns
    assert not np.any(np.isnan(alpha["alpha"]))
    assert not np.any(np.isinf(alpha["alpha"]))


def test_mean_reversion_all_same_values():
    """Test mean reversion when all symbols have identical values."""
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D", tz="UTC")
    symbols = ["SAME1", "SAME2", "SAME3"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # All symbols have same z-score
    features = pd.DataFrame(
        {
            "ret_1d": 0.01,
            "zscore_20d": 1.5,  # All at 1.5 std dev
            "volatility_20d": 0.2,
        },
        index=index,
    )

    alpha_model = MeanReversion(lookback_days=20, entry_threshold=1.0)
    alpha = alpha_model.score(features)

    # Should produce same alpha for all symbols
    assert "alpha" in alpha.columns
    unique_alphas = alpha["alpha"].dropna().unique()
    assert len(unique_alphas) == 1  # All same value


@pytest.mark.slow
def test_momentum_very_long_history():
    """Test momentum with very long lookback (5 years)."""
    dates = pd.date_range("2014-01-01", "2024-01-01", freq="D", tz="UTC")
    symbols = ["LONG_HISTORY"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    cumulative = np.cumsum(returns)

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_1260d": cumulative,  # 5 year cumulative
            "volatility_20d": 0.2,
        },
        index=index,
    )

    # Long lookback momentum
    alpha_model = TimeSeriesMomentum(lookback_days=1260)
    alpha = alpha_model.score(features)

    # Should handle long history
    assert "alpha" in alpha.columns
    assert len(alpha) > 2000


def test_cross_sectional_with_two_symbols():
    """Test CS momentum with minimum viable symbols (2)."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    symbols = ["WINNER", "LOSER"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    # One goes up, one goes down
    returns_winner = np.random.normal(0.01, 0.02, len(dates))
    returns_loser = np.random.normal(-0.01, 0.02, len(dates))

    returns = np.empty(len(index))
    returns[::2] = returns_winner
    returns[1::2] = returns_loser

    features = pd.DataFrame(
        {
            "ret_1d": returns,
            "ret_126d": returns * 126,
            "volatility_20d": 0.2,
        },
        index=index,
    )

    alpha_model = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
    alpha = alpha_model.score(features)

    # Should produce opposite signs for winner/loser
    assert "alpha" in alpha.columns

    # Check that winner has positive alpha, loser has negative
    alpha_winner = alpha.loc[(slice(None), "WINNER"), "alpha"]
    alpha_loser = alpha.loc[(slice(None), "LOSER"), "alpha"]

    assert alpha_winner.mean() > alpha_loser.mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
