"""Tests for alpha models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.alpha.mean_reversion import MeanReversion, ResiduaMeanReversion
from alphalab.alpha.ml import MLAlpha
from alphalab.alpha.momentum import CrossSectionalMomentum, TimeSeriesMomentum
from alphalab.alpha.pairs import PairsTradingAlpha, SimplePairsAlpha


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature data for testing."""
    dates = pd.date_range("2020-01-01", periods=250, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]

    # Create MultiIndex
    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Generate synthetic features
    np.random.seed(42)
    data = {
        "ret_5d": np.random.randn(len(index)) * 0.02,
        "ret_20d": np.random.randn(len(index)) * 0.05,
        "ret_60d": np.random.randn(len(index)) * 0.08,
        "ret_126d": np.random.randn(len(index)) * 0.12,
        "vol_20d": np.abs(np.random.randn(len(index)) * 0.02) + 0.01,
        "vol_60d": np.abs(np.random.randn(len(index)) * 0.03) + 0.01,
        "zscore_20d": np.random.randn(len(index)),
        "beta_60d": np.random.randn(len(index)) * 0.5 + 1.0,
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range("2020-01-01", periods=250, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "SPY"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    # Generate synthetic prices (random walk)
    np.random.seed(42)
    base_prices = {"AAPL": 150, "MSFT": 200, "GOOGL": 1500, "SPY": 300}

    prices = []
    for date in dates:
        for symbol in symbols:
            # Random walk with drift
            if date == dates[0]:
                price = base_prices[symbol]
            else:
                prev_price = prices[-1] if symbol == symbols[-1] else prices[-1]
                price = prev_price * (1 + np.random.randn() * 0.02)
            prices.append(price)

    return pd.DataFrame({"close": prices}, index=index)


class TestTimeSeriesMomentum:
    """Tests for TSMOM alpha model."""

    def test_initialization(self):
        """Test TSMOM initialization."""
        tsmom = TimeSeriesMomentum(lookback_days=126, winsor_pct=0.01)
        assert tsmom.lookback_days == 126
        assert tsmom.winsor_pct == 0.01

    def test_score_shape(self, sample_features):
        """Test that TSMOM returns correct shape."""
        tsmom = TimeSeriesMomentum(lookback_days=126)
        alpha = tsmom.score(sample_features)

        assert isinstance(alpha, pd.DataFrame)
        assert "alpha" in alpha.columns
        assert len(alpha) == len(sample_features)

    def test_score_not_all_nan(self, sample_features):
        """Test that TSMOM produces some non-NaN values."""
        tsmom = TimeSeriesMomentum(lookback_days=126)
        alpha = tsmom.score(sample_features)

        n_valid = alpha["alpha"].notna().sum()
        assert n_valid > 0, "Should have some non-NaN alpha scores"

    def test_missing_feature_raises(self, sample_features):
        """Test that missing required feature raises error."""
        tsmom = TimeSeriesMomentum(lookback_days=999)  # Non-existent feature

        with pytest.raises(ValueError, match="Required column"):
            tsmom.score(sample_features)

    def test_cooldown_shifts_data(self, sample_features):
        """Test that cooldown parameter shifts returns."""
        tsmom_no_cooldown = TimeSeriesMomentum(lookback_days=126, cooldown_days=0)
        tsmom_with_cooldown = TimeSeriesMomentum(lookback_days=126, cooldown_days=5)

        alpha_no_cool = tsmom_no_cooldown.score(sample_features)
        alpha_cool = tsmom_with_cooldown.score(sample_features)

        # Should be different due to shift
        assert not alpha_no_cool["alpha"].equals(alpha_cool["alpha"])


class TestCrossSectionalMomentum:
    """Tests for XSMOM alpha model."""

    def test_initialization(self):
        """Test XSMOM initialization."""
        xsmom = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
        assert xsmom.lookback_days == 126
        assert xsmom.neutralization == "zscore"

    def test_score_shape(self, sample_features):
        """Test that XSMOM returns correct shape."""
        xsmom = CrossSectionalMomentum(lookback_days=126)
        alpha = xsmom.score(sample_features)

        assert isinstance(alpha, pd.DataFrame)
        assert "alpha" in alpha.columns
        assert len(alpha) == len(sample_features)

    def test_neutralization_zscore(self, sample_features):
        """Test that z-score neutralization centers values."""
        xsmom = CrossSectionalMomentum(lookback_days=126, neutralization="zscore")
        alpha = xsmom.score(sample_features)

        # Check that within each date, mean is close to 0
        date_means = alpha.groupby(level="date")["alpha"].mean()
        # Allow some tolerance due to NaN handling
        assert date_means.abs().mean() < 0.5

    def test_neutralization_rank(self, sample_features):
        """Test that rank neutralization produces [0, 1] range."""
        xsmom = CrossSectionalMomentum(lookback_days=126, neutralization="rank")
        alpha = xsmom.score(sample_features)

        # Check that values are in [0, 1] range (allowing for NaN)
        valid_alpha = alpha["alpha"].dropna()
        assert valid_alpha.min() >= 0
        assert valid_alpha.max() <= 1

    def test_invalid_neutralization_raises(self):
        """Test that invalid neutralization raises error."""
        with pytest.raises(ValueError, match="Invalid neutralization"):
            CrossSectionalMomentum(neutralization="invalid")


class TestMeanReversion:
    """Tests for mean reversion alpha model."""

    def test_initialization(self):
        """Test MR initialization."""
        mr = MeanReversion(lookback_days=20, entry_threshold=2.0)
        assert mr.lookback_days == 20
        assert mr.entry_threshold == 2.0

    def test_score_shape(self, sample_features):
        """Test that MR returns correct shape."""
        mr = MeanReversion(lookback_days=20)
        alpha = mr.score(sample_features)

        assert isinstance(alpha, pd.DataFrame)
        assert len(alpha) == len(sample_features)

    def test_score_inverted(self, sample_features):
        """Test that MR inverts z-scores (contrarian)."""
        mr = MeanReversion(lookback_days=20, entry_threshold=0)  # No threshold
        alpha = mr.score(sample_features)

        # Alpha should be negative of z-score
        if "zscore_20d" in sample_features.columns:
            zscore = sample_features["zscore_20d"]
            # Check that signs are opposite (where both are non-zero)
            valid = (alpha["alpha"].notna()) & (zscore.notna()) & (zscore != 0)
            signs_opposite = (alpha.loc[valid, "alpha"] * zscore[valid] < 0).sum()
            signs_same = (alpha.loc[valid, "alpha"] * zscore[valid] > 0).sum()

            assert signs_opposite > signs_same


class TestMLAlpha:
    """Tests for ML alpha model."""

    def test_initialization(self):
        """Test ML alpha initialization."""
        ml = MLAlpha(model_type="classification", estimator="gbm")
        assert ml.model_type == "classification"
        assert ml.estimator_type == "gbm"

    def test_score_shape(self, sample_features):
        """Test that ML alpha returns correct shape."""
        # Use small train window for test speed
        ml = MLAlpha(
            model_type="classification",
            estimator="logistic",
            train_window=50,
            forward_horizon=5,
        )
        alpha = ml.score(sample_features)

        assert isinstance(alpha, pd.DataFrame)
        assert "alpha" in alpha.columns
        assert len(alpha) == len(sample_features)

    def test_classification_output_range(self, sample_features):
        """Test that classification output is in reasonable range."""
        ml = MLAlpha(
            model_type="classification",
            estimator="logistic",
            train_window=50,
        )
        alpha = ml.score(sample_features)

        # Classification probabilities centered at 0 should be in [-0.5, 0.5]
        valid = alpha["alpha"].dropna()
        if len(valid) > 0:
            assert valid.min() >= -0.6  # Allow small tolerance
            assert valid.max() <= 0.6

    def test_feature_importance_available(self, sample_features):
        """Test that feature importance is extracted."""
        ml = MLAlpha(
            model_type="classification",
            estimator="gbm",
            train_window=50,
        )
        alpha = ml.score(sample_features)

        # For GBM, feature importance should be available
        importance = ml.get_feature_importance()
        assert importance is not None or len(alpha["alpha"].dropna()) == 0


class TestPairsTrading:
    """Tests for pairs trading alpha."""

    def test_initialization(self):
        """Test pairs trading initialization."""
        pairs = PairsTradingAlpha(formation_window=126, entry_threshold=2.0)
        assert pairs.formation_window == 126
        assert pairs.entry_threshold == 2.0

    def test_identify_pairs(self, sample_prices):
        """Test pair identification."""
        pairs = PairsTradingAlpha(formation_window=100, significance_level=0.1)

        # Get prices in wide format
        prices_wide = sample_prices["close"].unstack("symbol")

        identified = pairs.identify_pairs(prices_wide)

        # Should return a list
        assert isinstance(identified, list)
        # Each item should be a tuple of (sym1, sym2, hedge_ratio)
        for item in identified:
            assert len(item) == 3
            assert isinstance(item[0], str)
            assert isinstance(item[1], str)
            assert isinstance(item[2], float)

    def test_simple_pairs_score(self, sample_features, sample_prices):
        """Test simple pairs alpha."""
        simple = SimplePairsAlpha(min_correlation=0.5)
        alpha = simple.score(sample_features, sample_prices)

        assert isinstance(alpha, pd.DataFrame)
        assert "alpha" in alpha.columns
        assert len(alpha) == len(sample_features)


def test_all_alphas_return_dataframe(sample_features):
    """Test that all alpha models return DataFrame with 'alpha' column."""
    alphas = [
        TimeSeriesMomentum(lookback_days=126),
        CrossSectionalMomentum(lookback_days=126),
        MeanReversion(lookback_days=20),
        MLAlpha(model_type="classification", train_window=50),
    ]

    for alpha_model in alphas:
        result = alpha_model.score(sample_features)
        assert isinstance(result, pd.DataFrame), f"{type(alpha_model).__name__} should return DataFrame"
        assert "alpha" in result.columns, f"{type(alpha_model).__name__} should have 'alpha' column"
        assert len(result) == len(sample_features), f"{type(alpha_model).__name__} should preserve length"
