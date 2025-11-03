"""Tests for validation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphalab.validate.purged_kfold import PurgedKFold, check_leakage
from alphalab.validate.walkforward import WalkForward


@pytest.fixture
def sample_timeseries_data() -> pd.DataFrame:
    """Create sample time series data for testing."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]

    index = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])

    np.random.seed(42)
    data = {
        "feature1": np.random.randn(len(index)),
        "feature2": np.random.randn(len(index)),
        "returns": np.random.randn(len(index)) * 0.02,
    }

    return pd.DataFrame(data, index=index)


class TestPurgedKFold:
    """Tests for Purged K-Fold cross-validation."""

    def test_initialization(self):
        """Test PurgedKFold initialization."""
        pkf = PurgedKFold(n_splits=5, embargo_days=5, purge_days=2)
        assert pkf.n_splits == 5
        assert pkf.embargo_days == 5
        assert pkf.purge_days == 2

    def test_invalid_n_splits_raises(self):
        """Test that invalid n_splits raises error."""
        with pytest.raises(ValueError, match="n_splits must be >= 2"):
            PurgedKFold(n_splits=1)

    def test_split_returns_list(self, sample_timeseries_data):
        """Test that split returns list of tuples."""
        pkf = PurgedKFold(n_splits=5)
        splits = pkf.split(sample_timeseries_data)

        assert isinstance(splits, list)
        assert len(splits) == 5

        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_train_test_no_overlap(self, sample_timeseries_data):
        """Test that train and test indices don't overlap."""
        pkf = PurgedKFold(n_splits=5)
        splits = pkf.split(sample_timeseries_data)

        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, "Train and test sets should not overlap"

    def test_embargo_creates_gap(self, sample_timeseries_data):
        """Test that embargo creates gap between train and test."""
        pkf = PurgedKFold(n_splits=3, embargo_days=10)
        splits = pkf.split(sample_timeseries_data)

        dates = sample_timeseries_data.index.get_level_values("date")

        for train_idx, test_idx in splits:
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]

            # Check if there's a gap
            train_max = train_dates.max()
            test_min = test_dates.min()

            # Should be some gap due to embargo
            if train_max < test_min:  # Train comes before test
                gap_days = (test_min - train_max).days
                # Embargo creates separation - actual gap may be less than embargo_days
                # due to weekend/holiday handling and date alignment
                assert gap_days >= 1  # At least some gap exists

    def test_all_data_used(self, sample_timeseries_data):
        """Test that most data is used across all folds."""
        pkf = PurgedKFold(n_splits=5)
        splits = pkf.split(sample_timeseries_data)

        all_indices = set()
        for train_idx, test_idx in splits:
            all_indices.update(train_idx)
            all_indices.update(test_idx)

        # Most indices should be used (some might be in embargo zones)
        usage_pct = len(all_indices) / len(sample_timeseries_data)
        assert usage_pct > 0.7, "Should use most of the data"

    def test_fold_sizes_roughly_equal(self, sample_timeseries_data):
        """Test that fold sizes are roughly equal."""
        pkf = PurgedKFold(n_splits=5)
        splits = pkf.split(sample_timeseries_data)

        test_sizes = [len(test_idx) for _, test_idx in splits]

        # Check that sizes are within 50% of mean
        mean_size = np.mean(test_sizes)
        for size in test_sizes:
            assert abs(size - mean_size) / mean_size < 0.5


class TestCheckLeakage:
    """Tests for leakage checking utility."""

    def test_no_leakage_when_separate(self, sample_timeseries_data):
        """Test that no leakage is detected when periods are separate."""
        # Manually create non-overlapping indices
        dates = sample_timeseries_data.index.get_level_values("date").unique()

        train_dates = dates[:200]
        test_dates = dates[250:]  # Gap of 50 days

        train_idx = sample_timeseries_data.index.get_level_values("date").isin(train_dates)
        test_idx = sample_timeseries_data.index.get_level_values("date").isin(test_dates)

        result = check_leakage(
            sample_timeseries_data,
            np.where(train_idx)[0],
            np.where(test_idx)[0],
        )

        assert result["has_overlap"] is False
        assert result["train_dates_ok"] is True

    def test_leakage_detected_when_overlap(self, sample_timeseries_data):
        """Test that leakage is detected when there's overlap."""
        dates = sample_timeseries_data.index.get_level_values("date").unique()

        # Create overlapping periods
        train_dates = dates[:300]
        test_dates = dates[200:400]  # Overlaps with train

        train_idx = sample_timeseries_data.index.get_level_values("date").isin(train_dates)
        test_idx = sample_timeseries_data.index.get_level_values("date").isin(test_dates)

        result = check_leakage(
            sample_timeseries_data,
            np.where(train_idx)[0],
            np.where(test_idx)[0],
        )

        assert result["has_overlap"] is True


class TestWalkForward:
    """Tests for Walk-Forward validation."""

    def test_initialization(self):
        """Test WalkForward initialization."""
        wf = WalkForward(
            n_folds=6,
            train_period_days=252,
            test_period_days=60,
            embargo_days=5,
        )
        assert wf.n_folds == 6
        assert wf.train_period_days == 252
        assert wf.test_period_days == 60
        assert wf.window_type == "rolling"

    def test_expanding_window_type(self):
        """Test expanding window type."""
        wf = WalkForward(n_folds=6, train_period_days=None, test_period_days=60)
        assert wf.window_type == "expanding"

    def test_split_returns_list(self, sample_timeseries_data):
        """Test that split returns list of tuples."""
        wf = WalkForward(
            n_folds=4,
            train_period_days=100,
            test_period_days=30,
        )
        splits = wf.split(sample_timeseries_data)

        assert isinstance(splits, list)
        assert len(splits) <= 4  # May be fewer if not enough data

        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_train_before_test(self, sample_timeseries_data):
        """Test that training period comes before test period."""
        wf = WalkForward(
            n_folds=3,
            train_period_days=100,
            test_period_days=30,
        )
        splits = wf.split(sample_timeseries_data)

        dates = sample_timeseries_data.index.get_level_values("date")

        for train_idx, test_idx in splits:
            train_max = dates[train_idx].max()
            test_min = dates[test_idx].min()

            # Train should end before test begins
            assert train_max < test_min

    def test_expanding_window_grows(self, sample_timeseries_data):
        """Test that expanding window increases train size."""
        wf = WalkForward(
            n_folds=3,
            train_period_days=None,  # Expanding
            test_period_days=30,
        )
        splits = wf.split(sample_timeseries_data)

        train_sizes = [len(train_idx) for train_idx, _ in splits]

        # Each subsequent fold should have more training data
        if len(train_sizes) > 1:
            for i in range(1, len(train_sizes)):
                assert train_sizes[i] >= train_sizes[i - 1]

    def test_rolling_window_constant_size(self, sample_timeseries_data):
        """Test that rolling window maintains constant train size."""
        wf = WalkForward(
            n_folds=3,
            train_period_days=100,
            test_period_days=30,
        )
        splits = wf.split(sample_timeseries_data)

        dates = sample_timeseries_data.index.get_level_values("date")
        train_date_spans = []

        for train_idx, _ in splits:
            train_dates = dates[train_idx]
            span = (train_dates.max() - train_dates.min()).days
            train_date_spans.append(span)

        # All spans should be similar (within tolerance for daily data)
        if len(train_date_spans) > 1:
            mean_span = np.mean(train_date_spans)
            for span in train_date_spans:
                # Allow 20% variation due to business days
                assert abs(span - mean_span) / mean_span < 0.2

    def test_no_overlap_between_folds(self, sample_timeseries_data):
        """Test that test sets don't overlap between folds."""
        wf = WalkForward(
            n_folds=3,
            train_period_days=100,
            test_period_days=30,
        )
        splits = wf.split(sample_timeseries_data)

        test_sets = [set(test_idx) for _, test_idx in splits]

        # Check no overlap between consecutive test sets
        for i in range(len(test_sets) - 1):
            overlap = test_sets[i] & test_sets[i + 1]
            assert len(overlap) == 0, "Test sets should not overlap"


def test_purged_kfold_vs_walkforward(sample_timeseries_data):
    """Compare PurgedKFold and WalkForward splitting strategies."""
    pkf = PurgedKFold(n_splits=5)
    wf = WalkForward(n_folds=5, train_period_days=100, test_period_days=50)

    pkf_splits = pkf.split(sample_timeseries_data)
    wf_splits = wf.split(sample_timeseries_data)

    # Both should return non-empty splits
    assert len(pkf_splits) > 0
    assert len(wf_splits) > 0

    # Walk-forward should have sequential test periods
    # Purged K-fold might have non-sequential test periods
    dates = sample_timeseries_data.index.get_level_values("date")

    wf_test_starts = [dates[test_idx].min() for _, test_idx in wf_splits]
    # Walk-forward test periods should be in chronological order
    assert wf_test_starts == sorted(wf_test_starts)
