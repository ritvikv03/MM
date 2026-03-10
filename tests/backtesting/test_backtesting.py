"""
tests/backtesting/test_backtesting.py

TDD tests for src/backtesting/backtesting.py.
Written RED-first — all tests should fail before implementation.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.backtesting.backtesting import (
    WalkForwardSplit,
    BacktestResult,
    run_backtest,
    aggregate_backtest,
    compute_log_loss_np,
)


# ---------------------------------------------------------------------------
# WalkForwardSplit
# ---------------------------------------------------------------------------

class TestWalkForwardSplit:
    """Tests for WalkForwardSplit generator."""

    def _make_records(self, n: int):
        """Helper: create n dummy game records."""
        return [{"y_true": i % 2, "clv": 0.0} for i in range(n)]

    def test_yields_correct_number_of_splits(self):
        """With n=50 games, test_size=5, n_splits=3 -> 3 splits yielded."""
        records = self._make_records(50)
        splitter = WalkForwardSplit(n_splits=3, test_size=5)
        splits = list(splitter.split(records))
        assert len(splits) == 3

    def test_yields_tuples(self):
        """Each yielded element should be a tuple of (train_indices, test_indices)."""
        records = self._make_records(30)
        splitter = WalkForwardSplit(n_splits=2, test_size=5)
        for train_idx, test_idx in splitter.split(records):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_train_indices_always_less_than_test_indices(self):
        """PIT integrity: max train index < min test index for every split."""
        records = self._make_records(40)
        splitter = WalkForwardSplit(n_splits=3, test_size=5)
        for train_idx, test_idx in splitter.split(records):
            assert train_idx.max() < test_idx.min()

    def test_test_size_correct(self):
        """Each test set should have exactly test_size elements."""
        records = self._make_records(50)
        splitter = WalkForwardSplit(n_splits=4, test_size=6)
        for _, test_idx in splitter.split(records):
            assert len(test_idx) == 6

    def test_gap_respected(self):
        """With gap=2, there should be 2 games between train end and test start."""
        records = self._make_records(50)
        splitter = WalkForwardSplit(n_splits=3, test_size=5, gap=2)
        for train_idx, test_idx in splitter.split(records):
            # gap games between last train index and first test index
            assert test_idx.min() - train_idx.max() == 3  # gap + 1 means 3 positions apart (indices gap+1 apart)

    def test_gap_zero_by_default(self):
        """With no gap, test starts immediately after training ends."""
        records = self._make_records(30)
        splitter = WalkForwardSplit(n_splits=2, test_size=5)
        splits = list(splitter.split(records))
        train_idx, test_idx = splits[0]
        # test should start right after train ends
        assert test_idx.min() == train_idx.max() + 1

    def test_successive_splits_grow_training_set(self):
        """Later splits should have larger (or equal) training sets."""
        records = self._make_records(60)
        splitter = WalkForwardSplit(n_splits=4, test_size=5)
        train_sizes = [len(t) for t, _ in splitter.split(records)]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_raises_value_error_too_small(self):
        """Raises ValueError if data is too small for even one split."""
        records = self._make_records(5)
        splitter = WalkForwardSplit(n_splits=1, test_size=10)
        with pytest.raises(ValueError):
            list(splitter.split(records))

    def test_raises_value_error_zero_records(self):
        """Raises ValueError on empty record list."""
        splitter = WalkForwardSplit(n_splits=1, test_size=5)
        with pytest.raises(ValueError):
            list(splitter.split([]))

    def test_no_index_overlap(self):
        """Train and test indices must not overlap."""
        records = self._make_records(50)
        splitter = WalkForwardSplit(n_splits=3, test_size=5)
        for train_idx, test_idx in splitter.split(records):
            overlap = set(train_idx.tolist()) & set(test_idx.tolist())
            assert len(overlap) == 0

    def test_single_split(self):
        """n_splits=1 should produce exactly one split."""
        records = self._make_records(20)
        splitter = WalkForwardSplit(n_splits=1, test_size=5)
        splits = list(splitter.split(records))
        assert len(splits) == 1

    def test_all_indices_in_range(self):
        """All returned indices must be valid indices into the records list."""
        records = self._make_records(40)
        n = len(records)
        splitter = WalkForwardSplit(n_splits=3, test_size=5)
        for train_idx, test_idx in splitter.split(records):
            assert all(0 <= i < n for i in train_idx)
            assert all(0 <= i < n for i in test_idx)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_can_instantiate(self):
        """BacktestResult can be created with all required fields."""
        result = BacktestResult(
            split_idx=0,
            train_size=100,
            test_size=20,
            brier_score=0.2,
            log_loss=0.5,
            calibration_ece=0.05,
            clv_delta=0.01,
        )
        assert result.split_idx == 0
        assert result.train_size == 100
        assert result.test_size == 20
        assert isinstance(result.brier_score, float)
        assert isinstance(result.log_loss, float)
        assert isinstance(result.calibration_ece, float)
        assert isinstance(result.clv_delta, float)

    def test_fields_accessible(self):
        """All fields are accessible as attributes."""
        result = BacktestResult(
            split_idx=2, train_size=50, test_size=10,
            brier_score=0.15, log_loss=0.4, calibration_ece=0.03, clv_delta=0.0
        )
        assert result.split_idx == 2
        assert result.train_size == 50
        assert result.test_size == 10
        assert result.brier_score == 0.15
        assert result.log_loss == 0.4
        assert result.calibration_ece == 0.03
        assert result.clv_delta == 0.0


# ---------------------------------------------------------------------------
# compute_log_loss_np
# ---------------------------------------------------------------------------

class TestComputeLogLossNp:
    """Tests for compute_log_loss_np."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield loss near 0."""
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        y_true = np.array([1, 0, 1, 0])
        loss = compute_log_loss_np(y_pred, y_true)
        # With eps clamping, perfect predictions ≈ -log(1-eps) ≈ very small
        assert loss < 0.01

    def test_random_predictions(self):
        """Known value: all 0.5 predictions on balanced data."""
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        y_true = np.array([1, 0, 1, 0])
        loss = compute_log_loss_np(y_pred, y_true)
        expected = -math.log(0.5)  # ≈ 0.693
        assert abs(loss - expected) < 1e-6

    def test_eps_clamping_prevents_inf(self):
        """y_pred=0 with y_true=1 and y_pred=1 with y_true=0 must not return inf."""
        y_pred = np.array([0.0, 1.0])
        y_true = np.array([1, 0])
        loss = compute_log_loss_np(y_pred, y_true)
        assert math.isfinite(loss)
        assert loss > 0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        y_pred = np.array([0.5, 0.5, 0.5])
        y_true = np.array([1, 0])
        with pytest.raises(ValueError):
            compute_log_loss_np(y_pred, y_true)

    def test_all_correct(self):
        """All correct predictions (after clamping) should give small loss."""
        y_pred = np.array([0.9, 0.1])
        y_true = np.array([1, 0])
        loss = compute_log_loss_np(y_pred, y_true)
        assert loss < 0.15

    def test_single_element(self):
        """Single element arrays should work correctly."""
        loss = compute_log_loss_np(np.array([0.7]), np.array([1]))
        expected = -math.log(0.7)
        assert abs(loss - expected) < 1e-6

    def test_custom_eps(self):
        """Custom eps should be respected for clamping."""
        y_pred = np.array([0.0])
        y_true = np.array([1])
        eps = 0.1
        loss = compute_log_loss_np(y_pred, y_true, eps=eps)
        expected = -math.log(eps)
        assert abs(loss - expected) < 1e-6


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

class TestRunBacktest:
    """Tests for run_backtest."""

    def _make_records(self, n: int, with_clv: bool = True):
        """Create n game records with alternating outcomes."""
        records = []
        for i in range(n):
            r = {"y_true": i % 2}
            if with_clv:
                r["clv"] = 0.05
            records.append(r)
        return records

    def _constant_predict_fn(self, train_records, test_records):
        """Always predict 0.6 for every game."""
        return np.full(len(test_records), 0.6)

    def test_returns_correct_number_of_results(self):
        """run_backtest should return one BacktestResult per split."""
        records = self._make_records(50)
        results = run_backtest(
            predict_fn=self._constant_predict_fn,
            game_records=records,
            n_splits=3,
            test_size=5,
        )
        assert len(results) == 3

    def test_results_are_backtest_result_instances(self):
        """Each element in the returned list should be a BacktestResult."""
        records = self._make_records(50)
        results = run_backtest(
            predict_fn=self._constant_predict_fn,
            game_records=records,
            n_splits=2,
            test_size=5,
        )
        for r in results:
            assert isinstance(r, BacktestResult)

    def test_predict_fn_called_once_per_split(self):
        """predict_fn should be called exactly n_splits times."""
        records = self._make_records(50)
        mock_fn = MagicMock(side_effect=lambda tr, te: np.full(len(te), 0.5))
        run_backtest(mock_fn, records, n_splits=4, test_size=5)
        assert mock_fn.call_count == 4

    def test_split_idx_is_sequential(self):
        """split_idx in results should be 0, 1, 2, ..."""
        records = self._make_records(50)
        results = run_backtest(self._constant_predict_fn, records, n_splits=3, test_size=5)
        for i, r in enumerate(results):
            assert r.split_idx == i

    def test_brier_score_is_float(self):
        """brier_score field should be a Python float."""
        records = self._make_records(30)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert isinstance(r.brier_score, float)

    def test_log_loss_is_float(self):
        """log_loss field should be a Python float."""
        records = self._make_records(30)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert isinstance(r.log_loss, float)

    def test_calibration_ece_is_float(self):
        """calibration_ece field should be a Python float."""
        records = self._make_records(30)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert isinstance(r.calibration_ece, float)

    def test_missing_clv_defaults_to_zero(self):
        """If game records lack 'clv' key, clv_delta should be 0.0."""
        records = self._make_records(30, with_clv=False)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert r.clv_delta == 0.0

    def test_train_size_and_test_size_correct(self):
        """train_size and test_size in results should match actual split sizes."""
        records = self._make_records(50)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert r.test_size == 5
            assert r.train_size > 0

    def test_brier_score_in_valid_range(self):
        """Brier score must be in [0, 1]."""
        records = self._make_records(50)
        results = run_backtest(self._constant_predict_fn, records, n_splits=3, test_size=5)
        for r in results:
            assert 0.0 <= r.brier_score <= 1.0

    def test_clv_delta_with_clv_present(self):
        """With all clv=0.05, clv_delta should be ~0.05 (mean of clv values)."""
        records = self._make_records(50)
        results = run_backtest(self._constant_predict_fn, records, n_splits=2, test_size=5)
        for r in results:
            assert abs(r.clv_delta - 0.05) < 1e-9

    def test_gap_parameter_respected(self):
        """run_backtest with gap > 0 should still return n_splits results."""
        records = self._make_records(60)
        results = run_backtest(
            self._constant_predict_fn, records, n_splits=2, test_size=5, gap=2
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# aggregate_backtest
# ---------------------------------------------------------------------------

class TestAggregateBacktest:
    """Tests for aggregate_backtest."""

    def _make_results(self, n: int) -> list:
        """Create n BacktestResult objects with known values."""
        return [
            BacktestResult(
                split_idx=i,
                train_size=100 + i * 10,
                test_size=10,
                brier_score=0.2 + i * 0.01,
                log_loss=0.5 + i * 0.01,
                calibration_ece=0.05 + i * 0.005,
                clv_delta=0.01 * i,
            )
            for i in range(n)
        ]

    def test_returns_dict(self):
        """aggregate_backtest should return a dict."""
        results = self._make_results(3)
        agg = aggregate_backtest(results)
        assert isinstance(agg, dict)

    def test_expected_keys_present(self):
        """All required keys should be present in the returned dict."""
        results = self._make_results(3)
        agg = aggregate_backtest(results)
        expected_keys = {
            "mean_brier", "std_brier",
            "mean_log_loss", "std_log_loss",
            "mean_ece", "std_ece",
            "mean_clv_delta", "n_splits",
        }
        assert expected_keys.issubset(agg.keys())

    def test_n_splits_correct(self):
        """n_splits should equal the number of results."""
        results = self._make_results(4)
        agg = aggregate_backtest(results)
        assert agg["n_splits"] == 4

    def test_mean_brier_computed_correctly(self):
        """mean_brier should be arithmetic mean of all brier_scores."""
        results = self._make_results(3)
        expected = np.mean([r.brier_score for r in results])
        agg = aggregate_backtest(results)
        assert abs(agg["mean_brier"] - expected) < 1e-9

    def test_std_brier_computed_correctly(self):
        """std_brier should be std (population) of all brier_scores."""
        results = self._make_results(4)
        expected = np.std([r.brier_score for r in results])
        agg = aggregate_backtest(results)
        assert abs(agg["std_brier"] - expected) < 1e-9

    def test_single_result_std_is_zero(self):
        """With a single result, std values should be 0."""
        results = self._make_results(1)
        agg = aggregate_backtest(results)
        assert agg["std_brier"] == 0.0
        assert agg["std_log_loss"] == 0.0
        assert agg["std_ece"] == 0.0

    def test_mean_log_loss_correct(self):
        """mean_log_loss should be arithmetic mean of all log_loss values."""
        results = self._make_results(5)
        expected = np.mean([r.log_loss for r in results])
        agg = aggregate_backtest(results)
        assert abs(agg["mean_log_loss"] - expected) < 1e-9

    def test_mean_ece_correct(self):
        """mean_ece should be arithmetic mean of calibration_ece values."""
        results = self._make_results(3)
        expected = np.mean([r.calibration_ece for r in results])
        agg = aggregate_backtest(results)
        assert abs(agg["mean_ece"] - expected) < 1e-9

    def test_mean_clv_delta_correct(self):
        """mean_clv_delta should be mean of clv_delta values."""
        results = self._make_results(4)
        expected = np.mean([r.clv_delta for r in results])
        agg = aggregate_backtest(results)
        assert abs(agg["mean_clv_delta"] - expected) < 1e-9
