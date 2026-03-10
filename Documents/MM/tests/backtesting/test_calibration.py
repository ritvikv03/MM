"""
tests/backtesting/test_calibration.py

TDD tests for src/backtesting/calibration.py.
Written RED-first — all tests should fail before implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.backtesting.calibration import (
    reliability_diagram_data,
    ece_time_series,
    sharpness,
    brier_skill_score,
    compute_overround,
)


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------

class TestReliabilityDiagramData:
    """Tests for reliability_diagram_data."""

    def test_returns_dict_with_required_keys(self):
        """Return value must contain all required keys."""
        y_pred = np.linspace(0.1, 0.9, 20)
        y_true = (y_pred > 0.5).astype(float)
        result = reliability_diagram_data(y_pred, y_true)
        required_keys = {"bin_centers", "mean_predicted", "fraction_positive", "bin_counts", "ece"}
        assert required_keys.issubset(result.keys())

    def test_bin_centers_shape(self):
        """bin_centers should have shape (n_bins,)."""
        y_pred = np.linspace(0.0, 1.0, 50)
        y_true = np.ones(50)
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        assert result["bin_centers"].shape == (10,)

    def test_bin_centers_values(self):
        """bin_centers should be midpoints of [0,1] bins."""
        result = reliability_diagram_data(np.array([0.5]), np.array([1.0]), n_bins=10)
        # With n_bins=10, bin width=0.1, centers should be 0.05, 0.15, ..., 0.95
        expected = np.arange(10) * 0.1 + 0.05
        np.testing.assert_allclose(result["bin_centers"], expected, atol=1e-9)

    def test_mean_predicted_shape(self):
        """mean_predicted should have shape (n_bins,)."""
        y_pred = np.linspace(0.0, 1.0, 100)
        y_true = np.random.default_rng(42).integers(0, 2, 100).astype(float)
        result = reliability_diagram_data(y_pred, y_true, n_bins=5)
        assert result["mean_predicted"].shape == (5,)

    def test_fraction_positive_shape(self):
        """fraction_positive should have shape (n_bins,)."""
        y_pred = np.linspace(0.0, 1.0, 100)
        y_true = np.zeros(100)
        result = reliability_diagram_data(y_pred, y_true, n_bins=8)
        assert result["fraction_positive"].shape == (8,)

    def test_empty_bins_are_nan(self):
        """Bins with no samples should have nan for mean_predicted and fraction_positive."""
        # All predictions in [0.9, 1.0] -> bins 0..8 are empty
        y_pred = np.array([0.95, 0.96, 0.97])
        y_true = np.array([1.0, 1.0, 1.0])
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        # bins 0-8 (indices 0..8) should be nan (only last bin has samples)
        assert np.isnan(result["mean_predicted"][0])
        assert np.isnan(result["fraction_positive"][0])

    def test_ece_is_float(self):
        """ece should be a Python float."""
        y_pred = np.linspace(0.1, 0.9, 20)
        y_true = (y_pred > 0.5).astype(float)
        result = reliability_diagram_data(y_pred, y_true)
        assert isinstance(result["ece"], float)

    def test_ece_matches_compute_ece_logic(self):
        """ECE from reliability_diagram_data should match wandb_logger.compute_ece."""
        from src.utils.wandb_logger import compute_ece
        rng = np.random.default_rng(0)
        y_pred = rng.uniform(0, 1, 100)
        y_true = rng.integers(0, 2, 100).astype(float)
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        expected_ece = compute_ece(y_pred, y_true, n_bins=10)
        assert abs(result["ece"] - expected_ece) < 1e-9

    def test_bin_counts_sum_equals_n(self):
        """bin_counts should sum to total number of samples."""
        y_pred = np.linspace(0.05, 0.95, 50)
        y_true = np.ones(50)
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        assert result["bin_counts"].sum() == 50

    def test_n_bins_respected(self):
        """Custom n_bins should be reflected in output array shapes."""
        y_pred = np.linspace(0.0, 1.0, 30)
        y_true = np.zeros(30)
        for n in [5, 10, 20]:
            result = reliability_diagram_data(y_pred, y_true, n_bins=n)
            assert result["bin_centers"].shape == (n,)
            assert result["mean_predicted"].shape == (n,)
            assert result["fraction_positive"].shape == (n,)
            assert result["bin_counts"].shape == (n,)

    def test_perfect_calibration_ece_near_zero(self):
        """For perfectly calibrated predictions, ECE should be near 0."""
        # Uniform predictions with matching outcomes
        n = 1000
        rng = np.random.default_rng(42)
        y_pred = rng.uniform(0, 1, n)
        # y_true = Bernoulli(y_pred) — perfectly calibrated in expectation
        y_true = (rng.uniform(0, 1, n) < y_pred).astype(float)
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        # ECE should be reasonably small (not necessarily 0 due to noise)
        assert result["ece"] < 0.1

    def test_value_exactly_one_in_last_bin(self):
        """y_pred=1.0 must fall in the last bin (right-edge inclusive)."""
        y_pred = np.array([1.0])
        y_true = np.array([1.0])
        result = reliability_diagram_data(y_pred, y_true, n_bins=10)
        # last bin should have count 1
        assert result["bin_counts"][-1] == 1


# ---------------------------------------------------------------------------
# ece_time_series
# ---------------------------------------------------------------------------

class TestEceTimeSeries:
    """Tests for ece_time_series."""

    def test_returns_numpy_array(self):
        """Should return a numpy ndarray."""
        preds = [np.array([0.6, 0.4]), np.array([0.7, 0.3])]
        labels = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        result = ece_time_series(preds, labels)
        assert isinstance(result, np.ndarray)

    def test_correct_length(self):
        """Length of output should equal number of time steps."""
        preds = [np.linspace(0.1, 0.9, 10) for _ in range(5)]
        labels = [np.ones(10) for _ in range(5)]
        result = ece_time_series(preds, labels)
        assert len(result) == 5

    def test_each_element_is_float(self):
        """Each element should be a float (ECE value)."""
        preds = [np.array([0.5, 0.5, 0.5]), np.array([0.3, 0.7, 0.5])]
        labels = [np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])]
        result = ece_time_series(preds, labels)
        for val in result:
            assert isinstance(float(val), float)

    def test_raises_on_mismatched_list_lengths(self):
        """Raises ValueError if predictions_list and labels_list lengths differ."""
        preds = [np.array([0.5, 0.5])]
        labels = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        with pytest.raises(ValueError):
            ece_time_series(preds, labels)

    def test_single_time_step(self):
        """Should work with a single time step."""
        preds = [np.array([0.8, 0.2, 0.9, 0.1])]
        labels = [np.array([1.0, 0.0, 1.0, 0.0])]
        result = ece_time_series(preds, labels)
        assert len(result) == 1
        assert 0.0 <= float(result[0]) <= 1.0

    def test_ece_values_in_valid_range(self):
        """All ECE values should be in [0, 1]."""
        rng = np.random.default_rng(99)
        preds = [rng.uniform(0, 1, 20) for _ in range(8)]
        labels = [rng.integers(0, 2, 20).astype(float) for _ in range(8)]
        result = ece_time_series(preds, labels)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_empty_lists_raises(self):
        """Empty lists should raise ValueError."""
        with pytest.raises(ValueError):
            ece_time_series([], [np.array([1.0])])


# ---------------------------------------------------------------------------
# sharpness
# ---------------------------------------------------------------------------

class TestSharpness:
    """Tests for sharpness."""

    def test_empty_array_returns_zero(self):
        """Empty array should return 0.0."""
        assert sharpness(np.array([])) == 0.0

    def test_all_half(self):
        """All predictions = 0.5 -> mean of 0.5*(1-0.5) = 0.25."""
        y_pred = np.full(100, 0.5)
        result = sharpness(y_pred)
        assert abs(result - 0.25) < 1e-9

    def test_all_zeros(self):
        """All predictions = 0 -> mean of 0*(1-0) = 0.0."""
        y_pred = np.zeros(50)
        result = sharpness(y_pred)
        assert result == 0.0

    def test_all_ones(self):
        """All predictions = 1.0 -> mean of 1*(1-1) = 0.0."""
        y_pred = np.ones(50)
        result = sharpness(y_pred)
        assert result == 0.0

    def test_mixed_predictions(self):
        """Sharpness of [0, 1] should be mean of [0*(1-0), 1*(1-1)] = 0.0."""
        y_pred = np.array([0.0, 1.0])
        result = sharpness(y_pred)
        assert result == 0.0

    def test_known_value(self):
        """[0.25, 0.75] -> mean of [0.25*0.75, 0.75*0.25] = 0.1875."""
        y_pred = np.array([0.25, 0.75])
        result = sharpness(y_pred)
        assert abs(result - 0.1875) < 1e-9

    def test_lower_means_sharper(self):
        """More decisive predictions should yield lower sharpness."""
        decisive = np.array([0.1, 0.9, 0.05, 0.95])
        uncertain = np.array([0.4, 0.6, 0.45, 0.55])
        assert sharpness(decisive) < sharpness(uncertain)

    def test_returns_float(self):
        """sharpness should return a Python float."""
        result = sharpness(np.array([0.3, 0.7]))
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# brier_skill_score
# ---------------------------------------------------------------------------

class TestBrierSkillScore:
    """Tests for brier_skill_score."""

    def test_perfect_model(self):
        """Perfect predictions -> BSS = 1.0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        bss = brier_skill_score(y_pred, y_true)
        assert abs(bss - 1.0) < 1e-6

    def test_climatology_model(self):
        """Predicting base rate for all games -> BSS = 0.0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        climatology = y_true.mean()
        y_pred = np.full(4, climatology)
        bss = brier_skill_score(y_pred, y_true)
        assert abs(bss) < 1e-9

    def test_all_same_label_returns_zero(self):
        """If all labels are the same (climatology BS=0), return 0.0."""
        y_true = np.ones(10)
        y_pred = np.full(10, 0.5)
        bss = brier_skill_score(y_pred, y_true)
        assert bss == 0.0

    def test_worse_than_climatology(self):
        """Predictions worse than climatology -> BSS < 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        # Predicting the wrong class
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        bss = brier_skill_score(y_pred, y_true)
        assert bss < 0.0

    def test_known_value(self):
        """Compute known BSS: y_true=[1,0,1,0], y_pred=[0.7, 0.3, 0.7, 0.3]."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.7, 0.3, 0.7, 0.3])
        # BS_model = mean((0.7-1)^2, (0.3-0)^2, ...) = mean(0.09, 0.09, 0.09, 0.09) = 0.09
        # climatology = 0.5; BS_climatology = 0.5*(1-0.5) = 0.25
        # BSS = 1 - 0.09/0.25 = 1 - 0.36 = 0.64
        bss = brier_skill_score(y_pred, y_true)
        assert abs(bss - 0.64) < 1e-6

    def test_returns_float(self):
        """brier_skill_score should return a float."""
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([0.6, 0.4])
        assert isinstance(brier_skill_score(y_pred, y_true), float)


# ---------------------------------------------------------------------------
# compute_overround
# ---------------------------------------------------------------------------

class TestComputeOverround:
    """Tests for compute_overround."""

    def test_fair_market(self):
        """Two equal probabilities summing to 1.0 should return 1.0."""
        probs = np.array([0.5, 0.5])
        assert abs(compute_overround(probs) - 1.0) < 1e-9

    def test_vig_market(self):
        """Typical bookmaker market sums to > 1.0."""
        probs = np.array([0.526, 0.526])  # ~5% vig
        result = compute_overround(probs)
        assert result > 1.0

    def test_known_value(self):
        """Sum of given probs should equal their sum."""
        probs = np.array([0.55, 0.50])
        result = compute_overround(probs)
        assert abs(result - 1.05) < 1e-9

    def test_single_prob(self):
        """Single probability is returned as-is."""
        probs = np.array([0.7])
        assert abs(compute_overround(probs) - 0.7) < 1e-9

    def test_multiple_outcomes(self):
        """Works with more than two outcomes."""
        probs = np.array([0.4, 0.35, 0.30])  # sum > 1
        result = compute_overround(probs)
        assert abs(result - 1.05) < 1e-9

    def test_returns_float(self):
        """compute_overround should return a Python float."""
        assert isinstance(compute_overround(np.array([0.5, 0.6])), float)
