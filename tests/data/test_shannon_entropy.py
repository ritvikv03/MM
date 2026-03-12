"""
tests/data/test_shannon_entropy.py

Test suite for Shannon Entropy and Kill Shot feature engineering.
"""

import numpy as np
import pytest

from src.data.shannon_entropy import (
    compute_scoring_entropy,
    compute_normalized_entropy,
    compute_kill_shot_vulnerability,
    compute_kill_shot_markov_matrix,
    extract_entropy_features,
)


class TestScoringEntropy:

    def test_uniform_scoring_maximizes_entropy(self):
        """Perfectly uniform scoring should produce maximum entropy."""
        uniform = [10, 10, 10, 10, 10, 10, 10, 10]
        entropy = compute_scoring_entropy(uniform)
        assert abs(entropy - np.log2(8)) < 0.01, f"Expected max entropy ~3.0, got {entropy}"

    def test_bursty_scoring_has_lower_entropy(self):
        """Teams that score in bursts should have lower entropy."""
        bursty = [0, 30, 0, 0, 30, 0, 0, 20]
        uniform = [10, 10, 10, 10, 10, 10, 10, 10]
        assert compute_scoring_entropy(bursty) < compute_scoring_entropy(uniform)

    def test_single_period_scoring_is_zero_entropy(self):
        """All points in one period → entropy = 0."""
        single = [80, 0, 0, 0, 0, 0, 0, 0]
        assert compute_scoring_entropy(single) == 0.0

    def test_zero_total_returns_zero(self):
        """No points scored → entropy = 0."""
        assert compute_scoring_entropy([0, 0, 0, 0]) == 0.0

    def test_normalized_entropy_in_range(self):
        """Normalized entropy must be in [0, 1]."""
        for pts in [[10, 10, 10, 10, 10, 10, 10, 10], [80, 0, 0, 0, 0, 0, 0, 0], [5, 15, 3, 20, 8, 12, 7, 10]]:
            norm = compute_normalized_entropy(pts)
            assert 0.0 <= norm <= 1.0, f"Normalized entropy {norm} out of range"


class TestKillShot:

    def test_no_runs_returns_zero(self):
        assert compute_kill_shot_vulnerability([]) == 0.0

    def test_all_above_threshold(self):
        assert compute_kill_shot_vulnerability([10, 12, 8, 15], threshold=8) == 1.0

    def test_none_above_threshold(self):
        assert compute_kill_shot_vulnerability([1, 2, 3, 4, 5], threshold=8) == 0.0

    def test_partial_vulnerability(self):
        vuln = compute_kill_shot_vulnerability([2, 10, 5, 12, 3], threshold=8)
        assert abs(vuln - 0.4) < 0.01  # 2 out of 5


class TestMarkovMatrix:

    def test_matrix_is_row_stochastic(self):
        """Each row of the Markov matrix must sum to 1."""
        seq = ['H', 'H', 'H', 'A', 'A', 'H', 'D', 'A', 'A', 'A', 'H']
        M = compute_kill_shot_markov_matrix(seq)
        assert M.shape == (4, 4)
        for i in range(4):
            row_sum = M[i].sum()
            if row_sum > 0:  # Only check populated rows
                assert abs(row_sum - 1.0) < 1e-8, f"Row {i} sums to {row_sum}"

    def test_matrix_nonnegative(self):
        seq = ['H', 'A', 'H', 'A', 'H', 'H', 'H', 'A', 'D']
        M = compute_kill_shot_markov_matrix(seq)
        assert np.all(M >= 0)


class TestExtractFeatures:

    def test_returns_defaults_when_no_data(self):
        features = extract_entropy_features({})
        assert "scoring_entropy" in features
        assert "kill_shot_vulnerability" in features
        assert features["scoring_entropy_normalized"] == 0.85  # Default

    def test_returns_computed_when_data_present(self):
        features = extract_entropy_features({
            "points_by_period": [10, 10, 10, 10, 10, 10, 10, 10],
        })
        assert features["scoring_entropy_normalized"] > 0.99
