"""
tests/model/test_skellam.py

Test suite for the Zero-Truncated Skellam distribution implementation.
Validates mathematical properties that are critical for NCAA margin modeling.
"""

import numpy as np
import pytest

from src.model.skellam import (
    skellam_pmf,
    skellam_pmf_zero_truncated,
    skellam_log_likelihood,
    margin_to_poisson_rates,
)


class TestSkellamPMF:
    """Tests for the raw (non-truncated) Skellam PMF."""

    def test_pmf_is_nonnegative(self):
        """PMF must be non-negative for all integer margins."""
        margins = np.arange(-30, 31)
        pmf_values = skellam_pmf(margins, mu1=75.0, mu2=70.0)
        assert np.all(pmf_values >= 0), "Skellam PMF produced negative values."

    def test_pmf_sums_to_one(self):
        """PMF must sum to ~1.0 over a wide range of integers."""
        margins = np.arange(-100, 101)
        total = skellam_pmf(margins, mu1=75.0, mu2=70.0).sum()
        assert abs(total - 1.0) < 0.001, f"Skellam PMF sums to {total}, expected ~1.0"

    def test_pmf_symmetric_when_equal_rates(self):
        """When μ₁ = μ₂, the distribution should be symmetric around 0."""
        margins = np.arange(-20, 21)
        pmf = skellam_pmf(margins, mu1=70.0, mu2=70.0)
        # pmf[i] should equal pmf[len-1-i] (mirror)
        np.testing.assert_allclose(pmf, pmf[::-1], atol=1e-8)

    def test_mean_approximates_difference(self):
        """E[D] ≈ μ₁ - μ₂ for the Skellam distribution."""
        margins = np.arange(-100, 101)
        pmf = skellam_pmf(margins, mu1=75.0, mu2=70.0)
        expected_mean = np.sum(margins * pmf)
        assert abs(expected_mean - 5.0) < 0.5, f"Mean is {expected_mean}, expected ~5.0"


class TestZeroTruncatedSkellam:
    """Tests for the Zero-Truncated Skellam PMF."""

    def test_zero_margin_has_zero_probability(self):
        """P(D=0) MUST be exactly 0.0 — NCAA games cannot end in a tie."""
        p_zero = skellam_pmf_zero_truncated(0, mu1=75.0, mu2=70.0)
        assert p_zero == 0.0, f"ZT-Skellam P(0) = {p_zero}, must be exactly 0.0"

    def test_zt_pmf_sums_to_one(self):
        """Zero-Truncated PMF must sum to 1.0 over all non-zero margins."""
        margins = np.concatenate([np.arange(-100, 0), np.arange(1, 101)])
        total = skellam_pmf_zero_truncated(margins, mu1=75.0, mu2=70.0).sum()
        assert abs(total - 1.0) < 0.005, f"ZT-Skellam sums to {total}, expected ~1.0"

    def test_zt_pmf_nonnegative(self):
        """ZT-Skellam PMF is non-negative for all non-zero margins."""
        margins = np.concatenate([np.arange(-30, 0), np.arange(1, 31)])
        pmf = skellam_pmf_zero_truncated(margins, mu1=75.0, mu2=70.0)
        assert np.all(pmf >= 0), "ZT-Skellam PMF produced negative values."

    def test_zt_values_greater_than_raw_for_nonzero(self):
        """For k≠0, ZT-Skellam must assign higher probability than raw Skellam."""
        margins = np.array([-5, -1, 1, 5, 10])
        raw = skellam_pmf(margins, mu1=75.0, mu2=70.0)
        zt = skellam_pmf_zero_truncated(margins, mu1=75.0, mu2=70.0)
        assert np.all(zt > raw), "ZT-Skellam should be > raw Skellam for non-zero k."

    def test_fits_ncaa_variance_range(self):
        """NCAA games typically have margins between -30 and +30.
        The distribution should concentrate most mass in this range."""
        margins = np.concatenate([np.arange(-30, 0), np.arange(1, 31)])
        total = skellam_pmf_zero_truncated(margins, mu1=75.0, mu2=70.0).sum()
        assert total > 0.95, f"Only {total*100:.1f}% of mass in [-30,+30], expected >95%"


class TestLogLikelihood:
    """Tests for the Skellam log-likelihood function."""

    def test_log_likelihood_finite(self):
        """LL must not return NaN or Inf for typical NCAA margins."""
        margins = np.array([5, -3, 10, -7, 1, 15, -12])
        mu1 = np.full_like(margins, 75.0, dtype=float)
        mu2 = np.full_like(margins, 70.0, dtype=float)
        ll = skellam_log_likelihood(margins, mu1, mu2, zero_truncated=True)
        assert np.isfinite(ll), f"Log-likelihood is not finite: {ll}"

    def test_ll_higher_for_correct_prediction_direction(self):
        """A model predicting positive margin should have higher LL when margin is positive."""
        margins = np.array([5])
        # Model A: home team stronger (mu1 > mu2) — correct direction
        ll_correct = skellam_log_likelihood(margins, np.array([78.0]), np.array([70.0]))
        # Model B: away team stronger (mu1 < mu2) — wrong direction
        ll_wrong = skellam_log_likelihood(margins, np.array([70.0]), np.array([78.0]))
        assert ll_correct > ll_wrong, "Model predicting correct direction should have higher LL."


class TestMarginToRates:
    """Tests for the delta-to-Poisson-rates converter."""

    def test_positive_delta_gives_mu1_greater(self):
        mu1, mu2 = margin_to_poisson_rates(5.0)
        assert mu1 > mu2, "Positive delta should give mu1 > mu2."

    def test_zero_delta_gives_equal_rates(self):
        mu1, mu2 = margin_to_poisson_rates(0.0)
        assert abs(mu1 - mu2) < 1e-8, "Zero delta should give equal rates."

    def test_rates_clamped_to_bounds(self):
        mu1, mu2 = margin_to_poisson_rates(300.0)  # Extreme
        assert 1.0 <= mu1 <= 150.0
        assert 1.0 <= mu2 <= 150.0
