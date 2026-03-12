"""
tests/simulation/test_copula.py

Test suite for the Copula-driven cross-game correlation engine.
"""

import numpy as np
import pytest

from src.simulation.copula_engine import (
    build_conference_correlation_matrix,
    generate_copula_draws,
    apply_upset_contagion,
    compute_correlated_outcomes,
)


TEAMS = [
    {"name": "Kansas", "conference": "Big 12"},
    {"name": "Iowa State", "conference": "Big 12"},
    {"name": "Houston", "conference": "Big 12"},
    {"name": "Duke", "conference": "ACC"},
    {"name": "UNC", "conference": "ACC"},
    {"name": "Michigan", "conference": "Big Ten"},
]


class TestCorrelationMatrix:

    def test_shape(self):
        corr = build_conference_correlation_matrix(TEAMS)
        assert corr.shape == (6, 6)

    def test_diagonal_is_one(self):
        corr = build_conference_correlation_matrix(TEAMS)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-6)

    def test_symmetric(self):
        corr = build_conference_correlation_matrix(TEAMS)
        np.testing.assert_allclose(corr, corr.T, atol=1e-8)

    def test_positive_semidefinite(self):
        corr = build_conference_correlation_matrix(TEAMS)
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-8), "Correlation matrix is not PSD."

    def test_intra_conf_higher_than_inter(self):
        corr = build_conference_correlation_matrix(TEAMS)
        # Kansas ↔ Iowa State (same Big 12) should be > Kansas ↔ Duke (diff conf)
        assert corr[0, 1] > corr[0, 3]


class TestCopulaDraws:

    def test_draws_shape(self):
        corr = build_conference_correlation_matrix(TEAMS)
        draws = generate_copula_draws(corr, n_simulations=1000)
        assert draws.shape == (1000, 6)

    def test_draws_in_unit_interval(self):
        corr = build_conference_correlation_matrix(TEAMS)
        draws = generate_copula_draws(corr, n_simulations=5000)
        assert np.all(draws >= 0) and np.all(draws <= 1), "Draws must be in [0, 1]."

    def test_draws_mean_near_half(self):
        corr = build_conference_correlation_matrix(TEAMS)
        draws = generate_copula_draws(corr, n_simulations=50000)
        assert abs(draws.mean() - 0.5) < 0.02, f"Mean = {draws.mean()}, expected ~0.5"

    def test_same_conf_draws_are_correlated(self):
        """Teams in the same conference must have positively correlated draws."""
        corr = build_conference_correlation_matrix(TEAMS, base_intra_conf_corr=0.30)
        draws = generate_copula_draws(corr, n_simulations=50000)
        observed_corr = np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]
        assert observed_corr > 0.15, f"Same-conf correlation = {observed_corr}, expected >0.15"


class TestUpsetContagion:

    def test_contagion_reduces_draws(self):
        corr = build_conference_correlation_matrix(TEAMS)
        draws = generate_copula_draws(corr, n_simulations=1000)
        original_mean = draws[:, 1].mean()
        
        modified = apply_upset_contagion(draws, team_index=0, conference_indices=[1, 2])
        modified_mean = modified[:, 1].mean()
        
        # Iowa State's draws should be lower after Kansas upset contagion
        assert modified_mean <= original_mean + 0.001


class TestCorrelatedOutcomes:

    def test_outcomes_are_boolean(self):
        probs = np.array([0.8, 0.6, 0.5, 0.3])
        uniforms = np.array([0.5, 0.4, 0.6, 0.2])
        outcomes = compute_correlated_outcomes(probs, uniforms)
        assert outcomes.dtype == bool

    def test_certain_win(self):
        probs = np.array([1.0])
        uniforms = np.array([0.999])
        assert compute_correlated_outcomes(probs, uniforms)[0] == True

    def test_certain_loss(self):
        probs = np.array([0.0])
        uniforms = np.array([0.5])
        assert compute_correlated_outcomes(probs, uniforms)[0] == False
