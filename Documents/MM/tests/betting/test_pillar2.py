"""
tests/betting/test_pillar2.py

Combined test suite for Pillar 2 financial engineering modules:
- Options Pricing (Black-Scholes Vega)
- Prospect Theory (Prelec weighting / CLV)
- RL Bracket Optimization
"""

import numpy as np
import pytest

from src.betting.options_pricing import (
    compute_path_volatility,
    compute_vega,
    recommend_hedge_action,
)
from src.betting.prospect_theory import (
    prelec_weighting,
    compute_bias_magnitude,
    scan_for_clv_opportunities,
    identify_peak_irrationality_windows,
)
from src.simulation.rl_bracket import (
    BracketPoolEnvironment,
    GreedyLeverageAgent,
    run_rl_bracket_optimization,
)


class TestOptionsPricing:

    def test_path_volatility_nonnegative(self):
        vol = compute_path_volatility([0.97, 0.85, 0.72, 0.62])
        assert vol >= 0.0

    def test_chalk_path_low_volatility(self):
        """A smooth downward path should have low volatility."""
        vol = compute_path_volatility([0.95, 0.90, 0.85, 0.80, 0.75, 0.70])
        assert vol < 0.1

    def test_volatile_path_higher_vol(self):
        """A path with wild swings should have higher volatility."""
        wild = compute_path_volatility([0.90, 0.30, 0.80, 0.20, 0.70, 0.10])
        smooth = compute_path_volatility([0.90, 0.85, 0.80, 0.75, 0.70, 0.65])
        assert wild > smooth

    def test_vega_nonnegative(self):
        vega = compute_vega(0.54, 0.05, rounds_remaining=4)
        assert vega >= 0.0

    def test_vega_zero_when_no_volatility(self):
        vega = compute_vega(0.54, 0.0, rounds_remaining=4)
        assert vega == 0.0

    def test_hedge_recommendation_keys(self):
        rec = recommend_hedge_action(0.2, 0.7, 25000)
        assert "action" in rec
        assert "reason" in rec
        assert "urgency" in rec


class TestProspectTheory:

    def test_prelec_boundary_zero(self):
        assert prelec_weighting(0.0) == 0.0

    def test_prelec_boundary_one(self):
        assert prelec_weighting(1.0) == 1.0

    def test_prelec_overweights_small_probs(self):
        """Small probabilities should be overweighted (w(p) > p)."""
        assert prelec_weighting(0.05) > 0.05
        assert prelec_weighting(0.10) > 0.10

    def test_prelec_underweights_large_probs(self):
        """Large probabilities should be underweighted (w(p) < p)."""
        assert prelec_weighting(0.90) < 0.90
        assert prelec_weighting(0.95) < 0.95

    def test_bias_magnitude_returns_correct_keys(self):
        result = compute_bias_magnitude(0.60, 0.55)
        assert "raw_clv" in result
        assert "perceived_prob" in result
        assert "distortion" in result
        assert "bias_direction" in result

    def test_clv_scanner_filters_by_edge(self):
        matchups = [
            {"team": "A", "true_prob": 0.80, "market_prob": 0.75, "moneyline": -300},
            {"team": "B", "true_prob": 0.50, "market_prob": 0.50, "moneyline": 100},
        ]
        opps = scan_for_clv_opportunities(matchups, min_edge=0.03)
        assert len(opps) == 1  # Only Team A has edge > 3%
        assert opps[0]["team"] == "A"

    def test_peak_irrationality_with_no_data(self):
        result = identify_peak_irrationality_windows([])
        assert result["peak_round"] == "R64"
        assert result["peak_seed_matchup"] == "5-vs-12"


class TestRLBracket:

    @pytest.fixture
    def mock_matchups(self):
        np.random.seed(42)
        matchups = {f"game_{i}": np.clip(np.random.normal(0.6, 0.1), 0.05, 0.95) for i in range(10)}
        public = {f"game_{i}": np.clip(np.random.normal(0.5, 0.1), 0.05, 0.95) for i in range(10)}
        return matchups, public

    def test_environment_creates(self, mock_matchups):
        true_probs, public_pcts = mock_matchups
        env = BracketPoolEnvironment(true_probs, public_pcts, n_opponents=50)
        assert env.n_games == 10

    def test_scoring_returns_integer(self, mock_matchups):
        true_probs, public_pcts = mock_matchups
        env = BracketPoolEnvironment(true_probs, public_pcts, n_opponents=10)
        bracket = {k: 1 for k in true_probs}  # All TeamA
        score = env.score_bracket(bracket)
        assert isinstance(score, (int, np.integer))

    def test_rank_in_valid_range(self, mock_matchups):
        true_probs, public_pcts = mock_matchups
        env = BracketPoolEnvironment(true_probs, public_pcts, n_opponents=50)
        agent = GreedyLeverageAgent(true_probs, public_pcts)
        bracket = agent.select_bracket()
        rank, reward = env.step(bracket)
        assert 1 <= rank <= 51  # Could be 1st or last

    def test_rl_optimization_runs(self, mock_matchups):
        true_probs, public_pcts = mock_matchups
        results = run_rl_bracket_optimization(
            true_probs, public_pcts, n_episodes=5, n_opponents=20
        )
        assert results["n_episodes"] == 5
        assert results["mean_rank"] > 0
