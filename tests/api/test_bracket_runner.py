"""
tests/api/test_bracket_runner.py
TDD tests for src/api/bracket_runner.py — RED → GREEN → REFACTOR
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from src.api.schemas import SimulateResponse

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

MOCK_TRANK_16 = pd.DataFrame({
    "team":       [f"Team{i}" for i in range(16)],
    "adj_oe":     np.linspace(120, 100, 16),
    "adj_de":     np.linspace(88, 100, 16),
    "tempo":      [70.0] * 16,
    "luck":       [0.0] * 16,
    "seed":       list(range(1, 17)),
    "conference": ["ACC"] * 8 + ["SEC"] * 8,
})

TEAMS_16 = [f"Team{i}" for i in range(16)]


class MockLoader:
    """Loader that returns MOCK_TRANK_16 for any season."""

    def get_trank(self, season: int) -> pd.DataFrame:
        return MOCK_TRANK_16.copy()


# ---------------------------------------------------------------------------
# Helper — run once and cache
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim_result():
    from src.api.bracket_runner import build_real_simulation
    return build_real_simulation(TEAMS_16, 50, 2024, loader=MockLoader())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_simulate_response(sim_result):
    """Test 1: build_real_simulation returns a SimulateResponse."""
    assert isinstance(sim_result, SimulateResponse)


def test_n_simulations_matches(sim_result):
    """Test 2: n_simulations field equals the passed value (50)."""
    assert sim_result.n_simulations == 50


def test_advancements_length_matches_teams(sim_result):
    """Test 3: len(advancements) == 16, matching the teams list."""
    assert len(sim_result.advancements) == 16


def test_advancement_probs_has_six_rounds(sim_result):
    """Test 4: Each TeamAdvancementItem.advancement_probs has 6 entries."""
    for item in sim_result.advancements:
        assert len(item.advancement_probs) == 6, (
            f"{item.team} has {len(item.advancement_probs)} rounds, expected 6"
        )


def test_advancement_probs_generally_decrease(sim_result):
    """Test 5: advancement_probs[R64] >= advancement_probs[Championship] - 0.05."""
    for item in sim_result.advancements:
        probs = list(item.advancement_probs.values())
        first = probs[0]
        last  = probs[-1]
        assert first >= last - 0.05, (
            f"{item.team}: R64 prob {first:.4f} < Championship {last:.4f} - 0.05"
        )


def test_top_seed_higher_champ_prob_than_bottom(sim_result):
    """Test 6: Team0 (highest adj_em) has higher championship prob than Team15."""
    by_team = {item.team: item for item in sim_result.advancements}
    champ_key = "Championship"
    top_champ  = by_team["Team0"].advancement_probs[champ_key]
    bot_champ  = by_team["Team15"].advancement_probs[champ_key]
    assert top_champ > bot_champ, (
        f"Team0 champ={top_champ:.4f} should exceed Team15 champ={bot_champ:.4f}"
    )


def test_entropy_non_negative_and_finite(sim_result):
    """Test 7: entropy >= 0.0 and finite for all teams."""
    for item in sim_result.advancements:
        assert item.entropy >= 0.0, f"{item.team} entropy {item.entropy} < 0"
        assert math.isfinite(item.entropy), f"{item.team} entropy not finite"


def test_missing_team_gets_median_em_no_crash():
    """Test 8: Teams not in trank get median EM — no crash, valid response."""
    from src.api.bracket_runner import build_real_simulation

    # Only provide data for Team0–Team7; Team8–Team15 are missing
    partial_df = MOCK_TRANK_16.head(8).copy()

    class PartialLoader:
        def get_trank(self, season: int) -> pd.DataFrame:
            return partial_df

    result = build_real_simulation(TEAMS_16, 50, 2024, loader=PartialLoader())
    assert isinstance(result, SimulateResponse)
    assert len(result.advancements) == 16


def test_data_source_is_real(sim_result):
    """Test 9: data_source field equals 'real'."""
    assert sim_result.data_source == "real"
