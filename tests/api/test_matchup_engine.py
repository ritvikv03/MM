"""
tests/api/test_matchup_engine.py
TDD tests for src/api/matchup_engine.py — 11 required tests.
"""
from __future__ import annotations

import pytest
import pandas as pd

from src.api.schemas import MatchupResponse

# ---------------------------------------------------------------------------
# Mock T-Rank DataFrame
# ---------------------------------------------------------------------------

MOCK_TRANK = pd.DataFrame({
    "team":       ["Duke", "Kansas"],
    "adj_oe":     [118.0,  115.2],
    "adj_de":     [92.0,   94.1],
    "tempo":      [70.0,   71.5],
    "luck":       [0.02,  -0.01],
    "seed":       [1,       1],
    "conference": ["ACC",  "Big 12"],
})


class MockLoader:
    """Loader that returns MOCK_TRANK without hitting the network."""

    def __init__(self, df: pd.DataFrame = MOCK_TRANK) -> None:
        self._df = df

    def get_trank(self, season: int) -> pd.DataFrame:
        return self._df.copy()


class EmptyLoader:
    """Loader that always returns an empty DataFrame (simulates fetch failure)."""

    def get_trank(self, season: int) -> pd.DataFrame:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def result() -> MatchupResponse:
    from src.api.matchup_engine import build_real_matchup
    return build_real_matchup(
        home_team="Duke",
        away_team="Kansas",
        season=2024,
        neutral_site=False,
        loader=MockLoader(),
    )


# ---------------------------------------------------------------------------
# Test 1 — returns MatchupResponse
# ---------------------------------------------------------------------------

def test_returns_matchup_response(result):
    assert isinstance(result, MatchupResponse)


# ---------------------------------------------------------------------------
# Test 2 — home_team == "Duke"
# ---------------------------------------------------------------------------

def test_home_team_is_duke(result):
    assert result.home_team == "Duke"


# ---------------------------------------------------------------------------
# Test 3 — away_team == "Kansas"
# ---------------------------------------------------------------------------

def test_away_team_is_kansas(result):
    assert result.away_team == "Kansas"


# ---------------------------------------------------------------------------
# Test 4 — p_win_home between 0 and 1
# ---------------------------------------------------------------------------

def test_p_win_home_in_unit_interval(result):
    assert 0.0 <= result.p_win_home <= 1.0


# ---------------------------------------------------------------------------
# Test 5 — spread_samples length == 2000
# ---------------------------------------------------------------------------

def test_spread_samples_length(result):
    assert len(result.spread_samples) == 2000


# ---------------------------------------------------------------------------
# Test 6 — p_win_samples length == 2000
# ---------------------------------------------------------------------------

def test_p_win_samples_length(result):
    assert len(result.p_win_samples) == 2000


# ---------------------------------------------------------------------------
# Test 7 — Duke (higher adj_em) has p_win_home > 0.5 on neutral site
# ---------------------------------------------------------------------------

def test_duke_wins_neutral_site():
    from src.api.matchup_engine import build_real_matchup
    res = build_real_matchup(
        home_team="Duke",
        away_team="Kansas",
        season=2024,
        neutral_site=True,
        loader=MockLoader(),
    )
    # Duke: EM = 118 - 92 = 26, Kansas: EM = 115.2 - 94.1 = 21.1
    # Duke has higher EM so should win > 50% on neutral court
    assert res.p_win_home > 0.5


# ---------------------------------------------------------------------------
# Test 8 — MatchupNotFoundError raised for unknown team
# ---------------------------------------------------------------------------

def test_unknown_team_raises():
    from src.api.matchup_engine import build_real_matchup, MatchupNotFoundError
    with pytest.raises(MatchupNotFoundError):
        build_real_matchup(
            home_team="NonExistentTeam",
            away_team="Kansas",
            season=2024,
            neutral_site=False,
            loader=MockLoader(),
        )


# ---------------------------------------------------------------------------
# Test 9 — neutral site gives lower p_win than home-court (Duke hosting)
# ---------------------------------------------------------------------------

def test_home_court_boosts_p_win():
    from src.api.matchup_engine import build_real_matchup
    home_result = build_real_matchup(
        home_team="Duke",
        away_team="Kansas",
        season=2024,
        neutral_site=False,
        loader=MockLoader(),
    )
    neutral_result = build_real_matchup(
        home_team="Duke",
        away_team="Kansas",
        season=2024,
        neutral_site=True,
        loader=MockLoader(),
    )
    assert home_result.p_win_home > neutral_result.p_win_home


# ---------------------------------------------------------------------------
# Test 10 — spread_mean is a reasonable float (between -20 and 20)
# ---------------------------------------------------------------------------

def test_spread_mean_reasonable(result):
    assert -20.0 <= result.spread_mean <= 20.0


# ---------------------------------------------------------------------------
# Test 11 — data_source == "real"
# ---------------------------------------------------------------------------

def test_data_source_is_real(result):
    assert result.data_source == "real"


# ---------------------------------------------------------------------------
# Bonus: empty loader raises MatchupNotFoundError
# ---------------------------------------------------------------------------

def test_empty_loader_raises():
    from src.api.matchup_engine import build_real_matchup, MatchupNotFoundError
    with pytest.raises(MatchupNotFoundError):
        build_real_matchup(
            home_team="Duke",
            away_team="Kansas",
            season=2024,
            neutral_site=False,
            loader=EmptyLoader(),
        )
