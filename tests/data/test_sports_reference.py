"""
tests/data/test_sports_reference.py

RED phase — tests written before implementation exists.
All sportsipy calls are mocked; no real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — build a fake sportsipy Team object
# ---------------------------------------------------------------------------

_STAT_KEYS = [
    "team_id",
    "wins",
    "losses",
    "pace",
    "offensive_rating",
    "defensive_rating",
    "effective_field_goal_percentage",
    "opp_effective_field_goal_percentage",
    "turnover_percentage",
    "opp_turnover_percentage",
    "offensive_rebound_percentage",
    "free_throw_attempt_rate",
]

_EXPECTED_RETURN_KEYS = [
    "team_id",
    "season",
    "wins",
    "losses",
    "pace",
    "ortg",
    "drtg",
    "efg_pct",
    "opp_efg_pct",
    "tov_pct",
    "opp_tov_pct",
    "orb_pct",
    "ft_rate",
]


def _fake_team(name: str = "duke", wins: int = 25, losses: int = 8) -> MagicMock:
    """Return a MagicMock that mimics a sportsipy Team object."""
    team = MagicMock()
    team.name = name
    team.team_id = name.replace(" ", "_").lower()
    team.wins = wins
    team.losses = losses
    team.pace = 68.5
    team.offensive_rating = 115.2
    team.defensive_rating = 98.7
    team.effective_field_goal_percentage = 0.542
    team.opp_effective_field_goal_percentage = 0.478
    team.turnover_percentage = 0.155
    team.opp_turnover_percentage = 0.178
    team.offensive_rebound_percentage = 0.312
    team.free_throw_attempt_rate = 0.288
    return team


def _fake_teams_iterable(n: int = 3) -> list[MagicMock]:
    names = ["duke", "kentucky", "kansas"][:n]
    return [_fake_team(name=n) for n in names]


# ---------------------------------------------------------------------------
# normalize_team_name
# ---------------------------------------------------------------------------

class TestNormalizeTeamName:
    """Tests for normalize_team_name()."""

    def test_lowercase(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("Duke") == "duke"

    def test_spaces_to_underscores(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("North Carolina") == "north_carolina"

    def test_strips_punctuation(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("St. John's") == "st_johns"

    def test_strips_leading_trailing_whitespace(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("  Duke  ") == "duke"

    def test_multiple_spaces_collapsed(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("Michigan  State") == "michigan_state"

    def test_already_normalized_is_idempotent(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("duke") == "duke"

    def test_empty_string(self) -> None:
        from src.data.sports_reference import normalize_team_name

        assert normalize_team_name("") == ""

    def test_mixed_punctuation(self) -> None:
        from src.data.sports_reference import normalize_team_name

        result = normalize_team_name("Connecticut (UConn)")
        assert result == "connecticut_uconn"


# ---------------------------------------------------------------------------
# Season validation (shared across fetch_team_stats + fetch_all_teams)
# ---------------------------------------------------------------------------

class TestSeasonValidation:
    """Season boundary checks raise ValueError."""

    @patch("src.data.sports_reference.Teams")
    def test_season_too_early_raises(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        with pytest.raises(ValueError, match="season"):
            fetch_team_stats("duke", 2002)

    @patch("src.data.sports_reference.Teams")
    def test_season_too_late_raises(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        with pytest.raises(ValueError, match="season"):
            fetch_team_stats("duke", 2027)

    @patch("src.data.sports_reference.Teams")
    def test_season_boundary_low_valid(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        # Season 2003 is the minimum valid season — should not raise
        result = fetch_team_stats("duke", 2003)
        assert isinstance(result, dict)

    @patch("src.data.sports_reference.Teams")
    def test_season_boundary_high_valid(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_team_stats("duke", 2026)
        assert isinstance(result, dict)

    @patch("src.data.sports_reference.Teams")
    def test_fetch_all_season_too_early_raises(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_all_teams

        with pytest.raises(ValueError, match="season"):
            fetch_all_teams(2002)


# ---------------------------------------------------------------------------
# fetch_team_stats
# ---------------------------------------------------------------------------

class TestFetchTeamStats:
    """Tests for fetch_team_stats()."""

    @patch("src.data.sports_reference.Teams")
    def test_returns_dict(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_team_stats("duke", 2024)
        assert isinstance(result, dict)

    @patch("src.data.sports_reference.Teams")
    def test_result_has_all_expected_keys(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_team_stats("duke", 2024)
        for key in _EXPECTED_RETURN_KEYS:
            assert key in result, f"Key '{key}' missing from result"

    @patch("src.data.sports_reference.Teams")
    def test_season_key_matches_argument(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_team_stats("duke", 2024)
        assert result["season"] == 2024

    @patch("src.data.sports_reference.Teams")
    def test_stats_values_match_mock(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        fake = _fake_team("duke", wins=30, losses=5)
        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([fake]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_team_stats("duke", 2024)
        assert result["wins"] == 30
        assert result["losses"] == 5
        assert result["pace"] == pytest.approx(68.5)
        assert result["ortg"] == pytest.approx(115.2)
        assert result["drtg"] == pytest.approx(98.7)

    @patch("src.data.sports_reference.Teams")
    def test_team_name_matching_is_case_insensitive(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        # Should find "duke" even when queried as "Duke"
        result = fetch_team_stats("Duke", 2024)
        assert result["team_id"] is not None

    @patch("src.data.sports_reference.Teams")
    def test_team_not_found_raises_value_error(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("kentucky")]))
        mock_teams_cls.return_value = mock_teams

        with pytest.raises(ValueError, match="not found"):
            fetch_team_stats("duke", 2024)

    @patch("src.data.sports_reference.Teams")
    def test_sportsipy_exception_wrapped_as_runtime_error(
        self, mock_teams_cls: MagicMock
    ) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams_cls.side_effect = Exception("sportsipy network failure")

        with pytest.raises(RuntimeError, match="sportsipy"):
            fetch_team_stats("duke", 2024)

    @patch("src.data.sports_reference.Teams")
    def test_teams_constructed_with_correct_season(
        self, mock_teams_cls: MagicMock
    ) -> None:
        from src.data.sports_reference import fetch_team_stats

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([_fake_team("duke")]))
        mock_teams_cls.return_value = mock_teams

        fetch_team_stats("duke", 2024)
        mock_teams_cls.assert_called_once_with(2024)


# ---------------------------------------------------------------------------
# fetch_all_teams
# ---------------------------------------------------------------------------

class TestFetchAllTeams:
    """Tests for fetch_all_teams()."""

    @patch("src.data.sports_reference.Teams")
    def test_returns_dataframe(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_all_teams

        fake_list = _fake_teams_iterable(3)
        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter(fake_list))
        mock_teams_cls.return_value = mock_teams

        result = fetch_all_teams(2024)
        assert isinstance(result, pd.DataFrame)

    @patch("src.data.sports_reference.Teams")
    def test_dataframe_has_expected_columns(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_all_teams

        fake_list = _fake_teams_iterable(3)
        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter(fake_list))
        mock_teams_cls.return_value = mock_teams

        result = fetch_all_teams(2024)
        for col in _EXPECTED_RETURN_KEYS:
            assert col in result.columns, f"Column '{col}' missing"

    @patch("src.data.sports_reference.Teams")
    def test_dataframe_row_count_matches_teams(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_all_teams

        fake_list = _fake_teams_iterable(3)
        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter(fake_list))
        mock_teams_cls.return_value = mock_teams

        result = fetch_all_teams(2024)
        assert len(result) == 3

    @patch("src.data.sports_reference.Teams")
    def test_all_rows_have_same_season(self, mock_teams_cls: MagicMock) -> None:
        from src.data.sports_reference import fetch_all_teams

        fake_list = _fake_teams_iterable(3)
        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter(fake_list))
        mock_teams_cls.return_value = mock_teams

        result = fetch_all_teams(2024)
        assert (result["season"] == 2024).all()

    @patch("src.data.sports_reference.Teams")
    def test_sportsipy_exception_wrapped_as_runtime_error(
        self, mock_teams_cls: MagicMock
    ) -> None:
        from src.data.sports_reference import fetch_all_teams

        mock_teams_cls.side_effect = Exception("connection refused")

        with pytest.raises(RuntimeError, match="sportsipy"):
            fetch_all_teams(2024)

    @patch("src.data.sports_reference.Teams")
    def test_empty_teams_returns_empty_dataframe(
        self, mock_teams_cls: MagicMock
    ) -> None:
        from src.data.sports_reference import fetch_all_teams

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([]))
        mock_teams_cls.return_value = mock_teams

        result = fetch_all_teams(2024)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.data.sports_reference.Teams")
    def test_teams_constructed_with_correct_season(
        self, mock_teams_cls: MagicMock
    ) -> None:
        from src.data.sports_reference import fetch_all_teams

        mock_teams = MagicMock()
        mock_teams.__iter__ = MagicMock(return_value=iter([]))
        mock_teams_cls.return_value = mock_teams

        fetch_all_teams(2024)
        mock_teams_cls.assert_called_once_with(2024)
