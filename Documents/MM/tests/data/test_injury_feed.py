"""
tests/data/test_injury_feed.py

Tests for src/data/injury_feed.py
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

ROTOWIRE_COLUMNS = [
    "player",
    "team",
    "position",
    "status",
    "injury",
    "expected_return",
    "scraped_at",
]

FAKE_ROTOWIRE_HTML = """
<html><body>
<table class="injury-report">
  <tbody>
    <tr>
      <td class="player">John Smith</td>
      <td class="team">Duke</td>
      <td class="position">G</td>
      <td class="status">Out</td>
      <td class="injury">Ankle</td>
      <td class="expected_return">Day-to-Day</td>
    </tr>
    <tr>
      <td class="player">Mike Jones</td>
      <td class="team">Kentucky</td>
      <td class="position">F</td>
      <td class="status">Questionable</td>
      <td class="injury">Knee</td>
      <td class="expected_return">TBD</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

FAKE_INJURIES_DF = pd.DataFrame(
    [
        {
            "player": "John Smith",
            "team": "Duke",
            "position": "G",
            "status": "Out",
            "injury": "Ankle",
            "expected_return": "Day-to-Day",
            "scraped_at": "2024-03-15T00:00:00",
        }
    ]
)

FAKE_BPR_DF = pd.DataFrame(
    [
        {"player": "John Smith", "team": "Duke", "bpr": 8.2, "minutes_share": 0.35},
        {"player": "James Brown", "team": "Duke", "bpr": 6.1, "minutes_share": 0.28},
        {"player": "Kevin Lee", "team": "Duke", "bpr": 4.5, "minutes_share": 0.20},
    ]
)


# ---------------------------------------------------------------------------
# fetch_rotowire_injuries
# ---------------------------------------------------------------------------


class TestFetchRotowireInjuries:
    def test_returns_dataframe(self):
        """fetch_rotowire_injuries returns a pandas DataFrame."""
        from src.data.injury_feed import fetch_rotowire_injuries

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_ROTOWIRE_HTML
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp):
            df = fetch_rotowire_injuries(season=2024)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """DataFrame has all required columns."""
        from src.data.injury_feed import fetch_rotowire_injuries

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_ROTOWIRE_HTML
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp):
            df = fetch_rotowire_injuries(season=2024)
        for col in ROTOWIRE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_scraped_at_column_is_populated(self):
        """scraped_at column is non-null for all rows."""
        from src.data.injury_feed import fetch_rotowire_injuries

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_ROTOWIRE_HTML
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp):
            df = fetch_rotowire_injuries(season=2024)
        if len(df) > 0:
            assert df["scraped_at"].notna().all()

    def test_raises_runtime_error_on_non_200(self):
        """RuntimeError raised when HTTP response is not 200."""
        from src.data.injury_feed import fetch_rotowire_injuries

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="403"):
                fetch_rotowire_injuries(season=2024)

    def test_returns_empty_dataframe_on_no_injuries(self):
        """Returns a DataFrame with correct columns even when no injuries found."""
        from src.data.injury_feed import fetch_rotowire_injuries

        empty_html = "<html><body><p>No injuries reported.</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = empty_html
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp):
            df = fetch_rotowire_injuries(season=2024)
        assert isinstance(df, pd.DataFrame)
        for col in ROTOWIRE_COLUMNS:
            assert col in df.columns

    def test_sends_user_agent_header(self):
        """HTTP request includes a User-Agent header."""
        from src.data.injury_feed import fetch_rotowire_injuries

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = FAKE_ROTOWIRE_HTML
        with patch("src.data.injury_feed.httpx.get", return_value=mock_resp) as mock_get:
            fetch_rotowire_injuries(season=2024)
        call_kwargs = mock_get.call_args
        all_str = str(call_kwargs)
        assert "User-Agent" in all_str or "user-agent" in all_str.lower()


# ---------------------------------------------------------------------------
# build_availability_vector
# ---------------------------------------------------------------------------


class TestBuildAvailabilityVector:
    def test_returns_float(self):
        """build_availability_vector returns a float."""
        from src.data.injury_feed import build_availability_vector

        result = build_availability_vector("Duke", FAKE_INJURIES_DF, FAKE_BPR_DF)
        assert isinstance(result, float)

    def test_full_strength_when_no_injuries(self):
        """Returns 1.0 when no players from the team are injured."""
        from src.data.injury_feed import build_availability_vector

        empty_injuries = pd.DataFrame(columns=ROTOWIRE_COLUMNS)
        result = build_availability_vector("Duke", empty_injuries, FAKE_BPR_DF)
        assert result == pytest.approx(1.0)

    def test_value_between_zero_and_one(self):
        """Availability vector is always in [0.0, 1.0]."""
        from src.data.injury_feed import build_availability_vector

        result = build_availability_vector("Duke", FAKE_INJURIES_DF, FAKE_BPR_DF)
        assert 0.0 <= result <= 1.0

    def test_injured_star_reduces_availability(self):
        """Injuring the highest-minutes player reduces availability below 1.0."""
        from src.data.injury_feed import build_availability_vector

        result = build_availability_vector("Duke", FAKE_INJURIES_DF, FAKE_BPR_DF)
        assert result < 1.0

    def test_all_players_injured_gives_low_availability(self):
        """Injuring all tracked players gives low (near 0) availability."""
        from src.data.injury_feed import build_availability_vector

        all_injured = pd.DataFrame(
            [
                {
                    "player": "John Smith",
                    "team": "Duke",
                    "position": "G",
                    "status": "Out",
                    "injury": "Ankle",
                    "expected_return": "TBD",
                    "scraped_at": "2024-03-15T00:00:00",
                },
                {
                    "player": "James Brown",
                    "team": "Duke",
                    "position": "F",
                    "status": "Out",
                    "injury": "Knee",
                    "expected_return": "TBD",
                    "scraped_at": "2024-03-15T00:00:00",
                },
                {
                    "player": "Kevin Lee",
                    "team": "Duke",
                    "position": "C",
                    "status": "Out",
                    "injury": "Back",
                    "expected_return": "TBD",
                    "scraped_at": "2024-03-15T00:00:00",
                },
            ]
        )
        result = build_availability_vector("Duke", all_injured, FAKE_BPR_DF)
        assert result < 0.5

    def test_filters_by_team(self):
        """Only injuries for the specified team affect the availability vector."""
        from src.data.injury_feed import build_availability_vector

        other_team_injuries = pd.DataFrame(
            [
                {
                    "player": "Other Player",
                    "team": "Kentucky",
                    "position": "G",
                    "status": "Out",
                    "injury": "Foot",
                    "expected_return": "TBD",
                    "scraped_at": "2024-03-15T00:00:00",
                }
            ]
        )
        result = build_availability_vector("Duke", other_team_injuries, FAKE_BPR_DF)
        assert result == pytest.approx(1.0)

    def test_team_not_in_bpr_returns_one(self):
        """Returns 1.0 when BPR data has no entries for the team."""
        from src.data.injury_feed import build_availability_vector

        empty_bpr = pd.DataFrame(columns=["player", "team", "bpr", "minutes_share"])
        result = build_availability_vector("Duke", FAKE_INJURIES_DF, empty_bpr)
        assert result == pytest.approx(1.0)
