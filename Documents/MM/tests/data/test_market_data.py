"""
tests/data/test_market_data.py

Tests for src/data/market_data.py
"""

import json
import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures / shared constants
# ---------------------------------------------------------------------------

SBR_DF_COLUMNS = [
    "game_id",
    "home_team",
    "away_team",
    "open_spread",
    "close_spread",
    "open_total",
    "close_total",
    "date",
]

FAKE_SBR_HTML_WITH_TABLE = """
<html><body>
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th>Home Team</th>
      <th>Away Team</th>
      <th>Open Spread</th>
      <th>Close Spread</th>
      <th>Open Total</th>
      <th>Close Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2024-03-20</td>
      <td>Duke</td>
      <td>Kentucky</td>
      <td>-4.5</td>
      <td>-5.0</td>
      <td>141.5</td>
      <td>142.0</td>
    </tr>
    <tr>
      <td>2024-03-21</td>
      <td>Kansas</td>
      <td>Gonzaga</td>
      <td>-2.0</td>
      <td>-2.5</td>
      <td>138.0</td>
      <td>137.5</td>
    </tr>
  </tbody>
</table>
</body></html>
"""

FAKE_SBR_HTML_NO_TABLE = "<html><body><p>No data available.</p></body></html>"


def _make_mock_response(status_code=200, text=FAKE_SBR_HTML_WITH_TABLE):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        mock_resp.raise_for_status.side_effect = requests.HTTPError(
            response=mock_resp
        )
    return mock_resp


# ---------------------------------------------------------------------------
# fetch_sbr_lines
# ---------------------------------------------------------------------------


class TestFetchSbrLines:
    def test_returns_dataframe(self):
        """fetch_sbr_lines returns a pandas DataFrame."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response()
        with patch("src.data.market_data.requests.get", return_value=mock_resp):
            df = fetch_sbr_lines(year=2024)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        """DataFrame has all required columns."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response()
        with patch("src.data.market_data.requests.get", return_value=mock_resp):
            df = fetch_sbr_lines(year=2024)
        for col in SBR_DF_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_returns_empty_dataframe_when_no_table(self):
        """Returns empty DataFrame with correct columns when no table in HTML."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response(text=FAKE_SBR_HTML_NO_TABLE)
        with patch("src.data.market_data.requests.get", return_value=mock_resp):
            df = fetch_sbr_lines(year=2024)
        assert isinstance(df, pd.DataFrame)
        for col in SBR_DF_COLUMNS:
            assert col in df.columns

    def test_raises_runtime_error_after_retries(self):
        """RuntimeError raised after 3 failed network attempts."""
        import requests as req
        from src.data.market_data import fetch_sbr_lines

        with patch(
            "src.data.market_data.requests.get",
            side_effect=req.ConnectionError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Failed to fetch SBR lines"):
                fetch_sbr_lines(year=2024)

    def test_retries_on_failure(self):
        """requests.get is called up to 3 times on repeated failure."""
        import requests as req
        from src.data.market_data import fetch_sbr_lines

        with patch(
            "src.data.market_data.requests.get",
            side_effect=req.ConnectionError("timeout"),
        ) as mock_get, patch("src.data.market_data.time.sleep"):
            with pytest.raises(RuntimeError):
                fetch_sbr_lines(year=2024)
        assert mock_get.call_count == 3

    def test_cache_miss_then_cache_hit(self):
        """Second call reads from cache without making HTTP request."""
        from src.data.market_data import fetch_sbr_lines

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resp = _make_mock_response()
            with patch(
                "src.data.market_data.requests.get", return_value=mock_resp
            ) as mock_get:
                df1 = fetch_sbr_lines(year=2024, cache_dir=tmpdir)

            # Second call — should read from cache, no HTTP call
            with patch(
                "src.data.market_data.requests.get", return_value=mock_resp
            ) as mock_get2:
                df2 = fetch_sbr_lines(year=2024, cache_dir=tmpdir)

            assert mock_get2.call_count == 0
            assert list(df2.columns) == list(df1.columns)

    def test_cache_file_is_written(self):
        """Cache file is created after a successful fetch."""
        from src.data.market_data import fetch_sbr_lines
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_resp = _make_mock_response()
            with patch("src.data.market_data.requests.get", return_value=mock_resp):
                fetch_sbr_lines(year=2024, cache_dir=tmpdir)

            cache_path = Path(tmpdir) / "market_data" / "sbr_2024.json"
            assert cache_path.exists()

    def test_game_id_column_populated(self):
        """game_id column is non-null for all rows."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response()
        with patch("src.data.market_data.requests.get", return_value=mock_resp):
            df = fetch_sbr_lines(year=2024)
        if len(df) > 0:
            assert df["game_id"].notna().all()

    def test_year_passed_in_request(self):
        """Year parameter is passed as part of the HTTP request."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response()
        with patch(
            "src.data.market_data.requests.get", return_value=mock_resp
        ) as mock_get:
            fetch_sbr_lines(year=2025)
        all_str = str(mock_get.call_args)
        assert "2025" in all_str

    def test_sends_user_agent_header(self):
        """HTTP request includes a User-Agent header."""
        from src.data.market_data import fetch_sbr_lines

        mock_resp = _make_mock_response()
        with patch(
            "src.data.market_data.requests.get", return_value=mock_resp
        ) as mock_get:
            fetch_sbr_lines(year=2024)
        all_str = str(mock_get.call_args)
        assert "User-Agent" in all_str or "user-agent" in all_str.lower()


# ---------------------------------------------------------------------------
# compute_clv
# ---------------------------------------------------------------------------


class TestComputeClv:
    def test_returns_float(self):
        """compute_clv returns a float."""
        from src.data.market_data import compute_clv

        result = compute_clv(open_line=-3.5, close_line=-5.5, bet_side="home")
        assert isinstance(result, float)

    def test_positive_clv_when_line_moves_in_favor_home(self):
        """CLV is positive when line moves in bettor's favor (home -3.5 → -5.5 means home side got worse, CLV negative for home bet)."""
        from src.data.market_data import compute_clv

        # Bet home at -3.5; line moves to -5.5 (home now -5.5 = worse price for home bettor)
        # Closing line for home is -5.5, which is a worse price → negative CLV
        result = compute_clv(open_line=-3.5, close_line=-5.5, bet_side="home")
        assert result < 0.0

    def test_positive_clv_when_line_moves_favorably(self):
        """Bet home at -5.5 open; close is -3.5 → home bettors got the best of it (+CLV)."""
        from src.data.market_data import compute_clv

        result = compute_clv(open_line=-5.5, close_line=-3.5, bet_side="home")
        assert result > 0.0

    def test_zero_clv_when_lines_equal(self):
        """CLV is 0.0 when open and close lines are equal."""
        from src.data.market_data import compute_clv

        result = compute_clv(open_line=-3.5, close_line=-3.5, bet_side="home")
        assert result == pytest.approx(0.0)

    def test_away_side_clv_is_inverse_of_home(self):
        """Away CLV is the negative of home CLV for the same lines."""
        from src.data.market_data import compute_clv

        home_clv = compute_clv(open_line=-3.5, close_line=-5.5, bet_side="home")
        away_clv = compute_clv(open_line=-3.5, close_line=-5.5, bet_side="away")
        assert away_clv == pytest.approx(-home_clv)

    def test_raises_value_error_on_invalid_bet_side(self):
        """ValueError raised when bet_side is not 'home' or 'away'."""
        from src.data.market_data import compute_clv

        with pytest.raises(ValueError, match="bet_side"):
            compute_clv(open_line=-3.5, close_line=-5.5, bet_side="push")


# ---------------------------------------------------------------------------
# american_to_prob
# ---------------------------------------------------------------------------


class TestAmericanToProb:
    def test_returns_float(self):
        from src.data.market_data import american_to_prob

        result = american_to_prob(-110)
        assert isinstance(result, float)

    def test_negative_110_is_approx_52_pct(self):
        """-110 American → ~52.38% implied probability."""
        from src.data.market_data import american_to_prob

        result = american_to_prob(-110)
        assert result == pytest.approx(110 / 210, rel=1e-4)

    def test_plus_100_is_50_pct(self):
        """+100 American → 50% implied probability."""
        from src.data.market_data import american_to_prob

        result = american_to_prob(100)
        assert result == pytest.approx(0.5, rel=1e-4)

    def test_minus_200_is_approx_66_7_pct(self):
        """-200 American → ~66.67% implied probability."""
        from src.data.market_data import american_to_prob

        result = american_to_prob(-200)
        assert result == pytest.approx(200 / 300, rel=1e-4)

    def test_plus_150_is_approx_40_pct(self):
        """+150 American → ~40% implied probability."""
        from src.data.market_data import american_to_prob

        result = american_to_prob(150)
        assert result == pytest.approx(100 / 250, rel=1e-4)

    def test_result_is_between_0_and_1(self):
        """Implied probability is always between 0 and 1."""
        from src.data.market_data import american_to_prob

        for odds in [-300, -150, -110, 100, 110, 150, 300]:
            result = american_to_prob(odds)
            assert 0.0 < result < 1.0, f"Out of range for odds={odds}: {result}"

    def test_raises_value_error_on_zero(self):
        """ValueError raised when odds == 0 (invalid American odds)."""
        from src.data.market_data import american_to_prob

        with pytest.raises(ValueError):
            american_to_prob(0)


# ---------------------------------------------------------------------------
# remove_vig
# ---------------------------------------------------------------------------


class TestRemoveVig:
    def test_returns_tuple_of_two_floats(self):
        """remove_vig returns a tuple of two floats."""
        from src.data.market_data import remove_vig

        result = remove_vig(0.5238, 0.5238)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_probabilities_sum_to_one(self):
        """After vig removal, home_prob + away_prob == 1.0."""
        from src.data.market_data import remove_vig

        home_p, away_p = remove_vig(0.5238, 0.5238)
        assert home_p + away_p == pytest.approx(1.0)

    def test_equal_probs_remain_equal(self):
        """Equal overround probabilities remain equal after vig removal."""
        from src.data.market_data import remove_vig

        home_p, away_p = remove_vig(0.5238, 0.5238)
        assert home_p == pytest.approx(away_p)

    def test_correct_vig_removal_example(self):
        """-110 / -110 line: both sides ~52.38% raw → 50% each after vig removal."""
        from src.data.market_data import remove_vig

        raw = 110 / 210  # ~0.52381
        home_p, away_p = remove_vig(raw, raw)
        assert home_p == pytest.approx(0.5, abs=1e-4)
        assert away_p == pytest.approx(0.5, abs=1e-4)

    def test_favourite_stays_favourite_after_vig_removal(self):
        """The favourite's probability remains higher after vig removal."""
        from src.data.market_data import remove_vig

        # -200 favourite (~66.67%) vs +160 dog (~38.46%)
        from src.data.market_data import american_to_prob

        fav_raw = american_to_prob(-200)
        dog_raw = american_to_prob(160)
        fav_clean, dog_clean = remove_vig(fav_raw, dog_raw)
        assert fav_clean > dog_clean

    def test_raises_value_error_on_zero_total_prob(self):
        """ValueError raised when both probabilities are zero."""
        from src.data.market_data import remove_vig

        with pytest.raises((ValueError, ZeroDivisionError)):
            remove_vig(0.0, 0.0)
