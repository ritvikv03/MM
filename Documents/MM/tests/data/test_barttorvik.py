"""
Tests for src/data/barttorvik.py

RED phase — all tests are written before implementation exists.
Run with: pytest tests/data/test_barttorvik.py -v
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pandas as pd
import pytest
import responses as responses_lib  # HTTP-mocking library


# ---------------------------------------------------------------------------
# Minimal fixture HTML
# ---------------------------------------------------------------------------

_MINIMAL_TRANK_HTML = textwrap.dedent(
    """
    <html>
    <body>
    <table id="t-rank-table">
      <thead>
        <tr>
          <th>Rk</th><th>Team</th><th>Conf</th><th>Record</th>
          <th>AdjEM</th><th>AdjO</th><th>AdjD</th><th>AdjT</th>
          <th>Luck</th><th>SOS AdjEM</th><th>OppO</th><th>OppD</th>
          <th>NCSOS AdjEM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td><td>Houston</td><td>BE</td><td>31-5</td>
          <td>28.34</td><td>118.7</td><td>90.36</td><td>67.2</td>
          <td>0.021</td><td>11.45</td><td>107.3</td><td>95.8</td>
          <td>2.11</td>
        </tr>
        <tr>
          <td>2</td><td>Auburn</td><td>SEC</td><td>29-4</td>
          <td>27.91</td><td>120.1</td><td>92.19</td><td>70.1</td>
          <td>-0.003</td><td>10.88</td><td>106.9</td><td>96.0</td>
          <td>1.88</td>
        </tr>
      </tbody>
    </table>
    </body>
    </html>
    """
)

_EXPECTED_COLUMNS = [
    "team", "conf", "record", "adj_em", "adj_o", "adj_d",
    "adj_t", "luck", "sos_adj_em", "opp_o", "opp_d",
    "ncsos_adj_em", "rank",
]


# ---------------------------------------------------------------------------
# _parse_trank_html — pure parser (no network)
# ---------------------------------------------------------------------------


class TestParseTrankHtml:
    """Unit tests for the pure HTML parser — zero network I/O."""

    def setup_method(self):
        # Import deferred to RED phase; will NameError until GREEN
        from src.data.barttorvik import _parse_trank_html  # noqa: PLC0415
        self.parse = _parse_trank_html

    def test_returns_dataframe(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert list(df.columns) == _EXPECTED_COLUMNS

    def test_row_count(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert len(df) == 2

    def test_team_names(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df["team"].tolist() == ["Houston", "Auburn"]

    def test_rank_column_is_int(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df["rank"].dtype in (int, "int64", "Int64")

    def test_numeric_adj_em(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df["adj_em"].dtype == float or pd.api.types.is_float_dtype(df["adj_em"])
        assert abs(df.loc[0, "adj_em"] - 28.34) < 1e-6

    def test_numeric_adj_o(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "adj_o"] - 118.7) < 1e-6

    def test_numeric_adj_d(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "adj_d"] - 90.36) < 1e-6

    def test_numeric_adj_t(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "adj_t"] - 67.2) < 1e-6

    def test_numeric_luck(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "luck"] - 0.021) < 1e-6

    def test_numeric_sos_adj_em(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "sos_adj_em"] - 11.45) < 1e-6

    def test_numeric_opp_o(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "opp_o"] - 107.3) < 1e-6

    def test_numeric_opp_d(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "opp_d"] - 95.8) < 1e-6

    def test_numeric_ncsos_adj_em(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert abs(df.loc[0, "ncsos_adj_em"] - 2.11) < 1e-6

    def test_record_is_string(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df["record"].dtype == object
        assert df.loc[0, "record"] == "31-5"

    def test_empty_tbody_returns_empty_df(self):
        """Parser must handle tables with no data rows gracefully."""
        empty_html = textwrap.dedent(
            """
            <html><body>
            <table id="t-rank-table">
              <thead><tr><th>Rk</th><th>Team</th></tr></thead>
              <tbody></tbody>
            </table>
            </body></html>
            """
        )
        df = self.parse(empty_html)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_second_row_team(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df.loc[1, "team"] == "Auburn"

    def test_second_row_rank(self):
        df = self.parse(_MINIMAL_TRANK_HTML)
        assert df.loc[1, "rank"] == 2


# ---------------------------------------------------------------------------
# fetch_trank — mocked network
# ---------------------------------------------------------------------------


class TestFetchTrank:
    """Tests for fetch_trank(); network is mocked with `responses`."""

    def setup_method(self):
        from src.data.barttorvik import fetch_trank  # noqa: PLC0415
        self.fetch_trank = fetch_trank

    @responses_lib.activate
    def test_happy_path_returns_dataframe(self):
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        df = self.fetch_trank(season=2024)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == _EXPECTED_COLUMNS

    @responses_lib.activate
    def test_passes_year_param(self):
        """fetch_trank must include `year=<season>` in the query string."""
        # JSON endpoint is tried first; register it so it "fails" gracefully
        # (returns a non-JSON body) and the HTML fallback is exercised.
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/2022_team_results.json",
            body="not-json",
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        self.fetch_trank(season=2022)
        # Two calls total: JSON attempt + HTML fallback
        assert len(responses_lib.calls) == 2
        html_call_url = responses_lib.calls[1].request.url
        assert "year=2022" in html_call_url

    def test_raises_value_error_for_season_before_2008(self):
        with pytest.raises(ValueError, match="season"):
            self.fetch_trank(season=2007)

    def test_raises_value_error_for_season_2007(self):
        with pytest.raises(ValueError):
            self.fetch_trank(season=2007)

    def test_accepts_season_2008(self):
        """2008 is the earliest valid season — should not raise on validation."""
        with responses_lib.RequestsMock() as rsps:
            rsps.add(
                responses_lib.GET,
                "https://www.barttorvik.com/trank.php",
                body=_MINIMAL_TRANK_HTML,
                status=200,
            )
            df = self.fetch_trank(season=2008)
        assert isinstance(df, pd.DataFrame)

    @responses_lib.activate
    def test_retries_on_500_then_succeeds(self):
        """First HTML call returns 500; second returns 200 — must retry and succeed."""
        # JSON endpoint is tried first; register it to fail gracefully (non-JSON body)
        # so the HTML fallback path is exercised.
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/2024_team_results.json",
            body="not-json",
            status=200,
        )
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            status=500,
        )
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        # Patch sleep so tests don't hang on backoff delay
        with patch("src.data.barttorvik.time.sleep"):
            df = self.fetch_trank(season=2024)
        assert isinstance(df, pd.DataFrame)
        # 3 calls total: 1 JSON attempt + 2 HTML attempts (500 then 200)
        assert len(responses_lib.calls) == 3

    @responses_lib.activate
    def test_raises_after_max_retries(self):
        """Three consecutive 500s must raise an exception."""
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                "https://www.barttorvik.com/trank.php",
                status=500,
            )
        with patch("src.data.barttorvik.time.sleep"):
            with pytest.raises(Exception):
                self.fetch_trank(season=2024)


# ---------------------------------------------------------------------------
# cache_trank — filesystem & network mocked
# ---------------------------------------------------------------------------


class TestCacheTrank:
    """Tests for cache_trank(); both network and filesystem are mocked."""

    def setup_method(self):
        from src.data.barttorvik import cache_trank  # noqa: PLC0415
        self.cache_trank = cache_trank

    @responses_lib.activate
    def test_returns_path_object(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        result = self.cache_trank(season=2024, cache_dir=str(tmp_path))
        assert isinstance(result, Path)

    @responses_lib.activate
    def test_file_is_written(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        path = self.cache_trank(season=2024, cache_dir=str(tmp_path))
        assert path.exists()
        assert path.stat().st_size > 0

    @responses_lib.activate
    def test_file_contains_html(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        path = self.cache_trank(season=2024, cache_dir=str(tmp_path))
        assert "<html>" in path.read_text()

    @responses_lib.activate
    def test_filename_contains_season(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        path = self.cache_trank(season=2024, cache_dir=str(tmp_path))
        assert "2024" in path.name

    @responses_lib.activate
    def test_cache_first_skips_download(self, tmp_path):
        """If the cache file already exists, no HTTP request should be made."""
        # Pre-write the cache file
        cache_file = tmp_path / "trank_2024.html"
        cache_file.write_text(_MINIMAL_TRANK_HTML)

        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        path = self.cache_trank(season=2024, cache_dir=str(tmp_path))
        # The HTTP endpoint should NOT have been hit
        assert len(responses_lib.calls) == 0
        assert path == cache_file

    @responses_lib.activate
    def test_creates_cache_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "dir"
        responses_lib.add(
            responses_lib.GET,
            "https://www.barttorvik.com/trank.php",
            body=_MINIMAL_TRANK_HTML,
            status=200,
        )
        path = self.cache_trank(season=2024, cache_dir=str(nested))
        assert nested.exists()
        assert path.exists()

    def test_raises_value_error_for_invalid_season(self, tmp_path):
        with pytest.raises(ValueError, match="season"):
            self.cache_trank(season=2005, cache_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Minimal fixture HTML for player-level tables
# ---------------------------------------------------------------------------

_MINIMAL_PORPAGATU_HTML = textwrap.dedent(
    """
    <html>
    <body>
    <table>
      <thead>
        <tr>
          <th>Player</th><th>PORPAGATU!</th><th>Min%</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Alice Smith</td><td>2.50</td><td>35.2</td>
        </tr>
        <tr>
          <td>Bob Jones</td><td>1.80</td><td>28.7</td>
        </tr>
      </tbody>
    </table>
    </body>
    </html>
    """
)

_MINIMAL_BPM_HTML = textwrap.dedent(
    """
    <html>
    <body>
    <table>
      <thead>
        <tr>
          <th>Player</th><th>BPM</th><th>Min%</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Alice Smith</td><td>3.10</td><td>35.2</td>
        </tr>
        <tr>
          <td>Bob Jones</td><td>-0.50</td><td>28.7</td>
        </tr>
      </tbody>
    </table>
    </body>
    </html>
    """
)

_MINIMAL_CONTINUITY_HTML = textwrap.dedent(
    """
    <html>
    <body>
    <table>
      <thead>
        <tr>
          <th>Team</th><th>Returning%</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Duke</td><td>72.5</td>
        </tr>
        <tr>
          <td>Kentucky</td><td>45.0</td>
        </tr>
      </tbody>
    </table>
    </body>
    </html>
    """
)

_EMPTY_TABLE_HTML = textwrap.dedent(
    """
    <html>
    <body>
    <table>
      <thead><tr><th>Player</th><th>PORPAGATU!</th><th>Min%</th></tr></thead>
      <tbody></tbody>
    </table>
    </body>
    </html>
    """
)


# ---------------------------------------------------------------------------
# TestPlayerData — fetch_porpagatu, fetch_bpm, fetch_roster_continuity,
#                  compute_team_porpagatu_weighted
# ---------------------------------------------------------------------------


class TestPlayerData:
    """Tests for player-level data fetching functions."""

    def setup_method(self):
        from src.data.barttorvik import (  # noqa: PLC0415
            fetch_porpagatu,
            fetch_bpm,
            fetch_roster_continuity,
            compute_team_porpagatu_weighted,
        )
        self.fetch_porpagatu = fetch_porpagatu
        self.fetch_bpm = fetch_bpm
        self.fetch_roster_continuity = fetch_roster_continuity
        self.compute_team_porpagatu_weighted = compute_team_porpagatu_weighted

    # ------------------------------------------------------------------
    # fetch_porpagatu
    # ------------------------------------------------------------------

    @responses_lib.activate
    def test_fetch_porpagatu_returns_dataframe(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_PORPAGATU_HTML,
            status=200,
        )
        df = self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    @responses_lib.activate
    def test_fetch_porpagatu_required_columns(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_PORPAGATU_HTML,
            status=200,
        )
        df = self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)
        for col in ("player", "porpagatu", "minutes_pct"):
            assert col in df.columns, f"Missing column: {col}"

    @responses_lib.activate
    def test_fetch_porpagatu_row_count(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_PORPAGATU_HTML,
            status=200,
        )
        df = self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)
        assert len(df) == 2

    @responses_lib.activate
    def test_fetch_porpagatu_caches_result(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_PORPAGATU_HTML,
            status=200,
        )
        self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)
        # Second call should use cache — no additional HTTP request
        call_count_after_first = len(responses_lib.calls)
        self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)
        assert len(responses_lib.calls) == call_count_after_first

    @responses_lib.activate
    def test_fetch_porpagatu_raises_value_error_on_empty(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_EMPTY_TABLE_HTML,
            status=200,
        )
        with pytest.raises(ValueError):
            self.fetch_porpagatu("UnknownTeam", 2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_porpagatu_raises_runtime_error_after_retries(self, tmp_path):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                "https://barttorvik.com/trankf.php",
                status=500,
            )
        with patch("src.data.barttorvik.time.sleep"):
            with pytest.raises(RuntimeError):
                self.fetch_porpagatu("Duke", 2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_porpagatu_url_encodes_team(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_PORPAGATU_HTML,
            status=200,
        )
        self.fetch_porpagatu("North Carolina", 2024, cache_dir=tmp_path)
        assert len(responses_lib.calls) == 1
        url = responses_lib.calls[0].request.url
        assert "tvalue=" in url
        assert "year=2024" in url
        assert "type=porpagatu" in url

    # ------------------------------------------------------------------
    # fetch_bpm
    # ------------------------------------------------------------------

    @responses_lib.activate
    def test_fetch_bpm_returns_dataframe(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_BPM_HTML,
            status=200,
        )
        df = self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    @responses_lib.activate
    def test_fetch_bpm_required_columns(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_BPM_HTML,
            status=200,
        )
        df = self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        for col in ("player", "bpm", "minutes_pct"):
            assert col in df.columns, f"Missing column: {col}"

    @responses_lib.activate
    def test_fetch_bpm_row_count(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_BPM_HTML,
            status=200,
        )
        df = self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        assert len(df) == 2

    @responses_lib.activate
    def test_fetch_bpm_caches_result(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_BPM_HTML,
            status=200,
        )
        self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        call_count_after_first = len(responses_lib.calls)
        self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        assert len(responses_lib.calls) == call_count_after_first

    @responses_lib.activate
    def test_fetch_bpm_raises_value_error_on_empty(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_EMPTY_TABLE_HTML,
            status=200,
        )
        with pytest.raises(ValueError):
            self.fetch_bpm("UnknownTeam", 2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_bpm_raises_runtime_error_after_retries(self, tmp_path):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                "https://barttorvik.com/trankf.php",
                status=500,
            )
        with patch("src.data.barttorvik.time.sleep"):
            with pytest.raises(RuntimeError):
                self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_bpm_url_params(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/trankf.php",
            body=_MINIMAL_BPM_HTML,
            status=200,
        )
        self.fetch_bpm("Duke", 2024, cache_dir=tmp_path)
        url = responses_lib.calls[0].request.url
        assert "type=bpm" in url
        assert "year=2024" in url

    # ------------------------------------------------------------------
    # fetch_roster_continuity
    # ------------------------------------------------------------------

    @responses_lib.activate
    def test_fetch_roster_continuity_returns_dataframe(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        df = self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    @responses_lib.activate
    def test_fetch_roster_continuity_required_columns(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        df = self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        assert "team" in df.columns
        assert "returning_pct" in df.columns

    @responses_lib.activate
    def test_fetch_roster_continuity_returning_pct_range(self, tmp_path):
        """returning_pct must be in [0.0, 1.0] (fraction, not percentage)."""
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        df = self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        assert df["returning_pct"].between(0.0, 1.0).all(), (
            f"returning_pct out of [0,1]: {df['returning_pct'].tolist()}"
        )

    @responses_lib.activate
    def test_fetch_roster_continuity_row_count(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        df = self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        assert len(df) == 2

    @responses_lib.activate
    def test_fetch_roster_continuity_caches_result(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        call_count_after_first = len(responses_lib.calls)
        self.fetch_roster_continuity(2024, cache_dir=tmp_path)
        assert len(responses_lib.calls) == call_count_after_first

    @responses_lib.activate
    def test_fetch_roster_continuity_raises_value_error_on_empty(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_EMPTY_TABLE_HTML,
            status=200,
        )
        with pytest.raises(ValueError):
            self.fetch_roster_continuity(2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_roster_continuity_raises_runtime_error_after_retries(self, tmp_path):
        for _ in range(3):
            responses_lib.add(
                responses_lib.GET,
                "https://barttorvik.com/continuity.php",
                status=500,
            )
        with patch("src.data.barttorvik.time.sleep"):
            with pytest.raises(RuntimeError):
                self.fetch_roster_continuity(2024, cache_dir=tmp_path)

    @responses_lib.activate
    def test_fetch_roster_continuity_url_contains_year(self, tmp_path):
        responses_lib.add(
            responses_lib.GET,
            "https://barttorvik.com/continuity.php",
            body=_MINIMAL_CONTINUITY_HTML,
            status=200,
        )
        self.fetch_roster_continuity(2025, cache_dir=tmp_path)
        url = responses_lib.calls[0].request.url
        assert "year=2025" in url

    # ------------------------------------------------------------------
    # compute_team_porpagatu_weighted
    # ------------------------------------------------------------------

    def test_weighted_mean_known_values(self):
        """player A: porpagatu=2.0, min%=0.6; player B: porpagatu=1.0, min%=0.4 → 1.6"""
        df = pd.DataFrame({
            "player": ["A", "B"],
            "porpagatu": [2.0, 1.0],
            "minutes_pct": [0.6, 0.4],
        })
        result = self.compute_team_porpagatu_weighted(df)
        assert abs(result - 1.6) < 1e-9

    def test_weighted_mean_single_player(self):
        df = pd.DataFrame({
            "player": ["A"],
            "porpagatu": [3.5],
            "minutes_pct": [1.0],
        })
        result = self.compute_team_porpagatu_weighted(df)
        assert abs(result - 3.5) < 1e-9

    def test_weighted_mean_equal_weights(self):
        df = pd.DataFrame({
            "player": ["A", "B"],
            "porpagatu": [4.0, 2.0],
            "minutes_pct": [0.5, 0.5],
        })
        result = self.compute_team_porpagatu_weighted(df)
        assert abs(result - 3.0) < 1e-9

    def test_weighted_mean_raises_on_zero_minutes(self):
        df = pd.DataFrame({
            "player": ["A", "B"],
            "porpagatu": [2.0, 1.0],
            "minutes_pct": [0.0, 0.0],
        })
        with pytest.raises(ValueError):
            self.compute_team_porpagatu_weighted(df)

    def test_weighted_mean_returns_float(self):
        df = pd.DataFrame({
            "player": ["A"],
            "porpagatu": [2.0],
            "minutes_pct": [0.5],
        })
        result = self.compute_team_porpagatu_weighted(df)
        assert isinstance(result, float)
