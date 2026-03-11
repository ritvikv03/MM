"""
Tests for src/data/hoopmath.py — Sports Reference CBB advanced stats scraper.
(Replaces defunct hoop-math.com; now parses sports-reference.com tables.)

All HTTP calls are intercepted via unittest.mock.patch on requests.get.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

_VALID_SREF_HTML = textwrap.dedent("""\
    <html><body>
    <table id="adv_school_stats">
      <thead>
        <tr>
          <th data-stat="school_name">School</th>
          <th data-stat="fg3a_per_fga_pct">3PAr</th>
          <th data-stat="fta_per_fga_pct">FTAr</th>
          <th data-stat="efg_pct">eFG%</th>
          <th data-stat="ts_pct">TS%</th>
          <th data-stat="pace">Pace</th>
          <th data-stat="off_rtg">ORtg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td data-stat="school_name">Duke</td>
          <td data-stat="fg3a_per_fga_pct">.375</td>
          <td data-stat="fta_per_fga_pct">.343</td>
          <td data-stat="efg_pct">.546</td>
          <td data-stat="ts_pct">.576</td>
          <td data-stat="pace">67.1</td>
          <td data-stat="off_rtg">117.1</td>
        </tr>
        <tr>
          <td data-stat="school_name">KentuckyNCAA</td>
          <td data-stat="fg3a_per_fga_pct">.320</td>
          <td data-stat="fta_per_fga_pct">.410</td>
          <td data-stat="efg_pct">.530</td>
          <td data-stat="ts_pct">.560</td>
          <td data-stat="pace">70.2</td>
          <td data-stat="off_rtg">115.0</td>
        </tr>
      </tbody>
    </table>
    </body></html>
""")

_VALID_DEFENSE_HTML = textwrap.dedent("""\
    <html><body>
    <table id="adv_opp_stats">
      <thead>
        <tr>
          <th data-stat="school_name">School</th>
          <th data-stat="fg3a_per_fga_pct">3PAr</th>
          <th data-stat="fta_per_fga_pct">FTAr</th>
          <th data-stat="efg_pct">eFG%</th>
          <th data-stat="ts_pct">TS%</th>
          <th data-stat="pace">Pace</th>
          <th data-stat="off_rtg">ORtg</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td data-stat="school_name">Duke</td>
          <td data-stat="fg3a_per_fga_pct">.310</td>
          <td data-stat="fta_per_fga_pct">.290</td>
          <td data-stat="efg_pct">.480</td>
          <td data-stat="ts_pct">.510</td>
          <td data-stat="pace">67.1</td>
          <td data-stat="off_rtg">98.5</td>
        </tr>
      </tbody>
    </table>
    </body></html>
""")

_NO_TABLE_HTML = "<html><body><p>No table here.</p></body></html>"


def _make_mock_response(html: str, status: int = 200) -> MagicMock:
    mock = MagicMock()
    mock.status_code = status
    mock.text = html
    mock.raise_for_status = MagicMock()
    if status >= 400:
        mock.raise_for_status.side_effect = Exception(f"HTTP {status}")
    return mock


# ---------------------------------------------------------------------------
# _parse_sref_table
# ---------------------------------------------------------------------------

class TestParseSrefTable:
    def _fn(self):
        from src.data.hoopmath import _parse_sref_table
        return _parse_sref_table

    def test_returns_dataframe(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert len(df) == 2

    def test_required_columns_present(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        for col in ["team", "season", "side", "fg3a_per_fga_pct", "fta_per_fga_pct",
                    "efg_pct", "ts_pct", "pace", "off_rtg"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_duke_values(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        duke = df[df["team"] == "Duke"].iloc[0]
        assert duke["fg3a_per_fga_pct"] == pytest.approx(0.375)
        assert duke["fta_per_fga_pct"] == pytest.approx(0.343)
        assert duke["efg_pct"] == pytest.approx(0.546)
        assert duke["ts_pct"] == pytest.approx(0.576)
        assert duke["pace"] == pytest.approx(67.1)
        assert duke["off_rtg"] == pytest.approx(117.1)

    def test_ncaa_suffix_stripped(self):
        """Sports Reference appends 'NCAA' to tournament teams — must be stripped."""
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert "Kentucky" in df["team"].values
        assert "KentuckyNCAA" not in df["team"].values

    def test_season_column_set(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert (df["season"] == 2024).all()

    def test_side_column_set_offense(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert (df["side"] == "offense").all()

    def test_side_column_set_defense(self):
        df = self._fn()(_VALID_DEFENSE_HTML, 2024, "defense")
        assert (df["side"] == "defense").all()

    def test_defense_table_id(self):
        df = self._fn()(_VALID_DEFENSE_HTML, 2024, "defense")
        assert len(df) == 1
        assert df.iloc[0]["team"] == "Duke"

    def test_no_table_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not find table"):
            self._fn()(_NO_TABLE_HTML, 2024, "offense")

    def test_empty_html_raises_value_error(self):
        with pytest.raises(ValueError, match="Could not find table"):
            self._fn()("", 2024, "offense")

    def test_float_columns_are_float(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        for col in ["fg3a_per_fga_pct", "fta_per_fga_pct", "efg_pct", "ts_pct", "pace", "off_rtg"]:
            assert df[col].dtype == float, f"{col} should be float"

    def test_season_column_is_int(self):
        df = self._fn()(_VALID_SREF_HTML, 2024, "offense")
        assert df["season"].dtype in (int, "int64", "int32")


# ---------------------------------------------------------------------------
# fetch_all_teams_shots
# ---------------------------------------------------------------------------

class TestFetchAllTeamsShots:
    def test_returns_dataframe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            df = hoopmath.fetch_all_teams_shots(2024, "offense")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_uses_correct_url_for_offense(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)) as mock_get:
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            hoopmath.fetch_all_teams_shots(2024, "offense")
        url = mock_get.call_args[0][0]
        assert "advanced-school-stats" in url
        assert "2024" in url

    def test_uses_correct_url_for_defense(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_DEFENSE_HTML)) as mock_get:
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            hoopmath.fetch_all_teams_shots(2024, "defense")
        url = mock_get.call_args[0][0]
        assert "advanced-opponent-stats" in url

    def test_caches_to_disk(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)) as mock_get:
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            hoopmath.fetch_all_teams_shots(2024, "offense")
            hoopmath.fetch_all_teams_shots(2024, "offense")  # second call — should hit cache
        assert mock_get.call_count == 1

    def test_cache_file_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            hoopmath.fetch_all_teams_shots(2024, "offense")
        cache_files = list(tmp_path.glob("*.html"))
        assert len(cache_files) == 1

    def test_invalid_side_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        from src.data import hoopmath
        import importlib; importlib.reload(hoopmath)
        with pytest.raises(ValueError, match="Invalid side"):
            hoopmath.fetch_all_teams_shots(2024, "invalid")

    def test_runtime_error_after_retries(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        err_response = _make_mock_response("", 500)
        err_response.raise_for_status.side_effect = Exception("Server error")
        with patch("requests.get", return_value=err_response):
            with patch("time.sleep"):
                from src.data import hoopmath
                import importlib; importlib.reload(hoopmath)
                with pytest.raises((RuntimeError, Exception)):
                    hoopmath.fetch_all_teams_shots(2024, "offense")


# ---------------------------------------------------------------------------
# fetch_team_shots
# ---------------------------------------------------------------------------

class TestFetchTeamShots:
    def test_returns_single_row_for_known_team(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            df = hoopmath.fetch_team_shots("Duke", 2024, "offense")
        assert len(df) == 1
        assert df.iloc[0]["team"] == "Duke"

    def test_case_insensitive_team_match(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            df = hoopmath.fetch_team_shots("duke", 2024, "offense")
        assert len(df) == 1

    def test_unknown_team_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            with pytest.raises(ValueError, match="not found"):
                hoopmath.fetch_team_shots("Nonexistent Team", 2024, "offense")

    def test_invalid_side_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        from src.data import hoopmath
        import importlib; importlib.reload(hoopmath)
        with pytest.raises(ValueError, match="Invalid side"):
            hoopmath.fetch_team_shots("Duke", 2024, "bad_side")

    def test_ncaa_suffix_team_found(self, tmp_path, monkeypatch):
        """Kentucky is stored as 'KentuckyNCAA' in fixture — should resolve to 'Kentucky'."""
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_SREF_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            df = hoopmath.fetch_team_shots("Kentucky", 2024, "offense")
        assert len(df) == 1
        assert df.iloc[0]["team"] == "Kentucky"

    def test_defense_side(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOOPMATH_CACHE_DIR", str(tmp_path))
        with patch("requests.get", return_value=_make_mock_response(_VALID_DEFENSE_HTML)):
            from src.data import hoopmath
            import importlib; importlib.reload(hoopmath)
            df = hoopmath.fetch_team_shots("Duke", 2024, "defense")
        assert df.iloc[0]["side"] == "defense"
        assert df.iloc[0]["fg3a_per_fga_pct"] == pytest.approx(0.310)
