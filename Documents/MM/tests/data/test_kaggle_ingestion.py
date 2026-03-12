"""
tests/data/test_kaggle_ingestion.py

RED phase — tests written before implementation exists.
All kaggle API calls are mocked; no real network traffic.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_game_results_csv() -> str:
    return (
        "Season,DayNum,WTeamID,WScore,LTeamID,LScore,WLoc,NumOT\n"
        "2024,10,1101,75,1102,60,H,0\n"
        "2024,11,1103,80,1104,70,A,1\n"
        "2023,15,1105,65,1106,55,N,0\n"
    )


def _make_seeds_csv() -> str:
    return (
        "Season,Slot,TeamID,Seed\n"
        "2024,W01,1101,W01\n"
        "2024,X02,1102,X02\n"
        "2024,Y16,1103,Y16\n"
        "2024,Z11,1104,Z11\n"
    )


def _make_spellings_csv() -> str:
    return (
        "TeamNameSpelling,TeamID\n"
        "duke,1101\n"
        "duke blue devils,1101\n"
        "kentucky,1102\n"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def game_results_file(tmp_path: Path) -> Path:
    f = tmp_path / "MRegularSeasonCompactResults.csv"
    f.write_text(_make_game_results_csv())
    return f


@pytest.fixture()
def seeds_file(tmp_path: Path) -> Path:
    f = tmp_path / "MNCAATourneySeeds.csv"
    f.write_text(_make_seeds_csv())
    return f


@pytest.fixture()
def spellings_file(tmp_path: Path) -> Path:
    f = tmp_path / "MTeamSpellings.csv"
    f.write_text(_make_spellings_csv())
    return f


@pytest.fixture()
def bad_schema_file(tmp_path: Path) -> Path:
    """CSV that is missing required columns."""
    f = tmp_path / "bad_schema.csv"
    f.write_text("Col1,Col2\n1,2\n")
    return f


# ---------------------------------------------------------------------------
# download_march_mania
# ---------------------------------------------------------------------------

class TestDownloadMarchMania:
    """Tests for download_march_mania()."""

    def _make_fake_competition(self, tmp_path: Path) -> tuple[MagicMock, list[Path]]:
        """Return a mock kaggle KaggleApiExtended and pre-created stub files."""
        # Simulate kaggle writing files into tmp_path
        files = [
            tmp_path / "MRegularSeasonCompactResults.csv",
            tmp_path / "MNCAATourneySeeds.csv",
            tmp_path / "MTeamSpellings.csv",
            tmp_path / "MSampleSubmission2024.csv",
        ]
        for f in files:
            f.write_text("placeholder")

        mock_api = MagicMock()
        mock_api.competition_download_files.return_value = None
        return mock_api, files

    @patch("src.data.kaggle_ingestion.KaggleApi")
    @patch("src.data.kaggle_ingestion.load_dotenv")
    def test_returns_list_of_paths(
        self,
        mock_load_dotenv: MagicMock,
        mock_kaggle_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.data.kaggle_ingestion import download_march_mania

        mock_api, created_files = self._make_fake_competition(tmp_path)
        mock_kaggle_cls.return_value = mock_api

        result = download_march_mania(str(tmp_path))

        assert isinstance(result, list)
        assert all(isinstance(p, Path) for p in result)
        mock_api.authenticate.assert_called_once()
        mock_api.competition_download_files.assert_called_once()

    @patch("src.data.kaggle_ingestion.KaggleApi")
    @patch("src.data.kaggle_ingestion.load_dotenv")
    def test_season_filter(
        self,
        mock_load_dotenv: MagicMock,
        mock_kaggle_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.data.kaggle_ingestion import download_march_mania

        mock_api, _ = self._make_fake_competition(tmp_path)
        mock_kaggle_cls.return_value = mock_api

        result = download_march_mania(str(tmp_path), season=2024)

        # Only files whose name contains "2024" should be returned
        for p in result:
            assert "2024" in p.name

    @patch("src.data.kaggle_ingestion.KaggleApi")
    @patch("src.data.kaggle_ingestion.load_dotenv")
    def test_no_season_filter_returns_all_csvs(
        self,
        mock_load_dotenv: MagicMock,
        mock_kaggle_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.data.kaggle_ingestion import download_march_mania

        mock_api, _ = self._make_fake_competition(tmp_path)
        mock_kaggle_cls.return_value = mock_api

        result = download_march_mania(str(tmp_path))

        # All .csv files in tmp_path should be returned
        expected = sorted(tmp_path.glob("*.csv"))
        assert sorted(result) == expected

    @patch("src.data.kaggle_ingestion.KaggleApi")
    @patch("src.data.kaggle_ingestion.load_dotenv")
    def test_competition_slug_is_march_mania_2024(
        self,
        mock_load_dotenv: MagicMock,
        mock_kaggle_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from src.data.kaggle_ingestion import download_march_mania

        mock_api = MagicMock()
        mock_api.competition_download_files.return_value = None
        mock_kaggle_cls.return_value = mock_api

        download_march_mania(str(tmp_path))

        call_kwargs = mock_api.competition_download_files.call_args
        args, kwargs = call_kwargs
        slug = args[0] if args else kwargs.get("competition", kwargs.get("competition_name", ""))
        assert "march-machine-learning-mania-2024" in slug


# ---------------------------------------------------------------------------
# load_game_results
# ---------------------------------------------------------------------------

class TestLoadGameResults:
    """Tests for load_game_results()."""

    def test_returns_dataframe_with_correct_columns(self, game_results_file: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        df = load_game_results(game_results_file)
        assert isinstance(df, pd.DataFrame)
        expected_cols = {"Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"}
        assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_csv(self, game_results_file: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        df = load_game_results(game_results_file)
        assert len(df) == 3

    def test_numeric_columns_are_numeric(self, game_results_file: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        df = load_game_results(game_results_file)
        for col in ["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "NumOT"]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric"

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        with pytest.raises(FileNotFoundError):
            load_game_results(tmp_path / "nonexistent.csv")

    def test_raises_value_error_on_bad_schema(self, bad_schema_file: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        with pytest.raises(ValueError, match="missing columns"):
            load_game_results(bad_schema_file)

    def test_wloc_values_are_valid(self, game_results_file: Path) -> None:
        from src.data.kaggle_ingestion import load_game_results

        df = load_game_results(game_results_file)
        assert set(df["WLoc"].unique()).issubset({"H", "A", "N"})


# ---------------------------------------------------------------------------
# load_seeds
# ---------------------------------------------------------------------------

class TestLoadSeeds:
    """Tests for load_seeds()."""

    def test_returns_dataframe_with_correct_columns(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        assert isinstance(df, pd.DataFrame)
        expected = {"Season", "Slot", "TeamID", "Seed"}
        assert expected.issubset(set(df.columns))

    def test_region_column_is_single_char(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        assert "region" in df.columns
        assert df["region"].str.len().eq(1).all()

    def test_seed_num_column_is_integer(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        assert "seed_num" in df.columns
        assert pd.api.types.is_integer_dtype(df["seed_num"])

    def test_seed_num_values_are_in_range(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        assert df["seed_num"].between(1, 16).all()

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        with pytest.raises(FileNotFoundError):
            load_seeds(tmp_path / "nonexistent.csv")

    def test_raises_value_error_on_bad_schema(self, bad_schema_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        with pytest.raises(ValueError, match="missing columns"):
            load_seeds(bad_schema_file)

    def test_row_count_matches_csv(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        assert len(df) == 4

    def test_seed_parsing_w01(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        row = df[df["Seed"] == "W01"].iloc[0]
        assert row["region"] == "W"
        assert row["seed_num"] == 1

    def test_seed_parsing_y16(self, seeds_file: Path) -> None:
        from src.data.kaggle_ingestion import load_seeds

        df = load_seeds(seeds_file)
        row = df[df["Seed"] == "Y16"].iloc[0]
        assert row["region"] == "Y"
        assert row["seed_num"] == 16


# ---------------------------------------------------------------------------
# load_team_spellings
# ---------------------------------------------------------------------------

class TestLoadTeamSpellings:
    """Tests for load_team_spellings()."""

    def test_returns_dataframe_with_correct_columns(self, spellings_file: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        df = load_team_spellings(spellings_file)
        assert isinstance(df, pd.DataFrame)
        assert {"TeamNameSpelling", "TeamID"}.issubset(set(df.columns))

    def test_row_count_matches_csv(self, spellings_file: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        df = load_team_spellings(spellings_file)
        assert len(df) == 3

    def test_team_id_is_numeric(self, spellings_file: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        df = load_team_spellings(spellings_file)
        assert pd.api.types.is_numeric_dtype(df["TeamID"])

    def test_raises_file_not_found(self, tmp_path: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        with pytest.raises(FileNotFoundError):
            load_team_spellings(tmp_path / "nonexistent.csv")

    def test_raises_value_error_on_bad_schema(self, bad_schema_file: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        with pytest.raises(ValueError, match="missing columns"):
            load_team_spellings(bad_schema_file)

    def test_spellings_are_strings(self, spellings_file: Path) -> None:
        from src.data.kaggle_ingestion import load_team_spellings

        df = load_team_spellings(spellings_file)
        assert df["TeamNameSpelling"].dtype == object
