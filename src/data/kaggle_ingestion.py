"""
src/data/kaggle_ingestion.py

Kaggle March Mania data ingestion module.

Downloads the 'march-machine-learning-mania-2024' competition files and
provides loaders for the key CSVs used in the ST-GNN pipeline.

Environment variables required (loaded from .env via python-dotenv):
    KAGGLE_USERNAME
    KAGGLE_KEY
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import]

load_dotenv()

# ---------------------------------------------------------------------------
# Competition constant
# ---------------------------------------------------------------------------

_COMPETITION_SLUG = "march-machine-learning-mania-2024"

# ---------------------------------------------------------------------------
# Required column schemas
# ---------------------------------------------------------------------------

_GAME_RESULTS_COLS: list[str] = [
    "Season",
    "DayNum",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WLoc",
    "NumOT",
]

_SEEDS_COLS: list[str] = ["Season", "Slot", "TeamID", "Seed"]

_SPELLINGS_COLS: list[str] = ["TeamNameSpelling", "TeamID"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str], filepath: Path) -> None:
    """Raise ValueError if *df* is missing any of the *required* columns."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{filepath.name} missing columns: {sorted(missing)}"
        )


def _assert_exists(filepath: Path) -> None:
    """Raise FileNotFoundError if *filepath* does not exist."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_march_mania(
    dest_dir: str,
    season: int | None = None,
) -> list[Path]:
    """Download the March Machine Learning Mania 2024 competition files.

    Parameters
    ----------
    dest_dir:
        Directory where files will be written.
    season:
        If provided, return only files whose filename contains this year as a
        substring (e.g. ``2024`` matches ``MSampleSubmission2024.csv``).

    Returns
    -------
    list[Path]
        Sorted list of CSV ``Path`` objects present in *dest_dir* after
        downloading. Filtered by *season* when specified.

    Raises
    ------
    RuntimeError
        If the Kaggle API raises an unexpected exception.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        _COMPETITION_SLUG,
        path=str(dest),
        quiet=False,
        force=False,
    )

    all_csvs = sorted(dest.glob("*.csv"))

    if season is not None:
        return [p for p in all_csvs if str(season) in p.name]

    return all_csvs


def load_game_results(filepath: Path) -> pd.DataFrame:
    """Load a game-results CSV and return a validated DataFrame.

    Expected columns: Season, DayNum, WTeamID, WScore, LTeamID, LScore,
    WLoc, NumOT.

    Parameters
    ----------
    filepath:
        Path to the CSV file (e.g. ``MRegularSeasonCompactResults.csv`` or
        ``MNCAATourneyCompactResults.csv``).

    Returns
    -------
    pd.DataFrame
        Raw rows with numeric types enforced on all non-``WLoc`` columns.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are absent from the file.
    """
    _assert_exists(filepath)
    df = pd.read_csv(filepath)
    _validate_columns(df, _GAME_RESULTS_COLS, filepath)

    numeric_cols = [c for c in _GAME_RESULTS_COLS if c != "WLoc"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    return df


def load_seeds(filepath: Path) -> pd.DataFrame:
    """Load a tournament seeds CSV and return a validated, enriched DataFrame.

    In addition to the raw columns (Season, Slot, TeamID, Seed) two derived
    columns are added:

    * ``region``   — single character extracted from the seed string (e.g.
                     ``'W'`` for ``'W01'``).
    * ``seed_num`` — integer seed rank extracted from the seed string (e.g.
                     ``1`` for ``'W01'``).

    Parameters
    ----------
    filepath:
        Path to ``MNCAATourneySeeds.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are absent from the file.
    """
    _assert_exists(filepath)
    df = pd.read_csv(filepath)
    _validate_columns(df, _SEEDS_COLS, filepath)

    # Seed strings look like "W01", "X02", "Y16a", "Z11b".
    # Region  = first character (letter).
    # SeedNum = the next 1-2 digit number (ignore trailing play-in suffix).
    df["region"] = df["Seed"].str[0]
    df["seed_num"] = (
        df["Seed"]
        .str.extract(r"[A-Za-z](\d+)", expand=False)
        .astype(int)
    )

    return df


def load_team_spellings(filepath: Path) -> pd.DataFrame:
    """Load a team spellings crosswalk CSV and return a validated DataFrame.

    Expected columns: TeamNameSpelling, TeamID.

    Parameters
    ----------
    filepath:
        Path to ``MTeamSpellings.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    ValueError
        If required columns are absent from the file.
    """
    _assert_exists(filepath)
    df = pd.read_csv(filepath)
    _validate_columns(df, _SPELLINGS_COLS, filepath)

    df["TeamID"] = pd.to_numeric(df["TeamID"])
    df["TeamNameSpelling"] = df["TeamNameSpelling"].astype(str)

    return df
