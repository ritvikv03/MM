"""
src/data/sports_reference.py

Sports Reference (via sportsipy) team stats ingestion module.

Pulls per-team Four Factors, efficiency ratings, and win/loss records for a
given NCAA basketball season and exposes them as plain Python dicts or a
Pandas DataFrame for downstream graph construction.
"""

from __future__ import annotations

import re
import string

import pandas as pd
from sportsipy.ncaab.teams import Teams  # type: ignore[import]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_SEASON = 2003
_MAX_SEASON = 2026

# Mapping: canonical return key  →  sportsipy attribute name
_STAT_MAP: dict[str, str] = {
    "wins": "wins",
    "losses": "losses",
    "pace": "pace",
    "ortg": "offensive_rating",
    "drtg": "defensive_rating",
    "efg_pct": "effective_field_goal_percentage",
    "opp_efg_pct": "opp_effective_field_goal_percentage",
    "tov_pct": "turnover_percentage",
    "opp_tov_pct": "opp_turnover_percentage",
    "orb_pct": "offensive_rebound_percentage",
    "ft_rate": "free_throw_attempt_rate",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_season(season: int) -> None:
    """Raise ValueError when *season* is outside the supported range."""
    if season < _MIN_SEASON or season > _MAX_SEASON:
        raise ValueError(
            f"season must be between {_MIN_SEASON} and {_MAX_SEASON}, got {season}"
        )


def _team_to_dict(team: object, season: int) -> dict:
    """Convert a sportsipy Team object to the canonical stats dict."""
    record: dict = {
        "team_id": getattr(team, "team_id", None),
        "season": season,
    }
    for canonical_key, attr_name in _STAT_MAP.items():
        record[canonical_key] = getattr(team, attr_name, None)
    return record


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_team_name(raw: str) -> str:
    """Normalise a raw team name for crosswalk matching.

    Transformations applied (in order):
    1. Strip leading / trailing whitespace.
    2. Convert to lowercase.
    3. Remove all punctuation characters *except* spaces.
    4. Collapse runs of whitespace to a single space.
    5. Replace spaces with underscores.

    Parameters
    ----------
    raw:
        Raw team name string (e.g. ``"St. John's"``).

    Returns
    -------
    str
        Normalised string (e.g. ``"st_johns"``).
    """
    if not raw:
        return raw

    s = raw.strip().lower()
    # Remove punctuation except spaces
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    # Replace spaces with underscores
    s = s.replace(" ", "_")
    return s


def fetch_team_stats(team_name: str, season: int) -> dict:
    """Fetch a single team's season stats from Sports Reference via sportsipy.

    Parameters
    ----------
    team_name:
        Team name as it appears in sportsipy (case-insensitive; normalised
        internally before comparison).
    season:
        NCAA season year (e.g. 2024 means the 2023-24 season).

    Returns
    -------
    dict
        Keys: ``team_id``, ``season``, ``wins``, ``losses``, ``pace``,
        ``ortg``, ``drtg``, ``efg_pct``, ``opp_efg_pct``, ``tov_pct``,
        ``opp_tov_pct``, ``orb_pct``, ``ft_rate``.

    Raises
    ------
    ValueError
        If *season* is outside [2003, 2026].
    ValueError
        If *team_name* is not found in the sportsipy data for that season.
    RuntimeError
        Wraps any exception raised by sportsipy.
    """
    _validate_season(season)

    normalised_query = normalize_team_name(team_name)

    try:
        teams = Teams(season)
    except Exception as exc:
        raise RuntimeError(f"sportsipy error while fetching Teams({season}): {exc}") from exc

    for team in teams:
        team_norm = normalize_team_name(getattr(team, "name", "") or "")
        team_id_norm = normalize_team_name(getattr(team, "team_id", "") or "")
        if normalised_query in (team_norm, team_id_norm):
            return _team_to_dict(team, season)

    raise ValueError(
        f"Team '{team_name}' not found in sportsipy data for season {season}"
    )


def fetch_all_teams(season: int) -> pd.DataFrame:
    """Fetch stats for every team in a given season.

    Parameters
    ----------
    season:
        NCAA season year.

    Returns
    -------
    pd.DataFrame
        One row per team; columns match the keys returned by
        :func:`fetch_team_stats`.

    Raises
    ------
    ValueError
        If *season* is outside [2003, 2026].
    RuntimeError
        Wraps any exception raised by sportsipy.
    """
    _validate_season(season)

    try:
        teams = Teams(season)
    except Exception as exc:
        raise RuntimeError(f"sportsipy error while fetching Teams({season}): {exc}") from exc

    records: list[dict] = [_team_to_dict(team, season) for team in teams]

    if not records:
        # Return an empty DataFrame with the correct column order
        columns = ["team_id", "season"] + list(_STAT_MAP.keys())
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)
