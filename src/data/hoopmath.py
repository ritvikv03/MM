"""
src/data/hoopmath.py
====================
Shot-type proxy scraper — sourced from Sports Reference CBB advanced stats.

NOTE: hoop-math.com is defunct (DNS does not resolve as of 2026-03-10).
This module now scrapes Sports Reference CBB advanced team stats as a
replacement, providing the following shot-quality proxies:

  - ``fg3a_per_fga_pct`` — fraction of shot attempts that are 3-pointers
    (low value → interior/rim-heavy offence)
  - ``fta_per_fga_pct``  — free-throw attempt rate; proxy for attacking the rim
  - ``efg_pct``          — effective field-goal percentage (quality proxy)
  - ``ts_pct``           — true-shooting percentage
  - ``pace``             — possessions per 40 minutes (tempo context)
  - ``off_rtg``          — offensive rating (points per 100 possessions)

Public API
----------
fetch_team_shots(team, season, side)       -> pd.DataFrame
fetch_all_teams_shots(season, side)        -> pd.DataFrame
_parse_sref_table(html, season, side)      -> pd.DataFrame  (pure, testable)

Disk cache lives at data/raw/hoopmath/ (or HOOPMATH_CACHE_DIR env var).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

_SREF_BASE = "https://www.sports-reference.com/cbb/seasons/men"
_OFFENSE_TABLE_ID = "adv_school_stats"
_DEFENSE_TABLE_ID = "adv_opp_stats"
_RETRY_BACKOFF_BASE = 1.0  # seconds
_MAX_RETRIES = 3

_OUTPUT_COLUMNS: List[str] = [
    "team",
    "season",
    "side",
    "fg3a_per_fga_pct",   # 3PT attempt rate — proxy for perimeter tendency
    "fta_per_fga_pct",    # FTA rate       — proxy for rim-attack tendency
    "efg_pct",            # effective FG%
    "ts_pct",             # true shooting %
    "pace",               # possessions per 40 min
    "off_rtg",            # offensive rating (points per 100 poss)
]

_VALID_SIDES = ("offense", "defense")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cache_dir() -> Path:
    cache_env = os.environ.get("HOOPMATH_CACHE_DIR")
    if cache_env:
        cache_dir = Path(cache_env)
    else:
        cache_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "hoopmath"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_path(season: int, side: str) -> Path:
    return _get_cache_dir() / f"sref_{season}_{side}.html"


def _build_url(season: int, side: str) -> str:
    suffix = "advanced-school-stats" if side == "offense" else "advanced-opponent-stats"
    return f"{_SREF_BASE}/{season}-{suffix}.html"


def _validate_side(side: str) -> None:
    if side not in _VALID_SIDES:
        raise ValueError(f"Invalid side '{side}'. Must be one of {_VALID_SIDES}.")


def _fetch_html(url: str) -> str:
    """GET *url* with retry/backoff; raises RuntimeError after exhausting retries."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.HTTPError as exc:
            last_exc = exc
            logger.warning("HTTP error on attempt %d for %s: %s", attempt + 1, url, exc)
            if exc.response is not None and exc.response.status_code < 500:
                raise
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning("Request error on attempt %d for %s: %s", attempt + 1, url, exc)
        time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
    raise RuntimeError(f"Failed to fetch {url} after {_MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------------


def _parse_sref_table(html: str, season: int, side: str) -> pd.DataFrame:
    """Parse the Sports Reference advanced school/opponent stats table.

    Parameters
    ----------
    html:
        Raw HTML from sports-reference.com advanced stats page.
    season:
        Four-digit season year (injected into ``season`` column).
    side:
        ``"offense"`` or ``"defense"`` (injected into ``side`` column).

    Returns
    -------
    pd.DataFrame
        Columns: team, season, side, fg3a_per_fga_pct, fta_per_fga_pct,
        efg_pct, ts_pct, pace, off_rtg.
        Percentage columns are stored as floats (e.g. 0.343, not 34.3).

    Raises
    ------
    ValueError
        If the expected table is not found in the HTML.
    """
    soup = BeautifulSoup(html, "lxml")
    table_id = _OFFENSE_TABLE_ID if side == "offense" else _DEFENSE_TABLE_ID
    table = soup.find("table", id=table_id)
    if table is None:
        raise ValueError(
            f"Could not find table id='{table_id}' in the HTML. "
            "The Sports Reference page structure may have changed."
        )

    records = []
    for row in table.find_all("tr"):
        school_cell = row.find("td", {"data-stat": "school_name"})
        if school_cell is None:
            continue

        def _val(stat: str) -> str:
            cell = row.find("td", {"data-stat": stat})
            return cell.get_text(strip=True) if cell else ""

        team_raw = school_cell.get_text(strip=True)
        # Sports Reference appends "NCAA" to tournament teams — strip it.
        team = team_raw.replace("NCAA", "").strip()

        try:
            fg3a = float(_val("fg3a_per_fga_pct") or "nan")
            fta  = float(_val("fta_per_fga_pct")  or "nan")
            efg  = float(_val("efg_pct")           or "nan")
            ts   = float(_val("ts_pct")            or "nan")
            pace = float(_val("pace")              or "nan")
            ortg = float(_val("off_rtg")           or "nan")
        except (ValueError, TypeError):
            continue

        records.append({
            "team":             team,
            "season":           season,
            "side":             side,
            "fg3a_per_fga_pct": fg3a,
            "fta_per_fga_pct":  fta,
            "efg_pct":          efg,
            "ts_pct":           ts,
            "pace":             pace,
            "off_rtg":          ortg,
        })

    df = pd.DataFrame(records, columns=_OUTPUT_COLUMNS)
    df["season"] = df["season"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_team_shots(
    team: str,
    season: int,
    side: str = "offense",
) -> pd.DataFrame:
    """Fetch shot-type proxies for a single team from the season table.

    Fetches the full season table (cached to disk), then filters to *team*.

    Parameters
    ----------
    team:
        Team name as it appears on Sports Reference (e.g. ``"Duke"``).
        Matching is case-insensitive and tolerates the trailing ``"NCAA"``
        suffix that Sports Reference appends to tournament teams.
    season:
        Four-digit season year (e.g. ``2024``).
    side:
        ``"offense"`` (default) or ``"defense"``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns in :data:`_OUTPUT_COLUMNS`.

    Raises
    ------
    ValueError
        If *side* is invalid or *team* is not found in the season table.
    """
    _validate_side(side)
    all_df = fetch_all_teams_shots(season, side=side)
    match = all_df[all_df["team"].str.lower() == team.lower()]
    if match.empty:
        raise ValueError(
            f"Team '{team}' not found in Sports Reference {season} {side} stats."
        )
    return match.reset_index(drop=True)


def fetch_all_teams_shots(
    season: int,
    side: str = "offense",
) -> pd.DataFrame:
    """Fetch shot-type proxies for all D-I teams in *season*.

    Results are cached to disk after the first successful fetch.

    Parameters
    ----------
    season:
        Four-digit season year.
    side:
        ``"offense"`` (default) or ``"defense"``.

    Returns
    -------
    pd.DataFrame
        One row per team with columns in :data:`_OUTPUT_COLUMNS`.

    Raises
    ------
    ValueError
        If *side* is invalid or the page table cannot be found.
    """
    _validate_side(side)
    cache_file = _cache_path(season, side)

    if cache_file.exists():
        logger.info("Cache hit: %s", cache_file)
        html = cache_file.read_text(encoding="utf-8")
    else:
        url = _build_url(season, side)
        logger.info("Fetching Sports Reference advanced stats: %s", url)
        html = _fetch_html(url)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(html, encoding="utf-8")
        logger.info("Cached to %s", cache_file)

    return _parse_sref_table(html, season, side)
