"""
src/data/barttorvik.py

Barttorvik (T-Rank) data ingestion module.

Barttorvik is the sole source for both team efficiency AND player-level data.

Public API
----------
fetch_trank(season)                               -> pd.DataFrame
_parse_trank_html(html)                           -> pd.DataFrame
_parse_trank_json(data)                           -> pd.DataFrame
cache_trank(season, cache_dir)                    -> Path
fetch_porpagatu(team, year, cache_dir)            -> pd.DataFrame
fetch_bpm(team, year, cache_dir)                  -> pd.DataFrame
fetch_roster_continuity(year, cache_dir)          -> pd.DataFrame
compute_team_porpagatu_weighted(player_df)        -> float

Rules
-----
- season must be >= 2008.
- Exponential backoff: up to 3 attempts, sleeping 2^n seconds between failures.
- Cache-first: if the raw HTML file already exists on disk it is returned as-is.

Data Endpoint Notes (2026-03-12)
---------------------------------
barttorvik.com/trank.php renders its table entirely via JavaScript.  A plain
requests.get() call only receives a Cloudflare-style browser-verification
challenge page (HTML, HTTP 200, body="Verifying Browser..."), so BeautifulSoup
finds no table and returns an empty DataFrame.

The working endpoint is a static JSON file served without Cloudflare challenge:
    https://barttorvik.com/{year}_team_results.json

fetch_trank() tries this JSON endpoint first.  If the response is not valid
JSON (e.g. the site returns the JS-challenge HTML), it falls back to the legacy
HTML path (_parse_trank_html) so that unit tests that mock the HTML endpoint
continue to pass.

JSON column mapping (45-element arrays, confirmed against published 2024 data):
    [0]  = rank (T-Rank overall rank)
    [1]  = team
    [2]  = conf
    [3]  = record (W-L string)
    [4]  = adj_o  (AdjOE — adjusted offensive efficiency)
    [5]  = adj_o_rank
    [6]  = adj_d  (AdjDE — adjusted defensive efficiency)
    [7]  = adj_d_rank
    [8]  = barthag (power rating / Pythagorean win probability)
    [9]  = barthag_rank
    [10] = wins
    [11] = losses
    [12] = conf_wins
    [13] = conf_losses
    [14] = conf_record (string)
    [15–22] = shooting/efficiency split metrics (EFG%, TOR, ORB, FTR, etc.)
    [23] = opp_o  (average opponent AdjOE — SOS offensive component)
    [24] = opp_d  (average opponent AdjDE — SOS defensive component)
    [25–26] = duplicate of [23–24] (alternate calculation window)
    [27] = ncsos_opp_o (non-conf SOS offensive component)
    [28] = ncsos_opp_d (non-conf SOS defensive component)
    [33] = luck  (extra wins/losses from close games)
    [44] = adj_t  (adjusted tempo — possessions per 40 minutes)

Derived columns:
    adj_em      = adj_o − adj_d
    sos_adj_em  = opp_o − opp_d  ([23] − [24])
    ncsos_adj_em = ncsos_opp_o − ncsos_opp_d  ([27] − [28])
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://www.barttorvik.com/trank.php"
# Direct JSON endpoint — no Cloudflare JS challenge (confirmed working 2026-03-12)
_JSON_BASE_URL = "https://barttorvik.com/{year}_team_results.json"
_EARLIEST_SEASON = 2008
_MAX_RETRIES = 3

# Browser-like headers to avoid 403 from barttorvik on headless/CI environments.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://barttorvik.com/",
    "DNT": "1",
}

_COLUMN_ORDER = [
    "team",
    "conf",
    "record",
    "adj_em",
    "adj_o",
    "adj_d",
    "adj_t",
    "luck",
    "sos_adj_em",
    "opp_o",
    "opp_d",
    "ncsos_adj_em",
    "rank",
]

# Map from zero-based column index in the source table to our canonical name.
# T-Rank table structure (0-indexed):
#   0  Rk   1  Team   2  Conf   3  Record
#   4  AdjEM  5  AdjO  6  AdjD  7  AdjT
#   8  Luck   9  SOS AdjEM  10 OppO  11 OppD
#   12 NCSOS AdjEM
_IDX_MAP = {
    0: "rank",
    1: "team",
    2: "conf",
    3: "record",
    4: "adj_em",
    5: "adj_o",
    6: "adj_d",
    7: "adj_t",
    8: "luck",
    9: "sos_adj_em",
    10: "opp_o",
    11: "opp_d",
    12: "ncsos_adj_em",
}

_FLOAT_COLS = {"adj_em", "adj_o", "adj_d", "adj_t", "luck", "sos_adj_em", "opp_o", "opp_d", "ncsos_adj_em"}
_INT_COLS = {"rank"}


# ---------------------------------------------------------------------------
# JSON column index mapping (see module docstring for full derivation)
# ---------------------------------------------------------------------------

_JSON_IDX = {
    "rank":          0,
    "team":          1,
    "conf":          2,
    "record":        3,
    "adj_o":         4,
    "adj_d":         6,
    "opp_o":         23,
    "opp_d":         24,
    "ncsos_opp_o":   27,
    "ncsos_opp_d":   28,
    "luck":          33,
    "adj_t":         44,
}


# ---------------------------------------------------------------------------
# JSON parser — used when the direct JSON endpoint is available
# ---------------------------------------------------------------------------


def _parse_trank_json(data: list) -> pd.DataFrame:
    """
    Parse the raw list-of-arrays returned by barttorvik.com/{year}_team_results.json
    and return a normalised DataFrame with the same schema as _parse_trank_html.

    Parameters
    ----------
    data : list
        Parsed JSON response — a list of 45-element arrays, one per team.

    Returns
    -------
    pd.DataFrame
        Columns: team, conf, record, adj_em, adj_o, adj_d, adj_t, luck,
                 sos_adj_em, opp_o, opp_d, ncsos_adj_em, rank
    """
    if not data:
        return _empty_df()

    records: list[dict] = []
    for row in data:
        if not isinstance(row, list) or len(row) <= max(_JSON_IDX.values()):
            continue
        try:
            adj_o = float(row[_JSON_IDX["adj_o"]])
            adj_d = float(row[_JSON_IDX["adj_d"]])
            opp_o = float(row[_JSON_IDX["opp_o"]])
            opp_d = float(row[_JSON_IDX["opp_d"]])
            ncsos_opp_o = float(row[_JSON_IDX["ncsos_opp_o"]])
            ncsos_opp_d = float(row[_JSON_IDX["ncsos_opp_d"]])
            records.append({
                "team":          str(row[_JSON_IDX["team"]]),
                "conf":          str(row[_JSON_IDX["conf"]]),
                "record":        str(row[_JSON_IDX["record"]]),
                "adj_o":         adj_o,
                "adj_d":         adj_d,
                "adj_em":        adj_o - adj_d,
                "adj_t":         float(row[_JSON_IDX["adj_t"]]),
                "luck":          float(row[_JSON_IDX["luck"]]),
                "opp_o":         opp_o,
                "opp_d":         opp_d,
                "sos_adj_em":    opp_o - opp_d,
                "ncsos_adj_em":  ncsos_opp_o - ncsos_opp_d,
                "rank":          int(row[_JSON_IDX["rank"]]),
            })
        except (TypeError, ValueError, IndexError):
            continue

    if not records:
        return _empty_df()

    df = pd.DataFrame(records, columns=_COLUMN_ORDER)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pure parser
# ---------------------------------------------------------------------------


def _parse_trank_html(html: str) -> pd.DataFrame:
    """
    Parse raw T-Rank HTML and return a normalised DataFrame.

    Parameters
    ----------
    html : str
        Raw HTML string from barttorvik.com/trank.php.

    Returns
    -------
    pd.DataFrame
        Columns: team, conf, record, adj_em, adj_o, adj_d, adj_t, luck,
                 sos_adj_em, opp_o, opp_d, ncsos_adj_em, rank
    """
    soup = BeautifulSoup(html, "html.parser")

    # Attempt to locate the main rankings table — try id first, fall back to
    # first <table> in the document.
    table = soup.find("table", {"id": "t-rank-table"}) or soup.find("table")

    if table is None:
        return _empty_df()

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []

    if not rows:
        return _empty_df()

    records: list[dict] = []
    for tr in rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        # Guard: need at least 13 columns to be a data row
        if len(cells) < 13:
            continue
        row: dict = {}
        for idx, col_name in _IDX_MAP.items():
            row[col_name] = cells[idx]
        records.append(row)

    if not records:
        return _empty_df()

    df = pd.DataFrame(records, columns=list(_IDX_MAP.values()))

    # Cast types
    for col in _FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in _INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")

    # Reorder to canonical column order and reset index
    df = df[_COLUMN_ORDER].reset_index(drop=True)
    return df


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column schema."""
    return pd.DataFrame(columns=_COLUMN_ORDER)


# ---------------------------------------------------------------------------
# Network layer with exponential backoff
# ---------------------------------------------------------------------------


def _get_with_backoff(url: str, params: dict | None = None, max_retries: int = _MAX_RETRIES) -> requests.Response:
    """
    HTTP GET with exponential backoff on non-2xx responses.

    Sleeps 2^attempt seconds between retries (patched in tests via
    ``src.data.barttorvik.time.sleep``).

    Raises
    ------
    requests.HTTPError
        After *max_retries* consecutive failures.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=_REQUEST_HEADERS, timeout=30)
        if response.ok:
            return response
        last_exc = requests.HTTPError(
            f"HTTP {response.status_code} after attempt {attempt + 1}",
            response=response,
        )
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_trank(season: int) -> pd.DataFrame:
    """
    Fetch the T-Rank table for *season* from barttorvik.com and return a
    normalised DataFrame.

    Strategy (as of 2026-03-12):
    1. Try the direct JSON endpoint ``barttorvik.com/{year}_team_results.json``.
       This endpoint is served without a Cloudflare JS challenge and returns
       data immediately.
    2. If the JSON response cannot be parsed (e.g. the site returns the HTML
       challenge page), fall back to the legacy HTML path so that unit tests
       that mock ``https://www.barttorvik.com/trank.php`` continue to pass.

    Parameters
    ----------
    season : int
        The academic year (e.g. 2024 = 2023-24 season).  Must be >= 2008.

    Returns
    -------
    pd.DataFrame
        Columns: team, conf, record, adj_em, adj_o, adj_d, adj_t, luck,
                 sos_adj_em, opp_o, opp_d, ncsos_adj_em, rank

    Raises
    ------
    ValueError
        If *season* < 2008.
    requests.HTTPError
        After 3 failed HTTP attempts (HTML fallback path only).
    """
    _validate_season(season)

    # --- Attempt 1: direct JSON endpoint (no JS challenge) ---
    json_url = _JSON_BASE_URL.format(year=season)
    try:
        json_response = requests.get(json_url, headers=_REQUEST_HEADERS, timeout=30)
        if json_response.ok:
            raw = json_response.json()
            df = _parse_trank_json(raw)
            if not df.empty:
                return df
    except (requests.RequestException, ValueError):
        # ValueError covers json.JSONDecodeError (site returned HTML instead)
        pass

    # --- Attempt 2: legacy HTML endpoint (used by mocked unit tests) ---
    params = {"year": season}
    response = _get_with_backoff(_BASE_URL, params=params)
    return _parse_trank_html(response.text)


def cache_trank(season: int, cache_dir: str = "data/raw/barttorvik") -> Path:
    """
    Fetch and cache the raw T-Rank HTML for *season* to *cache_dir*.

    Cache-first: if the target file already exists it is returned immediately
    without hitting the network.

    Parameters
    ----------
    season : int
        Must be >= 2008.
    cache_dir : str
        Directory to store raw HTML files.

    Returns
    -------
    Path
        Absolute path to the cached HTML file.

    Raises
    ------
    ValueError
        If *season* < 2008.
    """
    _validate_season(season)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    file_path = cache_path / f"trank_{season}.html"

    if file_path.exists():
        return file_path

    params = {"year": season}
    response = _get_with_backoff(_BASE_URL, params=params)
    file_path.write_text(response.text, encoding="utf-8")
    return file_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_season(season: int) -> None:
    if season < _EARLIEST_SEASON:
        raise ValueError(
            f"season must be >= {_EARLIEST_SEASON}; received {season}."
        )


# ---------------------------------------------------------------------------
# Player-level constants
# ---------------------------------------------------------------------------

_PLAYER_BASE_URL = "https://barttorvik.com/trankf.php"
_CONTINUITY_URL = "https://barttorvik.com/continuity.php"


# ---------------------------------------------------------------------------
# Player-level HTML parsers
# ---------------------------------------------------------------------------


def _parse_player_html(html: str, value_col: str) -> pd.DataFrame:
    """
    Parse a generic Barttorvik player table (PORPAGATU! or BPM layout).

    The first column is assumed to be the player name, the second column is
    the target metric (*value_col*), and the third column is Min%.  Any
    additional columns are included with auto-generated names.

    Parameters
    ----------
    html : str
        Raw HTML from barttorvik.com/trankf.php.
    value_col : str
        Canonical name for the primary metric column (e.g. ``"porpagatu"``
        or ``"bpm"``).

    Returns
    -------
    pd.DataFrame
        Columns: player, <value_col>, minutes_pct[, extra…]
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame(columns=["player", value_col, "minutes_pct"])

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []

    records: list[dict] = []
    for tr in rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 3:
            continue
        row: dict = {
            "player": cells[0],
            value_col: cells[1],
            "minutes_pct": cells[2],
        }
        # Preserve any extra columns with positional names
        for extra_idx in range(3, len(cells)):
            row[f"col_{extra_idx}"] = cells[extra_idx]
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["player", value_col, "minutes_pct"])

    df = pd.DataFrame(records)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    df["minutes_pct"] = pd.to_numeric(df["minutes_pct"], errors="coerce").astype(float)
    return df.reset_index(drop=True)


def _parse_continuity_html(html: str) -> pd.DataFrame:
    """
    Parse the Barttorvik season-continuity table.

    Expects the first column to be team name and the second to be returning
    possession-minutes as a percentage (0–100).  The value is divided by 100
    so that ``returning_pct`` is returned in [0.0, 1.0].

    Returns
    -------
    pd.DataFrame
        Columns: team, returning_pct[, extra…]
    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame(columns=["team", "returning_pct"])

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else []

    records: list[dict] = []
    for tr in rows:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 2:
            continue
        row: dict = {
            "team": cells[0],
            "returning_pct": cells[1],
        }
        for extra_idx in range(2, len(cells)):
            row[f"col_{extra_idx}"] = cells[extra_idx]
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["team", "returning_pct"])

    df = pd.DataFrame(records)
    df["returning_pct"] = pd.to_numeric(df["returning_pct"], errors="coerce").astype(float)

    # Normalise percentage → fraction if values appear to be in 0–100 range
    if df["returning_pct"].dropna().gt(1.0).any():
        df["returning_pct"] = df["returning_pct"] / 100.0

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Retry helper for player-level functions
# ---------------------------------------------------------------------------


def _get_with_backoff_runtime(url: str, params: dict | None = None, max_retries: int = _MAX_RETRIES) -> requests.Response:
    """
    Like _get_with_backoff but raises ``RuntimeError`` (not HTTPError) after
    exhausting retries, as required by the player-level API contract.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        response = requests.get(url, params=params, headers=_REQUEST_HEADERS, timeout=30)
        if response.ok:
            return response
        last_exc = requests.HTTPError(
            f"HTTP {response.status_code} after attempt {attempt + 1}",
            response=response,
        )
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to fetch {url} after {max_retries} attempts."
    ) from last_exc


# ---------------------------------------------------------------------------
# Public API — player-level
# ---------------------------------------------------------------------------


def fetch_porpagatu(
    team: str,
    year: int,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Scrape Barttorvik's PORPAGATU! player table for *team* and *year*.

    URL: https://barttorvik.com/trankf.php?tvalue=<team>&year=<year>&type=porpagatu

    Parameters
    ----------
    team : str
        Team name as it appears in Barttorvik URLs (will be URL-encoded).
    year : int
        Season year (e.g. 2024 = 2023-24).
    cache_dir : Path | None
        Root cache directory.  Results are stored under
        ``{cache_dir}/barttorvik/porpagatu_{team}_{year}.json``.
        Defaults to the current working directory if *None*.

    Returns
    -------
    pd.DataFrame
        Columns: player, porpagatu, minutes_pct[, extra…]

    Raises
    ------
    ValueError
        If the parsed DataFrame is empty (team/year not found).
    RuntimeError
        After 3 failed HTTP attempts.
    """
    cache_path = _player_cache_path(cache_dir, f"porpagatu_{team}_{year}.json")

    if cache_path.exists():
        return pd.read_json(cache_path)

    team_url = quote(team, safe="")
    params = {"tvalue": team_url, "year": year, "type": "porpagatu"}
    response = _get_with_backoff_runtime(_PLAYER_BASE_URL, params=params)

    df = _parse_player_html(response.text, "porpagatu")
    if df.empty:
        raise ValueError(
            f"No PORPAGATU! data found for team='{team}', year={year}."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(df.to_json(), encoding="utf-8")
    return df


def fetch_bpm(
    team: str,
    year: int,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Scrape Barttorvik's BPM (Box Plus-Minus) player table for *team* and *year*.

    URL: https://barttorvik.com/trankf.php?tvalue=<team>&year=<year>&type=bpm

    Parameters
    ----------
    team : str
        Team name (will be URL-encoded).
    year : int
        Season year.
    cache_dir : Path | None
        Root cache directory.  Results stored under
        ``{cache_dir}/barttorvik/bpm_{team}_{year}.json``.

    Returns
    -------
    pd.DataFrame
        Columns: player, bpm, minutes_pct[, extra…]

    Raises
    ------
    ValueError
        If the parsed DataFrame is empty.
    RuntimeError
        After 3 failed HTTP attempts.
    """
    cache_path = _player_cache_path(cache_dir, f"bpm_{team}_{year}.json")

    if cache_path.exists():
        return pd.read_json(cache_path)

    team_url = quote(team, safe="")
    params = {"tvalue": team_url, "year": year, "type": "bpm"}
    response = _get_with_backoff_runtime(_PLAYER_BASE_URL, params=params)

    df = _parse_player_html(response.text, "bpm")
    if df.empty:
        raise ValueError(
            f"No BPM data found for team='{team}', year={year}."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(df.to_json(), encoding="utf-8")
    return df


def fetch_roster_continuity(
    year: int,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Scrape the Barttorvik season continuity/returning-production table for all
    teams.

    URL: https://barttorvik.com/continuity.php?year=<year>

    Parameters
    ----------
    year : int
        Season year.
    cache_dir : Path | None
        Root cache directory.  Results stored under
        ``{cache_dir}/barttorvik/continuity_{year}.json``.

    Returns
    -------
    pd.DataFrame
        Columns: team, returning_pct[, extra…]
        ``returning_pct`` is in [0.0, 1.0].

    Raises
    ------
    ValueError
        If the parsed DataFrame is empty.
    RuntimeError
        After 3 failed HTTP attempts.
    """
    cache_path = _player_cache_path(cache_dir, f"continuity_{year}.json")

    if cache_path.exists():
        return pd.read_json(cache_path)

    params = {"year": year}
    response = _get_with_backoff_runtime(_CONTINUITY_URL, params=params)

    df = _parse_continuity_html(response.text)
    if df.empty:
        raise ValueError(f"No continuity data found for year={year}.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(df.to_json(), encoding="utf-8")
    return df


def compute_team_porpagatu_weighted(player_df: pd.DataFrame) -> float:
    """
    Compute the minutes-weighted mean PORPAGATU! for a team roster.

    Formula: ``sum(porpagatu * minutes_pct) / sum(minutes_pct)``

    Parameters
    ----------
    player_df : pd.DataFrame
        DataFrame returned by :func:`fetch_porpagatu`.  Must contain columns
        ``porpagatu`` and ``minutes_pct``.

    Returns
    -------
    float
        Minutes-weighted mean PORPAGATU! value.

    Raises
    ------
    ValueError
        If ``minutes_pct`` sums to zero.
    """
    total_minutes = player_df["minutes_pct"].sum()
    if total_minutes == 0:
        raise ValueError("minutes_pct sums to 0 — cannot compute weighted mean.")
    return float((player_df["porpagatu"] * player_df["minutes_pct"]).sum() / total_minutes)


# ---------------------------------------------------------------------------
# Internal cache helpers for player-level functions
# ---------------------------------------------------------------------------


def _player_cache_path(cache_dir: Path | None, filename: str) -> Path:
    """
    Resolve the full path for a player-level cache file.

    Files are always placed under ``{cache_dir}/barttorvik/{filename}``.
    """
    base = Path(cache_dir) if cache_dir is not None else Path(".")
    return base / "barttorvik" / filename
