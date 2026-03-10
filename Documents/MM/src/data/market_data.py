"""
src/data/market_data.py

Market data ingestion module.
Sources:
  - Sportsbook Review (SBR) — historical NCAAB opening / closing lines (HTML scrape)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

_SBR_BASE_URL = "https://www.sportsbookreview.com/betting-odds/college-basketball/"

_SBR_DF_COLUMNS = [
    "game_id",
    "home_team",
    "away_team",
    "open_spread",
    "close_spread",
    "open_total",
    "close_total",
    "date",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds between retries


# ---------------------------------------------------------------------------
# Public functions — SBR scraping
# ---------------------------------------------------------------------------


def fetch_sbr_lines(year: int, cache_dir: str | None = None) -> pd.DataFrame:
    """
    Scrape Sportsbook Review NCAAB historical opening/closing lines.

    Parameters
    ----------
    year:
        Season year (e.g. 2024).
    cache_dir:
        Optional directory path for caching results.  When provided the
        scraped payload is written to
        ``{cache_dir}/market_data/sbr_{year}.json`` and re-read on
        subsequent calls (same pattern as barttorvik.py).

    Returns
    -------
    pd.DataFrame
        Columns: game_id, home_team, away_team, open_spread,
        close_spread, open_total, close_total, date.
        Returns an empty DataFrame (with the correct columns) when SBR
        provides no parseable table — does *not* raise in that case.

    Raises
    ------
    RuntimeError
        On network failure after :data:`_MAX_RETRIES` attempts.
    """
    # ------------------------------------------------------------------
    # Cache hit
    # ------------------------------------------------------------------
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "market_data" / f"sbr_{year}.json"
        if cache_path.exists():
            logger.info(f"Loading SBR lines from cache: {cache_path}")
            with cache_path.open() as fh:
                records = json.load(fh)
            return pd.DataFrame(records, columns=_SBR_DF_COLUMNS)

    # ------------------------------------------------------------------
    # Fetch with retries
    # ------------------------------------------------------------------
    url = _SBR_BASE_URL
    params: dict[str, Any] = {"league": "ncaab", "season": str(year)}

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                f"Fetching SBR NCAAB lines (year={year}, attempt={attempt})"
            )
            response = requests.get(
                url, headers=_HEADERS, params=params, timeout=30.0
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                f"SBR request failed (attempt {attempt}/{_MAX_RETRIES}): {exc}"
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF)
    else:
        raise RuntimeError(
            f"Failed to fetch SBR lines after {_MAX_RETRIES} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Parse HTML
    # ------------------------------------------------------------------
    records = _parse_sbr_html(response.text, year=year)

    if not records:
        logger.info(f"No SBR table found for year={year}; returning empty DataFrame")
        return pd.DataFrame(columns=_SBR_DF_COLUMNS)

    # ------------------------------------------------------------------
    # Cache write
    # ------------------------------------------------------------------
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as fh:
            json.dump(records, fh)
        logger.info(f"Cached SBR lines to {cache_path}")

    df = pd.DataFrame(records, columns=_SBR_DF_COLUMNS)
    logger.info(f"Parsed {len(df)} SBR lines rows for year={year}")
    return df


# ---------------------------------------------------------------------------
# Public functions — CLV & odds math
# ---------------------------------------------------------------------------


def compute_clv(open_line: float, close_line: float, bet_side: str) -> float:
    """
    Compute Closing Line Value (CLV) for a spread bet.

    CLV measures whether you beat the closing number.
    Positive CLV means the line moved in the bettor's favour
    (you got a better number than the close).

    For the *home* side:
        CLV = close_line − open_line
        (a lower/less-negative close than open is better for the home side)

    For the *away* side:
        CLV = open_line − close_line
        (a lower/more-negative open line is better for the away side)

    Parameters
    ----------
    open_line:
        Opening spread for the home team (e.g., ``-3.5`` means home
        is a 3.5-point favourite at open).
    close_line:
        Closing spread for the home team.
    bet_side:
        ``"home"`` or ``"away"``.

    Returns
    -------
    float
        CLV value; positive = beat the closing line.

    Raises
    ------
    ValueError
        If ``bet_side`` is not ``"home"`` or ``"away"``.
    """
    if bet_side == "home":
        return float(close_line - open_line)
    elif bet_side == "away":
        return float(open_line - close_line)
    else:
        raise ValueError(
            f"bet_side must be 'home' or 'away', got {bet_side!r}. "
        )


def american_to_prob(american_odds: int) -> float:
    """
    Convert American odds to vig-inclusive implied probability.

    Parameters
    ----------
    american_odds:
        American-format odds (e.g., ``-110``, ``+150``).
        Must not be zero.

    Returns
    -------
    float
        Implied probability in ``(0, 1)``.

    Raises
    ------
    ValueError
        If ``american_odds`` is 0 (invalid).
    """
    if american_odds == 0:
        raise ValueError("american_odds cannot be 0 — invalid odds value.")

    if american_odds < 0:
        return float(-american_odds / (-american_odds + 100))
    else:
        return float(100 / (american_odds + 100))


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """
    Remove bookmaker vig via additive (sum-to-one) normalisation.

    Parameters
    ----------
    home_prob:
        Vig-inclusive implied probability for the home side.
    away_prob:
        Vig-inclusive implied probability for the away side.

    Returns
    -------
    tuple[float, float]
        ``(home_prob_clean, away_prob_clean)`` that sum to exactly 1.0.

    Raises
    ------
    ValueError
        If the total implied probability is zero (both inputs are 0.0).
    """
    total = home_prob + away_prob
    if total == 0.0:
        raise ValueError(
            "Total implied probability is 0.0 — both home_prob and away_prob "
            "cannot be zero."
        )
    return float(home_prob / total), float(away_prob / total)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_sbr_html(html: str, year: int) -> list[dict]:
    """
    Parse SBR HTML into a list of flat line record dicts.

    SBR does not expose a structured JSON API; this function tries to
    locate any ``<table>`` on the page and map columns to the canonical
    schema.  Returns an empty list when no table is found.
    """
    soup = BeautifulSoup(html, "lxml")
    records: list[dict] = []

    table = soup.find("table")
    if table is None:
        return records

    # Attempt to identify header columns
    header_cells = table.find_all("th")
    headers = [th.get_text(strip=True).lower() for th in header_cells]

    tbody = table.find("tbody")
    if tbody is None:
        return records

    for idx, tr in enumerate(tbody.find_all("tr")):
        cells = tr.find_all("td")
        if not cells:
            continue

        # Build a raw row dict keyed by header (best-effort)
        raw: dict[str, str] = {}
        for col_idx, td in enumerate(cells):
            key = headers[col_idx] if col_idx < len(headers) else str(col_idx)
            raw[key] = td.get_text(strip=True)

        # Map to canonical columns (graceful fallback to None)
        records.append(
            {
                "game_id": f"sbr_{year}_{idx}",
                "home_team": _pick(raw, ["home", "home team", "home_team"]),
                "away_team": _pick(raw, ["away", "away team", "away_team", "visitor"]),
                "open_spread": _to_float(_pick(raw, ["open", "open spread", "open_spread"])),
                "close_spread": _to_float(_pick(raw, ["close", "close spread", "close_spread", "closing"])),
                "open_total": _to_float(_pick(raw, ["open total", "open_total", "ou open"])),
                "close_total": _to_float(_pick(raw, ["close total", "close_total", "ou close"])),
                "date": _pick(raw, ["date", "game date", "game_date"]),
            }
        )

    return records


def _pick(row: dict[str, str], candidates: list[str]) -> str | None:
    """Return the first matching key value from *row*, or None."""
    for key in candidates:
        if key in row:
            return row[key] or None
    return None


def _to_float(value: str | None) -> float | None:
    """Convert a string to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
