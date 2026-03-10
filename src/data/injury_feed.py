"""
src/data/injury_feed.py

Injury and roster-availability ingestion module.
Primary source: Rotowire CBB injury page (HTML scrape).
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

_ROTOWIRE_CBB_URL = "https://www.rotowire.com/basketball/college/injury-report.php"

_ROTOWIRE_COLUMNS = [
    "player",
    "team",
    "position",
    "status",
    "injury",
    "expected_return",
    "scraped_at",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def fetch_rotowire_injuries(season: int) -> pd.DataFrame:
    """
    Scrape the Rotowire CBB injury report page.

    Parameters
    ----------
    season:
        Four-digit season year (used for metadata; Rotowire always returns
        current injuries).

    Returns
    -------
    pd.DataFrame
        Columns: player, team, position, status, injury,
        expected_return, scraped_at.

    Raises
    ------
    RuntimeError
        If the HTTP response is not 200.
    """
    logger.info(f"Fetching Rotowire CBB injuries (season={season})")
    response = httpx.get(_ROTOWIRE_CBB_URL, headers=_HEADERS, timeout=30.0)

    if response.status_code != 200:
        raise RuntimeError(
            f"Rotowire returned HTTP {response.status_code}: {response.text[:200]}"
        )

    scraped_at = datetime.now(tz=timezone.utc).isoformat()
    rows = _parse_rotowire_html(response.text, scraped_at=scraped_at)

    df = pd.DataFrame(rows, columns=_ROTOWIRE_COLUMNS)
    logger.info(f"Parsed {len(df)} injury entries from Rotowire")
    return df


def build_availability_vector(
    team: str,
    injuries_df: pd.DataFrame,
    bpr_df: pd.DataFrame,
) -> float:
    """
    Compute the fraction of a team's BPR-weighted minutes that are available.

    An availability of 1.0 means full strength; lower values indicate
    that high-BPR players are ruled out.

    Parameters
    ----------
    team:
        Team name to filter both DataFrames.
    injuries_df:
        Injury DataFrame as returned by :func:`fetch_rotowire_injuries`.
        Must contain columns: player, team, status.
    bpr_df:
        BPR DataFrame (from EvanMiya module).
        Must contain columns: player, team, minutes_share.

    Returns
    -------
    float
        Fraction in [0.0, 1.0] representing available BPR minutes.
        Returns 1.0 if no BPR data exists for the team.
    """
    # Filter BPR to this team
    team_bpr = bpr_df[bpr_df["team"] == team].copy()
    if team_bpr.empty:
        return 1.0

    total_minutes_share = team_bpr["minutes_share"].sum()
    if total_minutes_share == 0.0:
        return 1.0

    # Determine which players are injured/out for this team
    team_injuries = injuries_df[injuries_df["team"] == team]
    injured_players: set[str] = set(team_injuries["player"].str.strip().str.lower())

    # Sum minutes_share of injured players
    team_bpr["_name_lower"] = team_bpr["player"].str.strip().str.lower()
    lost_minutes = team_bpr.loc[
        team_bpr["_name_lower"].isin(injured_players), "minutes_share"
    ].sum()

    availability = 1.0 - (lost_minutes / total_minutes_share)
    # Clamp to [0, 1] for safety
    return float(max(0.0, min(1.0, availability)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_rotowire_html(html: str, scraped_at: str) -> list[dict]:
    """
    Parse Rotowire HTML into a list of injury record dicts.

    The page uses a table with class ``injury-report``.  Each ``<tr>``
    child of ``<tbody>`` is one player entry.  Falls back to an empty
    list gracefully when no table is found.
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    table = soup.find("table", class_="injury-report")
    if table is None:
        # Try generic table if class selector doesn't match
        table = soup.find("table")

    if table is None:
        return rows

    tbody = table.find("tbody")
    if tbody is None:
        return rows

    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 6:
            continue
        rows.append(
            {
                "player": _cell_text(cells, 0),
                "team": _cell_text(cells, 1),
                "position": _cell_text(cells, 2),
                "status": _cell_text(cells, 3),
                "injury": _cell_text(cells, 4),
                "expected_return": _cell_text(cells, 5),
                "scraped_at": scraped_at,
            }
        )

    return rows


def _cell_text(cells: list, index: int) -> str:
    """Return stripped text of a BeautifulSoup td cell by index."""
    try:
        return cells[index].get_text(strip=True)
    except (IndexError, AttributeError):
        return ""
