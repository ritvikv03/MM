"""
src/data/ncaa_roster_feed.py
Fetch active NCAA D-I rosters from Sports Reference and produce a clean JSON
feed that the frontend can consume for the news filtering gatekeeper.

Usage (CLI):
    python -m src.data.ncaa_roster_feed --season 2026 --output artifacts/ncaa_rosters.json

Output JSON shape:
{
  "season": 2026,
  "fetched_at": "<ISO timestamp>",
  "players": [
    {"name": "...", "team": "...", "class": "...", "position": "..."},
    ...
  ],
  "teams": ["Duke", "Florida", ...],
  "nba_draft_class": ["Cooper Flagg", ...],   // players who left for NBA
  "data_source": "sports-reference"
}
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.sports-reference.com/cbb"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MadnessMatrix/2.0; educational research tool)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_RATE_LIMIT_SECONDS = 3.0   # polite delay between requests


# ──────────────────────────────────────────────────────────────────────────────
# Scraping helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_soup(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a URL with exponential backoff and return a BeautifulSoup object."""
    delay = _RATE_LIMIT_SECONDS
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = delay * (2 ** attempt)
                logger.warning("Rate limited on %s — waiting %.1fs", url, wait)
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            logger.warning("HTTP %s for %s", resp.status_code, url)
            return None
        except Exception as exc:
            logger.error("Request failed (%s/%s): %s", attempt + 1, retries, exc)
            time.sleep(delay * (2 ** attempt))
    return None


def _fetch_tournament_teams(season: int) -> list[str]:
    """Scrape the list of NCAA tournament teams for a given season."""
    url = f"{_BASE_URL}/seasons/men/{season}-ncaa.html"
    soup = _get_soup(url)
    if soup is None:
        logger.error("Could not fetch tournament teams for season %s", season)
        return []

    teams = []
    # Look for team links in the tournament bracket table
    for a in soup.select("table a[href*='/cbb/schools/']"):
        team_name = a.get_text(strip=True)
        if team_name and team_name not in teams:
            teams.append(team_name)
        if len(teams) >= 68:
            break

    logger.info("Fetched %d tournament teams for season %s", len(teams), season)
    return teams


def _fetch_team_roster(team_slug: str, season: int) -> list[dict[str, str]]:
    """Fetch the roster for a single team from Sports Reference."""
    url = f"{_BASE_URL}/schools/{team_slug}/{season}.html"
    soup = _get_soup(url)
    if soup is None:
        return []

    players = []
    roster_table = soup.find("table", {"id": "roster"})
    if roster_table is None:
        return players

    for row in roster_table.select("tbody tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 3:
            continue
        name_cell = row.find("td", {"data-stat": "player"})
        class_cell = row.find("td", {"data-stat": "class"})
        pos_cell = row.find("td", {"data-stat": "pos"})
        if name_cell:
            players.append({
                "name": name_cell.get_text(strip=True),
                "class": class_cell.get_text(strip=True) if class_cell else "",
                "position": pos_cell.get_text(strip=True) if pos_cell else "",
            })

    time.sleep(_RATE_LIMIT_SECONDS)  # polite rate limiting
    return players


def _fetch_nba_draft_class(season: int) -> list[str]:
    """Fetch the list of college players who declared for the NBA Draft after the season."""
    # NBA Draft class = players who left after season-1 tournament
    draft_year = season  # e.g., 2026 → 2026 NBA Draft
    url = f"https://www.sports-reference.com/friv/colleges.fcgi?draft_year={draft_year}"
    soup = _get_soup(url)
    if soup is None:
        return []

    nba_players = []
    for row in soup.select("table#stats tbody tr"):
        name_cell = row.find("td", {"data-stat": "player"})
        if name_cell and name_cell.get_text(strip=True):
            nba_players.append(name_cell.get_text(strip=True))

    logger.info("Fetched %d NBA draft declarations for %s", len(nba_players), draft_year)
    return nba_players


# ──────────────────────────────────────────────────────────────────────────────
# News filtering gatekeeper
# ──────────────────────────────────────────────────────────────────────────────

def filter_news_items(
    news_items: list[dict[str, Any]],
    active_players: set[str],
    nba_departed: set[str],
    active_teams: set[str],
) -> list[dict[str, Any]]:
    """Filter news items to only those relevant to active college basketball players.

    A news item is EXCLUDED if:
    1. It mentions a player who has departed for the NBA (nba_departed set).
    2. It mentions no active D-I teams or active players at all.
    3. The item's 'team' field (if present) is not in the active tournament teams.

    A news item PASSES if:
    - It mentions an active college player OR an active tournament team.
    - It does not name a departed NBA player.

    Parameters
    ----------
    news_items:
        List of news dicts with keys: title, body, team (optional), player (optional).
    active_players:
        Set of currently active college player names (lowercase for matching).
    nba_departed:
        Set of players who have left for the NBA (filter these out).
    active_teams:
        Set of active NCAA tournament team names.

    Returns
    -------
    Filtered list of news items relevant to active college players.
    """
    filtered = []
    nba_lower = {p.lower() for p in nba_departed}
    active_lower = {p.lower() for p in active_players}
    teams_lower = {t.lower() for t in active_teams}

    for item in news_items:
        title_lower = item.get("title", "").lower()
        body_lower = item.get("body", "").lower()
        combined = f"{title_lower} {body_lower}"

        # Check if any NBA-departed player is mentioned → EXCLUDE
        mentions_departed = any(name in combined for name in nba_lower)
        if mentions_departed:
            logger.debug("Filtered out news item mentioning departed player: %s", item.get("title", "")[:60])
            continue

        # Check if mentions an active team
        mentions_active_team = any(team in combined for team in teams_lower)

        # Check if mentions an active player by name
        mentions_active_player = any(player in combined for player in active_lower)

        # Direct team field match
        item_team = item.get("team", "").lower()
        team_field_match = item_team in teams_lower if item_team else False

        if mentions_active_team or mentions_active_player or team_field_match:
            filtered.append(item)

    logger.info(
        "News filter: %d/%d items passed (%.0f%% retention)",
        len(filtered), len(news_items),
        100 * len(filtered) / max(len(news_items), 1),
    )
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Main build function
# ──────────────────────────────────────────────────────────────────────────────

def build_roster_feed(season: int = 2026, output_path: Path | None = None) -> dict[str, Any]:
    """Fetch active rosters and produce a clean JSON feed.

    This is the primary entry point for CLI and pipeline usage.
    """
    logger.info("Building NCAA roster feed for season %s", season)

    # Fetch tournament teams
    teams = _fetch_tournament_teams(season)
    if not teams:
        logger.warning("No tournament teams found — using 2026 bracket fallback")
        # Fallback to known 2026 bracket teams
        from src.api.bracket_2026 import get_bracket_teams_flat
        teams = get_bracket_teams_flat()

    # Fetch NBA draft departed players (who left the previous season)
    nba_departed = _fetch_nba_draft_class(season)

    # Collect player rosters for tournament teams (limited sample for performance)
    all_players: list[dict[str, str]] = []
    # Note: Full roster scraping is rate-limited. We fetch up to 30 teams in pipeline runs.
    max_teams_to_scrape = min(len(teams), 30)
    for i, team in enumerate(teams[:max_teams_to_scrape]):
        # Convert team name to slug (simplified — real impl uses name→slug mapping)
        slug = team.lower().replace(" ", "-").replace("'", "").replace(".", "")
        roster = _fetch_team_roster(slug, season)
        for player in roster:
            player["team"] = team
        all_players.extend(roster)
        logger.debug("Fetched %d players for %s (%d/%d)", len(roster), team, i + 1, max_teams_to_scrape)

    active_player_names = [p["name"] for p in all_players]

    result: dict[str, Any] = {
        "season": season,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "players": all_players,
        "teams": teams,
        "nba_draft_class": nba_departed,
        "active_player_count": len(active_player_names),
        "data_source": "sports-reference",
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        logger.info("Roster feed written to %s", output_path)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fetch active NCAA rosters for news filtering")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ncaa_rosters.json"))
    args = parser.parse_args()

    feed = build_roster_feed(season=args.season, output_path=args.output)
    print(f"Done — {feed['active_player_count']} active players, {len(feed['teams'])} teams, {len(feed['nba_draft_class'])} NBA departures")
