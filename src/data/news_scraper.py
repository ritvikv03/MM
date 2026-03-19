"""
src/data/news_scraper.py

Phase 4: Continuous "Information Asymmetry" Scraper
Scrapes Rotowire CBB injury page for real roster intelligence.
Per CLAUDE.md §16, no synthetic data is returned; on failure the function
returns an empty list and logs a warning.

Primary source: Rotowire CBB injuries (sole injury source per CLAUDE.md §3.4)
"""
from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_ROTOWIRE_URL = "https://www.rotowire.com/basketball/ncaa-injuries.php"
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.rotowire.com/",
}

ALERT_KEYWORDS = [
    "out",
    "doubtful",
    "questionable",
    "day-to-day",
    "injured",
    "suspension",
    "suspended",
    "concussion",
    "ankle",
    "knee",
    "foot",
    "fracture",
    "mri",
    "torn",
]


def _stable_alert_id(team: str, player: str, status: str) -> str:
    """Deterministic, dedup-safe ID — hash of team+player+status."""
    raw = f"{team}::{player}::{status}".lower()
    return "rotowire_" + hashlib.sha1(raw.encode()).hexdigest()[:12]


def _scrape_rotowire() -> list[dict[str, Any]]:
    """
    Scrape the Rotowire NCB injury page and return structured alert dicts.
    Returns an empty list (never raises) so callers are not blocked.
    """
    try:
        resp = requests.get(_ROTOWIRE_URL, headers=_REQUEST_HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Rotowire fetch failed: %s", exc)
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    alerts: list[dict[str, Any]] = []
    now = time.time()

    # Rotowire injury table rows have class "injury-report__table-row"
    # Each row contains: team, player, position, injury, status, est. return
    rows = soup.select(".injury-report__table-row")
    if not rows:
        # Fallback: try generic table rows inside the injury section
        rows = soup.select("ul.injury-report__table li")

    for row in rows:
        team_el = row.select_one(".injury-report__team-name, .team-name")
        player_el = row.select_one(".injury-report__player-name, .player-name")
        injury_el = row.select_one(".injury-report__injury, .injury")
        status_el = row.select_one(".injury-report__status, .status")

        team = team_el.get_text(strip=True) if team_el else ""
        player = player_el.get_text(strip=True) if player_el else ""
        injury = injury_el.get_text(strip=True) if injury_el else ""
        status = status_el.get_text(strip=True) if status_el else ""

        if not player or not team:
            continue

        content = f"{player} ({team}) — {injury} — {status}"
        keywords_found = [kw for kw in ALERT_KEYWORDS if kw in content.lower()]
        severity = _classify_severity(status)

        alert: dict[str, Any] = {
            "alert_id":           _stable_alert_id(team, player, status),
            "source":             "Rotowire",
            "content":            content,
            "keywords":           keywords_found,
            "severity":           severity,
            "teams_mentioned":    [team] if team else [],
            "needs_verification": severity in ("high", "critical"),
            "url":                _ROTOWIRE_URL,
            "alerted_at":         now,
            "resolved":           False,
        }
        alerts.append(alert)

    logger.info("Rotowire scrape: %d injury alerts parsed", len(alerts))
    return alerts


def _classify_severity(status: str) -> str:
    """Map Rotowire status string to our severity tiers."""
    s = status.lower()
    if any(w in s for w in ("out", "season", "suspension")):
        return "critical"
    if "doubtful" in s:
        return "high"
    if "questionable" in s:
        return "medium"
    return "low"


def scrape_intel_alerts() -> list[dict[str, Any]]:
    """
    Public entry-point called by PipelineRunner._run_intel().

    Scrapes Rotowire for real CBB injury/roster intelligence.
    Returns a list of alert dicts ready for SupabaseWriter.insert_intel_alert().
    Returns [] on any network or parse failure — never raises.
    """
    return _scrape_rotowire()
