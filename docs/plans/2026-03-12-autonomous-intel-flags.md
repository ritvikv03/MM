# Autonomous Intel Flag System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a live, continuously-updating Intel Feed that automatically scrapes injuries, roster changes (e.g., NBA Draft departures like Cooper Flagg), and news from free public sources — then surfaces alerts by team/region in the frontend with "See All" filtering.

**Architecture:** Four authoritative scrapers + one unverified signal feed into a unified `IntelAggregator`: (1) **Rotowire CBB** — injuries (authoritative), (2) **ESPN CBB Injuries** — cross-check verification, (3) **NBA.com Early Entry list** — official NBA Draft departures, (4) **On3 Transfer Portal** — transfer moves, (5) **Reddit r/CollegeBasketball** — early rumors only, flagged `needs_verification=True`. An APScheduler background job runs every 15 minutes and writes to `data/intel_cache.json`. A `/api/intel` FastAPI endpoint reads the cache with region/team filtering. The frontend `IntelFeed.tsx` polls every 60s; unverified Reddit alerts display a "UNVERIFIED" badge.

**Tech Stack:** `httpx`, `beautifulsoup4`, `APScheduler>=3.10`, `responses` (mocking), FastAPI, React/Next.js, Zod

---

## Context for Implementer

### Existing Files (read before modifying)
- `src/data/news_scraper.py` — stub `InformationAsymmetryScraper` class; has `ALERT_KEYWORDS` list and `analyze_texts_for_asymmetry()`; all fetch methods are stubs returning hardcoded data
- `src/data/injury_feed.py` — `fetch_rotowire_injuries(season)` scrapes `https://www.rotowire.com/basketball/college/injury-report.php`; `_parse_rotowire_html()` parses table
- `src/api/server.py` — FastAPI app; has CORS middleware; module-level `_data_loader = DataLoader()`; pattern: `os.getenv("USE_REAL_DATA","")` check before real vs. stub
- `src/api/schemas.py` — Pydantic models; all use `Literal["real","stub"]` for `data_source`
- `frontend/lib/api-types.ts` — Zod schemas mirroring Pydantic
- `frontend/lib/api.ts` — fetch helpers; calls `http://localhost:8000`
- `frontend/components/nav/Sidebar.tsx` — `NavPage` type; icons defined inline as SVG components
- `frontend/app/page.tsx` — main page; renders page based on `activePage` state

### Source Reliability Hierarchy
| Source | URL | Reliability | What it covers |
|---|---|---|---|
| Rotowire CBB | `rotowire.com/basketball/college/injury-report.php` | ★★★★★ Authoritative | Injuries |
| ESPN CBB | `espn.com/mens-college-basketball/injuries` | ★★★★★ Authoritative | Injuries (cross-check) |
| NBA.com Early Entry | `nba.com/draft/2026/early-entry` (HTML scrape) | ★★★★★ Official | NBA Draft departures |
| On3 Transfer Portal | `on3.com/transfer-portal/sport/basketball/` | ★★★★☆ Reliable | Transfer portal moves |
| Sports Reference CBB | `sports-reference.com/cbb/schools/{slug}/{year}.html` | ★★★★★ Factual | Roster comparison |
| Reddit r/CBB | `reddit.com/r/CollegeBasketball/new.json` | ★★☆☆☆ Unverified | Early rumors only |

### Key Patterns
- **NBA.com Early Entry**: `GET https://www.nba.com/draft/2026/early-entry` — HTML page; parse `<table>` or `<div class="early-entry-list">` for player names and colleges. Scrape with BeautifulSoup + httpx. Each entry = official NBA Draft departure.
- **ESPN CBB Injuries**: `GET https://www.espn.com/mens-college-basketball/injuries` — HTML; parse injury table rows: player name, team, status, comment. Used to cross-validate Rotowire.
- **On3 Transfer Portal**: `GET https://www.on3.com/transfer-portal/sport/basketball/` — HTML table of active portal entries; each row has player, from-school, to-school, position, status.
- **Reddit public JSON**: `GET https://www.reddit.com/r/CollegeBasketball/new.json?limit=50` — no auth; `needs_verification=True` on all Reddit alerts.
- **Sports Reference CBB roster**: `https://www.sports-reference.com/cbb/schools/{slug}/2026.html` — HTML table id `roster`
- APScheduler in server.py: use `BackgroundScheduler`, call `.start()` after app init, job function reads/writes `data/intel_cache.json`
- Severity = max SEVERITY_WEIGHTS value for keywords found (0.0–1.0)
- `SEVERITY_WEIGHTS` is in `src/model/encoders/sentiment_encoder.py`: `{"walking boot": 0.70, "torn": 1.00, "concussion": 0.85, "suspension": 0.75, "suspended": 0.75, "sprain": 0.60, "questionable": 0.40, "mri": 0.65, "out for season": 1.00, "transfer portal": 0.50, "nba draft": 0.80, "declared": 0.70, "not at practice": 0.55, "limited": 0.30}`
- Region mapping: `{"East": ["ACC", "Big East"], "West": ["Pac12", "WCC", "MWC"], "South": ["SEC", "CUSA"], "Midwest": ["Big10", "Big12", "MVC"]}` — any team not mapped gets "Unknown"
- Team name extraction: match any team name from `_TEAMS_BY_CONF` keys in server.py against alert text (case-insensitive)

### What Exists vs. What's Needed
- `fetch_rotowire_injuries()` → **works** (real scraper); needs severity scoring added in aggregator
- `InformationAsymmetryScraper.fetch_reddit_cbb_new()` → **stub**; replace with real Reddit JSON + `needs_verification=True`
- `InformationAsymmetryScraper.fetch_twitter_reporters()` → **stub**; replace with ESPN + On3 + NBA.com scrapers
- New: `src/data/roster_validator.py` — Sports Reference roster diff + NBA.com early entry list
- New: `src/api/intel.py` — aggregator + `/api/intel` endpoint
- New: `src/api/schemas.py` additions — `IntelAlert` (with `needs_verification` field), `IntelResponse`
- New: `frontend/components/intel/IntelFeed.tsx`
- Modify: `frontend/lib/api-types.ts`, `frontend/lib/api.ts`, `frontend/components/nav/Sidebar.tsx`, `frontend/app/page.tsx`

---

## Task 1: Roster Validator (`src/data/roster_validator.py`)

Detects NBA Draft / transfer portal departures by scraping Sports Reference CBB roster pages.

**Files:**
- Create: `src/data/roster_validator.py`
- Create: `tests/data/test_roster_validator.py`

### Step 1: Write the failing test

```python
# tests/data/test_roster_validator.py
import responses as resp_lib
import pytest
from src.data.roster_validator import (
    parse_roster_html,
    detect_departures,
    build_roster_alerts,
    RosterAlert,
)

ROSTER_HTML_2025 = """
<html><body>
<table id="roster">
  <thead><tr><th>Player</th><th>Cl.</th><th>Pos</th></tr></thead>
  <tbody>
    <tr><td><a href="/cbb/players/cooper-flagg-1.html">Cooper Flagg</a></td><td>FR</td><td>F</td></tr>
    <tr><td><a href="/cbb/players/tyrese-proctor-1.html">Tyrese Proctor</a></td><td>SO</td><td>G</td></tr>
  </tbody>
</table>
</body></html>
"""

ROSTER_HTML_2026 = """
<html><body>
<table id="roster">
  <thead><tr><th>Player</th><th>Cl.</th><th>Pos</th></tr></thead>
  <tbody>
    <tr><td><a href="/cbb/players/tyrese-proctor-1.html">Tyrese Proctor</a></td><td>JR</td><td>G</td></tr>
  </tbody>
</table>
</body></html>
"""


def test_parse_roster_html_returns_player_list():
    players = parse_roster_html(ROSTER_HTML_2025)
    assert len(players) == 2
    assert "Cooper Flagg" in players
    assert "Tyrese Proctor" in players


def test_parse_roster_html_empty_table():
    players = parse_roster_html("<html><body><p>No table</p></body></html>")
    assert players == []


def test_detect_departures_finds_missing_player():
    prev = ["Cooper Flagg", "Tyrese Proctor"]
    curr = ["Tyrese Proctor"]
    departed = detect_departures(prev, curr)
    assert "Cooper Flagg" in departed
    assert "Tyrese Proctor" not in departed


def test_detect_departures_empty_inputs():
    assert detect_departures([], []) == []
    assert detect_departures(["Alice"], ["Alice"]) == []


def test_build_roster_alerts_returns_dataclass():
    alerts = build_roster_alerts("Duke", ["Cooper Flagg"], season=2026)
    assert len(alerts) == 1
    a = alerts[0]
    assert isinstance(a, RosterAlert)
    assert a.team == "Duke"
    assert "Cooper Flagg" in a.player
    assert a.severity >= 0.5


@resp_lib.activate
def test_build_roster_alerts_http_fetch():
    resp_lib.add(resp_lib.GET,
        "https://www.sports-reference.com/cbb/schools/duke/2025.html",
        body=ROSTER_HTML_2025, status=200)
    resp_lib.add(resp_lib.GET,
        "https://www.sports-reference.com/cbb/schools/duke/2026.html",
        body=ROSTER_HTML_2026, status=200)
    from src.data.roster_validator import fetch_team_departures
    alerts = fetch_team_departures("Duke", "duke", current_season=2026)
    assert any("Cooper Flagg" in a.player for a in alerts)
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/data/test_roster_validator.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.roster_validator'`

### Step 3: Write implementation

```python
# src/data/roster_validator.py
"""
src/data/roster_validator.py

Detects player departures (NBA Draft, transfer portal) by comparing
Sports Reference CBB roster pages across consecutive seasons.

Free source: https://www.sports-reference.com/cbb/schools/{slug}/{year}.html
No API key required.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SR_ROSTER_URL = "https://www.sports-reference.com/cbb/schools/{slug}/{year}.html"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
_REQUEST_DELAY = 2.0   # seconds between requests (polite scraping)


@dataclass
class RosterAlert:
    team: str
    player: str
    source: str       # "roster_validator"
    title: str        # human-readable title
    severity: float   # 0.0–1.0
    url: str
    timestamp: float  # Unix timestamp


def parse_roster_html(html: str) -> list[str]:
    """Return list of player names from a Sports Reference CBB roster HTML page."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="roster")
    if table is None:
        return []
    players: list[str] = []
    tbody = table.find("tbody")
    if tbody is None:
        return []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if not cells:
            continue
        # First cell is player name; may be wrapped in <a>
        name = cells[0].get_text(strip=True)
        if name:
            players.append(name)
    return players


def detect_departures(prev_roster: list[str], curr_roster: list[str]) -> list[str]:
    """Return names in prev_roster but not in curr_roster (departed players)."""
    prev_set = {n.strip().lower() for n in prev_roster}
    curr_set = {n.strip().lower() for n in curr_roster}
    departed_lower = prev_set - curr_set
    # Return original-casing names from prev_roster
    return [n for n in prev_roster if n.strip().lower() in departed_lower]


def build_roster_alerts(
    team: str,
    departed_players: list[str],
    season: int,
) -> list[RosterAlert]:
    """Convert a list of departed player names into RosterAlert objects."""
    alerts: list[RosterAlert] = []
    for player in departed_players:
        alerts.append(RosterAlert(
            team=team,
            player=player,
            source="roster_validator",
            title=f"⚠ {player} no longer on {team} roster for {season}",
            severity=0.80,   # departure is high-impact
            url=f"https://www.sports-reference.com/cbb/schools/",
            timestamp=time.time(),
        ))
    return alerts


def _fetch_roster(slug: str, year: int) -> list[str]:
    """Fetch and parse a roster from Sports Reference. Returns [] on failure."""
    url = _SR_ROSTER_URL.format(slug=slug, year=year)
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=20.0, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning("SR roster fetch %s → HTTP %s", url, resp.status_code)
            return []
        return parse_roster_html(resp.text)
    except Exception as exc:
        logger.warning("SR roster fetch failed (%s): %s", url, exc)
        return []


def fetch_team_departures(
    team_name: str,
    team_slug: str,
    current_season: int,
) -> list[RosterAlert]:
    """
    Compare previous and current rosters for a team.

    Parameters
    ----------
    team_name:
        Display name (e.g. "Duke")
    team_slug:
        Sports Reference URL slug (e.g. "duke", "north-carolina")
    current_season:
        The season year being checked (e.g. 2026)
    """
    prev_year = current_season - 1
    prev_roster = _fetch_roster(team_slug, prev_year)
    time.sleep(_REQUEST_DELAY)
    curr_roster = _fetch_roster(team_slug, current_season)

    if not prev_roster:
        logger.info("No prior roster found for %s %s", team_name, prev_year)
        return []

    departed = detect_departures(prev_roster, curr_roster)
    if not departed:
        return []

    logger.info("%s: %d departed players detected", team_name, len(departed))
    return build_roster_alerts(team_name, departed, current_season)
```

### Step 4: Run tests

```bash
python -m pytest tests/data/test_roster_validator.py -v
```
Expected: All 6 tests PASS.

### Step 5: Commit

```bash
git add src/data/roster_validator.py tests/data/test_roster_validator.py
git commit -m "feat(intel): roster departure detector via Sports Reference CBB"
```

---

## Task 2: Multi-Source News Scraper (replace stubs in `news_scraper.py`)

Replace the stub scrapers with four real, authoritative sources. ESPN + Rotowire + NBA.com + On3 are primary (verified); Reddit is secondary (unverified, `needs_verification=True`). Each source method is independent so failures don't cascade.

**Files:**
- Modify: `src/data/news_scraper.py`
- Create: `tests/data/test_news_scraper.py`

### Step 1: Write the failing tests

```python
# tests/data/test_news_scraper.py
import time
import responses as resp_lib
import pytest
from src.data.news_scraper import (
    InformationAsymmetryScraper,
    ALERT_KEYWORDS,
    score_severity,
    extract_team_mentions,
    KNOWN_TEAMS,
    parse_espn_injuries_html,
    parse_on3_transfer_html,
    parse_nba_early_entry_html,
)

# --- Fixtures ---------------------------------------------------------------

REDDIT_JSON = {
    "data": {"children": [
        {"data": {"title": "Kansas guard in walking boot at practice",
                  "permalink": "/r/CollegeBasketball/def", "created_utc": time.time() - 300}},
        {"data": {"title": "Post-game discussion: regular stuff",
                  "permalink": "/r/CollegeBasketball/ghi", "created_utc": time.time() - 600}},
    ]}
}

ESPN_HTML = """<html><body>
<table class="Table">
  <tbody>
    <tr><td><a>Cooper Flagg</a></td><td>Duke</td><td>F</td><td>Out</td><td>Knee</td><td>TBD</td></tr>
    <tr><td><a>Zach Edey</a></td><td>Purdue</td><td>C</td><td>Questionable</td><td>Ankle</td><td>Game-time</td></tr>
  </tbody>
</table>
</body></html>"""

ON3_HTML = """<html><body>
<table class="transfer-portal">
  <tbody>
    <tr><td>Marcus Adams</td><td>Duke</td><td>Arizona</td><td>G</td><td>Committed</td></tr>
    <tr><td>Pending Player</td><td>Kansas</td><td></td><td>F</td><td>Undecided</td></tr>
  </tbody>
</table>
</body></html>"""

NBA_HTML = """<html><body>
<div class="early-entry-list">
  <ul>
    <li>Cooper Flagg, Duke University</li>
    <li>Tre Johnson, Texas Longhorns</li>
  </ul>
</div>
</body></html>"""

# --- Unit tests -------------------------------------------------------------

def test_known_teams_is_populated():
    assert len(KNOWN_TEAMS) >= 20
    assert "Duke" in KNOWN_TEAMS
    assert "Kansas" in KNOWN_TEAMS


def test_score_severity_torn():
    assert score_severity(["torn"]) == 1.00

def test_score_severity_walking_boot():
    assert abs(score_severity(["walking boot"]) - 0.70) < 0.01

def test_score_severity_empty():
    assert score_severity([]) == 0.0

def test_score_severity_multiple_takes_max():
    assert abs(score_severity(["sprain", "mri"]) - 0.65) < 0.01


def test_extract_team_mentions_single():
    assert "Duke" in extract_team_mentions("Duke star exits with injury")

def test_extract_team_mentions_multiple():
    teams = extract_team_mentions("Kansas and Baylor both advancing")
    assert "Kansas" in teams
    assert "Baylor" in teams

def test_extract_team_mentions_none():
    assert extract_team_mentions("The game was exciting") == []


def test_parse_espn_injuries_returns_rows():
    rows = parse_espn_injuries_html(ESPN_HTML)
    assert len(rows) == 2
    assert rows[0]["player"] == "Cooper Flagg"
    assert rows[0]["team"] == "Duke"
    assert rows[0]["source"] == "ESPN"

def test_parse_espn_injuries_empty_html():
    assert parse_espn_injuries_html("<html><body></body></html>") == []


def test_parse_on3_transfer_returns_rows():
    rows = parse_on3_transfer_html(ON3_HTML)
    assert len(rows) >= 1
    assert rows[0]["player"] == "Marcus Adams"
    assert rows[0]["from_school"] == "Duke"
    assert rows[0]["source"] == "On3"

def test_parse_on3_transfer_empty_html():
    assert parse_on3_transfer_html("<html><body></body></html>") == []


def test_parse_nba_early_entry_returns_players():
    players = parse_nba_early_entry_html(NBA_HTML)
    assert "Cooper Flagg" in players
    assert "Tre Johnson" in players

def test_parse_nba_early_entry_empty_html():
    assert parse_nba_early_entry_html("<html><body></body></html>") == []


@resp_lib.activate
def test_fetch_reddit_returns_posts():
    resp_lib.add(resp_lib.GET,
        "https://www.reddit.com/r/CollegeBasketball/new.json",
        json=REDDIT_JSON, status=200)
    scraper = InformationAsymmetryScraper()
    posts = scraper.fetch_reddit_cbb_new()
    assert len(posts) == 2
    assert posts[0]["source"] == "Reddit"
    assert posts[0]["needs_verification"] is True


@resp_lib.activate
def test_fetch_reddit_http_error_returns_empty():
    resp_lib.add(resp_lib.GET,
        "https://www.reddit.com/r/CollegeBasketball/new.json",
        status=429)
    scraper = InformationAsymmetryScraper()
    assert scraper.fetch_reddit_cbb_new() == []


def test_analyze_texts_flags_walking_boot():
    scraper = InformationAsymmetryScraper()
    texts = [{"title": "Kansas guard in walking boot", "url": "https://example.com",
               "source": "ESPN", "timestamp": time.time(), "needs_verification": False}]
    alerts = scraper.analyze_texts_for_asymmetry(texts)
    assert len(alerts) == 1
    assert "walking boot" in alerts[0]["keywords_found"]
    assert alerts[0]["needs_verification"] is False


def test_analyze_texts_reddit_preserves_needs_verification():
    scraper = InformationAsymmetryScraper()
    texts = [{"title": "Rumor: Duke starter torn ACL", "url": "https://reddit.com/x",
               "source": "Reddit", "timestamp": time.time(), "needs_verification": True}]
    alerts = scraper.analyze_texts_for_asymmetry(texts)
    assert alerts[0]["needs_verification"] is True


def test_analyze_texts_no_keywords_returns_empty():
    scraper = InformationAsymmetryScraper()
    texts = [{"title": "Post-game thread", "url": "https://example.com",
               "source": "Reddit", "timestamp": time.time(), "needs_verification": True}]
    assert scraper.analyze_texts_for_asymmetry(texts) == []
```

### Step 2: Run tests to verify they fail

```bash
python -m pytest tests/data/test_news_scraper.py -v
```
Expected: Multiple FAILs (`parse_espn_injuries_html`, `parse_on3_transfer_html`, `parse_nba_early_entry_html`, `needs_verification` not in items)

### Step 3: Write implementation (replace news_scraper.py)

```python
# src/data/news_scraper.py
"""
src/data/news_scraper.py

Phase 4: Continuous "Information Asymmetry" Scraper

Source reliability:
  ★★★★★  ESPN CBB Injuries  — authoritative, needs_verification=False
  ★★★★★  NBA.com Early Entry — official NBA Draft list, needs_verification=False
  ★★★★☆  On3 Transfer Portal — reliable transfer tracker, needs_verification=False
  ★★☆☆☆  Reddit r/CBB        — early rumors only, needs_verification=True
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_ALERTS_PATH = Path(__file__).parent.parent.parent / "data" / "asymmetry_alerts.json"
_HEADERS = {"User-Agent": "MarchMadnessOracle/1.0 (educational; free-tier-only)"}

_ESPN_URL   = "https://www.espn.com/mens-college-basketball/injuries"
_NBA_URL    = "https://www.nba.com/draft/2026/early-entry"
_ON3_URL    = "https://www.on3.com/transfer-portal/sport/basketball/"
_REDDIT_URL = "https://www.reddit.com/r/CollegeBasketball/new.json"

SEVERITY_WEIGHTS: dict[str, float] = {
    "torn": 1.00, "out for season": 1.00,
    "nba draft": 0.80, "concussion": 0.85,
    "suspension": 0.75, "suspended": 0.75,
    "walking boot": 0.70, "declared": 0.70,
    "mri": 0.65, "sprain": 0.60,
    "transfer portal": 0.50, "not at practice": 0.55,
    "questionable": 0.40, "limited": 0.30,
}
ALERT_KEYWORDS = list(SEVERITY_WEIGHTS.keys())

KNOWN_TEAMS: list[str] = [
    "Duke", "UNC", "Virginia", "Miami",
    "Houston", "Kansas", "Baylor", "TCU",
    "Alabama", "Tennessee", "Auburn", "Mississippi State",
    "Purdue", "Michigan State", "Illinois", "Ohio State",
    "Arizona", "UCLA", "Colorado", "Oregon",
    "UConn", "Marquette", "Creighton", "Seton Hall",
    "Drake", "Missouri State", "Gonzaga", "Saint Mary's",
    "San Diego State", "Nevada", "UAB", "UTEP",
    "Kentucky", "Florida", "Arkansas", "LSU",
    "North Carolina", "NC State", "Syracuse",
    "Indiana", "Wisconsin", "Iowa", "Minnesota",
    "BYU", "Iowa State", "West Virginia", "Oklahoma",
]


def score_severity(keywords_found: list[str]) -> float:
    if not keywords_found:
        return 0.0
    return max(SEVERITY_WEIGHTS.get(kw, 0.0) for kw in keywords_found)


def extract_team_mentions(text: str) -> list[str]:
    lower = text.lower()
    return [t for t in KNOWN_TEAMS if t.lower() in lower]


# ---------------------------------------------------------------------------
# HTML parsers (pure functions — easy to unit-test with mock HTML)
# ---------------------------------------------------------------------------

def parse_espn_injuries_html(html: str) -> list[dict[str, str]]:
    """Parse ESPN CBB injury table HTML → list of injury dicts."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, str]] = []
    table = soup.find("table", class_=lambda c: c and "Table" in c)
    if table is None:
        return rows
    tbody = table.find("tbody")
    if tbody is None:
        return rows
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        rows.append({
            "player":   cells[0].get_text(strip=True),
            "team":     cells[1].get_text(strip=True),
            "position": cells[2].get_text(strip=True) if len(cells) > 2 else "",
            "status":   cells[3].get_text(strip=True) if len(cells) > 3 else "",
            "injury":   cells[4].get_text(strip=True) if len(cells) > 4 else "",
            "source":   "ESPN",
        })
    return rows


def parse_on3_transfer_html(html: str) -> list[dict[str, str]]:
    """Parse On3 transfer portal table HTML → list of transfer dicts."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict[str, str]] = []
    table = soup.find("table", class_=lambda c: c and "transfer" in (c or "").lower())
    if table is None:
        table = soup.find("table")
    if table is None:
        return rows
    tbody = table.find("tbody")
    if tbody is None:
        return rows
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        rows.append({
            "player":      cells[0].get_text(strip=True),
            "from_school": cells[1].get_text(strip=True) if len(cells) > 1 else "",
            "to_school":   cells[2].get_text(strip=True) if len(cells) > 2 else "",
            "position":    cells[3].get_text(strip=True) if len(cells) > 3 else "",
            "status":      cells[4].get_text(strip=True) if len(cells) > 4 else "",
            "source":      "On3",
        })
    return rows


def parse_nba_early_entry_html(html: str) -> list[str]:
    """Parse NBA.com early entry page → list of player name strings."""
    soup = BeautifulSoup(html, "lxml")
    players: list[str] = []
    # Try <ul> inside early-entry-list div
    container = soup.find(attrs={"class": lambda c: c and "early-entry" in " ".join(c if isinstance(c, list) else [c]).lower()})
    if container:
        for li in container.find_all("li"):
            text = li.get_text(strip=True)
            # Format: "Player Name, School Name" — take part before comma
            name = text.split(",")[0].strip()
            if name:
                players.append(name)
    return players


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class InformationAsymmetryScraper:
    def __init__(self) -> None:
        self.flagged_items: list[dict[str, Any]] = []

    def _get(self, url: str, params: dict | None = None) -> str | None:
        """HTTP GET with error handling. Returns response text or None."""
        try:
            resp = httpx.get(url, params=params, headers=_HEADERS, timeout=20.0,
                             follow_redirects=True)
            if resp.status_code != 200:
                logger.warning("%s → HTTP %s", url, resp.status_code)
                return None
            return resp.text
        except Exception as exc:
            logger.warning("GET %s failed: %s", url, exc)
            return None

    def fetch_espn_injuries(self) -> list[dict[str, Any]]:
        """Authoritative: ESPN CBB injury report."""
        html = self._get(_ESPN_URL)
        if not html:
            return []
        rows = parse_espn_injuries_html(html)
        posts = []
        for r in rows:
            posts.append({
                "title": f"{r['player']} ({r['team']}): {r['injury']} — {r['status']}",
                "url": _ESPN_URL,
                "source": "ESPN",
                "timestamp": time.time(),
                "needs_verification": False,
            })
        return posts

    def fetch_nba_early_entries(self) -> list[dict[str, Any]]:
        """Official: NBA.com early entry list (NBA Draft departures)."""
        html = self._get(_NBA_URL)
        if not html:
            return []
        players = parse_nba_early_entry_html(html)
        posts = []
        for player in players:
            teams = extract_team_mentions(player)
            posts.append({
                "title": f"{player} declared for NBA Draft (early entry)",
                "url": _NBA_URL,
                "source": "NBA.com",
                "timestamp": time.time(),
                "needs_verification": False,
            })
        return posts

    def fetch_on3_transfers(self) -> list[dict[str, Any]]:
        """Reliable: On3 transfer portal tracker."""
        html = self._get(_ON3_URL)
        if not html:
            return []
        rows = parse_on3_transfer_html(html)
        posts = []
        for r in rows:
            dest = f" → {r['to_school']}" if r.get("to_school") else ""
            posts.append({
                "title": f"Transfer Portal: {r['player']} ({r['from_school']}){dest} — {r.get('status','')}",
                "url": _ON3_URL,
                "source": "On3",
                "timestamp": time.time(),
                "needs_verification": False,
            })
        return posts

    def fetch_reddit_cbb_new(self, limit: int = 50) -> list[dict[str, Any]]:
        """Unverified rumors: Reddit r/CollegeBasketball. needs_verification=True always."""
        text = self._get(_REDDIT_URL, params={"limit": limit})
        if not text:
            return []
        try:
            data = json.loads(text)
        except Exception:
            return []
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {})
            posts.append({
                "title": d.get("title", ""),
                "url": f"https://reddit.com{d.get('permalink', '')}",
                "source": "Reddit",
                "timestamp": d.get("created_utc", time.time()),
                "needs_verification": True,   # ALWAYS unverified
            })
        return posts

    def analyze_texts_for_asymmetry(
        self, texts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        alerts = []
        for item in texts:
            content = item.get("title", "").lower()
            found = [kw for kw in ALERT_KEYWORDS if kw in content]
            if not found:
                continue
            severity = score_severity(found)
            alert: dict[str, Any] = {
                "alert_id": f"alert_{int(item.get('timestamp', time.time()))}_{hash(item.get('title',''))&0xFFFF:04x}",
                "source": item.get("source", "Unknown"),
                "content": item.get("title", ""),
                "keywords_found": found,
                "severity": severity,
                "teams_mentioned": extract_team_mentions(item.get("title", "")),
                "url": item.get("url", ""),
                "timestamp": item.get("timestamp", time.time()),
                "needs_verification": item.get("needs_verification", True),
                "status": "REQUIRES_MANUAL_REVIEW",
            }
            alerts.append(alert)
            logger.warning("🚨 severity=%.2f needs_verification=%s: %.60s",
                           severity, alert["needs_verification"], item.get("title", ""))
        self.flagged_items.extend(alerts)
        return alerts

    def run_cycle(self) -> int:
        """Run full cycle across all sources. Returns new alert count."""
        all_texts = (
            self.fetch_espn_injuries()
            + self.fetch_nba_early_entries()
            + self.fetch_on3_transfers()
            + self.fetch_reddit_cbb_new()
        )
        alerts = self.analyze_texts_for_asymmetry(all_texts)
        _ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_ALERTS_PATH, "w") as fh:
            json.dump(self.flagged_items, fh, indent=2)
        return len(alerts)
```

### Step 4: Run tests

```bash
python -m pytest tests/data/test_news_scraper.py -v
```
Expected: All 20 tests PASS.

### Step 5: Commit

```bash
git add src/data/news_scraper.py tests/data/test_news_scraper.py
git commit -m "feat(intel): multi-source scraper — ESPN/NBA.com/On3 authoritative + Reddit unverified"
```

---

## Task 3: Intel Schemas (`src/api/schemas.py` additions)

Add `IntelAlert` and `IntelResponse` Pydantic models that mirror the frontend Zod contracts.

**Files:**
- Modify: `src/api/schemas.py`
- Modify: `tests/api/test_schemas.py`

### Step 1: Write the failing tests (append to existing test_schemas.py)

Read `tests/api/test_schemas.py` first. Then add at the bottom:

```python
# Append to tests/api/test_schemas.py
import time as _time
from src.api.schemas import IntelAlert, IntelResponse


def test_intel_alert_valid():
    alert = IntelAlert(
        alert_id="alert_123_abcd",
        source="ESPN",
        content="Cooper Flagg (Duke): Knee — Out",
        keywords_found=["out for season"],
        severity=1.00,
        teams_mentioned=["Duke"],
        region="East",
        url="https://espn.com/mens-college-basketball/injuries",
        timestamp=_time.time(),
        needs_verification=False,
        status="REQUIRES_MANUAL_REVIEW",
    )
    assert alert.severity == 1.00
    assert alert.needs_verification is False
    assert "Duke" in alert.teams_mentioned


def test_intel_alert_severity_clamp():
    """severity must be 0.0–1.0"""
    with pytest.raises(Exception):
        IntelAlert(
            alert_id="x", source="Reddit", content="test",
            keywords_found=[], severity=1.5,
            teams_mentioned=[], region="", url="", timestamp=0.0,
            needs_verification=True, status="REQUIRES_MANUAL_REVIEW",
        )


def test_intel_alert_reddit_needs_verification_true():
    """Reddit alerts must always be flagged as unverified."""
    alert = IntelAlert(
        alert_id="r", source="Reddit", content="rumor: torn ACL",
        keywords_found=["torn"], severity=1.0,
        teams_mentioned=["Duke"], region="East",
        url="https://reddit.com/x", timestamp=_time.time(),
        needs_verification=True, status="REQUIRES_MANUAL_REVIEW",
    )
    assert alert.needs_verification is True


def test_intel_response_valid():
    alert = IntelAlert(
        alert_id="a", source="ESPN", content="Edey questionable",
        keywords_found=["questionable"], severity=0.4,
        teams_mentioned=["Purdue"], region="Midwest",
        url="https://espn.com/", timestamp=_time.time(),
        needs_verification=False, status="REQUIRES_MANUAL_REVIEW",
    )
    resp = IntelResponse(alerts=[alert], last_updated="2026-03-12T10:00:00Z", total=1)
    assert resp.total == 1
    assert len(resp.alerts) == 1
```

### Step 2: Run to verify fail

```bash
python -m pytest tests/api/test_schemas.py::test_intel_alert_valid -v
```
Expected: FAIL with `ImportError: cannot import name 'IntelAlert'`

### Step 3: Write implementation (append to schemas.py)

Add the following to the bottom of `src/api/schemas.py`:

```python
# ── Intel / Autonomous Flags ───────────────────────────────────────────────

class IntelAlert(BaseModel):
    alert_id:           str
    source:             str               # "ESPN", "NBA.com", "On3", "Rotowire", "roster_validator", "Reddit"
    content:            str               # original headline/text
    keywords_found:     List[str]
    severity:           float = Field(ge=0.0, le=1.0)
    teams_mentioned:    List[str]
    region:             str               # "East" | "West" | "South" | "Midwest" | "Unknown"
    url:                str
    timestamp:          float             # Unix timestamp
    needs_verification: bool = False      # True for Reddit/unverified sources only
    status:             str = "REQUIRES_MANUAL_REVIEW"


class IntelResponse(BaseModel):
    alerts:       List[IntelAlert]
    last_updated: str                     # ISO 8601
    total:        int
```

### Step 4: Run tests

```bash
python -m pytest tests/api/test_schemas.py -v
```
Expected: All schema tests PASS (including the new intel tests).

### Step 5: Commit

```bash
git add src/api/schemas.py tests/api/test_schemas.py
git commit -m "feat(intel): IntelAlert + IntelResponse Pydantic schemas"
```

---

## Task 4: Intel Aggregator + `/api/intel` Endpoint

Builds the aggregation layer and FastAPI endpoint with region/team filtering and APScheduler background refresh.

**Files:**
- Create: `src/api/intel.py`
- Create: `tests/api/test_intel.py`
- Modify: `src/api/server.py` (add endpoint + scheduler startup)

### Step 1: Write the failing tests

```python
# tests/api/test_intel.py
import json
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


def test_intel_endpoint_exists():
    """GET /api/intel returns 200 without crashing."""
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/api/intel")
    assert resp.status_code == 200


def test_intel_endpoint_returns_intel_response_shape():
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/api/intel")
    data = resp.json()
    assert "alerts" in data
    assert "last_updated" in data
    assert "total" in data
    assert isinstance(data["alerts"], list)


def test_intel_endpoint_filter_by_team():
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/api/intel?team=Duke")
    data = resp.json()
    # All returned alerts must mention Duke (or be empty)
    for alert in data["alerts"]:
        assert "Duke" in alert["teams_mentioned"]


def test_intel_endpoint_filter_by_region():
    from src.api.server import app
    client = TestClient(app)
    resp = client.get("/api/intel?region=East")
    data = resp.json()
    for alert in data["alerts"]:
        assert alert["region"] in ("East", "Unknown")


def test_intel_aggregator_assigns_region():
    from src.api.intel import assign_region
    assert assign_region(["Duke"]) == "East"
    assert assign_region(["Kansas"]) == "Midwest"
    assert assign_region(["Alabama"]) == "South"
    assert assign_region(["Arizona"]) == "West"
    assert assign_region([]) == "Unknown"


def test_intel_aggregator_from_rotowire_alerts():
    from src.api.intel import rotowire_to_intel_alerts
    import pandas as pd
    df = pd.DataFrame([{
        "player": "John Doe",
        "team": "Duke",
        "status": "Out",
        "injury": "sprained ankle",
        "expected_return": "2-3 weeks",
        "scraped_at": "2026-03-12T10:00:00Z",
    }])
    alerts = rotowire_to_intel_alerts(df)
    assert len(alerts) == 1
    assert alerts[0].source == "Rotowire"
    assert "Duke" in alerts[0].teams_mentioned
    assert alerts[0].region == "East"


def test_intel_aggregator_deduplicates_by_alert_id():
    from src.api.intel import deduplicate_alerts
    from src.api.schemas import IntelAlert
    a1 = IntelAlert(alert_id="x", source="Reddit", content="t",
                    keywords_found=[], severity=0.5, teams_mentioned=[],
                    region="", url="", timestamp=time.time())
    a2 = IntelAlert(alert_id="x", source="Reddit", content="t2",
                    keywords_found=[], severity=0.5, teams_mentioned=[],
                    region="", url="", timestamp=time.time())
    result = deduplicate_alerts([a1, a2])
    assert len(result) == 1
```

### Step 2: Run to verify fail

```bash
python -m pytest tests/api/test_intel.py -v
```
Expected: FAILs (`/api/intel` not found, `src.api.intel` not found)

### Step 3: Write `src/api/intel.py`

```python
# src/api/intel.py
"""
src/api/intel.py

Intel aggregator: combines Rotowire injuries + Reddit news + roster departures
into a unified IntelAlert stream with region/team tagging.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.api.schemas import IntelAlert, IntelResponse
from src.data.news_scraper import (
    SEVERITY_WEIGHTS,
    ALERT_KEYWORDS,
    score_severity,
    extract_team_mentions,
    InformationAsymmetryScraper,
)

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/intel_cache.json")

# Region mapping: team name → bracket region
_TEAM_REGION: dict[str, str] = {
    # East (ACC + Big East)
    "Duke": "East", "UNC": "East", "Virginia": "East", "Miami": "East",
    "UConn": "East", "Marquette": "East", "Creighton": "East", "Seton Hall": "East",
    "North Carolina": "East", "NC State": "East", "Syracuse": "East",
    # Midwest (Big Ten + Big 12 + MVC)
    "Purdue": "Midwest", "Michigan State": "Midwest", "Illinois": "Midwest",
    "Ohio State": "Midwest", "Indiana": "Midwest", "Wisconsin": "Midwest",
    "Iowa": "Midwest", "Minnesota": "Midwest",
    "Houston": "Midwest", "Kansas": "Midwest", "Baylor": "Midwest", "TCU": "Midwest",
    "BYU": "Midwest", "Iowa State": "Midwest", "West Virginia": "Midwest",
    "Drake": "Midwest", "Missouri State": "Midwest",
    # South (SEC + CUSA)
    "Alabama": "South", "Tennessee": "South", "Auburn": "South",
    "Mississippi State": "South", "Kentucky": "South", "Florida": "South",
    "Arkansas": "South", "LSU": "South",
    "UAB": "South", "UTEP": "South",
    # West (Pac-12 + WCC + MWC)
    "Arizona": "West", "UCLA": "West", "Colorado": "West", "Oregon": "West",
    "Gonzaga": "West", "Saint Mary's": "West",
    "San Diego State": "West", "Nevada": "West",
}


def assign_region(teams_mentioned: list[str]) -> str:
    """Return the bracket region for the first recognized team, else 'Unknown'."""
    for team in teams_mentioned:
        region = _TEAM_REGION.get(team)
        if region:
            return region
    return "Unknown"


def rotowire_to_intel_alerts(df: pd.DataFrame) -> list[IntelAlert]:
    """Convert a Rotowire injury DataFrame into IntelAlert objects."""
    alerts: list[IntelAlert] = []
    for _, row in df.iterrows():
        injury_text = str(row.get("injury", "")).lower()
        found_kws = [kw for kw in ALERT_KEYWORDS if kw in injury_text]
        severity = score_severity(found_kws) if found_kws else 0.40  # default for any injury
        if not found_kws:
            found_kws = ["injured"]
        team = str(row.get("team", ""))
        player = str(row.get("player", ""))
        teams_mentioned = [team] if team else extract_team_mentions(team)
        region = assign_region(teams_mentioned)
        alerts.append(IntelAlert(
            alert_id=f"rotowire_{int(time.time())}_{hash(player + team) & 0xFFFF:04x}",
            source="Rotowire",
            content=f"{player} ({team}): {row.get('injury','unknown')} — Status: {row.get('status','')}",
            keywords_found=found_kws,
            severity=severity,
            teams_mentioned=teams_mentioned,
            region=region,
            url="https://www.rotowire.com/basketball/college/injury-report.php",
            timestamp=time.time(),
        ))
    return alerts


def reddit_to_intel_alerts(posts: list[dict]) -> list[IntelAlert]:
    """Convert raw Reddit posts to IntelAlert objects (only keyword-matched ones)."""
    scraper = InformationAsymmetryScraper()
    raw_alerts = scraper.analyze_texts_for_asymmetry(posts)
    alerts: list[IntelAlert] = []
    for r in raw_alerts:
        teams = r.get("teams_mentioned", [])
        region = assign_region(teams)
        alerts.append(IntelAlert(
            alert_id=r["alert_id"],
            source=r["source"],
            content=r["content"],
            keywords_found=r["keywords_found"],
            severity=r.get("severity", 0.5),
            teams_mentioned=teams,
            region=region,
            url=r.get("url", ""),
            timestamp=r.get("timestamp", time.time()),
            status=r.get("status", "REQUIRES_MANUAL_REVIEW"),
        ))
    return alerts


def deduplicate_alerts(alerts: list[IntelAlert]) -> list[IntelAlert]:
    """Remove duplicate alerts by alert_id, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[IntelAlert] = []
    for a in alerts:
        if a.alert_id not in seen:
            seen.add(a.alert_id)
            result.append(a)
    return result


def build_intel_feed(
    region: Optional[str] = None,
    team: Optional[str] = None,
    limit: int = 50,
) -> IntelResponse:
    """
    Load alerts from cache and apply region/team filters.

    Falls back to an empty response if no cache exists yet.
    """
    alerts: list[IntelAlert] = []

    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH) as fh:
                raw = json.load(fh)
            for item in raw:
                try:
                    alerts.append(IntelAlert(**item))
                except Exception:
                    continue
        except Exception as exc:
            logger.warning("Intel cache read failed: %s", exc)

    # Filters
    if region and region != "All":
        alerts = [a for a in alerts if a.region == region]
    if team:
        alerts = [a for a in alerts if team in a.teams_mentioned]

    # Sort by severity desc, then recency
    alerts.sort(key=lambda a: (-a.severity, -a.timestamp))
    alerts = alerts[:limit]

    return IntelResponse(
        alerts=alerts,
        last_updated=datetime.now(tz=timezone.utc).isoformat(),
        total=len(alerts),
    )


def refresh_intel_cache() -> int:
    """
    Full scrape cycle: Reddit + Rotowire. Writes merged results to cache.
    Returns total alert count.
    Called by APScheduler every 15 minutes.
    """
    all_alerts: list[IntelAlert] = []

    # Reddit
    try:
        scraper = InformationAsymmetryScraper()
        posts = scraper.fetch_reddit_cbb_new()
        all_alerts.extend(reddit_to_intel_alerts(posts))
        logger.info("Intel refresh: %d Reddit alerts", len(all_alerts))
    except Exception as exc:
        logger.warning("Reddit refresh failed: %s", exc)

    # Rotowire
    try:
        from src.data.injury_feed import fetch_rotowire_injuries
        df = fetch_rotowire_injuries(season=2026)
        all_alerts.extend(rotowire_to_intel_alerts(df))
        logger.info("Intel refresh: added Rotowire alerts, total=%d", len(all_alerts))
    except Exception as exc:
        logger.warning("Rotowire refresh failed: %s", exc)

    # Deduplicate + write
    all_alerts = deduplicate_alerts(all_alerts)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w") as fh:
        json.dump([a.model_dump() for a in all_alerts], fh, indent=2)

    logger.info("Intel cache written: %d alerts", len(all_alerts))
    return len(all_alerts)
```

### Step 4: Add endpoint to `src/api/server.py`

Add the following **import** near the top (after existing imports):

```python
from src.api.intel import build_intel_feed, refresh_intel_cache
from src.api.schemas import IntelResponse
```

Add the **endpoint** before the `/health` route:

```python
@app.get("/api/intel", response_model=IntelResponse)
async def get_intel(
    region: str = Query(default="All"),
    team: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=200),
) -> IntelResponse:
    return build_intel_feed(
        region=region if region != "All" else None,
        team=team if team else None,
        limit=limit,
    )
```

Add the **APScheduler startup** after `app = FastAPI(...)` block (after CORS middleware):

```python
from apscheduler.schedulers.background import BackgroundScheduler as _BGScheduler

_scheduler = _BGScheduler()
_scheduler.add_job(refresh_intel_cache, "interval", minutes=15, id="intel_refresh")
_scheduler.start()
```

### Step 5: Run all tests

```bash
python -m pytest tests/api/test_intel.py tests/api/test_server.py -v
```
Expected: All intel tests PASS. Existing server tests PASS.

### Step 6: Commit

```bash
git add src/api/intel.py src/api/server.py tests/api/test_intel.py
git commit -m "feat(intel): aggregator + /api/intel endpoint + APScheduler 15-min refresh"
```

---

## Task 5: Frontend Intel Types + API Helper

Add Zod schemas and `fetchIntel` helper so TypeScript stays end-to-end typed.

**Files:**
- Modify: `frontend/lib/api-types.ts`
- Modify: `frontend/lib/api.ts`

### Step 1: Read the existing files first

Read `frontend/lib/api-types.ts` and `frontend/lib/api.ts` fully before editing.

### Step 2: Write the failing vitest test (append to existing test file)

Find `frontend/lib/` test files via `ls frontend/lib/`:

```typescript
// Append to the relevant api-types test file (find with: ls frontend/lib/__tests__/ or frontend/__tests__/)
import { IntelAlertSchema, IntelResponseSchema } from '@/lib/api-types';

test('IntelAlertSchema parses valid alert', () => {
  const raw = {
    alert_id: 'alert_123_abcd',
    source: 'Reddit',
    content: "Duke Cooper Flagg declared for NBA Draft",
    keywords_found: ['nba draft'],
    severity: 0.80,
    teams_mentioned: ['Duke'],
    region: 'East',
    url: 'https://reddit.com/r/CollegeBasketball/abc',
    timestamp: Date.now() / 1000,
    status: 'REQUIRES_MANUAL_REVIEW',
  };
  const result = IntelAlertSchema.safeParse(raw);
  expect(result.success).toBe(true);
});

test('IntelResponseSchema parses valid response', () => {
  const raw = { alerts: [], last_updated: '2026-03-12T10:00:00Z', total: 0 };
  const result = IntelResponseSchema.safeParse(raw);
  expect(result.success).toBe(true);
});
```

### Step 3: Add Zod schemas to `frontend/lib/api-types.ts`

Append at the bottom of the file:

```typescript
// ── Intel / Autonomous Flags ──────────────────────────────────────────────

export const IntelAlertSchema = z.object({
  alert_id:           z.string(),
  source:             z.string(),
  content:            z.string(),
  keywords_found:     z.array(z.string()),
  severity:           z.number().min(0).max(1),
  teams_mentioned:    z.array(z.string()),
  region:             z.string(),
  url:                z.string(),
  timestamp:          z.number(),
  needs_verification: z.boolean().default(false),
  status:             z.string().default('REQUIRES_MANUAL_REVIEW'),
});

export const IntelResponseSchema = z.object({
  alerts:       z.array(IntelAlertSchema),
  last_updated: z.string(),
  total:        z.number(),
});

export type IntelAlert   = z.infer<typeof IntelAlertSchema>;
export type IntelResponse = z.infer<typeof IntelResponseSchema>;
```

### Step 4: Add `fetchIntel` to `frontend/lib/api.ts`

Append at the bottom:

```typescript
export interface IntelParams {
  region?: string;
  team?: string;
  limit?: number;
}

export async function fetchIntel(params: IntelParams = {}): Promise<IntelResponse> {
  const { region = 'All', team = '', limit = 50 } = params;
  const qs = new URLSearchParams();
  if (region !== 'All') qs.set('region', region);
  if (team) qs.set('team', team);
  qs.set('limit', String(limit));
  const url = `http://localhost:8000/api/intel?${qs}`;
  const res = await fetch(url);
  if (!res.ok) return { alerts: [], last_updated: new Date().toISOString(), total: 0 };
  const raw = await res.json();
  const parsed = IntelResponseSchema.safeParse(raw);
  if (!parsed.success) return { alerts: [], last_updated: new Date().toISOString(), total: 0 };
  return parsed.data;
}
```

Also add the import at the top of `api.ts` (if `IntelResponse` and `IntelResponseSchema` aren't already imported):

```typescript
import { IntelResponseSchema } from './api-types';
import type { IntelResponse } from './api-types';
```

### Step 5: Run vitest

```bash
cd frontend && npx vitest run --reporter=dot
```
Expected: All tests PASS including new intel schema tests.

### Step 6: Commit

```bash
git add frontend/lib/api-types.ts frontend/lib/api.ts
git commit -m "feat(intel): Zod schemas + fetchIntel API helper"
```

---

## Task 6: IntelFeed Component (`frontend/components/intel/IntelFeed.tsx`)

The main UI panel: shows all alerts with region/team filters, severity badges, auto-poll, and "See All" mode.

**Files:**
- Create: `frontend/components/intel/IntelFeed.tsx`

### Step 1: No vitest test needed (UI component — covered by e2e smoke test in Task 7)

### Step 2: Write `frontend/components/intel/IntelFeed.tsx`

```tsx
'use client';
// frontend/components/intel/IntelFeed.tsx
import { useState, useEffect, useCallback } from 'react';
import { fetchIntel } from '@/lib/api';
import type { IntelAlert } from '@/lib/api-types';

const REGIONS = ['All', 'East', 'West', 'South', 'Midwest'] as const;

function SeverityBadge({ severity }: { severity: number }) {
  const [label, color] =
    severity >= 0.7 ? ['🚨 CRITICAL', '#ff2d55'] :
    severity >= 0.4 ? ['⚠ MAJOR',    '#ffb800'] :
                      ['ℹ MINOR',    '#00f5ff'];
  return (
    <span style={{
      display: 'inline-block', padding: '2px 8px',
      borderRadius: 4, fontSize: 10, fontWeight: 700,
      color, border: `1px solid ${color}`,
      marginRight: 6, letterSpacing: '0.06em',
    }}>
      {label}
    </span>
  );
}

function AlertCard({ alert }: { alert: IntelAlert }) {
  const date = new Date(alert.timestamp * 1000);
  const timeAgo = (() => {
    const mins = Math.floor((Date.now() - date.getTime()) / 60000);
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    return `${Math.floor(hrs / 24)}d ago`;
  })();

  return (
    <div style={{
      padding: '10px 14px',
      borderBottom: '1px solid rgba(255,255,255,0.07)',
      transition: 'background 0.15s',
    }}
    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.04)')}
    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <SeverityBadge severity={alert.severity} />
        {alert.needs_verification && (
          <span style={{ fontSize: 9, color: '#888', border: '1px solid #444',
            borderRadius: 4, padding: '1px 6px', fontStyle: 'italic' }}>
            UNVERIFIED
          </span>
        )}
        {alert.region !== 'Unknown' && (
          <span style={{ fontSize: 10, color: '#888', border: '1px solid #333',
            borderRadius: 4, padding: '1px 6px' }}>
            {alert.region}
          </span>
        )}
        {alert.teams_mentioned.slice(0, 3).map(t => (
          <span key={t} style={{ fontSize: 10, color: '#00f5ff', border: '1px solid #00f5ff33',
            borderRadius: 4, padding: '1px 6px' }}>
            {t}
          </span>
        ))}
        <span style={{ marginLeft: 'auto', fontSize: 10, color: '#555' }}>{timeAgo}</span>
      </div>
      <p style={{ margin: 0, fontSize: 13, color: '#ccc', lineHeight: 1.4 }}>
        {alert.content}
      </p>
      <div style={{ marginTop: 4, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {alert.keywords_found.map(kw => (
          <span key={kw} style={{ fontSize: 9, color: '#888', background: 'rgba(255,255,255,0.05)',
            borderRadius: 3, padding: '1px 5px' }}>
            {kw}
          </span>
        ))}
        <a href={alert.url} target="_blank" rel="noreferrer"
          style={{ fontSize: 9, color: '#555', marginLeft: 'auto' }}>
          source ↗
        </a>
      </div>
    </div>
  );
}

export function IntelFeed() {
  const [alerts, setAlerts]           = useState<IntelAlert[]>([]);
  const [loading, setLoading]         = useState(true);
  const [lastUpdated, setLastUpdated] = useState('');
  const [region, setRegion]           = useState<string>('All');
  const [teamSearch, setTeamSearch]   = useState('');

  const load = useCallback(async () => {
    const data = await fetchIntel({
      region: region !== 'All' ? region : undefined,
      team: teamSearch.trim() || undefined,
      limit: 100,
    });
    setAlerts(data.alerts);
    setLastUpdated(data.last_updated);
    setLoading(false);
  }, [region, teamSearch]);

  useEffect(() => {
    load();
    const id = setInterval(load, 60_000); // poll every 60s
    return () => clearInterval(id);
  }, [load]);

  const criticalCount = alerts.filter(a => a.severity >= 0.7).length;

  return (
    <div style={{
      background: 'rgba(10,10,20,0.95)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 12,
      height: 'calc(100vh - 120px)',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '16px 18px',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: '#fff' }}>
          🚨 Intel Feed
        </span>
        {criticalCount > 0 && (
          <span style={{
            background: '#ff2d55', color: '#fff', borderRadius: 10,
            fontSize: 10, fontWeight: 700, padding: '1px 7px',
          }}>
            {criticalCount} CRITICAL
          </span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: 10, color: '#555' }}>
          {alerts.length} alerts · updated {lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : '—'}
        </span>
      </div>

      {/* Filters */}
      <div style={{
        padding: '10px 14px',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        display: 'flex', gap: 8, flexWrap: 'wrap',
      }}>
        {REGIONS.map(r => (
          <button key={r}
            onClick={() => setRegion(r)}
            style={{
              padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
              cursor: 'pointer', border: 'none',
              background: region === r ? '#00f5ff22' : 'rgba(255,255,255,0.05)',
              color: region === r ? '#00f5ff' : '#888',
              outline: region === r ? '1px solid #00f5ff66' : 'none',
            }}>
            {r}
          </button>
        ))}
        <input
          placeholder="Filter by team…"
          value={teamSearch}
          onChange={e => setTeamSearch(e.target.value)}
          style={{
            marginLeft: 'auto', padding: '4px 10px', borderRadius: 6,
            fontSize: 11, background: 'rgba(255,255,255,0.05)',
            border: '1px solid rgba(255,255,255,0.1)', color: '#ccc',
            outline: 'none', width: 140,
          }}
        />
      </div>

      {/* Alert list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {loading && (
          <div style={{ padding: 32, textAlign: 'center', color: '#555' }}>
            Scanning sources…
          </div>
        )}
        {!loading && alerts.length === 0 && (
          <div style={{ padding: 32, textAlign: 'center', color: '#555' }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>✅</div>
            <p style={{ margin: 0 }}>No active alerts for the current filter.</p>
            <p style={{ margin: '4px 0 0', fontSize: 11, color: '#444' }}>
              Auto-refreshes every 60 seconds.
            </p>
          </div>
        )}
        {alerts.map(alert => <AlertCard key={alert.alert_id} alert={alert} />)}
      </div>
    </div>
  );
}
```

### Step 3: Commit

```bash
git add frontend/components/intel/IntelFeed.tsx
git commit -m "feat(intel): IntelFeed component with region/team filters + auto-poll"
```

---

## Task 7: Wire Intel into Nav + Page

Add the Intel page to navigation and render it from the main page.

**Files:**
- Modify: `frontend/components/nav/Sidebar.tsx`
- Modify: `frontend/app/page.tsx`

### Step 1: Read both files fully before editing

Read `frontend/components/nav/Sidebar.tsx` and `frontend/app/page.tsx` fully.

### Step 2: Add `'intel'` to `NavPage` in Sidebar.tsx

Change:
```typescript
export type NavPage = 'rankings' | 'matchup' | 'bracket' | 'projections' | 'warroom' | 'graph';
```
To:
```typescript
export type NavPage = 'rankings' | 'matchup' | 'bracket' | 'projections' | 'warroom' | 'graph' | 'intel';
```

Add an `IntelIcon` SVG component (after existing icon components):

```tsx
function IntelIcon({ color }: { color: string }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
    </svg>
  );
}
```

Add the Intel nav item to the nav items array (wherever `navItems` or the nav list is defined):

Find the existing array of nav items (pattern: `{ id: 'rankings', label: ..., icon: ... }`) and add:
```typescript
{ id: 'intel' as NavPage, label: 'Intel', icon: IntelIcon },
```

### Step 3: Add Intel page rendering to `page.tsx`

Add the import:
```tsx
import { IntelFeed } from '@/components/intel/IntelFeed';
```

Find the section where pages are conditionally rendered (pattern: `activePage === 'rankings'`) and add:
```tsx
{activePage === 'intel' && <IntelFeed />}
```

### Step 4: Run vitest + type check

```bash
cd frontend && npx vitest run --reporter=dot && npx tsc --noEmit
```
Expected: All tests PASS, no TypeScript errors.

### Step 5: Commit

```bash
git add frontend/components/nav/Sidebar.tsx frontend/app/page.tsx
git commit -m "feat(intel): wire IntelFeed into nav + page routing"
```

---

## Verification Steps

### 1. Run full backend test suite

```bash
python -m pytest tests/ -q
```
Expected: All existing tests + new intel tests pass (total should be ≥ 980).

### 2. Run frontend tests

```bash
cd frontend && npx vitest run --reporter=dot
```
Expected: All tests pass.

### 3. End-to-end smoke test

```bash
# Start backend
uvicorn src.api.server:app --port 8000 --reload

# In another terminal, test the endpoint
curl http://localhost:8000/api/intel
# Expected: {"alerts":[], "last_updated":"...", "total":0}

curl "http://localhost:8000/api/intel?region=East"
# Expected: same shape, filtered

# Trigger a manual refresh cycle
python -c "from src.api.intel import refresh_intel_cache; n=refresh_intel_cache(); print(f'{n} alerts cached')"

# Start frontend
cd frontend && npm run dev
# Navigate to http://localhost:3000
# Click the ⚠ Intel icon in the sidebar → IntelFeed renders
# Region buttons filter alerts
# Team search filters alerts
# Page auto-polls every 60s
```

### 4. APScheduler confirmation

```bash
# Observe server logs after startup — should see:
# INFO: Scheduler started
# INFO: Intel refresh: X Reddit alerts
# (every 15 minutes automatically)
```
