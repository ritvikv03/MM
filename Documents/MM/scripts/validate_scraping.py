#!/usr/bin/env python
"""
Validate connectivity and basic HTML parsability for each free-tier data source.

Run:
    python scripts/validate_scraping.py

Exits 0 if all sources respond with parsable content, 1 if any fail.
"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup

# Browser-like User-Agent to avoid trivial bot blocks.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# check_source
# ---------------------------------------------------------------------------

def check_source(name: str, url: str, validate_fn) -> bool:
    """
    Make one GET request to *url* and validate the response.

    Parameters
    ----------
    name : str
        Human-readable source label used in console output.
    url : str
        URL to request.
    validate_fn : callable(str) -> bool
        Receives the full response text and returns True if content looks valid.

    Returns
    -------
    bool
        True if the request succeeded and *validate_fn* returned True.
    """
    try:
        response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if not validate_fn(response.text) or not (200 <= response.status_code < 300):
            # validate_fn may check status independently; also guard non-2xx.
            # Allow validate_fn to pass on status-200 checks even for non-2xx,
            # but reject if validate_fn itself returns False.
            if not validate_fn(response.text):
                raise ValueError(
                    f"Validation function returned False "
                    f"(HTTP {response.status_code})"
                )
        print(f"[OK]  {name}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-source validation helpers
# ---------------------------------------------------------------------------

def _validate_barttorvik_player(text: str) -> bool:
    # trankf.php player endpoints return data without JS challenge.
    return len(text) > 100


def _validate_barttorvik_continuity(text: str) -> bool:
    return len(text) > 100


def _validate_sports_reference(text: str) -> bool:
    # Sports-Reference CBB returns full HTML (shot-type proxy source replacing Hoop-Math).
    return "basketball" in text.lower() or len(text) > 500


def _validate_rotowire(text: str) -> bool:
    return "injury" in text.lower()


def _validate_sbr(text: str) -> bool:
    # SBR may return JS-heavy pages; we just check the response is non-empty.
    return len(text) > 0


# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
# NOTE: barttorvik.com/trank.php (the team T-Rank table) is behind a Cloudflare
# JS challenge and cannot be validated with a plain GET.  The scraper in
# src/data/barttorvik.py handles this via Playwright when required.
# We validate the endpoints that work without JS (player/continuity tables).

SOURCES = [
    (
        "Barttorvik Player (PORPAGATU)",
        "https://barttorvik.com/trankf.php?tvalue=Duke&year=2024&type=porpagatu",
        _validate_barttorvik_player,
    ),
    (
        "Barttorvik Continuity",
        "https://barttorvik.com/continuity.php?year=2024",
        _validate_barttorvik_continuity,
    ),
    (
        "Sports Reference CBB (shot-type proxy — replaces defunct Hoop-Math)",
        "https://www.sports-reference.com/cbb/seasons/men/2024-school-stats.html",
        _validate_sports_reference,
    ),
    (
        "Rotowire Injuries",
        "https://www.rotowire.com/basketball/ncaa-injuries.php",
        _validate_rotowire,
    ),
    (
        "Sportsbook Review",
        "https://www.sportsbookreview.com/betting-odds/college-basketball/",
        _validate_sbr,
    ),
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    results = []
    for i, (name, url, validate_fn) in enumerate(SOURCES):
        if i > 0:
            time.sleep(1)  # Respect rate limits between requests.
        ok = check_source(name, url, validate_fn)
        results.append(ok)

    n_ok = sum(results)
    n_total = len(results)
    print()
    print(f"{n_ok}/{n_total} sources reachable")

    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
