"""
src/api/bracket_2026.py
Official 2026 NCAA Tournament bracket field — Selection Sunday March 15, 2026.

68 teams across 4 regions.
First Four play-in games: UMBC/Howard (16 Midwest), Prairie View/Lehigh (16 South),
Miami (OH)/SMU (11 Midwest), Texas/NC State (11 West).

Public API
----------
get_bracket_field_2026()      -> dict   (region -> list of {team, seed})
get_bracket_teams_ordered()   -> list   (all 64 teams in bracket seeding order)
get_bracket_teams_flat()      -> list   (all 68 team names)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Official 2026 NCAA Tournament Bracket
# Source: Selection Sunday, March 15, 2026
# 1-seeds: Duke (East), Florida (South), Arizona (West), Michigan (Midwest)
# ---------------------------------------------------------------------------

BRACKET_2026: dict[str, list[dict[str, object]]] = {
    "East": [
        {"team": "Duke",                "seed": 1},
        {"team": "Siena",               "seed": 16},
        {"team": "Ohio St.",            "seed": 8},
        {"team": "TCU",                 "seed": 9},
        {"team": "St. John's",          "seed": 5},
        {"team": "Northern Iowa",       "seed": 12},
        {"team": "Kansas",              "seed": 4},
        {"team": "Cal Baptist",         "seed": 13},
        {"team": "Louisville",          "seed": 6},
        {"team": "South Florida",       "seed": 11},
        {"team": "Michigan St.",        "seed": 3},
        {"team": "North Dakota St.",    "seed": 14},
        {"team": "UCLA",               "seed": 7},
        {"team": "UCF",                "seed": 10},
        {"team": "Connecticut",         "seed": 2},
        {"team": "Furman",              "seed": 15},
    ],
    "South": [
        {"team": "Florida",             "seed": 1},
        {"team": "Prairie View",        "seed": 16},  # First Four: Prairie View/Lehigh
        {"team": "Clemson",             "seed": 8},
        {"team": "Iowa",               "seed": 9},
        {"team": "Vanderbilt",          "seed": 5},
        {"team": "McNeese",             "seed": 12},
        {"team": "Nebraska",            "seed": 4},
        {"team": "Troy",               "seed": 13},
        {"team": "North Carolina",      "seed": 6},
        {"team": "VCU",                "seed": 11},
        {"team": "Illinois",            "seed": 3},
        {"team": "Penn",               "seed": 14},
        {"team": "Saint Mary's",        "seed": 7},
        {"team": "Texas A&M",           "seed": 10},
        {"team": "Houston",             "seed": 2},
        {"team": "Idaho",              "seed": 15},
    ],
    "West": [
        {"team": "Arizona",             "seed": 1},
        {"team": "LIU",                "seed": 16},
        {"team": "Villanova",           "seed": 8},
        {"team": "Utah St.",            "seed": 9},
        {"team": "Wisconsin",           "seed": 5},
        {"team": "High Point",          "seed": 12},
        {"team": "Arkansas",            "seed": 4},
        {"team": "Hawai'i",             "seed": 13},
        {"team": "BYU",                "seed": 6},
        {"team": "Texas",              "seed": 11},  # First Four: Texas/NC State
        {"team": "Gonzaga",             "seed": 3},
        {"team": "Kennesaw St.",        "seed": 14},
        {"team": "Miami FL",            "seed": 7},
        {"team": "Missouri",            "seed": 10},
        {"team": "Purdue",              "seed": 2},
        {"team": "Queens",              "seed": 15},
    ],
    "Midwest": [
        {"team": "Michigan",            "seed": 1},
        {"team": "UMBC",               "seed": 16},  # First Four: UMBC/Howard
        {"team": "Georgia",             "seed": 8},
        {"team": "Saint Louis",         "seed": 9},
        {"team": "Texas Tech",          "seed": 5},
        {"team": "Akron",              "seed": 12},
        {"team": "Alabama",             "seed": 4},
        {"team": "Hofstra",             "seed": 13},
        {"team": "Tennessee",           "seed": 6},
        {"team": "Miami OH",            "seed": 11},  # First Four: Miami (OH)/SMU
        {"team": "Virginia",            "seed": 3},
        {"team": "Wright St.",          "seed": 14},
        {"team": "Kentucky",            "seed": 7},
        {"team": "Santa Clara",         "seed": 10},
        {"team": "Iowa St.",            "seed": 2},
        {"team": "Tennessee St.",       "seed": 15},
    ],
}

# ---------------------------------------------------------------------------
# Standard bracket pairing order: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
# ---------------------------------------------------------------------------
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGIONS = ["East", "South", "West", "Midwest"]


def get_bracket_field_2026() -> dict[str, list[dict[str, object]]]:
    """Return the full 2026 bracket field grouped by region."""
    return BRACKET_2026


def get_bracket_teams_ordered() -> list[str]:
    """Return all 64 team names in bracket seeding order (4 regions × 16 seeds).

    Order within each region follows standard bracket pairing:
    1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15.
    """
    ordered: list[str] = []
    for region in REGIONS:
        teams_in_region = BRACKET_2026[region]
        seed_to_team = {int(t["seed"]): str(t["team"]) for t in teams_in_region}
        for seed in SEED_ORDER:
            if seed in seed_to_team:
                ordered.append(seed_to_team[seed])
    return ordered


def get_bracket_teams_flat() -> list[str]:
    """Return all 64 team names (one per region slot, no duplicates)."""
    teams: list[str] = []
    for region in REGIONS:
        for entry in BRACKET_2026[region]:
            teams.append(str(entry["team"]))
    return teams
