"""
src/api/intel_engine.py
Auto-generate intel flags from Barttorvik T-Rank data.

All flags are computed from real efficiency metrics — no hardcoded player names
or narratives. Called by the /api/intel endpoint in server.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from src.api.bracket_2026 import BRACKET_2026

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntelFlag:
    type: str          # "risk" | "alert" | "surge" | "cinderella"
    severity: str      # "EXTREME" | "HIGH" | "MODERATE" | "LOW"
    team: str
    seed: int
    region: str
    headline: str      # short summary, data-driven
    detail: str        # longer computed narrative
    metric: str        # e.g. "luck=0.105" or "adjde_rank=118"
    emoji: str         # for UI display

@dataclass
class CinderellaEntry:
    team: str
    seed: int
    region: str
    opponent: str
    opponent_seed: int
    upset_pct: float   # computed from EM delta
    edge_summary: str  # data-driven explanation

@dataclass
class MatchupDeepDive:
    team_a: str
    seed_a: int
    team_b: str
    seed_b: int
    region: str
    round: str         # "R64"
    p_win_a: float
    tempo_clash: bool
    tempo_diff: float
    edge_team: str     # which team has the EM edge
    em_delta: float
    narrative: str     # computed narrative
    recommendation: str  # "HIGH-LEVERAGE UPSET CANDIDATE" etc.

@dataclass
class IntelResponse:
    season: int
    flags: list[dict]
    false_favorites: list[dict]
    cinderellas: list[dict]
    deep_dives: list[dict]
    optimal_path: dict  # R64 differentiators, deep run value

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Luck threshold above which a team's close-game record is unsustainable
LUCK_RISK_THRESHOLD = 0.05
# AdjDE threshold for surging defensive teams (lower = better)
SURGE_ADJDE_THRESHOLD = 96.0
# AdjOE threshold for elite offensive teams
SURGE_ADJOE_THRESHOLD = 122.0
# EM delta at which win probability crosses ~40% (upset territory)
UPSET_EM_THRESHOLD = 6.0
# Logistic scale (must match bracket_runner._win_prob divisor)
_EM_SCALE = 4.0

# R64 seeding pairings: (high_seed, low_seed)
R64_PAIRS = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _win_prob(em_a: float, em_b: float) -> float:
    """Same logistic formula as bracket_runner._win_prob."""
    import math
    return 1.0 / (1.0 + math.exp(-(em_a - em_b) / _EM_SCALE))


def _build_team_stats(trank_df: pd.DataFrame, seeds: dict[str, dict]) -> dict[str, dict]:
    """Merge T-Rank efficiency metrics with bracket seed/region data.

    Returns a dict mapping team_name -> {adj_oe, adj_de, em, tempo, luck, seed, region, conference}
    for every team in the 2026 bracket.
    """
    stats: dict[str, dict] = {}

    if trank_df.empty:
        return stats

    name_col = "team" if "team" in trank_df.columns else trank_df.columns[0]

    # Build a lookup from the DataFrame
    df_lookup: dict[str, dict] = {}
    for _, row in trank_df.iterrows():
        name = str(row[name_col])
        df_lookup[name] = row.to_dict()

    for team_name, seed_info in seeds.items():
        row = df_lookup.get(team_name)
        if row is None:
            continue
        adj_oe = float(row.get("adj_oe", 100.0))
        adj_de = float(row.get("adj_de", 100.0))
        tempo = float(row.get("tempo", 67.0))
        luck = float(row.get("luck", 0.0))
        conference = str(row.get("conference", ""))
        em = adj_oe - adj_de
        stats[team_name] = {
            "adj_oe": adj_oe,
            "adj_de": adj_de,
            "em": em,
            "tempo": tempo,
            "luck": luck,
            "conference": conference,
            "seed": seed_info["seed"],
            "region": seed_info["region"],
        }

    return stats


def _get_seeds_from_bracket() -> dict[str, dict]:
    """Flatten BRACKET_2026 into {team_name: {seed, region}} lookup."""
    seeds: dict[str, dict] = {}
    for region, entries in BRACKET_2026.items():
        for e in entries:
            seeds[str(e["team"])] = {"seed": int(e["seed"]), "region": region}
    return seeds


def _get_r64_pairs(region: str, stats: dict[str, dict]) -> list[tuple[str, str]]:
    """Return R64 matchup pairs for a region in (high_seed, low_seed) order."""
    region_teams = {t: s for t, s in stats.items() if s["region"] == region}
    pairs = []
    for high, low in R64_PAIRS:
        high_team = next((t for t, s in region_teams.items() if s["seed"] == high), None)
        low_team = next((t for t, s in region_teams.items() if s["seed"] == low), None)
        if high_team and low_team:
            pairs.append((high_team, low_team))
    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# Flag generators
# ──────────────────────────────────────────────────────────────────────────────

def _generate_luck_risks(stats: dict[str, dict]) -> list[IntelFlag]:
    """Flag teams with luck > LUCK_RISK_THRESHOLD — close-game record unsustainable."""
    flags = []
    # Rank all teams by EM to detect over-seeding
    em_ranked = sorted(stats.keys(), key=lambda t: stats[t]["em"], reverse=True)
    em_rank = {t: i + 1 for i, t in enumerate(em_ranked)}

    for team, s in stats.items():
        luck = s["luck"]
        if luck > LUCK_RISK_THRESHOLD:
            seed = s["seed"]
            rank = em_rank[team]
            # Severity based on luck magnitude
            if luck > 0.15:
                severity = "EXTREME"
                emoji = "🔴"
            elif luck > 0.08:
                severity = "HIGH"
                emoji = "🚨"
            else:
                severity = "MODERATE"
                emoji = "⚠️"

            # Over-seeded if EM rank is much worse than seed rank
            seed_expected_rank = seed * 4  # rough: seed 1 → top 4, seed 5 → top 20
            over_seeded = rank > seed_expected_rank * 1.5

            detail_parts = [
                f"Luck metric +{luck:.3f} — close-game wins are regressing to mean over 35-game sample.",
                f"AdjEM rank: #{rank} nationally (seed #{seed} → expected ~top {seed_expected_rank}).",
            ]
            if s["adj_de"] > 102:
                detail_parts.append(f"Defensive rating {s['adj_de']:.1f} AdjDE is a liability in neutral-site tournament play.")
            if over_seeded:
                detail_parts.append("Efficiency rank suggests this team is over-seeded relative to their AdjEM.")

            flags.append(IntelFlag(
                type="risk",
                severity=severity,
                team=team,
                seed=seed,
                region=s["region"],
                headline=f"{team} close-game luck (+{luck:.3f}) is unsustainable",
                detail=" ".join(detail_parts),
                metric=f"luck={luck:.3f}, adjEM_rank={rank}",
                emoji=emoji,
            ))

    return sorted(flags, key=lambda f: (
        {"EXTREME": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}[f.severity]
    ))


def _generate_defensive_alerts(stats: dict[str, dict]) -> list[IntelFlag]:
    """Flag high-seed teams with weak or collapsed AdjDE."""
    flags = []
    for team, s in stats.items():
        seed = s["seed"]
        # Only flag teams seeded 1–6 with poor defense — these are "expected" to advance
        if seed > 6:
            continue
        if s["adj_de"] > 103:
            severity = "HIGH" if s["adj_de"] > 106 else "MODERATE"
            flags.append(IntelFlag(
                type="alert",
                severity=severity,
                team=team,
                seed=seed,
                region=s["region"],
                headline=f"{team} (#{seed}) defensive rating {s['adj_de']:.1f} AdjDE is a tournament liability",
                detail=(
                    f"AdjDE: {s['adj_de']:.1f} is outside top-50 nationally. "
                    f"Tournament games on neutral courts amplify defensive vulnerabilities. "
                    f"AdjOE {s['adj_oe']:.1f} compensates, but offense-only profiles carry high variance."
                ),
                metric=f"adj_de={s['adj_de']:.1f}",
                emoji="📉",
            ))
    return flags


def _generate_surge_flags(stats: dict[str, dict]) -> list[IntelFlag]:
    """Flag teams with elite two-way efficiency profiles — surging contenders."""
    flags = []
    for team, s in stats.items():
        if s["adj_oe"] >= SURGE_ADJOE_THRESHOLD and s["adj_de"] <= SURGE_ADJDE_THRESHOLD:
            seed = s["seed"]
            em = s["em"]
            flags.append(IntelFlag(
                type="surge",
                severity="LOW",
                team=team,
                seed=seed,
                region=s["region"],
                headline=f"{team} elite two-way profile: AdjOE {s['adj_oe']:.1f} + AdjDE {s['adj_de']:.1f}",
                detail=(
                    f"Efficiency margin +{em:.1f} places {team} among the strongest profiles in the field. "
                    f"Both offense ({s['adj_oe']:.1f}) and defense ({s['adj_de']:.1f}) rank in top tier nationally. "
                    f"Extremely dangerous in single-elimination format."
                ),
                metric=f"adj_oe={s['adj_oe']:.1f}, adj_de={s['adj_de']:.1f}, em=+{em:.1f}",
                emoji="📈",
            ))
    return flags


def _generate_false_favorites(stats: dict[str, dict]) -> list[dict]:
    """Generate false favorites: teams whose EM rank doesn't justify their seed or high luck."""
    # Rank all teams by EM
    em_ranked = sorted(stats.keys(), key=lambda t: stats[t]["em"], reverse=True)
    em_rank = {t: i + 1 for i, t in enumerate(em_ranked)}

    false_favs = []
    for team, s in stats.items():
        seed = s["seed"]
        rank = em_rank[team]
        luck = s["luck"]

        # Over-seeded: seed top-4 but EM rank is 20+ worse than expected
        seed_expected_rank = seed * 4
        is_overseed = rank > seed_expected_rank * 2.0 and seed <= 5
        is_lucky = luck > LUCK_RISK_THRESHOLD and seed <= 5
        is_def_weak = s["adj_de"] > 104 and seed <= 4

        if not (is_overseed or is_lucky or is_def_weak):
            continue

        risk_parts = []
        severity = "MODERATE"

        if is_lucky:
            risk_parts.append(f"Barttorvik luck +{luck:.3f} — close-game record will regress")
            severity = "HIGH" if luck > 0.08 else severity

        if is_overseed:
            risk_parts.append(f"AdjEM rank #{rank} nationally vs seed #{seed} (expected ~top {seed_expected_rank})")
            severity = "HIGH"

        if is_def_weak:
            risk_parts.append(f"AdjDE {s['adj_de']:.1f} (weak for a {seed}-seed in March)")
            severity = "HIGH" if s["adj_de"] > 106 else severity

        false_favs.append({
            "team": team,
            "seed": seed,
            "region": s["region"],
            "seed_label": f"#{seed} ({s['region']})",
            "risk_level": severity,
            "detail": ". ".join(risk_parts) + ".",
            "em": round(s["em"], 1),
            "luck": round(luck, 3),
            "adj_de": round(s["adj_de"], 1),
        })

    false_favs.sort(key=lambda x: (
        {"EXTREME": 0, "HIGH": 1, "MODERATE": 2}[x["risk_level"]], x["seed"]
    ))
    return false_favs[:6]  # top 6


def _generate_cinderellas(stats: dict[str, dict]) -> list[CinderellaEntry]:
    """Generate Cinderella watchlist: low-seed teams with strong EM vs R64 opponent."""
    cinderellas = []

    for region in ["East", "South", "West", "Midwest"]:
        pairs = _get_r64_pairs(region, stats)
        for fav, dog in pairs:
            s_fav = stats.get(fav)
            s_dog = stats.get(dog)
            if not s_fav or not s_dog:
                continue

            seed_diff = s_dog["seed"] - s_fav["seed"]
            if seed_diff < 5:  # only low-seed upsets (5-12, 4-13, etc.)
                continue

            p_win_dog = _win_prob(s_dog["em"], s_fav["em"])
            if p_win_dog < 0.20:  # filter out no-hopers
                continue

            upset_pct = round(p_win_dog * 100, 1)
            em_delta = round(s_dog["em"] - s_fav["em"], 1)

            edge_parts = []
            if s_dog["adj_de"] < s_fav["adj_de"]:
                edge_parts.append(f"superior defense (AdjDE {s_dog['adj_de']:.1f} vs {s_fav['adj_de']:.1f})")
            if abs(s_dog["tempo"] - s_fav["tempo"]) > 5:
                slower = s_dog if s_dog["tempo"] < s_fav["tempo"] else s_fav
                faster = s_dog if s_dog["tempo"] > s_fav["tempo"] else s_fav
                edge_parts.append(f"tempo clash ({faster.get('name', faster)} fast vs {slower.get('name', slower)} slow — fewer possessions = higher variance)")
            if s_dog["luck"] < 0.02:
                edge_parts.append(f"near-zero luck ({s_dog['luck']:.3f}) suggests sustainable performance")
            if not edge_parts:
                edge_parts.append(f"efficiency margin {em_delta:+.1f} relative to opponent")

            cinderellas.append(CinderellaEntry(
                team=dog,
                seed=s_dog["seed"],
                region=region,
                opponent=fav,
                opponent_seed=s_fav["seed"],
                upset_pct=upset_pct,
                edge_summary=". ".join(edge_parts).capitalize() + ".",
            ))

    cinderellas.sort(key=lambda c: c.upset_pct, reverse=True)
    return cinderellas[:8]


def _generate_deep_dives(stats: dict[str, dict]) -> list[MatchupDeepDive]:
    """Generate narrative deep dives for the most interesting R64 matchups."""
    dives = []

    for region in ["East", "South", "West", "Midwest"]:
        pairs = _get_r64_pairs(region, stats)
        for fav, dog in pairs:
            s_fav = stats.get(fav)
            s_dog = stats.get(dog)
            if not s_fav or not s_dog:
                continue

            seed_diff = s_dog["seed"] - s_fav["seed"]
            if seed_diff < 5:
                continue  # only write up interesting matchups

            p_win_fav = _win_prob(s_fav["em"], s_dog["em"])
            tempo_diff = abs(s_fav["tempo"] - s_dog["tempo"])
            tempo_clash = tempo_diff > 5.0
            em_delta = s_fav["em"] - s_dog["em"]
            edge_team = fav if em_delta > 0 else dog

            # Build narrative from data
            narrative_parts = []
            if tempo_clash:
                slow = fav if s_fav["tempo"] < s_dog["tempo"] else dog
                fast = fav if s_fav["tempo"] > s_dog["tempo"] else dog
                s_slow = s_fav if slow == fav else s_dog
                s_fast = s_fav if fast == fav else s_dog
                narrative_parts.append(
                    f"Tempo clash: {fast} (pace {s_fast['tempo']:.0f}) vs {slow} ({s_slow['tempo']:.0f} poss/40). "
                    f"Fewer possessions = higher single-game variance for the {dog}."
                )
            narrative_parts.append(
                f"Efficiency delta: {fav} EM +{s_fav['em']:.1f} vs {dog} EM +{s_dog['em']:.1f} ({em_delta:+.1f})."
            )
            if s_dog["luck"] < 0.02:
                narrative_parts.append(
                    f"{dog}'s near-zero luck ({s_dog['luck']:.3f}) indicates performance is skill-driven, not variance-inflated."
                )
            if s_fav["luck"] > LUCK_RISK_THRESHOLD:
                narrative_parts.append(
                    f"{fav}'s luck +{s_fav['luck']:.3f} suggests close-game overperformance that won't sustain."
                )

            # Recommendation
            upset_prob = 1 - p_win_fav
            if upset_prob > 0.38:
                recommendation = f"HIGH-VALUE LEVERAGE: {dog} is an upset candidate with {upset_prob*100:.0f}% probability — differentiate in large bracket contests."
            elif upset_prob > 0.28:
                recommendation = f"MODERATE LEVERAGE: {dog} ({upset_prob*100:.0f}% upset probability) worth targeting in large-field bracket contests."
            else:
                recommendation = f"CHALK: {fav} is the safe pick at {p_win_fav*100:.0f}% win probability."

            dives.append(MatchupDeepDive(
                team_a=fav,
                seed_a=s_fav["seed"],
                team_b=dog,
                seed_b=s_dog["seed"],
                region=region,
                round="R64",
                p_win_a=round(p_win_fav, 3),
                tempo_clash=tempo_clash,
                tempo_diff=round(tempo_diff, 1),
                edge_team=edge_team,
                em_delta=round(em_delta, 1),
                narrative=" ".join(narrative_parts),
                recommendation=recommendation,
            ))

    # Sort by upset probability descending
    dives.sort(key=lambda d: 1 - d.p_win_a, reverse=True)
    return dives[:6]


def _generate_optimal_path(stats: dict[str, dict]) -> dict:
    """Identify highest-leverage R64 picks based on EM differentials."""
    r64_upsets = []
    deep_run = []

    for region in ["East", "South", "West", "Midwest"]:
        pairs = _get_r64_pairs(region, stats)
        for fav, dog in pairs:
            s_fav = stats.get(fav)
            s_dog = stats.get(dog)
            if not s_fav or not s_dog:
                continue
            p_dog = _win_prob(s_dog["em"], s_fav["em"])
            if p_dog >= 0.30:
                r64_upsets.append({
                    "winner": dog,
                    "loser": fav,
                    "winner_seed": s_dog["seed"],
                    "loser_seed": s_fav["seed"],
                    "region": region,
                    "upset_pct": round(p_dog * 100, 1),
                })

    r64_upsets.sort(key=lambda x: x["upset_pct"], reverse=True)

    # Deep run value: top EM teams seeded 2-6
    em_ranked = sorted(stats.items(), key=lambda t: t[1]["em"], reverse=True)
    for team, s in em_ranked[:20]:
        if 2 <= s["seed"] <= 6:
            deep_run.append({
                "team": team,
                "seed": s["seed"],
                "region": s["region"],
                "em": round(s["em"], 1),
                "adj_oe": round(s["adj_oe"], 1),
                "adj_de": round(s["adj_de"], 1),
            })
        if len(deep_run) >= 5:
            break

    # Championship edge: top 2 by EM
    top2 = em_ranked[:2]
    championship_edge = None
    if len(top2) >= 2:
        t1, s1 = top2[0]
        t2, s2 = top2[1]
        p1 = _win_prob(s1["em"], s2["em"])
        championship_edge = {
            "team_a": t1,
            "team_b": t2,
            "p_a": round(p1 * 100, 1),
            "p_b": round((1 - p1) * 100, 1),
            "note": f"Pick {t1} in small bracket contests (<1000 entries). Pick {t2} for Upset Edge advantage in mega-contests.",
        }

    return {
        "r64_differentiators": r64_upsets[:5],
        "deep_run_value": deep_run,
        "championship_edge": championship_edge,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_intel(season: int, loader: Any = None) -> IntelResponse:
    """Build the full intel payload for a given season.

    Parameters
    ----------
    season:
        NCAA season year (e.g. 2026).
    loader:
        DataLoader instance with get_trank(season) returning a DataFrame.
        If None, a new DataLoader is created.
    """
    if loader is None:
        from src.api.data_cache import DataLoader
        loader = DataLoader()

    trank_df: pd.DataFrame
    try:
        trank_df = loader.get_trank(season)
    except Exception as exc:
        logger.error("Intel engine: failed to load T-Rank data for %s: %s", season, exc)
        raise

    seeds = _get_seeds_from_bracket()
    stats = _build_team_stats(trank_df, seeds)

    if not stats:
        raise RuntimeError(f"No team stats could be built for season {season} — T-Rank data may not be available yet.")

    luck_risks = _generate_luck_risks(stats)
    def_alerts = _generate_defensive_alerts(stats)
    surge_flags = _generate_surge_flags(stats)
    all_flags = luck_risks + def_alerts + surge_flags

    false_favorites = _generate_false_favorites(stats)
    cinderellas = _generate_cinderellas(stats)
    deep_dives = _generate_deep_dives(stats)
    optimal_path = _generate_optimal_path(stats)

    return IntelResponse(
        season=season,
        flags=[asdict(f) for f in all_flags],
        false_favorites=false_favorites,
        cinderellas=[asdict(c) for c in cinderellas],
        deep_dives=[asdict(d) for d in deep_dives],
        optimal_path=optimal_path,
    )
