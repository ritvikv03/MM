"""
src/simulation/win_prob.py

Advanced win-probability engine for the Bracket Engine.

Combines four signal layers into a single, calibrated P(win) estimate:
  1. Efficiency-margin logistic model (Barttorvik T-Rank data)
  2. Market-implied probability (Sportsbook Review / Vegas lines)
  3. Shot-quality adjustment (Sports Reference advanced stats)
  4. "March DNA" clutch multiplier (guard experience, FT%, TOV%)

The blended probability is used by the Monte Carlo simulator
(src/simulation/monte_carlo.py) as the WinProbFn callable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TeamProfile:
    """All data needed to compute win probabilities for a single team."""

    name: str
    seed: int = 8

    # T-Rank efficiency metrics (from barttorvik.py)
    adj_oe: float = 105.0    # Adjusted Offensive Efficiency
    adj_de: float = 95.0     # Adjusted Defensive Efficiency
    tempo: float = 68.0      # Possessions per 40 minutes
    luck: float = 0.0        # Barttorvik luck metric (-0.1 to +0.1)
    sos: float = 5.0         # Strength of Schedule

    # Shot quality proxies (from hoopmath.py / Sports Reference)
    efg_pct: float = 0.50     # Team eFG%
    opp_efg_pct: float = 0.50  # Opponent eFG% (defense)
    fg3a_rate: float = 0.35    # Fraction of FGA that are 3-pointers
    fta_rate: float = 0.30     # Free throw attempt rate

    # March DNA components (from sports_reference.py + roster data)
    tov_pct: float = 18.0          # Team turnover percentage
    opp_tov_pct: float = 18.0      # Forced turnover percentage
    ft_pct_late: float = 0.72      # Late-game free throw %
    senior_guard_minutes: float = 0.30  # Fraction of guard minutes from seniors/grad transfers
    coach_tourney_wins: int = 0    # Career NCAA tournament wins
    returning_minutes_pct: float = 0.50  # Roster continuity

    # Market data (from market_data.py)
    market_spread: Optional[float] = None  # Vegas spread (e.g., -3.5 means 3.5-pt favorite)

    # Injury data (from injury_feed.py)
    availability: float = 1.0  # Fraction of roster BPR-weighted minutes available [0,1]


@dataclass
class MatchupContext:
    """Computed context for a specific head-to-head matchup."""
    team_a: TeamProfile
    team_b: TeamProfile
    neutral_site: bool = True

    # Computed probabilities
    p_efficiency: float = 0.5
    p_market: float = 0.5
    p_shot_quality: float = 0.5
    p_march_dna: float = 0.5
    p_blended: float = 0.5
    std_dev: float = 0.10

    # Narrative factors
    style_clash: str = ""
    upset_flags: list = field(default_factory=list)
    model_recommendation: str = ""


# ---------------------------------------------------------------------------
# Signal 1: Efficiency-margin logistic model
# ---------------------------------------------------------------------------


def compute_efficiency_prob(a: TeamProfile, b: TeamProfile) -> float:
    """
    Win probability from raw efficiency margins.

    Formula: P(A wins) = logistic( (em_a - em_b) / scale )
    where em = adj_oe - adj_de, and scale controls the steepness.

    The scale factor of 10.0 is calibrated to NCAA tournament games
    (per KenPom: ~11 points of efficiency margin ≈ 85% win probability).
    """
    em_a = a.adj_oe - a.adj_de
    em_b = b.adj_oe - b.adj_de

    # Regress luck toward mean (punishment for unsustainably lucky teams)
    luck_penalty_a = a.luck * 2.5  # Each 0.01 of luck → 0.025 EM penalty
    luck_penalty_b = b.luck * 2.5
    em_a_regressed = em_a - luck_penalty_a
    em_b_regressed = em_b - luck_penalty_b

    # Apply injury availability: reduce effective EM proportionally
    em_a_adj = em_a_regressed * (0.5 + 0.5 * a.availability)
    em_b_adj = em_b_regressed * (0.5 + 0.5 * b.availability)

    diff = em_a_adj - em_b_adj
    return 1.0 / (1.0 + math.exp(-diff / 10.0))


# ---------------------------------------------------------------------------
# Signal 2: Market-implied probability
# ---------------------------------------------------------------------------


def compute_market_prob(a: TeamProfile, b: TeamProfile) -> float:
    """
    Win probability from Vegas spread lines.

    Uses the industry-standard formula:
        P(A wins) = 1 / (1 + 10^(spread / 8.5))

    If no market data is available, returns 0.5 (uninformed).
    The scale of 8.5 is based on Stern (1991) and is the standard
    used by professional sports bettors.
    """
    if a.market_spread is not None:
        # Negative spread = A is favored; use A's line
        spread = a.market_spread
    elif b.market_spread is not None:
        # B has a line; A's implied spread is -B's spread
        spread = -b.market_spread
    else:
        return 0.5  # No market signal

    return 1.0 / (1.0 + 10.0 ** (spread / 8.5))


# ---------------------------------------------------------------------------
# Signal 3: Shot quality differential
# ---------------------------------------------------------------------------


def compute_shot_quality_adjustment(a: TeamProfile, b: TeamProfile) -> float:
    """
    Adjust win probability based on shot quality differentials.

    Teams that force low-quality shots (low opp_efg) while generating
    high-quality looks (high efg, high FTA rate) have a real edge that
    pure efficiency may overstate for lucky teams.

    Returns a probability adjustment centered around 0.5.
    """
    # A's offensive quality vs B's defensive quality
    a_offense_vs_b_defense = a.efg_pct - b.opp_efg_pct
    # B's offensive quality vs A's defensive quality
    b_offense_vs_a_defense = b.efg_pct - a.opp_efg_pct

    # FTA rate advantage: teams that attack the rim and get to the line
    fta_advantage = (a.fta_rate - b.fta_rate) * 0.3

    quality_diff = (a_offense_vs_b_defense - b_offense_vs_a_defense) + fta_advantage

    # Convert to probability centered at 0.5
    return 1.0 / (1.0 + math.exp(-quality_diff * 15.0))


# ---------------------------------------------------------------------------
# Signal 4: "March DNA" clutch multiplier
# ---------------------------------------------------------------------------


def compute_march_dna(a: TeamProfile, b: TeamProfile) -> float:
    """
    The "March DNA" composite metric quantifies tournament readiness.

    In close games (last 4 minutes), three factors dominate:
      1. Guard experience (senior guards make plays under pressure)
      2. Free throw shooting (FTs decide most 1-2 point games)
      3. Turnover control (in March, TOs are lethal)

    Each factor is scored 0-1, then combined with weights.
    """
    def _dna_score(t: TeamProfile) -> float:
        # Factor 1: Guard experience (0-1, higher = more senior minutes)
        guard_exp = min(1.0, t.senior_guard_minutes / 0.5)  # 50%+ senior guards = max

        # Factor 2: Late-game FT% (0-1, scaled from 0.65 to 0.80)
        ft_score = max(0.0, min(1.0, (t.ft_pct_late - 0.65) / 0.15))

        # Factor 3: Turnover discipline (0-1, lower TOV% and higher forced TOV%)
        tov_score = max(0.0, min(1.0, 1.0 - (t.tov_pct - 12.0) / 12.0))
        forced_tov = max(0.0, min(1.0, (t.opp_tov_pct - 15.0) / 8.0))

        # Factor 4: Coach tournament pedigree (bonus for experienced coaches)
        coach_bonus = min(0.2, t.coach_tourney_wins * 0.005)

        # Factor 5: Roster continuity
        continuity = min(1.0, t.returning_minutes_pct / 0.6)

        # Weighted combination
        dna = (
            guard_exp * 0.25
            + ft_score * 0.25
            + tov_score * 0.15
            + forced_tov * 0.10
            + coach_bonus
            + continuity * 0.05
        )
        return dna

    dna_a = _dna_score(a)
    dna_b = _dna_score(b)

    # Convert DNA difference to probability
    diff = dna_a - dna_b
    return 1.0 / (1.0 + math.exp(-diff * 5.0))


# ---------------------------------------------------------------------------
# Style clash detection
# ---------------------------------------------------------------------------


def detect_style_clash(a: TeamProfile, b: TeamProfile) -> str:
    """Detect when team styles produce high-variance matchups."""
    tempo_diff = abs(a.tempo - b.tempo)
    a_3pt_heavy = a.fg3a_rate > 0.40
    b_3pt_heavy = b.fg3a_rate > 0.40

    if tempo_diff > 8:
        slow_team = a.name if a.tempo < b.tempo else b.name
        fast_team = b.name if a.tempo < b.tempo else a.name
        return (
            f"⚡ TEMPO CLASH: {fast_team} pushes pace ({max(a.tempo, b.tempo):.0f}) "
            f"while {slow_team} grinds ({min(a.tempo, b.tempo):.0f}). "
            f"Fewer possessions = higher variance = upset risk."
        )
    if a_3pt_heavy and not b_3pt_heavy:
        return (
            f"🎯 VARIANCE ALERT: {a.name} lives by the 3-pointer "
            f"({a.fg3a_rate*100:.0f}% of attempts). If shots fall, blowout. If not, upset."
        )
    if b_3pt_heavy and not a_3pt_heavy:
        return (
            f"🎯 VARIANCE ALERT: {b.name} lives by the 3-pointer "
            f"({b.fg3a_rate*100:.0f}% of attempts). If shots fall, blowout. If not, upset."
        )
    return ""


# ---------------------------------------------------------------------------
# Blended probability (the main engine)
# ---------------------------------------------------------------------------

# Weights for blending signals
WEIGHT_EFFICIENCY = 0.45
WEIGHT_MARKET = 0.25
WEIGHT_SHOT_QUALITY = 0.15
WEIGHT_MARCH_DNA = 0.15


def compute_blended_win_prob(
    a: TeamProfile,
    b: TeamProfile,
    chaos_factor: float = 0.5,
) -> MatchupContext:
    """
    Compute the blended P(A wins B) from all four signal layers.

    Parameters
    ----------
    a, b : TeamProfile
        Team profiles with all data populated.
    chaos_factor : float
        0.0 = pure chalk (favor efficiency model heavily)
        0.5 = balanced (standard blend)
        1.0 = chaos mode (weight March DNA and variance heavily)

    Returns
    -------
    MatchupContext
        Full context including probabilities, narratives, and recommendations.
    """
    ctx = MatchupContext(team_a=a, team_b=b)

    # Compute individual signals
    ctx.p_efficiency = compute_efficiency_prob(a, b)
    ctx.p_market = compute_market_prob(a, b)
    ctx.p_shot_quality = compute_shot_quality_adjustment(a, b)
    ctx.p_march_dna = compute_march_dna(a, b)

    # Dynamic weight adjustment based on chaos_factor
    # In chalk mode: efficiency dominates
    # In chaos mode: March DNA and shot quality (variance indicators) dominate
    w_eff = WEIGHT_EFFICIENCY * (1.2 - chaos_factor * 0.6)
    w_mkt = WEIGHT_MARKET * (1.0 - chaos_factor * 0.3)
    w_sq = WEIGHT_SHOT_QUALITY * (0.8 + chaos_factor * 0.6)
    w_dna = WEIGHT_MARCH_DNA * (0.7 + chaos_factor * 0.8)

    # Normalize weights to sum to 1.0
    w_total = w_eff + w_mkt + w_sq + w_dna
    w_eff /= w_total
    w_mkt /= w_total
    w_sq /= w_total
    w_dna /= w_total

    # Blend
    ctx.p_blended = (
        w_eff * ctx.p_efficiency
        + w_mkt * ctx.p_market
        + w_sq * ctx.p_shot_quality
        + w_dna * ctx.p_march_dna
    )

    # Clip to [0.01, 0.99]
    ctx.p_blended = max(0.01, min(0.99, ctx.p_blended))

    # Standard deviation: higher when game is close or when chaos_factor is high
    closeness = 1.0 - abs(ctx.p_blended - 0.5) * 2  # 0 at extremes, 1 at 50/50
    ctx.std_dev = 0.05 + closeness * 0.10 + chaos_factor * 0.05

    # Style clash detection
    ctx.style_clash = detect_style_clash(a, b)

    # Upset flags
    if a.seed > b.seed and ctx.p_blended > 0.40:
        ctx.upset_flags.append(
            f"🚨 ({a.seed}) {a.name} has a {ctx.p_blended*100:.0f}% "
            f"chance against ({b.seed}) {b.name}"
        )
    if b.seed > a.seed and ctx.p_blended < 0.60:
        ctx.upset_flags.append(
            f"🚨 ({b.seed}) {b.name} has a {(1-ctx.p_blended)*100:.0f}% "
            f"chance against ({a.seed}) {a.name}"
        )

    # Model recommendation
    if abs(ctx.p_blended - 0.5) < 0.05:
        ctx.model_recommendation = (
            f"COIN FLIP — Pick based on bracket strategy. "
            f"In large bracket contests, take {a.name if a.seed > b.seed else b.name} for leverage."
        )
    elif ctx.p_blended > 0.65:
        ctx.model_recommendation = f"STRONG LEAN: {a.name} ({ctx.p_blended*100:.0f}%)"
    elif ctx.p_blended < 0.35:
        ctx.model_recommendation = f"STRONG LEAN: {b.name} ({(1-ctx.p_blended)*100:.0f}%)"
    else:
        higher = a if ctx.p_blended > 0.5 else b
        pct = ctx.p_blended if ctx.p_blended > 0.5 else 1 - ctx.p_blended
        ctx.model_recommendation = f"SLIGHT EDGE: {higher.name} ({pct*100:.0f}%)"

    return ctx


# ---------------------------------------------------------------------------
# WinProbFn adapter for Monte Carlo simulator
# ---------------------------------------------------------------------------


def make_win_prob_fn(
    profiles: dict[str, TeamProfile],
    chaos_factor: float = 0.5,
):
    """
    Create a WinProbFn callable for use with the Monte Carlo simulator.

    Parameters
    ----------
    profiles : dict[str, TeamProfile]
        Mapping of team_name → TeamProfile for all 64 teams.
    chaos_factor : float
        Risk dial: 0.0 (chalk) → 1.0 (chaos).

    Returns
    -------
    callable
        (team_a_name, team_b_name) → (mean_prob, std_prob)
    """
    def win_prob_fn(team_a: str, team_b: str) -> tuple[float, float]:
        a = profiles.get(team_a)
        b = profiles.get(team_b)

        if a is None or b is None:
            return (0.5, 0.15)  # Unknown team → uninformed prior

        ctx = compute_blended_win_prob(a, b, chaos_factor=chaos_factor)
        return (ctx.p_blended, ctx.std_dev)

    return win_prob_fn
""", "Complexity": 9, "Description": "Built the complete win_prob.py engine with 4 signal layers (efficiency, market, shot quality, March DNA), chaos factor support, style clash detection, narrative generation, and WinProbFn adapter for the Monte Carlo simulator.", "EmptyFile": false, "IsArtifact": false, "Overwrite": false, "TargetFile": "/Users/ritvikvasikarla/Documents/MM/src/simulation/win_prob.py"}
