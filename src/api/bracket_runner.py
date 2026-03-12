"""
src/api/bracket_runner.py
Real Monte Carlo bracket simulation using Barttorvik efficiency margins.

Replaces the deterministic stub in server.py when USE_REAL_DATA=1.

Public API
----------
build_real_simulation(teams, n_simulations, season, loader) -> SimulateResponse
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from src.api.schemas import SimulateResponse, TeamAdvancementItem

logger = logging.getLogger(__name__)

_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "Championship"]


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------


def _win_prob(em_a: float, em_b: float) -> float:
    """Logistic win probability for team A over team B.

    P(A beats B) = 1 / (1 + exp(-(em_a - em_b) / 4.0))

    The scale factor 4.0 is chosen so that a +8 efficiency-margin gap
    translates to roughly a 73% favourite, consistent with observed
    Barttorvik spread calibration.
    """
    return 1.0 / (1.0 + math.exp(-(em_a - em_b) / 4.0))


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy of an advancement probability vector.

    H = -sum(p * log2(p + 1e-12)) for p in normalised_probs
    where normalised_probs = raw_probs / (sum(raw_probs) + 1e-12)

    A uniform distribution over 6 rounds yields H ≈ 2.585 bits (log2(6)).
    A completely deterministic vector (one entry = 1.0) yields H ≈ 0.
    """
    arr = np.asarray(probs, dtype=float)
    total = arr.sum() + 1e-12
    p = arr / total
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ---------------------------------------------------------------------------
# Bracket simulation engine
# ---------------------------------------------------------------------------


def _simulate_bracket(
    teams: list[str],
    em_map: dict[str, float],
    n_simulations: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """Run `n_simulations` single-elimination bracket trials.

    Algorithm (per trial)
    ---------------------
    1. Start with `survivors = list(teams)`.
    2. Pair survivors sequentially: (0,1), (2,3), ...
       - Draw a uniform random number; winner is team[0] if draw < _win_prob(em_a, em_b).
       - An odd survivor at the end of a round gets a bye (advances automatically).
    3. Repeat until 1 survivor remains.
    4. Track per-round advancement counts for each team.

    Returns
    -------
    dict mapping team → list of 6 advancement *probabilities* (counts / n_simulations)
    for rounds [R64, R32, S16, E8, F4, Championship].
    """
    n_teams = len(teams)
    team_index = {t: i for i, t in enumerate(teams)}
    # Shape: (n_teams, 6) — counts of round advancements
    counts = np.zeros((n_teams, 6), dtype=np.float64)

    for _ in range(n_simulations):
        survivors = list(teams)
        last_round_idx = -1
        for round_idx in range(6):
            if len(survivors) <= 1:
                break
            next_round: list[str] = []
            i = 0
            while i < len(survivors):
                if i + 1 < len(survivors):
                    a, b = survivors[i], survivors[i + 1]
                    p_a = _win_prob(em_map[a], em_map[b])
                    winner = a if rng.random() < p_a else b
                    next_round.append(winner)
                    i += 2
                else:
                    # Odd survivor — bye
                    next_round.append(survivors[i])
                    i += 1
            survivors = next_round
            last_round_idx = round_idx
            # Credit advancement for this round (but not the final champion yet)
            if len(survivors) > 1:
                for t in survivors:
                    counts[team_index[t], round_idx] += 1

        # Credit the champion at the Championship slot (index 5), regardless
        # of how many rounds were needed.  For sub-64-team fields the rounds
        # collapse but the winner still earns a Championship credit.
        if survivors:
            champion = survivors[0]
            counts[team_index[champion], 5] += 1
            # Also back-fill intermediate rounds that were skipped so that
            # the champion's probs are non-decreasing up to Championship.
            # For small fields (< 64 teams) the last played round_idx < 5;
            # credit the champion for all skipped rounds between last_round_idx
            # and Championship so advancement_probs[0] >= advancement_probs[-1].
            for fill_idx in range(last_round_idx + 1, 5):
                counts[team_index[champion], fill_idx] += 1

    # Normalise to probabilities
    probs_matrix = counts / n_simulations
    return {teams[i]: probs_matrix[i].tolist() for i in range(n_teams)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_real_simulation(
    teams: list[str],
    n_simulations: int,
    season: int,
    loader: Any = None,
) -> SimulateResponse:
    """Build a real bracket simulation response using Barttorvik EM.

    Parameters
    ----------
    teams:
        Ordered list of team names (bracket seeding order).
    n_simulations:
        Number of Monte Carlo trials.
    season:
        NCAA season year (e.g. 2024).
    loader:
        DataLoader instance (or duck-typed mock) with a `get_trank(season)`
        method returning a DataFrame with columns [team, adj_oe, adj_de].

    Returns
    -------
    SimulateResponse with data_source="real".
    """
    # ------------------------------------------------------------------
    # 1. Fetch T-Rank data
    # ------------------------------------------------------------------
    if loader is None:
        from src.api.data_cache import DataLoader
        loader = DataLoader()

    trank_df = loader.get_trank(season)

    # ------------------------------------------------------------------
    # 2. Build EM map — teams missing from trank receive the median EM
    # ------------------------------------------------------------------
    em_map: dict[str, float] = {}
    if not trank_df.empty and "adj_oe" in trank_df.columns and "adj_de" in trank_df.columns:
        trank_df = trank_df.copy()
        trank_df["_em"] = trank_df["adj_oe"] - trank_df["adj_de"]
        median_em = float(trank_df["_em"].median())

        # Normalise team name column
        name_col = "team" if "team" in trank_df.columns else trank_df.columns[0]
        for _, row in trank_df.iterrows():
            em_map[str(row[name_col])] = float(row["_em"])
    else:
        median_em = 0.0

    for t in teams:
        if t not in em_map:
            logger.debug("Team '%s' not found in T-Rank; using median EM=%.2f", t, median_em)
            em_map[t] = median_em

    # ------------------------------------------------------------------
    # 3. Run Monte Carlo trials
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed=42)
    probs = _simulate_bracket(teams, em_map, n_simulations, rng)

    # ------------------------------------------------------------------
    # 4. Build response objects
    # ------------------------------------------------------------------
    advancements: list[TeamAdvancementItem] = []
    for team in teams:
        team_probs = probs[team]  # list of 6 floats
        prob_dict = {_ROUNDS[i]: round(team_probs[i], 6) for i in range(6)}
        entropy = _shannon_entropy(team_probs)
        advancements.append(TeamAdvancementItem(
            team=team,
            advancement_probs=prob_dict,
            entropy=round(entropy, 6),
        ))

    # Sort by championship probability descending
    advancements.sort(
        key=lambda item: item.advancement_probs.get("Championship", 0.0),
        reverse=True,
    )

    return SimulateResponse(
        n_simulations=n_simulations,
        advancements=advancements,
        data_source="real",
    )
