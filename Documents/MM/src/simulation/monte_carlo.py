"""
src/simulation/monte_carlo.py

Monte Carlo bracket simulation engine for NCAA March Madness.

Simulates the full 64-team bracket by drawing per-game win probabilities
from Normal distributions (as provided by the Bayesian head) and running
n_simulations independent bracket realizations to build aggregate statistics.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

WinProbFn = Callable[[str, str], tuple[float, float]]


# ---------------------------------------------------------------------------
# Standalone utility functions
# ---------------------------------------------------------------------------


def default_win_prob_fn(team_a: str, team_b: str) -> tuple[float, float]:
    """Baseline stub win-probability function.

    Returns (mean=0.5, std=0.1) regardless of the teams, representing a
    completely uninformed prior.  Useful for testing and sanity checks.

    Parameters
    ----------
    team_a:
        Name of the first team (challenger / home / left side).
    team_b:
        Name of the second team.

    Returns
    -------
    tuple[float, float]
        (mean_win_prob_for_team_a, std_win_prob_for_team_a)
    """
    return (0.5, 0.1)


def compute_bracket_entropy(champion_probs: dict[str, float]) -> float:
    """Compute the Shannon entropy (bits) of a champion probability distribution.

    Entropy measures how uncertain the champion outcome is:
    - 0 bits  → one team is a certain champion
    - log2(64) bits → all 64 teams equally likely

    Parameters
    ----------
    champion_probs:
        Mapping of team name → probability of winning the championship.
        Zero-probability entries are silently skipped (they contribute 0 to
        entropy by convention: 0·log2(0) = 0).

    Returns
    -------
    float
        Shannon entropy in bits.
    """
    entropy = 0.0
    for p in champion_probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def build_bracket_structure(seeds: dict[str, int]) -> list[list[str]]:
    """Organize 64 teams into a standard 4-region × 16-team bracket structure.

    Each region receives one team of each seed (1-16).  Teams are placed into
    their region in NCAA seeded order: [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3,
    14, 7, 10, 2, 15], which encodes the standard first-round matchup pairings.

    Parameters
    ----------
    seeds:
        Mapping of team_name → seed number (1–16).  Must contain exactly 64
        teams with each seed represented exactly 4 times (once per region).

    Returns
    -------
    list[list[str]]
        4-element list where each element is a list of 16 team names for that
        region, arranged in seeded matchup order.

    Raises
    ------
    ValueError
        If the number of teams is not exactly 64.
    ValueError
        If any seed value falls outside the valid range [1, 16].
    """
    if len(seeds) != 64:
        raise ValueError(
            f"Expected exactly 64 teams, got {len(seeds)}. "
            "The bracket requires one team per seed (1-16) per region (4 regions)."
        )

    # Validate all seed values
    for team, seed in seeds.items():
        if seed < 1 or seed > 16:
            raise ValueError(
                f"Invalid seed {seed} for team '{team}'. Seeds must be in [1, 16]."
            )

    # Group teams by seed
    seed_to_teams: dict[int, list[str]] = {s: [] for s in range(1, 17)}
    for team, seed in seeds.items():
        seed_to_teams[seed].append(team)

    # Standard NCAA seeded order within a region
    seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    # Build 4 regions; region index i gets the i-th team from each seed bucket
    bracket: list[list[str]] = [[] for _ in range(4)]
    for s in range(1, 17):
        teams_for_seed = seed_to_teams[s]
        for region_idx in range(4):
            bracket[region_idx].append(teams_for_seed[region_idx])

    # Reorder each region to match the seeded order
    # At this point each region contains teams indexed by seed position.
    # We need to map seed → team for this region, then emit in seed_order.
    structured_bracket: list[list[str]] = []
    for region_idx in range(4):
        # bracket[region_idx] has teams in seed 1..16 order (index = seed-1)
        seed_indexed = {s: bracket[region_idx][s - 1] for s in range(1, 17)}
        region_teams = [seed_indexed[s] for s in seed_order]
        structured_bracket.append(region_teams)

    return structured_bracket


def simulate_game(
    team_a: str,
    team_b: str,
    win_prob_fn: WinProbFn,
    rng: np.random.Generator,
) -> str:
    """Simulate a single game between team_a and team_b.

    The win probability for team_a is drawn from
    Normal(mean_prob, std_prob) and clipped to [0.01, 0.99] before a
    Bernoulli draw determines the winner.

    Parameters
    ----------
    team_a:
        Name of the first team (team_a winning is the "success" outcome).
    team_b:
        Name of the second team.
    win_prob_fn:
        Callable that accepts (team_a, team_b) and returns
        (mean_win_prob, std_win_prob) for team_a.
    rng:
        NumPy random Generator for reproducible draws.

    Returns
    -------
    str
        Name of the winning team (either team_a or team_b).
    """
    mean_prob, std_prob = win_prob_fn(team_a, team_b)
    # Draw win probability from Normal distribution
    sampled_prob = rng.normal(loc=mean_prob, scale=std_prob)
    # Clip to valid probability range
    sampled_prob = float(np.clip(sampled_prob, 0.01, 0.99))
    # Bernoulli draw: True → team_a wins
    return team_a if rng.random() < sampled_prob else team_b


def simulate_region(
    teams: list[str],
    win_prob_fn: WinProbFn,
    rng: np.random.Generator,
) -> str:
    """Simulate a single NCAA region bracket (16 → 1 team).

    The 16 teams must be provided in seeded matchup order:
    [1-seed, 16-seed, 8-seed, 9-seed, 5-seed, 12-seed, 4-seed, 13-seed,
     6-seed, 11-seed, 3-seed, 14-seed, 7-seed, 10-seed, 2-seed, 15-seed]

    Round 1 matchups (by index): 0v1, 2v3, 4v5, 6v7, 8v9, 10v11, 12v13, 14v15
    Winners advance and the pattern repeats for subsequent rounds.

    Parameters
    ----------
    teams:
        16 team names in seeded matchup order.
    win_prob_fn:
        Callable (team_a, team_b) → (mean_prob, std_prob).
    rng:
        NumPy random Generator.

    Returns
    -------
    str
        Name of the regional champion.
    """
    current_round = list(teams)  # copy
    while len(current_round) > 1:
        next_round: list[str] = []
        for i in range(0, len(current_round), 2):
            winner = simulate_game(current_round[i], current_round[i + 1], win_prob_fn, rng)
            next_round.append(winner)
        current_round = next_round
    return current_round[0]


def simulate_full_bracket(
    bracket_structure: list[list[str]],
    win_prob_fn: WinProbFn,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Simulate a complete 64-team NCAA bracket, returning per-team win counts.

    Structure
    ---------
    - 4 regional brackets (each 16 teams → 1 regional champion)
    - Final Four semi-finals: region 0 champion vs region 1 champion,
      region 2 champion vs region 3 champion
    - Championship game between the two Final Four winners

    Parameters
    ----------
    bracket_structure:
        4-element list of regions; each region is a list of 16 team names in
        seeded matchup order.
    win_prob_fn:
        Callable (team_a, team_b) → (mean_prob, std_prob).
    rng:
        NumPy random Generator.

    Returns
    -------
    dict[str, int]
        Every one of the 64 teams mapped to the number of wins accumulated in
        this single bracket simulation (0–6).  Exactly 63 total wins are
        distributed across all 64 teams.
    """
    # Initialize win counts to zero for every team
    wins: dict[str, int] = {}
    for region in bracket_structure:
        for team in region:
            wins[team] = 0

    # --- Rounds 1-4: simulate each of the 4 regional brackets ---
    regional_champions: list[str] = []
    for region_teams in bracket_structure:
        current_round = list(region_teams)
        while len(current_round) > 1:
            next_round: list[str] = []
            for i in range(0, len(current_round), 2):
                winner = simulate_game(
                    current_round[i], current_round[i + 1], win_prob_fn, rng
                )
                wins[winner] += 1
                next_round.append(winner)
            current_round = next_round
        regional_champions.append(current_round[0])

    # --- Round 5: Final Four semi-finals ---
    # Region 0 vs Region 1
    ff_winner_01 = simulate_game(
        regional_champions[0], regional_champions[1], win_prob_fn, rng
    )
    wins[ff_winner_01] += 1

    # Region 2 vs Region 3
    ff_winner_23 = simulate_game(
        regional_champions[2], regional_champions[3], win_prob_fn, rng
    )
    wins[ff_winner_23] += 1

    # --- Round 6: Championship ---
    champion = simulate_game(ff_winner_01, ff_winner_23, win_prob_fn, rng)
    wins[champion] += 1

    return wins


# ---------------------------------------------------------------------------
# BracketSimulator class
# ---------------------------------------------------------------------------


class BracketSimulator:
    """Monte Carlo simulator for the full 64-team NCAA March Madness bracket.

    Runs ``n_simulations`` independent bracket realizations to accumulate
    aggregate statistics:  champion probabilities, Final Four probabilities,
    Sweet 16 probabilities, expected wins per team, and bracket entropy.

    Parameters
    ----------
    n_simulations:
        Number of independent bracket simulations to run.  Default: 10,000.
    random_seed:
        Seed for the NumPy random Generator, ensuring reproducibility.
        Default: 42.
    """

    def __init__(
        self,
        n_simulations: int = 10_000,
        random_seed: int = 42,
    ) -> None:
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def simulate(
        self,
        seeds: dict[str, int],
        win_prob_fn: WinProbFn,
        bracket_structure: list[list[str]],
    ) -> dict:
        """Run the Monte Carlo simulation and return aggregated bracket statistics.

        Parameters
        ----------
        seeds:
            Mapping of team_name → seed (1–16).  Used for metadata; the actual
            simulation is driven by ``bracket_structure``.
        win_prob_fn:
            Callable (team_a, team_b) → (mean_prob, std_prob) for team_a.
            This should be the output of the Bayesian head for real usage, or
            ``default_win_prob_fn`` for testing.
        bracket_structure:
            4-region bracket where each region is a list of 16 team names in
            seeded matchup order (output of ``build_bracket_structure``).

        Returns
        -------
        dict with keys:
            champion_probs   : dict[str, float]  team → P(win championship)
            final_four_probs : dict[str, float]  team → P(reach Final Four)
            sweet_16_probs   : dict[str, float]  team → P(reach Sweet 16)
            expected_wins    : dict[str, float]  team → expected wins (0–6)
            bracket_entropy  : float             Shannon entropy of champion dist
        """
        rng = np.random.default_rng(self.random_seed)

        all_teams: list[str] = [t for region in bracket_structure for t in region]

        # Accumulators
        champion_counts: dict[str, int] = {t: 0 for t in all_teams}
        final_four_counts: dict[str, int] = {t: 0 for t in all_teams}
        sweet_16_counts: dict[str, int] = {t: 0 for t in all_teams}
        total_wins: dict[str, float] = {t: 0.0 for t in all_teams}

        for _ in range(self.n_simulations):
            wins = simulate_full_bracket(bracket_structure, win_prob_fn, rng)

            # Update total wins
            for team, w in wins.items():
                total_wins[team] += w

            # Determine which teams reached Sweet 16, Final Four, and won
            # Sweet 16: teams that won at least 2 games (survived R1 and R2)
            # Final Four: teams that won at least 4 games (4 regional rounds)
            # Champion: team that won exactly 6 games
            for team, w in wins.items():
                if w >= 2:
                    sweet_16_counts[team] += 1
                if w >= 4:
                    final_four_counts[team] += 1
                if w == 6:
                    champion_counts[team] += 1

        n = self.n_simulations

        champion_probs = {t: champion_counts[t] / n for t in all_teams}
        final_four_probs = {t: final_four_counts[t] / n for t in all_teams}
        sweet_16_probs = {t: sweet_16_counts[t] / n for t in all_teams}
        expected_wins_result = {t: total_wins[t] / n for t in all_teams}
        bracket_entropy = compute_bracket_entropy(champion_probs)

        return {
            "champion_probs": champion_probs,
            "final_four_probs": final_four_probs,
            "sweet_16_probs": sweet_16_probs,
            "expected_wins": expected_wins_result,
            "bracket_entropy": bracket_entropy,
        }
