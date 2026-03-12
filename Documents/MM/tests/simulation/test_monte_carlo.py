"""
tests/simulation/test_monte_carlo.py

RED phase — tests written before implementation.
All tests should FAIL until src/simulation/monte_carlo.py is implemented.
"""

import math
import pytest
import numpy as np

from src.simulation.monte_carlo import (
    BracketSimulator,
    simulate_game,
    simulate_region,
    simulate_full_bracket,
    build_bracket_structure,
    compute_bracket_entropy,
    default_win_prob_fn,
)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_64_teams() -> dict[str, int]:
    """Return 64 teams with exactly 4 teams per seed (1-16)."""
    teams = {}
    regions = ["East", "West", "South", "Midwest"]
    for seed in range(1, 17):
        for region in regions:
            teams[f"{region}_seed{seed}"] = seed
    return teams


def _make_bracket_structure() -> list[list[str]]:
    """Return a 4-region bracket with 16 teams each, seeds in NCAA order."""
    seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    regions = ["East", "West", "South", "Midwest"]
    bracket = []
    for region in regions:
        region_teams = [f"{region}_seed{s}" for s in seed_order]
        bracket.append(region_teams)
    return bracket


def _certain_win_fn(team_a: str, team_b: str) -> tuple[float, float]:
    """team_a always wins (mean=0.99, std near 0)."""
    return (0.99, 0.001)


def _certain_loss_fn(team_a: str, team_b: str) -> tuple[float, float]:
    """team_a always loses (mean=0.01, std near 0)."""
    return (0.01, 0.001)


# ---------------------------------------------------------------------------
# 1. default_win_prob_fn
# ---------------------------------------------------------------------------

class TestDefaultWinProbFn:
    def test_returns_tuple(self):
        result = default_win_prob_fn("TeamA", "TeamB")
        assert isinstance(result, tuple)

    def test_returns_two_elements(self):
        result = default_win_prob_fn("TeamA", "TeamB")
        assert len(result) == 2

    def test_mean_is_0_5(self):
        mean, _ = default_win_prob_fn("TeamA", "TeamB")
        assert mean == pytest.approx(0.5)

    def test_std_is_0_1(self):
        _, std = default_win_prob_fn("TeamA", "TeamB")
        assert std == pytest.approx(0.1)

    def test_symmetric_for_any_teams(self):
        r1 = default_win_prob_fn("Duke", "UNC")
        r2 = default_win_prob_fn("Kansas", "Kentucky")
        assert r1 == r2


# ---------------------------------------------------------------------------
# 2. compute_bracket_entropy
# ---------------------------------------------------------------------------

class TestComputeBracketEntropy:
    def test_certain_champion_entropy_is_zero(self):
        probs = {"TeamA": 1.0, "TeamB": 0.0, "TeamC": 0.0}
        assert compute_bracket_entropy(probs) == pytest.approx(0.0)

    def test_uniform_64_teams_entropy_is_log2_64(self):
        probs = {f"Team{i}": 1 / 64 for i in range(64)}
        entropy = compute_bracket_entropy(probs)
        assert entropy == pytest.approx(math.log2(64), rel=1e-6)

    def test_uniform_2_teams_entropy_is_1(self):
        probs = {"TeamA": 0.5, "TeamB": 0.5}
        assert compute_bracket_entropy(probs) == pytest.approx(1.0)

    def test_skips_zero_probs(self):
        # Should not raise ZeroDivisionError or math domain error
        probs = {"TeamA": 0.8, "TeamB": 0.2, "TeamC": 0.0}
        entropy = compute_bracket_entropy(probs)
        assert entropy > 0

    def test_entropy_nonnegative(self):
        probs = {"A": 0.3, "B": 0.5, "C": 0.2}
        assert compute_bracket_entropy(probs) >= 0.0

    def test_single_team_certain(self):
        probs = {"OnlyTeam": 1.0}
        assert compute_bracket_entropy(probs) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. build_bracket_structure
# ---------------------------------------------------------------------------

class TestBuildBracketStructure:
    def test_returns_4_regions(self):
        teams = _make_64_teams()
        structure = build_bracket_structure(teams)
        assert len(structure) == 4

    def test_each_region_has_16_teams(self):
        teams = _make_64_teams()
        structure = build_bracket_structure(teams)
        for region in structure:
            assert len(region) == 16

    def test_total_teams_is_64(self):
        teams = _make_64_teams()
        structure = build_bracket_structure(teams)
        all_teams = [t for region in structure for t in region]
        assert len(all_teams) == 64

    def test_all_teams_present(self):
        teams = _make_64_teams()
        structure = build_bracket_structure(teams)
        all_teams = set(t for region in structure for t in region)
        assert all_teams == set(teams.keys())

    def test_raises_on_wrong_team_count_too_few(self):
        teams = {f"Team{i}": (i % 16) + 1 for i in range(32)}
        with pytest.raises(ValueError, match="64"):
            build_bracket_structure(teams)

    def test_raises_on_wrong_team_count_too_many(self):
        teams = {f"Team{i}": (i % 16) + 1 for i in range(68)}
        with pytest.raises(ValueError, match="64"):
            build_bracket_structure(teams)

    def test_raises_on_seed_out_of_range_too_high(self):
        teams = _make_64_teams()
        # Replace one valid seed with 17
        first_team = next(iter(teams))
        teams[first_team] = 17
        with pytest.raises(ValueError):
            build_bracket_structure(teams)

    def test_raises_on_seed_out_of_range_zero(self):
        teams = _make_64_teams()
        first_team = next(iter(teams))
        teams[first_team] = 0
        with pytest.raises(ValueError):
            build_bracket_structure(teams)

    def test_each_region_contains_all_seeds_1_through_16(self):
        teams = _make_64_teams()
        structure = build_bracket_structure(teams)
        for region in structure:
            # Each region should have one team per seed
            assert len(region) == 16


# ---------------------------------------------------------------------------
# 4. simulate_game
# ---------------------------------------------------------------------------

class TestSimulateGame:
    def test_returns_one_of_the_two_teams(self):
        rng = _make_rng(42)
        winner = simulate_game("TeamA", "TeamB", default_win_prob_fn, rng)
        assert winner in ("TeamA", "TeamB")

    def test_deterministic_with_fixed_seed(self):
        rng1 = _make_rng(99)
        rng2 = _make_rng(99)
        w1 = simulate_game("Alpha", "Beta", default_win_prob_fn, rng1)
        w2 = simulate_game("Alpha", "Beta", default_win_prob_fn, rng2)
        assert w1 == w2

    def test_certain_win_fn_always_returns_team_a(self):
        # With mean=0.99 and tiny std, team_a should almost always win
        rng = _make_rng(0)
        wins = sum(
            simulate_game("A", "B", _certain_win_fn, rng) == "A"
            for _ in range(200)
        )
        assert wins >= 180  # at least 90% wins

    def test_certain_loss_fn_almost_always_returns_team_b(self):
        rng = _make_rng(0)
        wins_b = sum(
            simulate_game("A", "B", _certain_loss_fn, rng) == "B"
            for _ in range(200)
        )
        assert wins_b >= 180

    def test_returns_string(self):
        rng = _make_rng(7)
        winner = simulate_game("X", "Y", default_win_prob_fn, rng)
        assert isinstance(winner, str)

    def test_different_seeds_can_give_different_outcomes(self):
        # Run many games; both teams should win at least once with uniform prob
        rng = _make_rng(42)
        results = {simulate_game("A", "B", default_win_prob_fn, rng) for _ in range(100)}
        assert len(results) == 2  # both "A" and "B" should appear

    def test_win_prob_clipped_above_0_01(self):
        """Even with extreme std, probability is clipped to [0.01, 0.99]."""
        def extreme_fn(a, b):
            return (0.5, 100.0)  # huge std, should be clipped
        rng = _make_rng(0)
        # Should not raise; just return one of the teams
        results = {simulate_game("A", "B", extreme_fn, rng) for _ in range(50)}
        assert results.issubset({"A", "B"})


# ---------------------------------------------------------------------------
# 5. simulate_region
# ---------------------------------------------------------------------------

class TestSimulateRegion:
    def _make_region(self) -> list[str]:
        seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        return [f"Team_seed{s}" for s in seed_order]

    def test_returns_one_of_16_teams(self):
        teams = self._make_region()
        rng = _make_rng(0)
        champion = simulate_region(teams, default_win_prob_fn, rng)
        assert champion in teams

    def test_returns_string(self):
        teams = self._make_region()
        rng = _make_rng(1)
        result = simulate_region(teams, default_win_prob_fn, rng)
        assert isinstance(result, str)

    def test_deterministic_with_fixed_rng(self):
        teams = self._make_region()
        rng1 = _make_rng(123)
        rng2 = _make_rng(123)
        assert simulate_region(teams, default_win_prob_fn, rng1) == simulate_region(
            teams, default_win_prob_fn, rng2
        )

    def test_1_seed_always_wins_with_certain_fn(self):
        """1-seed is index 0; with certain_win_fn, team at index 0 always advances."""
        teams = self._make_region()
        rng = _make_rng(0)
        # With _certain_win_fn, team_a always wins. In index order [0v1, 2v3, ...],
        # team at index 0 wins round 1, then [0, 2, 4, 6, 8, 10, 12, 14] advance,
        # and so on — index 0 ultimately wins.
        champion = simulate_region(teams, _certain_win_fn, rng)
        assert champion == teams[0]

    def test_last_index_wins_with_certain_loss_fn(self):
        """With certain_loss_fn team_b always wins.

        Tracing the bracket: R1 winners are odd indices [1,3,5,7,9,11,13,15].
        R2: game pairs (1,3),(5,7),(11,9),(13,15) → winners [3,7,11,15].
        R3: game pairs (3,7),(11,15) → winners [7,15].
        R4: game pair (7,15) → winner teams[15].
        """
        teams = self._make_region()
        rng = _make_rng(0)
        champion = simulate_region(teams, _certain_loss_fn, rng)
        assert champion == teams[15]

    def test_multiple_simulations_variety(self):
        """With default prob fn, region champion varies across many simulations."""
        teams = self._make_region()
        champions = set()
        for seed in range(50):
            rng = _make_rng(seed)
            champions.add(simulate_region(teams, default_win_prob_fn, rng))
        assert len(champions) > 1  # should see variety

    def test_exactly_16_teams_input(self):
        """Sanity: the function consumes exactly 16 teams."""
        teams = self._make_region()
        assert len(teams) == 16


# ---------------------------------------------------------------------------
# 6. simulate_full_bracket
# ---------------------------------------------------------------------------

class TestSimulateFullBracket:
    def test_returns_dict_with_all_64_teams(self):
        bracket = _make_bracket_structure()
        all_teams = [t for region in bracket for t in region]
        rng = _make_rng(0)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert set(result.keys()) == set(all_teams)

    def test_win_counts_sum_to_63(self):
        bracket = _make_bracket_structure()
        rng = _make_rng(0)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert sum(result.values()) == 63

    def test_all_win_counts_nonnegative(self):
        bracket = _make_bracket_structure()
        rng = _make_rng(7)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert all(v >= 0 for v in result.values())

    def test_max_wins_is_at_most_6(self):
        bracket = _make_bracket_structure()
        rng = _make_rng(7)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert max(result.values()) <= 6

    def test_returns_dict(self):
        bracket = _make_bracket_structure()
        rng = _make_rng(0)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert isinstance(result, dict)

    def test_deterministic_with_fixed_rng(self):
        bracket = _make_bracket_structure()
        rng1 = _make_rng(42)
        rng2 = _make_rng(42)
        r1 = simulate_full_bracket(bracket, default_win_prob_fn, rng1)
        r2 = simulate_full_bracket(bracket, default_win_prob_fn, rng2)
        assert r1 == r2

    def test_champion_has_6_wins_with_certain_win_fn(self):
        """With certain_win_fn, the team at index 0 of region 0 wins all 6 games."""
        bracket = _make_bracket_structure()
        rng = _make_rng(0)
        result = simulate_full_bracket(bracket, _certain_win_fn, rng)
        champion = bracket[0][0]
        assert result[champion] == 6

    def test_exactly_one_team_has_6_wins(self):
        bracket = _make_bracket_structure()
        rng = _make_rng(0)
        result = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        six_win_teams = [t for t, w in result.items() if w == 6]
        assert len(six_win_teams) == 1


# ---------------------------------------------------------------------------
# 7. BracketSimulator
# ---------------------------------------------------------------------------

class TestBracketSimulator:
    def _run_simulate(self, n: int = 500) -> dict:
        seeds = _make_64_teams()
        bracket = _make_bracket_structure()
        sim = BracketSimulator(n_simulations=n, random_seed=42)
        return sim.simulate(seeds, default_win_prob_fn, bracket)

    def test_returns_dict(self):
        result = self._run_simulate()
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = self._run_simulate()
        expected_keys = {
            "champion_probs",
            "final_four_probs",
            "sweet_16_probs",
            "expected_wins",
            "bracket_entropy",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_champion_probs_sum_to_1(self):
        result = self._run_simulate()
        total = sum(result["champion_probs"].values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_final_four_probs_sum_to_4(self):
        """Exactly 4 teams reach Final Four per simulation → expected sum = 4."""
        result = self._run_simulate()
        total = sum(result["final_four_probs"].values())
        assert total == pytest.approx(4.0, abs=1e-6)

    def test_sweet_16_probs_sum_to_16(self):
        """Exactly 16 teams reach Sweet 16 per simulation → expected sum = 16."""
        result = self._run_simulate()
        total = sum(result["sweet_16_probs"].values())
        assert total == pytest.approx(16.0, abs=1e-6)

    def test_expected_wins_sum_to_63(self):
        result = self._run_simulate()
        total = sum(result["expected_wins"].values())
        assert total == pytest.approx(63.0, abs=1e-4)

    def test_champion_probs_all_nonnegative(self):
        result = self._run_simulate()
        assert all(v >= 0 for v in result["champion_probs"].values())

    def test_champion_probs_all_at_most_1(self):
        result = self._run_simulate()
        assert all(v <= 1.0 + 1e-9 for v in result["champion_probs"].values())

    def test_champion_probs_covers_all_64_teams(self):
        seeds = _make_64_teams()
        bracket = _make_bracket_structure()
        sim = BracketSimulator(n_simulations=500, random_seed=42)
        result = sim.simulate(seeds, default_win_prob_fn, bracket)
        assert set(result["champion_probs"].keys()) == set(seeds.keys())

    def test_bracket_entropy_is_float(self):
        result = self._run_simulate()
        assert isinstance(result["bracket_entropy"], float)

    def test_bracket_entropy_nonnegative(self):
        result = self._run_simulate()
        assert result["bracket_entropy"] >= 0.0

    def test_bracket_entropy_at_most_log2_64(self):
        result = self._run_simulate()
        assert result["bracket_entropy"] <= math.log2(64) + 1e-9

    def test_deterministic_with_same_seed(self):
        seeds = _make_64_teams()
        bracket = _make_bracket_structure()
        sim1 = BracketSimulator(n_simulations=200, random_seed=7)
        sim2 = BracketSimulator(n_simulations=200, random_seed=7)
        r1 = sim1.simulate(seeds, default_win_prob_fn, bracket)
        r2 = sim2.simulate(seeds, default_win_prob_fn, bracket)
        assert r1["champion_probs"] == r2["champion_probs"]

    def test_certain_champion_entropy_near_zero(self):
        """With near-certain win function, entropy should be very low."""
        seeds = _make_64_teams()
        bracket = _make_bracket_structure()
        sim = BracketSimulator(n_simulations=200, random_seed=42)
        result = sim.simulate(seeds, _certain_win_fn, bracket)
        assert result["bracket_entropy"] < 1.0  # effectively one champion

    def test_expected_wins_nonnegative(self):
        result = self._run_simulate()
        assert all(v >= 0 for v in result["expected_wins"].values())

    def test_final_four_probs_between_0_and_1(self):
        result = self._run_simulate()
        for v in result["final_four_probs"].values():
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_sweet_16_probs_between_0_and_1(self):
        result = self._run_simulate()
        for v in result["sweet_16_probs"].values():
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_n_simulations_stored(self):
        sim = BracketSimulator(n_simulations=1234, random_seed=0)
        assert sim.n_simulations == 1234

    def test_random_seed_stored(self):
        sim = BracketSimulator(n_simulations=100, random_seed=99)
        assert sim.random_seed == 99

    def test_default_n_simulations(self):
        sim = BracketSimulator()
        assert sim.n_simulations == 10_000

    def test_default_random_seed(self):
        sim = BracketSimulator()
        assert sim.random_seed == 42


# ===========================================================================
# TestChaosEngine — _apply_chaos_engine topology disruption rule
# ===========================================================================

from src.simulation.monte_carlo import _apply_chaos_engine


def _make_base_state(region: int = 0, survivors: list | None = None) -> dict:
    if survivors is None:
        survivors = ["Cinderella", "TeamB", "TeamC", "TeamD"]
    return {
        "surviving_teams": {region: list(survivors)},
        "win_prob_adjustments": {},
        "titan_killer": "Cinderella",
        "ot_teams": set(),
    }


class TestChaosEngine:
    """Tests for _apply_chaos_engine."""

    def test_returns_dict(self):
        state = _make_base_state()
        rng = _make_rng()
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        assert isinstance(result, dict)

    def test_result_has_win_prob_adjustments_key(self):
        state = _make_base_state()
        rng = _make_rng()
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        assert "win_prob_adjustments" in result

    def test_titan_killer_receives_fatigue_penalty(self):
        state = _make_base_state()
        rng = _make_rng()
        result = _apply_chaos_engine(
            state, "Kansas", 0, {}, rng, chaos_fatigue_penalty=-0.02
        )
        adj = result["win_prob_adjustments"]
        # Cinderella is the titan killer — must have a negative adjustment.
        assert "Cinderella" in adj
        assert adj["Cinderella"] < 0.0

    def test_titan_killer_penalty_matches_parameter(self):
        state = _make_base_state()
        rng = _make_rng()
        penalty = -0.05
        result = _apply_chaos_engine(
            state, "Kansas", 0, {}, rng, chaos_fatigue_penalty=penalty
        )
        adj = result["win_prob_adjustments"]
        assert adj["Cinderella"] == pytest.approx(penalty)

    def test_non_killer_survivors_get_path_relief(self):
        """Non-killer survivors should receive a positive path-relief boost."""
        survivors = ["Cinderella", "TeamB", "TeamC"]
        state = _make_base_state(survivors=survivors)
        rng = _make_rng()
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        adj = result["win_prob_adjustments"]
        # TeamB and TeamC are not the killer — should get positive adjustment.
        assert adj.get("TeamB", 0.0) > 0.0
        assert adj.get("TeamC", 0.0) > 0.0

    def test_ot_penalty_applied_to_titan_killer_in_ot(self):
        state = {
            "surviving_teams": {0: ["Cinderella", "TeamB"]},
            "win_prob_adjustments": {},
            "titan_killer": "Cinderella",
            "ot_teams": {"Cinderella"},
        }
        rng = _make_rng()
        result = _apply_chaos_engine(
            state, "Kansas", 0, {}, rng,
            chaos_fatigue_penalty=-0.02, chaos_ot_penalty=-0.015,
        )
        adj = result["win_prob_adjustments"]
        # Killer gets base fatigue + OT penalty.
        assert adj["Cinderella"] == pytest.approx(-0.02 + -0.015)

    def test_ot_penalty_applied_to_non_killer_ot_teams(self):
        state = {
            "surviving_teams": {0: ["Cinderella", "TeamB", "TeamC"]},
            "win_prob_adjustments": {},
            "titan_killer": "Cinderella",
            "ot_teams": {"TeamB"},
        }
        rng = _make_rng()
        result = _apply_chaos_engine(
            state, "Kansas", 0, {}, rng,
            chaos_fatigue_penalty=-0.02, chaos_ot_penalty=-0.015,
        )
        adj = result["win_prob_adjustments"]
        # TeamB played OT but is not the killer → path_relief + ot_penalty.
        assert "TeamB" in adj
        assert adj["TeamB"] < 0.015  # OT penalty dominates

    def test_no_adjustment_for_wrong_region(self):
        """Teams in a different region should not receive any adjustment."""
        state = {
            "surviving_teams": {0: ["Cinderella"], 1: ["OtherTeam"]},
            "win_prob_adjustments": {},
            "titan_killer": "Cinderella",
            "ot_teams": set(),
        }
        rng = _make_rng()
        # Chaos in region 0 — OtherTeam in region 1 should be untouched.
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        adj = result["win_prob_adjustments"]
        assert "OtherTeam" not in adj

    def test_preserves_surviving_teams_key(self):
        state = _make_base_state()
        rng = _make_rng()
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        assert "surviving_teams" in result

    def test_empty_survivors_no_crash(self):
        state = {
            "surviving_teams": {0: []},
            "win_prob_adjustments": {},
            "titan_killer": None,
            "ot_teams": set(),
        }
        rng = _make_rng()
        # Should not raise.
        result = _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        assert isinstance(result, dict)

    def test_chaos_does_not_mutate_input_state(self):
        state = _make_base_state()
        original_adj = dict(state["win_prob_adjustments"])
        rng = _make_rng()
        _apply_chaos_engine(state, "Kansas", 0, {}, rng)
        # Input state's adjustments should not have been mutated.
        assert state["win_prob_adjustments"] == original_adj


class TestSimulateFullBracketWithChaos:
    """simulate_full_bracket with seeds= activates the Chaos Engine."""

    def test_chaos_mode_returns_wins_dict(self):
        rng = _make_rng(1)
        seeds = _make_64_teams()
        bracket = build_bracket_structure(seeds)
        wins = simulate_full_bracket(bracket, default_win_prob_fn, rng, seeds=seeds)
        assert isinstance(wins, dict)

    def test_chaos_mode_total_wins_63(self):
        rng = _make_rng(2)
        seeds = _make_64_teams()
        bracket = build_bracket_structure(seeds)
        wins = simulate_full_bracket(bracket, default_win_prob_fn, rng, seeds=seeds)
        assert sum(wins.values()) == 63

    def test_chaos_mode_all_teams_present(self):
        rng = _make_rng(3)
        seeds = _make_64_teams()
        bracket = build_bracket_structure(seeds)
        wins = simulate_full_bracket(bracket, default_win_prob_fn, rng, seeds=seeds)
        assert len(wins) == 64

    def test_no_seeds_still_works(self):
        """Without seeds=, Chaos Engine is skipped and simulation runs normally."""
        rng = _make_rng(4)
        seeds = _make_64_teams()
        bracket = build_bracket_structure(seeds)
        wins = simulate_full_bracket(bracket, default_win_prob_fn, rng)
        assert sum(wins.values()) == 63
