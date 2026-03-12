"""
src/betting/leverage.py

Executive-Level Game Theory & Pool Optimization ("The Leverage Engine").
Calculates "Leverage Score" = (True_Win_Probability / Public_Pick_Percentage).
Generates three distinct bracket topologies based on pool size and variance needs.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, List, Callable

logger = logging.getLogger(__name__)


def calculate_leverage_score(true_win_prob: float, public_pick_pct: float) -> float:
    """
    Calculate the leverage score for a specific team in a matchup.
    
    Numerator: Our model's true win probability.
    Denominator: The percentage of the public picking this team.
    
    A Leverage Score > 1.0 means the team is under-owned relative to their
    true probability (Value / Leverage).
    A Leverage Score < 1.0 means the team is over-owned (Toxic Chalk).
    """
    # Prevent division by zero; no team has exactly 0% public ownership
    # but bounding it at 0.01 provides numerical stability.
    bounded_public_pct = max(0.01, public_pick_pct)
    return true_win_prob / bounded_public_pct


class BracketOptimizer:
    def __init__(self, true_win_probs: Callable[[str, str], float], public_picks: Callable[[str, int, int], float]):
        """
        Initialize the Bracket Optimizer.
        
        :param true_win_probs: A callable `fn(team_a, team_b)` returning P(A wins).
        :param public_picks: A callable `fn(team_a, seed_a, seed_b)` returning public pick %.
        """
        self.get_true_win_prob = true_win_probs
        self.get_public_pick_pct = public_picks

    def generate_chalk_bracket_pick(self, team_a: str, seed_a: int, team_b: str, seed_b: int) -> str:
        """
        Optimized for small pools (< 20 people). 
        Strategy: Pure expected value maximation. Ignore ownership.
        Action: Simply pick whichever team has a True Win Probability > 50%.
        """
        p_win_a = self.get_true_win_prob(team_a, team_b)
        if p_win_a >= 0.50:
            return team_a
        return team_b

    def generate_leverage_bracket_pick(self, team_a: str, seed_a: int, team_b: str, seed_b: int) -> str:
        """
        Optimized for medium pools (50-100 people).
        Strategy: Fade over-owned favorites. Optimize for the highest Leverage Score.
        Action: Pick the team with the higher leverage score, provided their true 
                win probability meets a minimum threshold (e.g., >35%) so we aren't
                picking massive underdogs just for leverage.
        """
        p_win_a = self.get_true_win_prob(team_a, team_b)
        p_win_b = 1.0 - p_win_a

        pub_a = self.get_public_pick_pct(team_a, seed_a, seed_b)
        pub_b = 1.0 - pub_a

        lev_a = calculate_leverage_score(p_win_a, pub_a)
        lev_b = calculate_leverage_score(p_win_b, pub_b)

        # Baseline: who is simply more likely to win
        favorite = team_a if p_win_a >= 0.5 else team_b

        # If it's a tight game (e.g. 8 vs 9), strictly pick the better leverage
        if 0.40 <= p_win_a <= 0.60:
            return team_a if lev_a > lev_b else team_b

        # For heavily favored teams, see if the underdog has massive leverage 
        # AND a viable path to victory (> 30% true win prob)
        if lev_a > lev_b and p_win_a > 0.30:
            return team_a
        if lev_b > lev_a and p_win_b > 0.30:
            return team_b

        return favorite

    def generate_chaos_bracket_pick(self, team_a: str, seed_a: int, team_b: str, seed_b: int) -> str:
        """
        Optimized for massive contests (1,000+ people).
        Strategy: High variance. Correlate low-ownership deep runs.
        Action: Heavily weight the leverage score. Willing to take risks on 
                15-25% true probability teams if public ownership is < 5%.
        """
        p_win_a = self.get_true_win_prob(team_a, team_b)
        p_win_b = 1.0 - p_win_a

        pub_a = self.get_public_pick_pct(team_a, seed_a, seed_b)
        pub_b = 1.0 - pub_a

        lev_a = calculate_leverage_score(p_win_a, pub_a)
        lev_b = calculate_leverage_score(p_win_b, pub_b)

        # In giant pools, true coin-flips go strictly to the massive leverage side
        if lev_a > 1.25 and p_win_a > 0.18:
            return team_a
        if lev_b > 1.25 and p_win_b > 0.18:
            return team_b

        return team_a if lev_a > lev_b else team_b


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.public_picks import PublicPicksScraper

    # Mock engine for demonstration
    def mock_true_win_prob(a: str, b: str) -> float:
        # P(Purdue wins) = 0.62 (12-5 upset alert)
        if a == "Purdue" and b == "Liberty": return 0.62
        if a == "Liberty" and b == "Purdue": return 0.38
        return 0.5

    scraper = PublicPicksScraper()
    def mock_public_pct(team: str, seed: int, opp_seed: int) -> float:
        return scraper.get_public_pick_percentage(team, seed, opp_seed)

    optimizer = BracketOptimizer(mock_true_win_prob, mock_public_pct)

    logger.info("TEST: (5) Purdue vs (12) Liberty")
    chalk = optimizer.generate_chalk_bracket_pick("Purdue", 5, "Liberty", 12)
    leverage = optimizer.generate_leverage_bracket_pick("Purdue", 5, "Liberty", 12)
    chaos = optimizer.generate_chaos_bracket_pick("Purdue", 5, "Liberty", 12)

    logger.info(f"Chalk Bracket Pick: {chalk}")
    logger.info(f"Leverage Bracket Pick: {leverage}")
    logger.info(f"Chaos Bracket Pick: {chaos}")
