"""
src/data/shannon_entropy.py

Shannon Entropy Feature Engineering for NCAA Scoring Variance.

Mathematical Background
-----------------------
Shannon Entropy quantifies the "unpredictability" of a team's scoring distribution
across time segments (e.g., minute-by-minute or 5-minute bins).

For a team that scores uniformly across all periods, H is maximized (high entropy = 
consistent offense). For a team that scores in explosive bursts (10-0 runs) followed 
by droughts, H is lower (low entropy = high variance = "Kill Shot" tendency).

Formula:
    H(X) = -Σ p(xᵢ) * log₂(p(xᵢ))

where p(xᵢ) is the proportion of total points scored in time bin i.

Kill Shot / 10-0 Run Vulnerability
-----------------------------------
A complementary feature: the "Kill Shot Markov Matrix" models the transition
probabilities between scoring states:
  - State 0: Neither team scores (dead ball / timeout)
  - State 1: Home team on a run (3+ consecutive scores)
  - State 2: Away team on a run
  - State 3: Trading baskets (alternating scores)

Teams with high P(State 1 | State 3) are "Kill Shot" teams — they can flip
from trading baskets to a devastating 10-0 run. This is a hidden variable
that standard efficiency metrics miss entirely.

Integration
-----------
Both features (Shannon Entropy + Kill Shot Vulnerability) are injected as 
additional node features into the ST-GNN's team node feature vector.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any


def compute_scoring_entropy(
    points_by_period: List[int],
    n_bins: int = 8,
) -> float:
    """
    Compute the Shannon Entropy of a team's scoring distribution across time bins.
    
    Parameters
    ----------
    points_by_period : list of int
        Points scored in each time bin (e.g., 8 five-minute intervals for a 40-min game).
    n_bins : int
        Number of time bins the game is divided into. Default 8 (5-min bins).
        
    Returns
    -------
    float
        Shannon Entropy in bits. Higher = more consistent; Lower = more bursty.
        Ranges from 0 (all points in one period) to log₂(n_bins) (perfectly uniform).
    """
    points = np.array(points_by_period, dtype=float)
    total = points.sum()
    
    if total == 0:
        return 0.0  # No points → zero entropy (degenerate)
    
    # Compute probability distribution
    probs = points / total
    
    # Filter out zero-probability bins (log(0) is undefined)
    probs = probs[probs > 0]
    
    # Shannon Entropy: H = -Σ p * log₂(p)
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)


def compute_normalized_entropy(
    points_by_period: List[int],
    n_bins: int = 8,
) -> float:
    """
    Compute normalized Shannon Entropy ∈ [0, 1].
    
    Normalized by the maximum possible entropy (log₂(n_bins)) so that 
    1.0 = perfectly uniform scoring, 0.0 = all points in one bin.
    """
    max_entropy = np.log2(n_bins)
    if max_entropy == 0:
        return 0.0
    raw_entropy = compute_scoring_entropy(points_by_period, n_bins)
    return float(raw_entropy / max_entropy)


def compute_kill_shot_vulnerability(
    scoring_runs: List[int],
    threshold: int = 8,
) -> float:
    """
    Compute the "Kill Shot Vulnerability" — the probability that a team
    will allow a scoring run of `threshold` or more unanswered points.
    
    Parameters
    ----------
    scoring_runs : list of int
        List of opponent scoring run lengths observed in the season.
        E.g., [0, 3, 8, 2, 12, 5, 0, 10] means the opponent had runs of
        0, 3, 8, 2, 12, 5, 0, 10 unanswered points at various points.
    threshold : int
        The minimum run length that qualifies as a "Kill Shot." Default 8.
        
    Returns
    -------
    float
        Proportion of observed runs that meet or exceed the threshold.
        Higher = more vulnerable to Kill Shots.
    """
    if len(scoring_runs) == 0:
        return 0.0
    
    runs = np.array(scoring_runs)
    return float(np.mean(runs >= threshold))


def compute_kill_shot_markov_matrix(
    scoring_sequence: List[str],
) -> np.ndarray:
    """
    Build a 4x4 Markov transition matrix from a scoring sequence.
    
    States:
        0: Dead ball / timeout
        1: Home run (3+ consecutive home scores)
        2: Away run (3+ consecutive away scores)
        3: Trading baskets (alternating)
    
    Parameters
    ----------
    scoring_sequence : list of str
        Sequence of scoring events: 'H' (home scores), 'A' (away scores), 
        'D' (dead ball/timeout).
    
    Returns
    -------
    np.ndarray (4, 4)
        Row-stochastic Markov transition matrix.
    """
    # Map events to states
    state_map = {'D': 0, 'H': 1, 'A': 2}
    
    # Build simple state sequence
    states = []
    consecutive_h = 0
    consecutive_a = 0
    
    for event in scoring_sequence:
        if event == 'D':
            states.append(0)
            consecutive_h = 0
            consecutive_a = 0
        elif event == 'H':
            consecutive_h += 1
            consecutive_a = 0
            states.append(1 if consecutive_h >= 3 else 3)
        elif event == 'A':
            consecutive_a += 1
            consecutive_h = 0
            states.append(2 if consecutive_a >= 3 else 3)
    
    # Build transition count matrix
    T = np.zeros((4, 4), dtype=float)
    for i in range(len(states) - 1):
        T[states[i], states[i + 1]] += 1
    
    # Normalize to row-stochastic
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid div by zero
    T = T / row_sums
    
    return T


def extract_entropy_features(
    team_data: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract the full Shannon Entropy feature set for a team node.
    
    Returns a dict containing:
    - scoring_entropy: Raw Shannon Entropy
    - scoring_entropy_normalized: Normalized ∈ [0, 1]
    - kill_shot_vulnerability: P(opponent run >= 8)
    - kill_shot_p_run_given_trading: P(State 1 | State 3) from Markov matrix
    """
    # Default fallback values when data is missing (graceful degradation)
    defaults = {
        "scoring_entropy": np.log2(8) * 0.85,  # Assume ~85% of max entropy
        "scoring_entropy_normalized": 0.85,
        "kill_shot_vulnerability": 0.15,  # Average NCAA vulnerability
        "kill_shot_p_run_given_trading": 0.12,
    }
    
    points_by_period = team_data.get("points_by_period")
    if points_by_period is not None and len(points_by_period) > 0:
        defaults["scoring_entropy"] = compute_scoring_entropy(points_by_period)
        defaults["scoring_entropy_normalized"] = compute_normalized_entropy(points_by_period)
    
    scoring_runs = team_data.get("opponent_scoring_runs")
    if scoring_runs is not None and len(scoring_runs) > 0:
        defaults["kill_shot_vulnerability"] = compute_kill_shot_vulnerability(scoring_runs)
    
    scoring_seq = team_data.get("scoring_sequence")
    if scoring_seq is not None and len(scoring_seq) > 1:
        markov = compute_kill_shot_markov_matrix(scoring_seq)
        # P(State 1 → Home Run | State 3 → Trading baskets)
        defaults["kill_shot_p_run_given_trading"] = float(markov[3, 1])
    
    return defaults


if __name__ == "__main__":
    # Sanity check with synthetic data
    print("=== Shannon Entropy Feature Engineering ===")
    
    # Consistent team: scores ~10 pts per 5-min period
    consistent = [10, 9, 11, 10, 10, 9, 11, 10]
    burst = [0, 25, 2, 0, 30, 3, 0, 20]
    
    print(f"  Consistent Team Entropy: {compute_scoring_entropy(consistent):.3f} bits")
    print(f"  Bursty Team Entropy:     {compute_scoring_entropy(burst):.3f} bits")
    print(f"  Max Possible Entropy:    {np.log2(8):.3f} bits")
    
    print(f"\n  Consistent Normalized:   {compute_normalized_entropy(consistent):.3f}")
    print(f"  Bursty Normalized:       {compute_normalized_entropy(burst):.3f}")
    
    # Kill Shot test
    runs = [0, 3, 8, 2, 12, 5, 0, 10, 2, 4, 1, 9, 3]
    print(f"\n  Kill Shot Vulnerability (>=8pt runs): {compute_kill_shot_vulnerability(runs):.3f}")
