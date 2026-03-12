"""
src/simulation/copula_engine.py

Copula-Driven Cross-Game Correlation Engine for Monte Carlo Bracket Simulation.

Mathematical Background
-----------------------
In standard Monte Carlo bracket simulators, each game outcome is treated as 
independent. This fails to capture a critical empirical observation:

When a team from a heavily favored conference is upset, other teams from 
that same conference tend to slightly underperform. This is driven by:
  1. Market correction (bookmakers adjust lines for remaining conference teams)
  2. Shared coaching philosophy vulnerabilities exposed by the upset
  3. Travel fatigue clustering (teams from the same conference often play 
     in the same regional venues)

The Gaussian Copula provides a principled way to inject this correlation 
without changing the marginal win probabilities. The copula function 
transforms uniform marginals through a multivariate normal correlation 
structure, then maps back to the original marginal distributions.

Implementation
--------------
For each simulation trial in the Monte Carlo engine:
  1. Generate correlated uniform random variables via a Gaussian Copula
  2. Use these correlated uniforms (instead of independent uniforms) to 
     determine game outcomes
  3. When a conference upset occurs, the correlation structure automatically 
     downgrades other teams from that conference

References
----------
- Nelsen (2006). "An Introduction to Copulas."
- McNeil, Frey, Embrechts (2005). "Quantitative Risk Management."
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional


def build_conference_correlation_matrix(
    teams: List[Dict],
    base_intra_conf_corr: float = 0.15,
    base_inter_conf_corr: float = 0.02,
) -> np.ndarray:
    """
    Build a correlation matrix for N teams based on conference membership.
    
    Teams within the same conference share `base_intra_conf_corr` correlation.
    Teams across conferences share `base_inter_conf_corr` (near zero, but 
    slightly positive to capture general "chalk vs chaos" market dynamics).
    
    Parameters
    ----------
    teams : list of dict
        Each dict must have 'name' and 'conference' keys.
    base_intra_conf_corr : float
        Pearson correlation between teams in the same conference. Default 0.15.
    base_inter_conf_corr : float
        Pearson correlation between teams in different conferences. Default 0.02.
        
    Returns
    -------
    np.ndarray (N, N)
        Positive semi-definite correlation matrix.
    """
    n = len(teams)
    corr = np.full((n, n), base_inter_conf_corr)
    
    for i in range(n):
        corr[i, i] = 1.0  # Diagonal
        for j in range(i + 1, n):
            if teams[i].get("conference") == teams[j].get("conference"):
                corr[i, j] = base_intra_conf_corr
                corr[j, i] = base_intra_conf_corr
    
    # Ensure positive semi-definiteness via eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Re-normalize diagonal to 1.0
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    
    return corr


def generate_copula_draws(
    correlation_matrix: np.ndarray,
    n_simulations: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate correlated uniform random variables using a Gaussian Copula.
    
    Process:
    1. Sample from a multivariate normal with the given correlation matrix
    2. Apply the standard normal CDF (Φ) to each marginal to get uniform [0,1]
    
    Parameters
    ----------
    correlation_matrix : (N, N) ndarray
        Correlation matrix for N teams.
    n_simulations : int
        Number of simulation draws.
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    np.ndarray (n_simulations, N)
        Correlated uniform [0, 1] draws.
    """
    from scipy.stats import norm
    
    rng = np.random.default_rng(seed)
    n_teams = correlation_matrix.shape[0]
    
    # Step 1: Multivariate normal draws
    z = rng.multivariate_normal(
        mean=np.zeros(n_teams),
        cov=correlation_matrix,
        size=n_simulations,
    )
    
    # Step 2: Apply standard normal CDF → uniform marginals
    u = norm.cdf(z)
    
    return u


def apply_upset_contagion(
    copula_draws: np.ndarray,
    team_index: int,
    conference_indices: List[int],
    contagion_factor: float = 0.03,
) -> np.ndarray:
    """
    When a team is upset (their copula draw falls below the upset threshold),
    apply a small downward shift to all other teams from the same conference.
    
    This implements the "Conference Contagion Effect" — when Kansas loses in R1,
    all other Big 12 teams get slightly worse probabilities in subsequent rounds.
    
    Parameters
    ----------
    copula_draws : (n_sims, N) ndarray
        Correlated uniform draws.
    team_index : int
        Index of the upset team.
    conference_indices : list of int
        Indices of all other teams sharing the upset team's conference.
    contagion_factor : float
        Amount by which to shift affected teams' draws downward. Default 0.03 
        (a 3% reduction in advancement probability).
        
    Returns
    -------
    np.ndarray
        Modified copula draws with contagion applied.
    """
    modified = copula_draws.copy()
    
    for sim in range(copula_draws.shape[0]):
        # Check if this team was upset in this simulation
        if copula_draws[sim, team_index] < 0.5:  # Below expected → upset occurred
            for conf_idx in conference_indices:
                if conf_idx != team_index:
                    modified[sim, conf_idx] = max(
                        0.001,  # Floor to prevent degenerate cases
                        copula_draws[sim, conf_idx] - contagion_factor
                    )
    
    return modified


def compute_correlated_outcomes(
    win_probs: np.ndarray,
    copula_uniforms: np.ndarray,
) -> np.ndarray:
    """
    Determine game outcomes using correlated copula draws instead of 
    independent random numbers.
    
    A team wins if their copula draw is less than their win probability.
    Because the draws are correlated, upsets will cluster within conferences.
    
    Parameters
    ----------
    win_probs : (N,) ndarray
        Model's true win probability for each of N teams.
    copula_uniforms : (N,) ndarray
        Correlated uniform [0, 1] draws for this simulation.
        
    Returns
    -------
    np.ndarray (N,) of bool
        True if team wins, False if team loses.
    """
    return copula_uniforms < win_probs


if __name__ == "__main__":
    print("=== Copula Engine Sanity Check ===")
    
    teams = [
        {"name": "Kansas", "conference": "Big 12"},
        {"name": "Iowa State", "conference": "Big 12"},
        {"name": "Houston", "conference": "Big 12"},
        {"name": "Duke", "conference": "ACC"},
        {"name": "UNC", "conference": "ACC"},
        {"name": "Michigan", "conference": "Big Ten"},
    ]
    
    corr = build_conference_correlation_matrix(teams)
    print(f"\n  Correlation matrix shape: {corr.shape}")
    print(f"  Kansas ↔ Iowa State (same conf): {corr[0, 1]:.3f}")
    print(f"  Kansas ↔ Duke (diff conf):       {corr[0, 3]:.3f}")
    
    draws = generate_copula_draws(corr, n_simulations=10000)
    print(f"\n  Copula draws shape: {draws.shape}")
    print(f"  Mean of draws (should be ~0.5): {draws.mean():.3f}")
    print(f"  Correlation of Kansas ↔ Iowa State draws: {np.corrcoef(draws[:, 0], draws[:, 1])[0, 1]:.3f}")
    print(f"  Correlation of Kansas ↔ Duke draws:       {np.corrcoef(draws[:, 0], draws[:, 3])[0, 1]:.3f}")
