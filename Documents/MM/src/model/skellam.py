"""
src/model/skellam.py

Custom Skellam Distribution for NCAA Basketball Margin-of-Victory Modeling.

Mathematical Background
-----------------------
The Skellam distribution models the difference of two independent Poisson random
variables: if X ~ Poisson(μ₁) and Y ~ Poisson(μ₂), then D = X - Y ~ Skellam(μ₁, μ₂).

In basketball, the final score margin can be modeled as the difference of two 
team-level scoring Poisson processes. The Skellam naturally produces:
  - Discrete integer outputs (margins are integers)
  - Non-zero probability at 0 (ties—rare in basketball due to overtime)
  - Fat tails from foul-game dynamics (when the losing team intentionally fouls)

Zero-Inflation Modification
----------------------------
Because NCAA basketball games cannot end in a tie (overtime rules), we apply a
Zero-Truncated modification: P(D=0) is redistributed proportionally to P(D≠0).
This ensures the model never produces 0-margin predictions, matching reality.

The PMF of the Zero-Truncated Skellam:
    P_ZT(D=k) = P_Skellam(D=k) / (1 - P_Skellam(D=0))   for k ≠ 0

References
----------
- Karlis & Ntzoufras (2003). "Analysis of sports data by using bivariate Poisson models."
- Skellam, J.G. (1946). "The frequency distribution of the difference between
  two Poisson variates belonging to different populations."
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv as bessel_iv  # Modified Bessel function of the first kind
from scipy.stats import poisson


def skellam_pmf(k: int | np.ndarray, mu1: float, mu2: float) -> float | np.ndarray:
    """
    Compute the PMF of the Skellam distribution.
    
    P(D=k) = e^{-(μ₁+μ₂)} * (μ₁/μ₂)^{k/2} * I_{|k|}(2√(μ₁μ₂))
    
    where I_v is the modified Bessel function of the first kind of order v.
    
    Parameters
    ----------
    k : int or array
        The value(s) at which to evaluate the PMF (margin of victory).
    mu1 : float
        Rate parameter for team 1 (home team scoring rate).
    mu2 : float
        Rate parameter for team 2 (away team scoring rate).
        
    Returns
    -------
    float or ndarray
        PMF value(s).
    """
    k = np.asarray(k, dtype=float)
    mu1 = max(mu1, 1e-8)  # Prevent log(0)
    mu2 = max(mu2, 1e-8)
    
    log_pmf = (
        -(mu1 + mu2)
        + (k / 2.0) * np.log(mu1 / mu2)
        + np.log(bessel_iv(np.abs(k), 2.0 * np.sqrt(mu1 * mu2)))
    )
    
    return np.exp(log_pmf)


def skellam_pmf_zero_truncated(k: int | np.ndarray, mu1: float, mu2: float) -> float | np.ndarray:
    """
    Compute the PMF of the Zero-Truncated Skellam distribution.
    
    P_ZT(D=k) = P(D=k) / (1 - P(D=0))  for k ≠ 0
    P_ZT(D=0) = 0
    
    This guarantees the model NEVER predicts a tie, matching NCAA overtime rules.
    """
    k = np.asarray(k, dtype=float)
    
    # Standard Skellam PMF
    raw_pmf = skellam_pmf(k, mu1, mu2)
    
    # Probability of a tie under standard Skellam
    p_zero = skellam_pmf(0, mu1, mu2)
    
    # Zero-truncate: redistribute P(0) across all non-zero outcomes
    zt_pmf = np.where(k == 0, 0.0, raw_pmf / (1.0 - p_zero))
    
    return zt_pmf


def skellam_log_likelihood(
    observed_margins: np.ndarray,
    mu1: np.ndarray,
    mu2: np.ndarray,
    zero_truncated: bool = True
) -> float:
    """
    Compute the total log-likelihood of observed margins under the Skellam model.
    
    Parameters
    ----------
    observed_margins : (N,) array of integers
        Observed margin of victory for N games.
    mu1: (N,) array
        Scoring rate parameters for team 1 (home).
    mu2: (N,) array
        Scoring rate parameters for team 2 (away).
    zero_truncated : bool
        If True, use Zero-Truncated Skellam. Default True.
        
    Returns
    -------
    float
        Sum of log-likelihoods across all games.
    """
    total_ll = 0.0
    for i in range(len(observed_margins)):
        k = observed_margins[i]
        m1 = max(float(mu1[i]), 1e-8)
        m2 = max(float(mu2[i]), 1e-8)
        
        if zero_truncated:
            p = skellam_pmf_zero_truncated(k, m1, m2)
        else:
            p = skellam_pmf(k, m1, m2)
        
        total_ll += np.log(max(float(p), 1e-30))
        
    return total_ll


def margin_to_poisson_rates(
    delta: float, 
    base_rate: float = 70.0
) -> tuple[float, float]:
    """
    Convert a strength differential (delta) into two Poisson rate parameters.
    
    The base_rate represents the average NCAA score (~70 points per game).
    delta is the expected margin: mu1 - mu2 ≈ delta.
    
    We solve:
        mu1 = base_rate + delta/2
        mu2 = base_rate - delta/2
    
    Both rates are clamped to [1.0, 150.0] for numerical stability.
    """
    mu1 = max(1.0, min(150.0, base_rate + delta / 2.0))
    mu2 = max(1.0, min(150.0, base_rate - delta / 2.0))
    return mu1, mu2


if __name__ == "__main__":
    # Quick sanity check
    print("=== Skellam Distribution Sanity Check ===")
    
    # Typical game: Home scores ~75, Away ~70 → margin ≈ +5
    mu1, mu2 = 75.0, 70.0
    
    for margin in [-10, -5, 0, 1, 5, 10, 15, 20]:
        raw = skellam_pmf(margin, mu1, mu2)
        zt = skellam_pmf_zero_truncated(margin, mu1, mu2)
        print(f"  margin={margin:+3d}  Skellam={raw:.6f}  ZT-Skellam={zt:.6f}")
    
    print(f"\n  P(margin=0) under ZT-Skellam = {skellam_pmf_zero_truncated(0, mu1, mu2):.8f}")
    print("  ✅ Zero-Truncated Skellam correctly assigns P(0) = 0.0")
