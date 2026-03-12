"""
src/betting/options_pricing.py

Pillar 2.1: Real Options Valuation (Black-Scholes) for NCAA Tournament Hedging.

Mathematical Background
-----------------------
We treat a team's advancement through the tournament bracket as a 
European-style option chain. Each round is an "expiry date," and the 
team's probability of reaching that round is its option value.

Key Metrics:
- **Delta**: The marginal change in bracket value per unit change in win probability.
  This is just the advancement probability itself.
- **Vega**: The sensitivity of the option value to changes in volatility.
  High-Vega paths are those where small changes in team quality assessment 
  dramatically alter the expected bracket outcome.
- **Theta**: Time decay — as rounds are completed, the option "expires" and 
  remaining value concentrates in fewer teams.

The hedging module uses Vega to determine WHEN to hedge:
- High Vega (volatile path) → hedge aggressively (lock in value)
- Low Vega (stable path) → let it ride (the path is either chalk or dead)

Black-Scholes Analogy
---------------------
The Black-Scholes formula for a European call is:
    C = S * N(d1) - K * e^(-rT) * N(d2)

where:
    d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
    d2 = d1 - σ√T

In our bracket context:
    S = Current team advancement probability (portfolio equity)
    K = Break-even threshold (pool entry fee / prize ratio)
    r = 0 (no risk-free rate in bracket pools)
    T = Rounds remaining / Total rounds
    σ = Path volatility (derived from opponent quality variance)
"""

from __future__ import annotations

import math
import numpy as np
from typing import Dict, Any

from scipy.stats import norm


def compute_path_volatility(
    opponent_win_probs: list[float],
) -> float:
    """
    Compute the volatility (σ) of a team's tournament path.
    
    Volatility = standard deviation of the log-returns of round-by-round 
    advancement probabilities. A path facing all ~50% matchups has high 
    volatility; a path of chalk (90%+ matchups) has low volatility.
    
    Parameters
    ----------
    opponent_win_probs : list of float
        Our model's win probability against each projected opponent in 
        sequence (R64, R32, S16, E8, F4, Championship).
        
    Returns
    -------
    float
        Annualized path volatility (σ). Range: [0, ~1.5].
    """
    if len(opponent_win_probs) < 2:
        return 0.0
    
    # Compute log-returns of advancement probability
    log_probs = [math.log(max(p, 0.01)) for p in opponent_win_probs]
    returns = [log_probs[i] - log_probs[i - 1] for i in range(1, len(log_probs))]
    
    if len(returns) == 0:
        return 0.0
    
    return float(np.std(returns))


def compute_vega(
    current_prob: float,
    path_volatility: float,
    rounds_remaining: int,
    total_rounds: int = 6,
) -> float:
    """
    Compute the Vega (sensitivity to volatility) of a team's bracket position.
    
    Vega = ∂C/∂σ = S * √T * φ(d1)
    
    where φ is the standard normal PDF and T is the time-to-expiration ratio.
    
    Parameters
    ----------
    current_prob : float
        Current advancement probability (S in Black-Scholes terms).
    path_volatility : float
        Path volatility (σ).
    rounds_remaining : int
        How many rounds the team must win to reach the championship.
    total_rounds : int
        Total tournament rounds (default 6: R64→R32→S16→E8→F4→Champ).
        
    Returns
    -------
    float
        Vega value. Higher = more sensitive to volatility changes.
    """
    if path_volatility <= 0 or rounds_remaining <= 0:
        return 0.0
    
    T = rounds_remaining / total_rounds
    sqrt_T = math.sqrt(T)
    
    # d1 from Black-Scholes (with K=0.5 as breakeven, r=0)
    S = max(current_prob, 0.01)
    K = 0.5  # Breakeven threshold
    
    d1 = (math.log(S / K) + (path_volatility ** 2 / 2) * T) / (path_volatility * sqrt_T)
    
    vega = S * sqrt_T * norm.pdf(d1)
    return float(vega)


def recommend_hedge_action(
    vega: float,
    current_prob: float,
    pool_equity: float,
) -> Dict[str, Any]:
    """
    Based on the team's Vega and current position, recommend a hedging action.
    
    Strategy:
    - High Vega + High Equity → HEDGE AGGRESSIVELY (lock in profit)
    - High Vega + Low Equity → LET IT RIDE (high-variance speculation)
    - Low Vega + High Equity → HOLD (stable chalk path)
    - Low Vega + Low Equity → DEAD PATH (no action needed)
    
    Parameters
    ----------
    vega : float
        Team's computed Vega.
    current_prob : float
        Current advancement probability.
    pool_equity : float
        Current dollar value of our bracket position if this team wins.
    """
    if vega > 0.15 and current_prob > 0.6:
        return {
            "action": "HEDGE_AGGRESSIVELY",
            "reason": f"High Vega ({vega:.3f}) with strong position ({current_prob:.0%}). Lock in profit.",
            "urgency": "HIGH"
        }
    elif vega > 0.15 and current_prob <= 0.6:
        return {
            "action": "LET_IT_RIDE",
            "reason": f"High Vega ({vega:.3f}) but speculative position ({current_prob:.0%}). Risk-seeking is optimal in large pools.",
            "urgency": "LOW"
        }
    elif vega <= 0.15 and current_prob > 0.6:
        return {
            "action": "HOLD",
            "reason": f"Low Vega ({vega:.3f}) — stable chalk path ({current_prob:.0%}). No hedge needed yet.",
            "urgency": "NONE"
        }
    else:
        return {
            "action": "NO_ACTION",
            "reason": f"Dead path (prob={current_prob:.0%}, vega={vega:.3f}). Move on.",
            "urgency": "NONE"
        }


if __name__ == "__main__":
    print("=== Options Pricing / Vega Calculator ===")
    
    # Duke's path: high probability, stable
    duke_probs = [0.97, 0.85, 0.72, 0.62, 0.58, 0.54]
    duke_vol = compute_path_volatility(duke_probs)
    duke_vega = compute_vega(0.54, duke_vol, rounds_remaining=6)
    print(f"  Duke path volatility: {duke_vol:.4f}")
    print(f"  Duke Vega:            {duke_vega:.4f}")
    print(f"  Duke Hedge Rec:       {recommend_hedge_action(duke_vega, 0.54, 25000)}")
    
    # Liberty's path: wild variance
    liberty_probs = [0.38, 0.30, 0.15, 0.08, 0.03, 0.01]
    liberty_vol = compute_path_volatility(liberty_probs)
    liberty_vega = compute_vega(0.01, liberty_vol, rounds_remaining=6)
    print(f"\n  Liberty path volatility: {liberty_vol:.4f}")
    print(f"  Liberty Vega:            {liberty_vega:.4f}")
    print(f"  Liberty Hedge Rec:       {recommend_hedge_action(liberty_vega, 0.01, 25000)}")
