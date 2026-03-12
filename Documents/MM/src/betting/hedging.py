"""
src/betting/hedging.py

Phase 2: Live Dynamic Hedging ("The Risk Desk").
Calculates fractional Kelly-Criterion sizing and exact EV Lock (Guaranteed Profit) 
hedge amounts for deep tournament runs.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def american_to_decimal(american_odds: int) -> float:
    """Convert American moneyline odds to Decimal odds."""
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    elif american_odds < 0:
        return (100.0 / abs(american_odds)) + 1.0
    return 1.0


def calculate_quarter_kelly(
    true_prob_a: float, 
    moneyline_a: int, 
    bankroll: float
) -> Dict[str, float]:
    """
    Calculate the conservative (Quarter-Kelly) bet size for a straight bet.
    
    Formula: f* = (bp - q) / b
    where: 
      b = net fractional odds received on the bet (Decimal - 1)
      p = probability of winning
      q = probability of losing (1 - p)
    """
    decimal_odds = american_to_decimal(moneyline_a)
    b = decimal_odds - 1.0
    p = true_prob_a
    q = 1.0 - p
    
    # Full Kelly fraction
    if b <= 0:
        return {"kelly_fraction": 0.0, "suggested_bet": 0.0, "edge": 0.0}

    k_fraction = (b * p - q) / b
    
    # If edge is negative, do not bet
    if k_fraction <= 0:
        return {"kelly_fraction": 0.0, "suggested_bet": 0.0, "edge": (p * decimal_odds) - 1.0}
        
    # Apply 1/4th fractional sizing for risk management
    quarter_kelly = k_fraction / 4.0
    
    # Cap maximum exposure at 5% of bankroll per game to prevent ruin
    capped_kelly = min(0.05, quarter_kelly)
    
    return {
        "kelly_fraction": quarter_kelly,
        "capped_kelly_fraction": capped_kelly,
        "suggested_bet": capped_kelly * bankroll,
        "edge": (p * decimal_odds) - 1.0  # Expected return per dollar
    }


def calculate_ev_lock(
    pool_prize_if_a_wins: float,
    moneyline_b: int
) -> Dict[str, float]:
    """
    Calculate the exact hedge amount needed on Team B's moneyline to lock in 
    a risk-free profit, assuming we win `pool_prize_if_a_wins` if Team A wins.
    
    Math:
    Let X be the bet amount on Team B. 
    Payout if A wins = pool_prize_if_a_wins - X
    Payout if B wins = X * decimal_odds_b - X  (which is X * fractional_odds)
    For equal risk-free profit: pool_prize_if_a_wins - X = X * decimal_odds_b - X
    => pool_prize_if_a_wins = X * decimal_odds_b
    => X = pool_prize_if_a_wins / decimal_odds_b
    """
    decimal_b = american_to_decimal(moneyline_b)
    
    if decimal_b <= 1.0:
        return {"hedge_amount": 0.0, "guaranteed_profit": 0.0}
        
    hedge_amount = pool_prize_if_a_wins / decimal_b
    guaranteed_profit = pool_prize_if_a_wins - hedge_amount
    
    return {
        "hedge_amount": round(hedge_amount, 2),
        "guaranteed_profit": round(guaranteed_profit, 2),
        "decimal_odds_b": round(decimal_b, 3)
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example 1: Kelly Sizing
    # Model says Liberty has 38% chance to beat Purdue.
    # Purdue is -700, Liberty is +500 (implied ~16.6%).
    # We have a massive edge.
    bankroll = 10000
    res = calculate_quarter_kelly(0.38, 500, bankroll)
    logger.info(f"Quarter-Kelly on Liberty (+500): {res}")
    
    # Example 2: EV Lock Hedge
    # It's the Final Four. We have Duke winning the bracket pool, which pays $25,000.
    # Duke plays Michigan. Michigan ML is +120.
    # How much do we bet on Michigan to guarantee profit no matter what?
    prize = 25000
    michigan_ml = 120
    hedge = calculate_ev_lock(prize, michigan_ml)
    logger.info(f"EV Lock Hedge on Michigan (+120): {hedge}")
