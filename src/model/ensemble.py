"""
src/model/ensemble.py

Phase 3: The "Board of Directors" Ensemble Weighting

Refactors the modeling pipeline to use an Ensemble approach. The primary engine 
remains the ST-GNN + Bayesian head, supported by three lightweight secondary models:
1. The Fundamentalist: Efficiency and Rebounding
2. The Market Reader: Sharp money and spread movement
3. The Chaos Agent: Travel fatigue, altitude, officiating, momentum
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class VoterModel:
    def __init__(self, name: str):
        self.name = name

    def vote(self, team_a: Dict[str, Any], team_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the matchup and return:
        - predicted_winner: str
        - confidence: float [0, 1]
        - reasoning: str
        """
        raise NotImplementedError

class FundamentalistModel(VoterModel):
    def __init__(self):
        super().__init__("The Fundamentalist")

    def vote(self, team_a: Dict[str, Any], team_b: Dict[str, Any]) -> Dict[str, Any]:
        # Simple heuristic based on Adjusted OE - Adjusted DE (Net Efficiency)
        em_a = team_a.get("adj_oe", 100) - team_a.get("adj_de", 100)
        em_b = team_b.get("adj_oe", 100) - team_b.get("adj_de", 100)
        
        # Add slight rebounding modifier
        reb_adv_a = team_a.get("rebound_margin", 0)
        reb_adv_b = team_b.get("rebound_margin", 0)
        
        score_a = em_a + (reb_adv_a * 0.5)
        score_b = em_b + (reb_adv_b * 0.5)
        
        margin = score_a - score_b
        
        if margin > 0:
            winner = team_a["name"]
            conf = min(0.99, 0.5 + (margin / 40.0))
            reason = f"Superior net efficiency (+{margin:.1f} AdjEM) and rebounding profile."
        else:
            winner = team_b["name"]
            conf = min(0.99, 0.5 + (abs(margin) / 40.0))
            reason = f"Superior net efficiency (+{abs(margin):.1f} AdjEM) and rebounding profile."
            
        return {"predicted_winner": winner, "confidence": conf, "reasoning": reason}

class MarketReaderModel(VoterModel):
    def __init__(self):
        super().__init__("The Market Reader")

    def vote(self, team_a: Dict[str, Any], team_b: Dict[str, Any]) -> Dict[str, Any]:
        # Evaluates sharp money line movement and historical ATS performance
        ats_a = team_a.get("ats_cover_pct", 0.5)
        ats_b = team_b.get("ats_cover_pct", 0.5)
        sharp_money_on_a = team_a.get("sharp_money_indicator", 0.0) # >0 means sharp money on A
        
        score_a = (ats_a - 0.5) + sharp_money_on_a
        score_b = (ats_b - 0.5) - sharp_money_on_a
        
        if score_a > score_b:
            winner = team_a["name"]
            conf = min(0.99, 0.5 + (abs(score_a - score_b)))
            reason = "Market indicators and sharp money flow favor the line movement."
        else:
            winner = team_b["name"]
            conf = min(0.99, 0.5 + (abs(score_b - score_a)))
            reason = "Sharp money fading the public heavily on this side."
            
        return {"predicted_winner": winner, "confidence": conf, "reasoning": reason}

class ChaosAgentModel(VoterModel):
    def __init__(self):
        super().__init__("The Chaos Agent")

    def vote(self, team_a: Dict[str, Any], team_b: Dict[str, Any]) -> Dict[str, Any]:
        # Factors: travel fatigue, altitude, momentum (kill-shots), officiating
        fatigue_a = team_a.get("travel_fatigue", 0)  # >0 means more fatigued
        fatigue_b = team_b.get("travel_fatigue", 0)
        
        foul_trouble_vuln_a = team_a.get("foul_dependency", 0) # High reliance on FTs or prone to fouling out
        foul_trouble_vuln_b = team_b.get("foul_dependency", 0)
        
        score_a = -fatigue_a - foul_trouble_vuln_a
        score_b = -fatigue_b - foul_trouble_vuln_b
        
        if score_a > score_b:
            winner = team_a["name"]
            conf = min(0.99, 0.5 + (score_a - score_b) * 0.1)
            reason = f"Less travel fatigue and lower officiating variance risk."
        else:
            winner = team_b["name"]
            conf = min(0.99, 0.5 + (score_b - score_a) * 0.1)
            reason = "Massive travel/altitude disadvantage for the opponent combined with foul trouble vulnerability."
            
        return {"predicted_winner": winner, "confidence": conf, "reasoning": reason}

class BoardOfDirectors:
    def __init__(self, primary_engine_weight: float = 0.7):
        self.primary_weight = primary_engine_weight
        self.fundamentalist = FundamentalistModel()
        self.market_reader = MarketReaderModel()
        self.chaos_agent = ChaosAgentModel()
        
    def evaluate_matchup(self, team_a: Dict[str, Any], team_b: Dict[str, Any], primary_prob_a: float) -> Dict[str, Any]:
        primary_winner = team_a["name"] if primary_prob_a >= 0.5 else team_b["name"]
        primary_conf = primary_prob_a if primary_prob_a >= 0.5 else (1.0 - primary_prob_a)
        
        votes = {
            "ST-GNN Primary": {
                "predicted_winner": primary_winner,
                "confidence": primary_conf,
                "reasoning": "Spatio-Temporal Graph representation combined with Bayesian posterior inference."
            },
            self.fundamentalist.name: self.fundamentalist.vote(team_a, team_b),
            self.market_reader.name: self.market_reader.vote(team_a, team_b),
            self.chaos_agent.name: self.chaos_agent.vote(team_a, team_b)
        }
        
        # Calculate consensus
        a_score = 0.0
        b_score = 0.0
        
        # Primary gets major weight
        if primary_winner == team_a["name"]:
            a_score += self.primary_weight * primary_conf
        else:
            b_score += self.primary_weight * primary_conf
            
        # Secondary gets 10% each
        secondary_weight = (1.0 - self.primary_weight) / 3.0
        for model in [self.fundamentalist.name, self.market_reader.name, self.chaos_agent.name]:
            v = votes[model]
            if v["predicted_winner"] == team_a["name"]:
                a_score += secondary_weight * v["confidence"]
            else:
                b_score += secondary_weight * v["confidence"]
                
        total_score = a_score + b_score
        final_winner = team_a["name"] if a_score > b_score else team_b["name"]
        final_conf = (a_score / total_score) if total_score > 0 else 0.5
        
        # Build Summary
        dissents = []
        for name, v in votes.items():
            if v["predicted_winner"] != final_winner:
                dissents.append(f"{name} dissents ({v['predicted_winner']}): {v['reasoning']}")
                
        consensus_summary = f"Total Consensus: {final_conf * 100:.1f}% confidence for {final_winner}. "
        if not dissents:
            consensus_summary += "Unanimous agreement across all four models."
        else:
            consensus_summary += " ".join(dissents)
            
        return {
            "final_winner": final_winner,
            "consensus_confidence": final_conf,
            "votes": votes,
            "summary": consensus_summary
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Mock data
    team_a = {
        "name": "Purdue", "adj_oe": 120, "adj_de": 95, "rebound_margin": +8,
        "ats_cover_pct": 0.45, "sharp_money_indicator": -0.2, # Sharps fading
        "travel_fatigue": 2, "foul_dependency": 0.8
    }
    team_b = {
        "name": "Liberty", "adj_oe": 110, "adj_de": 92, "rebound_margin": -2,
        "ats_cover_pct": 0.60, "sharp_money_indicator": +0.2, # Sharps love them
        "travel_fatigue": 0, "foul_dependency": 0.2
    }
    
    # Let's say ST-GNN gives Purdue 62%
    board = BoardOfDirectors()
    decision = board.evaluate_matchup(team_a, team_b, primary_prob_a=0.62)
    
    logger.info(f"Summary: {decision['summary']}")
