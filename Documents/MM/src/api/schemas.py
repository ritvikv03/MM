"""
src/api/schemas.py
Pydantic request/response models for the FastAPI server.
These exactly mirror the Zod schemas in frontend/lib/api-types.ts.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Graph ─────────────────────────────────────────────────────────────────────

class TeamNodeResponse(BaseModel):
    id:         str
    name:       str
    conference: str
    seed:       Optional[int]
    adj_oe:     float        # offensive rating
    adj_de:     float        # defensive rating
    tempo:      float        # possessions per 40 min
    x:          float
    y:          float
    z:          float
    color:      int          # 0xRRGGBB integer


class ConferenceNodeResponse(BaseModel):
    id:    str
    name:  str
    x:     float
    y:     float
    z:     float
    color: int               # 0xRRGGBB integer


class GameEdgeResponse(BaseModel):
    source:   str
    target:   str
    home_win: Optional[bool]
    spread:   Optional[float]
    date:     Optional[str]


class ConferenceEdgeResponse(BaseModel):
    source:    str
    target:    str
    edge_type: str = "member_of"


class GraphResponse(BaseModel):
    teams:            List[TeamNodeResponse]
    conferences:      List[ConferenceNodeResponse]
    games:            List[GameEdgeResponse]
    conference_edges: List[ConferenceEdgeResponse]


# ── Matchup ───────────────────────────────────────────────────────────────────

class MatchupRequest(BaseModel):
    home_team:    str
    away_team:    str
    season:       int  = 2024
    neutral_site: bool = False


class MatchupResponse(BaseModel):
    home_team:      str
    away_team:      str
    p_win_home:     float
    p_win_samples:  List[float]
    spread_mean:    float
    spread_samples: List[float]
    luck_compressed: bool


# ── Bracket simulate ──────────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    teams:         List[str]
    n_simulations: int = Field(default=1000, ge=100, le=50000)


class TeamAdvancementItem(BaseModel):
    team:             str
    advancement_probs: Dict[str, float]   # keys: R64 R32 S16 E8 F4 Championship
    entropy:          float


class SimulateResponse(BaseModel):
    n_simulations: int
    advancements:  List[TeamAdvancementItem]
