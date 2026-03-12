"""
src/api/server.py
FastAPI server exposing three endpoints for the Ethereal Oracle frontend.

Run with:
    uvicorn src.api.server:app --port 8000 --reload

Endpoints:
    GET  /api/graph?season=2024        → graph nodes + edges
    POST /api/matchup                  → posterior win/spread samples
    POST /api/bracket/simulate         → bracket advancement probabilities

All responses match the Zod contracts in frontend/lib/api-types.ts exactly.
"""
from __future__ import annotations

import math
import os
import random
from typing import List

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    GraphResponse, TeamNodeResponse, ConferenceNodeResponse,
    GameEdgeResponse, ConferenceEdgeResponse,
    MatchupRequest, MatchupResponse,
    SimulateRequest, SimulateResponse, TeamAdvancementItem,
)

app = FastAPI(
    title="Ethereal Oracle API",
    description="ST-GNN + Bayesian inference backend for NCAA March Madness",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

_CONF_COLORS: dict[str, int] = {
    "ACC":      0x00f5ff,   # cyan
    "Big12":    0x00ff88,   # tritium green
    "SEC":      0xff2d55,   # blood red
    "Big10":    0x7b2fff,   # violet
    "Pac12":    0xffb800,   # amber
    "BigEast":  0x00f5ff,
    "MVC":      0x888888,
    "WCC":      0x888888,
    "MWC":      0x888888,
    "CUSA":     0x888888,
}

_TEAMS_BY_CONF: dict[str, list[dict]] = {
    "ACC":     [{"name": "Duke",            "seed": 2},  {"name": "UNC",           "seed": 4},
                {"name": "Virginia",         "seed": 5},  {"name": "Miami",         "seed": 10}],
    "Big12":   [{"name": "Houston",         "seed": 1},  {"name": "Kansas",        "seed": 3},
                {"name": "Baylor",           "seed": 6},  {"name": "TCU",           "seed": 12}],
    "SEC":     [{"name": "Alabama",         "seed": 1},  {"name": "Tennessee",     "seed": 4},
                {"name": "Auburn",           "seed": 8},  {"name": "Mississippi State", "seed": 14}],
    "Big10":   [{"name": "Purdue",          "seed": 1},  {"name": "Michigan State","seed": 7},
                {"name": "Illinois",         "seed": 9},  {"name": "Ohio State",    "seed": 3}],
    "Pac12":   [{"name": "Arizona",         "seed": 2},  {"name": "UCLA",          "seed": 5},
                {"name": "Colorado",         "seed": 11}, {"name": "Oregon",        "seed": 13}],
    "BigEast": [{"name": "UConn",           "seed": 1},  {"name": "Marquette",     "seed": 3},
                {"name": "Creighton",        "seed": 6},  {"name": "Seton Hall",    "seed": 14}],
    "MVC":     [{"name": "Drake",           "seed": 11}, {"name": "Missouri State","seed": 16}],
    "WCC":     [{"name": "Gonzaga",         "seed": 3},  {"name": "Saint Mary's",  "seed": 12}],
    "MWC":     [{"name": "San Diego State", "seed": 5},  {"name": "Nevada",        "seed": 13}],
    "CUSA":    [{"name": "UAB",             "seed": 15}, {"name": "UTEP",          "seed": 16}],
}

_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "Championship"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int | str, offset: int = 0) -> random.Random:
    return random.Random(hash((str(seed), offset)) & 0xFFFFFFFF)


def _fibonacci_sphere(n: int, radius: float = 8.0) -> list[tuple[float, float, float]]:
    golden = math.pi * (1 + math.sqrt(5))
    pts = []
    for i in range(n):
        phi   = math.acos(1 - (2 * (i + 0.5)) / n)
        theta = golden * i
        pts.append((
            round(radius * math.sin(phi) * math.cos(theta), 3),
            round(radius * math.cos(phi),                  3),
            round(radius * math.sin(phi) * math.sin(theta), 3),
        ))
    return pts


def _box_muller(mean: float, std: float, n: int, rng: random.Random) -> list[float]:
    out = []
    for _ in range(n):
        u1 = max(1e-10, rng.random())
        u2 = rng.random()
        z  = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        out.append(mean + std * z)
    return out


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))


# ---------------------------------------------------------------------------
# Stub graph
# ---------------------------------------------------------------------------

def _build_stub_graph(season: int) -> GraphResponse:
    rng = _rng(season)
    conferences_list = list(_TEAMS_BY_CONF.keys())
    conf_positions   = _fibonacci_sphere(len(conferences_list), radius=9.0)

    conferences: list[ConferenceNodeResponse] = []
    teams:       list[TeamNodeResponse]       = []
    conf_edges:  list[ConferenceEdgeResponse] = []
    games:       list[GameEdgeResponse]       = []

    conf_pos_map: dict[str, tuple[float, float, float]] = {}

    for i, conf in enumerate(conferences_list):
        cx, cy, cz = conf_positions[i]
        conf_pos_map[conf] = (cx, cy, cz)
        color = _CONF_COLORS.get(conf, 0x888888)
        conferences.append(ConferenceNodeResponse(
            id=conf.lower(), name=conf, x=cx, y=cy, z=cz, color=color,
        ))

        for info in _TEAMS_BY_CONF.get(conf, []):
            team_id = info["name"].replace(" ", "_").lower()
            tx = round(cx + (rng.random() - 0.5) * 5, 3)
            ty = round(cy + (rng.random() - 0.5) * 5, 3)
            tz = round(cz + (rng.random() - 0.5) * 5, 3)
            adj_em = rng.uniform(5, 20)
            teams.append(TeamNodeResponse(
                id=team_id,
                name=info["name"],
                conference=conf,
                seed=info.get("seed"),
                adj_oe=round(100 + adj_em * 0.4 + rng.uniform(-2, 2), 1),
                adj_de=round(100 - adj_em * 0.4 + rng.uniform(-2, 2), 1),
                tempo=round(rng.uniform(65, 78), 1),
                x=tx, y=ty, z=tz,
                color=color,
            ))
            conf_edges.append(ConferenceEdgeResponse(
                source=team_id, target=conf.lower(), edge_type="member_of",
            ))

    # Generate game edges
    all_ids = [t.id for t in teams]
    for i in range(min(80, len(all_ids) * 3)):
        src, tgt = rng.sample(all_ids, 2)
        spread   = round(rng.gauss(0, 10), 1)
        month    = rng.randint(11, 12)
        day      = rng.randint(1, 28)
        games.append(GameEdgeResponse(
            source=src, target=tgt,
            home_win=rng.random() > 0.5 if rng.random() > 0.1 else None,
            spread=spread if rng.random() > 0.05 else None,
            date=f"{season - 1}-{month:02d}-{day:02d}",
        ))

    return GraphResponse(
        teams=teams, conferences=conferences,
        games=games, conference_edges=conf_edges,
    )


# ---------------------------------------------------------------------------
# Stub matchup
# ---------------------------------------------------------------------------

def _build_stub_matchup(
    home_team: str, away_team: str, season: int, neutral_site: bool,
) -> MatchupResponse:
    rng = _rng(f"{home_team}_{away_team}_{season}_{neutral_site}")

    home_adv   = 0.0 if neutral_site else rng.uniform(1.5, 3.5)
    spread_mean = round(rng.gauss(home_adv, 8), 2)
    spread_std  = round(abs(rng.gauss(6, 1.5)), 2)

    spread_samples  = [round(v, 3) for v in _box_muller(spread_mean, spread_std, 2000, rng)]
    p_win_samples   = [round(_sigmoid(v / 10), 4) for v in spread_samples]
    p_win_home      = round(sum(p_win_samples) / len(p_win_samples), 4)

    luck = round(rng.uniform(-0.09, 0.09), 3)

    return MatchupResponse(
        home_team=home_team,
        away_team=away_team,
        p_win_home=p_win_home,
        p_win_samples=p_win_samples,
        spread_mean=spread_mean,
        spread_samples=spread_samples,
        luck_compressed=abs(luck) > 0.05,
    )


# ---------------------------------------------------------------------------
# Stub bracket simulate
# ---------------------------------------------------------------------------

def _build_stub_simulate(teams: list[str], n_simulations: int) -> SimulateResponse:
    rng = _rng(f"{'_'.join(sorted(teams))}_{n_simulations}")

    advancements: list[TeamAdvancementItem] = []

    for i, team in enumerate(teams):
        # Seed-like strength: earlier in list = stronger
        strength = max(0.05, 1.0 - i / max(len(teams), 1) * 0.85)
        probs: dict[str, float] = {}
        p = 1.0
        for rnd in _ROUNDS:
            noise = rng.gauss(0, 0.04)
            p = max(0.01, min(p, p * (strength + noise)))
            probs[rnd] = round(p, 4)

        # Per-team entropy from distribution over rounds
        vals = list(probs.values())
        ent = 0.0
        total = sum(vals) or 1.0
        for v in vals:
            q = v / total
            if q > 0:
                ent -= q * math.log2(q)

        advancements.append(TeamAdvancementItem(
            team=team,
            advancement_probs=probs,
            entropy=round(ent, 4),
        ))

    return SimulateResponse(
        n_simulations=n_simulations,
        advancements=advancements,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(
    season: int = Query(default=2024, ge=2010, le=2030),
) -> GraphResponse:
    return _build_stub_graph(season)


@app.post("/api/matchup", response_model=MatchupResponse)
async def post_matchup(req: MatchupRequest) -> MatchupResponse:
    home = req.home_team.strip()
    away = req.away_team.strip()
    if not home or not away:
        raise HTTPException(status_code=422, detail="home_team and away_team must be non-empty")
    if home.lower() == away.lower():
        raise HTTPException(status_code=422, detail="home_team and away_team must be different")
    return _build_stub_matchup(home, away, req.season, req.neutral_site)


@app.post("/api/bracket/simulate", response_model=SimulateResponse)
async def post_simulate(req: SimulateRequest) -> SimulateResponse:
    return _build_stub_simulate(req.teams, req.n_simulations)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "ethereal-oracle-v1"}
