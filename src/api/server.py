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
If the data pipeline has not run yet, endpoints return HTTP 503.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    GraphResponse,
    MatchupRequest, MatchupResponse,
    SimulateRequest, SimulateResponse,
)
from src.api.data_cache import DataLoader
from src.api.graph_builder import build_real_graph
from src.api.matchup_engine import build_real_matchup, MatchupNotFoundError
from src.api.bracket_runner import build_real_simulation

logger = logging.getLogger(__name__)

_UNAVAILABLE = "Data unavailable — pipeline has not run yet. Check back after 6 AM ET."

# Module-level singleton — avoids re-initialising the cache directory on every
# request and allows the DataLoader to serve cached parquet/JSON files cheaply.
_data_loader = DataLoader()

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
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(
    season: int = Query(default=2024, ge=2010, le=2030),
) -> GraphResponse:
    try:
        return build_real_graph(season=season, loader=_data_loader)
    except Exception as exc:
        logger.error("Graph build failed for season %s: %s", season, exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.post("/api/matchup", response_model=MatchupResponse)
async def post_matchup(req: MatchupRequest) -> MatchupResponse:
    home = req.home_team.strip()
    away = req.away_team.strip()
    if not home or not away:
        raise HTTPException(status_code=422, detail="home_team and away_team must be non-empty")
    if home.lower() == away.lower():
        raise HTTPException(status_code=422, detail="home_team and away_team must be different")
    try:
        return build_real_matchup(home, away, req.season, req.neutral_site,
                                  loader=_data_loader)
    except MatchupNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Matchup build failed for %s vs %s: %s", home, away, exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.post("/api/bracket/simulate", response_model=SimulateResponse)
async def post_simulate(req: SimulateRequest) -> SimulateResponse:
    try:
        season = getattr(req, "season", 2024)
        return build_real_simulation(req.teams, req.n_simulations, season, loader=_data_loader)
    except Exception as exc:
        logger.error("Bracket simulation failed: %s", exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "ethereal-oracle-v1"}
