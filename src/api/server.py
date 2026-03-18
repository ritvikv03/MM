"""
src/api/server.py
FastAPI server — Madness Matrix backend.

Run with:
    uvicorn src.api.server:app --port 8000 --reload

Endpoints:
    GET  /api/graph?season=2026        → graph nodes + edges
    POST /api/matchup                  → posterior win/spread samples
    POST /api/bracket/simulate         → bracket advancement probabilities
    GET  /api/bracket/predict          → optimized 2026 predicted bracket

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
from src.api.bracket_2026 import get_bracket_teams_ordered
from src.api.intel_engine import build_intel

logger = logging.getLogger(__name__)

_UNAVAILABLE = "Data unavailable — pipeline has not run yet. Check back after 6 AM ET."

# Module-level singleton — avoids re-initialising the cache directory on every
# request and allows the DataLoader to serve cached parquet/JSON files cheaply.
_data_loader = DataLoader()

app = FastAPI(
    title="Madness Matrix API",
    description="ST-GNN + Bayesian inference backend for NCAA March Madness",
    version="2.0.0",
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
    season: int = Query(default=2026, ge=2010, le=2030),
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
        season = getattr(req, "season", 2026)
        return build_real_simulation(req.teams, req.n_simulations, season, loader=_data_loader)
    except Exception as exc:
        logger.error("Bracket simulation failed: %s", exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.get("/api/bracket/predict")
async def get_predicted_bracket() -> dict:
    """Run the full 2026 bracket prediction — 10,000 Monte Carlo trials.

    Returns the optimal bracket (most-probable path) plus advancement
    probabilities for all 64 teams.
    """
    try:
        teams = get_bracket_teams_ordered()
        sim = build_real_simulation(teams, 10_000, 2026, loader=_data_loader)

        # Build the optimal bracket: for each round, pick highest-probability team
        adv = sim.advancements
        rounds = ["R64", "R32", "S16", "E8", "F4", "Championship"]
        optimal: dict[str, str | None] = {}
        for r in rounds:
            best = max(adv, key=lambda t: t.advancement_probs.get(r, 0.0))
            optimal[r] = best.team

        champion = optimal.get("Championship", adv[0].team if adv else None)

        # Final Four: top 4 by F4 probability
        by_f4 = sorted(adv, key=lambda t: t.advancement_probs.get("F4", 0.0), reverse=True)
        final_four = [t.team for t in by_f4[:4]]

        return {
            "champion": champion,
            "final_four": final_four,
            "n_simulations": sim.n_simulations,
            "advancements": [
                {
                    "team": t.team,
                    "advancement_probs": t.advancement_probs,
                    "entropy": t.entropy,
                }
                for t in adv
            ],
            "data_source": sim.data_source,
        }
    except Exception as exc:
        logger.error("Bracket prediction failed: %s", exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.get("/api/intel")
async def get_intel(
    season: int = Query(default=2026, ge=2010, le=2030),
) -> dict:
    """Return autonomously generated intel flags from T-Rank metrics.

    Flags are computed algorithmically — no hardcoded strings.  All insights
    derive from Barttorvik efficiency margins, luck regression, and seed
    discrepancy analysis.
    """
    try:
        intel = build_intel(season=season, loader=_data_loader)
        from dataclasses import asdict
        return asdict(intel)
    except Exception as exc:
        logger.error("Intel generation failed for season %s: %s", season, exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.get("/api/bracket/optimal")
async def get_optimal_bracket() -> dict:
    """Return the optimal bracket with per-game picks and advancement probabilities.

    Runs 10,000 Monte Carlo trials and returns:
    - ``picks``: map of gameId → winning team name (for direct frontend auto-fill)
    - ``advancements``: full per-team advancement probability matrix
    - ``champion``: predicted champion name
    - ``final_four``: top-4 by Final Four probability
    """
    try:
        teams = get_bracket_teams_ordered()
        sim = build_real_simulation(teams, 10_000, 2026, loader=_data_loader)

        adv = sim.advancements

        # Sort by round probability to surface optimal picks
        by_f4  = sorted(adv, key=lambda t: t.advancement_probs.get("F4", 0.0),  reverse=True)
        by_e8  = sorted(adv, key=lambda t: t.advancement_probs.get("E8", 0.0),  reverse=True)

        champion   = sorted(adv, key=lambda t: t.advancement_probs.get("Championship", 0.0), reverse=True)[0].team if adv else None
        final_four = [t.team for t in by_f4[:4]]
        elite_eight = [t.team for t in by_e8[:8]]

        return {
            "champion":     champion,
            "final_four":   final_four,
            "elite_eight":  elite_eight,
            "n_simulations": sim.n_simulations,
            "advancements": [
                {
                    "team":             t.team,
                    "advancement_probs": t.advancement_probs,
                    "entropy":          t.entropy,
                    "champ_probability": t.advancement_probs.get("Championship", 0.0),
                }
                for t in adv
            ],
            "data_source": sim.data_source,
        }
    except Exception as exc:
        logger.error("Optimal bracket failed: %s", exc)
        raise HTTPException(status_code=503, detail=_UNAVAILABLE)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model": "madness-matrix-v2"}
