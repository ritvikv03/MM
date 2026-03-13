"""
src/api/graph_builder.py

Builds a real GraphResponse from Barttorvik T-Rank data via the DataLoader
cache.

Algorithm
---------
1. Load T-Rank DataFrame for the requested season from DataLoader.
2. Load tournament seeds (may be empty if seeds are unavailable yet).
3. Compute adj_em = adj_oe - adj_de per team.
4. Normalise adj_oe → x, adj_em → y, adj_de → z to [-1, 1] then scale to
   [-8, 8] / [-6, 6] / [-8, 8] respectively.
5. Emit one TeamNodeResponse per row with conference colour from
   _CONF_COLORS.
6. Emit one ConferenceNodeResponse per unique conference, positioned at the
   centroid of its member teams × 1.5 scale-out.
7. Emit one ConferenceEdgeResponse per team (team → conference).
8. games=[] for now (Kaggle game history added in a later task).
9. data_source="real".
"""
from __future__ import annotations

import hashlib
import logging
import math
from typing import Optional

import pandas as pd

from src.api.schemas import (
    ConferenceEdgeResponse,
    ConferenceNodeResponse,
    GameEdgeResponse,
    GraphResponse,
    TeamNodeResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conference colour palette (spec-mandated exact values)
# ---------------------------------------------------------------------------

_CONF_COLORS: dict[str, int] = {
    "ACC":      0x4A90D9,
    "Big 12":   0xE74C3C,
    "SEC":      0xF39C12,
    "Big Ten":  0x2ECC71,
    "Pac-12":   0x9B59B6,
    "WCC":      0x1ABC9C,
    "Big East": 0x3498DB,
    "AAC":      0xE67E22,
    "MWC":      0x95A5A6,
    "A-10":     0xD35400,
    "MAC":      0x27AE60,
    "CUSA":     0x8E44AD,
    "SBC":      0x2980B9,
    "MVC":      0xC0392B,
    "WAC":      0x16A085,
}
_DEFAULT_COLOR: int = 0x888888


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_id(name: str) -> str:
    """Return an 8-hex-char stable ID derived from *name*."""
    return hashlib.md5(name.encode()).hexdigest()[:8]


def _normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalise *series* to [-1, 1].

    If all values are identical (zero range), returns a zero Series to avoid
    division by zero.
    """
    lo = series.min()
    hi = series.max()
    rng = hi - lo
    if rng == 0.0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - lo) / rng * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_real_graph(
    season: int,
    loader=None,
) -> GraphResponse:
    """Build a real *GraphResponse* from Barttorvik T-Rank data.

    Parameters
    ----------
    season:
        NCAA season year (e.g. 2024).
    loader:
        Optional :class:`~src.api.data_cache.DataLoader` instance.  If
        *None*, a fresh ``DataLoader`` is instantiated (uses default cache
        directory).

    Returns
    -------
    GraphResponse
        Populated with real team names, efficiency ratings, conference
        topology, and ``data_source="real"``.  If T-Rank data is unavailable,
        returns an empty ``GraphResponse`` with ``data_source="real"``.
    """
    # ------------------------------------------------------------------
    # 0. Resolve loader lazily to avoid circular import at module import
    # ------------------------------------------------------------------
    if loader is None:
        from src.api.data_cache import DataLoader  # type: ignore
        loader = DataLoader()

    # ------------------------------------------------------------------
    # 1. Fetch T-Rank
    # ------------------------------------------------------------------
    trank_df: pd.DataFrame = loader.get_trank(season)

    if trank_df is None or trank_df.empty:
        logger.info("build_real_graph: empty T-Rank for season %s — returning empty graph", season)
        return GraphResponse(
            teams=[],
            conferences=[],
            games=[],
            conference_edges=[],
            data_source="real",
        )

    # ------------------------------------------------------------------
    # 2. Fetch tournament seeds
    # ------------------------------------------------------------------
    seeds_dict: dict[str, int] = loader.get_tournament_seeds(season)

    # ------------------------------------------------------------------
    # 3. Ensure required columns exist with sensible defaults
    # ------------------------------------------------------------------
    if "adj_oe" not in trank_df.columns:
        trank_df = trank_df.copy()
        trank_df["adj_oe"] = 100.0
    if "adj_de" not in trank_df.columns:
        trank_df = trank_df.copy()
        trank_df["adj_de"] = 100.0
    if "tempo" not in trank_df.columns:
        trank_df = trank_df.copy()
        trank_df["tempo"] = 70.0
    if "conference" not in trank_df.columns:
        trank_df = trank_df.copy()
        trank_df["conference"] = "Unknown"

    # ------------------------------------------------------------------
    # 4. Compute adj_em and normalised positions
    # ------------------------------------------------------------------
    trank_df = trank_df.copy()
    trank_df["adj_em"] = trank_df["adj_oe"] - trank_df["adj_de"]

    x_raw = _normalize_series(trank_df["adj_oe"]) * 8.0
    y_raw = _normalize_series(trank_df["adj_em"]) * 6.0   # stronger teams float up
    z_raw = _normalize_series(trank_df["adj_de"]) * 8.0

    # ------------------------------------------------------------------
    # 5. Build team nodes
    # ------------------------------------------------------------------
    teams: list[TeamNodeResponse] = []
    team_positions: dict[str, tuple[float, float, float]] = {}

    for idx, row in trank_df.iterrows():
        name: str = str(row["team"])
        conf: str = str(row.get("conference", "Unknown"))
        color: int = _CONF_COLORS.get(conf, _DEFAULT_COLOR)
        team_id = _make_id(name)

        # Seed: seeds_dict > trank seed column > default 16
        if name in seeds_dict:
            seed: Optional[int] = int(seeds_dict[name])
        elif "seed" in trank_df.columns and not pd.isna(row.get("seed")):
            seed = int(row["seed"])
        else:
            seed = 16

        tx = float(x_raw.loc[idx])
        ty = float(y_raw.loc[idx])
        tz = float(z_raw.loc[idx])

        # Guard against NaN / inf (e.g., single-row DataFrames after norm)
        tx = tx if math.isfinite(tx) else 0.0
        ty = ty if math.isfinite(ty) else 0.0
        tz = tz if math.isfinite(tz) else 0.0

        team_positions[team_id] = (tx, ty, tz)

        teams.append(TeamNodeResponse(
            id=team_id,
            name=name,
            conference=conf,
            seed=seed,
            adj_oe=float(row["adj_oe"]),
            adj_de=float(row["adj_de"]),
            tempo=float(row.get("tempo", 70.0)),
            x=tx,
            y=ty,
            z=tz,
            color=color,
        ))

    # ------------------------------------------------------------------
    # 6. Build conference nodes (centroid × 1.5)
    # ------------------------------------------------------------------
    # Group team positions by conference
    conf_to_team_ids: dict[str, list[str]] = {}
    for t, row in zip(teams, trank_df.itertuples()):
        conf = t.conference
        conf_to_team_ids.setdefault(conf, []).append(t.id)

    conferences: list[ConferenceNodeResponse] = []
    conf_id_map: dict[str, str] = {}  # conf_name → conf_id

    for conf_name, member_ids in conf_to_team_ids.items():
        conf_id = _make_id(conf_name)
        conf_id_map[conf_name] = conf_id
        color = _CONF_COLORS.get(conf_name, _DEFAULT_COLOR)

        # Centroid of member teams
        xs = [team_positions[tid][0] for tid in member_ids]
        ys = [team_positions[tid][1] for tid in member_ids]
        zs = [team_positions[tid][2] for tid in member_ids]
        n = len(member_ids)
        cx = sum(xs) / n * 1.5
        cy = sum(ys) / n * 1.5
        cz = sum(zs) / n * 1.5

        conferences.append(ConferenceNodeResponse(
            id=conf_id,
            name=conf_name,
            x=round(cx, 3),
            y=round(cy, 3),
            z=round(cz, 3),
            color=color,
        ))

    # ------------------------------------------------------------------
    # 7. Build conference edges (one per team)
    # ------------------------------------------------------------------
    conference_edges: list[ConferenceEdgeResponse] = []
    for team in teams:
        conf_id = conf_id_map.get(team.conference, _make_id(team.conference))
        conference_edges.append(ConferenceEdgeResponse(
            source=team.id,
            target=conf_id,
            edge_type="member_of",
        ))

    # ------------------------------------------------------------------
    # 8. Games — populated in a later task
    # ------------------------------------------------------------------
    games: list[GameEdgeResponse] = []

    return GraphResponse(
        teams=teams,
        conferences=conferences,
        games=games,
        conference_edges=conference_edges,
        data_source="real",
    )
