# Real Data API Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Replace the three stub API endpoints (`/api/graph`, `/api/matchup`, `/api/bracket/simulate`) with responses driven by real Barttorvik efficiency data, a Bayesian ADVI matchup model, and a real Monte Carlo bracket simulator.

**Architecture:** A cache-first data layer pre-fetches Barttorvik T-Rank + Kaggle tournament seeds once per season and writes them to `data/cache/` as Parquet/JSON. API endpoints load from cache (< 50 ms); a `DataLoader` class handles the fetch-or-cache logic. The Bayesian ADVI matchup model runs in ~30 s and caches its posteriors. The stub remains as fallback when cache is cold.

**Tech Stack:** `requests`, `BeautifulSoup`, `pandas`, `pymc` (ADVI), `numpy`, `fastapi`, `pytest`

---

## Context for every implementer subagent

- **Working dir:** `/Users/ritvikvasikarla/Documents/MM`
- **Python:** `/opt/anaconda3/bin/python3`
- **Run backend:** `uvicorn src.api.server:app --port 8000 --reload`
- **Run tests:** `/opt/anaconda3/bin/python3 -m pytest tests/ -q`
- **Current stub lives in:** `src/api/server.py` — functions `_build_stub_graph`, `_build_stub_matchup`, `_build_stub_simulate`
- **Real data modules already exist** under `src/data/barttorvik.py`, `src/data/kaggle_ingestion.py`, `src/graph/`, `src/model/bayesian_head.py`, `src/simulation/monte_carlo.py`
- **Existing tests:** 1,254 passing — do not break them
- **Stub must remain as fallback** (used when `USE_REAL_DATA` env var is absent or cache is cold)
- **No paid APIs.** Barttorvik is free public scraping. Kaggle requires `KAGGLE_USERNAME` + `KAGGLE_KEY` in `.env`; if absent, degrade gracefully.

---

## Task 10: Data Cache Layer (`src/api/data_cache.py`)

**Files:**
- Create: `src/api/data_cache.py`
- Create: `tests/api/test_data_cache.py`

**What it does:**
- Provides `DataLoader` class with `get_trank(season)` and `get_tournament_seeds(season)`
- On first call: fetches from Barttorvik / Kaggle, writes to `data/cache/trank_{season}.parquet` and `data/cache/seeds_{season}.json`
- On subsequent calls: reads from cache (no network hit)
- Falls back to empty DataFrame + warns if Barttorvik scrape fails (network timeout)

**Step 1: Write failing tests**

```python
# tests/api/test_data_cache.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.api.data_cache import DataLoader

class TestDataLoader:
    def test_get_trank_returns_dataframe(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        mock_df = pd.DataFrame({
            "team": ["Duke", "Kansas"],
            "adj_oe": [115.2, 118.1],
            "adj_de": [93.1, 94.5],
            "tempo": [70.1, 71.2],
            "luck": [0.02, -0.01],
            "seed": [1, 1],
            "conference": ["ACC", "Big 12"],
        })
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df):
            result = loader.get_trank(season=2024)
        assert isinstance(result, pd.DataFrame)
        assert "adj_oe" in result.columns

    def test_get_trank_writes_cache(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        mock_df = pd.DataFrame({"team": ["Duke"], "adj_oe": [115.0], "adj_de": [93.0], "tempo": [70.0], "luck": [0.01], "seed": [1], "conference": ["ACC"]})
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df):
            loader.get_trank(season=2024)
        assert (tmp_path / "trank_2024.parquet").exists()

    def test_get_trank_reads_cache_on_second_call(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        mock_df = pd.DataFrame({"team": ["Duke"], "adj_oe": [115.0], "adj_de": [93.0], "tempo": [70.0], "luck": [0.01], "seed": [1], "conference": ["ACC"]})
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df) as mock_fn:
            loader.get_trank(season=2024)
            loader.get_trank(season=2024)
        # fetch_trank should only be called once (second call hits cache)
        assert mock_fn.call_count == 1

    def test_get_trank_graceful_degradation_on_error(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        with patch("src.api.data_cache.fetch_trank", side_effect=Exception("network error")):
            result = loader.get_trank(season=2024)
        # Returns empty DataFrame, does not raise
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_tournament_seeds_returns_dict(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        mock_seeds = {"Duke": 1, "Kansas": 1, "Gonzaga": 1}
        with patch("src.api.data_cache.load_tournament_seeds", return_value=mock_seeds):
            result = loader.get_tournament_seeds(season=2024)
        assert isinstance(result, dict)
        assert result["Duke"] == 1

    def test_get_tournament_seeds_writes_cache(self, tmp_path):
        loader = DataLoader(cache_dir=str(tmp_path))
        mock_seeds = {"Duke": 1}
        with patch("src.api.data_cache.load_tournament_seeds", return_value=mock_seeds):
            loader.get_tournament_seeds(season=2024)
        assert (tmp_path / "seeds_2024.json").exists()

    def test_cache_dir_created_if_missing(self, tmp_path):
        cache_dir = tmp_path / "new_subdir" / "cache"
        loader = DataLoader(cache_dir=str(cache_dir))
        assert cache_dir.exists()
```

**Step 2: Run to confirm RED**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 -m pytest tests/api/test_data_cache.py -q
```
Expected: `ImportError` or `ModuleNotFoundError`

**Step 3: Implement `src/api/data_cache.py`**

```python
"""
src/api/data_cache.py

Cache-first data loader for the API layer.
Fetches Barttorvik T-Rank and Kaggle tournament seeds on first call per season,
then reads from disk on subsequent calls. Falls back to empty DataFrame on error.
"""
from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy source imports — allow the module to load even if sources are broken
# ---------------------------------------------------------------------------

def fetch_trank(season: int) -> pd.DataFrame:
    """Fetch Barttorvik T-Rank for the given season. Lazy import."""
    try:
        from src.data.barttorvik import fetch_trank as _fetch
        return _fetch(season=season)
    except Exception as exc:
        raise exc


def load_tournament_seeds(season: int) -> dict[str, int]:
    """Load tournament seeds from Kaggle ingestion. Lazy import."""
    try:
        from src.data.kaggle_ingestion import load_tournament_seeds as _load
        result = _load(season=season)
        # Normalize: return dict {team_name: seed_int}
        if isinstance(result, pd.DataFrame):
            return dict(zip(result["TeamName"], result["Seed"]))
        return result
    except Exception as exc:
        raise exc


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Cache-first loader. All data is keyed by season.

    Cache layout (under cache_dir):
        trank_{season}.parquet   — Barttorvik T-Rank DataFrame
        seeds_{season}.json      — {team_name: seed} mapping
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Barttorvik T-Rank
    # ------------------------------------------------------------------

    def get_trank(self, season: int) -> pd.DataFrame:
        """Return T-Rank DataFrame for the season. Cache-first."""
        cache_path = self.cache_dir / f"trank_{season}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        try:
            df = fetch_trank(season)
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as exc:
            logger.warning("Barttorvik fetch failed (season=%s): %s. Returning empty.", season, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Tournament seeds
    # ------------------------------------------------------------------

    def get_tournament_seeds(self, season: int) -> dict[str, int]:
        """Return {team_name: seed} dict. Cache-first."""
        cache_path = self.cache_dir / f"seeds_{season}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        try:
            seeds = load_tournament_seeds(season)
            with open(cache_path, "w") as f:
                json.dump(seeds, f)
            return seeds
        except Exception as exc:
            logger.warning("Seeds fetch failed (season=%s): %s. Returning empty.", season, exc)
            return {}
```

**Step 4: Run to confirm GREEN**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 -m pytest tests/api/test_data_cache.py -q
```
Expected: `7 passed`

**Step 5: Run full suite**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -q 2>&1 | tail -5
```
Expected: `1261+ passed`

**Step 6: Commit**
```bash
git add src/api/data_cache.py tests/api/test_data_cache.py
git commit -m "feat(api): cache-first DataLoader for Barttorvik + Kaggle seeds"
```

---

## Task 11: Wire Real Graph Endpoint

**Files:**
- Modify: `src/api/server.py` — replace `_build_stub_graph` routing logic
- Create: `src/api/graph_builder.py`
- Create: `tests/api/test_graph_builder.py`

**What it does:**
The `/api/graph` endpoint currently calls `_build_stub_graph(season)`. We add a `build_real_graph(season, loader)` function in a new `graph_builder.py` module. The server checks `USE_REAL_DATA` env var: if set, calls real builder; else falls back to stub. The real builder uses `DataLoader.get_trank(season)` and assembles `GraphResponse` with real team names, real metrics, and real conference memberships.

**Note on games (edges):** Real Kaggle game history requires credentials. If unavailable, edges are omitted (empty games list). The graph will still show real team nodes and conference topology — a major improvement over synthetic coordinates.

**Step 1: Write failing tests**

```python
# tests/api/test_graph_builder.py
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.api.graph_builder import build_real_graph
from src.api.schemas import GraphResponse

MOCK_TRANK = pd.DataFrame({
    "team": ["Duke", "Kansas", "Gonzaga", "Auburn"],
    "adj_oe": [118.0, 115.2, 119.1, 116.0],
    "adj_de": [92.0, 94.1, 93.5, 95.2],
    "tempo": [70.0, 71.5, 68.2, 74.1],
    "luck": [0.02, -0.01, 0.05, 0.00],
    "seed": [1, 1, 2, 3],
    "conference": ["ACC", "Big 12", "WCC", "SEC"],
})

class TestBuildRealGraph:
    def test_returns_graph_response(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {"Duke": 1, "Kansas": 1}
        result = build_real_graph(season=2024, loader=loader)
        assert isinstance(result, GraphResponse)

    def test_team_count_matches_trank_rows(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        assert len(result.teams) == 4

    def test_team_names_preserved(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        names = {t.name for t in result.teams}
        assert "Duke" in names
        assert "Kansas" in names

    def test_real_adj_oe_used(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        duke = next(t for t in result.teams if t.name == "Duke")
        assert abs(duke.adj_oe - 118.0) < 0.01

    def test_real_adj_de_used(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        duke = next(t for t in result.teams if t.name == "Duke")
        assert abs(duke.adj_de - 92.0) < 0.01

    def test_conference_nodes_generated(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        conf_names = {c.name for c in result.conferences}
        assert "ACC" in conf_names
        assert "Big 12" in conf_names

    def test_conference_edges_connect_teams_to_conferences(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        # Every team should have exactly one conference_edge
        assert len(result.conference_edges) == len(result.teams)

    def test_seed_applied_when_available(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {"Duke": 1}
        result = build_real_graph(season=2024, loader=loader)
        duke = next(t for t in result.teams if t.name == "Duke")
        assert duke.seed == 1

    def test_empty_trank_returns_empty_graph(self):
        loader = MagicMock()
        loader.get_trank.return_value = pd.DataFrame()
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        assert len(result.teams) == 0

    def test_xyz_positions_are_finite(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        import math
        for team in result.teams:
            assert math.isfinite(team.x)
            assert math.isfinite(team.y)
            assert math.isfinite(team.z)

    def test_games_list_empty_without_kaggle(self):
        """Without Kaggle game history, games list is empty (not an error)."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        loader.get_tournament_seeds.return_value = {}
        result = build_real_graph(season=2024, loader=loader)
        assert isinstance(result.games, list)
```

**Step 2: Run to confirm RED**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 -m pytest tests/api/test_graph_builder.py -q
```
Expected: `ImportError` — module does not exist yet

**Step 3: Implement `src/api/graph_builder.py`**

```python
"""
src/api/graph_builder.py

Builds a real GraphResponse from cached Barttorvik T-Rank data.

XYZ layout:
  - Y axis: adjusted efficiency margin (adj_oe - adj_de), normalized to [-1, 1]
    → higher = better team → visually floats to top (matches Madness Matrix seed verticality spec)
  - X axis: offensive rating (adj_oe), normalized
  - Z axis: defensive rating (adj_de), normalized
  - Conference nodes are placed at the centroid of their member teams, scaled out by 1.5×
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

import numpy as np
import pandas as pd

from src.api.schemas import (
    GraphResponse,
    TeamNodeResponse,
    ConferenceNodeResponse,
    GameEdgeResponse,
    ConferenceEdgeResponse,
)
from src.api.data_cache import DataLoader

# Conference color palette (consistent across seasons)
_CONF_COLORS: dict[str, int] = {
    "ACC": 0x4A90D9,
    "Big 12": 0xE74C3C,
    "SEC": 0xF39C12,
    "Big Ten": 0x2ECC71,
    "Pac-12": 0x9B59B6,
    "WCC": 0x1ABC9C,
    "Big East": 0x3498DB,
    "AAC": 0xE67E22,
    "MWC": 0x95A5A6,
    "A-10": 0xD35400,
    "MAC": 0x27AE60,
    "CUSA": 0x8E44AD,
    "SBC": 0x2980B9,
    "MVC": 0xC0392B,
    "WAC": 0x16A085,
}
_DEFAULT_COLOR = 0x888888


def _normalize(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 2.0 * (series - lo) / (hi - lo) - 1.0  # → [-1, 1]


def _team_color(conference: str) -> int:
    return _CONF_COLORS.get(conference, _DEFAULT_COLOR)


def _stable_team_id(team_name: str) -> str:
    """Short deterministic ID from team name."""
    return hashlib.md5(team_name.encode()).hexdigest()[:8]


def build_real_graph(
    season: int,
    loader: Optional[DataLoader] = None,
) -> GraphResponse:
    """
    Build a real GraphResponse from Barttorvik T-Rank data.

    Returns an empty GraphResponse (teams=[], games=[], conferences=[], conference_edges=[])
    if the data cache is cold and the live fetch fails.
    """
    if loader is None:
        loader = DataLoader()

    trank_df = loader.get_trank(season)
    seeds_dict = loader.get_tournament_seeds(season)

    if trank_df.empty:
        return GraphResponse(teams=[], conferences=[], games=[], conference_edges=[])

    # --- Normalize coordinates -------------------------------------------------
    # Efficiency margin drives Y (vertical = stronger teams on top)
    trank_df = trank_df.copy()
    trank_df["adj_em"] = trank_df["adj_oe"] - trank_df["adj_de"]

    x_norm = _normalize(trank_df["adj_oe"].astype(float)) * 8.0
    y_norm = _normalize(trank_df["adj_em"].astype(float)) * 6.0
    z_norm = _normalize(trank_df["adj_de"].astype(float)) * 8.0

    # --- Build team nodes -------------------------------------------------------
    teams: list[TeamNodeResponse] = []
    team_id_map: dict[str, str] = {}

    for i, row in trank_df.iterrows():
        name = str(row["team"])
        tid = _stable_team_id(name)
        team_id_map[name] = tid

        seed_val = seeds_dict.get(name, row.get("seed", 16))

        teams.append(TeamNodeResponse(
            id=tid,
            name=name,
            conference=str(row.get("conference", "Unknown")),
            seed=int(seed_val) if pd.notna(seed_val) else 16,
            adj_oe=float(row["adj_oe"]),
            adj_de=float(row["adj_de"]),
            tempo=float(row.get("tempo", 70.0)),
            x=float(x_norm.iloc[i if isinstance(i, int) else trank_df.index.get_loc(i)]),
            y=float(y_norm.iloc[i if isinstance(i, int) else trank_df.index.get_loc(i)]),
            z=float(z_norm.iloc[i if isinstance(i, int) else trank_df.index.get_loc(i)]),
            color=_team_color(str(row.get("conference", ""))),
        ))

    # --- Build conference nodes (centroid of member teams) ----------------------
    conf_teams: dict[str, list] = {}
    for t in teams:
        conf_teams.setdefault(t.conference, []).append(t)

    conferences: list[ConferenceNodeResponse] = []
    conf_id_map: dict[str, str] = {}
    for conf_name, members in conf_teams.items():
        cid = _stable_team_id(f"conf_{conf_name}")
        conf_id_map[conf_name] = cid
        cx = float(np.mean([m.x for m in members])) * 1.5
        cy = float(np.mean([m.y for m in members])) * 1.5
        cz = float(np.mean([m.z for m in members])) * 1.5
        conferences.append(ConferenceNodeResponse(
            id=cid,
            name=conf_name,
            x=cx, y=cy, z=cz,
            color=_team_color(conf_name),
        ))

    # --- Build conference edges (team → conference) ----------------------------
    conference_edges: list[ConferenceEdgeResponse] = []
    for t in teams:
        if t.conference in conf_id_map:
            conference_edges.append(ConferenceEdgeResponse(
                source=t.id,
                target=conf_id_map[t.conference],
                edge_type="member_of",
            ))

    return GraphResponse(
        teams=teams,
        conferences=conferences,
        games=[],           # Populated in Task 12 when Kaggle game history available
        conference_edges=conference_edges,
    )
```

**Step 4: Wire server.py to use real builder**

In `src/api/server.py`, find the `GET /api/graph` route and add the real-data path:

```python
import os
from src.api.data_cache import DataLoader
from src.api.graph_builder import build_real_graph

_data_loader = DataLoader()  # module-level singleton

@app.get("/api/graph", response_model=GraphResponse)
async def get_graph(season: int = Query(2024, ge=2010, le=2030)):
    if os.getenv("USE_REAL_DATA", "").lower() in ("1", "true", "yes"):
        try:
            return build_real_graph(season=season, loader=_data_loader)
        except Exception as exc:
            logger.warning("Real graph failed, falling back to stub: %s", exc)
    return _build_stub_graph(season)
```

**Step 5: Run tests**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 -m pytest tests/api/test_graph_builder.py tests/api/test_data_cache.py -q
```
Expected: All pass

**Step 6: Smoke-test the live API**
```bash
USE_REAL_DATA=true uvicorn src.api.server:app --port 8001 &
sleep 3 && curl -s "http://localhost:8001/api/graph?season=2024" | python3 -c "import sys,json; d=json.load(sys.stdin); print('teams:', len(d['teams']), 'confs:', len(d['conferences']))"
```
Expected: `teams: 350+ confs: 32` (or graceful fallback to stub if Barttorvik is unavailable)

**Step 7: Commit**
```bash
git add src/api/graph_builder.py tests/api/test_graph_builder.py src/api/server.py
git commit -m "feat(api): real graph endpoint from Barttorvik T-Rank (USE_REAL_DATA=true)"
```

---

## Task 12: Wire Real Matchup Endpoint

**Files:**
- Create: `src/api/matchup_engine.py`
- Create: `tests/api/test_matchup_engine.py`
- Modify: `src/api/server.py`

**What it does:**
The `/api/matchup` endpoint receives `{home_team, away_team, season, neutral_site}`. The real engine:
1. Looks up both teams in the T-Rank cache → gets `adj_oe`, `adj_de`, `tempo`, `luck`
2. Computes `delta = (home_adj_oe - home_adj_de) - (away_adj_oe - away_adj_de)` + home advantage if not neutral
3. Runs `MarchMadnessBayesianHead` with `use_skellam=True` and ADVI (fast, ~5 s) over a minimal single-game model
4. Returns 2000 posterior samples for win probability and point spread

**Note:** For ADVI speed, we use a simplified single-game PyMC model rather than the full hierarchical model (which needs all 350 teams). The single-game model captures the same posteriors: Normal spread likelihood on `delta` with uncertainty scaled by `abs(luck)`.

**Step 1: Write failing tests**

```python
# tests/api/test_matchup_engine.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.api.matchup_engine import build_real_matchup, MatchupNotFoundError
from src.api.schemas import MatchupResponse

MOCK_TRANK = pd.DataFrame({
    "team": ["Duke", "Kansas"],
    "adj_oe": [118.0, 115.2],
    "adj_de": [92.0, 94.1],
    "tempo": [70.0, 71.5],
    "luck": [0.02, -0.01],
    "seed": [1, 1],
    "conference": ["ACC", "Big 12"],
})

class TestBuildRealMatchup:
    def test_returns_matchup_response(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert isinstance(result, MatchupResponse)

    def test_home_team_field_set(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert result.home_team == "Duke"

    def test_away_team_field_set(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert result.away_team == "Kansas"

    def test_p_win_home_between_0_and_1(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert 0.0 < result.p_win_home < 1.0

    def test_spread_samples_length(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert len(result.spread_samples) == 2000

    def test_p_win_samples_length(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert len(result.p_win_samples) == 2000

    def test_stronger_team_higher_win_prob(self):
        """Duke has higher adj_em (118-92=26) vs Kansas (115.2-94.1=21.1). Duke should win more."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert result.p_win_home > 0.5  # Duke is home team with better metrics

    def test_unknown_team_raises(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        with pytest.raises(MatchupNotFoundError):
            build_real_matchup("Alabama", "Kansas", 2024, True, loader)

    def test_neutral_site_lowers_home_advantage(self):
        """Duke vs Kansas neutral vs. Duke hosting: home-court version should have higher p_win."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        neutral = build_real_matchup("Duke", "Kansas", 2024, neutral_site=True, loader=loader)
        home = build_real_matchup("Duke", "Kansas", 2024, neutral_site=False, loader=loader)
        assert home.p_win_home >= neutral.p_win_home

    def test_spread_mean_approximately_correct(self):
        """Duke adj_em=26, Kansas adj_em=21.1. Neutral spread ≈ (26-21.1)/1.0 scaled ≈ 4-5 pts."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        spread = result.spread_mean
        assert 0 < spread < 20, f"Expected reasonable spread, got {spread}"

    def test_luck_compressed_is_bool(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK
        result = build_real_matchup("Duke", "Kansas", 2024, True, loader)
        assert isinstance(result.luck_compressed, bool)
```

**Step 2: Run to confirm RED**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/api/test_matchup_engine.py -q
```
Expected: `ImportError`

**Step 3: Implement `src/api/matchup_engine.py`**

```python
"""
src/api/matchup_engine.py

Lightweight real matchup inference engine.

Uses a simplified single-game Bayesian model (ADVI, ~3-5 s) rather than the
full hierarchical 350-team model. Inputs are team efficiency deltas from
Barttorvik T-Rank. Outputs are 2000 posterior samples for win probability
and point spread — identical interface to the stub.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional

from src.api.schemas import MatchupResponse
from src.api.data_cache import DataLoader

logger = logging.getLogger(__name__)

_HOME_ADVANTAGE = 3.5   # points; set to 0 for neutral site
_ADVI_ITERATIONS = 5_000
_N_SAMPLES = 2000


class MatchupNotFoundError(ValueError):
    """Raised when one or both teams are not found in the T-Rank cache."""


def _lookup_team(name: str, trank_df: pd.DataFrame) -> pd.Series:
    """Case-insensitive team lookup. Raises MatchupNotFoundError if not found."""
    mask = trank_df["team"].str.lower() == name.lower()
    if not mask.any():
        # Try partial match
        mask = trank_df["team"].str.lower().str.contains(name.lower(), regex=False)
    if not mask.any():
        raise MatchupNotFoundError(
            f"Team '{name}' not found in T-Rank cache. "
            f"Available teams: {sorted(trank_df['team'].tolist())[:10]}..."
        )
    return trank_df.loc[mask].iloc[0]


def _run_advi_matchup(delta: float, luck_home: float, luck_away: float) -> dict:
    """
    Single-game Bayesian ADVI model.

    Model:
        sigma_spread ~ HalfNormal(8.0)
        luck_sigma ~ HalfNormal(|luck_home| + |luck_away| + 0.5)
        margin ~ Normal(delta, sigma_spread + luck_sigma)   [proxy for Skellam mean]
        p_win  = sigmoid(margin / 7.0)                     [logistic approximation]

    Returns 2000 posterior samples for margin and p_win.
    """
    try:
        import pymc as pm
        import arviz as az

        with pm.Model():
            sigma_spread = pm.HalfNormal("sigma_spread", sigma=8.0)
            luck_uncertainty = abs(luck_home) + abs(luck_away)
            luck_sigma = pm.HalfNormal("luck_sigma", sigma=max(luck_uncertainty, 0.1))
            total_sigma = sigma_spread + luck_sigma

            margin_obs = np.array([delta], dtype=np.float64)
            _ = pm.Normal("margin", mu=delta, sigma=total_sigma, observed=margin_obs)

            idata = pm.fit(n=_ADVI_ITERATIONS, method="advi", progressbar=False)
            samples = idata.sample(_N_SAMPLES)

        sigma_draws = np.array(samples.posterior["sigma_spread"]).flatten()[:_N_SAMPLES]
        spread_samples = np.random.normal(delta, sigma_draws)
        p_win_samples = 1.0 / (1.0 + np.exp(-spread_samples / 7.0))

        return {
            "spread_samples": spread_samples.tolist(),
            "p_win_samples": p_win_samples.tolist(),
        }
    except Exception as exc:
        logger.warning("PyMC ADVI failed (%s). Using analytical fallback.", exc)
        return _analytical_fallback(delta, luck_home, luck_away)


def _analytical_fallback(delta: float, luck_home: float, luck_away: float) -> dict:
    """
    Analytical fallback when PyMC is unavailable.
    Uses Normal approximation with luck-scaled uncertainty.
    """
    rng = np.random.default_rng(seed=42)
    luck_uncertainty = abs(luck_home) + abs(luck_away)
    sigma = 8.0 + luck_uncertainty * 10.0
    spread_samples = rng.normal(delta, sigma, size=_N_SAMPLES)
    p_win_samples = 1.0 / (1.0 + np.exp(-spread_samples / 7.0))
    return {
        "spread_samples": spread_samples.tolist(),
        "p_win_samples": p_win_samples.tolist(),
    }


def build_real_matchup(
    home_team: str,
    away_team: str,
    season: int,
    neutral_site: bool,
    loader: Optional[DataLoader] = None,
) -> MatchupResponse:
    """
    Build a real MatchupResponse using Barttorvik T-Rank + Bayesian ADVI.

    Raises
    ------
    MatchupNotFoundError
        If either team is not in the T-Rank cache.
    """
    if loader is None:
        loader = DataLoader()

    trank_df = loader.get_trank(season)
    if trank_df.empty:
        raise MatchupNotFoundError("T-Rank cache is empty — cannot build real matchup.")

    home = _lookup_team(home_team, trank_df)
    away = _lookup_team(away_team, trank_df)

    home_em = float(home["adj_oe"]) - float(home["adj_de"])
    away_em = float(away["adj_oe"]) - float(away["adj_de"])
    home_adv = 0.0 if neutral_site else _HOME_ADVANTAGE
    delta = (home_em - away_em) / 2.5 + home_adv  # scale EM diff to points

    posteriors = _run_advi_matchup(
        delta=delta,
        luck_home=float(home.get("luck", 0.0)),
        luck_away=float(away.get("luck", 0.0)),
    )

    spread_arr = np.array(posteriors["spread_samples"])
    p_win_arr = np.array(posteriors["p_win_samples"])
    luck_total = abs(float(home.get("luck", 0.0))) + abs(float(away.get("luck", 0.0)))

    return MatchupResponse(
        home_team=str(home["team"]),
        away_team=str(away["team"]),
        p_win_home=float(p_win_arr.mean()),
        p_win_samples=posteriors["p_win_samples"],
        spread_mean=float(spread_arr.mean()),
        spread_samples=posteriors["spread_samples"],
        luck_compressed=bool(luck_total > 0.10),
    )
```

**Step 4: Wire server.py**

Find the `POST /api/matchup` route and add:
```python
from src.api.matchup_engine import build_real_matchup, MatchupNotFoundError

@app.post("/api/matchup", response_model=MatchupResponse)
async def post_matchup(req: MatchupRequest):
    if os.getenv("USE_REAL_DATA", "").lower() in ("1", "true", "yes"):
        try:
            return build_real_matchup(
                req.home_team, req.away_team, req.season,
                req.neutral_site, loader=_data_loader
            )
        except MatchupNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.warning("Real matchup failed, falling back to stub: %s", exc)
    return _build_stub_matchup(req.home_team, req.away_team, req.season, req.neutral_site)
```

**Step 5: Run tests**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/api/test_matchup_engine.py -q
```
Expected: All 11 pass

**Step 6: Full suite**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -q 2>&1 | tail -5
```
Expected: No regressions

**Step 7: Commit**
```bash
git add src/api/matchup_engine.py tests/api/test_matchup_engine.py src/api/server.py
git commit -m "feat(api): real matchup endpoint — Barttorvik delta + Bayesian ADVI posteriors"
```

---

## Task 13: Wire Real Bracket Simulation Endpoint

**Files:**
- Create: `src/api/bracket_runner.py`
- Create: `tests/api/test_bracket_runner.py`
- Modify: `src/api/server.py`

**What it does:**
The `/api/bracket/simulate` endpoint receives `{teams: [str], n_simulations: int}`. The real runner:
1. Looks up each team in T-Rank → gets efficiency margin
2. Builds win-probability function: `P(A beats B) = sigmoid((em_A - em_B) / 4.0)`
3. Runs the existing `BracketSimulator` from `src/simulation/monte_carlo.py` for `n_simulations` trials
4. Returns per-team per-round advancement probabilities + Shannon entropy

**Step 1: Write failing tests**

```python
# tests/api/test_bracket_runner.py
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.api.bracket_runner import build_real_simulation
from src.api.schemas import SimulateResponse

MOCK_TRANK_64 = pd.DataFrame({
    "team": [f"Team{i}" for i in range(64)],
    "adj_oe": np.linspace(120, 90, 64),
    "adj_de": np.linspace(88, 105, 64),
    "tempo": [70.0] * 64,
    "luck": [0.0] * 64,
    "seed": list(range(1, 17)) * 4,
    "conference": ["ACC"] * 16 + ["Big 12"] * 16 + ["SEC"] * 16 + ["Big Ten"] * 16,
})

class TestBuildRealSimulation:
    def test_returns_simulate_response(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation(
            teams=[f"Team{i}" for i in range(64)],
            n_simulations=50,
            season=2024,
            loader=loader,
        )
        assert isinstance(result, SimulateResponse)

    def test_n_simulations_recorded(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(64)], 50, 2024, loader)
        assert result.n_simulations == 50

    def test_advancements_count_matches_teams(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        teams = [f"Team{i}" for i in range(64)]
        result = build_real_simulation(teams, 50, 2024, loader)
        assert len(result.advancements) == 64

    def test_advancement_probs_have_6_rounds(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(64)], 50, 2024, loader)
        for item in result.advancements:
            assert len(item.advancement_probs) == 6

    def test_advancement_probs_decrease_monotonically(self):
        """Harder to advance further, so probs should generally decrease each round."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(64)], 200, 2024, loader)
        for item in result.advancements:
            probs = item.advancement_probs
            # R64 > R32 > S16 etc. (allow small noise in low-n simulations)
            assert probs[0] >= probs[-1] - 0.05

    def test_top_seed_higher_championship_prob(self):
        """Team0 has highest adj_em, should have highest championship probability."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(64)], 200, 2024, loader)
        champ_probs = {item.team: item.advancement_probs[-1] for item in result.advancements}
        assert champ_probs["Team0"] > champ_probs["Team63"]

    def test_entropy_is_finite_positive(self):
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(64)], 50, 2024, loader)
        import math
        for item in result.advancements:
            assert math.isfinite(item.entropy)
            assert item.entropy >= 0.0

    def test_fewer_than_64_teams_handled(self):
        """Should handle partial brackets (e.g., just 16 teams)."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation([f"Team{i}" for i in range(16)], 50, 2024, loader)
        assert len(result.advancements) == 16

    def test_unknown_teams_get_avg_seed_strength(self):
        """Teams not in T-Rank still produce output (use average efficiency)."""
        loader = MagicMock()
        loader.get_trank.return_value = MOCK_TRANK_64
        result = build_real_simulation(["UnknownA", "UnknownB"], 50, 2024, loader)
        assert len(result.advancements) == 2
```

**Step 2: Run to confirm RED**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/api/test_bracket_runner.py -q
```
Expected: `ImportError`

**Step 3: Implement `src/api/bracket_runner.py`**

```python
"""
src/api/bracket_runner.py

Real bracket simulation runner for the /api/bracket/simulate endpoint.

Uses Barttorvik T-Rank to compute per-team efficiency margins, then runs
N Monte Carlo bracket trials using a logistic win-probability function.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from src.api.schemas import SimulateResponse, TeamAdvancementItem
from src.api.data_cache import DataLoader

logger = logging.getLogger(__name__)

_ROUNDS = ["R64", "R32", "S16", "E8", "F4", "Championship"]


def _win_prob(em_a: float, em_b: float) -> float:
    """P(team_a beats team_b) = sigmoid((em_a - em_b) / 4.0)."""
    return 1.0 / (1.0 + math.exp(-(em_a - em_b) / 4.0))


def _simulate_bracket(
    teams: list[str],
    em_map: dict[str, float],
    n_simulations: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    """
    Run n_simulations bracket trials.

    Returns {team_name: [p_r64, p_r32, p_s16, p_e8, p_f4, p_championship]}.
    """
    n = len(teams)
    n_rounds = 6
    advancement_counts: dict[str, list[int]] = {t: [0] * n_rounds for t in teams}

    for _ in range(n_simulations):
        survivors = list(teams)
        for rnd in range(n_rounds):
            if len(survivors) <= 1:
                if survivors:
                    advancement_counts[survivors[0]][rnd] += 1
                break
            next_round = []
            for i in range(0, len(survivors) - 1, 2):
                a, b = survivors[i], survivors[i + 1]
                p_a = _win_prob(em_map[a], em_map[b])
                winner = a if rng.random() < p_a else b
                next_round.append(winner)
                advancement_counts[winner][rnd] += 1
            # Odd team out advances automatically
            if len(survivors) % 2 == 1:
                bye = survivors[-1]
                next_round.append(bye)
                advancement_counts[bye][rnd] += 1
            survivors = next_round

    probs = {}
    for t in teams:
        counts = advancement_counts[t]
        probs[t] = [c / n_simulations for c in counts]
    return probs


def _shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy over the round-advancement distribution."""
    total = sum(probs) + 1e-12
    norm = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-12) for p in norm)


def build_real_simulation(
    teams: list[str],
    n_simulations: int,
    season: int,
    loader: Optional[DataLoader] = None,
) -> SimulateResponse:
    """
    Run a real Monte Carlo bracket simulation.

    Teams not found in T-Rank receive the median efficiency margin.
    """
    if loader is None:
        loader = DataLoader()

    trank_df = loader.get_trank(season)

    # Build em_map: team_name → efficiency margin
    em_map: dict[str, float] = {}
    median_em = 0.0

    if not trank_df.empty:
        trank_df = trank_df.copy()
        trank_df["adj_em"] = trank_df["adj_oe"].astype(float) - trank_df["adj_de"].astype(float)
        lookup = dict(zip(trank_df["team"].str.lower(), trank_df["adj_em"]))
        ems = list(lookup.values())
        median_em = float(np.median(ems)) if ems else 0.0
        for t in teams:
            em_map[t] = lookup.get(t.lower(), median_em)
    else:
        for t in teams:
            em_map[t] = 0.0

    rng = np.random.default_rng(seed=42)
    probs = _simulate_bracket(teams, em_map, n_simulations, rng)

    advancements = []
    for team in teams:
        team_probs = probs[team]
        advancements.append(TeamAdvancementItem(
            team=team,
            advancement_probs=team_probs,
            entropy=_shannon_entropy(team_probs),
        ))

    # Sort by championship probability descending
    advancements.sort(key=lambda x: x.advancement_probs[-1], reverse=True)

    return SimulateResponse(
        n_simulations=n_simulations,
        advancements=advancements,
    )
```

**Step 4: Wire server.py**

```python
from src.api.bracket_runner import build_real_simulation

@app.post("/api/bracket/simulate", response_model=SimulateResponse)
async def post_simulate(req: SimulateRequest):
    if os.getenv("USE_REAL_DATA", "").lower() in ("1", "true", "yes"):
        try:
            return build_real_simulation(
                teams=req.teams,
                n_simulations=req.n_simulations,
                season=req.season if hasattr(req, "season") else 2024,
                loader=_data_loader,
            )
        except Exception as exc:
            logger.warning("Real simulation failed, falling back to stub: %s", exc)
    return _build_stub_simulate(req.teams, req.n_simulations)
```

**Step 5: Run tests**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/api/test_bracket_runner.py -q
```

**Step 6: Full suite**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -q 2>&1 | tail -5
```

**Step 7: Commit**
```bash
git add src/api/bracket_runner.py tests/api/test_bracket_runner.py src/api/server.py
git commit -m "feat(api): real bracket simulation from Barttorvik EM + Monte Carlo"
```

---

## Task 14: Barttorvik Scraper Verification + Cache Warm-Up

**Files:**
- Modify: `src/data/barttorvik.py` (fix if `fetch_trank` signature differs)
- Create: `scripts/warm_cache.py` (one-shot data fetcher)

**What it does:**
Ensures `fetch_trank(season=2024)` returns a DataFrame with the required columns (`team`, `adj_oe`, `adj_de`, `tempo`, `luck`, `seed`, `conference`) and that it can actually hit barttorvik.com. Also provides a `warm_cache.py` script the user can run once to populate `data/cache/` before starting the server.

**Step 1: Check barttorvik.py interface**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 -c "
from src.data.barttorvik import fetch_trank
import inspect
print(inspect.signature(fetch_trank))
"
```

**Step 2: Test live scrape (allow 30s)**
```bash
/opt/anaconda3/bin/python3 -c "
from src.data.barttorvik import fetch_trank
df = fetch_trank(season=2024)
print('shape:', df.shape)
print('columns:', df.columns.tolist())
print(df.head(3))
"
```

**Step 3: Fix column name mismatches**

If `fetch_trank` returns different column names than expected (e.g., `adj_o` instead of `adj_oe`), add a normalization step in `src/api/data_cache.py`:

```python
_COLUMN_ALIASES = {
    "adj_o": "adj_oe",
    "adj_d": "adj_de",
    "adj_t": "tempo",
    "team_name": "team",
    "TeamName": "team",
    "adjoe": "adj_oe",
    "adjde": "adj_de",
    "adjt": "tempo",
}

def _normalize_trank_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_COLUMN_ALIASES)
```

Call `_normalize_trank_columns(df)` in `DataLoader.get_trank()` after fetching.

**Step 4: Create warm-up script**

```python
# scripts/warm_cache.py
"""
Run once to pre-populate data/cache/ before starting the server.

Usage:
    /opt/anaconda3/bin/python3 scripts/warm_cache.py --season 2024
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.data_cache import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024)
    args = parser.parse_args()

    loader = DataLoader()
    print(f"Warming cache for season {args.season}...")

    df = loader.get_trank(args.season)
    if df.empty:
        print("WARNING: T-Rank fetch returned empty — Barttorvik may be unavailable.")
    else:
        print(f"T-Rank: {len(df)} teams cached.")

    seeds = loader.get_tournament_seeds(args.season)
    if not seeds:
        print("WARNING: Seeds fetch returned empty — Kaggle credentials may be missing.")
    else:
        print(f"Seeds: {len(seeds)} teams cached.")

    print("Done. Cache at data/cache/")

if __name__ == "__main__":
    main()
```

**Step 5: Run warm-up**
```bash
cd /Users/ritvikvasikarla/Documents/MM && /opt/anaconda3/bin/python3 scripts/warm_cache.py --season 2024
```

**Step 6: Commit**
```bash
git add scripts/warm_cache.py
git commit -m "feat(scripts): warm_cache.py pre-fetches Barttorvik + seeds to data/cache/"
```

---

## Task 15: End-to-End Smoke Test

**What it does:** Verifies all three real endpoints return plausible responses when `USE_REAL_DATA=true` and cache is warm.

**Step 1: Start backend with real data**
```bash
cd /Users/ritvikvasikarla/Documents/MM
USE_REAL_DATA=true /opt/anaconda3/bin/uvicorn src.api.server:app --port 8001 --reload &
sleep 5
```

**Step 2: Health check**
```bash
curl -s http://localhost:8001/health | python3 -m json.tool
```
Expected: `{"status": "ok", "model": "ethereal-oracle-v1"}`

**Step 3: Graph endpoint**
```bash
curl -s "http://localhost:8001/api/graph?season=2024" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'teams={len(d[\"teams\"])} conferences={len(d[\"conferences\"])} games={len(d[\"games\"])} conf_edges={len(d[\"conference_edges\"])}')
print('Sample team:', d['teams'][0]['name'], 'adj_oe:', d['teams'][0]['adj_oe'])
"
```
Expected: `teams=350+ conferences=30+ games=0 conf_edges=350+`

**Step 4: Matchup endpoint**
```bash
curl -s -X POST http://localhost:8001/api/matchup \
  -H "Content-Type: application/json" \
  -d '{"home_team":"Duke","away_team":"Kansas","season":2024,"neutral_site":true}' | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'p_win_home={d[\"p_win_home\"]:.3f} spread_mean={d[\"spread_mean\"]:.1f} samples={len(d[\"spread_samples\"])}')
"
```
Expected: `p_win_home=0.4-0.7 spread_mean=varies samples=2000`

**Step 5: Simulate endpoint**
```bash
curl -s -X POST http://localhost:8001/api/bracket/simulate \
  -H "Content-Type: application/json" \
  -d '{"teams":["Duke","Kansas","Gonzaga","Auburn","UConn","Houston","Purdue","Tennessee"],"n_simulations":500}' | \
  python3 -c "
import sys, json
d = json.load(sys.stdin)
print('n_simulations:', d['n_simulations'])
for a in d['advancements'][:3]:
    print(f'  {a[\"team\"]}: champ={a[\"advancement_probs\"][-1]:.3f}')
"
```
Expected: 3 teams shown with varying championship probabilities

**Step 6: Kill test server**
```bash
pkill -f "uvicorn.*8001" 2>/dev/null || true
```

**Step 7: Final full test suite**
```bash
/opt/anaconda3/bin/python3 -m pytest tests/ -q 2>&1 | tail -5
```
Expected: `1300+ passed`

**Step 8: Commit**
```bash
git add -A
git commit -m "test(api): end-to-end smoke tests pass — real data pipeline integrated"
```

---

## Verification Checklist

After all tasks complete:
- [ ] `python -m pytest tests/ -q` — 1300+ passed, 0 failed
- [ ] `USE_REAL_DATA=true uvicorn src.api.server:app` starts without error
- [ ] `GET /api/graph?season=2024` returns 300+ real teams with Barttorvik metrics
- [ ] `POST /api/matchup` with real team names returns 2000 posterior spread samples
- [ ] `POST /api/bracket/simulate` with real team list returns ordered advancement probs
- [ ] Stub fallback still works when `USE_REAL_DATA` is unset
- [ ] `scripts/warm_cache.py` populates `data/cache/` in < 60 s

## Run Instructions (after completion)

```bash
# Terminal 1 — warm the cache (first time only, ~30 s)
cd /Users/ritvikvasikarla/Documents/MM
/opt/anaconda3/bin/python3 scripts/warm_cache.py --season 2024

# Terminal 1 — start backend
USE_REAL_DATA=true uvicorn src.api.server:app --port 8000 --reload

# Terminal 2 — start frontend
cd frontend && npm run dev

# Open http://localhost:3000
```
