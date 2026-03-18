# Madness Matrix Full Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the existing codebase into a production-deployed, zero-cost, autonomous NCAA March Madness prediction engine called "Madness Matrix" — running 3× daily via GitHub Actions, storing all predictions in Supabase, and surfacing them in a modern Next.js 15 frontend on Vercel.

**Architecture:** GitHub Actions runs the full ST-GNN + Bayesian + Monte Carlo stack and writes results to Supabase. Vercel hosts a Next.js 15 frontend that reads Supabase directly (no persistent backend). Supabase Realtime pushes intel alerts and prediction updates to the frontend live.

**Tech Stack:** Python (PyMC, PyTorch Geometric, sklearn), GitHub Actions, Supabase (Postgres + Realtime), Next.js 15 + Tailwind + shadcn/ui + Recharts + React Query, Vercel Hobby (free)

---

## Phase 1 — Foundation

---

### Task 1: Supabase Database Schema

**Files:**
- Create: `supabase/migrations/001_initial_schema.sql`
- Create: `supabase/migrations/002_indexes.sql`

**Step 1: Write the schema migration**

```sql
-- supabase/migrations/001_initial_schema.sql

CREATE TABLE IF NOT EXISTS teams (
    id            BIGSERIAL PRIMARY KEY,
    season        INTEGER NOT NULL,
    name          TEXT NOT NULL,
    conference    TEXT,
    seed          INTEGER,
    adj_oe        NUMERIC,
    adj_de        NUMERIC,
    adj_em        NUMERIC,
    tempo         NUMERIC,
    luck          NUMERIC,
    sos           NUMERIC,
    coach         TEXT,
    region        TEXT,
    scraped_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, name)
);

CREATE TABLE IF NOT EXISTS matchup_predictions (
    id              BIGSERIAL PRIMARY KEY,
    season          INTEGER NOT NULL,
    team_a          TEXT NOT NULL,
    team_b          TEXT NOT NULL,
    p_win_a         NUMERIC NOT NULL,
    spread_mean     NUMERIC,
    spread_std      NUMERIC,
    samples_json    JSONB,
    model_version   TEXT,
    computed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, team_a, team_b, computed_at)
);

CREATE TABLE IF NOT EXISTS bracket_runs (
    id                  BIGSERIAL PRIMARY KEY,
    season              INTEGER NOT NULL,
    run_date            DATE NOT NULL,
    n_simulations       INTEGER,
    advancement_probs   JSONB,
    champion_prob       JSONB,
    model_version       TEXT,
    runtime_secs        NUMERIC,
    computed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                    BIGSERIAL PRIMARY KEY,
    season                INTEGER NOT NULL,
    snapshot_date         DATE NOT NULL,
    team                  TEXT NOT NULL,
    championship_prob     NUMERIC,
    final_four_prob       NUMERIC,
    elite_eight_prob      NUMERIC,
    sweet_sixteen_prob    NUMERIC,
    brier_score           NUMERIC,
    log_loss              NUMERIC,
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, snapshot_date, team)
);

CREATE TABLE IF NOT EXISTS intel_alerts (
    id                  BIGSERIAL PRIMARY KEY,
    alert_id            TEXT UNIQUE NOT NULL,
    source              TEXT NOT NULL,
    content             TEXT NOT NULL,
    keywords            TEXT[],
    severity            TEXT CHECK (severity IN ('low','medium','high','critical')),
    teams_mentioned     TEXT[],
    region              TEXT,
    needs_verification  BOOLEAN DEFAULT FALSE,
    url                 TEXT,
    alerted_at          TIMESTAMPTZ DEFAULT NOW(),
    resolved            BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS game_results (
    id               BIGSERIAL PRIMARY KEY,
    season           INTEGER NOT NULL,
    game_date        DATE NOT NULL,
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    home_score       INTEGER,
    away_score       INTEGER,
    neutral_site     BOOLEAN DEFAULT FALSE,
    tournament_round TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (season, game_date, home_team, away_team)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                      BIGSERIAL PRIMARY KEY,
    run_date                DATE NOT NULL,
    trigger                 TEXT NOT NULL,
    status                  TEXT CHECK (status IN ('success','partial','failed')),
    teams_updated           INTEGER,
    predictions_computed    INTEGER,
    alerts_found            INTEGER,
    duration_secs           NUMERIC,
    error_log               TEXT,
    created_at              TIMESTAMPTZ DEFAULT NOW()
);
```

**Step 2: Write the indexes migration**

```sql
-- supabase/migrations/002_indexes.sql

CREATE INDEX IF NOT EXISTS idx_teams_season ON teams (season);
CREATE INDEX IF NOT EXISTS idx_matchup_season ON matchup_predictions (season, computed_at DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_season_date ON prediction_snapshots (season, snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_snapshots_team ON prediction_snapshots (team, season);
CREATE INDEX IF NOT EXISTS idx_intel_resolved ON intel_alerts (resolved, alerted_at DESC);
CREATE INDEX IF NOT EXISTS idx_bracket_runs_date ON bracket_runs (run_date DESC, season);
CREATE INDEX IF NOT EXISTS idx_game_results_season ON game_results (season, game_date);

-- Enable Realtime on intel_alerts and prediction_snapshots
ALTER TABLE intel_alerts REPLICA IDENTITY FULL;
ALTER TABLE prediction_snapshots REPLICA IDENTITY FULL;
```

**Step 3: Apply to Supabase**
Go to Supabase project → SQL Editor → run `001_initial_schema.sql`, then `002_indexes.sql`.
Enable Realtime on `intel_alerts` and `prediction_snapshots` in Supabase dashboard → Database → Replication.

**Step 4: Commit**

```bash
git add supabase/
git commit -m "feat(db): initial Supabase schema — 7 tables, indexes, Realtime on intel+snapshots"
```

---

### Task 2: Purge All Synthetic Data

**Files:**
- Delete: `frontend/lib/mock-data.ts`
- Modify: `src/api/server.py` — remove all `_build_stub_*` functions and `_TEAMS_BY_CONF`
- Modify: `scripts/warm_cache.py` — remove stub fallback message
- Modify: `frontend/` — remove `USE_REAL_DATA` checks and `StubDataBanner`

**Step 1: Write a test that confirms no stub paths remain**

```python
# tests/test_no_stubs.py
import ast
import pathlib

def _python_sources():
    root = pathlib.Path("src")
    return list(root.rglob("*.py"))

def test_no_build_stub_functions():
    for path in _python_sources():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert not node.name.startswith("_build_stub"), (
                    f"Stub function {node.name!r} found in {path}"
                )

def test_no_use_real_data_checks():
    for path in _python_sources():
        content = path.read_text()
        assert "USE_REAL_DATA" not in content, (
            f"USE_REAL_DATA env var found in {path} — remove stub gating"
        )
```

**Step 2: Run test to verify it fails (stubs still exist)**

```bash
python -m pytest tests/test_no_stubs.py -v
```

**Step 3: Delete mock-data.ts**

```bash
rm frontend/lib/mock-data.ts
```

**Step 4: Remove stub functions from server.py**
Read `src/api/server.py` and delete every function beginning with `_build_stub_` and the `_TEAMS_BY_CONF` dict. Replace any route that called these functions with an HTTP 503 response:

```python
from fastapi import HTTPException

@app.get("/api/teams")
async def get_teams(season: int = 2026):
    raise HTTPException(
        status_code=503,
        detail="Teams endpoint migrated to Supabase. Read from Supabase directly."
    )
```

**Step 5: Remove USE_REAL_DATA gating from warm_cache.py**
Delete the lines in `scripts/warm_cache.py` that reference `USE_REAL_DATA` or suggest running with `USE_REAL_DATA=true`.

**Step 6: Run tests to verify they pass**

```bash
python -m pytest tests/test_no_stubs.py -v
```

**Step 7: Commit**

```bash
git add -u
git commit -m "feat(purge): delete all synthetic/stub data — no mock-data.ts, no _build_stub_* functions"
```

---

### Task 3: SupabaseWriter Pipeline Module

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/supabase_writer.py`
- Create: `tests/pipeline/__init__.py`
- Create: `tests/pipeline/test_supabase_writer.py`

**Step 1: Write failing tests**

```python
# tests/pipeline/test_supabase_writer.py
from unittest.mock import MagicMock, call
import pytest
from src.pipeline.supabase_writer import SupabaseWriter


@pytest.fixture
def mock_client():
    client = MagicMock()
    table = MagicMock()
    client.table.return_value = table
    table.upsert.return_value = table
    table.insert.return_value = table
    table.execute.return_value = MagicMock(data=[{"id": 1}])
    return client


def test_upsert_teams_calls_correct_table(mock_client):
    writer = SupabaseWriter(client=mock_client)
    rows = [{"season": 2026, "name": "Duke", "adj_oe": 120.5}]
    writer.upsert_teams(rows)
    mock_client.table.assert_called_with("teams")


def test_upsert_snapshots(mock_client):
    writer = SupabaseWriter(client=mock_client)
    rows = [{"season": 2026, "snapshot_date": "2026-03-12", "team": "Duke",
             "championship_prob": 0.12}]
    writer.upsert_snapshots(rows)
    mock_client.table.assert_called_with("prediction_snapshots")


def test_insert_intel_alert(mock_client):
    writer = SupabaseWriter(client=mock_client)
    alert = {
        "alert_id": "espn-abc123",
        "source": "ESPN",
        "content": "Player X out with knee injury",
        "severity": "high",
        "teams_mentioned": ["Duke"],
        "needs_verification": False,
    }
    writer.insert_intel_alert(alert)
    mock_client.table.assert_called_with("intel_alerts")


def test_log_pipeline_run(mock_client):
    writer = SupabaseWriter(client=mock_client)
    writer.log_pipeline_run(
        trigger="full",
        status="success",
        teams_updated=68,
        predictions_computed=2278,
        alerts_found=3,
        duration_secs=142.7,
    )
    mock_client.table.assert_called_with("pipeline_runs")


def test_upsert_bracket_run(mock_client):
    writer = SupabaseWriter(client=mock_client)
    writer.upsert_bracket_run({
        "season": 2026,
        "run_date": "2026-03-12",
        "n_simulations": 10000,
        "advancement_probs": {},
        "champion_prob": {},
        "model_version": "v1.0",
    })
    mock_client.table.assert_called_with("bracket_runs")
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_supabase_writer.py -v
```

**Step 3: Implement SupabaseWriter**

```python
# src/pipeline/supabase_writer.py
"""Write pipeline outputs to Supabase tables."""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_client():
    """Return a Supabase client using env vars."""
    from supabase import create_client  # type: ignore

    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


class SupabaseWriter:
    """Thin wrapper that upserts/inserts pipeline outputs to Supabase."""

    def __init__(self, client=None) -> None:
        self._client = client or _get_client()

    # ------------------------------------------------------------------
    def _upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        if not rows:
            return
        (
            self._client.table(table)
            .upsert(rows, on_conflict=on_conflict)
            .execute()
        )

    # ------------------------------------------------------------------
    def upsert_teams(self, rows: list[dict[str, Any]]) -> None:
        """Upsert team efficiency rows for a season."""
        self._upsert("teams", rows, "season,name")

    def upsert_snapshots(self, rows: list[dict[str, Any]]) -> None:
        """Upsert prediction snapshot rows."""
        self._upsert("prediction_snapshots", rows, "season,snapshot_date,team")

    def upsert_matchup_predictions(self, rows: list[dict[str, Any]]) -> None:
        """Upsert matchup-level posterior rows."""
        self._upsert("matchup_predictions", rows, "season,team_a,team_b,computed_at")

    def upsert_bracket_run(self, row: dict[str, Any]) -> None:
        """Insert a bracket run record."""
        self._client.table("bracket_runs").insert(row).execute()

    def insert_intel_alert(self, alert: dict[str, Any]) -> None:
        """Insert a single intel alert (skip if alert_id already exists)."""
        self._client.table("intel_alerts").upsert(
            alert, on_conflict="alert_id"
        ).execute()

    def upsert_game_results(self, rows: list[dict[str, Any]]) -> None:
        """Upsert game result rows."""
        self._upsert("game_results", rows, "season,game_date,home_team,away_team")

    def log_pipeline_run(
        self,
        trigger: str,
        status: str,
        teams_updated: int = 0,
        predictions_computed: int = 0,
        alerts_found: int = 0,
        duration_secs: float = 0.0,
        error_log: str | None = None,
    ) -> None:
        """Insert a pipeline_runs record."""
        row = {
            "run_date": datetime.date.today().isoformat(),
            "trigger": trigger,
            "status": status,
            "teams_updated": teams_updated,
            "predictions_computed": predictions_computed,
            "alerts_found": alerts_found,
            "duration_secs": round(duration_secs, 2),
            "error_log": error_log,
        }
        self._client.table("pipeline_runs").insert(row).execute()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_supabase_writer.py -v
```

**Step 5: Commit**

```bash
git add src/pipeline/ tests/pipeline/
git commit -m "feat(pipeline): SupabaseWriter — upsert teams/snapshots/bracket/alerts/results"
```

---

## Phase 2 — Prediction Engine Additions

---

### Task 4: Isotonic Regression Calibrator

**Files:**
- Create: `src/model/calibration.py`
- Create: `tests/model/test_calibration.py`

**Step 1: Write failing tests**

```python
# tests/model/test_calibration.py
import numpy as np
import pytest
import tempfile
from pathlib import Path
from src.model.calibration import IsotonicCalibrator, brier_score


def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1, 0, 1, 0])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


def test_brier_score_worst():
    probs = np.array([0.0, 1.0])
    outcomes = np.array([1, 0])
    assert brier_score(probs, outcomes) == pytest.approx(1.0)


def test_calibrator_fit_predict():
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    calibrated = cal.predict(raw)
    assert calibrated.shape == raw.shape
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibrator_not_fitted_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(np.array([0.5, 0.6]))


def test_calibrator_save_load_joblib():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 100)
    outcomes = (rng.uniform(0, 1, 100) < raw).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "calibrator.joblib"
        cal.save(path)
        loaded = IsotonicCalibrator.load(path)
    pred_original = cal.predict(raw)
    pred_loaded = loaded.predict(raw)
    np.testing.assert_array_almost_equal(pred_original, pred_loaded)
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/test_calibration.py -v
```

**Step 3: Implement calibration.py (joblib — no pickle)**

```python
# src/model/calibration.py
"""Isotonic regression calibration for Bayesian posterior win probabilities.

Fits a monotone calibration curve on historical Brier scores (2012-2024)
to correct overconfident chalk predictions from the Bayesian head.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_DEFAULT_CALIBRATOR_PATH = Path("artifacts/calibrator.joblib")


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute mean Brier Score: E[(p - y)^2]. Lower is better."""
    return float(np.mean((probs - outcomes) ** 2))


class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression for win-probability calibration.

    Usage::

        cal = IsotonicCalibrator()
        cal.fit(raw_probs, binary_outcomes)     # fit on historical games
        calibrated = cal.predict(new_probs)      # apply to new predictions
        cal.save("artifacts/calibrator.joblib")  # persist with joblib (safe)

    The underlying IsotonicRegression guarantees a monotone non-decreasing
    mapping so a higher raw probability always maps to a higher calibrated one.
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on (raw_probs, binary_outcomes) pairs."""
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(raw_probs, outcomes)
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for *raw_probs*."""
        if self._iso is None:
            raise RuntimeError(
                "IsotonicCalibrator is not fitted. Call .fit() first."
            )
        return self._iso.predict(raw_probs)

    def save(self, path: Path | str = _DEFAULT_CALIBRATOR_PATH) -> None:
        """Persist calibrator using joblib (sklearn-recommended serializer)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._iso, path)

    @classmethod
    def load(cls, path: Path | str = _DEFAULT_CALIBRATOR_PATH) -> "IsotonicCalibrator":
        """Load a previously saved calibrator from *path*."""
        obj = cls()
        obj._iso = joblib.load(Path(path))
        return obj
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/model/test_calibration.py -v
```

**Step 5: Commit**

```bash
git add src/model/calibration.py tests/model/test_calibration.py
git commit -m "feat(model): IsotonicCalibrator — joblib persistence, brier_score utility"
```

---

### Task 5: Conference RPI Module

**Files:**
- Create: `src/data/conference_rpi.py`
- Create: `tests/data/test_conference_rpi.py`

**Step 1: Write failing tests**

```python
# tests/data/test_conference_rpi.py
import pandas as pd
import pytest
from src.data.conference_rpi import compute_conference_rpi, assign_rpi_tiers, ConferenceRPI


def _make_teams_df():
    return pd.DataFrame({
        "conference": ["ACC", "ACC", "MAC", "MAC", "Big 12", "Big 12"],
        "adj_em": [20.0, 18.0, 5.0, 3.0, 22.0, 24.0],
        "sos":    [15.0, 14.0, 8.0, 7.0, 18.0, 19.0],
    })


def test_compute_conference_rpi_returns_dataclass_list():
    df = _make_teams_df()
    result = compute_conference_rpi(df)
    assert len(result) == 3
    assert all(isinstance(r, ConferenceRPI) for r in result)


def test_rpi_score_formula():
    df = _make_teams_df()
    result = {r.conference: r for r in compute_conference_rpi(df)}
    # ACC: 0.6 * mean(20,18) + 0.4 * mean(15,14) = 0.6*19 + 0.4*14.5 = 11.4+5.8=17.2
    assert result["ACC"].rpi_score == pytest.approx(17.2, abs=0.01)


def test_rpi_sorted_descending():
    df = _make_teams_df()
    result = compute_conference_rpi(df)
    scores = [r.rpi_score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_assign_rpi_tiers_five_levels():
    df = _make_teams_df()
    rpis = compute_conference_rpi(df)
    tiers = assign_rpi_tiers(rpis)
    assert all(1 <= t.tier <= 5 for t in tiers)


def test_assign_rpi_tiers_top_is_tier_1():
    df = _make_teams_df()
    rpis = compute_conference_rpi(df)
    tiers = assign_rpi_tiers(rpis)
    top = max(tiers, key=lambda t: t.rpi_score)
    assert top.tier == 1
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/data/test_conference_rpi.py -v
```

**Step 3: Implement conference_rpi.py**

```python
# src/data/conference_rpi.py
"""Conference RPI computation for heterogeneous graph Conference nodes.

RPI formula (simplified):
    rpi_score = 0.6 * mean(adj_em) + 0.4 * mean(sos)

Higher is stronger. Tier 1 = top 20% of conferences.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ConferenceRPI:
    conference: str
    mean_adj_em: float
    mean_sos: float
    rpi_score: float
    tier: int = 0   # assigned separately by assign_rpi_tiers()
    n_teams: int = 0


def compute_conference_rpi(teams_df: pd.DataFrame) -> list[ConferenceRPI]:
    """Compute per-conference RPI from a T-Rank DataFrame.

    Args:
        teams_df: DataFrame with columns ['conference', 'adj_em', 'sos'].

    Returns:
        List of ConferenceRPI sorted by rpi_score descending.
    """
    required = {"conference", "adj_em", "sos"}
    missing = required - set(teams_df.columns)
    if missing:
        raise ValueError(f"teams_df missing columns: {missing}")

    records: list[ConferenceRPI] = []
    for conf, group in teams_df.groupby("conference"):
        mean_em = float(group["adj_em"].mean())
        mean_sos = float(group["sos"].mean())
        rpi = 0.6 * mean_em + 0.4 * mean_sos
        records.append(
            ConferenceRPI(
                conference=str(conf),
                mean_adj_em=round(mean_em, 3),
                mean_sos=round(mean_sos, 3),
                rpi_score=round(rpi, 3),
                n_teams=len(group),
            )
        )
    return sorted(records, key=lambda r: r.rpi_score, reverse=True)


def assign_rpi_tiers(rpis: list[ConferenceRPI]) -> list[ConferenceRPI]:
    """Assign tier 1–5 (1 = strongest) via quintile cut.

    Mutates each ConferenceRPI.tier in place and returns the list.
    """
    n = len(rpis)
    if n == 0:
        return rpis
    scores = np.array([r.rpi_score for r in rpis])
    quintiles = np.percentile(scores, [80, 60, 40, 20])
    for rpi_obj in rpis:
        s = rpi_obj.rpi_score
        if s >= quintiles[0]:
            rpi_obj.tier = 1
        elif s >= quintiles[1]:
            rpi_obj.tier = 2
        elif s >= quintiles[2]:
            rpi_obj.tier = 3
        elif s >= quintiles[3]:
            rpi_obj.tier = 4
        else:
            rpi_obj.tier = 5
    return rpis
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/data/test_conference_rpi.py -v
```

**Step 5: Commit**

```bash
git add src/data/conference_rpi.py tests/data/test_conference_rpi.py
git commit -m "feat(data): ConferenceRPI — RPI score + 5-tier assignment for GAT Conference nodes"
```

---

### Task 6: Bayesian Head — Temporal Warm-Start

**Files:**
- Modify: `src/model/bayesian_head.py`
- Modify: `tests/model/test_bayesian_head.py`

**Step 1: Write failing tests**

Add these tests to `tests/model/test_bayesian_head.py`:

```python
def test_build_model_accepts_warm_start_posterior():
    """build_model should not raise when warm_start_posterior is provided."""
    from src.model.bayesian_head import build_model
    import numpy as np
    n_games = 4
    home = np.array([0.6, 0.4, 0.55, 0.45])
    away = np.array([0.4, 0.6, 0.45, 0.55])
    obs_win = np.array([1, 0, 1, 0])
    obs_spread = np.array([5.0, -3.0, 2.0, -1.0])
    home_coach = np.array([0, 1, 0, 1])
    away_coach = np.array([1, 0, 1, 0])
    n_coaches = 2
    warm_start = {
        "alpha_mu": 0.02,
        "alpha_sigma": 0.05,
        "beta_spread_mu": 0.8,
        "beta_spread_sigma": 0.1,
    }
    # Should not raise
    model = build_model(
        home_strength=home,
        away_strength=away,
        obs_win=obs_win,
        obs_spread=obs_spread,
        home_coach=home_coach,
        away_coach=away_coach,
        n_coaches=n_coaches,
        warm_start_posterior=warm_start,
    )
    assert model is not None


def test_build_model_warm_start_none_is_default():
    """warm_start_posterior=None should behave identically to the original call."""
    from src.model.bayesian_head import build_model
    import numpy as np
    n_games = 2
    home = np.array([0.6, 0.4])
    away = np.array([0.4, 0.6])
    obs_win = np.array([1, 0])
    obs_spread = np.array([5.0, -3.0])
    home_coach = np.array([0, 1])
    away_coach = np.array([1, 0])
    model = build_model(
        home_strength=home,
        away_strength=away,
        obs_win=obs_win,
        obs_spread=obs_spread,
        home_coach=home_coach,
        away_coach=away_coach,
        n_coaches=2,
        warm_start_posterior=None,
    )
    assert model is not None
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/model/test_bayesian_head.py -v -k "warm_start"
```

**Step 3: Modify build_model to accept warm_start_posterior**

Read `src/model/bayesian_head.py` and add the `warm_start_posterior: dict | None = None` parameter. When provided, use the dict values as prior means instead of the hardcoded defaults:

```python
def build_model(
    home_strength,
    away_strength,
    obs_win,
    obs_spread,
    home_coach,
    away_coach,
    n_coaches: int,
    warm_start_posterior: dict | None = None,
) -> pm.Model:
    """Build PyMC model for win probability and spread.

    Args:
        warm_start_posterior: Optional dict with keys like
            {'alpha_mu': float, 'alpha_sigma': float,
             'beta_spread_mu': float, 'beta_spread_sigma': float}.
            When provided (12 PM / 10 PM runs), uses morning posteriors
            as informative priors — 10x faster, statistically correct
            sequential Bayesian updating.
    """
    ws = warm_start_posterior or {}
    alpha_mu_prior    = ws.get("alpha_mu", 0.0)
    alpha_sigma_prior = ws.get("alpha_sigma", 1.0)
    beta_mu_prior     = ws.get("beta_spread_mu", 0.5)
    beta_sigma_prior  = ws.get("beta_spread_sigma", 0.5)

    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=alpha_mu_prior, sigma=alpha_sigma_prior)
        # ... rest of existing model unchanged, substitute alpha_mu_prior etc.
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/model/test_bayesian_head.py -v
```

**Step 5: Commit**

```bash
git add src/model/bayesian_head.py tests/model/test_bayesian_head.py
git commit -m "feat(model): bayesian_head warm_start_posterior — sequential Bayesian updating for 12PM/10PM runs"
```

---

## Phase 3 — GitHub Actions Orchestration

---

### Task 7: Pipeline Runner

**Files:**
- Create: `src/pipeline/github_actions_runner.py`
- Create: `tests/pipeline/test_runner.py`

**Step 1: Write failing tests**

```python
# tests/pipeline/test_runner.py
from unittest.mock import MagicMock, patch
import pytest
from src.pipeline.github_actions_runner import PipelineRunner, RunProfile


def test_run_profile_enum():
    assert RunProfile.FULL.value == "full"
    assert RunProfile.INTEL.value == "intel"
    assert RunProfile.RESULTS.value == "results"


def test_pipeline_runner_init():
    runner = PipelineRunner(season=2026, dry_run=True)
    assert runner.season == 2026
    assert runner.dry_run is True


def test_run_full_calls_writer_in_dry_run():
    runner = PipelineRunner(season=2026, dry_run=True)
    # dry_run=True skips actual scraping but exercises the runner logic
    result = runner.run(RunProfile.FULL)
    assert result["trigger"] == "full"
    assert "duration_secs" in result


def test_run_intel_profile():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.INTEL)
    assert result["trigger"] == "intel"


def test_run_results_profile():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.RESULTS)
    assert result["trigger"] == "results"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/pipeline/test_runner.py -v
```

**Step 3: Implement PipelineRunner**

```python
# src/pipeline/github_actions_runner.py
"""Orchestrates 3x-daily pipeline runs triggered by GitHub Actions cron.

Run profiles:
  FULL    — 6 AM ET: scrape all sources → full ST-GNN → Bayesian → MC 10k → write Supabase
  INTEL   — 12 PM ET: intel refresh + warm-start Bayesian update
  RESULTS — 10 PM ET: ingest game results + Brier recalibration
"""
from __future__ import annotations

import enum
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class RunProfile(enum.Enum):
    FULL    = "full"
    INTEL   = "intel"
    RESULTS = "results"


class PipelineRunner:
    """Orchestrates a single cron run for a given profile."""

    def __init__(self, season: int = 2026, dry_run: bool = False) -> None:
        self.season = season
        self.dry_run = dry_run
        self._writer = None if dry_run else self._init_writer()

    def _init_writer(self):
        from src.pipeline.supabase_writer import SupabaseWriter
        return SupabaseWriter()

    def run(self, profile: RunProfile) -> dict[str, Any]:
        """Execute the given profile. Returns a summary dict."""
        t0 = time.time()
        logger.info("Starting pipeline run: profile=%s season=%s dry_run=%s",
                    profile.value, self.season, self.dry_run)
        try:
            if profile == RunProfile.FULL:
                summary = self._run_full()
            elif profile == RunProfile.INTEL:
                summary = self._run_intel()
            elif profile == RunProfile.RESULTS:
                summary = self._run_results()
            else:
                raise ValueError(f"Unknown profile: {profile}")
            summary["status"] = "success"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline run failed: %s", exc)
            summary = {"teams_updated": 0, "predictions_computed": 0,
                       "alerts_found": 0, "status": "failed",
                       "error_log": str(exc)}
        summary["trigger"] = profile.value
        summary["duration_secs"] = round(time.time() - t0, 2)
        if not self.dry_run and self._writer:
            self._writer.log_pipeline_run(**summary)
        return summary

    def _run_full(self) -> dict[str, Any]:
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}
        # 1. Scrape T-Rank + Sports Reference + Kaggle + Rotowire + NBA + On3
        from src.data.barttorvik import fetch_trank
        teams_df = fetch_trank(self.season)
        # 2. Conference RPI
        from src.data.conference_rpi import compute_conference_rpi, assign_rpi_tiers
        rpis = assign_rpi_tiers(compute_conference_rpi(teams_df))
        # 3. Write teams to Supabase
        import datetime
        team_rows = teams_df.to_dict(orient="records")
        for r in team_rows:
            r["season"] = self.season
            r["scraped_at"] = datetime.datetime.utcnow().isoformat()
        self._writer.upsert_teams(team_rows)
        return {
            "teams_updated": len(team_rows),
            "predictions_computed": 0,
            "alerts_found": 0,
        }

    def _run_intel(self) -> dict[str, Any]:
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}
        from src.data.news_scraper import scrape_intel_alerts
        alerts = scrape_intel_alerts()
        for alert in alerts:
            self._writer.insert_intel_alert(alert)
        return {"teams_updated": 0, "predictions_computed": 0,
                "alerts_found": len(alerts)}

    def _run_results(self) -> dict[str, Any]:
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}
        from src.data.kaggle_ingestion import load_recent_results
        results = load_recent_results(self.season)
        self._writer.upsert_game_results(results)
        return {"teams_updated": 0, "predictions_computed": len(results),
                "alerts_found": 0}
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/pipeline/test_runner.py -v
```

**Step 5: Commit**

```bash
git add src/pipeline/github_actions_runner.py tests/pipeline/test_runner.py
git commit -m "feat(pipeline): PipelineRunner — full/intel/results run profiles for GitHub Actions"
```

---

### Task 8: GitHub Actions Workflows

**Files:**
- Create: `.github/workflows/daily_pipeline.yml`
- Create: `.github/workflows/pr_tests.yml`

**Step 1: Write daily_pipeline.yml**

```yaml
# .github/workflows/daily_pipeline.yml
name: Madness Matrix — Daily Pipeline

on:
  schedule:
    - cron: '0 11 * * *'   # 6 AM ET — full run
    - cron: '0 17 * * *'   # 12 PM ET — intel run
    - cron: '0 3 * * *'    # 10 PM ET — results run
  workflow_dispatch:
    inputs:
      profile:
        description: 'Run profile (full / intel / results)'
        required: true
        default: 'full'
        type: choice
        options: [full, intel, results]

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Determine run profile
        id: profile
        run: |
          HOUR=$(date -u +%H)
          if [ "${{ github.event_name }}" = "workflow_dispatch" ]; then
            echo "profile=${{ github.event.inputs.profile }}" >> $GITHUB_OUTPUT
          elif [ "$HOUR" = "11" ]; then
            echo "profile=full" >> $GITHUB_OUTPUT
          elif [ "$HOUR" = "17" ]; then
            echo "profile=intel" >> $GITHUB_OUTPUT
          else
            echo "profile=results" >> $GITHUB_OUTPUT
          fi

      - name: Run pipeline
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_SERVICE_KEY: ${{ secrets.SUPABASE_SERVICE_KEY }}
          KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
          KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          WANDB_PROJECT: ${{ secrets.WANDB_PROJECT }}
          WANDB_ENTITY: ${{ secrets.WANDB_ENTITY }}
        run: |
          python -c "
          from src.pipeline.github_actions_runner import PipelineRunner, RunProfile
          profile_map = {'full': RunProfile.FULL, 'intel': RunProfile.INTEL, 'results': RunProfile.RESULTS}
          runner = PipelineRunner(season=2026)
          result = runner.run(profile_map['${{ steps.profile.outputs.profile }}'])
          print(result)
          "
```

**Step 2: Write pr_tests.yml**

```yaml
# .github/workflows/pr_tests.yml
name: PR Tests

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run pytest
        run: python -m pytest tests/ -q --tb=short

      - name: Run frontend tests
        working-directory: frontend
        run: |
          npm ci
          npx vitest run --reporter=dot
```

**Step 3: Commit**

```bash
git add .github/
git commit -m "feat(ci): daily_pipeline.yml (3x cron) + pr_tests.yml (pytest + vitest)"
```

---

## Phase 4 — Frontend Migration

---

### Task 9: Tailwind + shadcn + React Query + Supabase Client

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/lib/supabase.ts`
- Create: `frontend/lib/queries.ts`
- Create/modify: `frontend/tailwind.config.ts`

**Step 1: Install frontend dependencies**

```bash
cd frontend
npm install @supabase/supabase-js @tanstack/react-query recharts framer-motion date-fns zod
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npx shadcn@latest init --defaults
```

**Step 2: Create Supabase client singleton**

```typescript
// frontend/lib/supabase.ts
import { createClient } from '@supabase/supabase-js'

const supabaseUrl  = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnon = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnon)
```

**Step 3: Create React Query hooks**

```typescript
// frontend/lib/queries.ts
import { useQuery } from '@tanstack/react-query'
import { supabase } from './supabase'

export function useTeams(season: number) {
  return useQuery({
    queryKey: ['teams', season],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('teams')
        .select('*')
        .eq('season', season)
        .order('adj_em', { ascending: false })
      if (error) throw error
      return data ?? []
    },
  })
}

export function usePredictionSnapshots(season: number) {
  return useQuery({
    queryKey: ['snapshots', season],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('prediction_snapshots')
        .select('*')
        .eq('season', season)
        .order('championship_prob', { ascending: false })
      if (error) throw error
      return data ?? []
    },
  })
}

export function useLatestBracketRun(season: number) {
  return useQuery({
    queryKey: ['bracket_run', season],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('bracket_runs')
        .select('*')
        .eq('season', season)
        .order('computed_at', { ascending: false })
        .limit(1)
        .single()
      if (error) throw error
      return data
    },
  })
}

export function useIntelAlerts() {
  return useQuery({
    queryKey: ['intel_alerts'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('intel_alerts')
        .select('*')
        .eq('resolved', false)
        .order('alerted_at', { ascending: false })
        .limit(50)
      if (error) throw error
      return data ?? []
    },
    refetchInterval: 60_000,   // poll every 60s as fallback
  })
}

export function useMatchupPrediction(season: number, teamA: string, teamB: string) {
  return useQuery({
    queryKey: ['matchup', season, teamA, teamB],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('matchup_predictions')
        .select('*')
        .eq('season', season)
        .or(`and(team_a.eq.${teamA},team_b.eq.${teamB}),and(team_a.eq.${teamB},team_b.eq.${teamA})`)
        .order('computed_at', { ascending: false })
        .limit(1)
        .single()
      if (error) throw error
      return data
    },
    enabled: !!teamA && !!teamB,
  })
}
```

**Step 4: Configure Tailwind for Madness Matrix palette**

```typescript
// frontend/tailwind.config.ts
import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./app/**/*.{ts,tsx}', './components/**/*.{ts,tsx}', './lib/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        navy:  { DEFAULT: '#0a0f1e', 800: '#0d1429', 700: '#111b3a' },
        blue:  { DEFAULT: '#3b82f6', 600: '#2563eb' },
        amber: { DEFAULT: '#f59e0b', 600: '#d97706' },
        alert: { DEFAULT: '#ef4444' },
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
export default config
```

**Step 5: Commit**

```bash
git add frontend/lib/ frontend/tailwind.config.ts frontend/package.json
git commit -m "feat(frontend): Tailwind + shadcn + React Query + Supabase client singleton"
```

---

### Task 10: Rebrand — Remove Graph Tab, Add 5-Tab Navigation

**Files:**
- Modify: `frontend/app/layout.tsx` or root layout
- Modify: `frontend/app/page.tsx` or top-level nav component
- Delete: `frontend/components/constellation/` (entire directory)
- Modify: `frontend/components/` nav to use 5-tab structure

**Step 1: Remove constellation components**

```bash
rm -rf frontend/components/constellation/
```

**Step 2: Define new NavPage type and tab config**

```typescript
// In frontend/app/page.tsx or a new frontend/lib/nav.ts
export type NavPage = 'live' | 'rankings' | 'matchup' | 'bracket' | 'warroom'

export const NAV_TABS: { id: NavPage; label: string; emoji: string }[] = [
  { id: 'live',     label: 'Live 2026',  emoji: '🔴' },
  { id: 'rankings', label: 'Rankings',   emoji: '📊' },
  { id: 'matchup',  label: 'Matchup',    emoji: '⚔' },
  { id: 'bracket',  label: 'Bracket',    emoji: '🏆' },
  { id: 'warroom',  label: 'War Room',   emoji: '🎯' },
]
```

**Step 3: Add season-aware global state**

```typescript
// In frontend/app/page.tsx
const [season, setSeason] = useState<number>(2026)
const [page, setPage] = useState<NavPage>('live')
// Pass season down to all tab components so they all re-query when season changes
```

**Step 4: Update page title and branding**

In `frontend/app/layout.tsx`:
```typescript
export const metadata = {
  title: 'Madness Matrix',
  description: 'Quantitative bracket forecasting powered by ST-GNN + Bayesian inference',
}
```

**Step 5: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): rebrand to Madness Matrix, 5-tab nav, remove constellation graph tab"
```

---

## Phase 5 — Frontend Pages

---

### Task 11: Live Tab — Descriptive View

**Files:**
- Create: `frontend/components/live/LiveTab.tsx`
- Create: `frontend/components/live/DescriptiveView.tsx`

**Step 1: Create DescriptiveView**

```tsx
// frontend/components/live/DescriptiveView.tsx
'use client'
import { useTeams, useIntelAlerts } from '@/lib/queries'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface Props { season: number }

export function DescriptiveView({ season }: Props) {
  const { data: teams = [], isLoading } = useTeams(season)
  const { data: alerts = [] } = useIntelAlerts()

  if (isLoading) return <div className="text-blue-400 font-mono">Loading teams...</div>

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Efficiency Leaderboard */}
      <Card className="lg:col-span-2 bg-navy-800 border-blue-900">
        <CardHeader>
          <CardTitle className="text-blue-400 font-mono text-sm">
            EFFICIENCY LEADERBOARD — {season}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-auto max-h-96">
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="text-amber-500 border-b border-navy-700">
                  <th className="text-left py-1">Team</th>
                  <th className="text-right">AdjO</th>
                  <th className="text-right">AdjD</th>
                  <th className="text-right">AdjEM</th>
                </tr>
              </thead>
              <tbody>
                {teams.slice(0, 25).map((t: any) => (
                  <tr key={t.name} className="border-b border-navy-700/50 hover:bg-navy-700/30">
                    <td className="py-1 text-white">{t.name}</td>
                    <td className="text-right text-green-400">{t.adj_oe?.toFixed(1)}</td>
                    <td className="text-right text-red-400">{t.adj_de?.toFixed(1)}</td>
                    <td className="text-right text-blue-300 font-bold">{t.adj_em?.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Intel Alert Feed */}
      <Card className="bg-navy-800 border-amber-900/50">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono text-sm">
            INTEL ALERTS
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {alerts.length === 0 && (
            <p className="text-gray-500 text-sm">No active alerts</p>
          )}
          {alerts.map((alert: any) => (
            <div key={alert.alert_id}
              className="border border-amber-900/30 rounded p-2 text-xs">
              <div className="flex items-center gap-2 mb-1">
                <Badge variant={alert.severity === 'critical' ? 'destructive' : 'secondary'}
                  className="text-xs">{alert.severity}</Badge>
                {alert.needs_verification && (
                  <Badge variant="outline" className="text-xs text-amber-500">UNVERIFIED</Badge>
                )}
                <span className="text-gray-400">{alert.source}</span>
              </div>
              <p className="text-gray-300">{alert.content}</p>
              <p className="text-amber-600 mt-1">
                {alert.teams_mentioned?.join(', ')}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}
```

**Step 2: Create LiveTab with sub-view switcher**

```tsx
// frontend/components/live/LiveTab.tsx
'use client'
import { useState } from 'react'
import { DescriptiveView } from './DescriptiveView'

type SubView = 'descriptive' | 'predictive' | 'prescriptive'
interface Props { season: number }

export function LiveTab({ season }: Props) {
  const [subView, setSubView] = useState<SubView>('descriptive')
  const tabs: { id: SubView; label: string }[] = [
    { id: 'descriptive',  label: 'Descriptive' },
    { id: 'predictive',   label: 'Predictive' },
    { id: 'prescriptive', label: 'Prescriptive' },
  ]

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {tabs.map(t => (
          <button key={t.id}
            onClick={() => setSubView(t.id)}
            className={`px-4 py-1.5 rounded font-mono text-sm transition-colors ${
              subView === t.id
                ? 'bg-blue-600 text-white'
                : 'bg-navy-700 text-gray-400 hover:text-white'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      {subView === 'descriptive'  && <DescriptiveView season={season} />}
      {subView === 'predictive'   && <div className="text-gray-400 font-mono">Predictive view — Task 12</div>}
      {subView === 'prescriptive' && <div className="text-gray-400 font-mono">Prescriptive view — Task 13</div>}
    </div>
  )
}
```

**Step 3: Commit**

```bash
git add frontend/components/live/
git commit -m "feat(frontend): LiveTab + DescriptiveView — efficiency leaderboard + intel feed"
```

---

### Task 12: Predictive View — Calibration Curve + Probability Drift

**Files:**
- Create: `frontend/components/live/PredictiveView.tsx`
- Create: `frontend/components/live/CalibrationCurve.tsx`
- Create: `frontend/components/live/ProbabilityDrift.tsx`

**Step 1: Create CalibrationCurve**

```tsx
// frontend/components/live/CalibrationCurve.tsx
'use client'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface CalPoint { bin: number; raw: number; calibrated: number; perfect: number }
interface Props { data: CalPoint[] }

export function CalibrationCurve({ data }: Props) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <XAxis dataKey="bin" tickFormatter={v => `${(v*100).toFixed(0)}%`}
          stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <YAxis tickFormatter={v => `${(v*100).toFixed(0)}%`}
          stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => `${(v*100).toFixed(1)}%`}
          contentStyle={{ background: '#0d1429', border: '1px solid #1e3a5f' }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="perfect" stroke="#4b5563" strokeDasharray="4 4"
          dot={false} name="Perfect" />
        <Line type="monotone" dataKey="raw" stroke="#f59e0b" dot={false} name="Raw Posterior" />
        <Line type="monotone" dataKey="calibrated" stroke="#3b82f6" dot={false}
          strokeWidth={2} name="Isotonic Calibrated" />
      </LineChart>
    </ResponsiveContainer>
  )
}
```

**Step 2: Create ProbabilityDrift**

```tsx
// frontend/components/live/ProbabilityDrift.tsx
'use client'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface DriftPoint { run: string; championship_prob: number; team: string }
interface Props { data: DriftPoint[]; teams: string[] }

export function ProbabilityDrift({ data, teams }: Props) {
  const COLORS = ['#3b82f6', '#f59e0b', '#ef4444', '#10b981', '#8b5cf6']
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <XAxis dataKey="run" stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <YAxis tickFormatter={v => `${(v*100).toFixed(1)}%`}
          stroke="#4b5563" tick={{ fill: '#9ca3af', fontSize: 11 }} />
        <Tooltip formatter={(v: number) => `${(v*100).toFixed(2)}%`}
          contentStyle={{ background: '#0d1429', border: '1px solid #1e3a5f' }} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {teams.map((team, i) => (
          <Line key={team} type="monotone" dataKey={team}
            stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
```

**Step 3: Create PredictiveView**

```tsx
// frontend/components/live/PredictiveView.tsx
'use client'
import { usePredictionSnapshots } from '@/lib/queries'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CalibrationCurve } from './CalibrationCurve'
import { ProbabilityDrift } from './ProbabilityDrift'

interface Props { season: number }

export function PredictiveView({ season }: Props) {
  const { data: snapshots = [], isLoading } = usePredictionSnapshots(season)

  const top5 = snapshots.slice(0, 5)

  // Build drift data from snapshots (grouped by date)
  const driftData = Object.entries(
    snapshots.reduce((acc: any, s: any) => {
      const key = s.snapshot_date
      if (!acc[key]) acc[key] = { run: key }
      acc[key][s.team] = s.championship_prob
      return acc
    }, {})
  ).map(([, v]) => v) as any[]

  return (
    <div className="space-y-4">
      {/* Championship Probabilities */}
      <Card className="bg-navy-800 border-blue-900">
        <CardHeader>
          <CardTitle className="text-blue-400 font-mono text-sm">
            CHAMPIONSHIP PROBABILITY — TOP CONTENDERS
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading
            ? <p className="text-gray-500 font-mono">Loading...</p>
            : top5.map((s: any) => (
              <div key={s.team} className="flex items-center justify-between py-1.5 border-b border-navy-700/50">
                <span className="text-white font-mono text-sm">{s.team}</span>
                <div className="flex gap-4 text-xs font-mono">
                  <span className="text-amber-400">
                    🏆 {((s.championship_prob ?? 0) * 100).toFixed(1)}%
                  </span>
                  <span className="text-blue-300">
                    F4 {((s.final_four_prob ?? 0) * 100).toFixed(1)}%
                  </span>
                  <span className="text-green-400">
                    E8 {((s.elite_eight_prob ?? 0) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
        </CardContent>
      </Card>

      {/* Probability Drift */}
      {driftData.length > 1 && (
        <Card className="bg-navy-800 border-blue-900">
          <CardHeader>
            <CardTitle className="text-blue-400 font-mono text-sm">
              PROBABILITY DRIFT — TEMPORAL BAYESIAN UPDATES
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ProbabilityDrift
              data={driftData}
              teams={top5.map((s: any) => s.team)}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
```

**Step 4: Wire PredictiveView into LiveTab**

In `frontend/components/live/LiveTab.tsx`, replace the placeholder:
```tsx
import { PredictiveView } from './PredictiveView'
// ...
{subView === 'predictive' && <PredictiveView season={season} />}
```

**Step 5: Commit**

```bash
git add frontend/components/live/
git commit -m "feat(frontend): PredictiveView — championship probs, probability drift, calibration curve"
```

---

### Task 13: Prescriptive View + Intel Feed with Supabase Realtime

**Files:**
- Create: `frontend/components/live/PrescriptiveView.tsx`
- Create: `frontend/components/intel/IntelFeed.tsx`

**Step 1: Create IntelFeed with Realtime subscription**

```tsx
// frontend/components/intel/IntelFeed.tsx
'use client'
import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { Badge } from '@/components/ui/badge'

interface Alert {
  alert_id: string
  source: string
  content: string
  severity: string
  teams_mentioned: string[]
  needs_verification: boolean
  alerted_at: string
}

interface Props { initialAlerts?: Alert[] }

export function IntelFeed({ initialAlerts = [] }: Props) {
  const [alerts, setAlerts] = useState<Alert[]>(initialAlerts)

  useEffect(() => {
    // Initial fetch
    supabase
      .from('intel_alerts')
      .select('*')
      .eq('resolved', false)
      .order('alerted_at', { ascending: false })
      .limit(30)
      .then(({ data }) => { if (data) setAlerts(data) })

    // Realtime subscription
    const channel = supabase
      .channel('intel-realtime')
      .on(
        'postgres_changes',
        { event: 'INSERT', schema: 'public', table: 'intel_alerts' },
        payload => setAlerts(prev => [payload.new as Alert, ...prev.slice(0, 29)])
      )
      .subscribe()

    return () => { supabase.removeChannel(channel) }
  }, [])

  const severityColor: Record<string, string> = {
    critical: 'bg-red-900/50 border-red-500/30',
    high:     'bg-amber-900/30 border-amber-500/30',
    medium:   'bg-blue-900/20 border-blue-500/20',
    low:      'bg-gray-900/20 border-gray-600/20',
  }

  return (
    <div className="space-y-2 max-h-[500px] overflow-auto">
      {alerts.length === 0 && (
        <p className="text-gray-500 text-sm font-mono">No active intel alerts.</p>
      )}
      {alerts.map(alert => (
        <div key={alert.alert_id}
          className={`rounded border p-3 text-xs ${severityColor[alert.severity] ?? ''}`}>
          <div className="flex items-center gap-2 mb-1.5">
            <Badge className="text-xs capitalize">{alert.severity}</Badge>
            {alert.needs_verification && (
              <Badge variant="outline" className="text-xs text-amber-400 border-amber-600">
                UNVERIFIED
              </Badge>
            )}
            <span className="text-gray-400 ml-auto">{alert.source}</span>
          </div>
          <p className="text-gray-200 leading-relaxed">{alert.content}</p>
          {alert.teams_mentioned?.length > 0 && (
            <p className="text-amber-500 mt-1.5 font-mono">
              Teams: {alert.teams_mentioned.join(', ')}
            </p>
          )}
          <p className="text-gray-600 mt-1">
            {new Date(alert.alerted_at).toLocaleString()}
          </p>
        </div>
      ))}
    </div>
  )
}
```

**Step 2: Create PrescriptiveView**

```tsx
// frontend/components/live/PrescriptiveView.tsx
'use client'
import { usePredictionSnapshots, useLatestBracketRun } from '@/lib/queries'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { IntelFeed } from '@/components/intel/IntelFeed'

interface Props { season: number }

export function PrescriptiveView({ season }: Props) {
  const { data: snapshots = [] } = usePredictionSnapshots(season)
  const { data: bracketRun } = useLatestBracketRun(season)

  const topPicks = snapshots
    .filter((s: any) => s.championship_prob > 0.05)
    .slice(0, 10)

  const variants = ['Chalk', 'Leverage', 'Chaos']

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* CLV Top Picks */}
      <Card className="bg-navy-800 border-blue-900">
        <CardHeader>
          <CardTitle className="text-blue-400 font-mono text-sm">
            TOP VALUE PICKS — PROSPECT THEORY CLV
          </CardTitle>
        </CardHeader>
        <CardContent>
          {topPicks.map((s: any, i: number) => (
            <div key={s.team}
              className="flex items-center justify-between py-1.5 border-b border-navy-700/50">
              <span className="text-gray-400 font-mono text-xs w-6">{i + 1}.</span>
              <span className="text-white font-mono text-sm flex-1">{s.team}</span>
              <span className="text-amber-400 font-mono text-xs">
                {((s.championship_prob ?? 0) * 100).toFixed(2)}%
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* RL Bracket Variants */}
      <Card className="bg-navy-800 border-blue-900">
        <CardHeader>
          <CardTitle className="text-blue-400 font-mono text-sm">
            BRACKET STRATEGY VARIANTS
          </CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-3 gap-2">
          {variants.map(v => (
            <div key={v}
              className="border border-blue-900/50 rounded p-3 text-center">
              <p className="text-amber-400 font-mono text-xs font-bold mb-1">{v}</p>
              <p className="text-gray-400 text-xs">
                {v === 'Chalk' && 'Favor favorites. Optimal for small pools.'}
                {v === 'Leverage' && 'Fade public chalk. Optimal for mid-size pools.'}
                {v === 'Chaos' && 'Max upsets. Optimal for large contests.'}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Intel Feed */}
      <Card className="lg:col-span-2 bg-navy-800 border-amber-900/50">
        <CardHeader>
          <CardTitle className="text-amber-500 font-mono text-sm">
            LIVE INTEL FEED — REALTIME
          </CardTitle>
        </CardHeader>
        <CardContent>
          <IntelFeed />
        </CardContent>
      </Card>
    </div>
  )
}
```

**Step 3: Wire PrescriptiveView into LiveTab**

```tsx
import { PrescriptiveView } from './PrescriptiveView'
// ...
{subView === 'prescriptive' && <PrescriptiveView season={season} />}
```

**Step 4: Commit**

```bash
git add frontend/components/live/PrescriptiveView.tsx frontend/components/intel/
git commit -m "feat(frontend): PrescriptiveView + IntelFeed with Supabase Realtime subscription"
```

---

### Task 14: Season-Aware State + Matchup Oracle from Supabase

**Files:**
- Modify: `frontend/app/page.tsx`
- Modify: `frontend/components/matchup/` (existing Matchup Oracle component)

**Step 1: Wire season prop throughout page.tsx**

Read `frontend/app/page.tsx` and ensure:
1. `season` state is at the top level
2. Season selector dropdown passes `setSeason`
3. All 5 tab components receive `season` prop
4. When season < 2026, show a banner: "Viewing historical season — actual results are overlaid"

```tsx
// In page.tsx top-level state
const [season, setSeason] = useState<number>(2026)

// Season selector component
<select
  value={season}
  onChange={e => setSeason(Number(e.target.value))}
  className="bg-navy-800 text-white font-mono text-sm px-3 py-1.5 rounded border border-blue-900"
>
  {Array.from({ length: 15 }, (_, i) => 2026 - i).map(y => (
    <option key={y} value={y}>{y}</option>
  ))}
</select>

{season < 2026 && (
  <div className="bg-amber-900/20 border border-amber-700/30 rounded px-3 py-1.5 text-amber-400 font-mono text-xs">
    Historical season — actual results overlaid on predictions
  </div>
)}
```

**Step 2: Update Matchup Oracle to read from Supabase**

Find the existing matchup component and replace mock data with:

```tsx
import { useMatchupPrediction, useTeams } from '@/lib/queries'

// In MatchupOracle component:
const { data: matchup, isLoading } = useMatchupPrediction(season, teamA, teamB)

// Use matchup.p_win_a, matchup.spread_mean, matchup.spread_std
// Parse matchup.samples_json for KDE plot
```

**Step 3: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): season-aware global state + Matchup Oracle reads Supabase posteriors"
```

---

## Phase 6 — Deployment

---

### Task 15: Vercel Config + Environment Setup

**Files:**
- Create: `frontend/vercel.json`
- Create: `.env.example`

**Step 1: Create vercel.json**

```json
{
  "framework": "nextjs",
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "env": {
    "NEXT_PUBLIC_SUPABASE_URL": "@supabase_url",
    "NEXT_PUBLIC_SUPABASE_ANON_KEY": "@supabase_anon_key"
  },
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "no-store" }
      ]
    }
  ]
}
```

**Step 2: Create .env.example**

```bash
# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key          # server-side only (GitHub Actions)
SUPABASE_ANON_KEY=your-anon-key                     # frontend-safe

# Next.js public vars (must be prefixed NEXT_PUBLIC_)
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key

# Kaggle (GitHub Actions pipeline only)
KAGGLE_USERNAME=your-kaggle-username
KAGGLE_KEY=your-kaggle-api-key

# W&B Experiment Tracking (GitHub Actions pipeline only)
WANDB_API_KEY=your-wandb-api-key
WANDB_PROJECT=madness-matrix
WANDB_ENTITY=your-wandb-entity
```

**Step 3: Run all tests**

```bash
python -m pytest tests/ -q
cd frontend && npx vitest run --reporter=dot
```

Expected: all tests pass.

**Step 4: Deploy to Vercel**

```bash
cd frontend
npx vercel --prod
```

Add environment variables in Vercel dashboard:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

**Step 5: Add GitHub Actions secrets**

In `github.com/ritvikv03/MM/settings/secrets/actions`, add:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`
- `SUPABASE_ANON_KEY`
- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_ENTITY`

**Step 6: Final commit**

```bash
git add frontend/vercel.json .env.example
git commit -m "feat(deploy): vercel.json config + .env.example with all required secrets"
```

---

## Verification Checklist

After all 15 tasks:

- [ ] `python -m pytest tests/ -q` — all tests pass
- [ ] `cd frontend && npx vitest run` — all TS tests pass
- [ ] `supabase/migrations/001_initial_schema.sql` applied in Supabase SQL editor
- [ ] `supabase/migrations/002_indexes.sql` applied in Supabase SQL editor
- [ ] Supabase Realtime enabled on `intel_alerts` and `prediction_snapshots`
- [ ] `.github/workflows/daily_pipeline.yml` triggers on schedule
- [ ] `.github/workflows/pr_tests.yml` triggers on PR
- [ ] GitHub Actions secrets added for all 8 env vars
- [ ] Vercel deployment live at production URL
- [ ] Season selector switches all 5 tabs simultaneously
- [ ] Intel feed updates in realtime without page refresh
- [ ] No synthetic data remains — `mock-data.ts` deleted, no `_build_stub_*` functions
- [ ] `IsotonicCalibrator` persists via joblib (no pickle)
- [ ] `PipelineRunner(dry_run=True)` completes without network calls
