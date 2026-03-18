-- 001_initial_schema.sql
-- Initial schema for the Madness Matrix NCAA March Madness prediction engine.
-- Creates 7 core tables. Run once against a fresh Supabase project.

-- ---------------------------------------------------------------------------
-- 1. teams
--    One row per team per season. Stores Barttorvik efficiency metrics,
--    conference, seed, coach, and region (populated after bracket release).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    id          BIGSERIAL PRIMARY KEY,
    season      INT          NOT NULL,
    name        TEXT         NOT NULL,
    conference  TEXT,
    seed        INT,
    adj_oe      NUMERIC,
    adj_de      NUMERIC,
    adj_em      NUMERIC,
    tempo       NUMERIC,
    luck        NUMERIC,
    sos         NUMERIC,
    coach       TEXT,
    region      TEXT,
    scraped_at  TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (season, name)
);

-- ---------------------------------------------------------------------------
-- 2. matchup_predictions
--    Stores the Bayesian posterior outputs for a specific head-to-head matchup.
--    samples_json holds raw posterior samples for downstream Monte Carlo use.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matchup_predictions (
    id            BIGSERIAL PRIMARY KEY,
    season        INT          NOT NULL,
    team_a        TEXT         NOT NULL,
    team_b        TEXT         NOT NULL,
    p_win_a       NUMERIC      NOT NULL,
    spread_mean   NUMERIC,
    spread_std    NUMERIC,
    samples_json  JSONB,
    model_version TEXT,
    computed_at   TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (season, team_a, team_b, computed_at)
);

-- ---------------------------------------------------------------------------
-- 3. bracket_runs
--    One row per full Monte Carlo bracket simulation run.
--    advancement_probs and champion_prob are JSONB keyed by team name.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bracket_runs (
    id                  BIGSERIAL PRIMARY KEY,
    season              INT          NOT NULL,
    run_date            DATE         NOT NULL,
    n_simulations       INT,
    advancement_probs   JSONB,
    champion_prob       JSONB,
    model_version       TEXT,
    runtime_secs        NUMERIC,
    computed_at         TIMESTAMPTZ  DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 4. prediction_snapshots
--    Daily point-in-time snapshot of per-team advancement probabilities and
--    calibration metrics. Enables time-series drift analysis.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id                   BIGSERIAL PRIMARY KEY,
    season               INT          NOT NULL,
    snapshot_date        DATE         NOT NULL,
    team                 TEXT         NOT NULL,
    championship_prob    NUMERIC,
    final_four_prob      NUMERIC,
    elite_eight_prob     NUMERIC,
    sweet_sixteen_prob   NUMERIC,
    brier_score          NUMERIC,
    log_loss             NUMERIC,
    created_at           TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (season, snapshot_date, team)
);

-- ---------------------------------------------------------------------------
-- 5. intel_alerts
--    Injury/roster intelligence scraped from Rotowire and news feeds.
--    severity drives triage priority; needs_verification flags items for
--    human review before the simulation path is committed.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS intel_alerts (
    id                 BIGSERIAL PRIMARY KEY,
    alert_id           TEXT         UNIQUE NOT NULL,
    source             TEXT         NOT NULL,
    content            TEXT         NOT NULL,
    keywords           TEXT[],
    severity           TEXT         CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    teams_mentioned    TEXT[],
    region             TEXT,
    needs_verification BOOLEAN      DEFAULT FALSE,
    url                TEXT,
    alerted_at         TIMESTAMPTZ  DEFAULT NOW(),
    resolved           BOOLEAN      DEFAULT FALSE,
    created_at         TIMESTAMPTZ  DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- 6. game_results
--    Historical and in-season game outcomes. neutral_site flag is TRUE for
--    all NCAA tournament games. tournament_round stores e.g. 'R64', 'R32'.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS game_results (
    id               BIGSERIAL PRIMARY KEY,
    season           INT          NOT NULL,
    game_date        DATE         NOT NULL,
    home_team        TEXT         NOT NULL,
    away_team        TEXT         NOT NULL,
    home_score       INT,
    away_score       INT,
    neutral_site     BOOLEAN      DEFAULT FALSE,
    tournament_round TEXT,
    created_at       TIMESTAMPTZ  DEFAULT NOW(),
    UNIQUE (season, game_date, home_team, away_team)
);

-- ---------------------------------------------------------------------------
-- 7. pipeline_runs
--    Audit log for every scheduled or manually triggered pipeline execution.
--    status uses a CHECK constraint to enforce a known finite set of states.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id                    BIGSERIAL PRIMARY KEY,
    run_date              DATE         NOT NULL,
    trigger               TEXT         NOT NULL,
    status                TEXT         CHECK (status IN ('success', 'partial', 'failed')),
    teams_updated         INT,
    predictions_computed  INT,
    alerts_found          INT,
    duration_secs         NUMERIC,
    error_log             TEXT,
    created_at            TIMESTAMPTZ  DEFAULT NOW()
);
-- 002_indexes.sql
-- Performance indexes and Supabase Realtime configuration.
-- Run after 001_initial_schema.sql.

-- ---------------------------------------------------------------------------
-- teams
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_teams_season
    ON teams (season);

-- ---------------------------------------------------------------------------
-- matchup_predictions
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_matchup_predictions_season_computed_at
    ON matchup_predictions (season, computed_at DESC);

-- ---------------------------------------------------------------------------
-- prediction_snapshots
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_season_snapshot_date
    ON prediction_snapshots (season, snapshot_date DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_snapshots_team_season
    ON prediction_snapshots (team, season);

-- ---------------------------------------------------------------------------
-- intel_alerts
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_intel_alerts_resolved_alerted_at
    ON intel_alerts (resolved, alerted_at DESC);

-- ---------------------------------------------------------------------------
-- bracket_runs
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_bracket_runs_run_date_season
    ON bracket_runs (run_date DESC, season);

-- ---------------------------------------------------------------------------
-- game_results
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_game_results_season_game_date
    ON game_results (season, game_date);

-- ---------------------------------------------------------------------------
-- Supabase Realtime — REPLICA IDENTITY FULL
--
-- Setting REPLICA IDENTITY FULL instructs Postgres to include the full row
-- image (before AND after) in its WAL logical replication stream.  Supabase
-- Realtime listens to this stream; without FULL identity it can only emit
-- INSERT events (not UPDATE/DELETE) for tables without a primary-key replica
-- identity.  Enabling it on intel_alerts and prediction_snapshots allows the
-- War Room UI to receive live push updates the moment a new alert is resolved
-- or a snapshot row is written, without polling.
-- ---------------------------------------------------------------------------

-- Enables Supabase Realtime (INSERT / UPDATE / DELETE) on intel_alerts.
ALTER TABLE intel_alerts REPLICA IDENTITY FULL;

-- Enables Supabase Realtime (INSERT / UPDATE / DELETE) on prediction_snapshots.
ALTER TABLE prediction_snapshots REPLICA IDENTITY FULL;
-- 003_intel_snapshots.sql
-- Stores the full IntelResponse JSON from intel_engine.py — one row per
-- pipeline run.  The frontend reads the latest row for a given season.

CREATE TABLE IF NOT EXISTS intel_snapshots (
    id          BIGSERIAL PRIMARY KEY,
    season      INT          NOT NULL,
    snapshot    JSONB        NOT NULL,  -- full IntelResponse as JSON
    computed_at TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intel_snapshots_season_computed_at
    ON intel_snapshots (season, computed_at DESC);

ALTER TABLE intel_snapshots REPLICA IDENTITY FULL;
