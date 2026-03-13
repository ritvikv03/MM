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
