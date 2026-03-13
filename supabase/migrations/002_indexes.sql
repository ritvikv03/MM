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
