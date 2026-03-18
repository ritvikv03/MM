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
