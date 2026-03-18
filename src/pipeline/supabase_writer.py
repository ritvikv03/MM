"""Write pipeline outputs to Supabase tables."""
from __future__ import annotations

import datetime
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _get_client():
    """Return a live Supabase client using environment variables."""
    from supabase import create_client  # type: ignore
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])


class SupabaseWriter:
    """Thin wrapper that upserts/inserts pipeline outputs to Supabase tables.

    Pass client=mock in tests to avoid live network calls.
    """

    def __init__(self, client=None) -> None:
        self._client = client or _get_client()

    def _upsert(self, table: str, rows: list[dict], on_conflict: str) -> None:
        if not rows:
            return
        self._client.table(table).upsert(rows, on_conflict=on_conflict).execute()

    def upsert_teams(self, rows: list[dict[str, Any]]) -> None:
        """Upsert team efficiency rows. Conflict key: (season, name)."""
        self._upsert("teams", rows, "season,name")

    def upsert_snapshots(self, rows: list[dict[str, Any]]) -> None:
        """Upsert daily prediction snapshots. Conflict key: (season, snapshot_date, team)."""
        self._upsert("prediction_snapshots", rows, "season,snapshot_date,team")

    def upsert_matchup_predictions(self, rows: list[dict[str, Any]]) -> None:
        """Upsert matchup posterior rows. Conflict key: (season, team_a, team_b, computed_at)."""
        self._upsert("matchup_predictions", rows, "season,team_a,team_b,computed_at")

    def upsert_bracket_run(self, row: dict[str, Any]) -> None:
        """Insert a bracket run record (no conflict key — always a new run)."""
        self._client.table("bracket_runs").insert(row).execute()

    def write_intel_snapshot(self, season: int, intel_dict: dict[str, Any]) -> None:
        """Insert a full IntelResponse snapshot for the frontend to read."""
        self._client.table("intel_snapshots").insert({
            "season": season,
            "snapshot": intel_dict,
        }).execute()

    def insert_intel_alert(self, alert: dict[str, Any]) -> None:
        """Upsert an intel alert, deduplicating by alert_id."""
        self._client.table("intel_alerts").upsert(alert, on_conflict="alert_id").execute()

    def upsert_game_results(self, rows: list[dict[str, Any]]) -> None:
        """Upsert game result rows. Conflict key: (season, game_date, home_team, away_team)."""
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
        """Insert a pipeline audit record."""
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
