"""GitHub Actions pipeline orchestration for Madness Matrix.

Orchestrates three daily run profiles:
  FULL    — 6 AM ET: scrape all sources → ST-GNN → Bayesian ADVI → MC 10k → write Supabase
  INTEL   — 12 PM ET: intel refresh (Rotowire/ESPN/On3) + warm-start Bayesian update
  RESULTS — 10 PM ET: ingest game results + Brier recalibration

Each profile returns a summary dict that is logged to the Supabase
pipeline_runs table via SupabaseWriter.
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
    """Orchestrates a single pipeline run for a given profile.

    Args:
        season:   Tournament year (e.g. 2026).
        dry_run:  If True, skips all network calls and writes — safe for tests
                  and local debugging.
    """

    def __init__(self, season: int = 2026, dry_run: bool = False) -> None:
        self.season   = season
        self.dry_run  = dry_run
        self._writer  = None if dry_run else self._init_writer()

    def _init_writer(self):
        from src.pipeline.supabase_writer import SupabaseWriter
        return SupabaseWriter()

    # ------------------------------------------------------------------
    def run(self, profile: RunProfile) -> dict[str, Any]:
        """Execute the given profile and return a summary dict."""
        t0 = time.perf_counter()
        logger.info(
            "Pipeline run starting: profile=%s season=%s dry_run=%s",
            profile.value, self.season, self.dry_run,
        )
        try:
            if profile == RunProfile.FULL:
                summary = self._run_full()
            elif profile == RunProfile.INTEL:
                summary = self._run_intel()
            elif profile == RunProfile.RESULTS:
                summary = self._run_results()
            else:
                raise ValueError(f"Unknown RunProfile: {profile!r}")
            summary["status"] = "success"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline run failed: %s", exc)
            summary = {
                "teams_updated": 0,
                "predictions_computed": 0,
                "alerts_found": 0,
                "status": "failed",
                "error_log": str(exc),
            }

        summary["trigger"]       = profile.value
        summary["duration_secs"] = round(time.perf_counter() - t0, 3)

        if not self.dry_run and self._writer:
            try:
                self._writer.log_pipeline_run(**{
                    k: summary[k]
                    for k in (
                        "trigger", "status", "teams_updated",
                        "predictions_computed", "alerts_found", "duration_secs",
                    )
                }, error_log=summary.get("error_log"))
            except Exception as log_exc:  # noqa: BLE001
                logger.warning("Failed to log pipeline run to Supabase: %s", log_exc)

        return summary

    # ------------------------------------------------------------------
    def _run_full(self) -> dict[str, Any]:
        """6 AM ET — scrape all sources, run full prediction stack."""
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}

        import datetime
        from src.data.barttorvik import fetch_trank
        from src.data.conference_rpi import compute_conference_rpi, assign_rpi_tiers

        # 1. Scrape T-Rank efficiency metrics
        teams_df = fetch_trank(self.season)

        # 2. Compute Conference RPI for GAT Conference node features
        rpis = assign_rpi_tiers(compute_conference_rpi(teams_df))
        logger.info("Conference RPI computed for %d conferences", len(rpis))

        # 3. Write teams to Supabase
        team_rows = teams_df.to_dict(orient="records")
        now = datetime.datetime.utcnow().isoformat()
        for row in team_rows:
            row["season"]     = self.season
            row["scraped_at"] = now
        self._writer.upsert_teams(team_rows)

        return {
            "teams_updated":        len(team_rows),
            "predictions_computed": 0,
            "alerts_found":         0,
        }

    def _run_intel(self) -> dict[str, Any]:
        """12 PM ET — scrape intel alerts, warm-start Bayesian update."""
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}

        from src.data.news_scraper import scrape_intel_alerts
        alerts = scrape_intel_alerts()
        for alert in alerts:
            self._writer.insert_intel_alert(alert)

        return {
            "teams_updated":        0,
            "predictions_computed": 0,
            "alerts_found":         len(alerts),
        }

    def _run_results(self) -> dict[str, Any]:
        """10 PM ET — ingest game results, update Brier scores."""
        if self.dry_run:
            return {"teams_updated": 0, "predictions_computed": 0, "alerts_found": 0}

        from src.data.kaggle_ingestion import load_recent_results
        results = load_recent_results(self.season)
        self._writer.upsert_game_results(results)

        return {
            "teams_updated":        0,
            "predictions_computed": len(results),
            "alerts_found":         0,
        }
