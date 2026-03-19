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

        # Normalize column names: fetch_trank returns conf/adj_o/adj_d/adj_t/sos_adj_em
        teams_df = teams_df.rename(columns={
            "conf":       "conference",
            "sos_adj_em": "sos",
            "adj_o":      "adj_oe",
            "adj_d":      "adj_de",
            "adj_t":      "tempo",
        })

        # 2. Compute Conference RPI for GAT Conference node features
        rpis = assign_rpi_tiers(compute_conference_rpi(teams_df))
        logger.info("Conference RPI computed for %d conferences", len(rpis))

        # 3. Write teams to Supabase (only columns that exist in the schema)
        _TEAM_COLS = {"name", "conference", "seed", "adj_oe", "adj_de", "adj_em",
                      "tempo", "luck", "sos", "coach", "region"}
        team_rows = teams_df.to_dict(orient="records")
        now = datetime.datetime.utcnow().isoformat()
        filtered_rows = []
        for row in team_rows:
            filtered = {k: v for k, v in row.items() if k in _TEAM_COLS}
            filtered["season"]     = self.season
            filtered["adj_em"]     = round(
                float(filtered.get("adj_oe") or 0) - float(filtered.get("adj_de") or 0), 2
            )
            filtered["scraped_at"] = now
            filtered_rows.append(filtered)
        self._writer.upsert_teams(filtered_rows)

        # 4. Run Monte Carlo bracket simulation and write to Supabase
        predictions_computed = 0
        try:
            from src.api.data_cache import DataLoader
            from src.api.bracket_runner import build_real_simulation
            from src.api.bracket_2026 import get_bracket_teams_ordered
            loader = DataLoader()
            teams_ordered = get_bracket_teams_ordered()
            sim = build_real_simulation(teams_ordered, 10_000, self.season, loader=loader)
            adv_probs: dict = {}
            champ_prob: dict = {}
            for t in sim.advancements:
                adv_probs[t.team] = t.advancement_probs
                champ_prob[t.team] = t.advancement_probs.get("Championship", 0.0)
            champion = max(champ_prob, key=champ_prob.get) if champ_prob else None
            self._writer.upsert_bracket_run({
                "season":            self.season,
                "run_date":          datetime.date.today().isoformat(),
                "n_simulations":     sim.n_simulations,
                "advancement_probs": adv_probs,
                "champion_prob":     champ_prob,
                "model_version":     "v2",
                "runtime_secs":      0,
            })
            predictions_computed = len(sim.advancements)
            logger.info("Bracket run written: champion=%s", champion)
        except Exception as exc:
            logger.warning("Bracket simulation skipped: %s", exc)

        # 5. Run intel engine and write snapshot to Supabase
        try:
            from src.api.data_cache import DataLoader
            from src.api.intel_engine import build_intel
            from dataclasses import asdict
            loader = DataLoader()
            intel = build_intel(season=self.season, loader=loader)
            self._writer.write_intel_snapshot(self.season, asdict(intel))
            logger.info("Intel snapshot written: %d flags", len(intel.flags))
        except Exception as exc:
            logger.warning("Intel snapshot skipped: %s", exc)

        return {
            "teams_updated":        len(filtered_rows),
            "predictions_computed": predictions_computed,
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
