"""Tests for PipelineRunner — GitHub Actions orchestration."""
import pytest
from src.pipeline.github_actions_runner import PipelineRunner, RunProfile


def test_run_profile_enum_values():
    assert RunProfile.FULL.value == "full"
    assert RunProfile.INTEL.value == "intel"
    assert RunProfile.RESULTS.value == "results"


def test_pipeline_runner_init_dry_run():
    runner = PipelineRunner(season=2026, dry_run=True)
    assert runner.season == 2026
    assert runner.dry_run is True


def test_run_full_profile_dry_run():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.FULL)
    assert result["trigger"] == "full"
    assert result["status"] == "success"
    assert "duration_secs" in result


def test_run_intel_profile_dry_run():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.INTEL)
    assert result["trigger"] == "intel"
    assert result["status"] == "success"


def test_run_results_profile_dry_run():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.RESULTS)
    assert result["trigger"] == "results"
    assert result["status"] == "success"


def test_run_returns_duration():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.FULL)
    assert isinstance(result["duration_secs"], float)
    assert result["duration_secs"] >= 0.0


def test_run_returns_team_counts():
    runner = PipelineRunner(season=2026, dry_run=True)
    result = runner.run(RunProfile.FULL)
    assert "teams_updated" in result
    assert "predictions_computed" in result
    assert "alerts_found" in result
