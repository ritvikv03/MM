# tests/pipeline/test_supabase_writer.py
from unittest.mock import MagicMock
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


def test_upsert_teams_empty_list_is_noop(mock_client):
    writer = SupabaseWriter(client=mock_client)
    writer.upsert_teams([])
    mock_client.table.assert_not_called()


def test_upsert_snapshots(mock_client):
    writer = SupabaseWriter(client=mock_client)
    rows = [{"season": 2026, "snapshot_date": "2026-03-12", "team": "Duke",
             "championship_prob": 0.12}]
    writer.upsert_snapshots(rows)
    mock_client.table.assert_called_with("prediction_snapshots")


def test_upsert_matchup_predictions(mock_client):
    writer = SupabaseWriter(client=mock_client)
    rows = [{"season": 2026, "team_a": "Duke", "team_b": "UNC",
             "p_win_a": 0.63, "spread_mean": -4.5, "model_version": "v1.0"}]
    writer.upsert_matchup_predictions(rows)
    mock_client.table.assert_called_with("matchup_predictions")


def test_insert_intel_alert_uses_upsert_on_conflict(mock_client):
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
    # upsert with on_conflict="alert_id" for idempotency
    table = mock_client.table.return_value
    table.upsert.assert_called_once()
    call_kwargs = table.upsert.call_args[1]
    assert call_kwargs.get("on_conflict") == "alert_id"


def test_upsert_bracket_run(mock_client):
    writer = SupabaseWriter(client=mock_client)
    writer.upsert_bracket_run({
        "season": 2026, "run_date": "2026-03-12",
        "n_simulations": 10000, "advancement_probs": {},
        "champion_prob": {}, "model_version": "v1.0",
    })
    mock_client.table.assert_called_with("bracket_runs")


def test_log_pipeline_run_inserts_with_all_fields(mock_client):
    writer = SupabaseWriter(client=mock_client)
    writer.log_pipeline_run(
        trigger="full", status="success",
        teams_updated=68, predictions_computed=2278,
        alerts_found=3, duration_secs=142.7,
    )
    mock_client.table.assert_called_with("pipeline_runs")
    table = mock_client.table.return_value
    table.insert.assert_called_once()
    row = table.insert.call_args[0][0]
    assert row["trigger"] == "full"
    assert row["status"] == "success"
    assert row["teams_updated"] == 68
    assert row["duration_secs"] == 142.7


def test_upsert_game_results(mock_client):
    writer = SupabaseWriter(client=mock_client)
    rows = [{"season": 2026, "game_date": "2026-03-20",
             "home_team": "Duke", "away_team": "UNC",
             "home_score": 78, "away_score": 72}]
    writer.upsert_game_results(rows)
    mock_client.table.assert_called_with("game_results")
