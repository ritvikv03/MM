"""Tests for conference RPI computation and tier assignment."""
import pandas as pd
import pytest
from src.data.conference_rpi import (
    ConferenceRPI,
    compute_conference_rpi,
    assign_rpi_tiers,
)


def _make_teams_df():
    return pd.DataFrame({
        "conference": ["ACC", "ACC", "MAC", "MAC", "Big 12", "Big 12"],
        "adj_em":    [20.0,  18.0,   5.0,   3.0,   22.0,    24.0],
        "sos":       [15.0,  14.0,   8.0,   7.0,   18.0,    19.0],
    })


def test_compute_returns_correct_count():
    df = _make_teams_df()
    result = compute_conference_rpi(df)
    assert len(result) == 3


def test_compute_returns_conference_rpi_objects():
    df = _make_teams_df()
    result = compute_conference_rpi(df)
    assert all(isinstance(r, ConferenceRPI) for r in result)


def test_rpi_score_formula():
    """rpi = 0.6 * mean_adj_em + 0.4 * mean_sos"""
    df = _make_teams_df()
    result = {r.conference: r for r in compute_conference_rpi(df)}
    # ACC: 0.6 * 19.0 + 0.4 * 14.5 = 11.4 + 5.8 = 17.2
    assert result["ACC"].rpi_score == pytest.approx(17.2, abs=0.01)
    # MAC: 0.6 * 4.0 + 0.4 * 7.5 = 2.4 + 3.0 = 5.4
    assert result["MAC"].rpi_score == pytest.approx(5.4, abs=0.01)
    # Big 12: 0.6 * 23.0 + 0.4 * 18.5 = 13.8 + 7.4 = 21.2
    assert result["Big 12"].rpi_score == pytest.approx(21.2, abs=0.01)


def test_results_sorted_descending():
    df = _make_teams_df()
    result = compute_conference_rpi(df)
    scores = [r.rpi_score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_n_teams_populated():
    df = _make_teams_df()
    result = {r.conference: r for r in compute_conference_rpi(df)}
    assert result["ACC"].n_teams == 2
    assert result["MAC"].n_teams == 2


def test_missing_column_raises():
    df = pd.DataFrame({"conference": ["ACC"], "adj_em": [20.0]})  # missing sos
    with pytest.raises(ValueError, match="missing columns"):
        compute_conference_rpi(df)


def test_assign_tiers_all_in_range():
    df = _make_teams_df()
    rpis = compute_conference_rpi(df)
    tiers = assign_rpi_tiers(rpis)
    assert all(1 <= t.tier <= 5 for t in tiers)


def test_assign_tiers_top_is_tier_1():
    df = _make_teams_df()
    rpis = compute_conference_rpi(df)
    tiers = assign_rpi_tiers(rpis)
    top = max(tiers, key=lambda t: t.rpi_score)
    assert top.tier == 1


def test_assign_tiers_empty_list():
    result = assign_rpi_tiers([])
    assert result == []
