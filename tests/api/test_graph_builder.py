"""
tests/api/test_graph_builder.py
TDD tests for src/api/graph_builder.py — build_real_graph()

RED phase: all tests are written before the implementation exists.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.api.schemas import GraphResponse

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_TRANK = pd.DataFrame({
    "team":       ["Duke", "Kansas", "Gonzaga", "Auburn"],
    "adj_oe":     [118.0, 115.2, 119.1, 116.0],
    "adj_de":     [92.0,  94.1,  93.5,  95.2],
    "tempo":      [70.0,  71.5,  68.2,  74.1],
    "luck":       [0.02, -0.01,  0.05,  0.00],
    "seed":       [1,     1,     2,     3],
    "conference": ["ACC", "Big 12", "WCC", "SEC"],
})

MOCK_SEEDS = {"Duke": 1, "Kansas": 2}


def _make_loader(trank_df: pd.DataFrame, seeds_dict: dict) -> MagicMock:
    loader = MagicMock()
    loader.get_trank.return_value = trank_df
    loader.get_tournament_seeds.return_value = seeds_dict
    return loader


# ---------------------------------------------------------------------------
# Import under test (deferred so RED → GREEN is clear)
# ---------------------------------------------------------------------------

from src.api.graph_builder import build_real_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_graph_response():
    """build_real_graph must return a GraphResponse instance."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    assert isinstance(result, GraphResponse)


def test_team_count_matches_trank_rows():
    """Number of team nodes must equal the number of rows in trank_df."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    assert len(result.teams) == 4


def test_team_names_preserved():
    """Team names Duke and Kansas must appear in the result."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    names = {t.name for t in result.teams}
    assert "Duke" in names
    assert "Kansas" in names


def test_real_adj_oe_used():
    """Duke's adj_oe must be exactly 118.0 (from MOCK_TRANK)."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    duke = next(t for t in result.teams if t.name == "Duke")
    assert duke.adj_oe == pytest.approx(118.0)


def test_real_adj_de_used():
    """Duke's adj_de must be exactly 92.0 (from MOCK_TRANK)."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    duke = next(t for t in result.teams if t.name == "Duke")
    assert duke.adj_de == pytest.approx(92.0)


def test_conference_nodes_generated():
    """ConferenceNodeResponse objects must be generated for ACC, Big 12, WCC, SEC."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    conf_names = {c.name for c in result.conferences}
    assert "ACC" in conf_names
    assert "Big 12" in conf_names


def test_conference_edges_count_equals_team_count():
    """One conference_edge per team (team → conference membership)."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    assert len(result.conference_edges) == len(result.teams)


def test_seed_applied_from_seeds_dict():
    """When a team appears in seeds_dict, its seed must come from there."""
    loader = _make_loader(MOCK_TRANK, {"Duke": 7})  # override seed to 7
    result = build_real_graph(season=2024, loader=loader)
    duke = next(t for t in result.teams if t.name == "Duke")
    assert duke.seed == 7


def test_empty_trank_returns_empty_graph():
    """An empty trank DataFrame must produce a GraphResponse with 0 teams."""
    loader = _make_loader(pd.DataFrame(), {})
    result = build_real_graph(season=2024, loader=loader)
    assert isinstance(result, GraphResponse)
    assert len(result.teams) == 0
    assert len(result.conferences) == 0
    assert len(result.conference_edges) == 0


def test_xyz_positions_are_finite_floats():
    """All team x, y, z positions must be finite (no NaN or inf)."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    for team in result.teams:
        assert math.isfinite(team.x), f"{team.name}.x is not finite"
        assert math.isfinite(team.y), f"{team.name}.y is not finite"
        assert math.isfinite(team.z), f"{team.name}.z is not finite"


def test_data_source_is_real():
    """GraphResponse.data_source must equal 'real'."""
    loader = _make_loader(MOCK_TRANK, MOCK_SEEDS)
    result = build_real_graph(season=2024, loader=loader)
    assert result.data_source == "real"
