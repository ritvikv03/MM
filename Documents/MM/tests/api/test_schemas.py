"""
tests/api/test_schemas.py
Tests for the data_source field on all three response schemas.
"""
import pytest


class TestDataSourceField:
    def test_graph_response_default_stub(self):
        from src.api.schemas import GraphResponse
        r = GraphResponse(teams=[], conferences=[], games=[], conference_edges=[])
        assert r.data_source == "stub"

    def test_matchup_response_default_stub(self):
        from src.api.schemas import MatchupResponse
        r = MatchupResponse(
            home_team="A",
            away_team="B",
            p_win_home=0.5,
            p_win_samples=[0.5] * 2000,
            spread_mean=0.0,
            spread_samples=[0.0] * 2000,
            luck_compressed=False,
        )
        assert r.data_source == "stub"

    def test_simulate_response_default_stub(self):
        from src.api.schemas import SimulateResponse
        r = SimulateResponse(n_simulations=100, advancements=[])
        assert r.data_source == "stub"

    def test_graph_response_real_value(self):
        from src.api.schemas import GraphResponse
        r = GraphResponse(
            teams=[], conferences=[], games=[], conference_edges=[],
            data_source="real",
        )
        assert r.data_source == "real"

    def test_simulate_response_real_value(self):
        from src.api.schemas import SimulateResponse
        r = SimulateResponse(n_simulations=100, advancements=[], data_source="real")
        assert r.data_source == "real"

    def test_matchup_response_real_value(self):
        from src.api.schemas import MatchupResponse
        r = MatchupResponse(
            home_team="Duke",
            away_team="UNC",
            p_win_home=0.6,
            p_win_samples=[0.6] * 2000,
            spread_mean=3.5,
            spread_samples=[3.5] * 2000,
            luck_compressed=True,
            data_source="real",
        )
        assert r.data_source == "real"
