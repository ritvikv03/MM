"""
Tests for src/api/server.py — FastAPI endpoint validation.
Response shapes match the Zod contracts in frontend/lib/api-types.ts.
Uses FastAPI's built-in TestClient (no real network calls).
"""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_ok(self):
        assert client.get("/health").json()["status"] == "ok"


class TestGraphEndpoint:
    def test_returns_200(self):
        assert client.get("/api/graph?season=2024").status_code == 200

    def test_has_teams_list(self):
        data = client.get("/api/graph?season=2024").json()
        assert isinstance(data["teams"], list)
        assert len(data["teams"]) > 0

    def test_has_conferences_list(self):
        data = client.get("/api/graph?season=2024").json()
        assert isinstance(data["conferences"], list)
        assert len(data["conferences"]) > 0

    def test_has_games_list(self):
        data = client.get("/api/graph?season=2024").json()
        assert isinstance(data["games"], list)

    def test_has_conference_edges(self):
        data = client.get("/api/graph?season=2024").json()
        assert isinstance(data["conference_edges"], list)
        assert len(data["conference_edges"]) > 0

    def test_team_node_fields_match_zod(self):
        team = client.get("/api/graph?season=2024").json()["teams"][0]
        for field in ["id", "name", "conference", "seed", "adj_oe", "adj_de", "tempo", "x", "y", "z", "color"]:
            assert field in team, f"Missing field: {field}"

    def test_team_color_is_integer(self):
        team = client.get("/api/graph?season=2024").json()["teams"][0]
        assert isinstance(team["color"], int)

    def test_team_adj_oe_is_float(self):
        team = client.get("/api/graph?season=2024").json()["teams"][0]
        assert isinstance(team["adj_oe"], float)

    def test_conference_node_fields_match_zod(self):
        conf = client.get("/api/graph?season=2024").json()["conferences"][0]
        for field in ["id", "name", "x", "y", "z", "color"]:
            assert field in conf, f"Missing field: {field}"

    def test_conference_edge_has_member_of_type(self):
        edges = client.get("/api/graph?season=2024").json()["conference_edges"]
        assert all(e["edge_type"] == "member_of" for e in edges)

    def test_game_edge_fields(self):
        games = client.get("/api/graph?season=2024").json()["games"]
        assert len(games) > 0
        g = games[0]
        for field in ["source", "target", "home_win", "spread", "date"]:
            assert field in g, f"Missing field: {field}"

    def test_deterministic_across_calls(self):
        a = client.get("/api/graph?season=2024").json()
        b = client.get("/api/graph?season=2024").json()
        assert a["teams"][0]["x"] == b["teams"][0]["x"]

    def test_different_seasons_differ(self):
        a = client.get("/api/graph?season=2024").json()
        b = client.get("/api/graph?season=2023").json()
        assert a["teams"][0]["tempo"] != b["teams"][0]["tempo"] or \
               a["teams"][0]["adj_oe"] != b["teams"][0]["adj_oe"]

    def test_rejects_invalid_season(self):
        assert client.get("/api/graph?season=1990").status_code == 422

    def test_no_season_field_in_response(self):
        # GraphResponse has no top-level season key (matches frontend Zod)
        data = client.get("/api/graph?season=2024").json()
        assert "season" not in data


class TestMatchupEndpoint:
    def _post(self, home="Duke", away="UNC", season=2024, neutral=False):
        return client.post("/api/matchup", json={
            "home_team": home, "away_team": away,
            "season": season, "neutral_site": neutral,
        })

    def test_returns_200(self):
        assert self._post().status_code == 200

    def test_has_home_and_away_team_fields(self):
        data = self._post().json()
        assert "home_team" in data and "away_team" in data
        assert data["home_team"] == "Duke"
        assert data["away_team"] == "UNC"

    def test_p_win_home_in_range(self):
        p = self._post().json()["p_win_home"]
        assert 0.0 <= p <= 1.0

    def test_p_win_samples_non_empty(self):
        assert len(self._post().json()["p_win_samples"]) > 100

    def test_p_win_samples_in_range(self):
        for p in self._post().json()["p_win_samples"][:50]:
            assert 0 <= p <= 1

    def test_spread_mean_is_float(self):
        assert isinstance(self._post().json()["spread_mean"], float)

    def test_spread_samples_non_empty(self):
        assert len(self._post().json()["spread_samples"]) > 100

    def test_luck_compressed_is_bool(self):
        assert isinstance(self._post().json()["luck_compressed"], bool)

    def test_neutral_site_reduces_home_advantage(self):
        home_adv   = self._post(neutral=False).json()["spread_mean"]
        neutral_g  = self._post(neutral=True).json()["spread_mean"]
        # Neutral site spread should differ from home-court spread
        assert home_adv != neutral_g

    def test_rejects_same_team(self):
        assert self._post("Duke", "Duke").status_code == 422

    def test_rejects_empty_team(self):
        assert self._post("", "UNC").status_code == 422

    def test_deterministic_for_same_matchup(self):
        a = self._post().json()
        b = self._post().json()
        assert a["spread_mean"] == b["spread_mean"]

    def test_different_matchups_differ(self):
        a = self._post("Duke", "UNC").json()
        b = self._post("Houston", "Kansas").json()
        assert a["spread_mean"] != b["spread_mean"]


class TestSimulateEndpoint:
    _TEAMS = ["Duke", "UNC", "Kansas", "Kentucky", "Gonzaga", "Houston"]

    def _post(self, teams=None, n=1000):
        return client.post("/api/bracket/simulate", json={
            "teams": teams or self._TEAMS,
            "n_simulations": n,
        })

    def test_returns_200(self):
        assert self._post().status_code == 200

    def test_has_advancements_list(self):
        data = self._post().json()
        assert isinstance(data["advancements"], list)
        assert len(data["advancements"]) == len(self._TEAMS)

    def test_advancement_item_has_required_fields(self):
        item = self._post().json()["advancements"][0]
        assert "team" in item
        assert "advancement_probs" in item
        assert "entropy" in item

    def test_advancement_probs_has_all_rounds(self):
        probs = self._post().json()["advancements"][0]["advancement_probs"]
        for rnd in ["R64", "R32", "S16", "E8", "F4", "Championship"]:
            assert rnd in probs, f"Missing round: {rnd}"

    def test_advancement_probs_monotone_decreasing(self):
        for item in self._post().json()["advancements"]:
            p = item["advancement_probs"]
            assert p["R64"] >= p["R32"] >= p["S16"] >= p["E8"] >= p["F4"] >= p["Championship"]

    def test_entropy_is_positive(self):
        for item in self._post().json()["advancements"]:
            assert item["entropy"] >= 0.0

    def test_n_simulations_in_response(self):
        data = self._post(n=500).json()
        assert data["n_simulations"] == 500

    def test_rejects_too_few_simulations(self):
        r = client.post("/api/bracket/simulate", json={"teams": self._TEAMS, "n_simulations": 10})
        assert r.status_code == 422

    def test_team_names_preserved(self):
        items = self._post().json()["advancements"]
        returned = {i["team"] for i in items}
        assert returned == set(self._TEAMS)

    def test_deterministic(self):
        a = self._post().json()
        b = self._post().json()
        assert a["advancements"][0]["entropy"] == b["advancements"][0]["entropy"]


class TestDataSourceField:
    def test_graph_response_includes_data_source_stub(self):
        resp = client.get("/api/graph?season=2024")
        assert resp.json()["data_source"] == "stub"

    def test_matchup_response_includes_data_source_stub(self):
        resp = client.post("/api/matchup", json={
            "home_team": "Duke", "away_team": "Kansas",
            "season": 2024, "neutral_site": True
        })
        assert resp.json()["data_source"] == "stub"

    def test_simulate_response_includes_data_source_stub(self):
        resp = client.post("/api/bracket/simulate", json={
            "teams": ["Duke", "Kansas"], "n_simulations": 100
        })
        assert resp.json()["data_source"] == "stub"
