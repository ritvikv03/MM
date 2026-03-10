"""
tests/graph/test_node_features.py
==================================
TDD test suite for src/graph/node_features.py

Tests follow RED → GREEN order.  All 25+ tests are organized into classes
mirroring the public API:

  TestNodeFeatureBuilderInit       — constructor validation
  TestNodeFeatureBuilderBuild      — build() happy path and edge cases
  TestNormalizeFeatures            — normalize_features()
  TestToTensor                     — to_tensor()
  TestBuildTeamIndex               — build_team_index()

PIT (Point-in-Time) note: build() must never fetch data internally.
Tests verify that only pre-assembled DataFrames are accepted and that
no network calls are made.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers to build minimal valid DataFrames for NodeFeatureBuilder.build()
# ---------------------------------------------------------------------------

EFFICIENCY_COLS = ["team_id", "adj_em", "adj_o", "adj_d", "adj_t", "luck"]
BPR_COLS = ["team_id", "team_bpr_weighted"]
SHOT_COLS = ["team_id", "rim_pct", "three_pct", "transition_pct", "efg"]

EXPECTED_OUTPUT_COLS = [
    "team_id", "season", "adj_em", "adj_o", "adj_d", "adj_t", "luck",
    "team_bpr_weighted", "rim_pct", "three_pct", "transition_pct", "efg",
    "availability", "roster_continuity", "effective_strength",
]


def _make_efficiency_df(teams=("Duke", "Kansas")) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(teams):
        rows.append({
            "team_id": t,
            "adj_em": 20.0 + i,
            "adj_o": 115.0 + i,
            "adj_d": 95.0 - i,
            "adj_t": 68.0 + i,
            "luck": 0.01 * i,
        })
    return pd.DataFrame(rows)


def _make_bpr_df(teams=("Duke", "Kansas")) -> pd.DataFrame:
    rows = [{"team_id": t, "team_bpr_weighted": 3.0 + i}
            for i, t in enumerate(teams)]
    return pd.DataFrame(rows)


def _make_shot_df(teams=("Duke", "Kansas")) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(teams):
        rows.append({
            "team_id": t,
            "rim_pct": 0.30 + 0.01 * i,
            "three_pct": 0.35 + 0.01 * i,
            "transition_pct": 0.15 + 0.01 * i,
            "efg": 0.52 + 0.01 * i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Import under test (deferred so a missing module causes a clear ImportError)
# ---------------------------------------------------------------------------

from src.graph.node_features import (  # noqa: E402
    NodeFeatureBuilder,
    build_team_index,
    normalize_features,
    to_tensor,
)


# ===========================================================================
# TestNodeFeatureBuilderInit
# ===========================================================================

class TestNodeFeatureBuilderInit:
    """Constructor smoke tests."""

    def test_init_stores_season(self):
        builder = NodeFeatureBuilder(season=2024)
        assert builder.season == 2024

    def test_init_different_seasons(self):
        for yr in [2008, 2015, 2024, 2025]:
            b = NodeFeatureBuilder(season=yr)
            assert b.season == yr

    def test_init_type(self):
        builder = NodeFeatureBuilder(season=2024)
        assert isinstance(builder, NodeFeatureBuilder)


# ===========================================================================
# TestNodeFeatureBuilderBuild — happy path
# ===========================================================================

class TestNodeFeatureBuilderBuild:
    """Tests for NodeFeatureBuilder.build()."""

    @pytest.fixture
    def builder(self):
        return NodeFeatureBuilder(season=2024)

    @pytest.fixture
    def dfs(self):
        teams = ("Duke", "Kansas")
        return (
            _make_efficiency_df(teams),
            _make_bpr_df(teams),
            _make_shot_df(teams),
        )

    # --- column contract ---------------------------------------------------

    def test_build_returns_dataframe(self, builder, dfs):
        eff, bpr, shot = dfs
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert isinstance(result, pd.DataFrame)

    def test_build_exact_columns(self, builder, dfs):
        eff, bpr, shot = dfs
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert list(result.columns) == EXPECTED_OUTPUT_COLS

    def test_build_row_count_matches_teams(self, builder, dfs):
        eff, bpr, shot = dfs
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert len(result) == 2

    def test_build_season_column_value(self, builder, dfs):
        eff, bpr, shot = dfs
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert (result["season"] == 2024).all()

    def test_build_team_id_preserved(self, builder, dfs):
        eff, bpr, shot = dfs
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert set(result["team_id"]) == {"Duke", "Kansas"}

    # --- effective_strength formula ----------------------------------------

    def test_effective_strength_formula_basic(self, builder):
        """effective_strength = adj_em * availability * (0.7 + 0.3 * roster_continuity)"""
        teams = ("Alpha",)
        eff = pd.DataFrame([{"team_id": "Alpha", "adj_em": 10.0,
                              "adj_o": 110.0, "adj_d": 100.0,
                              "adj_t": 68.0, "luck": 0.0}])
        bpr = pd.DataFrame([{"team_id": "Alpha", "team_bpr_weighted": 2.0}])
        shot = pd.DataFrame([{"team_id": "Alpha", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.50}])
        result = builder.build(eff, bpr, shot,
                                roster_continuity=1.0,
                                availability_vector=1.0)
        row = result.iloc[0]
        expected = 10.0 * 1.0 * (0.7 + 0.3 * 1.0)
        assert math.isclose(row["effective_strength"], expected, rel_tol=1e-6)

    def test_effective_strength_zero_availability(self, builder):
        """Zero availability must produce zero effective_strength."""
        teams = ("Alpha",)
        eff = pd.DataFrame([{"team_id": "Alpha", "adj_em": 25.0,
                              "adj_o": 118.0, "adj_d": 93.0,
                              "adj_t": 70.0, "luck": 0.05}])
        bpr = pd.DataFrame([{"team_id": "Alpha", "team_bpr_weighted": 4.0}])
        shot = pd.DataFrame([{"team_id": "Alpha", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.52}])
        result = builder.build(eff, bpr, shot,
                                roster_continuity=0.9,
                                availability_vector=0.0)
        assert result.iloc[0]["effective_strength"] == pytest.approx(0.0)

    def test_effective_strength_roster_continuity_zero(self, builder):
        """roster_continuity=0.0 uses factor 0.7."""
        eff = pd.DataFrame([{"team_id": "Beta", "adj_em": 10.0,
                              "adj_o": 110.0, "adj_d": 100.0,
                              "adj_t": 68.0, "luck": 0.0}])
        bpr = pd.DataFrame([{"team_id": "Beta", "team_bpr_weighted": 2.0}])
        shot = pd.DataFrame([{"team_id": "Beta", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.50}])
        result = builder.build(eff, bpr, shot,
                                roster_continuity=0.0,
                                availability_vector=1.0)
        expected = 10.0 * 1.0 * (0.7 + 0.3 * 0.0)
        assert result.iloc[0]["effective_strength"] == pytest.approx(expected)

    def test_effective_strength_roster_continuity_one(self, builder):
        """roster_continuity=1.0 uses factor 1.0."""
        eff = pd.DataFrame([{"team_id": "Gamma", "adj_em": 10.0,
                              "adj_o": 110.0, "adj_d": 100.0,
                              "adj_t": 68.0, "luck": 0.0}])
        bpr = pd.DataFrame([{"team_id": "Gamma", "team_bpr_weighted": 2.0}])
        shot = pd.DataFrame([{"team_id": "Gamma", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.50}])
        result = builder.build(eff, bpr, shot,
                                roster_continuity=1.0,
                                availability_vector=1.0)
        expected = 10.0 * 1.0 * 1.0
        assert result.iloc[0]["effective_strength"] == pytest.approx(expected)

    def test_availability_and_continuity_stored_as_scalar(self, builder):
        """Scalar availability_vector and roster_continuity broadcast to all rows."""
        teams = ("A", "B", "C")
        eff = _make_efficiency_df(teams)
        bpr = _make_bpr_df(teams)
        shot = _make_shot_df(teams)
        result = builder.build(eff, bpr, shot,
                                roster_continuity=0.75,
                                availability_vector=0.85)
        np.testing.assert_allclose(result["availability"].values, 0.85)
        np.testing.assert_allclose(result["roster_continuity"].values, 0.75)

    # --- NaN / missing value handling --------------------------------------

    def test_missing_bpr_filled_with_zero(self, builder):
        """Teams absent from bpr_df get team_bpr_weighted=0.0."""
        eff = _make_efficiency_df(("Duke", "Kansas"))
        # BPR only has Duke
        bpr = pd.DataFrame([{"team_id": "Duke", "team_bpr_weighted": 5.0}])
        shot = _make_shot_df(("Duke", "Kansas"))
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        kansas_row = result[result["team_id"] == "Kansas"].iloc[0]
        assert kansas_row["team_bpr_weighted"] == pytest.approx(0.0)

    def test_all_nan_efficiency_filled_with_zero(self, builder):
        """All-NaN adj_em values are filled with 0.0 before effective_strength."""
        eff = pd.DataFrame([{
            "team_id": "NanTeam",
            "adj_em": float("nan"),
            "adj_o": float("nan"),
            "adj_d": float("nan"),
            "adj_t": float("nan"),
            "luck": float("nan"),
        }])
        bpr = pd.DataFrame([{"team_id": "NanTeam", "team_bpr_weighted": 2.0}])
        shot = pd.DataFrame([{"team_id": "NanTeam", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.50}])
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        # adj_em was NaN → filled to 0.0 → effective_strength = 0.0
        assert result.iloc[0]["effective_strength"] == pytest.approx(0.0)
        assert result.iloc[0]["adj_em"] == pytest.approx(0.0)

    def test_missing_shot_cols_filled_with_zero(self, builder):
        """Teams absent from shot_df get shot columns filled with 0.0."""
        eff = _make_efficiency_df(("Duke", "Kansas"))
        bpr = _make_bpr_df(("Duke", "Kansas"))
        # shot only has Duke
        shot = pd.DataFrame([{"team_id": "Duke", "rim_pct": 0.3,
                               "three_pct": 0.35, "transition_pct": 0.15,
                               "efg": 0.52}])
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        kansas_row = result[result["team_id"] == "Kansas"].iloc[0]
        for col in ["rim_pct", "three_pct", "transition_pct", "efg"]:
            assert kansas_row[col] == pytest.approx(0.0)

    # --- empty DataFrame --------------------------------------------------

    def test_build_empty_efficiency_df_returns_empty(self, builder):
        """Empty efficiency_df should return an empty DataFrame with correct columns."""
        eff = pd.DataFrame(columns=EFFICIENCY_COLS)
        bpr = pd.DataFrame(columns=BPR_COLS)
        shot = pd.DataFrame(columns=SHOT_COLS)
        result = builder.build(eff, bpr, shot, 0.8, 0.9)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == EXPECTED_OUTPUT_COLS
        assert len(result) == 0

    # --- no internal fetches (PIT Integrity) -------------------------------

    def test_build_does_not_call_requests(self, builder, dfs, monkeypatch):
        """build() must not make HTTP requests — pure DataFrame transformation."""
        import unittest.mock as mock
        with mock.patch("requests.get") as mock_get:
            eff, bpr, shot = dfs
            builder.build(eff, bpr, shot, 0.8, 0.9)
            mock_get.assert_not_called()

    # --- large team set ----------------------------------------------------

    def test_build_many_teams(self, builder):
        """Handles 64 teams without error."""
        teams = [f"Team_{i:02d}" for i in range(64)]
        eff = _make_efficiency_df(teams)
        bpr = _make_bpr_df(teams)
        shot = _make_shot_df(teams)
        result = builder.build(eff, bpr, shot, 0.85, 0.95)
        assert len(result) == 64
        assert list(result.columns) == EXPECTED_OUTPUT_COLS


# ===========================================================================
# TestNormalizeFeatures
# ===========================================================================

class TestNormalizeFeatures:
    """Tests for normalize_features()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "team_id": ["A", "B", "C"],
            "adj_em": [10.0, 20.0, 30.0],
            "adj_o": [100.0, 110.0, 120.0],
        })

    def test_normalize_returns_copy(self, sample_df):
        result = normalize_features(sample_df, ["adj_em"])
        assert result is not sample_df

    def test_normalize_min_max_range(self, sample_df):
        result = normalize_features(sample_df, ["adj_em"])
        assert result["adj_em"].min() == pytest.approx(0.0)
        assert result["adj_em"].max() == pytest.approx(1.0)

    def test_normalize_multiple_cols(self, sample_df):
        result = normalize_features(sample_df, ["adj_em", "adj_o"])
        for col in ["adj_em", "adj_o"]:
            assert result[col].min() == pytest.approx(0.0)
            assert result[col].max() == pytest.approx(1.0)

    def test_normalize_does_not_touch_other_cols(self, sample_df):
        original_ids = sample_df["team_id"].copy()
        result = normalize_features(sample_df, ["adj_em"])
        pd.testing.assert_series_equal(result["team_id"], original_ids)

    def test_normalize_original_df_unchanged(self, sample_df):
        original_vals = sample_df["adj_em"].copy()
        normalize_features(sample_df, ["adj_em"])
        pd.testing.assert_series_equal(sample_df["adj_em"], original_vals)

    def test_normalize_missing_col_raises_value_error(self, sample_df):
        with pytest.raises(ValueError, match="nonexistent"):
            normalize_features(sample_df, ["nonexistent"])

    def test_normalize_constant_column(self):
        """Constant column (max==min) normalizes to 0.0 without division by zero."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0]})
        result = normalize_features(df, ["x"])
        assert (result["x"] == 0.0).all()

    def test_normalize_single_row(self):
        df = pd.DataFrame({"val": [42.0]})
        result = normalize_features(df, ["val"])
        # single value: min == max → normalized to 0.0
        assert result["val"].iloc[0] == pytest.approx(0.0)

    def test_normalize_preserves_row_order(self, sample_df):
        result = normalize_features(sample_df, ["adj_em"])
        # after min-max: 10→0.0, 20→0.5, 30→1.0
        assert result["adj_em"].tolist() == pytest.approx([0.0, 0.5, 1.0])


# ===========================================================================
# TestToTensor
# ===========================================================================

class TestToTensor:
    """Tests for to_tensor()."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "team_id": ["A", "B", "C"],
            "adj_em": [10.0, 20.0, 30.0],
            "adj_o": [100.0, 110.0, 120.0],
        })

    def test_to_tensor_shape(self, sample_df):
        t = to_tensor(sample_df, ["adj_em", "adj_o"])
        assert t.shape == (3, 2)

    def test_to_tensor_dtype_float32(self, sample_df):
        import torch
        t = to_tensor(sample_df, ["adj_em", "adj_o"])
        assert t.dtype == torch.float32

    def test_to_tensor_values_correct(self, sample_df):
        import torch
        t = to_tensor(sample_df, ["adj_em"])
        expected = torch.tensor([[10.0], [20.0], [30.0]], dtype=torch.float32)
        assert torch.allclose(t, expected)

    def test_to_tensor_missing_col_raises_value_error(self, sample_df):
        with pytest.raises(ValueError, match="missing_col"):
            to_tensor(sample_df, ["missing_col"])

    def test_to_tensor_single_feature(self, sample_df):
        t = to_tensor(sample_df, ["adj_em"])
        assert t.shape == (3, 1)

    def test_to_tensor_no_torch_at_module_level(self):
        """torch must not be imported at import time; lazy import only inside to_tensor."""
        import importlib
        import sys
        # Remove torch from sys.modules temporarily to test lazy import path.
        # We just verify the module loads successfully even if torch is present;
        # the real check is structural (no top-level import in source).
        import src.graph.node_features as nf_mod
        source_path = nf_mod.__file__
        with open(source_path) as fh:
            source = fh.read()
        # Top-level torch import lines look like "import torch" at column 0
        lines = source.splitlines()
        top_level_torch = [
            ln for ln in lines
            if ln.startswith("import torch") or ln.startswith("from torch")
        ]
        assert top_level_torch == [], (
            f"torch must not be imported at module level; found: {top_level_torch}"
        )


# ===========================================================================
# TestBuildTeamIndex
# ===========================================================================

class TestBuildTeamIndex:
    """Tests for build_team_index()."""

    def test_returns_dict(self):
        result = build_team_index(["Duke", "Kansas", "Arizona"])
        assert isinstance(result, dict)

    def test_alphabetical_order(self):
        result = build_team_index(["Kansas", "Duke", "Arizona"])
        assert result["Arizona"] == 0
        assert result["Duke"] == 1
        assert result["Kansas"] == 2

    def test_zero_based_indices(self):
        teams = ["C_Team", "A_Team", "B_Team"]
        result = build_team_index(teams)
        assert set(result.values()) == {0, 1, 2}

    def test_deterministic_across_calls(self):
        teams = ["Gonzaga", "Baylor", "Villanova", "Duke"]
        r1 = build_team_index(teams)
        r2 = build_team_index(teams)
        assert r1 == r2

    def test_single_team(self):
        result = build_team_index(["OnlyTeam"])
        assert result == {"OnlyTeam": 0}

    def test_empty_list(self):
        result = build_team_index([])
        assert result == {}

    def test_all_64_teams_unique_indices(self):
        teams = [f"Team_{i:03d}" for i in range(64)]
        result = build_team_index(teams)
        assert len(result) == 64
        assert set(result.values()) == set(range(64))
