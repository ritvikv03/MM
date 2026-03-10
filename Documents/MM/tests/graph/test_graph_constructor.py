"""
tests/graph/test_graph_constructor.py

RED phase — tests written before implementation exists.
Covers: TemporalSnapshot dataclass, GraphConstructor, and filter_games_pit().

All torch_geometric imports are mocked; no real PyG installation needed.
torch is available in this environment and used directly for tensor assertions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import sys

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def season() -> int:
    return 2024


@pytest.fixture()
def node_feature_cols() -> list[str]:
    return ["adj_o", "adj_d", "tempo"]


@pytest.fixture()
def edge_feature_cols() -> list[str]:
    return ["score_diff", "location_enc", "rest_disparity"]


@pytest.fixture()
def sample_games_df() -> pd.DataFrame:
    """10 games across days 1–20 for season 2024."""
    return pd.DataFrame({
        "Season":  [2024] * 10,
        "DayNum":  [1, 1, 5, 5, 10, 10, 15, 15, 20, 20],
        "WTeamID": ["Duke", "Kansas", "UNC", "Gonzaga", "Duke", "Villanova",
                    "Kansas", "UNC", "Gonzaga", "Duke"],
        "LTeamID": ["UNC", "Villanova", "Duke", "Kansas", "UNC", "Gonzaga",
                    "Villanova", "Gonzaga", "Villanova", "Kansas"],
        "score_diff":      [5.0, 10.0, 3.0, 8.0, 12.0, 4.0, 7.0, 6.0, 9.0, 2.0],
        "location_enc":    [1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0],
        "rest_disparity":  [0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 0.0, 2.0, 1.0, 0.0],
    })


@pytest.fixture()
def sample_node_df(node_feature_cols) -> pd.DataFrame:
    """Pre-built node features for 5 teams."""
    teams = ["Duke", "Kansas", "UNC", "Gonzaga", "Villanova"]
    data = {
        "team": teams,
        "adj_o":  [115.0, 112.0, 110.0, 118.0, 108.0],
        "adj_d":  [90.0,  92.0,  95.0,  88.0,  97.0],
        "tempo":  [70.0,  68.0,  72.0,  65.0,  71.0],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def constructor(season, node_feature_cols, edge_feature_cols):
    from src.graph.graph_constructor import GraphConstructor
    return GraphConstructor(
        season=season,
        node_feature_cols=node_feature_cols,
        edge_feature_cols=edge_feature_cols,
    )


@pytest.fixture()
def snapshot_day10(constructor, sample_games_df, sample_node_df):
    """TemporalSnapshot built up to DayNum=10."""
    games_up_to_10 = sample_games_df[sample_games_df["DayNum"] <= 10].copy()
    return constructor.build_snapshot(
        games_up_to_day=games_up_to_10,
        node_df=sample_node_df,
        day_num=10,
    )


# ---------------------------------------------------------------------------
# TestFilterGamesPit — standalone helper function
# ---------------------------------------------------------------------------

class TestFilterGamesPit:
    """Tests for filter_games_pit()."""

    def test_returns_only_rows_up_to_day_num(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=5)
        assert (result["DayNum"] <= 5).all()

    def test_excludes_rows_after_day_num(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=5)
        assert not (result["DayNum"] > 5).any()

    def test_includes_exact_day_num(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=10)
        assert 10 in result["DayNum"].values

    def test_returns_correct_row_count(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        # Days 1 and 5 each have 2 games → 4 rows
        result = filter_games_pit(sample_games_df, day_num=5)
        assert len(result) == 4

    def test_returns_dataframe(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=10)
        assert isinstance(result, pd.DataFrame)

    def test_raises_value_error_if_day_num_col_missing(self):
        from src.graph.graph_constructor import filter_games_pit

        bad_df = pd.DataFrame({"Season": [2024], "WTeamID": ["Duke"]})
        with pytest.raises(ValueError, match="DayNum"):
            filter_games_pit(bad_df, day_num=10)

    def test_empty_df_when_day_num_before_first_game(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=0)
        assert len(result) == 0

    def test_all_rows_returned_when_day_num_very_large(self, sample_games_df):
        from src.graph.graph_constructor import filter_games_pit

        result = filter_games_pit(sample_games_df, day_num=9999)
        assert len(result) == len(sample_games_df)


# ---------------------------------------------------------------------------
# TestTemporalSnapshot — dataclass field validation
# ---------------------------------------------------------------------------

class TestTemporalSnapshot:
    """Tests for the TemporalSnapshot dataclass structure."""

    def test_snapshot_has_required_fields(self, snapshot_day10):
        snap = snapshot_day10
        assert hasattr(snap, "day_num")
        assert hasattr(snap, "season")
        assert hasattr(snap, "node_features")
        assert hasattr(snap, "edge_index")
        assert hasattr(snap, "edge_attr")
        assert hasattr(snap, "team_index")
        assert hasattr(snap, "num_games")

    def test_snapshot_day_num_matches_input(self, snapshot_day10):
        assert snapshot_day10.day_num == 10

    def test_snapshot_season_matches_constructor(self, snapshot_day10, season):
        assert snapshot_day10.season == season

    def test_node_features_is_tensor(self, snapshot_day10):
        assert isinstance(snapshot_day10.node_features, torch.Tensor)

    def test_edge_index_is_tensor(self, snapshot_day10):
        assert isinstance(snapshot_day10.edge_index, torch.Tensor)

    def test_edge_attr_is_tensor(self, snapshot_day10):
        assert isinstance(snapshot_day10.edge_attr, torch.Tensor)

    def test_team_index_is_dict(self, snapshot_day10):
        assert isinstance(snapshot_day10.team_index, dict)

    def test_num_games_is_int(self, snapshot_day10):
        assert isinstance(snapshot_day10.num_games, int)


# ---------------------------------------------------------------------------
# TestBuildSnapshot — shape correctness and PIT filtering
# ---------------------------------------------------------------------------

class TestBuildSnapshot:
    """Tests for GraphConstructor.build_snapshot()."""

    def test_node_features_shape_rows_equal_num_teams(
        self, snapshot_day10
    ):
        N = len(snapshot_day10.team_index)
        F = snapshot_day10.node_features.shape[1]
        assert snapshot_day10.node_features.shape == (N, F)

    def test_node_features_shape_cols_equal_feature_count(
        self, snapshot_day10, node_feature_cols
    ):
        assert snapshot_day10.node_features.shape[1] == len(node_feature_cols)

    def test_edge_index_shape_is_2_by_E(self, snapshot_day10):
        assert snapshot_day10.edge_index.ndim == 2
        assert snapshot_day10.edge_index.shape[0] == 2

    def test_edge_attr_shape_rows_match_edge_count(self, snapshot_day10):
        E = snapshot_day10.edge_index.shape[1]
        assert snapshot_day10.edge_attr.shape[0] == E

    def test_edge_attr_shape_cols_equal_feature_count(
        self, snapshot_day10, edge_feature_cols
    ):
        assert snapshot_day10.edge_attr.shape[1] == len(edge_feature_cols)

    def test_edge_index_no_out_of_bounds_indices(self, snapshot_day10):
        N = len(snapshot_day10.team_index)
        assert snapshot_day10.edge_index.max().item() < N
        assert snapshot_day10.edge_index.min().item() >= 0

    def test_edge_index_dtype_is_long(self, snapshot_day10):
        assert snapshot_day10.edge_index.dtype == torch.long

    def test_num_games_equals_edge_count(self, snapshot_day10):
        E = snapshot_day10.edge_index.shape[1]
        assert snapshot_day10.num_games == E

    def test_pit_filtering_correct_edge_count(
        self, constructor, sample_games_df, sample_node_df
    ):
        """Only games with DayNum <= 5 should appear as edges at day 5."""
        games_up_to_5 = sample_games_df[sample_games_df["DayNum"] <= 5].copy()
        snap = constructor.build_snapshot(
            games_up_to_day=games_up_to_5,
            node_df=sample_node_df,
            day_num=5,
        )
        # Days 1 and 5 have 2 games each → 4 edges
        assert snap.edge_index.shape[1] == 4

    def test_winning_team_on_row_0_of_edge_index(
        self, constructor, sample_node_df
    ):
        """Row 0 = winning team index, row 1 = losing team index (directed)."""
        games = pd.DataFrame({
            "Season":  [2024],
            "DayNum":  [1],
            "WTeamID": ["Duke"],
            "LTeamID": ["UNC"],
            "score_diff":    [5.0],
            "location_enc":  [1.0],
            "rest_disparity":[0.0],
        })
        snap = constructor.build_snapshot(
            games_up_to_day=games,
            node_df=pd.DataFrame({
                "team": ["Duke", "UNC"],
                "adj_o": [115.0, 110.0],
                "adj_d": [90.0, 95.0],
                "tempo": [70.0, 72.0],
            }),
            day_num=1,
        )
        duke_idx = snap.team_index["Duke"]
        unc_idx  = snap.team_index["UNC"]
        assert snap.edge_index[0, 0].item() == duke_idx
        assert snap.edge_index[1, 0].item() == unc_idx

    def test_teams_not_in_node_df_get_zero_vectors(
        self, constructor, edge_feature_cols
    ):
        """Teams present in games but absent from node_df → zero feature rows."""
        games = pd.DataFrame({
            "Season":  [2024],
            "DayNum":  [1],
            "WTeamID": ["NewTeam"],
            "LTeamID": ["AnotherNew"],
            "score_diff":    [3.0],
            "location_enc":  [0.0],
            "rest_disparity":[1.0],
        })
        empty_node_df = pd.DataFrame(
            columns=["team", "adj_o", "adj_d", "tempo"]
        )
        snap = constructor.build_snapshot(
            games_up_to_day=games,
            node_df=empty_node_df,
            day_num=1,
        )
        # Both teams should exist in team_index and have zero feature vectors
        assert "NewTeam" in snap.team_index
        assert "AnotherNew" in snap.team_index
        for idx in snap.team_index.values():
            assert torch.all(snap.node_features[idx] == 0.0)

    def test_team_index_is_alphabetically_sorted(self, snapshot_day10):
        """team_index keys should be assigned in alphabetical order."""
        keys = list(snapshot_day10.team_index.keys())
        assert keys == sorted(keys)

    def test_team_index_values_are_contiguous_from_zero(self, snapshot_day10):
        indices = sorted(snapshot_day10.team_index.values())
        assert indices == list(range(len(indices)))

    def test_single_game_snapshot(self, constructor, sample_node_df):
        """A single-game DataFrame produces a snapshot with exactly 1 edge."""
        single_game = pd.DataFrame({
            "Season":  [2024],
            "DayNum":  [3],
            "WTeamID": ["Duke"],
            "LTeamID": ["Kansas"],
            "score_diff":    [7.0],
            "location_enc":  [1.0],
            "rest_disparity":[0.0],
        })
        snap = constructor.build_snapshot(
            games_up_to_day=single_game,
            node_df=sample_node_df,
            day_num=3,
        )
        assert snap.edge_index.shape[1] == 1
        assert snap.edge_attr.shape[0] == 1

    def test_empty_games_produces_empty_edge_index(self, constructor, sample_node_df):
        """Empty games DataFrame → edge_index with 0 columns."""
        empty_games = pd.DataFrame(
            columns=["Season", "DayNum", "WTeamID", "LTeamID",
                     "score_diff", "location_enc", "rest_disparity"]
        )
        snap = constructor.build_snapshot(
            games_up_to_day=empty_games,
            node_df=sample_node_df,
            day_num=1,
        )
        assert snap.edge_index.shape[1] == 0
        assert snap.num_games == 0

    def test_empty_games_still_returns_temporal_snapshot(self, constructor, sample_node_df):
        from src.graph.graph_constructor import TemporalSnapshot

        empty_games = pd.DataFrame(
            columns=["Season", "DayNum", "WTeamID", "LTeamID",
                     "score_diff", "location_enc", "rest_disparity"]
        )
        snap = constructor.build_snapshot(
            games_up_to_day=empty_games,
            node_df=sample_node_df,
            day_num=1,
        )
        assert isinstance(snap, TemporalSnapshot)

    def test_node_features_are_float_tensor(self, snapshot_day10):
        assert snapshot_day10.node_features.dtype in (torch.float32, torch.float64)

    def test_edge_attr_are_float_tensor(self, snapshot_day10):
        assert snapshot_day10.edge_attr.dtype in (torch.float32, torch.float64)


# ---------------------------------------------------------------------------
# TestBuildSeasonSnapshots — interval and ordering logic
# ---------------------------------------------------------------------------

class TestBuildSeasonSnapshots:
    """Tests for GraphConstructor.build_season_snapshots()."""

    def test_returns_list(self, constructor, sample_games_df, sample_node_df):
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        assert isinstance(snapshots, list)

    def test_snapshots_sorted_by_day_num_ascending(
        self, constructor, sample_games_df, sample_node_df
    ):
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        day_nums = [s.day_num for s in snapshots]
        assert day_nums == sorted(day_nums)

    def test_interval_7_correct_snapshot_count(
        self, constructor, sample_games_df, sample_node_df
    ):
        """
        Unique days: 1, 5, 10, 15, 20.
        Interval=7: day 1, day 8 (→next >= 1+7=8 is 10), day 15 (next >= 10+7=17 is 20), 20.
        With 'always include final day', and interval-based stepping:
        snaps at days that are multiples of interval from the first or based on stepping.
        The spec says 'build one snapshot every N days (e.g. days 1, 8, 15, ...)'.
        Implementation should produce at least 2 snapshots and include the final day.
        """
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        assert len(snapshots) >= 2

    def test_last_snapshot_is_final_day(
        self, constructor, sample_games_df, sample_node_df
    ):
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        final_day = sample_games_df["DayNum"].max()
        assert snapshots[-1].day_num == final_day

    def test_interval_1_produces_snapshot_per_unique_day(
        self, constructor, sample_games_df, sample_node_df
    ):
        """Interval=1 should produce one snapshot for each unique DayNum."""
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=1,
        )
        unique_days = sorted(sample_games_df["DayNum"].unique())
        snap_days = [s.day_num for s in snapshots]
        assert snap_days == unique_days

    def test_all_snapshots_are_temporal_snapshot_instances(
        self, constructor, sample_games_df, sample_node_df
    ):
        from src.graph.graph_constructor import TemporalSnapshot

        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        assert all(isinstance(s, TemporalSnapshot) for s in snapshots)

    def test_later_snapshots_have_gte_edges_than_earlier(
        self, constructor, sample_games_df, sample_node_df
    ):
        """Later PIT snapshots accumulate more games → at least as many edges."""
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=1,
        )
        for i in range(1, len(snapshots)):
            assert snapshots[i].num_games >= snapshots[i - 1].num_games

    def test_default_snapshot_interval_is_7(
        self, constructor, sample_games_df, sample_node_df
    ):
        """Calling without snapshot_interval kwarg should default to 7."""
        snapshots_default = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
        )
        snapshots_explicit = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=7,
        )
        assert len(snapshots_default) == len(snapshots_explicit)

    def test_large_interval_still_includes_final_day(
        self, constructor, sample_games_df, sample_node_df
    ):
        """Even with interval=999 (larger than all DayNums), final day is included."""
        snapshots = constructor.build_season_snapshots(
            games_df=sample_games_df,
            node_df=sample_node_df,
            snapshot_interval=999,
        )
        final_day = sample_games_df["DayNum"].max()
        assert snapshots[-1].day_num == final_day


# ---------------------------------------------------------------------------
# TestToPygData — PyG Data conversion with mocked torch_geometric
# ---------------------------------------------------------------------------

class TestToPygData:
    """Tests for GraphConstructor.to_pyg_data() with mocked torch_geometric."""

    def _make_mock_pyg(self):
        """Return a mock torch_geometric.data.Data class."""
        mock_data_cls = MagicMock()
        mock_data_instance = MagicMock()
        mock_data_cls.return_value = mock_data_instance

        mock_pyg_module = MagicMock()
        mock_pyg_module.data.Data = mock_data_cls
        return mock_pyg_module, mock_data_cls, mock_data_instance

    def test_to_pyg_data_calls_data_constructor(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            result = constructor.to_pyg_data(snapshot_day10)

        mock_data_cls.assert_called_once()

    def test_to_pyg_data_passes_node_features_as_x(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            constructor.to_pyg_data(snapshot_day10)

        _, kwargs = mock_data_cls.call_args
        assert "x" in kwargs
        assert torch.equal(kwargs["x"], snapshot_day10.node_features)

    def test_to_pyg_data_passes_edge_index(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            constructor.to_pyg_data(snapshot_day10)

        _, kwargs = mock_data_cls.call_args
        assert "edge_index" in kwargs
        assert torch.equal(kwargs["edge_index"], snapshot_day10.edge_index)

    def test_to_pyg_data_passes_edge_attr(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            constructor.to_pyg_data(snapshot_day10)

        _, kwargs = mock_data_cls.call_args
        assert "edge_attr" in kwargs
        assert torch.equal(kwargs["edge_attr"], snapshot_day10.edge_attr)

    def test_to_pyg_data_edge_index_is_long_tensor(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            constructor.to_pyg_data(snapshot_day10)

        _, kwargs = mock_data_cls.call_args
        assert kwargs["edge_index"].dtype == torch.long

    def test_to_pyg_data_raises_import_error_if_pyg_missing(
        self, constructor, snapshot_day10
    ):
        """When torch_geometric is not installed, raise ImportError with helpful message."""
        # Remove torch_geometric from sys.modules to simulate missing package
        original = sys.modules.pop("torch_geometric", None)
        original_data = sys.modules.pop("torch_geometric.data", None)
        try:
            with patch.dict(sys.modules, {"torch_geometric": None, "torch_geometric.data": None}):
                with pytest.raises((ImportError, ModuleNotFoundError)):
                    constructor.to_pyg_data(snapshot_day10)
        finally:
            if original is not None:
                sys.modules["torch_geometric"] = original
            if original_data is not None:
                sys.modules["torch_geometric.data"] = original_data

    def test_to_pyg_data_returns_data_instance(
        self, constructor, snapshot_day10
    ):
        mock_pyg, mock_data_cls, mock_instance = self._make_mock_pyg()

        with patch.dict(sys.modules, {"torch_geometric": mock_pyg, "torch_geometric.data": mock_pyg.data}):
            result = constructor.to_pyg_data(snapshot_day10)

        assert result is mock_instance


# ---------------------------------------------------------------------------
# TestGraphConstructorInit — constructor attribute storage
# ---------------------------------------------------------------------------

class TestGraphConstructorInit:
    """Tests for GraphConstructor.__init__() attribute storage."""

    def test_stores_season(self, season, node_feature_cols, edge_feature_cols):
        from src.graph.graph_constructor import GraphConstructor

        gc = GraphConstructor(season, node_feature_cols, edge_feature_cols)
        assert gc.season == season

    def test_stores_node_feature_cols(self, season, node_feature_cols, edge_feature_cols):
        from src.graph.graph_constructor import GraphConstructor

        gc = GraphConstructor(season, node_feature_cols, edge_feature_cols)
        assert gc.node_feature_cols == node_feature_cols

    def test_stores_edge_feature_cols(self, season, node_feature_cols, edge_feature_cols):
        from src.graph.graph_constructor import GraphConstructor

        gc = GraphConstructor(season, node_feature_cols, edge_feature_cols)
        assert gc.edge_feature_cols == edge_feature_cols
