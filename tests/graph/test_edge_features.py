"""
Tests for src/graph/edge_features.py

TDD RED phase — all tests must fail before implementation exists.

Coverage targets:
- EdgeFeatureBuilder.build()
- compute_rest_days()
- encode_court_location()
- to_edge_tensor()

At least 28 tests covering:
  - margin / total_points computation
  - court one-hot encoding (H / A / N)
  - unknown court value raises ValueError
  - OT flag detection
  - rest day first-game sentinel (-1)
  - rest day chaining across multiple sequential games
  - team appearing as both W and L in different games (cross-role tracking)
  - parallel games on same DayNum (PIT: use only DayNum < current)
  - rest_disparity sign and magnitude
  - multi-season isolation (rest days do not bleed across seasons)
  - output DataFrame shape and column completeness
  - tensor shape and dtype
  - edge cases: single-game DataFrame, all-neutral-court season
"""
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_games(**kwargs) -> pd.DataFrame:
    """
    Convenience builder.  Pass column arrays as kwargs; Season and NumOT
    default to 2024 / 0 if omitted.
    """
    n = len(next(iter(kwargs.values())))
    base = {
        "Season": [2024] * n,
        "NumOT":  [0] * n,
    }
    base.update(kwargs)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Imports — these will raise ImportError until implementation exists
# ---------------------------------------------------------------------------

from src.graph.edge_features import (
    EdgeFeatureBuilder,
    compute_rest_days,
    encode_court_location,
    to_edge_tensor,
)


# ===========================================================================
# encode_court_location
# ===========================================================================

class TestEncodeCourtLocation:
    """12 tests for the standalone encoder."""

    def test_home_encodes_correctly(self):
        result = encode_court_location(pd.Series(["H"]))
        assert result.loc[0, "court_home"] == 1
        assert result.loc[0, "court_away"] == 0
        assert result.loc[0, "court_neutral"] == 0

    def test_away_encodes_correctly(self):
        result = encode_court_location(pd.Series(["A"]))
        assert result.loc[0, "court_home"] == 0
        assert result.loc[0, "court_away"] == 1
        assert result.loc[0, "court_neutral"] == 0

    def test_neutral_encodes_correctly(self):
        result = encode_court_location(pd.Series(["N"]))
        assert result.loc[0, "court_home"] == 0
        assert result.loc[0, "court_away"] == 0
        assert result.loc[0, "court_neutral"] == 1

    def test_mixed_series_all_rows(self):
        result = encode_court_location(pd.Series(["H", "A", "N"]))
        assert list(result["court_home"])    == [1, 0, 0]
        assert list(result["court_away"])    == [0, 1, 0]
        assert list(result["court_neutral"]) == [0, 0, 1]

    def test_output_columns_present(self):
        result = encode_court_location(pd.Series(["H", "N"]))
        assert set(result.columns) == {"court_home", "court_away", "court_neutral"}

    def test_output_has_correct_length(self):
        series = pd.Series(["H", "A", "N", "H", "N"])
        result = encode_court_location(series)
        assert len(result) == 5

    def test_values_are_integers(self):
        result = encode_court_location(pd.Series(["H", "A", "N"]))
        for col in ["court_home", "court_away", "court_neutral"]:
            assert result[col].dtype in (np.int64, np.int32, int)

    def test_unknown_value_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown WLoc value"):
            encode_court_location(pd.Series(["H", "X"]))

    def test_lowercase_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown WLoc value"):
            encode_court_location(pd.Series(["h"]))

    def test_empty_series_returns_empty_dataframe(self):
        result = encode_court_location(pd.Series([], dtype=str))
        assert len(result) == 0
        assert set(result.columns) == {"court_home", "court_away", "court_neutral"}

    def test_all_neutral_encodes_neutral_column_all_ones(self):
        result = encode_court_location(pd.Series(["N", "N", "N"]))
        assert list(result["court_neutral"]) == [1, 1, 1]
        assert list(result["court_home"])    == [0, 0, 0]
        assert list(result["court_away"])    == [0, 0, 0]

    def test_index_is_preserved(self):
        series = pd.Series(["H", "N"], index=[10, 20])
        result = encode_court_location(series)
        assert list(result.index) == [10, 20]


# ===========================================================================
# compute_rest_days
# ===========================================================================

class TestComputeRestDays:
    """10 tests for PIT-correct rest day computation."""

    def test_first_game_winner_gets_sentinel(self):
        df = _make_games(
            DayNum=[5],
            WTeamID=[1], WScore=[70],
            LTeamID=[2], LScore=[60],
            WLoc=["H"],
        )
        result = compute_rest_days(df)
        assert result.loc[0, "w_rest_days"] == -1

    def test_first_game_loser_gets_sentinel(self):
        df = _make_games(
            DayNum=[5],
            WTeamID=[1], WScore=[70],
            LTeamID=[2], LScore=[60],
            WLoc=["H"],
        )
        result = compute_rest_days(df)
        assert result.loc[0, "l_rest_days"] == -1

    def test_sequential_games_same_team_as_winner(self):
        """Team 1 wins on day 5 then again on day 10 → 5 rest days."""
        df = _make_games(
            DayNum=[5, 10],
            WTeamID=[1, 1], WScore=[70, 75],
            LTeamID=[2, 3], LScore=[60, 65],
            WLoc=["H", "H"],
        )
        result = compute_rest_days(df)
        assert result.loc[1, "w_rest_days"] == 5

    def test_sequential_games_same_team_as_loser(self):
        """Team 2 loses on day 5 then loses again on day 12 → 7 rest days."""
        df = _make_games(
            DayNum=[5, 12],
            WTeamID=[1, 3], WScore=[70, 80],
            LTeamID=[2, 2], LScore=[60, 55],
            WLoc=["H", "H"],
        )
        result = compute_rest_days(df)
        assert result.loc[1, "l_rest_days"] == 7

    def test_team_wins_then_loses_tracks_across_roles(self):
        """
        Team 1 wins on day 5 as WTeam; then loses on day 9 as LTeam.
        l_rest_days for the second game should be 4.
        """
        df = _make_games(
            DayNum=[5, 9],
            WTeamID=[1, 2], WScore=[70, 80],
            LTeamID=[3, 1], LScore=[60, 75],
            WLoc=["H", "H"],
        )
        result = compute_rest_days(df)
        assert result.loc[1, "l_rest_days"] == 4

    def test_team_loses_then_wins_tracks_across_roles(self):
        """
        Team 2 loses on day 3 as LTeam; then wins on day 11 as WTeam.
        w_rest_days for the second game should be 8.
        """
        df = _make_games(
            DayNum=[3, 11],
            WTeamID=[1, 2], WScore=[70, 80],
            LTeamID=[2, 3], LScore=[60, 55],
            WLoc=["H", "H"],
        )
        result = compute_rest_days(df)
        assert result.loc[1, "w_rest_days"] == 8

    def test_parallel_games_same_daynumber_uses_prior_game(self):
        """
        Games on day 10 and day 10 for team 1 — PIT: both should use only
        info from DayNum < 10.  Team 1 appeared on day 5 → rest = 5 for
        the day-10 game; the other day-10 game where team 1 also appears
        should also see rest = 5 (not 0, which would imply same-day leakage).
        """
        df = _make_games(
            DayNum=[5, 10, 10],
            WTeamID=[1,  1,  4], WScore=[70, 75, 80],
            LTeamID=[2,  3,  1], LScore=[60, 65, 70],
            WLoc=["H", "H", "H"],
        )
        result = compute_rest_days(df)
        # Game at index 1: team 1 as winner on day 10 — last game was day 5
        assert result.loc[1, "w_rest_days"] == 5
        # Game at index 2: team 1 as loser on day 10 — last game was day 5
        assert result.loc[2, "l_rest_days"] == 5

    def test_output_has_w_and_l_rest_days_columns(self):
        df = _make_games(
            DayNum=[5],
            WTeamID=[1], WScore=[70],
            LTeamID=[2], LScore=[60],
            WLoc=["H"],
        )
        result = compute_rest_days(df)
        assert "w_rest_days" in result.columns
        assert "l_rest_days" in result.columns

    def test_output_row_count_unchanged(self):
        df = _make_games(
            DayNum=[5, 10, 15],
            WTeamID=[1, 2, 3], WScore=[70, 80, 65],
            LTeamID=[4, 5, 6], LScore=[60, 70, 55],
            WLoc=["H", "A", "N"],
        )
        result = compute_rest_days(df)
        assert len(result) == len(df)

    def test_seasons_do_not_bleed_rest_days(self):
        """
        Team 1 plays on day 30 of season 2023.  In season 2024 its first game
        is on day 5.  Rest days in 2024 game must be -1 (first of new season).
        """
        df = pd.DataFrame({
            "Season":  [2023, 2024],
            "DayNum":  [30,   5],
            "WTeamID": [1,    1],
            "WScore":  [70,   75],
            "LTeamID": [2,    3],
            "LScore":  [60,   65],
            "WLoc":    ["H",  "H"],
            "NumOT":   [0,    0],
        })
        result = compute_rest_days(df)
        season_2024_row = result[result["Season"] == 2024].iloc[0]
        assert season_2024_row["w_rest_days"] == -1


# ===========================================================================
# EdgeFeatureBuilder.build()
# ===========================================================================

class TestEdgeFeatureBuilder:
    """12 tests for the high-level builder."""

    @pytest.fixture
    def builder(self):
        return EdgeFeatureBuilder()

    @pytest.fixture
    def simple_df(self):
        return pd.DataFrame({
            "Season":  [2024, 2024, 2024],
            "DayNum":  [1,    2,    3],
            "WTeamID": [1101, 1102, 1103],
            "WScore":  [75,   80,   65],
            "LTeamID": [1201, 1202, 1203],
            "LScore":  [70,   60,   55],
            "WLoc":    ["H",  "A",  "N"],
            "NumOT":   [0,    1,    0],
        })

    def test_output_contains_all_required_columns(self, builder, simple_df):
        result = builder.build(simple_df)
        required = {
            "margin", "total_points",
            "court_home", "court_away", "court_neutral",
            "w_rest_days", "l_rest_days", "rest_disparity",
            "ot_flag",
        }
        assert required.issubset(set(result.columns))

    def test_row_count_preserved(self, builder, simple_df):
        result = builder.build(simple_df)
        assert len(result) == len(simple_df)

    def test_margin_computation(self, builder, simple_df):
        result = builder.build(simple_df)
        expected = simple_df["WScore"] - simple_df["LScore"]
        pd.testing.assert_series_equal(
            result["margin"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_margin_is_always_positive(self, builder, simple_df):
        result = builder.build(simple_df)
        assert (result["margin"] > 0).all()

    def test_total_points_computation(self, builder, simple_df):
        result = builder.build(simple_df)
        expected = simple_df["WScore"] + simple_df["LScore"]
        pd.testing.assert_series_equal(
            result["total_points"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_court_home_flag(self, builder, simple_df):
        result = builder.build(simple_df)
        assert list(result["court_home"])    == [1, 0, 0]
        assert list(result["court_away"])    == [0, 1, 0]
        assert list(result["court_neutral"]) == [0, 0, 1]

    def test_ot_flag_zero_when_no_ot(self, builder, simple_df):
        result = builder.build(simple_df)
        # rows 0 and 2 have NumOT == 0
        assert result.loc[0, "ot_flag"] == 0
        assert result.loc[2, "ot_flag"] == 0

    def test_ot_flag_one_when_ot(self, builder, simple_df):
        result = builder.build(simple_df)
        # row 1 has NumOT == 1
        assert result.loc[1, "ot_flag"] == 1

    def test_ot_flag_with_multiple_ot(self, builder):
        df = _make_games(
            DayNum=[5],
            WTeamID=[1], WScore=[90],
            LTeamID=[2], LScore=[85],
            WLoc=["N"],
            NumOT=[3],
        )
        result = builder.build(df)
        assert result.loc[0, "ot_flag"] == 1

    def test_rest_disparity_equals_w_minus_l(self, builder):
        """
        Three-game chain for team 1 (wins day 5, day 12) and team 2 (loses day 5,
        wins day 10).  Verify rest_disparity == w_rest_days - l_rest_days.
        """
        df = pd.DataFrame({
            "Season":  [2024, 2024, 2024],
            "DayNum":  [5,    10,   12],
            "WTeamID": [1,    2,    1],
            "WScore":  [70,   80,   75],
            "LTeamID": [3,    4,    5],
            "LScore":  [60,   70,   60],
            "WLoc":    ["H",  "A",  "N"],
            "NumOT":   [0,    0,    0],
        })
        result = builder.build(df)
        computed = result["w_rest_days"] - result["l_rest_days"]
        pd.testing.assert_series_equal(
            result["rest_disparity"].reset_index(drop=True),
            computed.reset_index(drop=True),
            check_names=False,
        )

    def test_first_game_sentinel_propagated_through_build(self, builder):
        df = _make_games(
            DayNum=[5],
            WTeamID=[99], WScore=[70],
            LTeamID=[100], LScore=[60],
            WLoc=["H"],
        )
        result = builder.build(df)
        assert result.loc[0, "w_rest_days"] == -1
        assert result.loc[0, "l_rest_days"] == -1

    def test_original_columns_preserved(self, builder, simple_df):
        """build() must not drop any input columns."""
        result = builder.build(simple_df)
        for col in simple_df.columns:
            assert col in result.columns


# ===========================================================================
# to_edge_tensor
# ===========================================================================

class TestToEdgeTensor:
    """4 tests for tensor conversion."""

    def _sample_edge_df(self):
        return pd.DataFrame({
            "margin":        [5.0, 10.0, 3.0],
            "total_points":  [145.0, 140.0, 120.0],
            "court_home":    [1, 0, 0],
            "court_away":    [0, 1, 0],
            "court_neutral": [0, 0, 1],
            "w_rest_days":   [-1, 5, 3],
            "l_rest_days":   [-1, 3, -1],
            "rest_disparity":[0, 2, 4],
            "ot_flag":       [0, 1, 0],
        })

    def test_tensor_shape(self):
        import torch
        df = self._sample_edge_df()
        cols = ["margin", "total_points", "court_home", "court_away",
                "court_neutral", "w_rest_days", "l_rest_days",
                "rest_disparity", "ot_flag"]
        t = to_edge_tensor(df, cols)
        assert t.shape == (3, 9)

    def test_tensor_dtype_is_float32(self):
        import torch
        df = self._sample_edge_df()
        cols = ["margin", "total_points"]
        t = to_edge_tensor(df, cols)
        assert t.dtype == torch.float32

    def test_tensor_values_match_dataframe(self):
        import torch
        df = self._sample_edge_df()
        cols = ["margin", "total_points"]
        t = to_edge_tensor(df, cols)
        expected = torch.tensor([[5.0, 145.0], [10.0, 140.0], [3.0, 120.0]],
                                dtype=torch.float32)
        assert torch.allclose(t, expected)

    def test_tensor_subset_of_columns(self):
        import torch
        df = self._sample_edge_df()
        cols = ["ot_flag"]
        t = to_edge_tensor(df, cols)
        assert t.shape == (3, 1)
