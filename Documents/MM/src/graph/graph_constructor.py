"""
src/graph/graph_constructor.py

Top-level orchestrator for NCAA March Madness ST-GNN graph construction.

Assembles node features, edge features, and temporal snapshots into
PyG-compatible Data objects. Point-in-Time (PIT) integrity is enforced
throughout: no future data ever leaks into a past snapshot.

Usage example
-------------
>>> from src.graph.graph_constructor import GraphConstructor
>>> gc = GraphConstructor(
...     season=2024,
...     node_feature_cols=["adj_o", "adj_d", "tempo"],
...     edge_feature_cols=["score_diff", "location_enc", "rest_disparity"],
... )
>>> snapshots = gc.build_season_snapshots(games_df, node_df, snapshot_interval=7)
>>> pyg_data = gc.to_pyg_data(snapshots[-1])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Only imported for type hints; never at runtime from this block.
    import torch


# ---------------------------------------------------------------------------
# TemporalSnapshot dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalSnapshot:
    """A single point-in-time graph snapshot of the season up to *day_num*.

    Attributes
    ----------
    day_num:
        The NCAA calendar day through which games are included (PIT cutoff).
    season:
        The season year (e.g. 2024).
    node_features:
        Float tensor of shape ``(N, F_node)`` — one row per team.
    edge_index:
        Long tensor of shape ``(2, E)`` in COO format.
        Row 0 = winning-team node indices; Row 1 = losing-team node indices.
    edge_attr:
        Float tensor of shape ``(E, F_edge)`` — one row per game/edge.
    team_index:
        Mapping from team name (str) to node index (int), assigned
        alphabetically so that the mapping is deterministic across calls.
    num_games:
        Number of game-edges present in this snapshot (equals ``E``).
    """

    day_num: int
    season: int
    node_features: "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_attr: "torch.Tensor"
    team_index: dict[str, int]
    num_games: int


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def filter_games_pit(games_df: pd.DataFrame, day_num: int) -> pd.DataFrame:
    """Return only rows from *games_df* where ``DayNum <= day_num``.

    Parameters
    ----------
    games_df:
        DataFrame that must contain a ``DayNum`` integer column.
    day_num:
        Inclusive upper-bound cutoff day.

    Returns
    -------
    pd.DataFrame
        Filtered copy (same columns, subset of rows).

    Raises
    ------
    ValueError
        If *games_df* does not have a ``DayNum`` column.
    """
    if "DayNum" not in games_df.columns:
        raise ValueError(
            "games_df is missing required column 'DayNum'. "
            "Ensure the DataFrame comes from the Kaggle game-results loader."
        )
    return games_df[games_df["DayNum"] <= day_num].copy()


# ---------------------------------------------------------------------------
# GraphConstructor
# ---------------------------------------------------------------------------

class GraphConstructor:
    """Assembles PyG-compatible temporal graph snapshots from season game data.

    Parameters
    ----------
    season:
        The NCAA season year (used to tag snapshots).
    node_feature_cols:
        Column names in *node_df* (passed to ``build_snapshot``) that form
        each team's node feature vector.
    edge_feature_cols:
        Column names in the games DataFrame that form each edge's feature
        vector (e.g. ``score_diff``, ``location_enc``, ``rest_disparity``).
    """

    def __init__(
        self,
        season: int,
        node_feature_cols: list[str],
        edge_feature_cols: list[str],
    ) -> None:
        self.season = season
        self.node_feature_cols = node_feature_cols
        self.edge_feature_cols = edge_feature_cols

    # ------------------------------------------------------------------
    # build_snapshot
    # ------------------------------------------------------------------

    def build_snapshot(
        self,
        games_up_to_day: pd.DataFrame,
        node_df: pd.DataFrame,
        day_num: int,
    ) -> TemporalSnapshot:
        """Build a single ``TemporalSnapshot`` for games with DayNum <= *day_num*.

        The method re-applies the PIT filter internally so callers may pass
        any superset of games; rows beyond *day_num* are silently dropped.

        Parameters
        ----------
        games_up_to_day:
            DataFrame of games (may be pre-filtered or the full season).
            Must contain ``DayNum``, ``WTeamID``, ``LTeamID``, and all
            columns listed in ``self.edge_feature_cols``.
        node_df:
            Pre-built node features.  Must have a ``"team"`` column plus all
            columns listed in ``self.node_feature_cols``.  Teams present in
            games but absent from *node_df* receive zero feature vectors.
        day_num:
            The inclusive PIT cutoff day.

        Returns
        -------
        TemporalSnapshot
        """
        import torch  # lazy import

        # ---- 1. Apply PIT filter ----------------------------------------
        if "DayNum" in games_up_to_day.columns:
            games = filter_games_pit(games_up_to_day, day_num)
        else:
            games = games_up_to_day.copy()

        # ---- 2. Collect all teams in alphabetical order -----------------
        if len(games) == 0:
            teams: list[str] = []
        else:
            winning_teams = games["WTeamID"].tolist() if "WTeamID" in games.columns else []
            losing_teams  = games["LTeamID"].tolist() if "LTeamID" in games.columns else []
            teams = sorted(set(winning_teams) | set(losing_teams))

        team_index: dict[str, int] = {name: idx for idx, name in enumerate(teams)}
        N = len(teams)
        F_node = len(self.node_feature_cols)

        # ---- 3. Build node feature matrix (N, F_node) -------------------
        node_feat_matrix = torch.zeros((N, F_node), dtype=torch.float32)

        if N > 0 and len(node_df) > 0 and "team" in node_df.columns:
            # Index node_df by team name for O(1) lookup
            node_df_indexed = node_df.set_index("team")
            for team_name, node_idx in team_index.items():
                if team_name in node_df_indexed.index:
                    row = node_df_indexed.loc[team_name]
                    for feat_idx, col in enumerate(self.node_feature_cols):
                        if col in node_df_indexed.columns:
                            node_feat_matrix[node_idx, feat_idx] = float(row[col])
                # Teams not found in node_df keep their zero row (already initialised)

        # ---- 4. Build edge_index (2, E) and edge_attr (E, F_edge) -------
        F_edge = len(self.edge_feature_cols)

        if len(games) == 0 or "WTeamID" not in games.columns:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, F_edge), dtype=torch.float32)
            num_games  = 0
        else:
            E = len(games)
            src_indices: list[int] = []
            dst_indices: list[int] = []

            for _, row in games.iterrows():
                src_indices.append(team_index[row["WTeamID"]])
                dst_indices.append(team_index[row["LTeamID"]])

            edge_index = torch.tensor(
                [src_indices, dst_indices], dtype=torch.long
            )

            # Edge attributes — pull the configured columns from games
            edge_rows: list[list[float]] = []
            for _, row in games.iterrows():
                feat_row = [float(row[col]) for col in self.edge_feature_cols]
                edge_rows.append(feat_row)

            edge_attr = torch.tensor(edge_rows, dtype=torch.float32)
            num_games = E

        return TemporalSnapshot(
            day_num=day_num,
            season=self.season,
            node_features=node_feat_matrix,
            edge_index=edge_index,
            edge_attr=edge_attr,
            team_index=team_index,
            num_games=num_games,
        )

    # ------------------------------------------------------------------
    # build_season_snapshots
    # ------------------------------------------------------------------

    def build_season_snapshots(
        self,
        games_df: pd.DataFrame,
        node_df: pd.DataFrame,
        snapshot_interval: int = 7,
    ) -> list[TemporalSnapshot]:
        """Build a time-ordered sequence of ``TemporalSnapshot`` objects.

        Snapshots are built at every *snapshot_interval* days starting from
        the first game day; the final game day is always included as the last
        snapshot (even if it falls inside an interval).

        Parameters
        ----------
        games_df:
            Full season games DataFrame with a ``DayNum`` column.
        node_df:
            Pre-built node features (forwarded to ``build_snapshot``).
        snapshot_interval:
            Number of days between consecutive snapshots.  Defaults to 7.

        Returns
        -------
        list[TemporalSnapshot]
            Snapshots sorted by ``day_num`` ascending.
        """
        if "DayNum" not in games_df.columns:
            raise ValueError(
                "games_df is missing required column 'DayNum'."
            )

        unique_days = sorted(games_df["DayNum"].unique())
        if not unique_days:
            return []

        # Determine which days to build snapshots for.
        # Step through unique_days, picking one at every `snapshot_interval`.
        selected_days: list[int] = []
        next_threshold = unique_days[0]  # start from the first game day

        for day in unique_days:
            if day >= next_threshold:
                selected_days.append(day)
                next_threshold = day + snapshot_interval

        # Always include the final day
        final_day = unique_days[-1]
        if selected_days[-1] != final_day:
            selected_days.append(final_day)

        # Build each snapshot (PIT: pass all games; build_snapshot filters)
        snapshots: list[TemporalSnapshot] = []
        for day in selected_days:
            games_pit = filter_games_pit(games_df, day)
            snap = self.build_snapshot(
                games_up_to_day=games_pit,
                node_df=node_df,
                day_num=day,
            )
            snapshots.append(snap)

        return snapshots

    # ------------------------------------------------------------------
    # to_pyg_data
    # ------------------------------------------------------------------

    def to_pyg_data(self, snapshot: TemporalSnapshot) -> "torch_geometric.data.Data":
        """Convert a ``TemporalSnapshot`` to a ``torch_geometric.data.Data`` object.

        Parameters
        ----------
        snapshot:
            The snapshot to convert.

        Returns
        -------
        torch_geometric.data.Data
            PyG ``Data`` object with ``x``, ``edge_index``, and ``edge_attr``
            attributes set.

        Raises
        ------
        ImportError
            If ``torch_geometric`` is not installed.  The error message
            includes the pip install command to resolve it.
        """
        try:
            import torch_geometric.data as pyg_data_module  # lazy import
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "torch_geometric is required for to_pyg_data() but is not installed. "
                "Install it with: pip install torch-geometric\n"
                "See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html "
                "for version-specific installation instructions."
            ) from exc

        import torch  # lazy import

        # Ensure edge_index is int64 (LongTensor) as required by PyG
        edge_index = snapshot.edge_index.to(torch.long)

        return pyg_data_module.Data(
            x=snapshot.node_features,
            edge_index=edge_index,
            edge_attr=snapshot.edge_attr,
        )
