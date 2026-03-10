"""
src/graph/node_features.py
==========================
Node feature assembly for the NCAA March Madness ST-GNN.

Each graph node = one NCAA team for one season.  Features are assembled
from pre-ingested DataFrames that callers must supply as-of the game date
(Point-in-Time integrity: this module never fetches data).

Public API
----------
NodeFeatureBuilder          — assembles per-team feature rows
normalize_features(df, cols) -> pd.DataFrame   — min-max normalisation
to_tensor(df, cols)          -> torch.Tensor   — converts to float32 tensor
build_team_index(teams)      -> dict[str, int]  — deterministic team→index map

Design notes
------------
- No top-level torch import: torch is imported lazily inside to_tensor() so
  the module loads even when torch is not installed.
- All missing values are filled with 0.0 before effective_strength is computed
  to avoid NaN propagation.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_COLUMNS: List[str] = [
    "team_id",
    "season",
    "adj_em",
    "adj_o",
    "adj_d",
    "adj_t",
    "luck",
    "team_bpr_weighted",
    "rim_pct",
    "three_pct",
    "transition_pct",
    "efg",
    "availability",
    "roster_continuity",
    "effective_strength",
]

# Columns sourced from each input DataFrame.
_EFFICIENCY_FEATURE_COLS = ["adj_em", "adj_o", "adj_d", "adj_t", "luck"]
_BPR_FEATURE_COLS = ["team_bpr_weighted"]
_SHOT_FEATURE_COLS = ["rim_pct", "three_pct", "transition_pct", "efg"]


# ===========================================================================
# NodeFeatureBuilder
# ===========================================================================

class NodeFeatureBuilder:
    """
    Assembles node-level feature vectors for a single NCAA season.

    Parameters
    ----------
    season : int
        The season year (e.g. 2024 for the 2023-24 season).
    """

    def __init__(self, season: int) -> None:
        self.season = season

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def build(
        self,
        efficiency_df: pd.DataFrame,
        bpr_df: pd.DataFrame,
        shot_df: pd.DataFrame,
        roster_continuity: float,
        availability_vector: float,
    ) -> pd.DataFrame:
        """
        Assemble per-team feature DataFrame from pre-ingested source frames.

        Parameters
        ----------
        efficiency_df : pd.DataFrame
            Must contain columns: team_id, adj_em, adj_o, adj_d, adj_t, luck.
            Typically sourced from barttorvik or kenpom ingestion modules.
        bpr_df : pd.DataFrame
            Must contain columns: team_id, team_bpr_weighted.
            Sourced from evanmiya ingestion module.
        shot_df : pd.DataFrame
            Must contain columns: team_id, rim_pct, three_pct, transition_pct, efg.
            Sourced from hoopmath ingestion module.
        roster_continuity : float
            Scalar in [0.0, 1.0].  Fraction of possessions returning from last
            season; sourced from barttorvik returning-possession data.
        availability_vector : float
            Scalar in [0.0, 1.0].  Output of injury_feed.build_availability_vector().

        Returns
        -------
        pd.DataFrame
            One row per team with columns exactly matching _OUTPUT_COLUMNS.
            Missing join values are filled with 0.0 before effective_strength
            is computed.

        Notes
        -----
        PIT integrity: this method is a pure DataFrame transformation.
        It never performs HTTP requests or file I/O.
        """
        # Handle empty input gracefully.
        if efficiency_df.empty:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS)

        # 1. Start from efficiency data; it defines the universe of teams.
        result = efficiency_df[["team_id"] + _EFFICIENCY_FEATURE_COLS].copy()

        # 2. Left-join BPR data.
        bpr_subset = bpr_df[["team_id"] + _BPR_FEATURE_COLS].copy()
        result = result.merge(bpr_subset, on="team_id", how="left")

        # 3. Left-join shot data.
        shot_subset = shot_df[["team_id"] + _SHOT_FEATURE_COLS].copy()
        result = result.merge(shot_subset, on="team_id", how="left")

        # 4. Fill all remaining NaN values with 0.0.
        result = result.fillna(0.0)

        # 5. Attach scalar metadata.
        result["availability"] = float(availability_vector)
        result["roster_continuity"] = float(roster_continuity)
        result["season"] = self.season

        # 6. Compute effective_strength.
        #    effective_strength = adj_em * availability * (0.7 + 0.3 * roster_continuity)
        continuity_factor = 0.7 + 0.3 * float(roster_continuity)
        result["effective_strength"] = (
            result["adj_em"]
            * float(availability_vector)
            * continuity_factor
        )

        # 7. Return with guaranteed column order.
        return result[_OUTPUT_COLUMNS].reset_index(drop=True)


# ===========================================================================
# normalize_features
# ===========================================================================

def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Apply min-max normalisation per column to a copy of *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    feature_cols : list[str]
        Columns to normalise.  All other columns are passed through unchanged.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with *feature_cols* replaced by their min-max scaled
        values in [0.0, 1.0].  Constant columns (min == max) are set to 0.0.

    Raises
    ------
    ValueError
        If any column in *feature_cols* is not present in *df*.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalize_features: columns not found in DataFrame: {missing}"
        )

    out = df.copy()
    for col in feature_cols:
        col_min = out[col].min()
        col_max = out[col].max()
        if col_max == col_min:
            out[col] = 0.0
        else:
            out[col] = (out[col] - col_min) / (col_max - col_min)
    return out


# ===========================================================================
# to_tensor
# ===========================================================================

def to_tensor(df: pd.DataFrame, feature_cols: List[str]):
    """
    Convert selected DataFrame columns to a float32 torch Tensor.

    torch is imported lazily so this module loads without torch installed.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    torch.Tensor
        Shape (N, len(feature_cols)), dtype=torch.float32.

    Raises
    ------
    ValueError
        If any column in *feature_cols* is missing from *df*.
    """
    # Lazy import — torch is not required at module load time.
    import torch  # noqa: PLC0415

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"to_tensor: columns not found in DataFrame: {missing}"
        )

    array = df[feature_cols].to_numpy(dtype=np.float32)
    return torch.tensor(array, dtype=torch.float32)


# ===========================================================================
# build_team_index
# ===========================================================================

def build_team_index(teams: List[str]) -> dict:
    """
    Build a deterministic mapping from team name to integer index.

    Teams are sorted alphabetically before assignment so the mapping is
    stable regardless of insertion order.

    Parameters
    ----------
    teams : list[str]
        Any collection of team name strings (duplicates are allowed but
        unusual; the last duplicate's index is retained).

    Returns
    -------
    dict[str, int]
        Maps each unique team name to its 0-based index in the sorted list.
    """
    return {team: idx for idx, team in enumerate(sorted(teams))}
