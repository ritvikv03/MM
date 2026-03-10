"""
src/graph/edge_features.py

Edge feature construction for the NCAA March Madness ST-GNN.

Each directed edge represents one game played:
    direction:  winning team  →  losing team
    (for in-progress seasons: home team → away team)

Edge feature columns produced
──────────────────────────────────────────────────────────────
margin          : WScore - LScore  (always positive)
total_points    : WScore + LScore
court_home      : 1 if WLoc == "H" else 0
court_away      : 1 if WLoc == "A" else 0  (upset-bonus signal)
court_neutral   : 1 if WLoc == "N" else 0  (tournament baseline)
w_rest_days     : days since winner's previous game in same season; -1 = first game
l_rest_days     : days since loser's previous game in same season; -1 = first game
rest_disparity  : w_rest_days - l_rest_days
ot_flag         : 1 if NumOT > 0 else 0

PIT Integrity
─────────────
compute_rest_days is strictly causal: for game G on DayNum D, we look only at
games with DayNum < D within the same season to find a team's last appearance.
Same-day (DayNum == D) games are never used as prior games, preventing leakage
when multiple games share the same DayNum.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

__all__ = [
    "EdgeFeatureBuilder",
    "compute_rest_days",
    "encode_court_location",
    "to_edge_tensor",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_WLOC = {"H", "A", "N"}
_SENTINEL = -1  # first game of season has no prior game


# ---------------------------------------------------------------------------
# encode_court_location
# ---------------------------------------------------------------------------

def encode_court_location(wloc_series: pd.Series) -> pd.DataFrame:
    """
    One-hot encode a Series of "H" / "A" / "N" court location codes.

    Parameters
    ----------
    wloc_series : pd.Series
        Series containing WLoc values ("H", "A", or "N").

    Returns
    -------
    pd.DataFrame
        Three-column DataFrame with columns
        ['court_home', 'court_away', 'court_neutral'], integer dtype,
        preserving the original Series index.

    Raises
    ------
    ValueError
        If any value in wloc_series is not in {"H", "A", "N"}.
    """
    unknown = set(wloc_series.dropna().unique()) - _VALID_WLOC
    if unknown:
        raise ValueError(
            f"Unknown WLoc value(s): {sorted(unknown)}. "
            f"Allowed values are {sorted(_VALID_WLOC)}."
        )

    index = wloc_series.index
    court_home    = (wloc_series == "H").astype(int)
    court_away    = (wloc_series == "A").astype(int)
    court_neutral = (wloc_series == "N").astype(int)

    return pd.DataFrame(
        {
            "court_home":    court_home.values,
            "court_away":    court_away.values,
            "court_neutral": court_neutral.values,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# compute_rest_days
# ---------------------------------------------------------------------------

def compute_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute point-in-time causal rest days for every team in every game.

    For each game (season, DayNum, WTeamID, LTeamID), we look only at games
    that occurred *strictly before* the current DayNum in the same season to
    find the last game day for each team.  Games on the same DayNum are
    excluded to prevent same-day leakage.

    First game of the season → rest days = -1 (sentinel).

    Parameters
    ----------
    games_df : pd.DataFrame
        Must contain columns: Season, DayNum, WTeamID, LTeamID.

    Returns
    -------
    pd.DataFrame
        Copy of games_df with two new integer columns appended:
        'w_rest_days' and 'l_rest_days'.
    """
    df = games_df.copy()

    # Build a per-team, per-season appearance log: {(season, team_id) → sorted list of DayNums}
    # We merge WTeam and LTeam appearances into a single structure.
    w_appearances = df[["Season", "DayNum", "WTeamID"]].rename(
        columns={"WTeamID": "TeamID"}
    )
    l_appearances = df[["Season", "DayNum", "LTeamID"]].rename(
        columns={"LTeamID": "TeamID"}
    )
    all_appearances = (
        pd.concat([w_appearances, l_appearances], ignore_index=True)
        .drop_duplicates()
        .sort_values(["Season", "TeamID", "DayNum"])
    )

    def _last_game_day_before(season: int, team_id: int, day_num: int) -> int:
        """
        Return the most recent DayNum strictly before *day_num* for this
        (season, team_id) pair, or -1 if none exists.
        """
        mask = (
            (all_appearances["Season"] == season)
            & (all_appearances["TeamID"] == team_id)
            & (all_appearances["DayNum"] < day_num)
        )
        prior = all_appearances.loc[mask, "DayNum"]
        if prior.empty:
            return _SENTINEL
        return int(prior.max())

    w_rest = []
    l_rest = []

    for _, row in df.iterrows():
        season  = row["Season"]
        day_num = row["DayNum"]
        w_team  = row["WTeamID"]
        l_team  = row["LTeamID"]

        last_w = _last_game_day_before(season, w_team, day_num)
        last_l = _last_game_day_before(season, l_team, day_num)

        w_rest.append(day_num - last_w if last_w != _SENTINEL else _SENTINEL)
        l_rest.append(day_num - last_l if last_l != _SENTINEL else _SENTINEL)

    df["w_rest_days"] = w_rest
    df["l_rest_days"] = l_rest
    return df


# ---------------------------------------------------------------------------
# to_edge_tensor
# ---------------------------------------------------------------------------

def to_edge_tensor(edge_df: pd.DataFrame, feature_cols: list[str]):
    """
    Convert selected columns of an edge DataFrame to a float32 PyTorch tensor.

    Torch is imported lazily so that modules without PyTorch installed can
    still use the rest of this file.

    Parameters
    ----------
    edge_df : pd.DataFrame
        DataFrame containing at minimum all columns listed in feature_cols.
    feature_cols : list[str]
        Ordered list of column names to extract.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (E, len(feature_cols)) where E = len(edge_df).
    """
    import torch  # lazy import

    values = edge_df[feature_cols].to_numpy(dtype=np.float32)
    return torch.tensor(values, dtype=torch.float32)


# ---------------------------------------------------------------------------
# EdgeFeatureBuilder
# ---------------------------------------------------------------------------

class EdgeFeatureBuilder:
    """
    High-level builder that transforms a raw game results DataFrame into an
    edge feature DataFrame suitable for use in a PyG graph.

    Usage
    -----
    >>> builder = EdgeFeatureBuilder()
    >>> edge_df = builder.build(games_df)

    Input columns expected in games_df
    ────────────────────────────────────
    Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT

    Output columns added (original columns are preserved)
    ──────────────────────────────────────────────────────
    margin, total_points, court_home, court_away, court_neutral,
    w_rest_days, l_rest_days, rest_disparity, ot_flag
    """

    def build(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build all edge features from a game results DataFrame.

        Parameters
        ----------
        games_df : pd.DataFrame
            Raw game results with columns:
            ['Season','DayNum','WTeamID','WScore','LTeamID','LScore','WLoc','NumOT']

        Returns
        -------
        pd.DataFrame
            New DataFrame (same rows) with original columns plus all edge
            feature columns.
        """
        df = games_df.copy()

        # --- Scalar features ------------------------------------------------
        df["margin"]       = df["WScore"] - df["LScore"]
        df["total_points"] = df["WScore"] + df["LScore"]
        df["ot_flag"]      = (df["NumOT"] > 0).astype(int)

        # --- Court location one-hot -----------------------------------------
        court_df = encode_court_location(df["WLoc"].reset_index(drop=True))
        # Re-attach with correct index
        court_df.index = df.index
        df["court_home"]    = court_df["court_home"]
        df["court_away"]    = court_df["court_away"]
        df["court_neutral"] = court_df["court_neutral"]

        # --- Rest days (PIT-causal) -----------------------------------------
        df = compute_rest_days(df)

        # --- Rest disparity -------------------------------------------------
        df["rest_disparity"] = df["w_rest_days"] - df["l_rest_days"]

        return df
