"""
src/graph/edge_features.py

Edge feature construction for the NCAA March Madness ST-GNN.

Each directed edge represents one game played:
    direction:  winning team  →  losing team
    (for in-progress seasons: home team → away team)

Edge feature columns produced
──────────────────────────────────────────────────────────────
margin              : WScore - LScore  (always positive)
total_points        : WScore + LScore
court_home          : 1 if WLoc == "H" else 0
court_away          : 1 if WLoc == "A" else 0  (upset-bonus signal)
court_neutral       : 1 if WLoc == "N" else 0  (tournament baseline)
w_rest_days         : days since winner's previous game; -1 = first game
l_rest_days         : days since loser's previous game; -1 = first game
rest_disparity      : w_rest_days - l_rest_days
ot_flag             : 1 if NumOT > 0 else 0

Travel fatigue (3-dim vector, 0.0 when coordinates unavailable):
  distance_miles      : great-circle distance (home campus → venue), miles
  time_zones_crossed  : |UTC_offset_delta|, discretized 0/1/2/3+
  elevation_flag      : 1 if venue elevation > 5,000 ft, else 0

PIT Integrity
─────────────
compute_rest_days is strictly causal: for game G on DayNum D, we look only at
games with DayNum < D within the same season to find a team's last appearance.
Same-day (DayNum == D) games are never used as prior games, preventing leakage
when multiple games share the same DayNum.
"""

from __future__ import annotations

import math

import pandas as pd
import numpy as np

__all__ = [
    "EdgeFeatureBuilder",
    "compute_rest_days",
    "compute_travel_fatigue",
    "encode_court_location",
    "to_edge_tensor",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_WLOC = {"H", "A", "N"}
_SENTINEL = -1  # first game of season has no prior game

# Elevation threshold (feet) for tournament venues that impose a meaningful
# altitude disadvantage — see CLAUDE.md §2 edge feature mandate.
_HIGH_ALTITUDE_FT = 5_000.0

# Earth radius for Haversine formula (miles).
_EARTH_RADIUS_MILES = 3_958.8


# ---------------------------------------------------------------------------
# Travel fatigue helpers
# ---------------------------------------------------------------------------


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two (lat, lon) points in degrees."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_MILES * c


def _longitude_to_utc_offset(lon: float) -> float:
    """Approximate UTC offset from longitude (every 15° = 1 hour)."""
    return lon / 15.0


# ---------------------------------------------------------------------------
# compute_travel_fatigue
# ---------------------------------------------------------------------------


def compute_travel_fatigue(
    games_df: pd.DataFrame,
    campus_coords: dict[int, tuple[float, float]] | None = None,
    venue_coords: dict[int, tuple[float, float]] | None = None,
    venue_elevation: dict[int, float] | None = None,
) -> pd.DataFrame:
    """Compute travel fatigue features for each game edge.

    For each game, computes a 3-dimensional travel fatigue vector for the
    **winning team** (the direction of the graph edge):

    - ``distance_miles``     : great-circle miles from the home campus of the
                               team assigned WTeamID to the game venue.
    - ``time_zones_crossed`` : absolute difference in approximate UTC offset
                               between home campus and venue, discretized to
                               the integer range 0–3 (capped at 3+).
    - ``elevation_flag``     : 1 if venue elevation > 5,000 ft, else 0.

    When coordinates/elevation are unavailable for a game, all three features
    default to 0.0 (graceful degradation, no errors raised).

    Parameters
    ----------
    games_df : pd.DataFrame
        Must contain at minimum: WTeamID, LTeamID.  A ``VenueID`` column is
        used to look up venue coordinates/elevation; if absent, falls back to
        0.0 for all venue-dependent features.
    campus_coords : dict[int, tuple[float, float]] | None
        Mapping team_id → (latitude_deg, longitude_deg) for each school's
        home campus.  When ``None``, ``distance_miles`` and
        ``time_zones_crossed`` default to 0.0.
    venue_coords : dict[int, tuple[float, float]] | None
        Mapping venue_id → (latitude_deg, longitude_deg) for tournament venues.
        Keyed by ``VenueID`` column in *games_df* when available.  When
        ``None`` or the row has no ``VenueID``, venue-dependent features are 0.
    venue_elevation : dict[int, float] | None
        Mapping venue_id → elevation in feet.  Used to compute
        ``elevation_flag``.  When ``None``, ``elevation_flag`` defaults to 0.

    Returns
    -------
    pd.DataFrame
        Copy of *games_df* with three new float columns appended:
        ``distance_miles``, ``time_zones_crossed``, ``elevation_flag``.
    """
    df = games_df.copy()

    distance_col: list[float] = []
    tz_col: list[float] = []
    elevation_col: list[float] = []

    has_venue_col = "VenueID" in df.columns

    for _, row in df.iterrows():
        w_team = int(row["WTeamID"])
        venue_id = int(row["VenueID"]) if has_venue_col else None

        dist = 0.0
        tz_cross = 0.0
        elev_flag = 0.0

        # --- Distance + time zones -----------------------------------------
        if (
            campus_coords is not None
            and w_team in campus_coords
            and venue_coords is not None
            and venue_id is not None
            and venue_id in venue_coords
        ):
            clat, clon = campus_coords[w_team]
            vlat, vlon = venue_coords[venue_id]
            dist = _haversine_miles(clat, clon, vlat, vlon)
            campus_tz = _longitude_to_utc_offset(clon)
            venue_tz = _longitude_to_utc_offset(vlon)
            tz_cross = float(min(abs(campus_tz - venue_tz), 3.0))

        # --- Elevation flag -------------------------------------------------
        if (
            venue_elevation is not None
            and venue_id is not None
            and venue_id in venue_elevation
        ):
            elev_flag = 1.0 if venue_elevation[venue_id] > _HIGH_ALTITUDE_FT else 0.0

        distance_col.append(dist)
        tz_col.append(tz_cross)
        elevation_col.append(elev_flag)

    df["distance_miles"] = distance_col
    df["time_zones_crossed"] = tz_col
    df["elevation_flag"] = elevation_col
    return df


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
    w_rest_days, l_rest_days, rest_disparity, ot_flag,
    distance_miles, time_zones_crossed, elevation_flag
    """

    def build(
        self,
        games_df: pd.DataFrame,
        campus_coords: dict | None = None,
        venue_coords: dict | None = None,
        venue_elevation: dict | None = None,
    ) -> pd.DataFrame:
        """
        Build all edge features from a game results DataFrame.

        Parameters
        ----------
        games_df : pd.DataFrame
            Raw game results with columns:
            ['Season','DayNum','WTeamID','WScore','LTeamID','LScore','WLoc','NumOT'].
            An optional 'VenueID' column enables travel fatigue computation.
        campus_coords : dict[int, tuple[float, float]] | None
            Mapping team_id → (lat, lon) for campus coordinates.  Required
            to compute ``distance_miles`` and ``time_zones_crossed``.
            Defaults to 0.0 when absent.
        venue_coords : dict[int, tuple[float, float]] | None
            Mapping venue_id → (lat, lon) for game venues.
        venue_elevation : dict[int, float] | None
            Mapping venue_id → elevation in feet.  Used for ``elevation_flag``.

        Returns
        -------
        pd.DataFrame
            New DataFrame (same rows) with original columns plus all edge
            feature columns including the 3-dim travel fatigue vector.
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

        # --- Travel fatigue (3-dim vector) -----------------------------------
        df = compute_travel_fatigue(
            df,
            campus_coords=campus_coords,
            venue_coords=venue_coords,
            venue_elevation=venue_elevation,
        )

        return df
