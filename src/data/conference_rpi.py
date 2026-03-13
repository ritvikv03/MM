"""Conference RPI computation for heterogeneous graph Conference nodes.

RPI formula:
    rpi_score = 0.6 * mean(adj_em) + 0.4 * mean(sos)

Tiers (1 = strongest, 5 = weakest) are assigned via quintile cut.
These scores are injected as Conference node features in the GAT graph
to enable cross-conference strength modeling (e.g., an average Big 12
team vs. a dominant MAC team) — see CLAUDE.md §2 Graph Processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ConferenceRPI:
    """Per-conference RPI summary for GAT node feature injection."""
    conference: str
    mean_adj_em: float
    mean_sos: float
    rpi_score: float
    n_teams: int = 0
    tier: int = 0  # assigned by assign_rpi_tiers(); 0 = unassigned


def compute_conference_rpi(teams_df: pd.DataFrame) -> list[ConferenceRPI]:
    """Compute per-conference RPI from a T-Rank efficiency DataFrame.

    Args:
        teams_df: DataFrame with columns ['conference', 'adj_em', 'sos'].

    Returns:
        List of ConferenceRPI sorted by rpi_score descending.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"conference", "adj_em", "sos"}
    missing = required - set(teams_df.columns)
    if missing:
        raise ValueError(f"teams_df missing columns: {missing}")

    records: list[ConferenceRPI] = []
    for conf, group in teams_df.groupby("conference"):
        mean_em  = float(group["adj_em"].mean())
        mean_sos = float(group["sos"].mean())
        rpi      = round(0.6 * mean_em + 0.4 * mean_sos, 3)
        records.append(
            ConferenceRPI(
                conference=str(conf),
                mean_adj_em=round(mean_em, 3),
                mean_sos=round(mean_sos, 3),
                rpi_score=rpi,
                n_teams=len(group),
            )
        )
    return sorted(records, key=lambda r: r.rpi_score, reverse=True)


def assign_rpi_tiers(rpis: list[ConferenceRPI]) -> list[ConferenceRPI]:
    """Assign tier 1–5 (1 = strongest) via quintile cut.

    Mutates each ConferenceRPI.tier in place and returns the same list.
    Tier 1 = top 20% by rpi_score; Tier 5 = bottom 20%.
    """
    if not rpis:
        return rpis
    scores   = np.array([r.rpi_score for r in rpis])
    cutoffs  = np.percentile(scores, [80, 60, 40, 20])
    for rpi_obj in rpis:
        s = rpi_obj.rpi_score
        if s >= cutoffs[0]:
            rpi_obj.tier = 1
        elif s >= cutoffs[1]:
            rpi_obj.tier = 2
        elif s >= cutoffs[2]:
            rpi_obj.tier = 3
        elif s >= cutoffs[3]:
            rpi_obj.tier = 4
        else:
            rpi_obj.tier = 5
    return rpis
