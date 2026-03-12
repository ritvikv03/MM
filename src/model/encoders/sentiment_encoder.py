"""
src/model/encoders/sentiment_encoder.py

Lightweight attention-based sentiment encoder that converts news/injury alert
dicts (as produced by src/data/news_scraper.py) into 8-dimensional latent
vectors per team.

Design
------
No HTTP calls, no pretrained LLM.  Pure numpy + stdlib.

Algorithm
---------
For each team, matching alerts are scored by:
  1. Keyword severity from SEVERITY_WEIGHTS.
  2. Recency decay: w = exp(-hours_since_alert / recency_decay_tau).
     Alerts with no timestamp receive weight 1.0.
  3. Weighted aggregation into an 8-channel vector:
       vec[0] = pressure = sum(sev * w) / max(1, n)
       vec[1] = n_alerts / 10.0
       vec[2] = max_severity (or 0)
       vec[3] = sum(recency_weights) / 10.0
       vec[4] = 1.0 if any keyword in {"torn", "out for season", "out"} else 0.0
       vec[5] = n_suspension / max(1, n)  — suspension fraction
       vec[6] = n_injury / max(1, n)      — physical injury fraction
       vec[7] = pressure * vec[4]         — pressure × availability interaction
  All values clipped to [0, 1].

No bare `import torch` anywhere in this module.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS: Dict[str, float] = {
    "walking boot":   0.70,
    "sprain":         0.60,
    "suspension":     0.80,
    "mri":            0.75,
    "torn":           1.00,
    "concussion":     0.90,
    "out for season": 1.00,
    "not at practice": 0.65,
    "questionable":   0.40,
    "doubtful":       0.70,
    "out":            0.85,
    "injured":        0.55,
    "knee":           0.65,
    "ankle":          0.50,
}

LATENT_DIM: int = 8

# Keywords that indicate the player is unavailable (availability flag, vec[4])
_AVAILABILITY_KEYWORDS = frozenset({"torn", "out for season", "out"})

# Keywords that count as suspensions (vec[5])
_SUSPENSION_KEYWORDS = frozenset({"suspension"})

# Keywords that count as physical injuries (vec[6])
_INJURY_KEYWORDS = frozenset({
    "walking boot", "sprain", "mri", "torn", "concussion",
    "out for season", "not at practice", "injured", "knee", "ankle",
    "doubtful", "questionable", "out",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string to a timezone-aware datetime.
    Returns None on parse failure."""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _hours_since(ts_str: str, reference_time: Optional[datetime] = None) -> Optional[float]:
    """Return hours between timestamp and reference_time (default: now UTC).
    Returns None if timestamp cannot be parsed."""
    dt = _parse_timestamp(ts_str)
    if dt is None:
        return None
    ref = reference_time if reference_time is not None else datetime.now(timezone.utc)
    delta = ref - dt
    return max(0.0, delta.total_seconds() / 3600.0)


# ---------------------------------------------------------------------------
# SentimentEncoder
# ---------------------------------------------------------------------------

class SentimentEncoder:
    """Lightweight keyword-severity + recency-decay encoder.

    Converts a list of injury/news alert dicts into per-team 8-dimensional
    latent vectors encoding injury pressure.

    Parameters
    ----------
    latent_dim:
        Output dimensionality.  Default 8.  When latent_dim > 8, the extra
        channels are filled with zeros (sparse representation).
    recency_decay_tau_hours:
        Exponential decay time constant (tau) in hours.  Default 24.0.
        Weight formula: w = exp(-hours_since_alert / tau).
        Weight at t=tau is exp(-1) ≈ 0.368.
        Must be > 0.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        recency_decay_tau_hours: float = 24.0,
    ) -> None:
        if recency_decay_tau_hours <= 0:
            raise ValueError(
                f"recency_decay_tau_hours must be > 0, got {recency_decay_tau_hours}."
            )
        self.latent_dim = latent_dim
        self.recency_decay_tau_hours = recency_decay_tau_hours

    # ------------------------------------------------------------------
    # encode_single_team
    # ------------------------------------------------------------------

    def encode_single_team(
        self,
        team_name: str,
        alerts: List[dict],
        reference_time: Optional[str] = None,
    ) -> np.ndarray:
        """Encode all alerts for *team_name* into a latent vector.

        Parameters
        ----------
        team_name:
            The team whose alerts should be aggregated.
        alerts:
            List of alert dicts from news_scraper.  Each should have at
            minimum ``"team"`` and ``"keyword"`` keys.  ``"timestamp"`` is
            optional; missing timestamps receive recency weight 1.0.
        reference_time:
            ISO-8601 string used as "now" for recency decay computation.
            If None, the actual wall-clock UTC time is used.

        Returns
        -------
        np.ndarray shape (latent_dim,) clipped to [0, 1].
        """
        ref_dt: Optional[datetime] = None
        if reference_time is not None:
            ref_dt = _parse_timestamp(reference_time)

        # Filter alerts that match this team
        team_alerts = [a for a in alerts if a.get("team") == team_name]

        vec = np.zeros(self.latent_dim, dtype=float)
        if not team_alerts:
            return vec

        n = len(team_alerts)
        severities: List[float] = []
        recency_weights: List[float] = []
        availability_triggered = False
        n_suspension = 0
        n_injury = 0

        for alert in team_alerts:
            keyword = alert.get("keyword", "")
            sev = SEVERITY_WEIGHTS.get(keyword, 0.5)
            severities.append(sev)

            # Recency weight
            ts_str = alert.get("timestamp")
            if ts_str is not None:
                hours = _hours_since(ts_str, ref_dt)
                if hours is not None:
                    w = math.exp(-hours / self.recency_decay_tau_hours)
                else:
                    w = 1.0
            else:
                w = 1.0
            recency_weights.append(w)

            # Availability flag
            if keyword in _AVAILABILITY_KEYWORDS:
                availability_triggered = True

            # Category counts
            if keyword in _SUSPENSION_KEYWORDS:
                n_suspension += 1
            if keyword in _INJURY_KEYWORDS:
                n_injury += 1

        # Compute aggregate scalars
        pressure = sum(s * w for s, w in zip(severities, recency_weights)) / max(1, n)
        max_sev = max(severities)
        recency_mass = sum(recency_weights)

        # Fill the 8 canonical channels
        vec[0] = pressure
        vec[1] = n / 10.0
        vec[2] = max_sev
        vec[3] = recency_mass / 10.0
        vec[4] = 1.0 if availability_triggered else 0.0
        vec[5] = n_suspension / max(1, n)
        vec[6] = n_injury / max(1, n)
        vec[7] = pressure * vec[4]

        # For latent_dim > 8, extra channels remain zero (sparse extension)
        # Clip all to [0, 1]
        vec[:8] = np.clip(vec[:8], 0.0, 1.0)
        if self.latent_dim > 8:
            vec[8:] = 0.0

        return vec

    # ------------------------------------------------------------------
    # encode_alerts
    # ------------------------------------------------------------------

    def encode_alerts(
        self,
        alerts: List[dict],
        reference_time: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Encode all alerts, returning a dict keyed by team name.

        Parameters
        ----------
        alerts:
            List of alert dicts.
        reference_time:
            ISO-8601 string used as "now" for recency decay.

        Returns
        -------
        dict[team_name -> np.ndarray (latent_dim,)]
        """
        # Collect unique team names that appear in alerts
        teams = list({a.get("team") for a in alerts if a.get("team")})
        result: Dict[str, np.ndarray] = {}
        for team in teams:
            result[team] = self.encode_single_team(team, alerts, reference_time)
        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def encode_team_matrix(
    teams: List[str],
    alerts: List[dict],
    latent_dim: int = LATENT_DIM,
    recency_decay_tau_hours: float = 24.0,
    reference_time: Optional[str] = None,
) -> np.ndarray:
    """Encode a fixed list of teams into a matrix of shape (len(teams), latent_dim).

    Teams with no matching alerts receive a zero row (graceful degradation).

    Parameters
    ----------
    teams:
        Ordered list of team names defining row order in the output matrix.
    alerts:
        List of alert dicts.
    latent_dim:
        Output dimensionality per team.  Default 8.
    recency_decay_tau_hours:
        Exponential decay time constant (tau) in hours.  Weight at t=tau is
        exp(-1) ≈ 0.368.
    reference_time:
        ISO-8601 string used as "now".

    Returns
    -------
    np.ndarray shape (len(teams), latent_dim).
    """
    enc = SentimentEncoder(
        latent_dim=latent_dim,
        recency_decay_tau_hours=recency_decay_tau_hours,
    )
    matrix = np.zeros((len(teams), latent_dim), dtype=float)
    for i, team in enumerate(teams):
        matrix[i] = enc.encode_single_team(team, alerts, reference_time)
    return matrix
