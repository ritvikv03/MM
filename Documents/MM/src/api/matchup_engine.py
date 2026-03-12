"""
src/api/matchup_engine.py
Real /api/matchup engine — Barttorvik efficiency delta + ADVI posterior samples.

Workflow
--------
1.  Load T-Rank DataFrame from the DataLoader cache.
2.  Resolve both teams via case-insensitive lookup.
3.  Compute efficiency margin delta + optional home-court advantage.
4.  Run a lightweight PyMC ADVI model (~3-5 s, 5000 iters) to draw 2000
    posterior spread / win-probability samples.
5.  On any PyMC failure, fall back to an analytical Normal approximation
    (deterministic, seeded with numpy RNG 42).

References
----------
- PyMC ADVI: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/variational_api_quickstart.html
- Closing Line Value spec: CLAUDE.md §3.5
"""
from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from src.api.schemas import MatchupResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOME_COURT_ADVANTAGE: float = 3.5   # points; 0 on neutral courts
_SCALE_FACTOR: float = 2.5           # divisor when converting EM delta → spread
_WIN_SPREAD_SCALE: float = 7.0       # logistic spread→win-probability denominator
_DEFAULT_SIGMA: float = 8.0          # base spread std-dev (points)
_LUCK_AMPLIFIER: float = 10.0        # multiplier on combined |luck| term
_N_SAMPLES: int = 2000
_ADVI_ITERS: int = 5000


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class MatchupNotFoundError(ValueError):
    """Raised when a team is not found in the T-Rank cache."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _lookup_team(name: str, trank_df: pd.DataFrame) -> pd.Series:
    """Return the first matching row for *name* in *trank_df*.

    Performs case-insensitive exact match first, then falls back to a
    case-insensitive substring (partial) match.

    Raises
    ------
    MatchupNotFoundError
        If neither strategy finds a result.
    """
    if trank_df.empty:
        raise MatchupNotFoundError(
            f"T-Rank cache is empty; cannot look up team '{name}'."
        )

    name_lower = name.strip().lower()
    col = trank_df["team"].str.strip().str.lower()

    # 1. Exact match (case-insensitive)
    exact = trank_df[col == name_lower]
    if not exact.empty:
        return exact.iloc[0]

    # 2. Partial / substring match
    partial = trank_df[col.str.contains(name_lower, regex=False, na=False)]
    if not partial.empty:
        return partial.iloc[0]

    raise MatchupNotFoundError(
        f"Team '{name}' not found in T-Rank cache. "
        f"Available teams: {trank_df['team'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Analytical fallback
# ---------------------------------------------------------------------------

def _analytical_fallback(
    delta: float,
    luck_home: float,
    luck_away: float,
) -> dict:
    """Pure NumPy Normal approximation — always succeeds.

    Parameters
    ----------
    delta:
        Expected spread (home perspective).
    luck_home, luck_away:
        Barttorvik luck values for each team.

    Returns
    -------
    dict with keys ``spread_samples`` (list[float]) and
    ``p_win_samples`` (list[float]).
    """
    rng = np.random.default_rng(seed=42)
    sigma = _DEFAULT_SIGMA + (abs(luck_home) + abs(luck_away)) * _LUCK_AMPLIFIER
    spread_samples: np.ndarray = rng.normal(delta, sigma, size=_N_SAMPLES)
    p_win_samples: np.ndarray = _sigmoid(spread_samples / _WIN_SPREAD_SCALE)
    return {
        "spread_samples": spread_samples.tolist(),
        "p_win_samples": p_win_samples.tolist(),
    }


# ---------------------------------------------------------------------------
# ADVI model
# ---------------------------------------------------------------------------

def _run_advi_matchup(
    delta: float,
    luck_home: float,
    luck_away: float,
) -> dict:
    """Attempt a lightweight PyMC ADVI model; fall back to analytical on failure.

    Model
    -----
    sigma_spread ~ HalfNormal(8.0)
    luck_sigma   ~ HalfNormal(max(|luck_home| + |luck_away|, 0.1))
    margin_obs   = pm.Normal("margin", mu=delta,
                             sigma=sigma_spread + luck_sigma,
                             observed=[delta])

    After fitting, draws *_N_SAMPLES* samples from the approximate posterior,
    then generates spread and win-probability arrays.

    Falls back to ``_analytical_fallback`` on any exception.
    """
    try:
        import pymc as pm  # noqa: PLC0415 — lazy import to keep tests fast

        luck_scale = max(abs(luck_home) + abs(luck_away), 0.1)

        with pm.Model():
            sigma_spread = pm.HalfNormal("sigma_spread", sigma=_DEFAULT_SIGMA)
            luck_sigma   = pm.HalfNormal("luck_sigma", sigma=luck_scale)
            _margin_obs  = pm.Normal(  # noqa: F841
                "margin",
                mu=delta,
                sigma=sigma_spread + luck_sigma,
                observed=[delta],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                approx = pm.fit(
                    n=_ADVI_ITERS,
                    method="advi",
                    progressbar=False,
                )

            trace = approx.sample(_N_SAMPLES)

        sigma_draws: np.ndarray = (
            np.asarray(trace.posterior["sigma_spread"]).flatten()
            + np.asarray(trace.posterior["luck_sigma"]).flatten()
        )
        # Ensure exactly _N_SAMPLES draws
        sigma_draws = sigma_draws[:_N_SAMPLES]
        if len(sigma_draws) < _N_SAMPLES:
            # Pad by repeating the mean
            pad = np.full(_N_SAMPLES - len(sigma_draws), sigma_draws.mean())
            sigma_draws = np.concatenate([sigma_draws, pad])

        rng = np.random.default_rng(seed=0)
        spread_samples: np.ndarray = rng.normal(delta, sigma_draws)
        p_win_samples: np.ndarray  = _sigmoid(spread_samples / _WIN_SPREAD_SCALE)

        return {
            "spread_samples": spread_samples.tolist(),
            "p_win_samples":  p_win_samples.tolist(),
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_run_advi_matchup failed (%s). Falling back to analytical approximation.",
            exc,
        )
        return _analytical_fallback(delta, luck_home, luck_away)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_real_matchup(
    home_team: str,
    away_team: str,
    season: int,
    neutral_site: bool,
    loader=None,
) -> MatchupResponse:
    """Build a real matchup response using Barttorvik T-Rank data.

    Parameters
    ----------
    home_team:
        Display name of the home (or first) team.
    away_team:
        Display name of the away (or second) team.
    season:
        NCAA season year (e.g., 2024).
    neutral_site:
        If ``True``, home-court advantage is set to 0.
    loader:
        Object with a ``get_trank(season)`` method returning a DataFrame.
        Defaults to a fresh ``DataLoader()`` instance.

    Returns
    -------
    MatchupResponse
        Contains 2000 posterior spread + win-probability samples, plus
        scalar summaries and metadata.

    Raises
    ------
    MatchupNotFoundError
        If the T-Rank cache is empty or a team cannot be resolved.
    """
    # ── 1. Resolve loader ────────────────────────────────────────────────────
    if loader is None:
        from src.api.data_cache import DataLoader  # noqa: PLC0415
        loader = DataLoader()

    trank_df: pd.DataFrame = loader.get_trank(season)

    if trank_df.empty:
        raise MatchupNotFoundError(
            f"T-Rank data is empty for season {season}. "
            "Check network access or pre-warm the cache."
        )

    # ── 2. Look up teams ─────────────────────────────────────────────────────
    home_row: pd.Series = _lookup_team(home_team, trank_df)
    away_row: pd.Series = _lookup_team(away_team, trank_df)

    # ── 3. Compute efficiency margins ────────────────────────────────────────
    home_em: float = float(home_row["adj_oe"]) - float(home_row["adj_de"])
    away_em: float = float(away_row["adj_oe"]) - float(away_row["adj_de"])

    home_adv: float = 0.0 if neutral_site else _HOME_COURT_ADVANTAGE

    # Delta: positive → home team is favoured
    delta: float = (home_em - away_em) / _SCALE_FACTOR + home_adv

    # ── 4. Luck values ───────────────────────────────────────────────────────
    luck_home: float = float(home_row.get("luck", 0.0))
    luck_away: float = float(away_row.get("luck", 0.0))

    # ── 5. Posterior samples ─────────────────────────────────────────────────
    samples: dict = _run_advi_matchup(delta, luck_home, luck_away)

    spread_arr = np.asarray(samples["spread_samples"])
    p_win_arr  = np.asarray(samples["p_win_samples"])

    # ── 6. Assemble response ─────────────────────────────────────────────────
    luck_compressed: bool = (abs(luck_home) + abs(luck_away)) > 0.10

    return MatchupResponse(
        home_team=home_team,
        away_team=away_team,
        p_win_home=float(np.mean(p_win_arr)),
        p_win_samples=[round(float(v), 6) for v in p_win_arr],
        spread_mean=float(np.mean(spread_arr)),
        spread_samples=[round(float(v), 6) for v in spread_arr],
        luck_compressed=luck_compressed,
        data_source="real",
    )
