"""Cache-first DataLoader for Barttorvik T-Rank and tournament seeds.

Fetches data once per season, writes to disk, and serves from cache on all
subsequent calls — preventing live scraping on every API request.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column normalization mapping
# ---------------------------------------------------------------------------

_COLUMN_ALIASES: dict[str, str] = {
    "adj_o": "adj_oe",
    "adj_d": "adj_de",
    "adj_t": "tempo",
    "team_name": "team",
    "TeamName": "team",
    "adjoe": "adj_oe",
    "adjde": "adj_de",
    "adjt": "tempo",
    # Barttorvik uses "conf" — normalize to "conference"
    "conf": "conference",
}


# ---------------------------------------------------------------------------
# Lazy-import wrappers (re-raise so DataLoader can catch cleanly)
# ---------------------------------------------------------------------------

def fetch_trank(season: int) -> pd.DataFrame:
    """Lazy wrapper around src.data.barttorvik.fetch_trank."""
    from src.data.barttorvik import fetch_trank as _fetch_trank  # type: ignore

    return _fetch_trank(season)


def load_tournament_seeds(season: int) -> dict[str, int]:
    """Lazy wrapper around src.data.kaggle_ingestion.load_seeds.

    Delegates to ``load_seeds`` from the kaggle_ingestion module.  The *season*
    argument is passed through; callers are responsible for supplying a valid
    filepath or season identifier as required by the underlying function.
    """
    from src.data.kaggle_ingestion import load_seeds as _load_seeds  # type: ignore

    return _load_seeds(season)


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Cache-first loader for season-level data assets.

    Writes fetched data to ``cache_dir`` as parquet / JSON files on the first
    call and reads from those files on every subsequent call, avoiding repeated
    live scraping of Barttorvik and Kaggle.
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # T-Rank (Barttorvik)
    # ------------------------------------------------------------------

    def get_trank(self, season: int) -> pd.DataFrame:
        """Return T-Rank DataFrame for *season*, using cache if available.

        Cache path: ``{cache_dir}/trank_{season}.parquet``

        On any exception (network failure, missing parquet dependencies, etc.)
        logs a warning and returns an empty :class:`~pandas.DataFrame`.
        If the cached parquet file is corrupt, it is deleted and the data is
        re-fetched from the source.
        """
        cache_path = self._cache_dir / f"trank_{season}.parquet"
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as read_exc:
                logger.warning(
                    "Corrupt parquet cache for season %s (%s). Deleting and re-fetching.",
                    season,
                    read_exc,
                )
                cache_path.unlink(missing_ok=True)
        try:
            df = fetch_trank(season)
            df = df.rename(columns=_COLUMN_ALIASES)
            # Ensure required columns exist with sensible defaults
            if "luck" not in df.columns:
                df["luck"] = 0.0
            if "seed" not in df.columns:
                df["seed"] = 16
            if "conference" not in df.columns:
                df["conference"] = "Unknown"
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Barttorvik fetch failed (season=%s): %s. Returning empty.",
                season,
                exc,
            )
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Tournament seeds (Kaggle)
    # ------------------------------------------------------------------

    def get_tournament_seeds(self, season: int) -> dict[str, int]:
        """Return tournament seed mapping for *season*, using cache if available.

        Cache path: ``{cache_dir}/seeds_{season}.json``

        On any exception logs a warning and returns ``{}``.
        """
        cache_path = self._cache_dir / f"seeds_{season}.json"

        try:
            if cache_path.exists():
                with open(cache_path) as fh:
                    return json.load(fh)

            seeds = load_tournament_seeds(season)
            with open(cache_path, "w") as fh:
                json.dump(seeds, fh)
            return seeds

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DataLoader.get_tournament_seeds(%s) failed — returning {}. "
                "Error: %s",
                season,
                exc,
            )
            return {}
