"""Tests for src/api/data_cache.py — cache-first DataLoader."""
import json
import os
import importlib
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_df() -> pd.DataFrame:
    """Minimal DataFrame matching the expected schema after column normalization."""
    return pd.DataFrame(
        {
            "team": ["Duke", "Kansas", "Gonzaga"],
            "adj_oe": [115.2, 120.1, 118.4],
            "adj_de": [92.3, 94.5, 90.1],
            "tempo": [68.5, 71.2, 69.8],
            "luck": [0.02, -0.01, 0.03],
            "seed": [1, 2, 1],
            "conference": ["ACC", "Big 12", "WCC"],
        }
    )


def _make_mock_seeds() -> dict:
    return {"Duke": 1, "Kansas": 2, "Gonzaga": 1}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_get_trank_returns_dataframe(self, tmp_path):
        """get_trank() returns a pandas DataFrame."""
        from src.api.data_cache import DataLoader

        mock_df = _make_mock_df()
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df):
            loader = DataLoader(cache_dir=str(tmp_path))
            result = loader.get_trank(2025)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_get_trank_writes_cache(self, tmp_path):
        """get_trank() writes a parquet file to the cache directory."""
        from src.api.data_cache import DataLoader

        mock_df = _make_mock_df()
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df):
            loader = DataLoader(cache_dir=str(tmp_path))
            loader.get_trank(2025)

        cache_file = tmp_path / "trank_2025.parquet"
        assert cache_file.exists(), "Expected parquet cache file to be written"

    def test_get_trank_reads_cache_on_second_call(self, tmp_path):
        """fetch_trank() is called only ONCE; second call reads from cache."""
        from src.api.data_cache import DataLoader

        mock_df = _make_mock_df()
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df) as mock_fetch:
            loader = DataLoader(cache_dir=str(tmp_path))
            loader.get_trank(2025)
            loader.get_trank(2025)

        mock_fetch.assert_called_once()

    def test_get_trank_graceful_degradation_on_error(self, tmp_path):
        """On fetch exception, get_trank() returns empty DataFrame without raising."""
        from src.api.data_cache import DataLoader

        with patch("src.api.data_cache.fetch_trank", side_effect=RuntimeError("network error")):
            loader = DataLoader(cache_dir=str(tmp_path))
            result = loader.get_trank(2025)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_get_tournament_seeds_returns_dict(self, tmp_path):
        """get_tournament_seeds() returns a dict mapping team name to seed."""
        from src.api.data_cache import DataLoader

        mock_seeds = _make_mock_seeds()
        with patch("src.api.data_cache.load_tournament_seeds", return_value=mock_seeds):
            loader = DataLoader(cache_dir=str(tmp_path))
            result = loader.get_tournament_seeds(2025)

        assert isinstance(result, dict)
        assert result["Duke"] == 1

    def test_get_tournament_seeds_writes_cache(self, tmp_path):
        """get_tournament_seeds() writes a JSON file to the cache directory."""
        from src.api.data_cache import DataLoader

        mock_seeds = _make_mock_seeds()
        with patch("src.api.data_cache.load_tournament_seeds", return_value=mock_seeds):
            loader = DataLoader(cache_dir=str(tmp_path))
            loader.get_tournament_seeds(2025)

        cache_file = tmp_path / "seeds_2025.json"
        assert cache_file.exists(), "Expected JSON cache file to be written"
        with open(cache_file) as f:
            loaded = json.load(f)
        assert loaded == mock_seeds

    def test_cache_dir_created_if_missing(self, tmp_path):
        """DataLoader creates cache_dir if it does not exist."""
        from src.api.data_cache import DataLoader

        new_dir = tmp_path / "nested" / "cache"
        assert not new_dir.exists()

        mock_df = _make_mock_df()
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df):
            loader = DataLoader(cache_dir=str(new_dir))
            loader.get_trank(2025)

        assert new_dir.exists(), "Expected cache_dir to be created automatically"

    def test_get_tournament_seeds_graceful_degradation_on_error(self, tmp_path):
        """On load exception, get_tournament_seeds() returns empty dict without raising."""
        from src.api.data_cache import DataLoader

        with patch(
            "src.api.data_cache.load_tournament_seeds",
            side_effect=FileNotFoundError("no data"),
        ):
            loader = DataLoader(cache_dir=str(tmp_path))
            result = loader.get_tournament_seeds(2025)

        assert isinstance(result, dict)
        assert result == {}

    def test_column_aliases_applied(self, tmp_path):
        """Column aliases are applied before caching (e.g., adj_o → adj_oe)."""
        from src.api.data_cache import DataLoader

        raw_df = pd.DataFrame(
            {
                "team_name": ["Duke"],
                "adj_o": [115.2],
                "adj_d": [92.3],
                "adj_t": [68.5],
                "luck": [0.02],
                "seed": [1],
                "conference": ["ACC"],
            }
        )
        with patch("src.api.data_cache.fetch_trank", return_value=raw_df):
            loader = DataLoader(cache_dir=str(tmp_path))
            result = loader.get_trank(2025)

        assert "adj_oe" in result.columns, "adj_o should be renamed to adj_oe"
        assert "adj_de" in result.columns, "adj_d should be renamed to adj_de"
        assert "tempo" in result.columns, "adj_t should be renamed to tempo"
        assert "team" in result.columns, "team_name should be renamed to team"

    def test_corrupt_parquet_triggers_refetch(self, tmp_path):
        """A corrupt parquet file is deleted and re-fetched, not silently returned as empty."""
        from src.api.data_cache import DataLoader

        loader = DataLoader(cache_dir=str(tmp_path))
        cache_path = tmp_path / "trank_2024.parquet"
        # Write junk bytes to simulate corruption
        cache_path.write_bytes(b"not a parquet file")
        mock_df = pd.DataFrame({
            "team": ["Duke"],
            "adj_oe": [118.0],
            "adj_de": [92.0],
            "tempo": [70.0],
            "luck": [0.01],
            "seed": [1],
            "conference": ["ACC"],
        })
        with patch("src.api.data_cache.fetch_trank", return_value=mock_df) as mock_fn:
            result = loader.get_trank(season=2024)

        assert mock_fn.call_count == 1       # re-fetched
        assert len(result) == 1              # real data returned, not empty
        assert not cache_path.exists() or pd.read_parquet(cache_path).shape[0] == 1  # new clean cache


# ---------------------------------------------------------------------------
# Lazy-import wrapper tests
# ---------------------------------------------------------------------------

class TestLazyImportWrappers:
    def test_fetch_trank_wrapper_delegates_to_barttorvik(self):
        """Module-level fetch_trank wrapper correctly imports and delegates."""
        mock_df = pd.DataFrame({
            "team": ["Duke"],
            "adj_oe": [118.0],
            "adj_de": [92.0],
            "tempo": [70.0],
            "luck": [0.01],
            "seed": [1],
            "conference": ["ACC"],
        })
        barttorvik = pytest.importorskip("src.data.barttorvik")
        with patch("src.data.barttorvik.fetch_trank", return_value=mock_df):
            from src.api import data_cache
            importlib.reload(data_cache)
            result = data_cache.fetch_trank(season=2024)
        assert isinstance(result, pd.DataFrame)

    def test_load_tournament_seeds_wrapper_delegates_to_kaggle(self):
        """Module-level load_tournament_seeds wrapper correctly imports and delegates."""
        mock_seeds = {"Duke": 1, "Kansas": 1}
        kaggle_ingestion = pytest.importorskip("src.data.kaggle_ingestion")
        with patch("src.data.kaggle_ingestion.load_seeds", return_value=mock_seeds):
            from src.api import data_cache
            importlib.reload(data_cache)
            result = data_cache.load_tournament_seeds(season=2024)
        assert isinstance(result, dict)

    def test_fetch_trank_wrapper_propagates_exception(self):
        """Wrapper re-raises exceptions so DataLoader can catch them."""
        barttorvik = pytest.importorskip("src.data.barttorvik")
        with patch("src.data.barttorvik.fetch_trank", side_effect=ConnectionError("network down")):
            from src.api import data_cache
            importlib.reload(data_cache)
            with pytest.raises(ConnectionError):
                data_cache.fetch_trank(season=2024)
