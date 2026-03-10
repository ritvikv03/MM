"""
Shared pytest fixtures for all data ingestion tests.
"""
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory that mimics data/raw/ layout."""
    for sub in ["kaggle", "barttorvik", "kenpom", "evanmiya", "hoopmath",
                "shotquality", "market"]:
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """
    Inject dummy env vars for all tests so modules that call load_dotenv()
    won't fail due to missing .env file.
    """
    monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
    monkeypatch.setenv("KAGGLE_KEY", "test_key_abc123")
    monkeypatch.setenv("KENPOM_EMAIL", "test@example.com")
    monkeypatch.setenv("KENPOM_PASSWORD", "test_password")
    monkeypatch.setenv("ODDS_API_KEY", "test_odds_key")
    monkeypatch.setenv("ACTION_NETWORK_API_KEY", "test_action_key")
    monkeypatch.setenv("WANDB_API_KEY", "test_wandb_key")
    monkeypatch.setenv("TWITTER_BEARER_TOKEN", "test_twitter_token")
    monkeypatch.setenv("SHOTQUALITY_API_KEY", "test_sq_key")
    monkeypatch.setenv("EVANMIYA_API_KEY", "test_evanmiya_key")


@pytest.fixture(autouse=True)
def isolate_cache_dirs(monkeypatch, tmp_path):
    """
    Redirect every module-level CACHE_DIR to a fresh tmp_path subtree for
    each test, preventing cache bleed-over between tests that don't explicitly
    patch the cache location themselves.

    Tests that call monkeypatch.setattr("src.data.X.CACHE_DIR", ...) after
    this fixture runs will simply override the value for their own duration
    (monkeypatch restores in LIFO order), so explicit per-test patches still
    take precedence.
    """
    import importlib
    modules = [
        "src.data.shotquality",
        "src.data.evanmiya",
        "src.data.hoopmath",
        "src.data.barttorvik",
        "src.data.kenpom",
        "src.data.market_data",
        "src.data.injury_feed",
    ]
    for mod_path in modules:
        try:
            mod = importlib.import_module(mod_path)
            if hasattr(mod, "CACHE_DIR"):
                isolated = tmp_path / mod_path.split(".")[-1]
                isolated.mkdir(parents=True, exist_ok=True)
                monkeypatch.setattr(mod, "CACHE_DIR", isolated)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Shared sample DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_game_results_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Season":   [2024, 2024, 2024],
        "DayNum":   [1, 2, 3],
        "WTeamID":  [1101, 1102, 1103],
        "WScore":   [75, 80, 65],
        "LTeamID":  [1201, 1202, 1203],
        "LScore":   [70, 60, 55],
        "WLoc":     ["H", "A", "N"],
        "NumOT":    [0, 1, 0],
    })


@pytest.fixture
def sample_seeds_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Season": [2024, 2024, 2024],
        "Slot":   ["W01", "X02", "Y16"],
        "TeamID": [1101, 1102, 1103],
        "Seed":   ["W01", "X02", "Y16"],
    })


@pytest.fixture
def sample_bpr_df() -> pd.DataFrame:
    return pd.DataFrame({
        "player":      ["Alice", "Bob", "Carol"],
        "team":        ["Duke"] * 3,
        "season":      [2024] * 3,
        "minutes_pct": [0.40, 0.35, 0.25],
        "obpr":        [2.1, 1.8, 0.9],
        "dbpr":        [1.5, 1.2, 0.7],
        "bpr":         [3.6, 3.0, 1.6],
        "pos":         ["G", "F", "C"],
    })
