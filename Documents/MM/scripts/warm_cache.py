#!/usr/bin/env python3
"""
scripts/warm_cache.py

Pre-populate data/cache/ before starting the server.
Run once per season (takes ~30-60s for live Barttorvik scrape).

Usage:
    python scripts/warm_cache.py --season 2024
    python scripts/warm_cache.py --season 2025
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.data_cache import DataLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-warm data/cache/ for a given season.")
    parser.add_argument("--season", type=int, default=2025, help="Season year (default: 2025)")
    args = parser.parse_args()

    loader = DataLoader()
    print(f"Warming cache for season {args.season}...")
    print("  Fetching Barttorvik T-Rank...", end=" ", flush=True)
    df = loader.get_trank(args.season)
    if df.empty:
        print("WARN: empty — Barttorvik may be unavailable or unreachable.")
    else:
        print(f"OK ({len(df)} teams cached)")

    print("  Fetching tournament seeds...", end=" ", flush=True)
    seeds = loader.get_tournament_seeds(args.season)
    if not seeds:
        print("WARN: empty — Kaggle credentials may be missing from .env")
    else:
        print(f"OK ({len(seeds)} teams seeded)")

    print(f"\nCache written to: {loader._cache_dir.resolve()}")
    print("Start the backend with: USE_REAL_DATA=true uvicorn src.api.server:app --port 8000 --reload")


if __name__ == "__main__":
    main()
