#!/usr/bin/env python3
"""
scripts/warm_cache.py

Pre-populate data/cache/ for one season or all tournament-era seasons
so the frontend can cycle through historical data without live scraping.

Barttorvik JSON data is available from 2008 onward.
Default range when --all is used: 2012-2025 (covers every NCAA Tournament
since the modern efficiency era took hold).

Usage:
    # Single season
    python scripts/warm_cache.py --season 2024

    # All seasons 2012-2025 (recommended first-run)
    python scripts/warm_cache.py --all

    # Custom range
    python scripts/warm_cache.py --all --start 2016 --end 2025

    # Force re-fetch even if already cached
    python scripts/warm_cache.py --all --force
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.data_cache import DataLoader

_DEFAULT_START = 2012
_DEFAULT_END   = 2025


def _warm_one(loader: DataLoader, season: int, force: bool) -> dict:
    """Warm cache for a single season. Returns a status dict."""
    result = {"season": season, "trank": "skip", "seeds": "skip"}

    trank_path = loader._cache_dir / f"trank_{season}.parquet"
    seeds_path = loader._cache_dir / f"seeds_{season}.json"

    # T-Rank
    if force and trank_path.exists():
        trank_path.unlink()
    if not trank_path.exists() or force:
        t0 = time.time()
        df = loader.get_trank(season)
        elapsed = round(time.time() - t0, 1)
        if df.empty:
            result["trank"] = f"WARN (empty, {elapsed}s)"
        else:
            result["trank"] = f"OK ({len(df)} teams, {elapsed}s)"
    else:
        result["trank"] = "cached"

    # Seeds (Kaggle — may fail without credentials)
    if force and seeds_path.exists():
        seeds_path.unlink()
    if not seeds_path.exists() or force:
        t0 = time.time()
        try:
            seeds = loader.get_tournament_seeds(season)
        except BaseException:
            seeds = {}
        elapsed = round(time.time() - t0, 1)
        if not seeds:
            result["seeds"] = f"WARN (Kaggle creds missing? {elapsed}s)"
        else:
            result["seeds"] = f"OK ({len(seeds)} teams, {elapsed}s)"
    else:
        result["seeds"] = "cached"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-warm data/cache/ so the server can serve real historical data.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--season", type=int, help="Single season year (e.g. 2024)")
    group.add_argument("--all", action="store_true",
                       help=f"Warm all seasons {_DEFAULT_START}-{_DEFAULT_END}")
    parser.add_argument("--start", type=int, default=_DEFAULT_START,
                        help="Start year for --all (default: 2012)")
    parser.add_argument("--end",   type=int, default=_DEFAULT_END,
                        help="End year inclusive for --all (default: 2025)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if already cached")
    args = parser.parse_args()

    # Determine season list
    if args.season:
        seasons = [args.season]
    elif args.all:
        seasons = list(range(args.start, args.end + 1))
    else:
        # Default: three most recent seasons
        seasons = [_DEFAULT_END - 2, _DEFAULT_END - 1, _DEFAULT_END]

    loader = DataLoader()
    print(f"Cache directory : {loader._cache_dir.resolve()}")
    print(f"Seasons to warm : {seasons[0]}-{seasons[-1]} ({len(seasons)} total)\n")

    results = []
    for season in seasons:
        r = _warm_one(loader, season, force=args.force)
        trank_str = r["trank"]
        seeds_str = r["seeds"]
        flag = "✓" if (trank_str.startswith("OK") or trank_str == "cached") else "✗"
        print(f"  {flag} [{season}]  T-Rank: {trank_str:<38} Seeds: {seeds_str}")
        results.append(r)

    n_trank_ok = sum(
        1 for r in results
        if r["trank"].startswith("OK") or r["trank"] == "cached"
    )
    n_seeds_ok = sum(
        1 for r in results
        if r["seeds"].startswith("OK") or r["seeds"] == "cached"
    )

    print(f"\n{'─' * 65}")
    print(f"T-Rank : {n_trank_ok}/{len(results)} seasons ready")
    print(f"Seeds  : {n_seeds_ok}/{len(results)} seasons ready  "
          f"(add KAGGLE_USERNAME + KAGGLE_KEY to .env to enable)")
    print()

    if n_trank_ok == len(results):
        print("All T-Rank data cached. Run the stack:")
        print()
        print("  USE_REAL_DATA=true uvicorn src.api.server:app --port 8000 --reload")
        print("  cd frontend && npm run dev")
        print()
        print("Then open http://localhost:3000 — cycle seasons in the sidebar to")
        print("browse historical efficiency data from 2012 through 2025.")
    else:
        failed = [r["season"] for r in results if r["trank"].startswith("WARN")]
        print(f"Some seasons failed: {failed}")
        print("Stub data will be served as fallback for failed seasons.")


if __name__ == "__main__":
    main()
