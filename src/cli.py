"""
src/cli.py
Madness Matrix CLI entry point — registered as the `mm` command.

Usage:
    mm run full       [--season 2026] [--dry-run]
    mm run intel      [--season 2026] [--dry-run]
    mm run results    [--season 2026] [--dry-run]
    mm serve          [--port 8000] [--reload]
    mm validate
    mm sweep
"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mm")


# ── subcommand: run ───────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    from src.pipeline.github_actions_runner import PipelineRunner, RunProfile

    try:
        profile = RunProfile(args.profile)
    except ValueError:
        logger.error("Unknown profile %r. Choose: full | intel | results", args.profile)
        return 1

    runner = PipelineRunner(season=args.season, dry_run=args.dry_run)
    summary = runner.run(profile)

    status = summary.get("status", "unknown")
    icon = "✅" if status == "success" else "❌"
    print(
        f"\n{icon}  Pipeline [{args.profile.upper()}] {status.upper()}"
        f"  season={args.season}"
        f"  duration={summary.get('duration_secs', '?')}s"
    )
    if status != "success":
        print(f"   Error: {summary.get('error_log', 'see logs above')}")
        return 1
    print(
        f"   teams_updated={summary.get('teams_updated', 0)}"
        f"  predictions={summary.get('predictions_computed', 0)}"
        f"  alerts={summary.get('alerts_found', 0)}"
    )
    return 0


# ── subcommand: serve ─────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=args.reload,
    )
    return 0


# ── subcommand: validate ──────────────────────────────────────────────────────

def cmd_validate(_args: argparse.Namespace) -> int:
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/validate_scraping.py"],
        check=False,
    )
    return result.returncode


# ── subcommand: sweep ─────────────────────────────────────────────────────────

def cmd_sweep(_args: argparse.Namespace) -> int:
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/sweep.py"],
        check=False,
    )
    return result.returncode


# ── parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mm",
        description="Madness Matrix — NCAA bracket prediction engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # mm run <profile>
    run_p = sub.add_parser("run", help="Execute a pipeline run profile")
    run_p.add_argument(
        "profile",
        choices=["full", "intel", "results"],
        help="full=6AM  intel=12PM  results=10PM",
    )
    run_p.add_argument("--season", type=int, default=2026, metavar="YEAR")
    run_p.add_argument("--dry-run", action="store_true", help="Skip all network/DB calls")

    # mm serve
    serve_p = sub.add_parser("serve", help="Start the FastAPI backend server")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    # mm validate
    sub.add_parser("validate", help="Check all data source connections")

    # mm sweep
    sub.add_parser("sweep", help="Launch W&B hyperparameter sweep")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "run":      cmd_run,
        "serve":    cmd_serve,
        "validate": cmd_validate,
        "sweep":    cmd_sweep,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()
