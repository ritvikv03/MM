#!/usr/bin/env python
"""
CLI entry point for the NCAA March Madness ST-GNN pipeline.

Usage:
    python scripts/run_pipeline.py --season 2024 --sampler advi --n-epochs 50

All flags correspond directly to fields on PipelineConfig.  Run with --help
for a full listing.
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of cwd.
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before any src imports so WANDB_API_KEY etc. are available.
from dotenv import load_dotenv
load_dotenv()

from src.pipeline import PipelineConfig, validate_config, run_pipeline
from src.utils.wandb_logger import ExperimentLogger, format_run_name


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NCAA March Madness ST-GNN + Bayesian pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    parser.add_argument(
        "--season",
        type=int,
        default=2024,
        help="NCAA season year.",
    )

    # --- Bayesian sampler ---
    parser.add_argument(
        "--sampler",
        type=str,
        default="advi",
        choices=["advi", "nuts"],
        help="Bayesian inference sampler.",
    )
    parser.add_argument(
        "--nuts-draws",
        type=int,
        default=500,
        dest="nuts_draws",
        help="Number of NUTS posterior draws (only used when --sampler nuts).",
    )
    parser.add_argument(
        "--nuts-chains",
        type=int,
        default=2,
        dest="nuts_chains",
        help="Number of NUTS chains (only used when --sampler nuts).",
    )

    # --- Training ---
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=50,
        dest="n_epochs",
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        dest="random_seed",
        help="Global random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Compute device (e.g. "cpu" or "cuda:0").',
    )

    # --- GAT encoder ---
    parser.add_argument(
        "--gat-hidden-dim",
        type=int,
        default=64,
        dest="gat_hidden_dim",
        help="Hidden dimension of the GAT encoder.",
    )
    parser.add_argument(
        "--gat-num-heads",
        type=int,
        default=4,
        dest="gat_num_heads",
        help="Number of attention heads in the GAT encoder.",
    )

    # --- Temporal encoder ---
    parser.add_argument(
        "--temporal-hidden-dim",
        type=int,
        default=128,
        dest="temporal_hidden_dim",
        help="Hidden dimension of the temporal encoder.",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="lstm",
        choices=["lstm", "transformer"],
        dest="encoder_type",
        help="Temporal encoder architecture.",
    )

    # --- Cross-validation / backtest ---
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        dest="n_splits",
        help="Number of time-series CV splits.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=20,
        dest="test_size",
        help="Number of games per test fold.",
    )

    # --- W&B ---
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="mm-stgnn",
        dest="wandb_project",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="",
        dest="wandb_entity",
        help="W&B entity (team or username).  Empty string uses default.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        dest="wandb_mode",
        help="W&B run mode.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1.0",
        dest="model_version",
        help="Model version tag.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build config from parsed args.
    cfg = PipelineConfig(
        season=args.season,
        sampler=args.sampler,
        nuts_draws=args.nuts_draws,
        nuts_chains=args.nuts_chains,
        n_epochs=args.n_epochs,
        lr=args.lr,
        random_seed=args.random_seed,
        device=args.device,
        gat_hidden_dim=args.gat_hidden_dim,
        gat_num_heads=args.gat_num_heads,
        temporal_hidden_dim=args.temporal_hidden_dim,
        encoder_type=args.encoder_type,
        n_splits=args.n_splits,
        test_size=args.test_size,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        model_version=args.model_version,
    )

    # Validate before touching any external resources.
    validate_config(cfg)

    # Initialise logger.
    logger = ExperimentLogger(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        season=cfg.season,
        model_version=cfg.model_version,
        mode=cfg.wandb_mode,
    )

    run_name = format_run_name(cfg.season, cfg.model_version, cfg.sampler)
    logger.init_run(config=vars(cfg), run_name=run_name)

    # Run the pipeline.
    try:
        summary = run_pipeline(cfg, logger)
    finally:
        logger.finish()

    # Print formatted backtest summary.
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    """Print backtest summary as formatted text to stdout."""
    print()
    print("=" * 50)
    print("  Backtest Summary")
    print("=" * 50)
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            print(f"  {label:<25} {value:.6f}")
        else:
            print(f"  {label:<25} {value}")
    print("=" * 50)
    print()


if __name__ == "__main__":
    main()
