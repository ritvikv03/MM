#!/usr/bin/env python
"""Launch a W&B hyperparameter sweep over key ST-GNN parameters."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.wandb_logger import make_sweep_config

SWEEP_PARAM_GRID = {
    "lr": [0.0001, 0.001, 0.01],
    "gat_hidden_dim": [64, 128],
    "gat_num_heads": [4, 8],
    "temporal_hidden_dim": [64, 128],
    "n_epochs": [30, 50],
    "sampler": ["advi"],   # keep NUTS off sweep to avoid long runtimes
}

def main():
    parser = argparse.ArgumentParser(description="Launch W&B sweep")
    parser.add_argument("--project", default="mm-stgnn")
    parser.add_argument("--entity", default="")
    parser.add_argument("--count", type=int, default=20, help="Max sweep runs")
    parser.add_argument("--dry-run", action="store_true", help="Print config without launching")
    args = parser.parse_args()

    sweep_config = make_sweep_config(SWEEP_PARAM_GRID)
    sweep_config["name"] = "stgnn-hyperparam-sweep"

    if args.dry_run:
        import json
        print(json.dumps(sweep_config, indent=2))
        return

    import wandb
    sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity or None)
    print(f"Sweep ID: {sweep_id}")
    wandb.agent(sweep_id, count=args.count)

if __name__ == "__main__":
    main()
