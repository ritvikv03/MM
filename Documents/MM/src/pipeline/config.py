"""
src/pipeline/config.py

PipelineConfig dataclass holding all pipeline hyperparameters, plus helpers:
  - validate_config  — raises ValueError for invalid field values
  - config_from_dict — construct from a plain dict (unknown keys ignored)
  - to_wandb_config  — flat dict for wandb.init(config=...)

No torch / pymc / wandb is imported at module level.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All configurable parameters for a single pipeline run."""

    # Data
    season: int = 2024
    data_dir: str = "data/raw"
    cache_dir: str = "data/cache"

    # Graph
    node_in_features: int = 12
    edge_in_features: int = 5

    # Shannon Entropy gating
    entropy_feat_dim: int = 3        # scoring_entropy_normalized, kill_shot_vulnerability,
                                     # kill_shot_p_run_given_trading
    use_entropy_gating: bool = True  # False falls back to plain GATEncoder

    # GAT encoder
    gat_hidden_dim: int = 64
    gat_num_heads: int = 4
    gat_num_layers: int = 2
    gat_dropout: float = 0.1

    # Temporal encoder
    temporal_hidden_dim: int = 128
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.1
    temporal_encoder_type: str = "lstm"   # "lstm" or "transformer"

    # Training
    learning_rate: float = 0.001
    n_epochs: int = 50
    device: str = "cpu"

    # Bayesian
    sampler: str = "advi"           # "advi" or "nuts"
    advi_iterations: int = 10_000
    nuts_draws: int = 500
    nuts_chains: int = 2
    nuts_tune: int = 200
    random_seed: int = 42

    # Backtesting
    n_splits: int = 5
    test_size: int = 20
    backtest_gap: int = 1

    # W&B
    wandb_project: str = "mm-stgnn"
    wandb_entity: str = ""
    wandb_mode: str = "disabled"    # "disabled" by default so tests never hit network
    model_version: str = "v1.0"


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

_VALID_SAMPLERS = {"advi", "nuts"}
_VALID_TEMPORAL_ENCODER_TYPES = {"lstm", "transformer"}


def validate_config(cfg: PipelineConfig) -> None:
    """
    Raise ValueError with a descriptive message if any field is invalid.

    Checks:
    - gat_hidden_dim % gat_num_heads == 0
    - n_epochs >= 1
    - sampler in {"advi", "nuts"}
    - temporal_encoder_type in {"lstm", "transformer"}
    - n_splits >= 1
    - test_size >= 1

    Parameters
    ----------
    cfg : PipelineConfig

    Raises
    ------
    ValueError
        On the first invalid field encountered.
    """
    if cfg.gat_hidden_dim % cfg.gat_num_heads != 0:
        raise ValueError(
            f"gat_hidden_dim ({cfg.gat_hidden_dim}) must be divisible by "
            f"gat_num_heads ({cfg.gat_num_heads})."
        )

    if cfg.n_epochs < 1:
        raise ValueError(
            f"n_epochs must be >= 1, got {cfg.n_epochs}."
        )

    if cfg.sampler not in _VALID_SAMPLERS:
        raise ValueError(
            f"sampler must be one of {_VALID_SAMPLERS}, got {cfg.sampler!r}."
        )

    if cfg.temporal_encoder_type not in _VALID_TEMPORAL_ENCODER_TYPES:
        raise ValueError(
            f"temporal_encoder_type must be one of "
            f"{_VALID_TEMPORAL_ENCODER_TYPES}, got {cfg.temporal_encoder_type!r}."
        )

    if cfg.n_splits < 1:
        raise ValueError(
            f"n_splits must be >= 1, got {cfg.n_splits}."
        )

    if cfg.test_size < 1:
        raise ValueError(
            f"test_size must be >= 1, got {cfg.test_size}."
        )


# ---------------------------------------------------------------------------
# config_from_dict
# ---------------------------------------------------------------------------

_FIELD_NAMES = {f.name for f in fields(PipelineConfig)}


def config_from_dict(d: dict) -> PipelineConfig:
    """
    Construct a PipelineConfig from a plain dict.

    Unknown keys are silently ignored; missing keys fall back to dataclass
    defaults.

    Parameters
    ----------
    d : dict
        Mapping of field names to values.

    Returns
    -------
    PipelineConfig
    """
    known = {k: v for k, v in d.items() if k in _FIELD_NAMES}
    return PipelineConfig(**known)


# ---------------------------------------------------------------------------
# to_wandb_config
# ---------------------------------------------------------------------------

_WANDB_EXCLUDED_KEYS = {"wandb_project", "wandb_entity", "wandb_mode"}


def to_wandb_config(cfg: PipelineConfig) -> dict:
    """
    Convert cfg to a flat dict suitable for ``wandb.init(config=...)``.

    Excludes ``wandb_project``, ``wandb_entity``, and ``wandb_mode``.

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    dict
        Flat mapping of hyperparameter name → scalar value.
    """
    raw = asdict(cfg)
    return {k: v for k, v in raw.items() if k not in _WANDB_EXCLUDED_KEYS}
