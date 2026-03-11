"""
src/pipeline/pipeline.py

End-to-end pipeline orchestration for the NCAA March Madness ST-GNN project.

All heavy dependencies (torch, pymc, wandb, real data I/O) are injected as
callable arguments so every function is 100 % mockable in tests.

No torch / pymc / wandb is imported at module level.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional

from src.pipeline.config import PipelineConfig, to_wandb_config, validate_config

if TYPE_CHECKING:
    from src.utils.wandb_logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Lazy import helper
# ---------------------------------------------------------------------------


def _import_train_one_epoch():
    """Deferred import of train_one_epoch to avoid torch at module level."""
    from src.model.stgnn import train_one_epoch  # noqa: PLC0415
    return train_one_epoch


# Module-level reference that tests can patch via
#   patch("src.pipeline.pipeline.train_one_epoch", ...)
try:
    from src.model.stgnn import train_one_epoch  # type: ignore[import]
except Exception:  # pragma: no cover — torch may not be installed
    train_one_epoch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# build_gat_config
# ---------------------------------------------------------------------------


def build_gat_config(cfg: PipelineConfig) -> dict:
    """
    Return the gat_config dict for make_stgnn_model.

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    dict
        Keys: node_in_features, hidden_dim, num_heads, num_layers, dropout.
    """
    return {
        "node_in_features": cfg.node_in_features,
        "hidden_dim": cfg.gat_hidden_dim,
        "num_heads": cfg.gat_num_heads,
        "num_layers": cfg.gat_num_layers,
        "dropout": cfg.gat_dropout,
    }


# ---------------------------------------------------------------------------
# build_temporal_config
# ---------------------------------------------------------------------------


def build_temporal_config(cfg: PipelineConfig) -> dict:
    """
    Return the temporal_config dict for make_stgnn_model.

    Parameters
    ----------
    cfg : PipelineConfig

    Returns
    -------
    dict
        Keys: hidden_dim, num_layers, dropout, encoder_type.
    """
    return {
        "hidden_dim": cfg.temporal_hidden_dim,
        "num_layers": cfg.temporal_num_layers,
        "dropout": cfg.temporal_dropout,
        "encoder_type": cfg.temporal_encoder_type,
    }


# ---------------------------------------------------------------------------
# run_training_loop
# ---------------------------------------------------------------------------


def run_training_loop(
    model,
    optimizer,
    snapshots_list: list,
    game_home_idx,
    game_away_idx,
    labels,
    n_epochs: int,
    logger: "Optional[ExperimentLogger]" = None,
    extra_metrics_fn: Optional[Callable] = None,
) -> List[dict]:
    """
    Run n_epochs of train_one_epoch, log each epoch to logger if provided.

    clv_delta and calibration_ece are set to 0.0 until backtest provides real
    values (placeholder for future wiring).

    Parameters
    ----------
    model : STGNNModel or mock
    optimizer : torch.optim.Optimizer or mock
    snapshots_list : list of TemporalSnapshot (or mock)
    game_home_idx : array-like of home-team indices
    game_away_idx : array-like of away-team indices
    labels : array-like of binary labels
    n_epochs : int
    logger : ExperimentLogger | None
        If provided, :meth:`log_epoch` is called after each epoch.
    extra_metrics_fn : callable | None
        Signature: ``extra_metrics_fn(epoch, probs, labels) -> dict | None``
        If provided, returned dict is merged into the epoch entry.

    Returns
    -------
    list[dict]
        Per-epoch metric dicts, one entry per epoch.
        Each entry contains at minimum:
        ``brier_score``, ``log_loss``, ``clv_delta``,
        ``calibration_ece``, ``epoch``.
    """
    history: List[dict] = []

    for epoch in range(n_epochs):
        # train_one_epoch is patched in tests; real signature:
        #   train_one_epoch(model, optimizer, snapshots, home_idx, away_idx, labels) -> dict
        epoch_metrics: dict = train_one_epoch(
            model,
            optimizer,
            snapshots_list,
            game_home_idx,
            game_away_idx,
            labels,
        )

        # Ensure mandatory W&B keys are always present
        entry: dict = dict(epoch_metrics)
        entry.setdefault("brier_score", 0.0)
        entry.setdefault("log_loss", 0.0)
        entry["clv_delta"] = 0.0          # filled by backtest later
        entry["calibration_ece"] = 0.0    # filled by backtest later
        entry["epoch"] = epoch

        # Merge optional extra metrics
        if extra_metrics_fn is not None:
            probs = entry.get("probs", labels)
            extra = extra_metrics_fn(epoch, probs, labels)
            if extra:
                entry.update(extra)

        history.append(entry)

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                brier_score=float(entry.get("brier_score", 0.0)),
                log_loss=float(entry.get("log_loss", 0.0)),
                clv_delta=0.0,
                calibration_ece=0.0,
            )

    return history


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: PipelineConfig,
    *,
    data_loader_fn: Optional[Callable] = None,
    model_factory_fn: Optional[Callable] = None,
    bayesian_head_fn: Optional[Callable] = None,
    logger: "Optional[ExperimentLogger]" = None,
) -> dict:
    """
    Full pipeline entry point.

    All heavy lifting is delegated to injected callables so the function is
    fully testable without real data or torch.

    Parameters
    ----------
    cfg : PipelineConfig
    data_loader_fn : callable(cfg) -> dict | None
        Expected keys: snapshots, home_idx, away_idx, labels, game_records.
        When None, empty stubs are used (no training, trivial backtest).
    model_factory_fn : callable(cfg) -> (model, optimizer) | None
        When None, (None, None) stubs are used.
    bayesian_head_fn : callable(cfg, embeddings) -> idata | None
        Called after training if provided.
    logger : ExperimentLogger | None
        If provided, final backtest summary is logged.

    Returns
    -------
    dict
        Keys:
        - ``"training_history"``  — list[dict] of per-epoch metrics
        - ``"backtest_results"``  — list of BacktestResult objects
        - ``"backtest_summary"``  — dict from aggregate_backtest
        - ``"config"``            — to_wandb_config(cfg) output dict
    """
    # Step 1: validate — raise immediately on invalid config
    validate_config(cfg)

    # Step 2: load data
    if data_loader_fn is not None:
        data: dict = data_loader_fn(cfg)
    else:
        data = {
            "snapshots": [],
            "home_idx": [],
            "away_idx": [],
            "labels": [],
            "game_records": [],
        }

    snapshots = data.get("snapshots", [])
    home_idx = data.get("home_idx", [])
    away_idx = data.get("away_idx", [])
    labels = data.get("labels", [])
    game_records = data.get("game_records", [])

    # Step 3: build model
    if model_factory_fn is not None:
        model, optimizer = model_factory_fn(cfg)
    else:
        model, optimizer = None, None

    # Step 4: training loop
    if model is not None and snapshots:
        training_history = run_training_loop(
            model=model,
            optimizer=optimizer,
            snapshots_list=snapshots,
            game_home_idx=home_idx,
            game_away_idx=away_idx,
            labels=labels,
            n_epochs=cfg.n_epochs,
            logger=logger,
        )
    else:
        # Stub: still produce n_epochs empty entries when model is available
        # but data is absent, OR produce exactly n_epochs entries when both
        # model and data are provided via the training_loop above.
        training_history = []
        if model is not None:
            # Model provided but no snapshots — run loop with empty lists
            training_history = run_training_loop(
                model=model,
                optimizer=optimizer,
                snapshots_list=snapshots,
                game_home_idx=home_idx,
                game_away_idx=away_idx,
                labels=labels,
                n_epochs=cfg.n_epochs,
                logger=logger,
            )

    # Step 5: Bayesian head (optional)
    idata = None
    if bayesian_head_fn is not None:
        embeddings = None  # real implementation would pass model embeddings
        idata = bayesian_head_fn(cfg, embeddings)

    # Step 6: backtest
    from src.backtesting.backtesting import (  # noqa: PLC0415
        run_backtest,
        aggregate_backtest,
    )

    def _default_predict_fn(train_records, test_records):
        """Trivial predict_fn: returns 0.5 for every game."""
        import numpy as np  # noqa: PLC0415
        return np.full(len(test_records), 0.5)

    if game_records:
        try:
            backtest_results = run_backtest(
                predict_fn=_default_predict_fn,
                game_records=game_records,
                n_splits=cfg.n_splits,
                test_size=cfg.test_size,
                gap=cfg.backtest_gap,
            )
        except ValueError:
            # Dataset too small for the requested splits — return empty results
            backtest_results = []
    else:
        backtest_results = []

    backtest_summary: dict = aggregate_backtest(backtest_results) if backtest_results else {}

    # Step 7: log final summary
    if logger is not None and backtest_summary:
        logger.log_epoch(
            epoch=-1,
            brier_score=float(backtest_summary.get("mean_brier", 0.0)),
            log_loss=float(backtest_summary.get("mean_log_loss", 0.0)),
            clv_delta=float(backtest_summary.get("mean_clv_delta", 0.0)),
            calibration_ece=float(backtest_summary.get("mean_ece", 0.0)),
        )

    return {
        "training_history": training_history,
        "backtest_results": backtest_results,
        "backtest_summary": backtest_summary,
        "config": to_wandb_config(cfg),
    }
