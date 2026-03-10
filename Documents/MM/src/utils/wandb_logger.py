"""
src/utils/wandb_logger.py

Weights & Biases experiment tracking wrapper for the NCAA March Madness
ST-GNN project.

Provides:
- ExperimentLogger: clean class interface over wandb.init / wandb.log / artifacts.
- compute_ece: Expected Calibration Error from raw prediction arrays.
- make_sweep_config: convert flat param grid → W&B sweep config dict.
- format_run_name: canonical run-name string.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Thin wrapper around W&B for logging metrics, hyperparameter configs,
    and model artifacts.

    Parameters
    ----------
    project : str
        W&B project name.
    entity : str | None
        W&B entity (team / user).  ``None`` uses the default entity.
    season : int
        NCAA season year.  Tagged on every run as ``season=YYYY``.
    model_version : str
        Semver-style model version string.  Tagged as ``model_version=vX.Y``.
    tags : list[str] | None
        Extra free-form tags appended after the mandatory season/version tags.
    mode : str
        One of ``"online"``, ``"offline"``, or ``"disabled"``.
        Passed directly to ``wandb.init``.
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        season: int = 2024,
        model_version: str = "v1.0",
        tags: Optional[list] = None,
        mode: str = "online",
    ) -> None:
        self.project = project
        self.entity = entity
        self.season = season
        self.model_version = model_version
        self.tags = tags
        self.mode = mode
        self.run = None

    # ------------------------------------------------------------------
    # init_run
    # ------------------------------------------------------------------

    def init_run(self, config: dict, run_name: Optional[str] = None) -> None:
        """
        Initialise a W&B run.

        Parameters
        ----------
        config : dict
            Hyperparameter configuration dictionary logged to W&B.
        run_name : str | None
            Human-readable run name.  ``None`` lets W&B auto-generate one.

        Raises
        ------
        RuntimeError
            If ``mode`` is ``"online"`` and the ``WANDB_API_KEY`` environment
            variable is not set.
        """
        if self.mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY environment variable is required for online mode. "
                "Set it in your .env file or export it before running."
            )

        import wandb  # noqa: PLC0415 — deferred to support mocking in tests

        mandatory_tags = [f"season={self.season}", f"model_version={self.model_version}"]
        extra_tags = list(self.tags) if self.tags else []
        all_tags = mandatory_tags + extra_tags

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            tags=all_tags,
            name=run_name,
            mode=self.mode,
        )

    # ------------------------------------------------------------------
    # log_epoch
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        brier_score: float,
        log_loss: float,
        clv_delta: float,
        calibration_ece: float,
        extra: Optional[dict] = None,
    ) -> None:
        """
        Log per-epoch metrics to W&B.

        All five CLAUDE.md-mandated keys are always written:
        ``brier_score``, ``log_loss``, ``clv_delta``, ``calibration_ece``,
        and ``epoch``.  Additional metrics can be passed via *extra*.

        Raises
        ------
        RuntimeError
            If :meth:`init_run` has not been called.
        """
        self._assert_initialized("log_epoch")

        import wandb  # noqa: PLC0415

        payload: dict = {
            "brier_score": brier_score,
            "log_loss": log_loss,
            "clv_delta": clv_delta,
            "calibration_ece": calibration_ece,
            "epoch": epoch,
        }
        if extra:
            payload.update(extra)

        wandb.log(payload)

    # ------------------------------------------------------------------
    # log_hyperparams
    # ------------------------------------------------------------------

    def log_hyperparams(self, params: dict) -> None:
        """
        Update the W&B run config with additional hyperparameters.

        Raises
        ------
        RuntimeError
            If :meth:`init_run` has not been called.
        """
        self._assert_initialized("log_hyperparams")

        import wandb  # noqa: PLC0415

        wandb.config.update(params)

    # ------------------------------------------------------------------
    # log_artifact
    # ------------------------------------------------------------------

    def log_artifact(
        self,
        filepath: str,
        artifact_type: str,
        name: Optional[str] = None,
    ) -> None:
        """
        Package a local file as a W&B artifact and attach it to the run.

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the file to upload.
        artifact_type : str
            W&B artifact type string, e.g. ``"model"`` or ``"dataset"``.
        name : str | None
            Artifact name.  Defaults to ``Path(filepath).stem``.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist on disk.
        RuntimeError
            If :meth:`init_run` has not been called.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Artifact file not found: {filepath}")

        self._assert_initialized("log_artifact")

        import wandb  # noqa: PLC0415

        artifact_name = name if name is not None else Path(filepath).stem
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(filepath)
        self.run.log_artifact(artifact)

    # ------------------------------------------------------------------
    # finish
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """
        Finish the active W&B run.  No-op if :meth:`init_run` was never called.
        """
        if self.run is None:
            return

        import wandb  # noqa: PLC0415

        wandb.finish()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_initialized(self, method_name: str) -> None:
        if self.run is None:
            raise RuntimeError(
                f"{method_name}() called before init_run(). "
                "Call ExperimentLogger.init_run(config) first."
            )


# ---------------------------------------------------------------------------
# compute_ece
# ---------------------------------------------------------------------------

def compute_ece(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions into *n_bins* equal-width bins on [0, 1], then
    computes the weighted average absolute difference between mean predicted
    probability and mean observed label in each bin::

        ECE = Σ_b (|bin_b| / N) × |mean_pred_b − mean_true_b|

    Parameters
    ----------
    y_pred : np.ndarray, shape (N,)
        Predicted probabilities in [0, 1].
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels (0 or 1), or continuous values in [0, 1].
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    float
        ECE ∈ [0, 1].  Returns 0.0 for empty input.

    Raises
    ------
    ValueError
        If *y_pred* and *y_true* have different shapes.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape} "
            f"but y_true has shape {y_true.shape}."
        )

    n = len(y_pred)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for the last bin to capture pred == 1.0
        if i < n_bins - 1:
            mask = (y_pred >= lo) & (y_pred < hi)
        else:
            mask = (y_pred >= lo) & (y_pred <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            continue

        mean_pred = y_pred[mask].mean()
        mean_true = y_true[mask].mean()
        ece += (bin_size / n) * abs(mean_pred - mean_true)

    return float(ece)


# ---------------------------------------------------------------------------
# make_sweep_config
# ---------------------------------------------------------------------------

def make_sweep_config(param_grid: dict) -> dict:
    """
    Convert a flat parameter grid into a W&B sweep configuration dict.

    Parameters
    ----------
    param_grid : dict
        Mapping of hyperparameter name → list of values, e.g.::

            {"lr": [0.001, 0.01], "hidden_dim": [64, 128]}

    Returns
    -------
    dict
        W&B sweep config with ``method="grid"`` and each parameter wrapped
        in ``{"values": [...]}``::

            {
                "method": "grid",
                "parameters": {
                    "lr": {"values": [0.001, 0.01]},
                    "hidden_dim": {"values": [64, 128]},
                },
            }
    """
    return {
        "method": "grid",
        "parameters": {key: {"values": values} for key, values in param_grid.items()},
    }


# ---------------------------------------------------------------------------
# format_run_name
# ---------------------------------------------------------------------------

def format_run_name(season: int, model_version: str, sampler: str) -> str:
    """
    Build a canonical W&B run name.

    Parameters
    ----------
    season : int
        NCAA season year (e.g. 2024).
    model_version : str
        Model version string (e.g. ``"v1.0"``).
    sampler : str
        Inference sampler identifier (e.g. ``"advi"`` or ``"nuts"``).

    Returns
    -------
    str
        ``f"mm_{season}_{model_version}_{sampler}"``
        e.g. ``"mm_2024_v1.0_advi"``.
    """
    return f"mm_{season}_{model_version}_{sampler}"
