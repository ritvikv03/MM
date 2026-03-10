"""
src/backtesting/backtesting.py

Walk-forward backtesting engine for NCAA March Madness ST-GNN project.

Provides:
- WalkForwardSplit: generator for rolling-window train/test splits.
- BacktestResult: dataclass for per-split evaluation results.
- run_backtest: execute a backtest with a caller-supplied predict function.
- aggregate_backtest: summarise a list of BacktestResult objects.
- compute_log_loss_np: numpy-only binary cross-entropy (no sklearn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterator, List

import numpy as np


# ---------------------------------------------------------------------------
# compute_log_loss_np
# ---------------------------------------------------------------------------

def compute_log_loss_np(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Numpy-only binary cross-entropy loss (no sklearn dependency).

    Parameters
    ----------
    y_pred : np.ndarray, shape (N,)
        Predicted probabilities in [0, 1].
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels (0 or 1).
    eps : float
        Clamp probabilities to [eps, 1-eps] to avoid log(0).

    Returns
    -------
    float
        Mean binary cross-entropy.

    Raises
    ------
    ValueError
        If y_pred and y_true have different shapes.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape} "
            f"but y_true has shape {y_true.shape}."
        )

    y_pred_clamped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred_clamped) + (1.0 - y_true) * np.log(1.0 - y_pred_clamped))
    return float(loss.mean())


# ---------------------------------------------------------------------------
# WalkForwardSplit
# ---------------------------------------------------------------------------

class WalkForwardSplit:
    """
    Generator for rolling-window (walk-forward) train/test splits.

    Yields (train_indices, test_indices) pairs where test indices are always
    strictly greater than every train index, guaranteeing no future leakage.

    Parameters
    ----------
    n_splits : int
        Number of splits to generate.
    test_size : int
        Number of games in each test fold.
    gap : int
        Number of games to skip between the end of training and the start
        of the test fold (to avoid look-ahead leakage).  Default 0.
    """

    def __init__(self, n_splits: int, test_size: int, gap: int = 0) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, records: list) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_indices, test_indices) for each walk-forward split.

        The splits are arranged so that:
        - split k uses records[0..end_k] as training
        - there are `gap` games skipped after the last training game
        - the test window is records[end_k+gap+1 .. end_k+gap+test_size]

        Splits are arranged so the *last* split exhausts the dataset.

        Parameters
        ----------
        records : list
            List of game record dicts.  Must be in chronological order.

        Yields
        ------
        (train_indices, test_indices) : tuple of np.ndarray
            Both arrays contain integer indices into `records`.

        Raises
        ------
        ValueError
            If the dataset is too small to accommodate even one split.
        """
        n = len(records)
        # Minimum size: at least 1 training sample + gap + test_size
        min_required = 1 + self.gap + self.test_size
        if n < min_required:
            raise ValueError(
                f"Dataset too small: need at least {min_required} records for one split "
                f"(gap={self.gap}, test_size={self.test_size}), but got {n}."
            )

        # Place splits so the last one exhausts the dataset.
        # The end of the k-th (0-indexed) test window:
        #   test_end_k = n - 1 - (n_splits - 1 - k) * test_size
        # The training window for split k ends at:
        #   train_end_k = test_end_k - test_size - gap
        # Training window = records[0 .. train_end_k] (inclusive)

        for k in range(self.n_splits):
            test_end = n - 1 - (self.n_splits - 1 - k) * self.test_size
            test_start = test_end - self.test_size + 1
            train_end = test_start - self.gap - 1

            if train_end < 0:
                raise ValueError(
                    f"Dataset too small for split {k}: train_end={train_end} < 0. "
                    f"Reduce n_splits, test_size, or gap."
                )

            train_indices = np.arange(0, train_end + 1, dtype=int)
            test_indices = np.arange(test_start, test_end + 1, dtype=int)

            yield train_indices, test_indices


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Per-split evaluation results from a walk-forward backtest.

    Attributes
    ----------
    split_idx : int
        Zero-based index of this split.
    train_size : int
        Number of training games in this split.
    test_size : int
        Number of test games in this split.
    brier_score : float
        Mean squared error between y_pred and y_true.
    log_loss : float
        Mean binary cross-entropy.
    calibration_ece : float
        Expected Calibration Error (10-bin).
    clv_delta : float
        Mean Closing Line Value across test games.
    """

    split_idx: int
    train_size: int
    test_size: int
    brier_score: float
    log_loss: float
    calibration_ece: float
    clv_delta: float


# ---------------------------------------------------------------------------
# run_backtest
# ---------------------------------------------------------------------------

def run_backtest(
    predict_fn: Callable[[list, list], np.ndarray],
    game_records: list,
    n_splits: int,
    test_size: int,
    gap: int = 0,
) -> List[BacktestResult]:
    """
    Execute a walk-forward backtest.

    For each split generated by WalkForwardSplit, this function:
    1. Calls ``predict_fn(train_records, test_records)`` to get predicted probs.
    2. Computes Brier score, log-loss, ECE, and mean CLV delta.
    3. Returns a list of BacktestResult, one per split.

    Parameters
    ----------
    predict_fn : callable
        Signature: ``predict_fn(train_records, test_records) -> np.ndarray``
        Must return an array of predicted probabilities with length
        ``len(test_records)``.
    game_records : list of dict
        Each dict must contain:
        - ``y_true`` (int/float): binary outcome label (0 or 1).
        - ``clv`` (float, optional): closing line value.  Defaults to 0.0.
    n_splits : int
        Number of walk-forward splits.
    test_size : int
        Number of games per test fold.
    gap : int
        Games to skip between train and test to avoid look-ahead.

    Returns
    -------
    list of BacktestResult
        One BacktestResult per split, in chronological order.
    """
    # Deferred import to keep top-level imports lightweight
    from src.utils.wandb_logger import compute_ece

    splitter = WalkForwardSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    results: List[BacktestResult] = []

    for k, (train_idx, test_idx) in enumerate(splitter.split(game_records)):
        train_records = [game_records[i] for i in train_idx]
        test_records = [game_records[i] for i in test_idx]

        y_pred = np.asarray(predict_fn(train_records, test_records), dtype=float)
        y_true = np.array([r["y_true"] for r in test_records], dtype=float)

        # Brier score: mean squared error
        brier = float(np.mean((y_pred - y_true) ** 2))

        # Log loss (numpy-only)
        ll = compute_log_loss_np(y_pred, y_true)

        # ECE
        ece = compute_ece(y_pred, y_true, n_bins=10)

        # Mean CLV delta (default 0.0 if key absent)
        clv_values = [r.get("clv", 0.0) for r in test_records]
        clv_delta = float(np.mean(clv_values))

        results.append(
            BacktestResult(
                split_idx=k,
                train_size=len(train_records),
                test_size=len(test_records),
                brier_score=brier,
                log_loss=ll,
                calibration_ece=ece,
                clv_delta=clv_delta,
            )
        )

    return results


# ---------------------------------------------------------------------------
# aggregate_backtest
# ---------------------------------------------------------------------------

def aggregate_backtest(results: List[BacktestResult]) -> dict:
    """
    Aggregate a list of BacktestResult objects into summary statistics.

    Parameters
    ----------
    results : list of BacktestResult
        Output from :func:`run_backtest`.

    Returns
    -------
    dict
        Keys:
        - ``mean_brier`` / ``std_brier``
        - ``mean_log_loss`` / ``std_log_loss``
        - ``mean_ece`` / ``std_ece``
        - ``mean_clv_delta``
        - ``n_splits``
    """
    briers = np.array([r.brier_score for r in results])
    log_losses = np.array([r.log_loss for r in results])
    eces = np.array([r.calibration_ece for r in results])
    clv_deltas = np.array([r.clv_delta for r in results])

    return {
        "mean_brier": float(np.mean(briers)),
        "std_brier": float(np.std(briers)),
        "mean_log_loss": float(np.mean(log_losses)),
        "std_log_loss": float(np.std(log_losses)),
        "mean_ece": float(np.mean(eces)),
        "std_ece": float(np.std(eces)),
        "mean_clv_delta": float(np.mean(clv_deltas)),
        "n_splits": len(results),
    }
