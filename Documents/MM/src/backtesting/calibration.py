"""
src/backtesting/calibration.py

Calibration analysis utilities for NCAA March Madness ST-GNN project.

Provides:
- reliability_diagram_data: bin-level statistics for reliability diagrams.
- ece_time_series: ECE computed at each time step / fold.
- sharpness: mean p*(1-p) as a measure of prediction decisiveness.
- brier_skill_score: BSS = 1 - BS_model / BS_climatology.
- compute_overround: sum of market probabilities (bookmaker vig measure).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# reliability_diagram_data
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute binned statistics for a reliability (calibration) diagram.

    Uses the same binning logic as ``compute_ece`` in wandb_logger.py:
    equal-width bins on [0, 1] with only the *last* bin including its right
    edge (so that predictions == 1.0 are captured).

    Parameters
    ----------
    y_pred : np.ndarray, shape (N,)
        Predicted probabilities in [0, 1].
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels (0 or 1).
    n_bins : int
        Number of equal-width bins (default 10).

    Returns
    -------
    dict with keys:
        - ``bin_centers`` (np.ndarray, shape (n_bins,)): midpoint of each bin.
        - ``mean_predicted`` (np.ndarray, shape (n_bins,)): mean y_pred per bin.
          np.nan for empty bins.
        - ``fraction_positive`` (np.ndarray, shape (n_bins,)): mean y_true per bin.
          np.nan for empty bins.
        - ``bin_counts`` (np.ndarray, shape (n_bins,)): number of samples per bin.
        - ``ece`` (float): Expected Calibration Error.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_width = 1.0 / n_bins
    bin_centers = bin_edges[:-1] + bin_width / 2.0

    mean_predicted = np.full(n_bins, np.nan)
    fraction_positive = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    n = len(y_pred)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Right edge inclusive only for the last bin (matches wandb_logger.compute_ece)
        if i < n_bins - 1:
            mask = (y_pred >= lo) & (y_pred < hi)
        else:
            mask = (y_pred >= lo) & (y_pred <= hi)

        bin_size = int(mask.sum())
        bin_counts[i] = bin_size

        if bin_size == 0:
            continue

        mean_predicted[i] = float(y_pred[mask].mean())
        fraction_positive[i] = float(y_true[mask].mean())

        if n > 0:
            ece += (bin_size / n) * abs(mean_predicted[i] - fraction_positive[i])

    return {
        "bin_centers": bin_centers,
        "mean_predicted": mean_predicted,
        "fraction_positive": fraction_positive,
        "bin_counts": bin_counts,
        "ece": float(ece),
    }


# ---------------------------------------------------------------------------
# ece_time_series
# ---------------------------------------------------------------------------

def ece_time_series(
    predictions_list: list,
    labels_list: list,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Compute ECE at each time step or fold.

    Parameters
    ----------
    predictions_list : list of np.ndarray
        Predicted probabilities for each time step.
    labels_list : list of np.ndarray
        Ground-truth labels for each time step.
    n_bins : int
        Number of bins for ECE computation (default 10).

    Returns
    -------
    np.ndarray, shape (T,)
        ECE value for each time step.

    Raises
    ------
    ValueError
        If ``predictions_list`` and ``labels_list`` have different lengths,
        or if either is empty.
    """
    if len(predictions_list) != len(labels_list):
        raise ValueError(
            f"predictions_list has {len(predictions_list)} elements but "
            f"labels_list has {len(labels_list)} elements."
        )
    if len(predictions_list) == 0:
        raise ValueError("predictions_list and labels_list must not be empty.")

    eces = []
    for y_pred, y_true in zip(predictions_list, labels_list):
        data = reliability_diagram_data(
            np.asarray(y_pred, dtype=float),
            np.asarray(y_true, dtype=float),
            n_bins=n_bins,
        )
        eces.append(data["ece"])

    return np.array(eces, dtype=float)


# ---------------------------------------------------------------------------
# sharpness
# ---------------------------------------------------------------------------

def sharpness(y_pred: np.ndarray) -> float:
    """
    Compute sharpness as mean p*(1-p).

    Sharpness measures how decisive (extreme) the predicted probabilities are.
    Lower values indicate sharper (more confident) predictions.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted probabilities in [0, 1].

    Returns
    -------
    float
        Mean of p*(1-p) across all predictions.  Returns 0.0 for empty array.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred * (1.0 - y_pred)))


# ---------------------------------------------------------------------------
# brier_skill_score
# ---------------------------------------------------------------------------

def brier_skill_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Brier Skill Score (BSS).

    BSS = 1 - (BS_model / BS_climatology)

    where BS_climatology = mean(y_true) * (1 - mean(y_true)) is the Brier
    score of the naive climatology forecast.

    Parameters
    ----------
    y_pred : np.ndarray, shape (N,)
        Model predicted probabilities.
    y_true : np.ndarray, shape (N,)
        Binary ground-truth labels.

    Returns
    -------
    float
        BSS in (-∞, 1].  Returns 0.0 if BS_climatology is zero
        (all labels are the same).
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    base_rate = float(y_true.mean())
    bs_climatology = base_rate * (1.0 - base_rate)

    if bs_climatology == 0.0:
        return 0.0

    bs_model = float(np.mean((y_pred - y_true) ** 2))
    return float(1.0 - bs_model / bs_climatology)


# ---------------------------------------------------------------------------
# compute_overround
# ---------------------------------------------------------------------------

def compute_overround(probs: np.ndarray) -> float:
    """
    Compute the overround (sum of market probabilities).

    In a fair market this equals 1.0.  Bookmakers build in vig so the sum
    exceeds 1.0; this function returns that sum.

    Parameters
    ----------
    probs : np.ndarray
        Array of implied probabilities from a betting market.

    Returns
    -------
    float
        Sum of all probabilities in the market.
    """
    return float(np.sum(np.asarray(probs, dtype=float)))
