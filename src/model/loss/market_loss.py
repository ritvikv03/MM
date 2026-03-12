"""
src/model/loss/market_loss.py

Market-Aligned Loss Function for the NCAA March Madness AlphaMarch engine.

Loss = (y_pred - y_true)² × (1 + λ × |y_pred - p_close|)

where p_close is the implied probability from the closing market line.

Design rationale
----------------
Standard Brier Score treats all prediction errors equally. This loss penalises
predictions that deviate from efficient-market consensus (closing lines), which
are the sharpest probability estimates available. A model should only deviate
from the market when it has genuine information edge; unnecessary deviation is
penalised by the CLV term.

References
----------
- Kelly, J.L. (1956): A New Interpretation of Information Rate.
- Buchdahl, J. (2003): Fixed-Odds Sports Betting.
- Closing Line Value (CLV) as the gold-standard accuracy metric for sharp bettors.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_prob(value: float, name: str) -> None:
    """Raise ValueError if value is outside [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _validate_lam(lam: float) -> None:
    """Raise ValueError if lam is negative."""
    if lam < 0.0:
        raise ValueError(f"lam must be >= 0, got {lam!r}")


def _validate_prob_array(arr: np.ndarray, name: str) -> None:
    """Raise ValueError if any element of arr is outside [0, 1]."""
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} elements must be in [0, 1]")


def _validate_shapes(*arrays: np.ndarray) -> None:
    """Raise ValueError if arrays do not all share the same shape."""
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) > 1:
        raise ValueError(
            f"All arrays must have the same shape; got shapes: {shapes}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_clv_weight(y_pred: float, p_close: float, lam: float = 1.0) -> float:
    """Return the CLV multiplier ``(1 + λ × |y_pred - p_close|)``.

    Accepts scalar inputs only; for batch use, see batch_market_loss.

    Parameters
    ----------
    y_pred:
        Model-predicted win probability in [0, 1].
    p_close:
        Closing-line implied probability in [0, 1].
    lam:
        CLV sensitivity weight (must be >= 0).

    Returns
    -------
    float
        Multiplier >= 1.  Equals 1.0 when y_pred == p_close (no CLV
        deviation).
    """
    _validate_lam(lam)
    _validate_prob(y_pred, "y_pred")
    _validate_prob(p_close, "p_close")
    return 1.0 + lam * abs(y_pred - p_close)


class MarketAlignedLoss:
    """CLV-penalised training loss for a single game.

    Loss = (y_pred - y_true)² × (1 + λ × |y_pred - p_close|)

    When y_pred == p_close the CLV term vanishes and the loss reduces to the
    standard Brier Score.  Deviation from the efficient closing line is
    penalised in proportion to λ.

    Parameters
    ----------
    lam:
        CLV sensitivity weight (must be >= 0).  Default 1.0.
    """

    def __init__(self, lam: float = 1.0) -> None:
        _validate_lam(lam)
        self.lam = lam

    def __call__(self, y_pred: float, y_true: float, p_close: float) -> float:
        """Compute the market-aligned loss for a single game.

        Parameters
        ----------
        y_pred:
            Predicted win probability in [0, 1].
        y_true:
            Binary game outcome (0 or 1) in [0, 1].
        p_close:
            Closing-line implied probability in [0, 1].

        Returns
        -------
        float
            Non-negative loss value.
        """
        _validate_prob(y_pred, "y_pred")
        _validate_prob(y_true, "y_true")
        _validate_prob(p_close, "p_close")
        brier = (y_pred - y_true) ** 2
        weight = 1.0 + self.lam * abs(y_pred - p_close)
        return brier * weight

    def gradient(self, y_pred: float, y_true: float, p_close: float) -> float:
        """Analytical gradient dL/dy_pred.

        dL/dy_pred = 2(y_pred - y_true)(1 + λ|y_pred - p_close|)
                   + (y_pred - y_true)² × λ × sign(y_pred - p_close)

        Parameters
        ----------
        y_pred:
            Predicted win probability in [0, 1].
        y_true:
            Binary game outcome (0 or 1) in [0, 1].
        p_close:
            Closing-line implied probability in [0, 1].

        Returns
        -------
        float
            Gradient with respect to y_pred.
            Note: the returned gradient may push y_pred outside [0, 1] —
            clamping is the optimizer's responsibility.
        """
        _validate_prob(y_pred, "y_pred")
        _validate_prob(y_true, "y_true")
        _validate_prob(p_close, "p_close")
        diff = y_pred - y_true
        clv_diff = y_pred - p_close
        weight = 1.0 + self.lam * abs(clv_diff)
        sign = np.sign(clv_diff)
        return 2.0 * diff * weight + diff ** 2 * self.lam * sign


def batch_market_loss(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    p_close: np.ndarray,
    lam: float = 1.0,
) -> float:
    """Vectorised mean market-aligned loss over a batch of games.

    Parameters
    ----------
    y_pred:
        Array of predicted win probabilities, shape (N,).
    y_true:
        Array of binary game outcomes, shape (N,).
    p_close:
        Array of closing-line implied probabilities, shape (N,).
    lam:
        CLV sensitivity weight (must be >= 0).  Default 1.0.

    Returns
    -------
    float
        Scalar mean loss over the batch.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    p_close = np.asarray(p_close, dtype=float)

    _validate_shapes(y_pred, y_true, p_close)
    _validate_lam(lam)
    _validate_prob_array(y_pred, "y_pred")
    _validate_prob_array(y_true, "y_true")
    _validate_prob_array(p_close, "p_close")

    brier = (y_pred - y_true) ** 2
    weight = 1.0 + lam * np.abs(y_pred - p_close)
    return float(np.mean(brier * weight))


def clv_decomposition(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    p_close: np.ndarray,
    lam: float = 1.0,
) -> dict:
    """Decompose the batch loss into Brier and CLV-penalty components.

    Useful for diagnostics: how much extra loss comes purely from CLV
    deviation versus the baseline Brier Score?

    Parameters
    ----------
    y_pred:
        Array of predicted win probabilities, shape (N,).
    y_true:
        Array of binary game outcomes, shape (N,).
    p_close:
        Array of closing-line implied probabilities, shape (N,).
    lam:
        CLV sensitivity weight.  Default 1.0.

    Returns
    -------
    dict with keys:
        ``brier``      — mean plain Brier Score (no CLV term).
        ``clv_penalty``— mean additional loss introduced by the CLV term.
        ``total``      — mean total market-aligned loss (brier + clv_penalty).
        ``clv_lift_pct``— (clv_penalty / brier) × 100; 0 when brier == 0.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    p_close = np.asarray(p_close, dtype=float)

    _validate_lam(lam)
    _validate_shapes(y_pred, y_true, p_close)
    _validate_prob_array(y_pred, "y_pred")
    _validate_prob_array(y_true, "y_true")
    _validate_prob_array(p_close, "p_close")

    sq_err = (y_pred - y_true) ** 2
    brier = float(np.mean(sq_err))
    total = float(np.mean(sq_err * (1.0 + lam * np.abs(y_pred - p_close))))
    clv_penalty = total - brier
    clv_lift_pct = 0.0 if brier == 0.0 else clv_penalty / brier * 100.0

    return {
        "brier": brier,
        "clv_penalty": clv_penalty,
        "total": total,
        "clv_lift_pct": clv_lift_pct,
    }
