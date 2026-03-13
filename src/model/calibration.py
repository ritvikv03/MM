"""Isotonic regression calibration for Bayesian posterior win probabilities.

Fits a monotone calibration curve on historical Brier scores (2012-2024)
to correct overconfident chalk predictions from the Bayesian ADVI head.

References:
    Zadrozny & Elkan (2002) — Transforming classifier scores into accurate
    multiclass probability estimates.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

_DEFAULT_PATH = Path("artifacts/calibrator.joblib")


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean Brier Score: E[(p - y)^2]. Range [0, 1]. Lower is better."""
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression for win-probability calibration.

    Guarantees a monotone non-decreasing mapping from raw posterior
    probabilities to calibrated probabilities — higher raw always maps
    to higher calibrated.

    Persistence uses joblib (sklearn-recommended serializer), which is
    safe and efficient for numpy arrays.

    Law of Large Numbers note: over a 35-game NCAA season, close-game
    outcomes revert strongly to mean. The Bayesian prior shrinks clutch
    metrics; this calibrator corrects any residual overconfidence in
    chalk predictions post-ADVI.
    """

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> "IsotonicCalibrator":
        """Fit calibration curve on historical (raw_prob, outcome) pairs.

        Args:
            raw_probs: Raw win probabilities from Bayesian head (0.0-1.0).
            outcomes:  Binary game outcomes (1 = home/team_a win, 0 = loss).

        Returns:
            self (for chaining)
        """
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(raw_probs), np.asarray(outcomes))
        return self

    def predict(self, raw_probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities for raw_probs.

        Raises:
            RuntimeError: if .fit() has not been called.
        """
        if self._iso is None:
            raise RuntimeError(
                "IsotonicCalibrator is not fitted. Call .fit() before .predict()."
            )
        return self._iso.predict(np.asarray(raw_probs))

    def save(self, path: Path | str = _DEFAULT_PATH) -> None:
        """Persist the calibrator to disk using joblib.

        joblib is sklearn's recommended serializer — safe, efficient,
        and handles numpy arrays well.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._iso, path)

    @classmethod
    def load(cls, path: Path | str = _DEFAULT_PATH) -> "IsotonicCalibrator":
        """Load a previously saved calibrator from path."""
        obj = cls()
        obj._iso = joblib.load(Path(path))
        return obj
