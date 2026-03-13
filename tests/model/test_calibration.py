"""Tests for IsotonicCalibrator and brier_score."""
import numpy as np
import pytest
import tempfile
from pathlib import Path
from src.model.calibration import IsotonicCalibrator, brier_score


def test_brier_score_perfect():
    probs = np.array([1.0, 0.0, 1.0, 0.0])
    outcomes = np.array([1, 0, 1, 0])
    assert brier_score(probs, outcomes) == pytest.approx(0.0)


def test_brier_score_worst():
    probs = np.array([0.0, 1.0])
    outcomes = np.array([1, 0])
    assert brier_score(probs, outcomes) == pytest.approx(1.0)


def test_brier_score_uniform():
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    outcomes = np.array([1, 0, 1, 0])
    assert brier_score(probs, outcomes) == pytest.approx(0.25)


def test_calibrator_fit_predict_shape():
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    calibrated = cal.predict(raw)
    assert calibrated.shape == raw.shape


def test_calibrator_output_in_bounds():
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 1, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    calibrated = cal.predict(raw)
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibrator_not_fitted_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.predict(np.array([0.5, 0.6]))


def test_calibrator_save_load_joblib():
    """Serialization uses joblib (sklearn-recommended, safe serializer)."""
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 100)
    outcomes = (rng.uniform(0, 1, 100) < raw).astype(int)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "calibrator.joblib"
        cal.save(path)
        assert path.exists()
        loaded = IsotonicCalibrator.load(path)
    np.testing.assert_array_almost_equal(cal.predict(raw), loaded.predict(raw))


def test_calibrator_fit_returns_self():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 50)
    outcomes = (rng.uniform(0, 1, 50) < raw).astype(int)
    cal = IsotonicCalibrator()
    result = cal.fit(raw, outcomes)
    assert result is cal
