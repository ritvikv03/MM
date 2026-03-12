"""
tests/model/test_market_loss.py

Test suite for src/model/loss/market_loss.py.

All expected values are derived analytically from the formula:
    Loss = (y_pred - y_true)^2 * (1 + lambda * |y_pred - p_close|)

No synthetic / guessed results — every fixture is computed from the math.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Import under test — will fail (RED) until the module is created
# ---------------------------------------------------------------------------
from src.model.loss.market_loss import (
    compute_clv_weight,
    MarketAlignedLoss,
    batch_market_loss,
    clv_decomposition,
)


# ===========================================================================
# compute_clv_weight
# ===========================================================================

class TestComputeClvWeight:
    """compute_clv_weight(y_pred, p_close, lam=1.0) -> float"""

    def test_no_penalty_when_equal(self):
        """When y_pred == p_close the weight is exactly 1.0 (no CLV term)."""
        assert compute_clv_weight(0.6, 0.6) == pytest.approx(1.0)

    def test_no_penalty_at_zero(self):
        assert compute_clv_weight(0.0, 0.0) == pytest.approx(1.0)

    def test_no_penalty_at_one(self):
        assert compute_clv_weight(1.0, 1.0) == pytest.approx(1.0)

    def test_formula_positive_deviation(self):
        # |0.7 - 0.5| = 0.2; lam=1.0 → weight = 1 + 1.0*0.2 = 1.2
        assert compute_clv_weight(0.7, 0.5, lam=1.0) == pytest.approx(1.2)

    def test_formula_negative_deviation(self):
        # |0.3 - 0.5| = 0.2; lam=1.0 → weight = 1.2 (absolute value)
        assert compute_clv_weight(0.3, 0.5, lam=1.0) == pytest.approx(1.2)

    def test_formula_custom_lam(self):
        # |0.8 - 0.5| = 0.3; lam=2.0 → weight = 1 + 2.0*0.3 = 1.6
        assert compute_clv_weight(0.8, 0.5, lam=2.0) == pytest.approx(1.6)

    def test_lam_zero_always_one(self):
        """lam=0 means no CLV penalty regardless of deviation."""
        assert compute_clv_weight(0.9, 0.1, lam=0.0) == pytest.approx(1.0)

    def test_weight_always_ge_one(self):
        """Weight must be >= 1 for any valid inputs."""
        for yp, pc in [(0.1, 0.9), (0.5, 0.5), (0.0, 1.0), (1.0, 0.0)]:
            assert compute_clv_weight(yp, pc) >= 1.0

    def test_invalid_lam_negative(self):
        with pytest.raises(ValueError, match="lam"):
            compute_clv_weight(0.5, 0.5, lam=-0.1)

    def test_invalid_y_pred_below_zero(self):
        with pytest.raises(ValueError):
            compute_clv_weight(-0.01, 0.5)

    def test_invalid_y_pred_above_one(self):
        with pytest.raises(ValueError):
            compute_clv_weight(1.01, 0.5)

    def test_invalid_p_close_below_zero(self):
        with pytest.raises(ValueError):
            compute_clv_weight(0.5, -0.01)

    def test_invalid_p_close_above_one(self):
        with pytest.raises(ValueError):
            compute_clv_weight(0.5, 1.01)

    def test_boundary_values_accepted(self):
        """Exact 0.0 and 1.0 are valid."""
        assert compute_clv_weight(0.0, 1.0, lam=1.0) == pytest.approx(2.0)
        assert compute_clv_weight(1.0, 0.0, lam=1.0) == pytest.approx(2.0)


# ===========================================================================
# MarketAlignedLoss.__call__  (single-game)
# ===========================================================================

class TestMarketAlignedLossCall:
    """MarketAlignedLoss(lam).__call__(y_pred, y_true, p_close) -> float"""

    def test_reduces_to_brier_when_on_market(self):
        """If y_pred == p_close, the CLV term vanishes → pure Brier Score."""
        loss = MarketAlignedLoss(lam=1.0)
        y_pred, y_true, p_close = 0.7, 1.0, 0.7
        # Brier = (0.7 - 1.0)^2 = 0.09
        assert loss(y_pred, y_true, p_close) == pytest.approx(0.09)

    def test_amplified_when_diverging(self):
        """Diverging from market amplifies the loss."""
        loss = MarketAlignedLoss(lam=1.0)
        # y_pred=0.7, y_true=1.0, p_close=0.5
        # brier = (0.7 - 1.0)^2 = 0.09
        # weight = 1 + 1.0 * |0.7 - 0.5| = 1.2
        # total = 0.09 * 1.2 = 0.108
        assert loss(0.7, 1.0, 0.5) == pytest.approx(0.108)

    def test_perfect_prediction_zero_loss(self):
        """Perfect prediction (y_pred = y_true) → zero loss regardless of p_close."""
        loss = MarketAlignedLoss(lam=1.0)
        assert loss(1.0, 1.0, 0.5) == pytest.approx(0.0)
        assert loss(0.0, 0.0, 0.5) == pytest.approx(0.0)

    def test_loss_nonneg(self):
        """Loss must be non-negative for all valid inputs."""
        loss = MarketAlignedLoss(lam=2.0)
        for yp, yt, pc in [(0.3, 1.0, 0.6), (0.9, 0.0, 0.5), (0.5, 0.5, 0.5)]:
            assert loss(yp, yt, pc) >= 0.0

    def test_default_lam_one(self):
        """Default lam=1.0 in the constructor."""
        loss = MarketAlignedLoss()
        # y_pred=0.6, y_true=0.0, p_close=0.5
        # brier = 0.36; weight = 1 + 0.1 = 1.1; total = 0.396
        assert loss(0.6, 0.0, 0.5) == pytest.approx(0.396)

    def test_lam_zero_pure_brier(self):
        """lam=0 → loss equals Brier Score regardless of p_close."""
        loss = MarketAlignedLoss(lam=0.0)
        y_pred, y_true, p_close = 0.8, 1.0, 0.2
        expected = (0.8 - 1.0) ** 2  # 0.04
        assert loss(y_pred, y_true, p_close) == pytest.approx(expected)

    def test_high_lam_amplifies_more(self):
        """Higher lam → higher penalty for same CLV deviation."""
        base = MarketAlignedLoss(lam=1.0)
        high = MarketAlignedLoss(lam=5.0)
        args = (0.7, 1.0, 0.5)
        assert high(*args) > base(*args)

    def test_invalid_inputs_raise(self):
        loss = MarketAlignedLoss()
        with pytest.raises(ValueError):
            loss(-0.1, 0.5, 0.5)
        with pytest.raises(ValueError):
            loss(0.5, 1.1, 0.5)
        with pytest.raises(ValueError):
            loss(0.5, 0.5, -0.1)

    def test_constructor_rejects_negative_lam(self):
        with pytest.raises(ValueError, match="lam"):
            MarketAlignedLoss(lam=-1.0)

    def test_known_values_lam2(self):
        """Spot-check with lam=2 for an explicit hand-calculation."""
        loss = MarketAlignedLoss(lam=2.0)
        # y_pred=0.4, y_true=1.0, p_close=0.6
        # brier = (0.4-1.0)^2 = 0.36
        # weight = 1 + 2.0*|0.4-0.6| = 1 + 0.4 = 1.4
        # total = 0.36 * 1.4 = 0.504
        assert loss(0.4, 1.0, 0.6) == pytest.approx(0.504)


# ===========================================================================
# MarketAlignedLoss.gradient
# ===========================================================================

class TestMarketAlignedLossGradient:
    """gradient(y_pred, y_true, p_close) -> float — analytical dL/dy_pred"""

    def _finite_diff(self, loss_fn, y_pred, y_true, p_close, eps=1e-7):
        """Central-difference approximation of the gradient."""
        return (loss_fn(y_pred + eps, y_true, p_close) -
                loss_fn(y_pred - eps, y_true, p_close)) / (2 * eps)

    def test_gradient_matches_finite_diff_below_market(self):
        """Analytical gradient matches finite-difference when y_pred < p_close."""
        loss = MarketAlignedLoss(lam=1.0)
        y_pred, y_true, p_close = 0.4, 1.0, 0.6  # y_pred < p_close
        analytic = loss.gradient(y_pred, y_true, p_close)
        numeric = self._finite_diff(loss, y_pred, y_true, p_close)
        assert analytic == pytest.approx(numeric, abs=1e-5)

    def test_gradient_matches_finite_diff_above_market(self):
        """Analytical gradient matches finite-difference when y_pred > p_close."""
        loss = MarketAlignedLoss(lam=1.0)
        y_pred, y_true, p_close = 0.8, 0.0, 0.5  # y_pred > p_close
        analytic = loss.gradient(y_pred, y_true, p_close)
        numeric = self._finite_diff(loss, y_pred, y_true, p_close)
        assert analytic == pytest.approx(numeric, abs=1e-5)

    def test_gradient_matches_finite_diff_on_market(self):
        """When y_pred == p_close, CLV sign term vanishes; gradient = 2*(y_pred-y_true)."""
        loss = MarketAlignedLoss(lam=1.0)
        y_pred, y_true, p_close = 0.6, 1.0, 0.6
        analytic = loss.gradient(y_pred, y_true, p_close)
        # At equality gradient = 2*(0.6-1.0)*(1+0) = -0.8
        assert analytic == pytest.approx(-0.8, abs=1e-5)

    def test_gradient_zero_at_perfect_on_market(self):
        """y_pred = y_true = p_close → gradient is 0."""
        loss = MarketAlignedLoss(lam=1.0)
        assert loss.gradient(0.7, 0.7, 0.7) == pytest.approx(0.0, abs=1e-10)

    def test_gradient_finite_diff_custom_lam(self):
        loss = MarketAlignedLoss(lam=3.0)
        y_pred, y_true, p_close = 0.35, 0.0, 0.65
        analytic = loss.gradient(y_pred, y_true, p_close)
        numeric = self._finite_diff(loss, y_pred, y_true, p_close)
        assert analytic == pytest.approx(numeric, abs=1e-5)

    def test_gradient_invalid_inputs_raise(self):
        loss = MarketAlignedLoss()
        with pytest.raises(ValueError):
            loss.gradient(1.5, 0.5, 0.5)


# ===========================================================================
# batch_market_loss
# ===========================================================================

class TestBatchMarketLoss:
    """batch_market_loss(y_pred, y_true, p_close, lam=1.0) -> float"""

    def test_single_element_matches_scalar(self):
        """Batch of one should match the scalar MarketAlignedLoss."""
        loss = MarketAlignedLoss(lam=1.0)
        y_pred = np.array([0.7])
        y_true = np.array([1.0])
        p_close = np.array([0.5])
        assert batch_market_loss(y_pred, y_true, p_close) == pytest.approx(
            loss(0.7, 1.0, 0.5)
        )

    def test_mean_aggregation(self):
        """Result is the arithmetic mean of per-game losses."""
        lam = 1.0
        loss_fn = MarketAlignedLoss(lam=lam)
        pairs = [(0.6, 1.0, 0.5), (0.8, 0.0, 0.7), (0.3, 1.0, 0.4)]
        y_pred = np.array([p[0] for p in pairs])
        y_true = np.array([p[1] for p in pairs])
        p_close = np.array([p[2] for p in pairs])
        expected = np.mean([loss_fn(*p) for p in pairs])
        assert batch_market_loss(y_pred, y_true, p_close, lam=lam) == pytest.approx(expected)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            batch_market_loss(
                np.array([0.5, 0.6]),
                np.array([1.0]),
                np.array([0.5, 0.6]),
            )

    def test_shape_mismatch_p_close_raises(self):
        with pytest.raises(ValueError, match="shape"):
            batch_market_loss(
                np.array([0.5, 0.6]),
                np.array([1.0, 0.0]),
                np.array([0.5]),
            )

    def test_out_of_range_y_pred_raises(self):
        with pytest.raises(ValueError):
            batch_market_loss(
                np.array([1.5, 0.5]),
                np.array([1.0, 0.0]),
                np.array([0.5, 0.5]),
            )

    def test_all_perfect_predictions_zero_loss(self):
        """All y_pred == y_true → loss is 0 regardless of p_close."""
        y_pred = np.array([0.3, 0.7, 0.5])
        y_true = np.array([0.3, 0.7, 0.5])
        p_close = np.array([0.9, 0.1, 0.8])
        assert batch_market_loss(y_pred, y_true, p_close) == pytest.approx(0.0)

    def test_nonneg_output(self):
        """Batch loss must always be non-negative."""
        rng = np.random.default_rng(42)
        y_pred = rng.random(50)
        y_true = rng.integers(0, 2, 50).astype(float)
        p_close = rng.random(50)
        assert batch_market_loss(y_pred, y_true, p_close) >= 0.0

    def test_higher_lam_increases_loss_when_diverging(self):
        """For inputs diverging from market, higher lam should increase mean loss."""
        y_pred = np.array([0.9, 0.1])
        y_true = np.array([0.0, 1.0])
        p_close = np.array([0.5, 0.5])
        low = batch_market_loss(y_pred, y_true, p_close, lam=0.5)
        high = batch_market_loss(y_pred, y_true, p_close, lam=2.0)
        assert high > low


# ===========================================================================
# clv_decomposition
# ===========================================================================

class TestClvDecomposition:
    """clv_decomposition(y_pred, y_true, p_close) -> dict"""

    def _expected_decomp(self, y_pred, y_true, p_close, lam=1.0):
        """Reference implementation derived directly from formula."""
        brier = float(np.mean((y_pred - y_true) ** 2))
        total = float(np.mean(
            (y_pred - y_true) ** 2 * (1 + lam * np.abs(y_pred - p_close))
        ))
        clv_penalty = total - brier
        clv_lift_pct = 0.0 if brier == 0.0 else clv_penalty / brier * 100.0
        return {"brier": brier, "clv_penalty": clv_penalty, "total": total,
                "clv_lift_pct": clv_lift_pct}

    def test_keys_present(self):
        y_pred = np.array([0.6, 0.4])
        y_true = np.array([1.0, 0.0])
        p_close = np.array([0.5, 0.5])
        result = clv_decomposition(y_pred, y_true, p_close)
        assert set(result.keys()) == {"brier", "clv_penalty", "total", "clv_lift_pct"}

    def test_clv_lift_zero_when_on_market(self):
        """When y_pred == p_close everywhere, clv_lift_pct must be 0."""
        y_pred = np.array([0.7, 0.3, 0.6])
        y_true = np.array([1.0, 0.0, 1.0])
        p_close = y_pred.copy()
        result = clv_decomposition(y_pred, y_true, p_close)
        assert result["clv_lift_pct"] == pytest.approx(0.0)

    def test_brier_component_correct(self):
        """brier key equals plain mean-squared error."""
        y_pred = np.array([0.6, 0.8, 0.3])
        y_true = np.array([1.0, 0.0, 1.0])
        p_close = np.array([0.5, 0.5, 0.5])
        result = clv_decomposition(y_pred, y_true, p_close)
        expected_brier = float(np.mean((y_pred - y_true) ** 2))
        assert result["brier"] == pytest.approx(expected_brier)

    def test_total_equals_brier_plus_penalty(self):
        y_pred = np.array([0.7, 0.2])
        y_true = np.array([1.0, 1.0])
        p_close = np.array([0.5, 0.5])
        result = clv_decomposition(y_pred, y_true, p_close)
        assert result["total"] == pytest.approx(result["brier"] + result["clv_penalty"])

    def test_clv_penalty_nonneg(self):
        """CLV penalty is always non-negative."""
        rng = np.random.default_rng(99)
        y_pred = rng.random(30)
        y_true = rng.integers(0, 2, 30).astype(float)
        p_close = rng.random(30)
        result = clv_decomposition(y_pred, y_true, p_close)
        assert result["clv_penalty"] >= -1e-10  # allow tiny float error

    def test_known_values(self):
        """Compare against reference implementation for a small batch."""
        y_pred = np.array([0.7, 0.4, 0.9])
        y_true = np.array([1.0, 0.0, 1.0])
        p_close = np.array([0.5, 0.6, 0.8])
        result = clv_decomposition(y_pred, y_true, p_close)
        expected = self._expected_decomp(y_pred, y_true, p_close)
        for key in expected:
            assert result[key] == pytest.approx(expected[key], rel=1e-6)

    def test_all_perfect_zero_brier_zero_lift(self):
        """Perfect predictions → brier=0, total=0, lift=0."""
        y_pred = np.array([1.0, 0.0, 1.0])
        y_true = np.array([1.0, 0.0, 1.0])
        p_close = np.array([0.5, 0.5, 0.5])
        result = clv_decomposition(y_pred, y_true, p_close)
        assert result["brier"] == pytest.approx(0.0)
        assert result["total"] == pytest.approx(0.0)
        assert result["clv_lift_pct"] == pytest.approx(0.0)
