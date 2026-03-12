"""
tests/model/test_fusion_engine.py

TDD test suite for CFA Fusion Engine.
Reference: Hsu et al. (2005/2006) Combinatorial Fusion Analysis.

All test data uses randomly generated numeric arrays — no hardcoded team names.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _make_predictions(n_models: int, n_games: int, rng=None) -> np.ndarray:
    """Return shape (n_models, n_games) float array of fake probabilities."""
    r = rng or RNG
    return r.uniform(0.0, 1.0, size=(n_models, n_games))


def _make_y_true(n_games: int, rng=None) -> np.ndarray:
    """Return shape (n_games,) binary array."""
    r = rng or RNG
    return r.integers(0, 2, size=n_games).astype(float)


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from src.model.fusion.fusion_engine import (
    CFAFusionEngine,
    rank_models,
    compute_rank_correlation,
    select_diverse_subset,
    _predictions_to_ranks,
)


# ===========================================================================
# Tests: _predictions_to_ranks
# ===========================================================================


class TestPredictionsToRanks:
    def test_shape_preserved(self):
        preds = _make_predictions(3, 10)
        ranks = _predictions_to_ranks(preds)
        assert ranks.shape == (3, 10)

    def test_rank_values_are_valid(self):
        """Each row must contain ranks 1..N_games exactly once."""
        preds = _make_predictions(4, 20)
        ranks = _predictions_to_ranks(preds)
        n_games = preds.shape[1]
        for row in ranks:
            assert set(row.astype(int)) == set(range(1, n_games + 1))

    def test_highest_prob_gets_rank_1(self):
        """Game with highest predicted probability should receive rank 1."""
        preds = np.array([[0.1, 0.9, 0.5]])  # game 1 is highest
        ranks = _predictions_to_ranks(preds)
        assert ranks[0, 1] == 1  # index 1 has highest prob → rank 1

    def test_lowest_prob_gets_rank_n(self):
        preds = np.array([[0.1, 0.9, 0.5]])
        ranks = _predictions_to_ranks(preds)
        n_games = preds.shape[1]
        assert ranks[0, 0] == n_games  # 0.1 is lowest → rank N


# ===========================================================================
# Tests: rank_models
# ===========================================================================


class TestRankModels:
    def test_returns_array_of_indices(self):
        preds = _make_predictions(3, 50)
        y = _make_y_true(50)
        result = rank_models(preds, y)
        assert len(result) == 3
        assert set(result) == {0, 1, 2}

    def test_better_model_ranked_first(self):
        """A perfect model should be ranked before a random one."""
        n_games = 100
        y = _make_y_true(n_games)
        perfect = y.copy()               # model 0: perfect predictions
        random = _make_predictions(1, n_games)[0]  # model 1: random
        preds = np.array([perfect, random])
        result = rank_models(preds, y)
        assert result[0] == 0  # perfect model is best

    def test_worst_model_ranked_last(self):
        n_games = 100
        y = np.ones(n_games)             # all wins
        worst = np.zeros(n_games)        # predicts all losses → terrible Brier
        best = np.ones(n_games) * 0.99   # confident and correct
        medium = np.ones(n_games) * 0.6  # correct but low confidence
        preds = np.array([medium, worst, best])
        result = rank_models(preds, y)
        assert result[-1] == 1  # worst model last

    def test_output_is_array(self):
        preds = _make_predictions(2, 20)
        y = _make_y_true(20)
        result = rank_models(preds, y)
        assert isinstance(result, np.ndarray)

    def test_single_model(self):
        preds = _make_predictions(1, 30)
        y = _make_y_true(30)
        result = rank_models(preds, y)
        assert list(result) == [0]

    def test_consistent_with_brier_score(self):
        """Ordering must match manual Brier Score calculation."""
        rng = np.random.default_rng(0)
        n_games = 80
        y = rng.integers(0, 2, n_games).astype(float)
        preds = rng.uniform(0, 1, (4, n_games))
        brier_scores = [np.mean((preds[i] - y) ** 2) for i in range(4)]
        expected_order = np.argsort(brier_scores)
        result = rank_models(preds, y)
        np.testing.assert_array_equal(result, expected_order)


# ===========================================================================
# Tests: compute_rank_correlation
# ===========================================================================


class TestComputeRankCorrelation:
    def test_identical_arrays_return_1(self):
        ranks = np.array([1, 2, 3, 4, 5], dtype=float)
        corr = compute_rank_correlation(ranks, ranks)
        assert abs(corr - 1.0) < 1e-6

    def test_inversely_ranked_returns_minus1(self):
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([5, 4, 3, 2, 1], dtype=float)
        corr = compute_rank_correlation(a, b)
        assert abs(corr - (-1.0)) < 1e-6

    def test_returns_scalar_float(self):
        a = np.arange(1, 11, dtype=float)
        b = RNG.permutation(10).astype(float) + 1
        corr = compute_rank_correlation(a, b)
        assert isinstance(corr, float)

    def test_in_range_minus1_to_1(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            a = rng.permutation(30).astype(float)
            b = rng.permutation(30).astype(float)
            corr = compute_rank_correlation(a, b)
            assert -1.0 <= corr <= 1.0

    def test_near_zero_for_uncorrelated(self):
        """Large random permutations should have near-zero correlation on average."""
        rng = np.random.default_rng(99)
        correlations = []
        for _ in range(100):
            a = rng.permutation(200).astype(float)
            b = rng.permutation(200).astype(float)
            correlations.append(compute_rank_correlation(a, b))
        assert abs(np.mean(correlations)) < 0.1


# ===========================================================================
# Tests: select_diverse_subset
# ===========================================================================


class TestSelectDiverseSubset:
    def test_returns_list(self):
        preds = _make_predictions(3, 50)
        y = _make_y_true(50)
        result = select_diverse_subset(preds, y)
        assert isinstance(result, list)

    def test_threshold_1_0_selects_all(self):
        """With diversity_threshold=1.0, no pair can exceed it → all models selected."""
        preds = _make_predictions(4, 60)
        y = _make_y_true(60)
        result = select_diverse_subset(preds, y, diversity_threshold=1.0)
        assert set(result) == {0, 1, 2, 3}

    def test_threshold_near_zero_selects_one(self):
        """With diversity_threshold near 0, the second model can almost never meet the bar."""
        rng = np.random.default_rng(21)
        n_games = 200
        y = rng.integers(0, 2, n_games).astype(float)
        # Create 5 models that are all highly correlated with each other
        base = rng.uniform(0, 1, n_games)
        preds = np.array([base + rng.normal(0, 0.001, n_games) for _ in range(5)])
        preds = np.clip(preds, 0.0, 1.0)
        # With a very small threshold, nearly all pairs will be rejected → only 1 selected
        result = select_diverse_subset(preds, y, diversity_threshold=1e-9)
        assert len(result) == 1

    def test_near_identical_models_dropped(self):
        """Two models with correlation > 0.95 should not both be selected at default threshold."""
        rng = np.random.default_rng(1)
        n_games = 200
        y = rng.integers(0, 2, n_games).astype(float)
        base = rng.uniform(0, 1, n_games)
        # model 0: best (close to y), model 1: near-identical clone of model 0, model 2: diverse
        model_0 = base.copy()
        model_1 = base + rng.normal(0, 0.001, n_games)  # nearly identical → corr > 0.95
        model_1 = np.clip(model_1, 0, 1)
        model_2 = rng.uniform(0, 1, n_games)              # diverse
        preds = np.array([model_0, model_1, model_2])
        result = select_diverse_subset(preds, y, diversity_threshold=0.95)
        # Both 0 and 1 should NOT both appear
        assert not (0 in result and 1 in result)

    def test_best_model_always_included(self):
        """The best-performing model must always be in the selected subset."""
        rng = np.random.default_rng(5)
        n_games = 100
        y = rng.integers(0, 2, n_games).astype(float)
        preds = rng.uniform(0, 1, (5, n_games))
        brier = [np.mean((preds[i] - y) ** 2) for i in range(5)]
        best_idx = int(np.argmin(brier))
        result = select_diverse_subset(preds, y, diversity_threshold=0.95)
        assert best_idx in result

    def test_invalid_threshold_raises(self):
        preds = _make_predictions(3, 20)
        y = _make_y_true(20)
        with pytest.raises(ValueError):
            select_diverse_subset(preds, y, diversity_threshold=0.0)
        with pytest.raises(ValueError):
            select_diverse_subset(preds, y, diversity_threshold=1.1)

    def test_indices_are_valid(self):
        preds = _make_predictions(5, 50)
        y = _make_y_true(50)
        result = select_diverse_subset(preds, y)
        for idx in result:
            assert 0 <= idx < 5


# ===========================================================================
# Tests: CFAFusionEngine
# ===========================================================================


class TestCFAFusionEngineInit:
    def test_default_params(self):
        engine = CFAFusionEngine()
        assert engine.diversity_threshold == 0.95
        assert engine.aggregation == "rank_sum"

    def test_custom_params(self):
        engine = CFAFusionEngine(diversity_threshold=0.8, aggregation="rank_sum")
        assert engine.diversity_threshold == 0.8

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError):
            CFAFusionEngine(aggregation="average")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            CFAFusionEngine(diversity_threshold=0.0)
        with pytest.raises(ValueError):
            CFAFusionEngine(diversity_threshold=1.5)


class TestCFAFusionEngineFit:
    def test_fit_sets_selected_indices(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 60)
        y = _make_y_true(60)
        engine.fit(preds, y)
        assert hasattr(engine, "selected_indices_")
        assert isinstance(engine.selected_indices_, list)

    def test_fit_returns_self(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 40)
        y = _make_y_true(40)
        result = engine.fit(preds, y)
        assert result is engine

    def test_fit_invalid_shape_1d_raises(self):
        engine = CFAFusionEngine()
        with pytest.raises(ValueError):
            engine.fit(np.array([0.5, 0.6, 0.7]), np.array([1, 0, 1]))

    def test_fit_mismatched_games_raises(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 50)
        y = _make_y_true(40)  # wrong length
        with pytest.raises(ValueError):
            engine.fit(preds, y)


class TestCFAFusionEnginePredict:
    def test_predict_before_fit_raises(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 30)
        with pytest.raises(ValueError):
            engine.predict(preds)

    def test_predict_shape(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 50)
        y = _make_y_true(50)
        engine.fit(preds, y)
        out = engine.predict(preds)
        assert out.shape == (50,)

    def test_predict_in_range_0_1(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(4, 80)
        y = _make_y_true(80)
        engine.fit(preds, y)
        out = engine.predict(preds)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_predict_invalid_shape_raises(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 50)
        y = _make_y_true(50)
        engine.fit(preds, y)
        with pytest.raises(ValueError):
            engine.predict(np.array([0.5, 0.6]))  # 1D

    def test_predict_higher_prob_for_confident_model(self):
        """Games where the selected model is very confident should yield higher fused prob."""
        rng = np.random.default_rng(13)
        n_games = 100
        y = rng.integers(0, 2, n_games).astype(float)
        # One strong model, two random
        strong = y * 0.95 + (1 - y) * 0.05   # near perfect
        weak1 = rng.uniform(0.4, 0.6, n_games)
        weak2 = rng.uniform(0.4, 0.6, n_games)
        preds = np.array([strong, weak1, weak2])
        engine = CFAFusionEngine(diversity_threshold=1.0)  # keep all
        engine.fit(preds, y)
        out = engine.predict(preds)
        # Fused output should be positively correlated with strong model
        # The 2 weak models dilute the signal, so a modest threshold of 0.3 is used
        corr = np.corrcoef(strong, out)[0, 1]
        assert corr > 0.3


class TestCFAFusionEngineDiversityReport:
    def test_report_has_required_keys(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 50)
        y = _make_y_true(50)
        engine.fit(preds, y)
        report = engine.diversity_report()
        assert "n_models_total" in report
        assert "n_models_selected" in report
        assert "pairwise_correlations" in report
        assert "selected_indices" in report

    def test_report_n_models_total(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(5, 50)
        y = _make_y_true(50)
        engine.fit(preds, y)
        report = engine.diversity_report()
        assert report["n_models_total"] == 5

    def test_report_selected_matches_selected_indices(self):
        engine = CFAFusionEngine()
        preds = _make_predictions(4, 50)
        y = _make_y_true(50)
        engine.fit(preds, y)
        report = engine.diversity_report()
        assert report["n_models_selected"] == len(engine.selected_indices_)
        assert report["selected_indices"] == engine.selected_indices_

    def test_report_pairwise_correlations_format(self):
        """Keys must be 'i-j' strings; values must be floats in [-1, 1]."""
        engine = CFAFusionEngine()
        preds = _make_predictions(3, 60)
        y = _make_y_true(60)
        engine.fit(preds, y)
        report = engine.diversity_report()
        corrs = report["pairwise_correlations"]
        assert isinstance(corrs, dict)
        for key, val in corrs.items():
            parts = key.split("-")
            assert len(parts) == 2
            assert parts[0].isdigit() and parts[1].isdigit()
            assert -1.0 <= val <= 1.0

    def test_report_before_fit_raises(self):
        engine = CFAFusionEngine()
        with pytest.raises(ValueError):
            engine.diversity_report()


# ===========================================================================
# Integration test
# ===========================================================================


class TestCFAIntegration:
    def test_near_identical_pair_excludes_duplicate(self):
        """
        3 models: 0 and 1 are near-identical (corr > 0.95), 2 is diverse.
        At default threshold, only 2 out of 3 should be selected, and both 0 and 1
        should not appear together.
        """
        rng = np.random.default_rng(77)
        n_games = 300
        y = rng.integers(0, 2, n_games).astype(float)

        base = rng.uniform(0, 1, n_games)
        model_0 = base.copy()
        model_1 = base + rng.normal(0, 0.002, n_games)
        model_1 = np.clip(model_1, 0.0, 1.0)
        model_2 = rng.uniform(0, 1, n_games)

        preds = np.array([model_0, model_1, model_2])
        engine = CFAFusionEngine(diversity_threshold=0.95)
        engine.fit(preds, y)

        selected = engine.selected_indices_
        assert len(selected) <= 2
        assert not (0 in selected and 1 in selected)

    def test_end_to_end_pipeline(self):
        """fit → predict → diversity_report without errors."""
        rng = np.random.default_rng(42)
        n_games = 64
        y = rng.integers(0, 2, n_games).astype(float)
        preds = rng.uniform(0, 1, (4, n_games))

        engine = CFAFusionEngine()
        engine.fit(preds, y)
        out = engine.predict(preds)
        report = engine.diversity_report()

        assert out.shape == (n_games,)
        assert report["n_models_total"] == 4

    def test_single_model_pipeline(self):
        """Edge case: single model should be selected and predict correctly."""
        rng = np.random.default_rng(3)
        n_games = 30
        y = rng.integers(0, 2, n_games).astype(float)
        preds = rng.uniform(0, 1, (1, n_games))

        engine = CFAFusionEngine()
        engine.fit(preds, y)
        assert engine.selected_indices_ == [0]
        out = engine.predict(preds)
        assert out.shape == (n_games,)
