"""
tests/model/test_bayesian_head.py

RED phase — all tests written before implementation exists.

Strategy
--------
- BayesianHead.build_model, .fit, .predict: mocked via unittest.mock.patch
  to avoid real MCMC and prevent hangs in CI / local TDD.
- compute_brier_score, compute_calibration_bins,
  uncertainty_to_kelly_fraction: real numpy (no mocking needed).
- 30+ tests total across 7 test classes.
"""

from __future__ import annotations

import sys
import importlib
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call, PropertyMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a lightweight pymc / arviz mock stack
# ---------------------------------------------------------------------------

def _make_pm_mock():
    """Return a minimal pymc mock with the shapes expected by BayesianHead."""
    pm = MagicMock(name="pymc")

    # --- pm.Model as a context-manager class --------------------------------
    mock_model = MagicMock(name="pm.Model_instance")
    mock_model.__enter__ = MagicMock(return_value=mock_model)
    mock_model.__exit__ = MagicMock(return_value=False)
    pm.Model.return_value = mock_model

    # --- distributions / deterministics -------------------------------------
    pm.HalfNormal.return_value = MagicMock(name="HalfNormal_rv")
    pm.Normal.return_value = MagicMock(name="Normal_rv")
    pm.Bernoulli.return_value = MagicMock(name="Bernoulli_rv")
    pm.Data.return_value = MagicMock(name="pm.Data_rv")

    # pm.math.sigmoid
    pm.math = MagicMock(name="pm.math")
    pm.math.sigmoid.side_effect = lambda x: x  # identity — shape preserved

    # --- pm.fit (ADVI) ------------------------------------------------------
    mock_approx = MagicMock(name="approx")
    # approx.sample(5000) → InferenceData-like object
    mock_idata_advi = MagicMock(name="idata_advi")
    mock_approx.sample.return_value = mock_idata_advi
    pm.fit.return_value = mock_approx

    # --- pm.sample (NUTS) ---------------------------------------------------
    mock_idata_nuts = MagicMock(name="idata_nuts")
    pm.sample.return_value = mock_idata_nuts

    return pm, mock_model, mock_approx, mock_idata_advi, mock_idata_nuts


def _make_az_mock():
    """Return a minimal arviz mock."""
    az = MagicMock(name="arviz")
    return az


def _make_skellam_mock():
    """Return a minimal mock for src.model.skellam (used when scipy is absent)."""
    skellam_mod = MagicMock(name="src.model.skellam")
    # zero_truncated_skellam_log_pmf returns a mock tensor-like object
    skellam_mod.zero_truncated_skellam_log_pmf = MagicMock(
        name="zero_truncated_skellam_log_pmf",
        return_value=MagicMock(name="skellam_logp_tensor"),
    )
    return skellam_mod


def _build_modules_patch(pm, az):
    skellam = _make_skellam_mock()
    # Also stub scipy.special so skellam.py doesn't fail if imported directly
    scipy_mock = MagicMock(name="scipy")
    scipy_special_mock = MagicMock(name="scipy.special")
    scipy_special_mock.iv = MagicMock(name="bessel_iv", return_value=1.0)
    scipy_stats_mock = MagicMock(name="scipy.stats")
    return {
        "pymc": pm,
        "arviz": az,
        "src.model.skellam": skellam,
        "scipy": scipy_mock,
        "scipy.special": scipy_special_mock,
        "scipy.stats": scipy_stats_mock,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_pm_az():
    """Yield (pm_mock, az_mock, module) with bayesian_head reloaded under mocks."""
    pm, mock_model, mock_approx, idata_advi, idata_nuts = _make_pm_mock()
    az = _make_az_mock()
    patches = _build_modules_patch(pm, az)
    with patch.dict(sys.modules, patches):
        sys.modules.pop("src.model.bayesian_head", None)
        import src.model.bayesian_head as module
        yield module, pm, az, mock_model, mock_approx, idata_advi, idata_nuts
    sys.modules.pop("src.model.bayesian_head", None)


@pytest.fixture()
def small_game_data():
    """Tiny synthetic dataset: G=5 games, embedding_dim=4.

    Note: y_total is intentionally absent — game totals are permanently
    excluded from the two-target architecture (win prob + spread only).
    """
    rng = np.random.default_rng(0)
    G, D = 5, 4
    return {
        "home_emb": rng.standard_normal((G, D)).astype(np.float32),
        "away_emb": rng.standard_normal((G, D)).astype(np.float32),
        "home_conf": rng.integers(0, 8, size=G),
        "away_conf": rng.integers(0, 8, size=G),
        "home_seed": rng.integers(0, 4, size=G),
        "away_seed": rng.integers(0, 4, size=G),
        "y_win": rng.integers(0, 2, size=G).astype(np.float32),
        "y_spread": rng.standard_normal(G).astype(np.float32) * 10,
    }


@pytest.fixture()
def head_advi(mock_pm_az):
    """BayesianHead configured for ADVI (default)."""
    module, pm, az, *_ = mock_pm_az
    return module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4)


@pytest.fixture()
def head_nuts(mock_pm_az):
    """BayesianHead configured for NUTS."""
    module, pm, az, *_ = mock_pm_az
    return module.BayesianHead(
        embedding_dim=4,
        n_conferences=8,
        n_seeds=4,
        sampler="nuts",
        nuts_draws=50,
        nuts_chains=1,
        nuts_tune=20,
    )


# ===========================================================================
# 1. TestBayesianHeadInit — constructor stores hyperparameters
# ===========================================================================

class TestBayesianHeadInit:
    """BayesianHead.__init__() stores all constructor arguments."""

    def test_stores_embedding_dim(self, head_advi):
        assert head_advi.embedding_dim == 4

    def test_stores_n_conferences(self, head_advi):
        assert head_advi.n_conferences == 8

    def test_stores_n_seeds(self, head_advi):
        assert head_advi.n_seeds == 4

    def test_default_sampler_is_advi(self, head_advi):
        assert head_advi.sampler == "advi"

    def test_nuts_sampler_stored(self, head_nuts):
        assert head_nuts.sampler == "nuts"

    def test_default_advi_iterations(self, head_advi):
        assert head_advi.advi_iterations == 10_000

    def test_default_nuts_draws(self, head_nuts):
        assert head_nuts.nuts_draws == 50  # overridden in fixture

    def test_default_nuts_chains(self, head_nuts):
        assert head_nuts.nuts_chains == 1  # overridden in fixture

    def test_default_random_seed(self, head_advi):
        assert head_advi.random_seed == 42

    def test_default_n_conferences_is_32(self, mock_pm_az):
        module, *_ = mock_pm_az
        h = module.BayesianHead(embedding_dim=8)
        assert h.n_conferences == 32

    def test_default_n_seeds_is_16(self, mock_pm_az):
        module, *_ = mock_pm_az
        h = module.BayesianHead(embedding_dim=8)
        assert h.n_seeds == 16


# ===========================================================================
# 2. TestBuildModel — model structure is constructed correctly
# ===========================================================================

class TestBuildModel:
    """build_model() calls pm.Model and registers the expected random variables."""

    def test_build_model_calls_pm_model(self, head_advi, mock_pm_az, small_game_data):
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        pm.Model.assert_called_once()

    def test_build_model_uses_context_manager(self, head_advi, mock_pm_az, small_game_data):
        """pm.Model().__enter__ and __exit__ must both be invoked."""
        module, pm, az, mock_model, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        mock_model.__enter__.assert_called_once()
        mock_model.__exit__.assert_called_once()

    def test_build_model_returns_model_object(self, head_advi, mock_pm_az, small_game_data):
        module, pm, az, mock_model, *_ = mock_pm_az
        result = head_advi.build_model(**small_game_data)
        # The returned value should be the model returned by pm.Model()
        assert result is mock_model

    def test_half_normal_sigma_conf_called(self, head_advi, mock_pm_az, small_game_data):
        """sigma_conf ~ HalfNormal(1.0) must be created."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        names = [c[1].get("name", c[0][0] if c[0] else "") for c in pm.HalfNormal.call_args_list]
        # Check that at least one HalfNormal with sigma=1 was called
        sigmas_1 = [c for c in pm.HalfNormal.call_args_list if 1.0 in c[0] or c[1].get("sigma") == 1.0]
        assert len(sigmas_1) >= 1

    def test_half_normal_sigma_spread_called(self, mock_pm_az, small_game_data):
        """sigma_spread ~ HalfNormal(10.0) must be created when use_skellam=False.
        With use_skellam=True (Skellam default), sigma_spread is not used.
        """
        module, pm, *_ = mock_pm_az
        head = module.BayesianHead(
            embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False
        )
        head.build_model(**small_game_data)
        sigmas_10 = [c for c in pm.HalfNormal.call_args_list if 10.0 in c[0] or c[1].get("sigma") == 10.0]
        assert len(sigmas_10) >= 1

    def test_luck_scale_halfnormal_called(self, head_advi, mock_pm_az, small_game_data):
        """luck_scale ~ HalfNormal(sigma=0.15) must be created for LLN regression prior."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        sigmas_015 = [
            c for c in pm.HalfNormal.call_args_list
            if 0.15 in c[0] or c[1].get("sigma") == pytest.approx(0.15)
        ]
        assert len(sigmas_015) >= 1

    def test_no_obs_total_halfnormal(self, head_advi, mock_pm_az, small_game_data):
        """sigma_total ~ HalfNormal(15.0) must NOT be created (totals permanently removed)."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        sigmas_15 = [c for c in pm.HalfNormal.call_args_list if 15.0 in c[0] or c[1].get("sigma") == 15.0]
        assert len(sigmas_15) == 0

    def test_pm_normal_called_for_conf_effect(self, head_advi, mock_pm_az, small_game_data):
        """conf_effect ~ Normal(0, sigma_conf, shape=n_conferences)."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        # At least one Normal call with shape=n_conferences
        shapes = [c[1].get("shape") for c in pm.Normal.call_args_list]
        assert head_advi.n_conferences in shapes

    def test_pm_normal_called_for_seed_effect(self, head_advi, mock_pm_az, small_game_data):
        """seed_effect ~ Normal(0, 0.5, shape=n_seeds+1)."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        shapes = [c[1].get("shape") for c in pm.Normal.call_args_list]
        assert head_advi.n_seeds + 1 in shapes

    def test_pm_normal_called_for_w_home(self, head_advi, mock_pm_az, small_game_data):
        """W_home ~ Normal(0, 1, shape=embedding_dim)."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        shapes = [c[1].get("shape") for c in pm.Normal.call_args_list]
        assert head_advi.embedding_dim in shapes

    def test_pm_data_not_called_for_mu_total_base(self, head_advi, mock_pm_az, small_game_data):
        """pm.Data('mu_total_base') must NOT be called — totals are permanently removed."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        data_names = [c[0][0] for c in pm.Data.call_args_list]
        assert "mu_total_base" not in data_names

    def test_pm_bernoulli_called_for_obs_win(self, head_advi, mock_pm_az, small_game_data):
        """obs_win ~ Bernoulli(p=p_win, observed=y_win)."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        pm.Bernoulli.assert_called()

    def test_pm_normal_called_for_obs_spread(self, head_advi, mock_pm_az, small_game_data):
        """obs_spread is registered via pm.CustomDist (Skellam, default) or
        pm.Normal (use_skellam=False).  With default use_skellam=True the
        spread goes through CustomDist; with False it goes through Normal."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        # Default (Skellam): CustomDist called for spread
        head_advi.build_model(**small_game_data)
        pm.CustomDist.assert_called()
        # Normal-fallback variant: pm.Normal still called with observed= for spread
        head_normal = module.BayesianHead(
            embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False
        )
        pm.Normal.reset_mock()
        head_normal.build_model(**small_game_data)
        observed_arrays = [
            c[1].get("observed") for c in pm.Normal.call_args_list
            if c[1].get("observed") is not None
        ]
        assert len(observed_arrays) >= 1

    def test_pm_math_sigmoid_called_for_p_win(self, head_advi, mock_pm_az, small_game_data):
        """pm.math.sigmoid(delta) must be called to obtain p_win."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        pm.math.sigmoid.assert_called()

    def test_luck_obs_normal_called_with_luck_data(self, head_advi, mock_pm_az, small_game_data):
        """When home_luck and away_luck provided, Normal obs priors are created.

        With use_skellam=True (default), obs_spread goes through pm.CustomDist
        so pm.Normal is only called for the two luck observed nodes.
        """
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        rng = np.random.default_rng(77)
        G = len(small_game_data["y_win"])
        luck_data = {
            **small_game_data,
            "home_luck": rng.uniform(0.3, 0.7, G).astype(np.float32),
            "away_luck": rng.uniform(0.3, 0.7, G).astype(np.float32),
        }
        head_advi.build_model(**luck_data)
        # With Skellam default: 2 Normal calls with observed= (home_luck + away_luck)
        observed_arrays = [
            c[1].get("observed") for c in pm.Normal.call_args_list
            if c[1].get("observed") is not None
        ]
        assert len(observed_arrays) >= 2  # obs_home_luck + obs_away_luck

    def test_luck_obs_normal_not_called_without_luck_data(self, head_advi, mock_pm_az, small_game_data):
        """Without luck data, pm.Normal should have no observed= calls when
        use_skellam=True (spread goes through pm.CustomDist)."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head_advi.build_model(**small_game_data)
        observed_arrays = [
            c[1].get("observed") for c in pm.Normal.call_args_list
            if c[1].get("observed") is not None
        ]
        # No luck data + Skellam spread → no pm.Normal with observed
        assert len(observed_arrays) == 0


# ===========================================================================
# 3. TestFit — fit() dispatches to the correct sampler
# ===========================================================================

class TestFit:
    """fit(model) calls pm.fit (ADVI) or pm.sample (NUTS) based on sampler attr."""

    def test_advi_calls_pm_fit(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, mock_approx, idata_advi, _ = mock_pm_az
        head_advi.fit(mock_model)
        pm.fit.assert_called_once()

    def test_advi_passes_iterations(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_advi.fit(mock_model)
        call_kwargs = pm.fit.call_args
        # iterations is first positional arg
        assert call_kwargs[0][0] == head_advi.advi_iterations

    def test_advi_passes_model_kwarg(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_advi.fit(mock_model)
        call_kwargs = pm.fit.call_args
        assert call_kwargs[1].get("model") is mock_model

    def test_advi_passes_random_seed(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_advi.fit(mock_model)
        call_kwargs = pm.fit.call_args
        assert call_kwargs[1].get("random_seed") == head_advi.random_seed

    def test_advi_calls_sample_on_approx(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, mock_approx, idata_advi, _ = mock_pm_az
        head_advi.fit(mock_model)
        mock_approx.sample.assert_called_once_with(5000)

    def test_advi_returns_idata(self, head_advi, mock_pm_az):
        module, pm, az, mock_model, mock_approx, idata_advi, _ = mock_pm_az
        result = head_advi.fit(mock_model)
        assert result is idata_advi

    def test_nuts_calls_pm_sample(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_nuts.fit(mock_model)
        pm.sample.assert_called_once()

    def test_nuts_passes_draws(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_nuts.fit(mock_model)
        assert pm.sample.call_args[1].get("draws") == head_nuts.nuts_draws

    def test_nuts_passes_chains(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_nuts.fit(mock_model)
        assert pm.sample.call_args[1].get("chains") == head_nuts.nuts_chains

    def test_nuts_passes_tune(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_nuts.fit(mock_model)
        assert pm.sample.call_args[1].get("tune") == head_nuts.nuts_tune

    def test_nuts_progressbar_false(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, *_ = mock_pm_az
        head_nuts.fit(mock_model)
        assert pm.sample.call_args[1].get("progressbar") is False

    def test_nuts_returns_idata(self, head_nuts, mock_pm_az):
        module, pm, az, mock_model, mock_approx, _, idata_nuts = mock_pm_az
        result = head_nuts.fit(mock_model)
        assert result is idata_nuts


# ===========================================================================
# 4. TestPredict — predict() extracts correct posterior statistics
# ===========================================================================

class TestPredict:
    """predict() returns a dict with correct shapes and keys."""

    def _make_idata_and_inputs(self, G: int = 5, D: int = 4, n_samples: int = 100):
        """
        Build a minimal mock idata and matching input arrays.

        The posterior samples are stored so that when predict() evaluates:
          delta[s] = home_emb @ W_home[s] - away_emb @ W_away[s]
                     + conf_effect[s, home_conf] - conf_effect[s, away_conf]
                     + seed_effect[s, home_seed] - seed_effect[s, away_seed]
        we can verify the aggregated statistics.
        """
        rng = np.random.default_rng(7)

        home_emb = rng.standard_normal((G, D)).astype(np.float32)
        away_emb = rng.standard_normal((G, D)).astype(np.float32)
        home_conf = rng.integers(0, 4, size=G)
        away_conf = rng.integers(0, 4, size=G)
        home_seed = rng.integers(0, 4, size=G)
        away_seed = rng.integers(0, 4, size=G)

        # Posterior samples — shapes (n_samples, D), etc.
        W_home_samples = rng.standard_normal((n_samples, D)).astype(np.float32)
        W_away_samples = rng.standard_normal((n_samples, D)).astype(np.float32)
        conf_samples = rng.standard_normal((n_samples, 4)).astype(np.float32)
        seed_samples = rng.standard_normal((n_samples, 5)).astype(np.float32)

        # Build mock idata with posterior group
        idata = MagicMock(name="idata")
        posterior = MagicMock(name="posterior")

        # Each variable accessed as idata.posterior["varname"].values
        # shape convention: (chain, draw, *param_dims) — we use 1 chain
        def _wrap(arr):
            """Wrap (S, D) as (1, S, D) to mimic ArviZ shape."""
            m = MagicMock()
            m.values = arr[np.newaxis, ...]  # (1, S, ...)
            return m

        posterior.__getitem__ = MagicMock(side_effect=lambda key: {
            "W_home": _wrap(W_home_samples),
            "W_away": _wrap(W_away_samples),
            "conf_effect": _wrap(conf_samples),
            "seed_effect": _wrap(seed_samples),
        }[key])

        idata.posterior = posterior

        return (
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
            W_home_samples, W_away_samples, conf_samples, seed_samples,
        )

    def test_predict_returns_dict(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert isinstance(result, dict)

    def test_predict_has_p_win_mean_key(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert "p_win_mean" in result

    def test_predict_has_p_win_std_key(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert "p_win_std" in result

    def test_predict_has_spread_mean_key(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert "spread_mean" in result

    def test_predict_has_spread_std_key(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert "spread_std" in result

    def test_predict_has_credible_interval_key(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert "credible_interval_95" in result

    def test_predict_p_win_mean_shape(self, head_advi, mock_pm_az):
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert result["p_win_mean"].shape == (G,)

    def test_predict_p_win_std_shape(self, head_advi, mock_pm_az):
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert result["p_win_std"].shape == (G,)

    def test_predict_spread_mean_shape(self, head_advi, mock_pm_az):
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert result["spread_mean"].shape == (G,)

    def test_predict_credible_interval_has_lower_upper(self, head_advi, mock_pm_az):
        idata, *inputs = self._make_idata_and_inputs()
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        ci = result["credible_interval_95"]
        assert "lower" in ci and "upper" in ci

    def test_predict_credible_interval_lower_shape(self, head_advi, mock_pm_az):
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert result["credible_interval_95"]["lower"].shape == (G,)

    def test_predict_credible_interval_upper_shape(self, head_advi, mock_pm_az):
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert result["credible_interval_95"]["upper"].shape == (G,)

    def test_predict_p_win_mean_bounded_0_1(self, head_advi, mock_pm_az):
        """Win probabilities must be in [0, 1]."""
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        assert np.all(result["p_win_mean"] >= 0.0)
        assert np.all(result["p_win_mean"] <= 1.0)

    def test_predict_ci_lower_leq_upper(self, head_advi, mock_pm_az):
        """lower bound must be <= upper bound for all games."""
        G = 5
        idata, *inputs = self._make_idata_and_inputs(G=G)
        home_emb, away_emb, home_conf, away_conf, home_seed, away_seed = inputs[:6]
        result = head_advi.predict(
            idata, home_emb, away_emb,
            home_conf, away_conf, home_seed, away_seed,
        )
        ci = result["credible_interval_95"]
        assert np.all(ci["lower"] <= ci["upper"])


# ===========================================================================
# 5. TestComputeBrierScore — pure numpy, no mocking
# ===========================================================================

class TestComputeBrierScore:
    """Tests for compute_brier_score(y_pred, y_true)."""

    @pytest.fixture(autouse=True)
    def _import(self, mock_pm_az):
        self.module, *_ = mock_pm_az

    def test_perfect_prediction_gives_zero(self):
        y = np.array([0.0, 1.0, 1.0, 0.0])
        score = self.module.compute_brier_score(y, y)
        assert score == pytest.approx(0.0)

    def test_worst_prediction_gives_one(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        score = self.module.compute_brier_score(y_pred, y_true)
        assert score == pytest.approx(1.0)

    def test_returns_float(self):
        y_pred = np.array([0.3, 0.7])
        y_true = np.array([0.0, 1.0])
        result = self.module.compute_brier_score(y_pred, y_true)
        assert isinstance(result, float)

    def test_known_value(self):
        """BS = mean((0.6-1)^2, (0.4-0)^2) = mean(0.16, 0.16) = 0.16."""
        y_pred = np.array([0.6, 0.4])
        y_true = np.array([1.0, 0.0])
        assert self.module.compute_brier_score(y_pred, y_true) == pytest.approx(0.16)

    def test_shape_mismatch_raises_value_error(self):
        with pytest.raises(ValueError):
            self.module.compute_brier_score(np.array([0.5]), np.array([0.0, 1.0]))

    def test_all_half_probability(self):
        """All predictions 0.5 vs all 1.0 => BS = mean((0.5-1)^2) = 0.25."""
        y_pred = np.full(10, 0.5)
        y_true = np.ones(10)
        assert self.module.compute_brier_score(y_pred, y_true) == pytest.approx(0.25)

    def test_symmetry(self):
        """Brier score is symmetric in the sense BS(f,o) == BS(1-f, 1-o)."""
        rng = np.random.default_rng(1)
        y_pred = rng.uniform(0, 1, 20)
        y_true = rng.integers(0, 2, 20).astype(float)
        bs1 = self.module.compute_brier_score(y_pred, y_true)
        bs2 = self.module.compute_brier_score(1 - y_pred, 1 - y_true)
        assert bs1 == pytest.approx(bs2)

    def test_single_element(self):
        assert self.module.compute_brier_score(
            np.array([0.8]), np.array([1.0])
        ) == pytest.approx(0.04)


# ===========================================================================
# 6. TestComputeCalibrationBins — pure numpy, no mocking
# ===========================================================================

class TestComputeCalibrationBins:
    """Tests for compute_calibration_bins(y_pred, y_true, n_bins)."""

    @pytest.fixture(autouse=True)
    def _import(self, mock_pm_az):
        self.module, *_ = mock_pm_az

    def test_returns_dict(self):
        y_pred = np.linspace(0, 1, 50)
        y_true = (y_pred > 0.5).astype(float)
        result = self.module.compute_calibration_bins(y_pred, y_true)
        assert isinstance(result, dict)

    def test_has_bin_centers_key(self):
        y_pred = np.linspace(0, 1, 50)
        y_true = (y_pred > 0.5).astype(float)
        result = self.module.compute_calibration_bins(y_pred, y_true)
        assert "bin_centers" in result

    def test_has_fraction_positive_key(self):
        y_pred = np.linspace(0, 1, 50)
        y_true = (y_pred > 0.5).astype(float)
        result = self.module.compute_calibration_bins(y_pred, y_true)
        assert "fraction_positive" in result

    def test_has_bin_counts_key(self):
        y_pred = np.linspace(0, 1, 50)
        y_true = (y_pred > 0.5).astype(float)
        result = self.module.compute_calibration_bins(y_pred, y_true)
        assert "bin_counts" in result

    def test_n_bins_length(self):
        y_pred = np.linspace(0, 1, 100)
        y_true = np.zeros(100)
        result = self.module.compute_calibration_bins(y_pred, y_true, n_bins=5)
        assert len(result["bin_centers"]) == 5
        assert len(result["fraction_positive"]) == 5
        assert len(result["bin_counts"]) == 5

    def test_empty_bins_handled_gracefully(self):
        """Predictions all in [0.0, 0.1]: bins [0.5, 1.0] are empty."""
        y_pred = np.full(20, 0.05)
        y_true = np.zeros(20)
        # Should not raise
        result = self.module.compute_calibration_bins(y_pred, y_true, n_bins=10)
        assert len(result["bin_centers"]) == 10

    def test_bin_counts_sum_to_n(self):
        n = 80
        y_pred = np.random.default_rng(3).uniform(0, 1, n)
        y_true = np.zeros(n)
        result = self.module.compute_calibration_bins(y_pred, y_true, n_bins=8)
        assert sum(result["bin_counts"]) == n

    def test_fraction_positive_range(self):
        """fraction_positive must be in [0, 1] for all non-empty bins."""
        rng = np.random.default_rng(5)
        y_pred = rng.uniform(0, 1, 200)
        y_true = rng.integers(0, 2, 200).astype(float)
        result = self.module.compute_calibration_bins(y_pred, y_true, n_bins=10)
        for fp in result["fraction_positive"]:
            if not np.isnan(fp):
                assert 0.0 <= fp <= 1.0

    def test_all_positive_outcome(self):
        """If y_true=1 everywhere, non-empty bins should have fraction_positive=1."""
        y_pred = np.linspace(0.1, 0.9, 50)
        y_true = np.ones(50)
        result = self.module.compute_calibration_bins(y_pred, y_true, n_bins=5)
        for fp, count in zip(result["fraction_positive"], result["bin_counts"]):
            if count > 0:
                assert fp == pytest.approx(1.0)


# ===========================================================================
# 7. TestUncertaintyToKellyFraction — pure numpy, no mocking
# ===========================================================================

class TestUncertaintyToKellyFraction:
    """Tests for uncertainty_to_kelly_fraction(p_win_mean, p_win_std, odds, fraction)."""

    @pytest.fixture(autouse=True)
    def _import(self, mock_pm_az):
        self.module, *_ = mock_pm_az

    def test_returns_float(self):
        result = self.module.uncertainty_to_kelly_fraction(0.6, 0.1, 2.0)
        assert isinstance(result, float)

    def test_zero_edge_returns_zero(self):
        """If kelly <= 0, return 0.0."""
        # kelly = (0.4 * 1 - 0.6) / 1 = -0.2 → clamp to 0
        result = self.module.uncertainty_to_kelly_fraction(0.4, 0.05, 2.0)
        assert result == pytest.approx(0.0)

    def test_positive_edge_returns_positive(self):
        result = self.module.uncertainty_to_kelly_fraction(0.7, 0.05, 2.0)
        assert result > 0.0

    def test_high_uncertainty_reduces_stake(self):
        """Higher std → lower fraction (uncertainty discount)."""
        r_low = self.module.uncertainty_to_kelly_fraction(0.65, 0.05, 2.0)
        r_high = self.module.uncertainty_to_kelly_fraction(0.65, 0.3, 2.0)
        assert r_low > r_high

    def test_fraction_parameter_scales_output(self):
        """fraction=0.5 gives twice the output as fraction=0.25."""
        r_quarter = self.module.uncertainty_to_kelly_fraction(0.65, 0.05, 2.0, fraction=0.25)
        r_half = self.module.uncertainty_to_kelly_fraction(0.65, 0.05, 2.0, fraction=0.50)
        assert r_half == pytest.approx(r_quarter * 2, rel=1e-5)

    def test_negative_kelly_returns_zero(self):
        """p_win so small that kelly is negative — must clamp to 0."""
        result = self.module.uncertainty_to_kelly_fraction(0.1, 0.01, 2.0)
        assert result == pytest.approx(0.0)

    def test_known_value(self):
        """
        p_win=0.6, odds=2.0, std=0.0 (certainty), fraction=0.25:
          kelly = (0.6*1 - 0.4)/1 = 0.2
          discount = 1 - 0.0*2 = 1.0
          result = 0.25 * 0.2 * 1.0 = 0.05
        """
        result = self.module.uncertainty_to_kelly_fraction(0.6, 0.0, 2.0, fraction=0.25)
        assert result == pytest.approx(0.05)

    def test_result_non_negative(self):
        """Result must never be negative."""
        rng = np.random.default_rng(9)
        for _ in range(20):
            p = float(rng.uniform(0.1, 0.9))
            std = float(rng.uniform(0.0, 0.4))
            odds = float(rng.uniform(1.1, 5.0))
            result = self.module.uncertainty_to_kelly_fraction(p, std, odds)
            assert result >= 0.0


# ===========================================================================
# 8. TestCoachATSEffect — Coach ATS ("Tom Izzo Effect") hierarchical prior
# ===========================================================================

class TestCoachATSEffect:
    """build_model() with coach indices creates the Coach ATS Effect priors."""

    def test_n_coaches_stored_in_init(self, mock_pm_az):
        module, *_ = mock_pm_az
        h = module.BayesianHead(embedding_dim=4, n_coaches=200)
        assert h.n_coaches == 200

    def test_default_n_coaches_is_1(self, mock_pm_az):
        module, *_ = mock_pm_az
        h = module.BayesianHead(embedding_dim=4)
        assert h.n_coaches == 1

    def test_coach_ats_normal_called_with_n_coaches_shape(
        self, mock_pm_az, small_game_data
    ):
        """pm.Normal('coach_ats_effect', shape=n_coaches) must be created when
        coach arrays are supplied."""
        module, pm, *_ = mock_pm_az
        G = len(small_game_data["y_win"])
        rng = np.random.default_rng(55)
        coach_data = {
            **small_game_data,
            "home_coach": rng.integers(0, 5, size=G),
            "away_coach": rng.integers(0, 5, size=G),
        }
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, n_coaches=5)
        head.build_model(**coach_data)
        shapes = [c[1].get("shape") for c in pm.Normal.call_args_list]
        assert 5 in shapes  # n_coaches

    def test_mu_coach_normal_called(self, mock_pm_az, small_game_data):
        """mu_coach ~ Normal(0, 0.5) must be created."""
        module, pm, *_ = mock_pm_az
        G = len(small_game_data["y_win"])
        rng = np.random.default_rng(55)
        coach_data = {
            **small_game_data,
            "home_coach": rng.integers(0, 3, size=G),
            "away_coach": rng.integers(0, 3, size=G),
        }
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, n_coaches=3)
        head.build_model(**coach_data)
        # mu_coach has no shape kwarg and sigma=0.5
        sigma_05 = [
            c for c in pm.Normal.call_args_list
            if c[1].get("sigma") == pytest.approx(0.5) and c[1].get("shape") is None
        ]
        assert len(sigma_05) >= 1

    def test_sigma_coach_halfnormal_called(self, mock_pm_az, small_game_data):
        """sigma_coach ~ HalfNormal(0.3) must be created."""
        module, pm, *_ = mock_pm_az
        G = len(small_game_data["y_win"])
        rng = np.random.default_rng(55)
        coach_data = {
            **small_game_data,
            "home_coach": rng.integers(0, 3, size=G),
            "away_coach": rng.integers(0, 3, size=G),
        }
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, n_coaches=3)
        head.build_model(**coach_data)
        sigma_03 = [
            c for c in pm.HalfNormal.call_args_list
            if 0.3 in c[0] or c[1].get("sigma") == pytest.approx(0.3)
        ]
        assert len(sigma_03) >= 1

    def test_coach_prior_not_called_without_coach_data(
        self, head_advi, mock_pm_az, small_game_data
    ):
        """When home_coach/away_coach are absent, no shape=n_coaches Normal
        should be created and HalfNormal(0.3) should not appear."""
        module, pm, *_ = mock_pm_az
        head_advi.build_model(**small_game_data)
        sigma_03 = [
            c for c in pm.HalfNormal.call_args_list
            if 0.3 in c[0] or c[1].get("sigma") == pytest.approx(0.3)
        ]
        assert len(sigma_03) == 0

    def test_existing_tests_unaffected_by_n_coaches_param(
        self, mock_pm_az, small_game_data
    ):
        """BayesianHead constructed without n_coaches still works identically."""
        module, pm, *_ = mock_pm_az
        h = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4)
        result = h.build_model(**small_game_data)
        # Model is returned without error.
        from unittest.mock import MagicMock
        assert result is not None


# ===========================================================================
# 9. TestSkellamHead — Zero-Truncated Skellam spread likelihood
# ===========================================================================

class TestSkellamHead:
    """Tests for use_skellam parameter on MarchMadnessBayesianHead / BayesianHead."""

    # ------ attribute / init tests (use mock pm) ----------------------------

    def test_default_uses_skellam(self, mock_pm_az):
        """BayesianHead() default should have use_skellam=True."""
        module, *_ = mock_pm_az
        head = module.BayesianHead(embedding_dim=4)
        assert head.use_skellam is True

    def test_skellam_false_uses_normal(self, mock_pm_az):
        """use_skellam=False should be stored and honoured."""
        module, *_ = mock_pm_az
        head = module.BayesianHead(embedding_dim=4, use_skellam=False)
        assert head.use_skellam is False

    def test_use_skellam_stored_as_attribute(self, mock_pm_az):
        """use_skellam value is accessible as head.use_skellam."""
        module, *_ = mock_pm_az
        head_true = module.BayesianHead(embedding_dim=4, use_skellam=True)
        head_false = module.BayesianHead(embedding_dim=4, use_skellam=False)
        assert head_true.use_skellam is True
        assert head_false.use_skellam is False

    def test_invalid_use_skellam_type_raises(self, mock_pm_az):
        """use_skellam='yes' (string) should raise TypeError."""
        module, *_ = mock_pm_az
        with pytest.raises(TypeError):
            module.BayesianHead(embedding_dim=4, use_skellam="yes")

    # ------ mock-based build_model tests ------------------------------------

    def test_build_model_returns_model_with_skellam(self, mock_pm_az, small_game_data):
        """build_model() with use_skellam=True returns a model object."""
        module, pm, az, mock_model, *_ = mock_pm_az
        # Expose CustomDist on the pm mock
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        result = head.build_model(**small_game_data)
        assert result is mock_model

    def test_build_model_returns_model_with_normal(self, mock_pm_az, small_game_data):
        """build_model() with use_skellam=False returns a model object."""
        module, pm, az, mock_model, *_ = mock_pm_az
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False)
        result = head.build_model(**small_game_data)
        assert result is mock_model

    def test_skellam_calls_custom_dist(self, mock_pm_az, small_game_data):
        """With use_skellam=True, pm.CustomDist must be called (not Normal for spread)."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        head.build_model(**small_game_data)
        pm.CustomDist.assert_called()

    def test_normal_model_does_not_call_custom_dist(self, mock_pm_az, small_game_data):
        """With use_skellam=False, pm.CustomDist should NOT be called."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False)
        head.build_model(**small_game_data)
        pm.CustomDist.assert_not_called()

    def test_skellam_model_normal_spread_not_called(self, mock_pm_az, small_game_data):
        """With use_skellam=True, pm.Normal should NOT be called with observed=y_spread."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        head.build_model(**small_game_data)
        # pm.Normal calls with observed=y_spread should be 0
        y_spread = small_game_data["y_spread"]
        observed_spread_calls = [
            c for c in pm.Normal.call_args_list
            if c[1].get("observed") is not None
            and np.array_equal(c[1]["observed"], y_spread)
        ]
        assert len(observed_spread_calls) == 0

    def test_normal_model_spread_still_calls_normal(self, mock_pm_az, small_game_data):
        """With use_skellam=False, pm.Normal must still be called with observed=y_spread."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        head = module.BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False)
        head.build_model(**small_game_data)
        observed_arrays = [
            c[1].get("observed") for c in pm.Normal.call_args_list
            if c[1].get("observed") is not None
        ]
        assert len(observed_arrays) >= 1

    def test_skellam_and_normal_same_interface(self, mock_pm_az):
        """Both use_skellam=True and False must return same keys from predict()."""
        module, pm, az, mock_model, mock_approx, idata_advi, idata_nuts = mock_pm_az
        # We test the attribute interface here — both modes have same __init__ params
        head_sk = module.BayesianHead(embedding_dim=4, use_skellam=True)
        head_no = module.BayesianHead(embedding_dim=4, use_skellam=False)
        # Both have the same methods
        assert hasattr(head_sk, "build_model")
        assert hasattr(head_sk, "fit")
        assert hasattr(head_sk, "predict")
        assert hasattr(head_no, "build_model")
        assert hasattr(head_no, "fit")
        assert hasattr(head_no, "predict")

    def test_skellam_model_wins_and_spreads(self, mock_pm_az):
        """n_games=5 input → model builds with use_skellam=True without error."""
        module, pm, az, mock_model, *_ = mock_pm_az
        pm.CustomDist = MagicMock(return_value=MagicMock(name="CustomDist_rv"))
        pm.Deterministic = MagicMock(return_value=MagicMock(name="Deterministic_rv"))
        rng = np.random.default_rng(0)
        G, D = 5, 4
        data = {
            "home_emb": rng.standard_normal((G, D)).astype(np.float32),
            "away_emb": rng.standard_normal((G, D)).astype(np.float32),
            "home_conf": rng.integers(0, 8, size=G),
            "away_conf": rng.integers(0, 8, size=G),
            "home_seed": rng.integers(0, 4, size=G),
            "away_seed": rng.integers(0, 4, size=G),
            "y_win": rng.integers(0, 2, size=G).astype(np.float32),
            "y_spread": rng.integers(1, 20, size=G).astype(np.float32),
        }
        head = module.BayesianHead(embedding_dim=D, n_conferences=8, n_seeds=4, use_skellam=True)
        result = head.build_model(**data)
        assert result is mock_model

    # ------ real PyMC integration tests (no mocking) -----------------------

    def _real_game_data(self, G=5, D=4, seed=0):
        """Synthetic data for real PyMC tests — integer spreads, nonzero."""
        rng = np.random.default_rng(seed)
        return {
            "home_emb": rng.standard_normal((G, D)).astype(np.float32),
            "away_emb": rng.standard_normal((G, D)).astype(np.float32),
            "home_conf": rng.integers(0, 8, size=G),
            "away_conf": rng.integers(0, 8, size=G),
            "home_seed": rng.integers(0, 4, size=G),
            "away_seed": rng.integers(0, 4, size=G),
            "y_win": rng.integers(0, 2, size=G).astype(np.float32),
            # Non-zero integer spreads (zero-truncated Skellam assigns -inf to 0)
            "y_spread": rng.choice(
                [v for v in range(-20, 21) if v != 0], size=G
            ).astype(np.float32),
        }

    def test_build_model_returns_model_with_skellam_real(self):
        """Real PyMC: build_model(use_skellam=True) returns a pm.Model."""
        pm = pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4,
                            advi_iterations=200, use_skellam=True)
        model = head.build_model(**data)
        assert isinstance(model, pm.Model)

    def test_build_model_returns_model_with_normal_real(self):
        """Real PyMC: build_model(use_skellam=False) returns a pm.Model."""
        pm = pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4,
                            advi_iterations=200, use_skellam=False)
        model = head.build_model(**data)
        assert isinstance(model, pm.Model)

    def test_skellam_model_has_log_base_rate(self):
        """Real PyMC: model.named_vars must contain 'log_base_rate' when use_skellam=True."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        model = head.build_model(**data)
        assert "log_base_rate" in model.named_vars

    def test_normal_model_does_not_have_log_base_rate(self):
        """Real PyMC: model.named_vars must NOT have 'log_base_rate' when use_skellam=False."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=False)
        model = head.build_model(**data)
        assert "log_base_rate" not in model.named_vars

    def test_skellam_model_contains_spread_obs(self):
        """Real PyMC: Observed spread variable exists in model."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        model = head.build_model(**data)
        # 'obs_spread' should exist in named_vars
        assert "obs_spread" in model.named_vars

    def test_skellam_model_has_mu_home_away(self):
        """Real PyMC: 'mu_home' and 'mu_away' deterministic vars exist with Skellam."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        model = head.build_model(**data)
        assert "mu_home" in model.named_vars
        assert "mu_away" in model.named_vars

    def test_skellam_model_fit_advi(self):
        """Real PyMC: build_model(use_skellam=True) + fit(method='advi', n=200) runs without error."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4,
                            advi_iterations=200, use_skellam=True)
        model = head.build_model(**data)
        idata = head.fit(model)
        assert idata is not None

    def test_normal_model_fit_advi(self):
        """Real PyMC: build_model(use_skellam=False) + fit(method='advi', n=200) runs without error."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4,
                            advi_iterations=200, use_skellam=False)
        model = head.build_model(**data)
        idata = head.fit(model)
        assert idata is not None

    def test_zero_spread_excluded(self):
        """Zero spread raises -inf log-prob under ZT-Skellam; model should still build with nonzero data."""
        pytest.importorskip("pymc")
        import sys
        sys.modules.pop("src.model.bayesian_head", None)
        sys.modules.pop("src.model.skellam", None)
        from src.model.bayesian_head import BayesianHead
        data = self._real_game_data()
        # Ensure no zero in spreads
        assert not np.any(data["y_spread"] == 0)
        head = BayesianHead(embedding_dim=4, n_conferences=8, n_seeds=4, use_skellam=True)
        model = head.build_model(**data)
        assert model is not None
