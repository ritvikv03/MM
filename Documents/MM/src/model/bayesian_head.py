"""
src/model/bayesian_head.py

Bayesian head for the NCAA March Madness ST-GNN.

Wraps ST-GNN team embeddings (produced by TemporalEncoder) in a PyMC
hierarchical model that yields full posterior distributions over two
simultaneous game outcomes:

  1. Win probability  (Bernoulli likelihood)
  2. Point spread     (Normal likelihood)

NOTE: Game totals (obs_total) are permanently excluded from this model.
Two-target architecture only: win probability + spread.

Clutch/Luck Regression Prior
-----------------------------
Per the Law of Large Numbers, close-game win % regresses strongly toward 0.5
over a ~35-game D-I season (Gilovich, Vallone & Tversky, 1985; see also
Berri & Schmidt "Stumbling on Wins").  A HalfNormal(sigma=0.15) prior on the
luck scale enforces this regression:  any luck effect >2σ (>0.30) is
strongly penalised, keeping luck adjustments bounded near zero.

Hierarchical structure
----------------------
- Conference random effects (per-conference intercept).
- Seed strength prior (informative; higher seed index = weaker team).
- Linear projection weights for home and away embeddings.

Sampler choice
--------------
Default: ADVI (Variational Inference) — fast, non-blocking, suitable for
local TDD and CI.  Pass ``sampler="nuts"`` to use full NUTS for production
runs (configure chain/draw counts via constructor kwargs).

PIT Integrity
-------------
This module never fetches data.  All embeddings and indices are supplied by
the caller from pre-computed arrays produced by the upstream ST-GNN pipeline.

Usage example
-------------
>>> from src.model.bayesian_head import BayesianHead
>>> head = BayesianHead(embedding_dim=128)
>>> model = head.build_model(home_emb, away_emb, ...)
>>> idata = head.fit(model)
>>> preds = head.predict(idata, home_emb_test, away_emb_test, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pymc as pm
    import arviz as az


# ---------------------------------------------------------------------------
# BayesianHead
# ---------------------------------------------------------------------------

class BayesianHead:
    """Hierarchical Bayesian model that consumes ST-GNN team embeddings and
    produces posterior distributions over game outcomes.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of the incoming team embeddings
        (= ``TemporalEncoder.hidden_dim``).
    n_conferences:
        Number of distinct conference indices.  Default 32.
    n_seeds:
        Number of distinct tournament seed values (1–16).  A zero index
        is reserved for teams not in the tournament.  Default 16.
    sampler:
        ``"advi"`` (default) — fast variational inference;
        ``"nuts"`` — full MCMC with No-U-Turn Sampler.
    advi_iterations:
        Number of ADVI optimisation steps.  Default 10 000.
    nuts_draws:
        Number of post-warmup NUTS draws per chain.  Default 500.
    nuts_chains:
        Number of independent NUTS chains.  Default 2.
    nuts_tune:
        Number of NUTS warmup (tuning) steps.  Default 200.
    random_seed:
        Global random seed for reproducibility.  Default 42.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_conferences: int = 32,
        n_seeds: int = 16,
        sampler: str = "advi",
        advi_iterations: int = 10_000,
        nuts_draws: int = 500,
        nuts_chains: int = 2,
        nuts_tune: int = 200,
        random_seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.n_conferences = n_conferences
        self.n_seeds = n_seeds
        self.sampler = sampler
        self.advi_iterations = advi_iterations
        self.nuts_draws = nuts_draws
        self.nuts_chains = nuts_chains
        self.nuts_tune = nuts_tune
        self.random_seed = random_seed

    # ------------------------------------------------------------------
    # build_model
    # ------------------------------------------------------------------

    def build_model(
        self,
        home_emb: np.ndarray,
        away_emb: np.ndarray,
        home_conf: np.ndarray,
        away_conf: np.ndarray,
        home_seed: np.ndarray,
        away_seed: np.ndarray,
        y_win: np.ndarray,
        y_spread: np.ndarray,
        home_luck: np.ndarray | None = None,
        away_luck: np.ndarray | None = None,
    ) -> "pm.Model":
        """Construct and return the PyMC hierarchical model.

        Two-target model: win probability (Bernoulli) + spread (Normal).
        Game totals are permanently excluded per architecture mandate.

        Parameters
        ----------
        home_emb : (G, embedding_dim)
            Home team embeddings from the ST-GNN for G games.
        away_emb : (G, embedding_dim)
            Away team embeddings.
        home_conf : (G,) int
            Conference index for the home team.
        away_conf : (G,) int
            Conference index for the away team.
        home_seed : (G,) int
            Tournament seed for the home team (1-16; 0 = not in tournament).
        away_seed : (G,) int
            Tournament seed for the away team.
        y_win : (G,) binary float
            Observed outcome: 1 if home team won.
        y_spread : (G,) float
            Observed point spread (home score − away score).
        home_luck : (G,) float | None
            Barttorvik close-game win fraction for the home team.  When
            provided, a regression-to-mean prior (LLN) is applied.
        away_luck : (G,) float | None
            Barttorvik close-game win fraction for the away team.

        Returns
        -------
        pm.Model
            The constructed PyMC model (not yet sampled).
        """
        import pymc as pm

        with pm.Model() as model:
            # ---- 1. Conference random effects (hierarchical) ---------------
            sigma_conf = pm.HalfNormal("sigma_conf", sigma=1.0)
            conf_effect = pm.Normal(
                "conf_effect", mu=0.0, sigma=sigma_conf, shape=self.n_conferences
            )

            # ---- 2. Seed strength ------------------------------------------
            # shape = n_seeds + 1 to handle index 0 (not in tournament)
            seed_effect = pm.Normal(
                "seed_effect", mu=0.0, sigma=0.5, shape=self.n_seeds + 1
            )

            # ---- 3. Embedding projection weights ---------------------------
            W_home = pm.Normal("W_home", mu=0.0, sigma=1.0, shape=self.embedding_dim)
            W_away = pm.Normal("W_away", mu=0.0, sigma=1.0, shape=self.embedding_dim)

            # ---- 4. Clutch/Luck regression-to-mean prior (LLN) ------------
            # Per LLN, close-game win % regresses toward 0.5 over a ~35-game
            # season.  sigma <= 0.15 bounds luck effects near zero.
            # (Gilovich et al. 1985; Berri & Schmidt "Stumbling on Wins")
            luck_scale = pm.HalfNormal("luck_scale", sigma=0.15)
            if home_luck is not None and away_luck is not None:
                # Soft prior: observed luck is pulled toward 0.5.
                pm.Normal("obs_home_luck", mu=0.5, sigma=luck_scale,
                          observed=home_luck)
                pm.Normal("obs_away_luck", mu=0.5, sigma=luck_scale,
                          observed=away_luck)

            # ---- 5. Linear predictors (deterministic) ----------------------
            # Use pm.math.dot so this stays in the PyTensor computation graph
            # and remains mockable in tests (avoids raw numpy @ on RV objects).
            home_strength = (
                pm.math.dot(home_emb, W_home)
                + conf_effect[home_conf]
                + seed_effect[home_seed]
            )
            away_strength = (
                pm.math.dot(away_emb, W_away)
                + conf_effect[away_conf]
                + seed_effect[away_seed]
            )
            delta = home_strength - away_strength

            # ---- 6. Win probability — Bernoulli likelihood -----------------
            p_win = pm.math.sigmoid(delta)
            obs_win = pm.Bernoulli("obs_win", p=p_win, observed=y_win)

            # ---- 7. Spread — Normal likelihood -----------------------------
            sigma_spread = pm.HalfNormal("sigma_spread", sigma=10.0)
            obs_spread = pm.Normal(
                "obs_spread", mu=delta, sigma=sigma_spread, observed=y_spread
            )

        return model

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, model: "pm.Model") -> "az.InferenceData":
        """Fit the model and return an ArviZ InferenceData object.

        Parameters
        ----------
        model:
            The PyMC model returned by :meth:`build_model`.

        Returns
        -------
        az.InferenceData
            Posterior samples (NUTS) or variational approximation samples
            (ADVI) wrapped in ArviZ InferenceData.
        """
        import pymc as pm

        if self.sampler == "advi":
            approx = pm.fit(
                self.advi_iterations,
                model=model,
                random_seed=self.random_seed,
            )
            return approx.sample(5000)

        # NUTS path
        with model:
            idata = pm.sample(
                draws=self.nuts_draws,
                chains=self.nuts_chains,
                tune=self.nuts_tune,
                random_seed=self.random_seed,
                progressbar=False,
            )
        return idata

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        idata: "az.InferenceData",
        home_emb: np.ndarray,
        away_emb: np.ndarray,
        home_conf: np.ndarray,
        away_conf: np.ndarray,
        home_seed: np.ndarray,
        away_seed: np.ndarray,
    ) -> dict:
        """Generate posterior predictive summaries for a set of games.

        Parameters
        ----------
        idata:
            ArviZ InferenceData from :meth:`fit`.
        home_emb : (G, embedding_dim)
        away_emb : (G, embedding_dim)
        home_conf : (G,) int
        away_conf : (G,) int
        home_seed : (G,) int
        away_seed : (G,) int

        Returns
        -------
        dict with keys:
            ``p_win_mean``        : (G,) posterior mean win probability
            ``p_win_std``         : (G,) posterior std of win probability
            ``spread_mean``       : (G,) posterior mean point spread
            ``spread_std``        : (G,) posterior std of point spread
            ``credible_interval_95`` : dict with ``"lower"`` and ``"upper"`` (G,)
        """
        posterior = idata.posterior

        # Stack chains and draws into a flat sample dimension.
        # ArviZ stores posterior variables as (chain, draw, *param_dims).
        W_home_samples = posterior["W_home"].values     # (chain, draw, D)
        W_away_samples = posterior["W_away"].values     # (chain, draw, D)
        conf_samples   = posterior["conf_effect"].values  # (chain, draw, n_conf)
        seed_samples   = posterior["seed_effect"].values  # (chain, draw, n_seeds+1)

        # Flatten chains: (chain, draw, ...) → (S, ...)
        def _flatten(arr: np.ndarray) -> np.ndarray:
            """Merge the first two (chain, draw) axes into one sample axis."""
            shape = arr.shape
            return arr.reshape(-1, *shape[2:])

        W_home_s = _flatten(W_home_samples)  # (S, D)
        W_away_s = _flatten(W_away_samples)  # (S, D)
        conf_s   = _flatten(conf_samples)    # (S, n_conf)
        seed_s   = _flatten(seed_samples)    # (S, n_seeds+1)

        S = W_home_s.shape[0]
        G = home_emb.shape[0]

        # Compute delta for each posterior sample: (S, G)
        # home_strength[s, g] = home_emb[g] @ W_home[s] + conf[s, home_conf[g]] + seed[s, home_seed[g]]
        home_str = (
            home_emb @ W_home_s.T                          # (G, S)
            + conf_s[:, home_conf].T                       # (G, S)
            + seed_s[:, home_seed].T                       # (G, S)
        )  # (G, S)

        away_str = (
            away_emb @ W_away_s.T                          # (G, S)
            + conf_s[:, away_conf].T                       # (G, S)
            + seed_s[:, away_seed].T                       # (G, S)
        )  # (G, S)

        delta_samples = home_str - away_str                # (G, S)

        # Sigmoid → win probability posterior samples
        p_win_samples = 1.0 / (1.0 + np.exp(-delta_samples))  # (G, S)

        p_win_mean = p_win_samples.mean(axis=1)  # (G,)
        p_win_std  = p_win_samples.std(axis=1)   # (G,)

        spread_mean = delta_samples.mean(axis=1)  # (G,)
        spread_std  = delta_samples.std(axis=1)   # (G,)

        # 95% HDI via quantiles on win probability
        lower = np.quantile(p_win_samples, 0.025, axis=1)  # (G,)
        upper = np.quantile(p_win_samples, 0.975, axis=1)  # (G,)

        return {
            "p_win_mean": p_win_mean,
            "p_win_std":  p_win_std,
            "spread_mean": spread_mean,
            "spread_std":  spread_std,
            "credible_interval_95": {"lower": lower, "upper": upper},
        }


# ---------------------------------------------------------------------------
# Standalone metric / utility functions
# ---------------------------------------------------------------------------

def compute_brier_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute the Brier Score: mean squared error between predicted
    probabilities and binary outcomes.

    Parameters
    ----------
    y_pred:
        Predicted probabilities in [0, 1], shape (N,).
    y_true:
        Binary observed outcomes in {0, 1}, shape (N,).

    Returns
    -------
    float
        Brier Score.  A perfect model scores 0.0; a worst-case model 1.0.

    Raises
    ------
    ValueError
        If ``y_pred`` and ``y_true`` have different shapes.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape} but "
            f"y_true has shape {y_true.shape}."
        )

    return float(np.mean((y_pred - y_true) ** 2))


def compute_calibration_bins(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Build calibration curve data by binning predicted probabilities.

    Each bin collects samples whose predicted probability falls within the
    bin's range and reports:
      - The bin centre (mid-point of the bin edges).
      - The fraction of positive outcomes within that bin.
      - The count of samples in that bin.

    Empty bins are handled gracefully: ``fraction_positive`` is set to
    ``np.nan`` for empty bins so they can be identified by callers.

    Parameters
    ----------
    y_pred:
        Predicted probabilities in [0, 1], shape (N,).
    y_true:
        Binary observed outcomes, shape (N,).
    n_bins:
        Number of equal-width bins across [0, 1].  Default 10.

    Returns
    -------
    dict with keys:
        ``bin_centers``      : np.ndarray (n_bins,)
        ``fraction_positive``: np.ndarray (n_bins,)  — nan for empty bins
        ``bin_counts``       : np.ndarray (n_bins,) int
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fraction_positive = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include the right edge only for the last bin to capture p=1.0
        if i < n_bins - 1:
            mask = (y_pred >= lo) & (y_pred < hi)
        else:
            mask = (y_pred >= lo) & (y_pred <= hi)

        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            fraction_positive[i] = float(y_true[mask].mean())

    return {
        "bin_centers": bin_centers,
        "fraction_positive": fraction_positive,
        "bin_counts": bin_counts,
    }


def uncertainty_to_kelly_fraction(
    p_win_mean: float,
    p_win_std: float,
    odds: float,
    fraction: float = 0.25,
) -> float:
    """Compute a fractional Kelly stake, discounted for prediction uncertainty.

    The raw Kelly criterion for a binary bet is::

        kelly = (p_win_mean * (odds - 1) - (1 - p_win_mean)) / (odds - 1)

    A positive Kelly value is then scaled by ``fraction`` (fractional Kelly)
    and by an uncertainty discount ``(1 - p_win_std * 2)`` that shrinks the
    stake when the model is less certain.  The final stake is clamped to
    ``[0, 1]``.

    Parameters
    ----------
    p_win_mean:
        Posterior mean win probability.
    p_win_std:
        Posterior standard deviation of win probability.
    odds:
        Decimal odds offered by the bookmaker (e.g. 2.0 for evens).
    fraction:
        Fractional Kelly multiplier.  Default 0.25 (quarter Kelly).

    Returns
    -------
    float
        Recommended stake fraction in ``[0, 1]``.  Returns 0.0 if the raw
        Kelly edge is non-positive.
    """
    b = odds - 1.0  # net profit per unit staked on a win
    kelly = (p_win_mean * b - (1.0 - p_win_mean)) / b

    if kelly <= 0.0:
        return 0.0

    # Clamp raw kelly to [0, 1]
    kelly = min(max(kelly, 0.0), 1.0)

    # Uncertainty discount: less confidence → smaller stake
    discount = 1.0 - p_win_std * 2.0

    stake = fraction * kelly * discount
    return float(max(stake, 0.0))
