"""
src/model/fusion/fusion_engine.py

Combinatorial Fusion Analysis (CFA) Fusion Engine.

Reference
---------
Hsu, D. F., Taksa, I. (2005). "Comparing Rank and Score Combination Methods for
Data Fusion in Information Retrieval." Information Retrieval, 8(3), 449-480.

Hsu, D. F., Chung, Y. S., Kristal, B. S. (2006). "Combinatorial Fusion Analysis:
Methods and Practices of Combining Multiple Scoring Systems." Advanced Data Mining
and Applications, LNCS 4093, Springer.

Design Rationale
----------------
The J-curve hypothesis from CFA explains why a *diverse* ensemble of models
outperforms even the single best individual model: plotting Spearman rank
correlation (diversity proxy) against combined accuracy reveals a characteristic
J-curve — performance improves as diversity increases, until models become
anti-correlated and performance degrades.  The greedy subset-selection algorithm
in `select_diverse_subset` exploits this by starting from the best individual
model and adding new models only when their rank correlation with all already-
selected models stays below `diversity_threshold` (default 0.95).  This prunes
near-duplicate models (corr > 0.95) that would add no information while
increasing variance.

Note: For rigorous model selection, use this module (Hsu et al. 2005) in
preference to the heuristic weighting in src.model.ensemble.BoardOfDirectors.
"""

from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------


def _predictions_to_ranks(predictions: np.ndarray) -> np.ndarray:
    """
    Convert per-model probability predictions to per-game ranks.

    For each model (row), rank its N_games predictions so that rank 1 is
    assigned to the game with the *highest* predicted win probability.

    Parameters
    ----------
    predictions : np.ndarray, shape (N_models, N_games)

    Returns
    -------
    np.ndarray, shape (N_models, N_games), dtype float
        Each row contains ranks 1..N_games (no ties handling needed for
        continuous probabilities; ties broken by natural order).
    """
    n_models, n_games = predictions.shape
    ranks = np.empty_like(predictions, dtype=float)
    for i in range(n_models):
        # argsort descending → the index with highest prob gets position 0 → rank 1
        order = np.argsort(-predictions[i])
        rank_row = np.empty(n_games, dtype=float)
        rank_row[order] = np.arange(1, n_games + 1, dtype=float)
        ranks[i] = rank_row
    return ranks


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def rank_models(predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Rank models by Brier Score (lower score = better = earlier in result).

    Parameters
    ----------
    predictions : np.ndarray, shape (N_models, N_games)
        Each row is one model's predicted win probabilities.
    y_true : np.ndarray, shape (N_games,)
        Binary game outcomes (0 or 1).

    Returns
    -------
    np.ndarray, shape (N_models,), dtype int
        Model indices sorted from best (lowest Brier Score) to worst.
    """
    n_models = predictions.shape[0]
    brier_scores = np.array(
        [np.mean((predictions[i] - y_true) ** 2) for i in range(n_models)]
    )
    return np.argsort(brier_scores)


def compute_rank_correlation(ranks_a: np.ndarray, ranks_b: np.ndarray) -> float:
    """
    Compute Spearman rank correlation between two models' game-level rank arrays.

    Parameters
    ----------
    ranks_a, ranks_b : np.ndarray, shape (N_games,)
        Per-game prediction ranks from two models (rank 1 = highest probability).

    Returns
    -------
    float
        Spearman correlation in [-1, 1].
    """
    corr, _ = spearmanr(ranks_a, ranks_b)
    return float(corr)


def select_diverse_subset(
    predictions: np.ndarray,
    y_true: np.ndarray,
    diversity_threshold: float = 0.95,
) -> List[int]:
    """
    Greedy J-curve subset selection following CFA (Hsu et al. 2005).

    Algorithm
    ---------
    1. Rank all models by Brier score (best first).
    2. Initialise selected set with the best model.
    3. Iterate remaining models in performance order.  Add a model only if its
       Spearman rank correlation with *every* already-selected model is <=
       diversity_threshold.
    4. Return selected model indices (original 0-based).

    Parameters
    ----------
    predictions : np.ndarray, shape (N_models, N_games)
    y_true : np.ndarray, shape (N_games,)
    diversity_threshold : float, in (0, 1]
        Maximum allowed rank correlation between any two selected models.

    Returns
    -------
    list[int]
        Original 0-based indices of selected models.

    Raises
    ------
    ValueError
        If `diversity_threshold` is not in (0, 1].
    """
    if not (0.0 < diversity_threshold <= 1.0):
        raise ValueError(
            f"diversity_threshold must be in (0, 1]; got {diversity_threshold}"
        )

    ranked_order = rank_models(predictions, y_true)  # best → worst
    per_model_ranks = _predictions_to_ranks(predictions)  # shape (N, G)

    selected_original_indices: List[int] = []

    for model_idx in ranked_order:
        if not selected_original_indices:
            # Always include the best model
            selected_original_indices.append(int(model_idx))
            continue

        # Check correlation against every already-selected model
        candidate_ranks = per_model_ranks[model_idx]
        diverse_enough = True
        for sel_idx in selected_original_indices:
            corr = compute_rank_correlation(candidate_ranks, per_model_ranks[sel_idx])
            if corr > diversity_threshold:
                diverse_enough = False
                break

        if diverse_enough:
            selected_original_indices.append(int(model_idx))

    return selected_original_indices


# ---------------------------------------------------------------------------
# CFAFusionEngine
# ---------------------------------------------------------------------------


class CFAFusionEngine:
    """
    Combinatorial Fusion Analysis engine for multi-model ensemble fusion.

    Implements J-curve diversity-based model selection (Hsu et al. 2005/2006)
    and rank-sum score aggregation.

    Parameters
    ----------
    diversity_threshold : float, default 0.95
        Spearman correlation ceiling for model pair acceptance.  Pairs with
        correlation > threshold are considered near-duplicate; the lower-
        performing model in the pair is discarded.
    aggregation : str, default "rank_sum"
        Aggregation strategy.  Currently only "rank_sum" is supported.

    Attributes
    ----------
    selected_indices_ : list[int]
        Indices of models selected after `fit()`.  Set only after `fit()`.
    """

    _VALID_AGGREGATIONS = {"rank_sum"}

    def __init__(
        self,
        diversity_threshold: float = 0.95,
        aggregation: str = "rank_sum",
    ) -> None:
        if not (0.0 < diversity_threshold <= 1.0):
            raise ValueError(
                f"diversity_threshold must be in (0, 1]; got {diversity_threshold}"
            )
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {self._VALID_AGGREGATIONS}; got {aggregation!r}"
            )
        self.diversity_threshold = diversity_threshold
        self.aggregation = aggregation
        self._fitted = False
        self._n_models_total: int = 0
        self._per_model_ranks: np.ndarray | None = None  # stored for diversity_report

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, predictions: np.ndarray, y_true: np.ndarray) -> "CFAFusionEngine":
        """
        Select diverse model subset from training predictions.

        Parameters
        ----------
        predictions : np.ndarray, shape (N_models, N_games)
        y_true : np.ndarray, shape (N_games,)

        Returns
        -------
        self
        """
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions must be 2D (N_models, N_games); got shape {predictions.shape}"
            )
        if predictions.shape[1] != len(y_true):
            raise ValueError(
                f"predictions.shape[1]={predictions.shape[1]} != len(y_true)={len(y_true)}"
            )

        self._n_models_total = predictions.shape[0]
        self._per_model_ranks = _predictions_to_ranks(predictions)

        self.selected_indices_: List[int] = select_diverse_subset(
            predictions, y_true, diversity_threshold=self.diversity_threshold
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        """
        Fuse predictions from selected models into a single probability array.

        For rank_sum aggregation:
            1. Compute per-game ranks for each selected model (rank 1 = highest prob).
            2. Average ranks across selected models.
            3. Convert mean rank to probability: fused = 1 - (mean_rank / N_games).

        Parameters
        ----------
        predictions : np.ndarray, shape (N_models, N_games)
            Same format as `fit()`.  Only rows corresponding to `selected_indices_`
            are used.

        Returns
        -------
        np.ndarray, shape (N_games,)
            Fused probabilities in [0, 1].
        """
        if not self._fitted:
            raise ValueError("CFAFusionEngine must be fit() before predict().")
        if predictions.ndim != 2:
            raise ValueError(
                f"predictions must be 2D (N_models, N_games); got shape {predictions.shape}"
            )

        n_games = predictions.shape[1]
        selected_preds = predictions[self.selected_indices_]  # (K, N_games)
        ranks = _predictions_to_ranks(selected_preds)         # (K, N_games)
        mean_rank = ranks.mean(axis=0)                        # (N_games,)
        fused = 1.0 - (mean_rank / n_games)
        # Clip to [0, 1] for floating-point safety
        return np.clip(fused, 0.0, 1.0)

    # ------------------------------------------------------------------
    # diversity_report
    # ------------------------------------------------------------------

    def diversity_report(self) -> dict:
        """
        Return a diagnostic report of the fusion state.

        Returns
        -------
        dict with keys:
            n_models_total : int
            n_models_selected : int
            pairwise_correlations : dict[str, float]
                Keys are "i-j" (original model indices), values are Spearman
                correlations computed from training-time rank arrays.
            selected_indices : list[int]

        Raises
        ------
        ValueError
            If called before `fit()`.
        """
        if not self._fitted:
            raise ValueError("CFAFusionEngine must be fit() before diversity_report().")

        n_total = self._n_models_total
        pairwise: dict[str, float] = {}
        ranks = self._per_model_ranks  # (N_models_total, N_games)

        for i, j in combinations(range(n_total), 2):
            corr = compute_rank_correlation(ranks[i], ranks[j])
            pairwise[f"{i}-{j}"] = corr

        return {
            "n_models_total": n_total,
            "n_models_selected": len(self.selected_indices_),
            "pairwise_correlations": pairwise,
            "selected_indices": list(self.selected_indices_),
        }
