"""
src/model/stgnn.py

Top-level Spatio-Temporal GNN model for NCAA March Madness win-probability
prediction.

Wires :class:`~src.model.gat_encoder.GATEncoder` and
:class:`~src.model.temporal_encoder.TemporalEncoder` together and adds a
lightweight prediction head.

Also provides training utilities:

* :func:`brier_score`
* :func:`compute_log_loss`
* :func:`make_stgnn_model`
* :func:`train_one_epoch`

Lazy imports
------------
``torch`` and ``torch.nn`` are imported *inside* every function and method,
never at module-import time.  This satisfies the constraint that the module
loads cleanly even if torch is not installed.

Usage example
-------------
>>> from src.model.stgnn import STGNNModel, make_stgnn_model, train_one_epoch
>>> model = make_stgnn_model(gat_config, temporal_config)
>>> logits = model(snapshots, home_idx, away_idx)  # (G, 1)
"""

from __future__ import annotations

# Import encoder factories at module level so they can be patched in tests.
# These are our own modules (not torch deps) and are safe to import early.
from src.model.gat_encoder import make_gat_encoder
from src.model.temporal_encoder import make_temporal_encoder


# Re-export stack_snapshots for callers.
def stack_snapshots(embeddings: list):
    """Delegate to :func:`~src.model.temporal_encoder.stack_snapshots`."""
    from src.model.temporal_encoder import stack_snapshots as _stack
    return _stack(embeddings)


# ---------------------------------------------------------------------------
# STGNNModel — built lazily inside a class factory so nn.Module is only
# resolved when torch is available (same pattern as temporal_encoder.py).
# ---------------------------------------------------------------------------

def _build_stgnn_class():
    """Return STGNNModel with nn.Module as a live base class (lazy torch)."""
    import torch
    import torch.nn as nn

    class STGNNModel(nn.Module):
        """Full Spatio-Temporal GNN producing win-probability logits.

        Architecture
        ------------
        1. Per-snapshot GAT encoding → ``(N, gat_hidden_dim)``
        2. Stack T snapshots → ``(T, N, gat_hidden_dim)``
        3. Temporal encoding → ``(N, temporal_hidden_dim)``
        4. Concatenate home + away team embeddings → ``(G, 2*temporal_hidden_dim)``
        5. Prediction head MLP → ``(G, output_dim)`` raw logits
        """

        def __init__(
            self,
            gat_config: dict,
            temporal_config: dict,
            output_dim: int = 1,
        ) -> None:
            super().__init__()

            self.output_dim = output_dim
            self.gat_encoder = make_gat_encoder(gat_config)
            self.temporal_encoder = make_temporal_encoder(temporal_config)

            temporal_hidden_dim = temporal_config.get("hidden_dim", 128)
            head_hidden_dim = max(temporal_hidden_dim, 32)

            self.prediction_head = nn.Sequential(
                nn.Linear(2 * temporal_hidden_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=temporal_config.get("dropout", 0.1)),
                nn.Linear(head_hidden_dim, output_dim),
            )

        def forward(self, snapshots: list, game_home_idx, game_away_idx):
            """Run the full ST-GNN pipeline; returns logits ``(G, output_dim)``."""
            snapshot_embeddings = []
            for snap in snapshots:
                emb = self.gat_encoder(
                    snap.node_features,
                    snap.edge_index,
                    snap.edge_attr,
                )
                snapshot_embeddings.append(emb)

            stacked = stack_snapshots(snapshot_embeddings)
            team_embeddings = self.temporal_encoder(stacked)

            home_emb = team_embeddings[game_home_idx]
            away_emb = team_embeddings[game_away_idx]
            game_features = torch.cat([home_emb, away_emb], dim=1)

            return self.prediction_head(game_features)

    return STGNNModel


# Lazy singleton — resolved on first attribute access via __getattr__.
_STGNNModel_cls = None


def __getattr__(name: str):
    global _STGNNModel_cls
    if name == "STGNNModel":
        if _STGNNModel_cls is None:
            _STGNNModel_cls = _build_stgnn_class()
        return _STGNNModel_cls
    raise AttributeError(f"module 'src.model.stgnn' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Loss / metric functions
# ---------------------------------------------------------------------------

def brier_score(
    y_pred_probs: "torch.Tensor",
    y_true: "torch.Tensor",
) -> "torch.Tensor":
    """Compute the Brier Score (mean squared error on probabilities).

    Parameters
    ----------
    y_pred_probs:
        Predicted probabilities (sigmoid already applied), shape ``(G,)`` or
        ``(G, 1)``.
    y_true:
        Binary labels ``{0, 1}``, same shape as *y_pred_probs*.

    Returns
    -------
    torch.Tensor
        Scalar Brier Score: ``mean((y_pred - y_true)^2)``.
    """
    import torch  # lazy

    diff = y_pred_probs - y_true
    return torch.mean(diff * diff)


def compute_log_loss(
    y_pred_probs: "torch.Tensor",
    y_true: "torch.Tensor",
    eps: float = 1e-7,
) -> "torch.Tensor":
    """Compute binary cross-entropy (log loss) with probability clamping.

    Parameters
    ----------
    y_pred_probs:
        Predicted probabilities (sigmoid already applied), shape ``(G,)`` or
        ``(G, 1)``.
    y_true:
        Binary labels ``{0, 1}``, same shape as *y_pred_probs*.
    eps:
        Clamp probabilities to ``[eps, 1 - eps]`` before computing the log to
        prevent ``log(0)`` → ``-inf`` / NaN.

    Returns
    -------
    torch.Tensor
        Scalar mean log loss.
    """
    import torch  # lazy

    p = torch.clamp(y_pred_probs, min=eps, max=1.0 - eps)
    loss = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
    return torch.mean(loss)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_stgnn_model(
    gat_config: dict,
    temporal_config: dict,
    output_dim: int = 1,
) -> "STGNNModel":
    """Construct an :class:`STGNNModel` from plain config dictionaries.

    Parameters
    ----------
    gat_config:
        Dict forwarded to :func:`~src.model.gat_encoder.make_gat_encoder`.
    temporal_config:
        Dict forwarded to
        :func:`~src.model.temporal_encoder.make_temporal_encoder`.
    output_dim:
        Number of output logits per game.  Defaults to ``1``.

    Returns
    -------
    STGNNModel
    """
    # Resolve lazily — STGNNModel lives behind module __getattr__
    global _STGNNModel_cls
    if _STGNNModel_cls is None:
        _STGNNModel_cls = _build_stgnn_class()
    return _STGNNModel_cls(
        gat_config=gat_config,
        temporal_config=temporal_config,
        output_dim=output_dim,
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: "STGNNModel",
    optimizer,
    snapshots_list: list,
    game_home_idx: "torch.Tensor",
    game_away_idx: "torch.Tensor",
    labels: "torch.Tensor",
    device: str = "cpu",
) -> dict:
    """Run a single training epoch.

    Performs one forward pass, computes Brier Score loss, backpropagates, and
    steps the optimizer.  Returns a dictionary of training metrics.

    Parameters
    ----------
    model:
        :class:`STGNNModel` instance to train.
    optimizer:
        PyTorch optimizer (e.g. ``torch.optim.Adam``).
    snapshots_list:
        List of :class:`~src.graph.graph_constructor.TemporalSnapshot` objects
        for the current epoch.
    game_home_idx:
        Long tensor of shape ``(G,)`` — home-team node indices.
    game_away_idx:
        Long tensor of shape ``(G,)`` — away-team node indices.
    labels:
        Float tensor of shape ``(G,)`` — binary win labels ``{0.0, 1.0}``.
    device:
        Device string for future use (defaults to ``"cpu"``).

    Returns
    -------
    dict
        ``{"loss": float, "brier_score": float, "log_loss": float}``
    """
    import torch  # lazy

    # Set model to training mode
    model.train()

    # Zero gradients
    optimizer.zero_grad()

    # Forward pass → raw logits (G, output_dim)
    logits = model(snapshots_list, game_home_idx, game_away_idx)

    # Flatten to (G,) for metric functions
    logits_flat = logits.view(-1)
    probs = torch.sigmoid(logits_flat)

    # Labels must also be flat
    labels_flat = labels.view(-1)

    # Compute Brier Score as training loss
    loss = brier_score(probs, labels_flat)

    # Backward + optimizer step
    loss.backward()
    optimizer.step()

    # Compute additional metrics (no grad needed)
    with torch.no_grad():
        bs_val = brier_score(probs, labels_flat).item()
        ll_val = compute_log_loss(probs, labels_flat).item()

    return {
        "loss": loss.item(),
        "brier_score": bs_val,
        "log_loss": ll_val,
    }
