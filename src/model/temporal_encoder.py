"""
src/model/temporal_encoder.py

Temporal encoder for the NCAA March Madness ST-GNN.

Consumes a sequence of per-snapshot GATEncoder output embeddings
(one embedding matrix per TemporalSnapshot in the season) and produces
a single final team embedding that captures momentum and trajectory.

Two encoder back-ends are supported:

* ``"lstm"``        — stacked LSTM; last hidden state is returned.
* ``"transformer"`` — stacked TransformerEncoder; last timestep is projected.

Lazy imports
------------
``torch`` and ``torch.nn`` are imported *inside* functions/methods, never at
module-import time.  This satisfies the constraint that the module loads
cleanly even if torch is not installed.

The :class:`TemporalEncoder` class itself is built the first time it is
referenced via the module-level ``_build_temporal_encoder_class()`` factory,
which defers the ``nn.Module`` base-class resolution until torch is available.

Usage example
-------------
>>> from src.model.temporal_encoder import make_temporal_encoder, stack_snapshots
>>> enc = make_temporal_encoder({"input_dim": 64, "encoder_type": "lstm"})
>>> x = stack_snapshots(snapshot_embeddings)   # list[Tensor(N, D)] → (T, N, D)
>>> team_embeddings = enc(x)                   # (N, hidden_dim)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only for static type checkers; never executed at runtime.
    import torch
    import torch.nn as nn


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def stack_snapshots(embeddings: list) -> "torch.Tensor":
    """Stack a list of per-snapshot embedding tensors into a single 3-D tensor.

    Parameters
    ----------
    embeddings:
        A list of ``T`` tensors, each of shape ``(N, D)`` where *N* is the
        number of teams and *D* is the feature dimension.

    Returns
    -------
    torch.Tensor
        Shape ``(T, N, D)`` — ready to feed into :class:`TemporalEncoder`.

    Raises
    ------
    ValueError
        If *embeddings* is empty or if the tensors have inconsistent shapes.
    """
    import torch  # lazy

    if len(embeddings) == 0:
        raise ValueError(
            "stack_snapshots received an empty list. "
            "At least one snapshot embedding is required."
        )

    reference_shape = embeddings[0].shape
    for i, t in enumerate(embeddings[1:], start=1):
        if t.shape != reference_shape:
            raise ValueError(
                f"Inconsistent embedding shapes: snapshot 0 has shape "
                f"{reference_shape} but snapshot {i} has shape {t.shape}."
            )

    return torch.stack(embeddings, dim=0)  # (T, N, D)


# ---------------------------------------------------------------------------
# TemporalEncoder class builder
# ---------------------------------------------------------------------------

def _build_temporal_encoder_class():
    """Return the TemporalEncoder class with nn.Module as a live base class.

    Called lazily the first time :data:`TemporalEncoder` is resolved so that
    torch is never imported at module-import time.
    """
    import torch.nn as nn  # lazy

    _VALID_TYPES = frozenset({"lstm", "transformer"})

    class TemporalEncoder(nn.Module):
        """Temporal encoder that converts a sequence of graph embeddings into
        a single momentum embedding per team.

        Parameters
        ----------
        input_dim:
            Dimensionality of the incoming embeddings
            (= ``GATEncoder.hidden_dim``).
        hidden_dim:
            Dimensionality of the output momentum embeddings.
        num_layers:
            Number of stacked LSTM / Transformer layers.
        dropout:
            Dropout probability.  For LSTM with a single layer this is forced
            to ``0.0`` to avoid a PyTorch ``UserWarning``.
        encoder_type:
            ``"lstm"`` or ``"transformer"``.  Any other value raises
            :class:`ValueError` immediately at construction time.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.1,
            encoder_type: str = "lstm",
        ) -> None:
            if encoder_type not in _VALID_TYPES:
                raise ValueError(
                    f"encoder_type must be one of {set(_VALID_TYPES)!r}, "
                    f"got {encoder_type!r}."
                )

            super().__init__()

            # Hyper-parameters (stored for introspection / config round-trips)
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.encoder_type = encoder_type

            # ---- LSTM back-end --------------------------------------------
            if encoder_type == "lstm":
                # PyTorch raises UserWarning when dropout > 0 with
                # num_layers == 1.
                lstm_dropout = dropout if num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=False,
                    dropout=lstm_dropout,
                )

            # ---- Transformer back-end -------------------------------------
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=4,
                    dim_feedforward=hidden_dim * 2,
                    dropout=dropout,
                    batch_first=False,  # expects (T, N, D)
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=num_layers,
                )
                # Project input_dim → hidden_dim after taking last timestep
                self.projection = nn.Linear(input_dim, hidden_dim)

        # ------------------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------------------

        def forward(self, x):
            """Run the temporal encoder.

            Parameters
            ----------
            x:
                Tensor of shape ``(T, N, input_dim)`` — *T* sequential
                snapshots, *N* teams, *D* feature dimensions.

            Returns
            -------
            torch.Tensor
                Shape ``(N, hidden_dim)`` — one momentum embedding per team.
            """
            if self.encoder_type == "lstm":
                # _output: (T, N, hidden_dim)
                # h_n:     (num_layers, N, hidden_dim)
                _output, (h_n, _c_n) = self.lstm(x)
                # Last layer's hidden state
                return h_n[-1]  # (N, hidden_dim)

            else:  # transformer
                # output: (T, N, input_dim)
                output = self.transformer(x)
                # Last timestep
                last = output[-1]        # (N, input_dim)
                return self.projection(last)  # (N, hidden_dim)

    return TemporalEncoder


# ---------------------------------------------------------------------------
# Lazy module-level attribute resolution
# ---------------------------------------------------------------------------

class _LazyModuleProxy:
    """Proxy that builds :class:`TemporalEncoder` on first attribute access."""

    _cls = None

    @classmethod
    def _get_cls(cls):
        if cls._cls is None:
            cls._cls = _build_temporal_encoder_class()
        return cls._cls


# Expose TemporalEncoder as if it were a top-level class.
# ``from src.model.temporal_encoder import TemporalEncoder`` triggers
# ``__getattr__`` on the module, which calls _get_cls() the first time.

def __getattr__(name: str):
    if name == "TemporalEncoder":
        return _LazyModuleProxy._get_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_temporal_encoder(config: dict) -> "TemporalEncoder":
    """Construct a :class:`TemporalEncoder` from a plain config dictionary.

    Parameters
    ----------
    config:
        Dictionary with the following keys (all optional except *input_dim*):

        * ``input_dim``    — (required) input feature dimension.
        * ``hidden_dim``   — default ``128``.
        * ``num_layers``   — default ``2``.
        * ``dropout``      — default ``0.1``.
        * ``encoder_type`` — default ``"lstm"``; must be ``"lstm"`` or
          ``"transformer"``, else :class:`ValueError` is raised.

    Returns
    -------
    TemporalEncoder

    Raises
    ------
    ValueError
        If ``encoder_type`` is not ``"lstm"`` or ``"transformer"``.
    """
    encoder_type = config.get("encoder_type", "lstm")

    _VALID_TYPES = {"lstm", "transformer"}
    if encoder_type not in _VALID_TYPES:
        raise ValueError(
            f"encoder_type must be one of {_VALID_TYPES!r}, "
            f"got {encoder_type!r}."
        )

    TemporalEncoder = _LazyModuleProxy._get_cls()
    return TemporalEncoder(
        input_dim=config["input_dim"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.1),
        encoder_type=encoder_type,
    )
