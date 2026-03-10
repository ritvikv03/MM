"""
tests/model/test_temporal_encoder.py

RED phase — tests written before implementation exists.
Covers: TemporalEncoder class (lstm + transformer paths), make_temporal_encoder(),
and stack_snapshots() helper.

torch is available in this environment and used directly for forward-pass
shape assertions. torch.nn internals (nn.LSTM, nn.TransformerEncoder) are
verified via structural inspection rather than full mock replacement, because
the lazy-import constraint means torch is only imported inside the module
under test — not at module-import time.
"""

from __future__ import annotations

import sys
import importlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_module():
    """Fresh import of src.model.temporal_encoder (bypasses cache if needed)."""
    import src.model.temporal_encoder as m
    return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def lstm_config() -> dict:
    return {
        "input_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "encoder_type": "lstm",
    }


@pytest.fixture()
def transformer_config() -> dict:
    return {
        "input_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1,
        "encoder_type": "transformer",
    }


@pytest.fixture()
def lstm_encoder(lstm_config):
    from src.model.temporal_encoder import TemporalEncoder
    return TemporalEncoder(**lstm_config)


@pytest.fixture()
def transformer_encoder(transformer_config):
    from src.model.temporal_encoder import TemporalEncoder
    return TemporalEncoder(**transformer_config)


@pytest.fixture()
def input_tensor() -> torch.Tensor:
    """(T=5, N=10, D=64) float tensor."""
    torch.manual_seed(42)
    return torch.randn(5, 10, 64)


@pytest.fixture()
def single_step_tensor() -> torch.Tensor:
    """(T=1, N=10, D=64) — single timestep edge case."""
    torch.manual_seed(0)
    return torch.randn(1, 10, 64)


# ===========================================================================
# 1. Lazy import — module loads without torch at import time
# ===========================================================================

class TestLazyImport:
    """Verify torch is NOT imported at module-import time."""

    def test_module_imports_cleanly_without_torch_in_sys_modules(self):
        """Temporarily hide torch and confirm the module can still be imported."""
        # Save originals
        saved = {k: v for k, v in sys.modules.items() if k == "torch" or k.startswith("torch.")}

        # Build a fake sys.modules that hides torch
        fake_modules: dict[str, Any] = {k: None for k in saved}

        # Remove the cached module so it re-imports fresh
        mod_key = "src.model.temporal_encoder"
        original_mod = sys.modules.pop(mod_key, None)

        try:
            with patch.dict(sys.modules, fake_modules):
                # Import should succeed — lazy torch means no top-level torch usage
                import importlib as il
                mod = il.import_module(mod_key)
                assert mod is not None
        finally:
            # Restore everything
            if original_mod is not None:
                sys.modules[mod_key] = original_mod
            for k, v in saved.items():
                sys.modules[k] = v

    def test_module_has_temporal_encoder_class(self):
        m = _import_module()
        assert hasattr(m, "TemporalEncoder")

    def test_module_has_make_temporal_encoder(self):
        m = _import_module()
        assert hasattr(m, "make_temporal_encoder")

    def test_module_has_stack_snapshots(self):
        m = _import_module()
        assert hasattr(m, "stack_snapshots")


# ===========================================================================
# 2. TemporalEncoder.__init__ — attribute storage
# ===========================================================================

class TestTemporalEncoderInit:
    """Constructor stores all hyperparameters correctly."""

    def test_stores_input_dim(self, lstm_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        assert enc.input_dim == lstm_config["input_dim"]

    def test_stores_hidden_dim(self, lstm_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        assert enc.hidden_dim == lstm_config["hidden_dim"]

    def test_stores_num_layers(self, lstm_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        assert enc.num_layers == lstm_config["num_layers"]

    def test_stores_dropout(self, lstm_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        assert enc.dropout == lstm_config["dropout"]

    def test_stores_encoder_type_lstm(self, lstm_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        assert enc.encoder_type == "lstm"

    def test_stores_encoder_type_transformer(self, transformer_config):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**transformer_config)
        assert enc.encoder_type == "transformer"

    def test_invalid_encoder_type_raises_value_error(self):
        from src.model.temporal_encoder import TemporalEncoder
        with pytest.raises(ValueError, match="encoder_type"):
            TemporalEncoder(input_dim=64, encoder_type="rnn")

    def test_invalid_encoder_type_message_contains_invalid_value(self):
        from src.model.temporal_encoder import TemporalEncoder
        with pytest.raises(ValueError, match="gru"):
            TemporalEncoder(input_dim=64, encoder_type="gru")

    def test_is_nn_module(self, lstm_encoder):
        import torch.nn as nn
        assert isinstance(lstm_encoder, nn.Module)

    def test_default_hidden_dim_is_128(self):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(input_dim=32, encoder_type="lstm")
        assert enc.hidden_dim == 128

    def test_default_num_layers_is_2(self):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(input_dim=32, encoder_type="lstm")
        assert enc.num_layers == 2

    def test_default_dropout_is_0_1(self):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(input_dim=32, encoder_type="lstm")
        assert enc.dropout == pytest.approx(0.1)

    def test_default_encoder_type_is_lstm(self):
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(input_dim=32)
        assert enc.encoder_type == "lstm"


# ===========================================================================
# 3. LSTM path — internal structure
# ===========================================================================

class TestLSTMPath:
    """LSTM encoder builds and exposes nn.LSTM sub-module."""

    def test_lstm_encoder_has_lstm_attribute(self, lstm_encoder):
        import torch.nn as nn
        assert hasattr(lstm_encoder, "lstm")
        assert isinstance(lstm_encoder.lstm, nn.LSTM)

    def test_lstm_input_size_matches_input_dim(self, lstm_config, lstm_encoder):
        assert lstm_encoder.lstm.input_size == lstm_config["input_dim"]

    def test_lstm_hidden_size_matches_hidden_dim(self, lstm_config, lstm_encoder):
        assert lstm_encoder.lstm.hidden_size == lstm_config["hidden_dim"]

    def test_lstm_num_layers_matches_config(self, lstm_config, lstm_encoder):
        assert lstm_encoder.lstm.num_layers == lstm_config["num_layers"]

    def test_lstm_batch_first_is_false(self, lstm_encoder):
        assert lstm_encoder.lstm.batch_first is False

    def test_lstm_dropout_nonzero_when_multilayer(self, lstm_config, lstm_encoder):
        """Dropout is set on the LSTM when num_layers > 1."""
        assert lstm_encoder.lstm.dropout == pytest.approx(lstm_config["dropout"])

    def test_lstm_dropout_zero_when_single_layer(self):
        """Dropout must be 0 for a single-layer LSTM (avoids PyTorch UserWarning)."""
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(input_dim=32, hidden_dim=64, num_layers=1, dropout=0.3, encoder_type="lstm")
        assert enc.lstm.dropout == pytest.approx(0.0)


# ===========================================================================
# 4. Transformer path — internal structure
# ===========================================================================

class TestTransformerPath:
    """Transformer encoder builds nn.TransformerEncoder with projection."""

    def test_transformer_encoder_has_transformer_attribute(self, transformer_encoder):
        import torch.nn as nn
        assert hasattr(transformer_encoder, "transformer")
        assert isinstance(transformer_encoder.transformer, nn.TransformerEncoder)

    def test_transformer_encoder_has_projection_attribute(self, transformer_encoder):
        import torch.nn as nn
        assert hasattr(transformer_encoder, "projection")
        assert isinstance(transformer_encoder.projection, nn.Linear)

    def test_transformer_projection_in_features_is_input_dim(
        self, transformer_config, transformer_encoder
    ):
        assert transformer_encoder.projection.in_features == transformer_config["input_dim"]

    def test_transformer_projection_out_features_is_hidden_dim(
        self, transformer_config, transformer_encoder
    ):
        assert transformer_encoder.projection.out_features == transformer_config["hidden_dim"]

    def test_transformer_num_layers_matches_config(
        self, transformer_config, transformer_encoder
    ):
        assert transformer_encoder.transformer.num_layers == transformer_config["num_layers"]


# ===========================================================================
# 5. Forward pass — LSTM output shapes
# ===========================================================================

class TestLSTMForwardShape:
    """LSTM forward pass produces correct output shapes."""

    def test_lstm_forward_output_shape(self, lstm_encoder, input_tensor, lstm_config):
        """(T, N, D) → (N, hidden_dim)."""
        lstm_encoder.eval()
        with torch.no_grad():
            out = lstm_encoder(input_tensor)
        N = input_tensor.shape[1]
        assert out.shape == (N, lstm_config["hidden_dim"])

    def test_lstm_forward_output_is_tensor(self, lstm_encoder, input_tensor):
        lstm_encoder.eval()
        with torch.no_grad():
            out = lstm_encoder(input_tensor)
        assert isinstance(out, torch.Tensor)

    def test_lstm_forward_single_timestep(self, lstm_encoder, single_step_tensor, lstm_config):
        """Single timestep input should still produce (N, hidden_dim)."""
        lstm_encoder.eval()
        with torch.no_grad():
            out = lstm_encoder(single_step_tensor)
        N = single_step_tensor.shape[1]
        assert out.shape == (N, lstm_config["hidden_dim"])

    def test_lstm_forward_single_team(self, lstm_config):
        """N=1 (single team) edge case."""
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**lstm_config)
        enc.eval()
        x = torch.randn(4, 1, lstm_config["input_dim"])
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, lstm_config["hidden_dim"])

    def test_lstm_forward_output_dtype_is_float32(self, lstm_encoder, input_tensor):
        lstm_encoder.eval()
        with torch.no_grad():
            out = lstm_encoder(input_tensor)
        assert out.dtype == torch.float32


# ===========================================================================
# 6. Forward pass — Transformer output shapes
# ===========================================================================

class TestTransformerForwardShape:
    """Transformer forward pass produces correct output shapes."""

    def test_transformer_forward_output_shape(
        self, transformer_encoder, input_tensor, transformer_config
    ):
        """(T, N, D) → (N, hidden_dim)."""
        transformer_encoder.eval()
        with torch.no_grad():
            out = transformer_encoder(input_tensor)
        N = input_tensor.shape[1]
        assert out.shape == (N, transformer_config["hidden_dim"])

    def test_transformer_forward_output_is_tensor(
        self, transformer_encoder, input_tensor
    ):
        transformer_encoder.eval()
        with torch.no_grad():
            out = transformer_encoder(input_tensor)
        assert isinstance(out, torch.Tensor)

    def test_transformer_forward_single_timestep(
        self, transformer_encoder, single_step_tensor, transformer_config
    ):
        """Single timestep input should still produce (N, hidden_dim)."""
        transformer_encoder.eval()
        with torch.no_grad():
            out = transformer_encoder(single_step_tensor)
        N = single_step_tensor.shape[1]
        assert out.shape == (N, transformer_config["hidden_dim"])

    def test_transformer_forward_single_team(self, transformer_config):
        """N=1 (single team) edge case."""
        from src.model.temporal_encoder import TemporalEncoder
        enc = TemporalEncoder(**transformer_config)
        enc.eval()
        x = torch.randn(4, 1, transformer_config["input_dim"])
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, transformer_config["hidden_dim"])


# ===========================================================================
# 7. make_temporal_encoder — factory function
# ===========================================================================

class TestMakeTemporalEncoder:
    """make_temporal_encoder() constructs TemporalEncoder from config dict."""

    def test_returns_temporal_encoder_instance(self, lstm_config):
        from src.model.temporal_encoder import make_temporal_encoder, TemporalEncoder
        enc = make_temporal_encoder(lstm_config)
        assert isinstance(enc, TemporalEncoder)

    def test_constructs_lstm_type(self, lstm_config):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder(lstm_config)
        assert enc.encoder_type == "lstm"

    def test_constructs_transformer_type(self, transformer_config):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder(transformer_config)
        assert enc.encoder_type == "transformer"

    def test_raises_value_error_for_invalid_encoder_type(self):
        from src.model.temporal_encoder import make_temporal_encoder
        with pytest.raises(ValueError):
            make_temporal_encoder({"input_dim": 32, "encoder_type": "bad_type"})

    def test_raises_value_error_message_contains_encoder_type(self):
        from src.model.temporal_encoder import make_temporal_encoder
        with pytest.raises(ValueError, match="encoder_type"):
            make_temporal_encoder({"input_dim": 32, "encoder_type": "cnn"})

    def test_default_hidden_dim_from_config(self):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder({"input_dim": 32})
        assert enc.hidden_dim == 128

    def test_default_num_layers_from_config(self):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder({"input_dim": 32})
        assert enc.num_layers == 2

    def test_default_dropout_from_config(self):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder({"input_dim": 32})
        assert enc.dropout == pytest.approx(0.1)

    def test_default_encoder_type_from_config(self):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder({"input_dim": 32})
        assert enc.encoder_type == "lstm"

    def test_config_values_propagate_to_encoder(self, lstm_config):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder(lstm_config)
        assert enc.input_dim == lstm_config["input_dim"]
        assert enc.hidden_dim == lstm_config["hidden_dim"]
        assert enc.num_layers == lstm_config["num_layers"]


# ===========================================================================
# 8. stack_snapshots — helper function
# ===========================================================================

class TestStackSnapshots:
    """stack_snapshots() correctly stacks a list of (N, D) tensors → (T, N, D)."""

    def test_basic_shape_T_N_D(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(10, 64) for _ in range(5)]
        result = stack_snapshots(tensors)
        assert result.shape == (5, 10, 64)

    def test_single_snapshot_gives_T1(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(10, 64)]
        result = stack_snapshots(tensors)
        assert result.shape == (1, 10, 64)

    def test_returns_tensor(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(8, 32) for _ in range(3)]
        result = stack_snapshots(tensors)
        assert isinstance(result, torch.Tensor)

    def test_raises_value_error_on_empty_list(self):
        from src.model.temporal_encoder import stack_snapshots
        with pytest.raises(ValueError, match="empty"):
            stack_snapshots([])

    def test_raises_value_error_on_inconsistent_N(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(10, 64), torch.randn(11, 64)]
        with pytest.raises(ValueError):
            stack_snapshots(tensors)

    def test_raises_value_error_on_inconsistent_D(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(10, 64), torch.randn(10, 32)]
        with pytest.raises(ValueError):
            stack_snapshots(tensors)

    def test_preserves_values_in_order(self):
        from src.model.temporal_encoder import stack_snapshots
        t0 = torch.ones(4, 8)
        t1 = torch.zeros(4, 8)
        result = stack_snapshots([t0, t1])
        assert torch.all(result[0] == 1.0)
        assert torch.all(result[1] == 0.0)

    def test_T_dimension_equals_list_length(self):
        from src.model.temporal_encoder import stack_snapshots
        T = 7
        tensors = [torch.randn(5, 16) for _ in range(T)]
        result = stack_snapshots(tensors)
        assert result.shape[0] == T

    def test_N_dimension_preserved(self):
        from src.model.temporal_encoder import stack_snapshots
        N = 12
        tensors = [torch.randn(N, 32) for _ in range(3)]
        result = stack_snapshots(tensors)
        assert result.shape[1] == N

    def test_D_dimension_preserved(self):
        from src.model.temporal_encoder import stack_snapshots
        D = 48
        tensors = [torch.randn(6, D) for _ in range(3)]
        result = stack_snapshots(tensors)
        assert result.shape[2] == D

    def test_float_dtype_preserved(self):
        from src.model.temporal_encoder import stack_snapshots
        tensors = [torch.randn(4, 8).float() for _ in range(2)]
        result = stack_snapshots(tensors)
        assert result.dtype == torch.float32


# ===========================================================================
# 9. end-to-end: stack_snapshots → TemporalEncoder
# ===========================================================================

class TestEndToEnd:
    """Combine stack_snapshots with TemporalEncoder.forward."""

    def test_stack_then_lstm_forward_shape(self, lstm_config):
        from src.model.temporal_encoder import stack_snapshots, TemporalEncoder
        snapshots = [torch.randn(10, lstm_config["input_dim"]) for _ in range(6)]
        x = stack_snapshots(snapshots)
        enc = TemporalEncoder(**lstm_config)
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (10, lstm_config["hidden_dim"])

    def test_stack_then_transformer_forward_shape(self, transformer_config):
        from src.model.temporal_encoder import stack_snapshots, TemporalEncoder
        snapshots = [torch.randn(10, transformer_config["input_dim"]) for _ in range(6)]
        x = stack_snapshots(snapshots)
        enc = TemporalEncoder(**transformer_config)
        enc.eval()
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (10, transformer_config["hidden_dim"])

    def test_make_encoder_then_forward(self, lstm_config, input_tensor):
        from src.model.temporal_encoder import make_temporal_encoder
        enc = make_temporal_encoder(lstm_config)
        enc.eval()
        with torch.no_grad():
            out = enc(input_tensor)
        assert out.shape == (input_tensor.shape[1], lstm_config["hidden_dim"])
