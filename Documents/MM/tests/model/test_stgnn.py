"""
tests/model/test_stgnn.py

RED phase — tests written before implementation exists.
Covers: STGNNModel construction, forward pass output shape,
brier_score formula, compute_log_loss clamping, train_one_epoch
return keys, make_stgnn_model factory, and lazy imports.

All torch_geometric / GATEncoder / TemporalEncoder internals are mocked via
sys.modules patching so no real PyG installation is needed to run the suite.
Real torch is used for tensor-math assertions (brier_score, log_loss, etc.).
"""

from __future__ import annotations

import sys
import pathlib
from unittest.mock import MagicMock, patch
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

GAT_HIDDEN = 16
TEMPORAL_HIDDEN = 8
N_TEAMS = 5
T_SNAPSHOTS = 3
N_GAMES = 4
OUTPUT_DIM = 1


# ---------------------------------------------------------------------------
# Fixtures — configs
# ---------------------------------------------------------------------------

@pytest.fixture()
def gat_config():
    """Valid gat_config keys for make_gat_encoder (gat_encoder.py)."""
    return {
        "node_in_features": 3,
        "edge_in_features": 3,
        "hidden_dim": GAT_HIDDEN,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
    }


@pytest.fixture()
def temporal_config():
    """Valid temporal_config keys for make_temporal_encoder."""
    return {
        "input_dim": GAT_HIDDEN,
        "hidden_dim": TEMPORAL_HIDDEN,
        "num_layers": 1,
        "dropout": 0.1,
    }


@pytest.fixture()
def fake_snapshots():
    """List of T_SNAPSHOTS fake TemporalSnapshot-like objects."""
    snapshots = []
    for _ in range(T_SNAPSHOTS):
        snap = MagicMock()
        snap.node_features = torch.randn(N_TEAMS, 3)
        snap.edge_index = torch.zeros(2, 2, dtype=torch.long)
        snap.edge_attr = torch.randn(2, 3)
        snapshots.append(snap)
    return snapshots


@pytest.fixture()
def game_indices():
    home_idx = torch.randint(0, N_TEAMS, (N_GAMES,))
    away_idx = torch.randint(0, N_TEAMS, (N_GAMES,))
    return home_idx, away_idx


@pytest.fixture()
def labels():
    return torch.randint(0, 2, (N_GAMES,)).float()


# ---------------------------------------------------------------------------
# Module-level encoder patch helper
# ---------------------------------------------------------------------------

def _make_encoder_mocks():
    """Return (mock_gat_encoder_obj, mock_temporal_encoder_obj, patch_ctx).

    Sets up two mock encoder objects and patches
    src.model.gat_encoder.make_gat_encoder and
    src.model.temporal_encoder.make_temporal_encoder so that STGNNModel.__init__
    receives predictable encoder instances.
    """
    mock_gat = MagicMock(name="gat_encoder_instance")
    mock_gat.return_value = torch.randn(N_TEAMS, GAT_HIDDEN)

    mock_temporal = MagicMock(name="temporal_encoder_instance")
    mock_temporal.return_value = torch.randn(N_TEAMS, TEMPORAL_HIDDEN)

    return mock_gat, mock_temporal


@pytest.fixture()
def model_with_mocked_encoders(gat_config, temporal_config):
    """STGNNModel whose gat_encoder and temporal_encoder are replaced with mocks.

    make_gat_encoder and make_temporal_encoder are patched in stgnn.py's
    namespace so __init__ never calls real PyG code.  After construction the
    encoders are swapped to fresh deterministic MagicMocks for clean
    call-count tracking.
    """
    mock_gat, mock_temporal = _make_encoder_mocks()
    with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
         patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
        from src.model.stgnn import STGNNModel
        model = STGNNModel(gat_config=gat_config, temporal_config=temporal_config)
    model.gat_encoder = mock_gat
    model.temporal_encoder = mock_temporal
    return model


# ---------------------------------------------------------------------------
# TestSTGNNModelConstruction
# ---------------------------------------------------------------------------

class TestSTGNNModelConstruction:
    """Verify STGNNModel can be constructed and stores configs."""

    def _make_model(self, gat_config, temporal_config, **kwargs):
        """Construct STGNNModel with patched encoder factories."""
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import STGNNModel
            return STGNNModel(gat_config=gat_config, temporal_config=temporal_config, **kwargs)

    def test_model_instantiation_does_not_raise(self, gat_config, temporal_config):
        """STGNNModel(gat_config, temporal_config) should not raise."""
        model = self._make_model(gat_config, temporal_config)
        assert model is not None

    def test_model_is_nn_module(self, gat_config, temporal_config):
        """STGNNModel must subclass torch.nn.Module."""
        model = self._make_model(gat_config, temporal_config)
        assert isinstance(model, nn.Module)

    def test_model_stores_output_dim_default(self, gat_config, temporal_config):
        """Default output_dim is 1."""
        model = self._make_model(gat_config, temporal_config)
        assert model.output_dim == 1

    def test_model_stores_custom_output_dim(self, gat_config, temporal_config):
        """Custom output_dim is stored correctly."""
        model = self._make_model(gat_config, temporal_config, output_dim=3)
        assert model.output_dim == 3

    def test_model_has_gat_encoder_attribute(self, gat_config, temporal_config):
        """Model must expose .gat_encoder attribute."""
        model = self._make_model(gat_config, temporal_config)
        assert hasattr(model, "gat_encoder")

    def test_model_has_temporal_encoder_attribute(self, gat_config, temporal_config):
        """Model must expose .temporal_encoder attribute."""
        model = self._make_model(gat_config, temporal_config)
        assert hasattr(model, "temporal_encoder")

    def test_model_has_prediction_head_attribute(self, gat_config, temporal_config):
        """Model must expose .prediction_head attribute."""
        model = self._make_model(gat_config, temporal_config)
        assert hasattr(model, "prediction_head")


# ---------------------------------------------------------------------------
# TestSTGNNModelForward
# ---------------------------------------------------------------------------

class TestSTGNNModelForward:
    """Verify forward() output shape and types."""

    def _build_model(self, gat_config, temporal_config, output_dim=1):
        """Build STGNNModel with mocked encoders for forward tests."""
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import STGNNModel
            model = STGNNModel(
                gat_config=gat_config,
                temporal_config=temporal_config,
                output_dim=output_dim,
            )
        # Fresh mocks with proper return tensors for forward calls
        gat_mock = MagicMock()
        gat_mock.return_value = torch.randn(N_TEAMS, GAT_HIDDEN)
        temporal_mock = MagicMock()
        temporal_mock.return_value = torch.randn(N_TEAMS, TEMPORAL_HIDDEN)
        model.gat_encoder = gat_mock
        model.temporal_encoder = temporal_mock
        return model

    def test_forward_returns_tensor(self, gat_config, temporal_config, fake_snapshots, game_indices):
        model = self._build_model(gat_config, temporal_config)
        home_idx, away_idx = game_indices
        out = model(fake_snapshots, home_idx, away_idx)
        assert isinstance(out, torch.Tensor)

    def test_forward_output_shape_G_by_output_dim(self, gat_config, temporal_config, fake_snapshots, game_indices):
        model = self._build_model(gat_config, temporal_config)
        home_idx, away_idx = game_indices
        out = model(fake_snapshots, home_idx, away_idx)
        assert out.shape == (N_GAMES, OUTPUT_DIM)

    def test_forward_output_shape_custom_output_dim(self, gat_config, temporal_config, fake_snapshots, game_indices):
        model = self._build_model(gat_config, temporal_config, output_dim=2)
        home_idx, away_idx = game_indices
        out = model(fake_snapshots, home_idx, away_idx)
        assert out.shape == (N_GAMES, 2)

    def test_forward_calls_gat_encoder_for_each_snapshot(
        self, gat_config, temporal_config, fake_snapshots, game_indices
    ):
        """gat_encoder should be called once per snapshot."""
        model = self._build_model(gat_config, temporal_config)
        home_idx, away_idx = game_indices
        model(fake_snapshots, home_idx, away_idx)
        assert model.gat_encoder.call_count == T_SNAPSHOTS

    def test_forward_calls_temporal_encoder_once(
        self, gat_config, temporal_config, fake_snapshots, game_indices
    ):
        model = self._build_model(gat_config, temporal_config)
        home_idx, away_idx = game_indices
        model(fake_snapshots, home_idx, away_idx)
        assert model.temporal_encoder.call_count == 1

    def test_forward_output_dtype_is_float(self, gat_config, temporal_config, fake_snapshots, game_indices):
        model = self._build_model(gat_config, temporal_config)
        home_idx, away_idx = game_indices
        out = model(fake_snapshots, home_idx, away_idx)
        assert out.dtype in (torch.float32, torch.float64)

    def test_forward_output_is_raw_logits_not_probabilities(
        self, gat_config, temporal_config, fake_snapshots, game_indices
    ):
        """Raw logits can exceed [0,1] range — probabilities cannot.
        Run several attempts to catch at least one value outside [0,1].
        """
        found_outside = False
        for _ in range(20):
            model = self._build_model(gat_config, temporal_config)
            home_idx, away_idx = game_indices
            out = model(fake_snapshots, home_idx, away_idx)
            if out.min().item() < 0.0 or out.max().item() > 1.0:
                found_outside = True
                break
        assert found_outside, "Expected raw logits to occasionally fall outside [0,1]"


# ---------------------------------------------------------------------------
# TestStackSnapshots — re-exported helper
# ---------------------------------------------------------------------------

class TestStackSnapshots:
    """Verify stack_snapshots re-export from stgnn module."""

    def test_stack_snapshots_shape(self):
        from src.model.stgnn import stack_snapshots
        embeddings = [torch.randn(N_TEAMS, GAT_HIDDEN) for _ in range(T_SNAPSHOTS)]
        stacked = stack_snapshots(embeddings)
        assert stacked.shape == (T_SNAPSHOTS, N_TEAMS, GAT_HIDDEN)

    def test_stack_snapshots_returns_tensor(self):
        from src.model.stgnn import stack_snapshots
        embeddings = [torch.randn(N_TEAMS, GAT_HIDDEN) for _ in range(T_SNAPSHOTS)]
        result = stack_snapshots(embeddings)
        assert isinstance(result, torch.Tensor)


# ---------------------------------------------------------------------------
# TestBrierScore
# ---------------------------------------------------------------------------

class TestBrierScore:
    """Unit tests for brier_score() with known numerical inputs."""

    def test_brier_score_perfect_prediction_is_zero(self):
        from src.model.stgnn import brier_score
        probs = torch.tensor([1.0, 0.0, 1.0, 0.0])
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = brier_score(probs, labels)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_brier_score_worst_prediction(self):
        from src.model.stgnn import brier_score
        # Always predicting 1.0 when true is 0.0 → BS = 1.0
        probs = torch.tensor([1.0, 1.0])
        labels = torch.tensor([0.0, 0.0])
        result = brier_score(probs, labels)
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_brier_score_known_value(self):
        from src.model.stgnn import brier_score
        # probs = [0.9], labels = [1.0] → (0.9-1.0)^2 = 0.01
        probs = torch.tensor([0.9])
        labels = torch.tensor([1.0])
        result = brier_score(probs, labels)
        assert result.item() == pytest.approx(0.01, abs=1e-5)

    def test_brier_score_two_samples_mean(self):
        from src.model.stgnn import brier_score
        # (0.8-1)^2 + (0.3-0)^2 = 0.04 + 0.09 = 0.13 → mean = 0.065
        probs = torch.tensor([0.8, 0.3])
        labels = torch.tensor([1.0, 0.0])
        result = brier_score(probs, labels)
        assert result.item() == pytest.approx(0.065, abs=1e-5)

    def test_brier_score_returns_scalar(self):
        from src.model.stgnn import brier_score
        probs = torch.tensor([0.5, 0.7, 0.2])
        labels = torch.tensor([1.0, 0.0, 1.0])
        result = brier_score(probs, labels)
        assert result.ndim == 0  # scalar tensor

    def test_brier_score_accepts_2d_input(self):
        from src.model.stgnn import brier_score
        # 2-sample mean: (0.9-1)^2 + (0.1-0)^2 = 0.01 + 0.01 = 0.02 → mean=0.01
        probs = torch.tensor([[0.9], [0.1]])
        labels = torch.tensor([[1.0], [0.0]])
        result = brier_score(probs, labels)
        assert result.item() == pytest.approx(0.01, abs=1e-5)

    def test_brier_score_is_nonnegative(self):
        from src.model.stgnn import brier_score
        probs = torch.rand(10)
        labels = torch.randint(0, 2, (10,)).float()
        result = brier_score(probs, labels)
        assert result.item() >= 0.0

    def test_brier_score_no_side_effects(self):
        """brier_score must not modify the input tensors."""
        from src.model.stgnn import brier_score
        probs = torch.tensor([0.7, 0.3])
        labels = torch.tensor([1.0, 0.0])
        probs_copy = probs.clone()
        labels_copy = labels.clone()
        brier_score(probs, labels)
        assert torch.equal(probs, probs_copy)
        assert torch.equal(labels, labels_copy)


# ---------------------------------------------------------------------------
# TestComputeLogLoss
# ---------------------------------------------------------------------------

class TestComputeLogLoss:
    """Unit tests for compute_log_loss()."""

    def test_log_loss_perfect_prediction_near_zero(self):
        from src.model.stgnn import compute_log_loss
        probs = torch.tensor([1.0 - 1e-6, 1e-6])
        labels = torch.tensor([1.0, 0.0])
        result = compute_log_loss(probs, labels)
        assert result.item() < 0.01

    def test_log_loss_returns_scalar(self):
        from src.model.stgnn import compute_log_loss
        probs = torch.tensor([0.6, 0.4, 0.8])
        labels = torch.tensor([1.0, 0.0, 1.0])
        result = compute_log_loss(probs, labels)
        assert result.ndim == 0

    def test_log_loss_clamping_prevents_nan_for_zero_prob(self):
        from src.model.stgnn import compute_log_loss
        # Without clamping, log(0) = -inf → NaN in BCE. With clamping, must be finite.
        probs = torch.tensor([0.0, 1.0])
        labels = torch.tensor([1.0, 0.0])
        result = compute_log_loss(probs, labels)
        assert torch.isfinite(result)

    def test_log_loss_clamping_prevents_nan_for_one_prob(self):
        from src.model.stgnn import compute_log_loss
        probs = torch.tensor([1.0])
        labels = torch.tensor([0.0])
        result = compute_log_loss(probs, labels)
        assert torch.isfinite(result)

    def test_log_loss_is_nonnegative(self):
        from src.model.stgnn import compute_log_loss
        probs = torch.rand(10).clamp(0.05, 0.95)
        labels = torch.randint(0, 2, (10,)).float()
        result = compute_log_loss(probs, labels)
        assert result.item() >= 0.0

    def test_log_loss_custom_eps_clamps_correctly(self):
        from src.model.stgnn import compute_log_loss
        # With eps=0.1, prob=0.0 should be clamped to 0.1
        probs = torch.tensor([0.0])
        labels = torch.tensor([1.0])
        result = compute_log_loss(probs, labels, eps=0.1)
        expected = -torch.log(torch.tensor(0.1)).item()
        assert result.item() == pytest.approx(expected, abs=1e-5)

    def test_log_loss_no_side_effects(self):
        """compute_log_loss must not modify the input tensors."""
        from src.model.stgnn import compute_log_loss
        probs = torch.tensor([0.7, 0.3])
        labels = torch.tensor([1.0, 0.0])
        probs_copy = probs.clone()
        labels_copy = labels.clone()
        compute_log_loss(probs, labels)
        assert torch.equal(probs, probs_copy)
        assert torch.equal(labels, labels_copy)


# ---------------------------------------------------------------------------
# TestMakeStgnnModel
# ---------------------------------------------------------------------------

class TestMakeStgnnModel:
    """Tests for the make_stgnn_model factory function."""

    def _make(self, gat_config, temporal_config, **kwargs):
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import make_stgnn_model, STGNNModel
            model = make_stgnn_model(gat_config, temporal_config, **kwargs)
            return model, STGNNModel

    def test_returns_stgnn_model_instance(self, gat_config, temporal_config):
        from src.model.stgnn import STGNNModel
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import make_stgnn_model
            model = make_stgnn_model(gat_config, temporal_config)
        assert isinstance(model, STGNNModel)

    def test_factory_passes_output_dim(self, gat_config, temporal_config):
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import make_stgnn_model
            model = make_stgnn_model(gat_config, temporal_config, output_dim=3)
        assert model.output_dim == 3

    def test_factory_default_output_dim_is_1(self, gat_config, temporal_config):
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import make_stgnn_model
            model = make_stgnn_model(gat_config, temporal_config)
        assert model.output_dim == 1


# ---------------------------------------------------------------------------
# TestTrainOneEpoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    """Tests for train_one_epoch() utility function."""

    def _make_model_for_training(self, gat_config, temporal_config):
        """Return model with mocked encoders that produce valid tensors."""
        mock_gat, mock_temporal = _make_encoder_mocks()
        with patch("src.model.stgnn.make_gat_encoder", return_value=mock_gat), \
             patch("src.model.stgnn.make_temporal_encoder", return_value=mock_temporal):
            from src.model.stgnn import STGNNModel
            model = STGNNModel(gat_config=gat_config, temporal_config=temporal_config)
        # Replace with fresh mocks for isolated call tracking
        gat_mock = MagicMock()
        gat_mock.return_value = torch.randn(N_TEAMS, GAT_HIDDEN)
        temporal_mock = MagicMock()
        temporal_mock.return_value = torch.randn(N_TEAMS, TEMPORAL_HIDDEN)
        model.gat_encoder = gat_mock
        model.temporal_encoder = temporal_mock
        return model

    def test_train_one_epoch_returns_dict(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert isinstance(result, dict)

    def test_train_one_epoch_returns_loss_key(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert "loss" in result

    def test_train_one_epoch_returns_brier_score_key(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert "brier_score" in result

    def test_train_one_epoch_returns_log_loss_key(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert "log_loss" in result

    def test_train_one_epoch_loss_is_float(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert isinstance(result["loss"], float)

    def test_train_one_epoch_brier_score_is_float(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert isinstance(result["brier_score"], float)

    def test_train_one_epoch_log_loss_is_float(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert isinstance(result["log_loss"], float)

    def test_train_one_epoch_loss_is_nonnegative(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert result["loss"] >= 0.0

    def test_train_one_epoch_sets_model_to_train_mode(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        """train_one_epoch must call model.train() so model is in training mode."""
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        # Force eval mode first
        model.eval()
        assert not model.training
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        # train_one_epoch must call model.train() at start, leaving model in training mode
        assert model.training

    def test_train_one_epoch_all_three_keys_present(
        self, gat_config, temporal_config, fake_snapshots, game_indices, labels
    ):
        from src.model.stgnn import train_one_epoch
        model = self._make_model_for_training(gat_config, temporal_config)
        optimizer = torch.optim.SGD(model.prediction_head.parameters(), lr=0.01)
        home_idx, away_idx = game_indices
        result = train_one_epoch(model, optimizer, fake_snapshots, home_idx, away_idx, labels)
        assert set(result.keys()) >= {"loss", "brier_score", "log_loss"}


# ---------------------------------------------------------------------------
# TestLazyImports
# ---------------------------------------------------------------------------

class TestLazyImports:
    """Verify torch is imported lazily inside functions, not at module level."""

    def test_stgnn_module_has_no_unconditional_top_level_torch_import(self):
        """
        Verify that 'import torch' does not appear at column 0 (module level)
        outside of a TYPE_CHECKING guard in stgnn.py.

        This confirms the lazy-import constraint: torch is only imported inside
        function/method bodies, never at module load time.
        """
        stgnn_path = (
            pathlib.Path(__file__).parents[2] / "src" / "model" / "stgnn.py"
        )
        source = stgnn_path.read_text()
        lines = source.splitlines()

        in_type_checking_block = False
        for line in lines:
            stripped = line.rstrip()
            # Detect entry into TYPE_CHECKING block (indented block follows)
            if "TYPE_CHECKING" in stripped and "if" in stripped:
                in_type_checking_block = True
                continue
            # A non-indented non-empty line that isn't a comment exits the block
            if in_type_checking_block and stripped and not stripped.startswith(" ") and not stripped.startswith("\t"):
                in_type_checking_block = False

            # Flag bare 'import torch' at column 0 outside TYPE_CHECKING
            if stripped.startswith("import torch") and not in_type_checking_block:
                pytest.fail(
                    f"Found unconditional top-level 'import torch' in stgnn.py: {stripped!r}. "
                    "All torch imports must be lazy (inside functions/methods)."
                )
