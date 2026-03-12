"""
tests/model/test_entropy_gated_gat.py

Test suite for EntropyGatedGATEncoder -- the Shannon-entropy-gated
extension of GATEncoder.

Architecture under test
-----------------------
After each GATConv layer aggregates neighbour messages, a learned sigmoid
gate conditioned on per-team entropy features modulates the output:

    gate_i  = sigmoid(Linear(entropy_feat_dim -> hidden_dim)(entropy_feats))
    h       = GATConv(h, edge_index, edge_attr) * gate_i
    h       = BatchNorm(h) -> ELU -> Dropout

Three entropy features per team (from shannon_entropy.py):
    0: scoring_entropy_normalized      in [0, 1]
    1: kill_shot_vulnerability         in [0, 1]
    2: kill_shot_p_run_given_trading   in [0, 1]

All torch / torch_geometric imports are mocked so the suite runs without
those packages installed.  Integration tests (real torch) are auto-skipped
when torch_geometric is absent.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import inspect
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mock_torch_stack():
    """Build minimal torch + torch_geometric mocks for unit tests."""
    mock_torch = MagicMock(name="torch")
    mock_nn = MagicMock(name="torch.nn")
    mock_pyg = MagicMock(name="torch_geometric")
    mock_pyg_nn = MagicMock(name="torch_geometric.nn")

    mock_nn.Module = MagicMock
    mock_nn.BatchNorm1d = MagicMock
    mock_nn.ELU = MagicMock
    mock_nn.Dropout = MagicMock
    mock_nn.Linear = MagicMock

    mock_pyg_nn.GATConv = MagicMock

    patches = {
        "torch": mock_torch,
        "torch.nn": mock_nn,
        "torch_geometric": mock_pyg,
        "torch_geometric.nn": mock_pyg_nn,
    }
    return patches, mock_torch, mock_nn, mock_pyg_nn.GATConv


@pytest.fixture()
def mock_modules():
    """Patch sys.modules and reload gat_encoder under mock torch."""
    patches, mock_torch, mock_nn, mock_gatconv = _build_mock_torch_stack()
    with patch.dict(sys.modules, patches):
        sys.modules.pop("src.model.gat_encoder", None)
        import src.model.gat_encoder as module
        yield module, mock_torch, mock_nn, mock_gatconv
    sys.modules.pop("src.model.gat_encoder", None)


# ===========================================================================
# 1. TestEntropyGatedGATEncoderExists
# ===========================================================================

class TestEntropyGatedGATEncoderExists:
    """The module must expose EntropyGatedGATEncoder and its factory."""

    def test_class_is_importable(self, mock_modules):
        module, *_ = mock_modules
        assert hasattr(module, "EntropyGatedGATEncoder"), (
            "src.model.gat_encoder must define EntropyGatedGATEncoder"
        )

    def test_factory_is_importable(self, mock_modules):
        module, *_ = mock_modules
        assert hasattr(module, "make_entropy_gated_gat_encoder"), (
            "src.model.gat_encoder must define make_entropy_gated_gat_encoder"
        )

    def test_class_is_callable(self, mock_modules):
        module, *_ = mock_modules
        assert callable(module.EntropyGatedGATEncoder)


# ===========================================================================
# 2. TestEntropyGatedGATEncoderConstruction
# ===========================================================================

class TestEntropyGatedGATEncoderConstruction:
    """Constructor stores all hyperparameters correctly."""

    def _make(self, mock_modules, **kwargs):
        module, *_ = mock_modules
        defaults = dict(node_in_features=10, edge_in_features=3, entropy_feat_dim=3)
        defaults.update(kwargs)
        return module.EntropyGatedGATEncoder(**defaults)

    def test_stores_entropy_feat_dim(self, mock_modules):
        enc = self._make(mock_modules, entropy_feat_dim=3)
        assert enc.entropy_feat_dim == 3

    def test_stores_node_in_features(self, mock_modules):
        enc = self._make(mock_modules, node_in_features=15)
        assert enc.node_in_features == 15

    def test_stores_hidden_dim(self, mock_modules):
        enc = self._make(mock_modules, hidden_dim=64)
        assert enc.hidden_dim == 64

    def test_stores_num_layers(self, mock_modules):
        enc = self._make(mock_modules, num_layers=3)
        assert enc.num_layers == 3

    def test_raises_on_indivisible_hidden_dim(self, mock_modules):
        module, *_ = mock_modules
        with pytest.raises(ValueError, match="hidden_dim"):
            module.EntropyGatedGATEncoder(
                node_in_features=10,
                edge_in_features=3,
                entropy_feat_dim=3,
                hidden_dim=65,
                num_heads=4,
            )

    def test_gate_layers_created_for_each_gat_layer(self, mock_modules):
        """One Linear gate projection must exist per GAT layer."""
        enc = self._make(mock_modules, num_layers=2)
        assert hasattr(enc, "_gate_layers"), (
            "EntropyGatedGATEncoder must have _gate_layers attribute"
        )
        assert len(enc._gate_layers) == 2

    def test_gate_layers_count_matches_num_layers(self, mock_modules):
        enc = self._make(mock_modules, num_layers=3)
        assert len(enc._gate_layers) == 3

    def test_default_entropy_feat_dim_is_three(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3,
        )
        assert enc.entropy_feat_dim == 3


# ===========================================================================
# 3. TestFactoryFunction
# ===========================================================================

class TestFactoryFunction:
    """make_entropy_gated_gat_encoder constructs from a config dict."""

    def test_returns_entropy_gated_instance(self, mock_modules):
        module, *_ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3, "entropy_feat_dim": 3}
        enc = module.make_entropy_gated_gat_encoder(cfg)
        assert isinstance(enc, module.EntropyGatedGATEncoder)

    def test_passes_through_optional_params(self, mock_modules):
        module, *_ = mock_modules
        cfg = {
            "node_in_features": 12,
            "edge_in_features": 5,
            "entropy_feat_dim": 3,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.2,
        }
        enc = module.make_entropy_gated_gat_encoder(cfg)
        assert enc.hidden_dim == 32
        assert enc.num_layers == 3

    def test_raises_on_missing_required_keys(self, mock_modules):
        module, *_ = mock_modules
        with pytest.raises(KeyError):
            module.make_entropy_gated_gat_encoder({})


# ===========================================================================
# 4. TestForwardSignature
# ===========================================================================

class TestForwardSignature:
    """forward() must accept entropy_feats as a keyword argument."""

    def test_forward_method_exists(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        assert hasattr(enc, "forward") and callable(enc.forward)

    def test_forward_accepts_entropy_feats_kwarg(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        sig = inspect.signature(enc.forward)
        assert "entropy_feats" in sig.parameters, (
            "forward() must have an entropy_feats parameter"
        )

    def test_forward_entropy_feats_defaults_to_none(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        sig = inspect.signature(enc.forward)
        param = sig.parameters.get("entropy_feats")
        assert param is not None
        assert param.default is None


# ===========================================================================
# 5. TestGatingMechanism
# ===========================================================================

class TestGatingMechanism:
    """Gate layers are called with entropy_feats and applied to h."""

    def _build_encoder_with_spy_layers(self, mock_modules, num_layers=2):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10,
            edge_in_features=3,
            entropy_feat_dim=3,
            hidden_dim=64,
            num_heads=4,
            num_layers=num_layers,
        )
        gat_spies = [MagicMock(name=f"gat_{i}") for i in range(num_layers)]
        bn_spies = [MagicMock(name=f"bn_{i}") for i in range(num_layers)]
        gate_spies = [MagicMock(name=f"gate_{i}") for i in range(num_layers)]
        enc._gat_layers = gat_spies
        enc._bn_layers = bn_spies
        enc._gate_layers = gate_spies
        enc._act = MagicMock(name="act")
        enc._dropout = MagicMock(name="dropout")
        enc._sigmoid = MagicMock(name="sigmoid")
        return enc, gat_spies, gate_spies

    def test_gate_layers_called_once_per_layer(self, mock_modules):
        enc, gat_spies, gate_spies = self._build_encoder_with_spy_layers(
            mock_modules, num_layers=2
        )
        fake_x = MagicMock(name="x")
        fake_ef = MagicMock(name="entropy_feats")
        for spy in gat_spies:
            spy.return_value = fake_x
        for spy in gate_spies:
            spy.return_value = fake_x
        enc._sigmoid.return_value = fake_x
        for bn in enc._bn_layers:
            bn.return_value = fake_x
        enc._act.return_value = fake_x
        enc._dropout.return_value = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(), entropy_feats=fake_ef)
        except Exception:
            pass

        for i, spy in enumerate(gate_spies):
            assert spy.call_count == 1, (
                f"gate_layers[{i}] called {spy.call_count} times, expected 1"
            )

    def test_gate_receives_entropy_feats_not_x(self, mock_modules):
        enc, gat_spies, gate_spies = self._build_encoder_with_spy_layers(
            mock_modules, num_layers=1
        )
        fake_x = MagicMock(name="x")
        fake_ef = MagicMock(name="entropy_feats")
        gat_spies[0].return_value = fake_x
        gate_spies[0].return_value = fake_x
        enc._sigmoid.return_value = fake_x
        enc._bn_layers[0].return_value = fake_x
        enc._act.return_value = fake_x
        enc._dropout.return_value = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(), entropy_feats=fake_ef)
        except Exception:
            pass

        args, _ = gate_spies[0].call_args
        assert args[0] is fake_ef, (
            "gate_layer must receive entropy_feats as its first argument"
        )

    def test_gat_called_same_number_of_times_as_layers(self, mock_modules):
        enc, gat_spies, gate_spies = self._build_encoder_with_spy_layers(
            mock_modules, num_layers=3
        )
        fake_x = MagicMock(name="x")
        fake_ef = MagicMock(name="entropy_feats")
        for spy in gat_spies + gate_spies:
            spy.return_value = fake_x
        enc._sigmoid.return_value = fake_x
        for bn in enc._bn_layers:
            bn.return_value = fake_x
        enc._act.return_value = fake_x
        enc._dropout.return_value = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(), entropy_feats=fake_ef)
        except Exception:
            pass

        assert sum(s.call_count for s in gat_spies) == 3

    def test_gate_layers_not_called_when_entropy_feats_is_none(self, mock_modules):
        """When entropy_feats=None, gate layers must not be called (identity gate)."""
        enc, gat_spies, gate_spies = self._build_encoder_with_spy_layers(
            mock_modules, num_layers=2
        )
        fake_x = MagicMock(name="x")
        for spy in gat_spies:
            spy.return_value = fake_x
        enc._sigmoid.return_value = fake_x
        for bn in enc._bn_layers:
            bn.return_value = fake_x
        enc._act.return_value = fake_x
        enc._dropout.return_value = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(), entropy_feats=None)
        except Exception:
            pass

        for i, spy in enumerate(gate_spies):
            assert spy.call_count == 0, (
                f"gate_layers[{i}] called when entropy_feats=None; "
                "should be identity (no-op)"
            )


# ===========================================================================
# 6. TestParameterCoverage
# ===========================================================================

class TestParameterCoverage:
    """Gate layer parameters must be included in parameters()."""

    def test_gate_layers_in_parameters_iterator(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3, num_layers=1
        )
        sentinel = MagicMock(name="gate_param")
        sentinel.numel.return_value = 999
        enc._gate_layers[0].parameters = MagicMock(return_value=iter([sentinel]))
        for layer in enc._gat_layers + enc._bn_layers:
            layer.parameters = MagicMock(return_value=iter([]))
        enc._act.parameters = MagicMock(return_value=iter([]))
        enc._dropout.parameters = MagicMock(return_value=iter([]))

        all_params = list(enc.parameters())
        assert sentinel in all_params, (
            "Gate layer parameters must appear in EntropyGatedGATEncoder.parameters()"
        )


# ===========================================================================
# 7. TestEntropyNodeFeatureCols (node_features.py integration)
# ===========================================================================

class TestEntropyNodeFeatureCols:
    """node_features.py must expose entropy-related constants and accept entropy_df."""

    def test_entropy_feature_cols_constant_exists(self):
        from src.graph import node_features as nf
        assert hasattr(nf, "_ENTROPY_FEATURE_COLS")

    def test_entropy_feature_cols_has_three_entries(self):
        from src.graph import node_features as nf
        assert len(nf._ENTROPY_FEATURE_COLS) == 3

    def test_entropy_feature_cols_correct_names(self):
        from src.graph import node_features as nf
        expected = {
            "scoring_entropy_normalized",
            "kill_shot_vulnerability",
            "kill_shot_p_run_given_trading",
        }
        assert set(nf._ENTROPY_FEATURE_COLS) == expected

    def test_build_accepts_entropy_df_param(self):
        from src.graph.node_features import NodeFeatureBuilder
        sig = inspect.signature(NodeFeatureBuilder.build)
        assert "entropy_df" in sig.parameters

    def test_build_with_entropy_df_adds_entropy_cols(self):
        import pandas as pd
        from src.graph.node_features import NodeFeatureBuilder, _ENTROPY_FEATURE_COLS

        builder = NodeFeatureBuilder(season=2024)
        eff_df = pd.DataFrame({
            "team_id": ["Duke", "Kansas"],
            "adj_em": [20.0, 18.0], "adj_o": [115.0, 112.0],
            "adj_d": [95.0, 94.0], "adj_t": [70.0, 68.0], "luck": [0.02, -0.01],
        })
        bpr_df = pd.DataFrame({"team_id": ["Duke", "Kansas"], "team_bpr_weighted": [5.0, 4.5]})
        shot_df = pd.DataFrame({
            "team_id": ["Duke", "Kansas"],
            "rim_pct": [0.4, 0.38], "three_pct": [0.35, 0.33],
            "transition_pct": [0.15, 0.18], "efg": [0.52, 0.50],
        })
        entropy_df = pd.DataFrame({
            "team_id": ["Duke", "Kansas"],
            "scoring_entropy_normalized": [0.92, 0.87],
            "kill_shot_vulnerability": [0.12, 0.18],
            "kill_shot_p_run_given_trading": [0.10, 0.15],
        })

        result = builder.build(
            efficiency_df=eff_df, bpr_df=bpr_df, shot_df=shot_df,
            roster_continuity=0.7, availability_vector=1.0,
            entropy_df=entropy_df,
        )

        for col in _ENTROPY_FEATURE_COLS:
            assert col in result.columns, f"Output missing entropy column: {col}"

    def test_build_without_entropy_df_uses_defaults(self):
        import pandas as pd
        from src.graph.node_features import NodeFeatureBuilder, _ENTROPY_FEATURE_COLS

        builder = NodeFeatureBuilder(season=2024)
        eff_df = pd.DataFrame({
            "team_id": ["Duke"], "adj_em": [20.0], "adj_o": [115.0],
            "adj_d": [95.0], "adj_t": [70.0], "luck": [0.02],
        })
        bpr_df = pd.DataFrame({"team_id": ["Duke"], "team_bpr_weighted": [5.0]})
        shot_df = pd.DataFrame({
            "team_id": ["Duke"], "rim_pct": [0.4], "three_pct": [0.35],
            "transition_pct": [0.15], "efg": [0.52],
        })

        result = builder.build(
            efficiency_df=eff_df, bpr_df=bpr_df, shot_df=shot_df,
            roster_continuity=0.7, availability_vector=1.0,
        )

        for col in _ENTROPY_FEATURE_COLS:
            assert col in result.columns, (
                f"Column {col!r} must exist even without entropy_df (use defaults)"
            )
        assert abs(result["scoring_entropy_normalized"].iloc[0] - 0.85) < 1e-6

    def test_entropy_values_from_df_override_defaults(self):
        import pandas as pd
        from src.graph.node_features import NodeFeatureBuilder

        builder = NodeFeatureBuilder(season=2024)
        eff_df = pd.DataFrame({
            "team_id": ["Duke"], "adj_em": [20.0], "adj_o": [115.0],
            "adj_d": [95.0], "adj_t": [70.0], "luck": [0.02],
        })
        bpr_df = pd.DataFrame({"team_id": ["Duke"], "team_bpr_weighted": [5.0]})
        shot_df = pd.DataFrame({
            "team_id": ["Duke"], "rim_pct": [0.4], "three_pct": [0.35],
            "transition_pct": [0.15], "efg": [0.52],
        })
        entropy_df = pd.DataFrame({
            "team_id": ["Duke"],
            "scoring_entropy_normalized": [0.55],
            "kill_shot_vulnerability": [0.30],
            "kill_shot_p_run_given_trading": [0.20],
        })

        result = builder.build(
            efficiency_df=eff_df, bpr_df=bpr_df, shot_df=shot_df,
            roster_continuity=0.7, availability_vector=1.0,
            entropy_df=entropy_df,
        )

        assert abs(result["scoring_entropy_normalized"].iloc[0] - 0.55) < 1e-6


# ===========================================================================
# 8. TestConfigEntropyFeatDim
# ===========================================================================

class TestConfigEntropyFeatDim:
    """PipelineConfig must expose entropy_feat_dim and use_entropy_gating."""

    def test_config_has_entropy_feat_dim(self):
        from src.pipeline.config import PipelineConfig
        assert hasattr(PipelineConfig(), "entropy_feat_dim")

    def test_default_entropy_feat_dim_is_three(self):
        from src.pipeline.config import PipelineConfig
        assert PipelineConfig().entropy_feat_dim == 3

    def test_config_has_use_entropy_gating(self):
        from src.pipeline.config import PipelineConfig
        assert hasattr(PipelineConfig(), "use_entropy_gating")

    def test_use_entropy_gating_default_true(self):
        from src.pipeline.config import PipelineConfig
        assert PipelineConfig().use_entropy_gating is True


# ===========================================================================
# 9. Integration tests -- require real torch + torch_geometric
# ===========================================================================

class TestEntropyGatedGATEncoderOutputShape:
    """Real-torch integration tests; auto-skipped without torch_geometric."""

    @pytest.fixture(autouse=True)
    def require_torch(self):
        pytest.importorskip("torch")

    @pytest.fixture(autouse=True)
    def require_pyg(self, require_torch):
        pytest.importorskip("torch_geometric")

    @pytest.fixture()
    def real_module(self):
        sys.modules.pop("src.model.gat_encoder", None)
        import src.model.gat_encoder as mod
        return mod

    def test_output_shape_with_entropy_feats(self, real_module):
        import torch
        N, node_in, edge_in, E = 15, 12, 5, 25
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in, edge_in_features=edge_in,
            entropy_feat_dim=3, hidden_dim=64, num_heads=4, num_layers=2,
        )
        enc.eval()
        out = enc(
            torch.randn(N, node_in),
            torch.randint(0, N, (2, E)),
            torch.randn(E, edge_in),
            entropy_feats=torch.rand(N, 3),
        )
        assert out.shape == (N, 64)

    def test_output_shape_without_entropy_feats(self, real_module):
        import torch
        N, node_in, edge_in, E = 10, 12, 5, 18
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in, edge_in_features=edge_in,
        )
        enc.eval()
        out = enc(
            torch.randn(N, node_in),
            torch.randint(0, N, (2, E)),
            torch.randn(E, edge_in),
        )
        assert out.shape == (N, 64)

    def test_output_dtype_float32(self, real_module):
        import torch
        N, node_in, edge_in, E = 8, 12, 5, 10
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in, edge_in_features=edge_in,
        )
        enc.eval()
        out = enc(
            torch.randn(N, node_in),
            torch.randint(0, N, (2, E)),
            torch.randn(E, edge_in),
            entropy_feats=torch.rand(N, 3),
        )
        assert out.dtype == torch.float32

    def test_gate_output_is_sigmoid(self, real_module):
        """Gate layer output through sigmoid must be strictly in (0, 1)."""
        import torch
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=12, edge_in_features=5,
        )
        enc.eval()
        entropy_feats = torch.rand(4, 3)
        with torch.no_grad():
            gate = torch.sigmoid(enc._gate_layers[0](entropy_feats))
        assert gate.min().item() > 0.0
        assert gate.max().item() < 1.0

    def test_gated_has_more_parameters_than_plain(self, real_module):
        plain = real_module.GATEncoder(node_in_features=12, edge_in_features=5)
        gated = real_module.EntropyGatedGATEncoder(
            node_in_features=12, edge_in_features=5, entropy_feat_dim=3,
        )
        assert real_module.count_parameters(gated) > real_module.count_parameters(plain)
