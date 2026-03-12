"""
tests/model/test_info_vec_integration.py

Tests for the info_vec parameter wired into EntropyGatedGATEncoder.forward().

info_vec: optional (N, latent_dim) tensor of sentiment information vectors.
When provided, it is concatenated to x before the first GAT layer:
    x_augmented = concat([x, info_vec], dim=-1)

All torch imports are mocked so these run without torch installed.
Integration tests are auto-skipped when torch_geometric is absent.
"""

from __future__ import annotations

import sys
import inspect
from unittest.mock import MagicMock, patch
import pytest


# ---------------------------------------------------------------------------
# Helpers (reuse the same mock stack pattern from test_entropy_gated_gat.py)
# ---------------------------------------------------------------------------

def _build_mock_torch_stack():
    mock_torch = MagicMock(name="torch")
    mock_nn = MagicMock(name="torch.nn")
    mock_pyg = MagicMock(name="torch_geometric")
    mock_pyg_nn = MagicMock(name="torch_geometric.nn")

    mock_nn.Module = MagicMock
    mock_nn.BatchNorm1d = MagicMock
    mock_nn.ELU = MagicMock
    mock_nn.Dropout = MagicMock
    mock_nn.Linear = MagicMock
    mock_nn.Sigmoid = MagicMock

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
    patches, mock_torch, mock_nn, mock_gatconv = _build_mock_torch_stack()
    with patch.dict(sys.modules, patches):
        sys.modules.pop("src.model.gat_encoder", None)
        import src.model.gat_encoder as module
        yield module, mock_torch, mock_nn, mock_gatconv
    sys.modules.pop("src.model.gat_encoder", None)


# ===========================================================================
# 1. Signature tests
# ===========================================================================

class TestInfoVecSignature:
    """forward() must accept info_vec as an optional keyword argument."""

    def test_forward_accepts_info_vec_kwarg(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        sig = inspect.signature(enc.forward)
        assert "info_vec" in sig.parameters, (
            "EntropyGatedGATEncoder.forward() must accept 'info_vec' parameter"
        )

    def test_info_vec_defaults_to_none(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        sig = inspect.signature(enc.forward)
        param = sig.parameters.get("info_vec")
        assert param is not None
        assert param.default is None, (
            "info_vec parameter must default to None"
        )

    def test_docstring_mentions_info_vec(self, mock_modules):
        module, *_ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3
        )
        doc = (enc.forward.__doc__ or "") + (enc.__doc__ or "")
        assert "info_vec" in doc, (
            "info_vec must be documented in EntropyGatedGATEncoder forward() or class docstring"
        )


# ===========================================================================
# 2. Behaviour with mocks
# ===========================================================================

class TestInfoVecBehaviourMocked:
    """Verify concatenation is triggered correctly with mock torch."""

    def _build_enc_with_spies(self, mock_modules, num_layers=1):
        module, mock_torch, mock_nn, _ = mock_modules
        enc = module.EntropyGatedGATEncoder(
            node_in_features=10, edge_in_features=3, entropy_feat_dim=3,
            hidden_dim=64, num_heads=4, num_layers=num_layers,
        )
        gat_spies = [MagicMock(name=f"gat_{i}") for i in range(num_layers)]
        bn_spies  = [MagicMock(name=f"bn_{i}")  for i in range(num_layers)]
        gate_spies = [MagicMock(name=f"gate_{i}") for i in range(num_layers)]
        enc._gat_layers  = gat_spies
        enc._bn_layers   = bn_spies
        enc._gate_layers = gate_spies
        enc._act         = MagicMock(name="act")
        enc._dropout     = MagicMock(name="dropout")
        enc._sigmoid     = MagicMock(name="sigmoid")
        return enc, mock_torch, gat_spies, gate_spies

    def test_torch_cat_called_when_info_vec_provided(self, mock_modules):
        """When info_vec is not None, torch.cat must be called to concatenate."""
        enc, mock_torch, gat_spies, gate_spies = self._build_enc_with_spies(
            mock_modules, num_layers=1
        )
        fake_x        = MagicMock(name="x")
        fake_info_vec = MagicMock(name="info_vec")
        fake_concat   = MagicMock(name="concat_result")
        mock_torch.cat.return_value = fake_concat

        gat_spies[0].return_value  = fake_x
        gate_spies[0].return_value = fake_x
        enc._sigmoid.return_value  = fake_x
        enc._bn_layers[0].return_value = fake_x
        enc._act.return_value      = fake_x
        enc._dropout.return_value  = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(),
                        entropy_feats=None, info_vec=fake_info_vec)
        except Exception:
            pass

        mock_torch.cat.assert_called_once()

    def test_torch_cat_not_called_when_info_vec_is_none(self, mock_modules):
        """When info_vec=None, torch.cat must NOT be called."""
        enc, mock_torch, gat_spies, gate_spies = self._build_enc_with_spies(
            mock_modules, num_layers=1
        )
        fake_x = MagicMock(name="x")

        gat_spies[0].return_value  = fake_x
        gate_spies[0].return_value = fake_x
        enc._sigmoid.return_value  = fake_x
        enc._bn_layers[0].return_value = fake_x
        enc._act.return_value      = fake_x
        enc._dropout.return_value  = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(),
                        entropy_feats=None, info_vec=None)
        except Exception:
            pass

        mock_torch.cat.assert_not_called()

    def test_gat_receives_concatenated_x_when_info_vec_provided(self, mock_modules):
        """The first GAT layer must receive the concatenated x, not raw x."""
        enc, mock_torch, gat_spies, gate_spies = self._build_enc_with_spies(
            mock_modules, num_layers=1
        )
        fake_x        = MagicMock(name="x")
        fake_info_vec = MagicMock(name="info_vec")
        fake_concat   = MagicMock(name="concat_result")
        mock_torch.cat.return_value = fake_concat

        gat_spies[0].return_value  = fake_x
        gate_spies[0].return_value = fake_x
        enc._sigmoid.return_value  = fake_x
        enc._bn_layers[0].return_value = fake_x
        enc._act.return_value      = fake_x
        enc._dropout.return_value  = fake_x

        try:
            enc.forward(fake_x, MagicMock(), MagicMock(),
                        entropy_feats=None, info_vec=fake_info_vec)
        except Exception:
            pass

        # The first argument to gat_spies[0] should be the concat result
        call_args = gat_spies[0].call_args
        if call_args is not None:
            first_arg = call_args[0][0] if call_args[0] else call_args[1].get("x")
            assert first_arg is fake_concat, (
                "First GAT layer must receive the concatenated tensor when info_vec is given"
            )


# ===========================================================================
# 3. Integration tests — require real torch + torch_geometric
# ===========================================================================

class TestInfoVecIntegration:
    """Real-torch integration tests."""

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

    def test_forward_with_info_vec_no_crash(self, real_module):
        """forward() with info_vec != None must not crash."""
        import torch
        N, node_in, edge_in, E = 10, 12, 5, 20
        latent_dim = 8
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in + latent_dim,  # augmented input dim
            edge_in_features=edge_in,
            entropy_feat_dim=3,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )
        enc.train(False)
        x          = torch.randn(N, node_in)
        info_vec   = torch.rand(N, latent_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr  = torch.randn(E, edge_in)
        out = enc(x, edge_index, edge_attr, info_vec=info_vec)
        assert out.shape == (N, 64)

    def test_forward_with_info_vec_differs_from_without(self, real_module):
        """Output with info_vec must differ from output without it."""
        import torch
        N, node_in, edge_in, E = 10, 12, 5, 20
        latent_dim = 8

        # Build encoder whose node_in = node_in + latent_dim
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in + latent_dim,
            edge_in_features=edge_in,
            entropy_feat_dim=3,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )
        enc.train(False)
        torch.manual_seed(0)
        x          = torch.randn(N, node_in)
        info_vec   = torch.rand(N, latent_dim)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr  = torch.randn(E, edge_in)

        with torch.no_grad():
            out_with    = enc(x, edge_index, edge_attr, info_vec=info_vec)
            out_without = enc(
                torch.cat([x, torch.zeros(N, latent_dim)], dim=-1),
                edge_index, edge_attr,
            )
        # Should differ because info_vec is non-zero
        assert not torch.allclose(out_with, out_without), (
            "Output with info_vec must differ from output with zero augmentation"
        )

    def test_forward_output_shape_with_info_vec(self, real_module):
        """Output shape must be (N, hidden_dim) regardless of info_vec."""
        import torch
        N, node_in, edge_in, E = 8, 10, 4, 15
        latent_dim = 8
        enc = real_module.EntropyGatedGATEncoder(
            node_in_features=node_in + latent_dim,
            edge_in_features=edge_in,
            entropy_feat_dim=3,
            hidden_dim=64,
            num_heads=4,
        )
        enc.train(False)
        x = torch.randn(N, node_in)
        out = enc(
            x,
            torch.randint(0, N, (2, E)),
            torch.randn(E, edge_in),
            info_vec=torch.rand(N, latent_dim),
        )
        assert out.shape == (N, 64)
