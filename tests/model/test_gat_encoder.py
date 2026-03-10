"""
tests/model/test_gat_encoder.py

RED phase — tests written before implementation exists.
Covers: GATEncoder class, make_gat_encoder(), count_parameters(), and lazy
import contract.

All torch / torch_geometric imports are mocked via patch.dict(sys.modules)
so the suite runs without installing those packages.  torch is available in
this environment, so a small number of integration-flavoured tests use it
directly for tensor shape assertions; those are clearly labelled.
"""

from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake torch + torch_geometric module graph
# ---------------------------------------------------------------------------

def _make_mock_nn_module_class():
    """Return a minimal mock of torch.nn.Module that records __init__ calls."""
    class FakeModule:
        def __init__(self):
            self._parameters: dict = {}
            self._modules: dict = {}
            self.training = True

        def parameters(self, recurse=True):
            return iter([])

        def named_parameters(self, prefix="", recurse=True):
            return iter([])

        def train(self, mode=True):
            self.training = mode
            return self

        def eval(self):
            return self.train(False)

    return FakeModule


def _build_mock_torch_stack():
    """
    Build a fake sys.modules entry for torch, torch.nn, and
    torch_geometric.nn so that importing src.model.gat_encoder works
    without real packages installed.

    Returns a dict suitable for patch.dict(sys.modules, ...).
    """
    # ---- torch ---------------------------------------------------------
    mock_torch = MagicMock(name="torch")
    mock_torch.Tensor = MagicMock(name="torch.Tensor")
    mock_torch.float32 = "float32"

    # ---- torch.nn ------------------------------------------------------
    mock_nn = MagicMock(name="torch.nn")
    FakeModule = _make_mock_nn_module_class()
    mock_nn.Module = FakeModule

    # Concrete layer mocks — each call returns a new MagicMock instance
    mock_nn.BatchNorm1d = MagicMock(
        name="BatchNorm1d",
        side_effect=lambda *a, **kw: MagicMock(name=f"BN({a})"),
    )
    mock_nn.ELU = MagicMock(
        name="ELU",
        side_effect=lambda *a, **kw: MagicMock(name="ELU()"),
    )
    mock_nn.Dropout = MagicMock(
        name="Dropout",
        side_effect=lambda *a, **kw: MagicMock(name=f"Dropout({a})"),
    )
    mock_nn.ModuleList = MagicMock(
        name="ModuleList",
        side_effect=lambda items=None: MagicMock(
            name="ModuleList", _inner=list(items or [])
        ),
    )
    mock_torch.nn = mock_nn

    # ---- torch_geometric.nn --------------------------------------------
    mock_gatconv = MagicMock(
        name="GATConv",
        side_effect=lambda *a, **kw: MagicMock(name=f"GATConv({a},{kw})"),
    )
    mock_pyg_nn = MagicMock(name="torch_geometric.nn")
    mock_pyg_nn.GATConv = mock_gatconv

    mock_pyg = MagicMock(name="torch_geometric")
    mock_pyg.nn = mock_pyg_nn

    modules_patch = {
        "torch": mock_torch,
        "torch.nn": mock_nn,
        "torch_geometric": mock_pyg,
        "torch_geometric.nn": mock_pyg_nn,
    }
    return modules_patch, mock_torch, mock_nn, mock_gatconv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_modules():
    """Patch sys.modules and reload src.model.gat_encoder under the mock."""
    patches, mock_torch, mock_nn, mock_gatconv = _build_mock_torch_stack()
    with patch.dict(sys.modules, patches):
        # Remove any cached real module so the reload picks up the mocks
        sys.modules.pop("src.model.gat_encoder", None)
        import src.model.gat_encoder as module
        yield module, mock_torch, mock_nn, mock_gatconv
    # Ensure the module is evicted after each test so mocks don't leak
    sys.modules.pop("src.model.gat_encoder", None)


@pytest.fixture()
def default_encoder(mock_modules):
    """GATEncoder with default hyperparams built under mocked torch."""
    module, _, _, _ = mock_modules
    return module.GATEncoder(
        node_in_features=10,
        edge_in_features=3,
    )


@pytest.fixture()
def custom_encoder(mock_modules):
    """GATEncoder with explicit non-default hyperparams."""
    module, _, _, _ = mock_modules
    return module.GATEncoder(
        node_in_features=8,
        edge_in_features=4,
        hidden_dim=128,
        num_heads=8,
        num_layers=3,
        dropout=0.2,
    )


# ===========================================================================
# 1. TestLazyImports — module must import cleanly without torch installed
# ===========================================================================

class TestLazyImports:
    """Verify that gat_encoder.py does not perform eager torch imports."""

    def test_module_imports_without_torch_in_sys_modules(self):
        """Removing torch from sys.modules should not break the module import."""
        # Evict any cached module
        sys.modules.pop("src.model.gat_encoder", None)
        saved_torch = sys.modules.pop("torch", None)
        saved_torch_nn = sys.modules.pop("torch.nn", None)
        saved_pyg = sys.modules.pop("torch_geometric", None)
        saved_pyg_nn = sys.modules.pop("torch_geometric.nn", None)

        # Replace with sentinel None to force ImportError on eager import
        blocked = {
            "torch": None,
            "torch.nn": None,
            "torch_geometric": None,
            "torch_geometric.nn": None,
        }
        try:
            with patch.dict(sys.modules, blocked):
                # If the module has ANY top-level torch import this will raise
                import src.model.gat_encoder  # noqa: F401
        except (ImportError, AttributeError):
            pytest.fail(
                "src.model.gat_encoder has eager torch/pyg imports at module level; "
                "all torch imports must be lazy (inside functions/methods)."
            )
        finally:
            sys.modules.pop("src.model.gat_encoder", None)
            for name, mod in [
                ("torch", saved_torch),
                ("torch.nn", saved_torch_nn),
                ("torch_geometric", saved_pyg),
                ("torch_geometric.nn", saved_pyg_nn),
            ]:
                if mod is not None:
                    sys.modules[name] = mod

    def test_source_file_has_no_top_level_torch_import(self):
        """Read the source file and confirm no bare 'import torch' at module scope."""
        import ast
        import pathlib

        src_path = pathlib.Path(__file__).parent.parent.parent / "src" / "model" / "gat_encoder.py"
        assert src_path.exists(), "gat_encoder.py must exist"
        tree = ast.parse(src_path.read_text())

        for node in ast.walk(tree):
            # Only check top-level (module-body) statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert not alias.name.startswith("torch"), (
                            f"Top-level 'import {alias.name}' found; must be lazy."
                        )
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("torch"):
                        # Allowed only inside TYPE_CHECKING guard
                        # We can't easily check the guard here, so we rely on
                        # test_module_imports_without_torch_in_sys_modules
                        pass


# ===========================================================================
# 2. TestGATEncoderInit — constructor stores hyperparameters
# ===========================================================================

class TestGATEncoderInit:
    """GATEncoder.__init__() should store all hyperparameters as attributes."""

    def test_stores_node_in_features(self, default_encoder):
        assert default_encoder.node_in_features == 10

    def test_stores_edge_in_features(self, default_encoder):
        assert default_encoder.edge_in_features == 3

    def test_stores_hidden_dim_default(self, default_encoder):
        assert default_encoder.hidden_dim == 64

    def test_stores_num_heads_default(self, default_encoder):
        assert default_encoder.num_heads == 4

    def test_stores_num_layers_default(self, default_encoder):
        assert default_encoder.num_layers == 2

    def test_stores_dropout_default(self, default_encoder):
        assert default_encoder.dropout == 0.1

    def test_stores_custom_hidden_dim(self, custom_encoder):
        assert custom_encoder.hidden_dim == 128

    def test_stores_custom_num_heads(self, custom_encoder):
        assert custom_encoder.num_heads == 8

    def test_stores_custom_num_layers(self, custom_encoder):
        assert custom_encoder.num_layers == 3

    def test_stores_custom_dropout(self, custom_encoder):
        assert custom_encoder.dropout == 0.2

    def test_correct_number_of_gat_layers_created(self, mock_modules):
        """num_layers GAT layer objects should be instantiated."""
        module, _, _, mock_gatconv = mock_modules
        mock_gatconv.reset_mock()
        enc = module.GATEncoder(
            node_in_features=6,
            edge_in_features=2,
            hidden_dim=32,
            num_heads=4,
            num_layers=3,
        )
        assert mock_gatconv.call_count == 3

    def test_correct_number_of_batch_norm_layers(self, mock_modules):
        """One BatchNorm1d per GAT layer."""
        module, _, mock_nn, _ = mock_modules
        mock_nn.BatchNorm1d.reset_mock()
        enc = module.GATEncoder(
            node_in_features=6,
            edge_in_features=2,
            hidden_dim=32,
            num_heads=4,
            num_layers=3,
        )
        assert mock_nn.BatchNorm1d.call_count == 3


# ===========================================================================
# 3. TestMakeGATEncoder — factory function validation
# ===========================================================================

class TestMakeGATEncoder:
    """Tests for the make_gat_encoder() factory function."""

    def test_returns_gat_encoder_instance(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = module.make_gat_encoder(cfg)
        assert isinstance(enc, module.GATEncoder)

    def test_factory_passes_node_in_features(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 12, "edge_in_features": 5}
        enc = module.make_gat_encoder(cfg)
        assert enc.node_in_features == 12

    def test_factory_passes_edge_in_features(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 12, "edge_in_features": 5}
        enc = module.make_gat_encoder(cfg)
        assert enc.edge_in_features == 5

    def test_factory_uses_default_hidden_dim(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = module.make_gat_encoder(cfg)
        assert enc.hidden_dim == 64

    def test_factory_uses_default_num_heads(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = module.make_gat_encoder(cfg)
        assert enc.num_heads == 4

    def test_factory_uses_default_num_layers(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = module.make_gat_encoder(cfg)
        assert enc.num_layers == 2

    def test_factory_uses_default_dropout(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = module.make_gat_encoder(cfg)
        assert enc.dropout == 0.1

    def test_factory_overrides_hidden_dim(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3, "hidden_dim": 128}
        enc = module.make_gat_encoder(cfg)
        assert enc.hidden_dim == 128

    def test_factory_overrides_num_heads(self, mock_modules):
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3, "num_heads": 8,
               "hidden_dim": 128}
        enc = module.make_gat_encoder(cfg)
        assert enc.num_heads == 8

    def test_factory_raises_value_error_hidden_not_divisible_by_heads(self, mock_modules):
        """hidden_dim=65 is not divisible by num_heads=4 → ValueError."""
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3,
               "hidden_dim": 65, "num_heads": 4}
        with pytest.raises(ValueError, match="hidden_dim"):
            module.make_gat_encoder(cfg)

    def test_factory_raises_value_error_heads_not_divisor(self, mock_modules):
        """hidden_dim=64, num_heads=3 (64 % 3 != 0) → ValueError."""
        module, _, _, _ = mock_modules
        cfg = {"node_in_features": 10, "edge_in_features": 3,
               "hidden_dim": 64, "num_heads": 3}
        with pytest.raises(ValueError):
            module.make_gat_encoder(cfg)

    def test_constructor_directly_raises_value_error_on_indivisible(self, mock_modules):
        """ValueError should also be raised by GATEncoder() directly."""
        module, _, _, _ = mock_modules
        with pytest.raises(ValueError, match="hidden_dim"):
            module.GATEncoder(
                node_in_features=10,
                edge_in_features=3,
                hidden_dim=65,
                num_heads=4,
            )


# ===========================================================================
# 4. TestCountParameters — trainable parameter counting
# ===========================================================================

class TestCountParameters:
    """Tests for count_parameters(model) utility."""

    def test_count_parameters_returns_int(self, mock_modules):
        module, _, _, _ = mock_modules
        mock_model = MagicMock()
        # Simulate two parameter tensors: 100 and 50 elements
        p1 = MagicMock()
        p1.numel.return_value = 100
        p1.requires_grad = True
        p2 = MagicMock()
        p2.numel.return_value = 50
        p2.requires_grad = True
        mock_model.parameters.return_value = iter([p1, p2])
        result = module.count_parameters(mock_model)
        assert isinstance(result, int)

    def test_count_parameters_sums_numel(self, mock_modules):
        module, _, _, _ = mock_modules
        mock_model = MagicMock()
        p1 = MagicMock(); p1.numel.return_value = 100; p1.requires_grad = True
        p2 = MagicMock(); p2.numel.return_value = 50;  p2.requires_grad = True
        mock_model.parameters.return_value = iter([p1, p2])
        assert module.count_parameters(mock_model) == 150

    def test_count_parameters_zero_for_empty_model(self, mock_modules):
        module, _, _, _ = mock_modules
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([])
        assert module.count_parameters(mock_model) == 0

    def test_count_parameters_single_param(self, mock_modules):
        module, _, _, _ = mock_modules
        mock_model = MagicMock()
        p = MagicMock(); p.numel.return_value = 256; p.requires_grad = True
        mock_model.parameters.return_value = iter([p])
        assert module.count_parameters(mock_model) == 256


# ===========================================================================
# 5. TestGATEncoderForward — forward() call contract (mock-based)
# ===========================================================================

class TestGATEncoderForward:
    """
    Tests for GATEncoder.forward() using mocked torch layers.

    Because the layer objects themselves are MagicMocks, we verify:
      - each GAT layer is called with (x, edge_index, edge_attr=edge_attr)
      - BatchNorm is applied after each GAT layer
      - the output tensor returned by the final step is propagated correctly
    """

    def _build_encoder_with_spy_layers(self, mock_modules, num_layers=2):
        """
        Build an encoder whose GAT layers and BN layers are individually
        accessible for call-count / argument inspection.
        """
        module, mock_torch, mock_nn, mock_gatconv = mock_modules

        # Capture created layers in order
        created_gat_layers = []
        created_bn_layers = []

        def make_gat(*args, **kwargs):
            layer = MagicMock(name=f"GATConv_{len(created_gat_layers)}")
            created_gat_layers.append(layer)
            return layer

        def make_bn(dim):
            bn = MagicMock(name=f"BN_{len(created_bn_layers)}")
            created_bn_layers.append(bn)
            return bn

        mock_gatconv.side_effect = make_gat
        mock_nn.BatchNorm1d.side_effect = make_bn

        enc = module.GATEncoder(
            node_in_features=10,
            edge_in_features=3,
            hidden_dim=64,
            num_heads=4,
            num_layers=num_layers,
        )
        return enc, created_gat_layers, created_bn_layers

    def test_forward_calls_first_gat_layer(self, mock_modules):
        enc, gat_layers, _ = self._build_encoder_with_spy_layers(mock_modules)

        fake_x = MagicMock(name="x")
        fake_edge_index = MagicMock(name="edge_index")
        fake_edge_attr = MagicMock(name="edge_attr")

        # Chain mock outputs so forward can run without error
        for layer in gat_layers:
            layer.return_value = fake_x
        for mod_name in ["_bn_layers", "_act", "_dropout"]:
            pass  # handled by MagicMock passthrough

        try:
            enc.forward(fake_x, fake_edge_index, fake_edge_attr)
        except Exception:
            pass  # We only care that the first GAT layer was called

        assert gat_layers[0].call_count >= 1

    def test_forward_passes_edge_attr_to_gat_layers(self, mock_modules):
        enc, gat_layers, _ = self._build_encoder_with_spy_layers(mock_modules)

        fake_x = MagicMock(name="x")
        fake_edge_index = MagicMock(name="edge_index")
        fake_edge_attr = MagicMock(name="edge_attr")

        for layer in gat_layers:
            layer.return_value = fake_x

        try:
            enc.forward(fake_x, fake_edge_index, fake_edge_attr)
        except Exception:
            pass

        # The first GAT layer must have received edge_attr as a kwarg
        _, kwargs = gat_layers[0].call_args
        assert kwargs.get("edge_attr") is fake_edge_attr

    def test_forward_passes_edge_index_to_gat_layers(self, mock_modules):
        enc, gat_layers, _ = self._build_encoder_with_spy_layers(mock_modules)

        fake_x = MagicMock(name="x")
        fake_edge_index = MagicMock(name="edge_index")
        fake_edge_attr = MagicMock(name="edge_attr")

        for layer in gat_layers:
            layer.return_value = fake_x

        try:
            enc.forward(fake_x, fake_edge_index, fake_edge_attr)
        except Exception:
            pass

        args, _ = gat_layers[0].call_args
        # edge_index should be the second positional argument
        assert args[1] is fake_edge_index

    def test_number_of_gat_layer_calls_equals_num_layers(self, mock_modules):
        enc, gat_layers, _ = self._build_encoder_with_spy_layers(
            mock_modules, num_layers=3
        )

        fake_x = MagicMock(name="x")
        fake_edge_index = MagicMock(name="edge_index")
        fake_edge_attr = MagicMock(name="edge_attr")

        for layer in gat_layers:
            layer.return_value = fake_x

        try:
            enc.forward(fake_x, fake_edge_index, fake_edge_attr)
        except Exception:
            pass

        total_calls = sum(l.call_count for l in gat_layers)
        assert total_calls == 3


# ===========================================================================
# 6. TestGATEncoderOutputShape — output shape with real torch (integration)
# ===========================================================================

class TestGATEncoderOutputShape:
    """
    Integration tests that use real torch (if available) to verify the output
    shape contract: forward(x, edge_index, edge_attr) -> (N, hidden_dim).

    These tests are skipped if torch or torch_geometric is not installed.
    """

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

    def test_output_shape_default_params(self, real_module):
        import torch
        N, node_in, edge_in, E = 15, 10, 3, 20
        enc = real_module.GATEncoder(
            node_in_features=node_in,
            edge_in_features=edge_in,
        )
        enc.eval()
        x = torch.randn(N, node_in)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.randn(E, edge_in)
        out = enc(x, edge_index, edge_attr)
        assert out.shape == (N, 64), f"Expected (15, 64), got {out.shape}"

    def test_output_shape_custom_hidden_dim(self, real_module):
        import torch
        N, node_in, edge_in, E = 8, 6, 4, 12
        enc = real_module.GATEncoder(
            node_in_features=node_in,
            edge_in_features=edge_in,
            hidden_dim=32,
            num_heads=4,
            num_layers=2,
        )
        enc.eval()
        x = torch.randn(N, node_in)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_in)
        out = enc(x, edge_index, edge_attr)
        assert out.shape == (N, 32)

    def test_output_dtype_is_float32(self, real_module):
        import torch
        N, node_in, edge_in, E = 10, 8, 3, 15
        enc = real_module.GATEncoder(
            node_in_features=node_in,
            edge_in_features=edge_in,
        )
        enc.eval()
        x = torch.randn(N, node_in)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_in)
        out = enc(x, edge_index, edge_attr)
        assert out.dtype == torch.float32

    def test_three_layer_encoder_output_shape(self, real_module):
        import torch
        N, node_in, edge_in, E = 12, 10, 5, 18
        enc = real_module.GATEncoder(
            node_in_features=node_in,
            edge_in_features=edge_in,
            hidden_dim=64,
            num_heads=4,
            num_layers=3,
        )
        enc.eval()
        x = torch.randn(N, node_in)
        edge_index = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(E, edge_in)
        out = enc(x, edge_index, edge_attr)
        assert out.shape == (N, 64)

    def test_count_parameters_real_model(self, real_module):
        enc = real_module.GATEncoder(node_in_features=10, edge_in_features=3)
        total = real_module.count_parameters(enc)
        assert isinstance(total, int)
        assert total > 0

    def test_make_gat_encoder_real_model(self, real_module):
        import torch
        cfg = {"node_in_features": 10, "edge_in_features": 3}
        enc = real_module.make_gat_encoder(cfg)
        assert isinstance(enc, real_module.GATEncoder)

    def test_make_gat_encoder_raises_on_indivisible_real(self, real_module):
        cfg = {"node_in_features": 10, "edge_in_features": 3,
               "hidden_dim": 65, "num_heads": 4}
        with pytest.raises(ValueError):
            real_module.make_gat_encoder(cfg)
