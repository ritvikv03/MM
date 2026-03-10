"""
tests/utils/test_wandb_logger.py

RED phase — all tests written before implementation exists.

Strategy
--------
- Mock `wandb` entirely via patch.dict(sys.modules, {"wandb": MagicMock()})
  so no real W&B network calls are made and the package need not be installed
  in CI at test time.
- ExperimentLogger: init_run, log_epoch, log_hyperparams, log_artifact, finish.
- Standalone helpers: compute_ece, make_sweep_config, format_run_name.
- 30+ tests across 8 test classes.
"""

from __future__ import annotations

import os
import sys
import types
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Module-level wandb mock — injected before src.utils.wandb_logger is loaded
# ---------------------------------------------------------------------------

def _make_wandb_mock() -> MagicMock:
    """Return a MagicMock that covers all wandb surface used by the module."""
    wandb = MagicMock(name="wandb")

    # wandb.init returns a run-like mock
    mock_run = MagicMock(name="wandb.run")
    mock_run.config = MagicMock(name="run.config")
    wandb.init.return_value = mock_run

    # wandb.config is a module-level attribute too (for wandb.config.update)
    wandb.config = MagicMock(name="wandb.config")

    # wandb.Artifact returns an artifact-like mock
    mock_artifact = MagicMock(name="wandb.Artifact_instance")
    wandb.Artifact.return_value = mock_artifact

    # wandb.finish, wandb.log are plain MagicMock by default
    return wandb


# Patch sys.modules before importing the module under test so the module
# captures our mock instead of the real (possibly absent) wandb package.
_wandb_mock = _make_wandb_mock()


@pytest.fixture(autouse=True)
def reset_wandb_mock():
    """Reset all call counts on the wandb mock between tests."""
    _wandb_mock.reset_mock()
    # Re-wire Artifact because reset_mock clears return_value chains
    mock_artifact = MagicMock(name="wandb.Artifact_instance")
    _wandb_mock.Artifact.return_value = mock_artifact
    mock_run = MagicMock(name="wandb.run")
    mock_run.config = MagicMock(name="run.config")
    _wandb_mock.init.return_value = mock_run
    # config is explicitly assigned (not auto-generated), so reset_mock()
    # does not clear its call history — reset it explicitly.
    _wandb_mock.config.reset_mock()
    yield


def _import_logger():
    """Import (or re-import) wandb_logger with the mocked wandb in place."""
    with patch.dict(sys.modules, {"wandb": _wandb_mock}):
        import importlib
        if "src.utils.wandb_logger" in sys.modules:
            mod = importlib.reload(sys.modules["src.utils.wandb_logger"])
        else:
            import src.utils.wandb_logger as mod
        return mod


# ---------------------------------------------------------------------------
# 1. ExperimentLogger — __init__
# ---------------------------------------------------------------------------

class TestExperimentLoggerInit:
    def test_default_attributes(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="test_proj")
        assert logger.project == "test_proj"
        assert logger.entity is None
        assert logger.season == 2024
        assert logger.model_version == "v1.0"
        assert logger.tags is None
        assert logger.mode == "online"

    def test_custom_attributes(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(
            project="mm_proj",
            entity="my_team",
            season=2025,
            model_version="v2.1",
            tags=["experiment"],
            mode="offline",
        )
        assert logger.entity == "my_team"
        assert logger.season == 2025
        assert logger.model_version == "v2.1"
        assert logger.tags == ["experiment"]
        assert logger.mode == "offline"

    def test_run_is_none_before_init(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        assert logger.run is None

    def test_disabled_mode_accepted(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", mode="disabled")
        assert logger.mode == "disabled"


# ---------------------------------------------------------------------------
# 2. ExperimentLogger.init_run
# ---------------------------------------------------------------------------

class TestInitRun:
    def test_calls_wandb_init(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="proj", entity="ent", season=2024, model_version="v1.0")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={"lr": 0.01}, run_name="run1")
        _wandb_mock.init.assert_called_once()

    def test_init_run_passes_project_and_entity(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="my_proj", entity="my_ent")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        kwargs = _wandb_mock.init.call_args.kwargs
        assert kwargs.get("project") == "my_proj"
        assert kwargs.get("entity") == "my_ent"

    def test_init_run_includes_season_tag(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", season=2025)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        tags = _wandb_mock.init.call_args.kwargs.get("tags", [])
        assert "season=2025" in tags

    def test_init_run_includes_model_version_tag(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", model_version="v3.0")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        tags = _wandb_mock.init.call_args.kwargs.get("tags", [])
        assert "model_version=v3.0" in tags

    def test_init_run_includes_extra_tags(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", tags=["prod", "sweep"])
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        tags = _wandb_mock.init.call_args.kwargs.get("tags", [])
        assert "prod" in tags
        assert "sweep" in tags

    def test_init_run_passes_config(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        cfg = {"lr": 0.001, "hidden_dim": 128}
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config=cfg)
        assert _wandb_mock.init.call_args.kwargs.get("config") == cfg

    def test_init_run_passes_run_name(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={}, run_name="my_run")
        assert _wandb_mock.init.call_args.kwargs.get("name") == "my_run"

    def test_init_run_passes_mode(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", mode="offline")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        assert _wandb_mock.init.call_args.kwargs.get("mode") == "offline"

    def test_init_run_stores_run(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        assert logger.run is not None

    def test_init_run_raises_runtime_error_when_api_key_missing_online(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", mode="online")
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            with patch.dict(sys.modules, {"wandb": _wandb_mock}):
                logger.init_run(config={})

    def test_init_run_no_error_when_api_key_missing_offline(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", mode="offline")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})  # should not raise
        assert logger.run is not None

    def test_init_run_no_error_when_api_key_missing_disabled(self, monkeypatch):
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", mode="disabled")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})  # should not raise
        assert logger.run is not None

    def test_init_run_no_extra_tags_when_tags_none(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p", season=2024, model_version="v1.0", tags=None)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        tags = _wandb_mock.init.call_args.kwargs.get("tags", [])
        # Should contain exactly 2 mandatory tags, no extras
        assert len(tags) == 2
        assert set(tags) == {"season=2024", "model_version=v1.0"}


# ---------------------------------------------------------------------------
# 3. ExperimentLogger.log_epoch
# ---------------------------------------------------------------------------

class TestLogEpoch:
    def _initialized_logger(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        return logger, mod

    def test_log_epoch_calls_wandb_log(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_epoch(epoch=1, brier_score=0.2, log_loss=0.5, clv_delta=0.05, calibration_ece=0.03)
        _wandb_mock.log.assert_called_once()

    def test_log_epoch_all_five_required_keys(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_epoch(epoch=3, brier_score=0.18, log_loss=0.45, clv_delta=0.02, calibration_ece=0.01)
        logged = _wandb_mock.log.call_args.args[0]
        assert "brier_score" in logged
        assert "log_loss" in logged
        assert "clv_delta" in logged
        assert "calibration_ece" in logged
        assert "epoch" in logged

    def test_log_epoch_correct_values(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_epoch(epoch=5, brier_score=0.15, log_loss=0.40, clv_delta=0.07, calibration_ece=0.02)
        logged = _wandb_mock.log.call_args.args[0]
        assert logged["epoch"] == 5
        assert logged["brier_score"] == pytest.approx(0.15)
        assert logged["log_loss"] == pytest.approx(0.40)
        assert logged["clv_delta"] == pytest.approx(0.07)
        assert logged["calibration_ece"] == pytest.approx(0.02)

    def test_log_epoch_extra_dict_merged(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_epoch(
                epoch=1, brier_score=0.2, log_loss=0.5, clv_delta=0.0, calibration_ece=0.0,
                extra={"custom_metric": 42.0}
            )
        logged = _wandb_mock.log.call_args.args[0]
        assert logged.get("custom_metric") == pytest.approx(42.0)

    def test_log_epoch_extra_none_no_error(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_epoch(epoch=1, brier_score=0.2, log_loss=0.5, clv_delta=0.0, calibration_ece=0.0, extra=None)
        _wandb_mock.log.assert_called_once()

    def test_log_epoch_raises_if_not_initialized(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with pytest.raises(RuntimeError, match="init_run"):
            with patch.dict(sys.modules, {"wandb": _wandb_mock}):
                logger.log_epoch(epoch=1, brier_score=0.2, log_loss=0.5, clv_delta=0.0, calibration_ece=0.0)

    def test_log_epoch_multiple_epochs(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            for ep in range(3):
                logger.log_epoch(epoch=ep, brier_score=0.2, log_loss=0.5, clv_delta=0.0, calibration_ece=0.0)
        assert _wandb_mock.log.call_count == 3


# ---------------------------------------------------------------------------
# 4. ExperimentLogger.log_hyperparams
# ---------------------------------------------------------------------------

class TestLogHyperparams:
    def _initialized_logger(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        return logger, mod

    def test_log_hyperparams_calls_config_update(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        params = {"lr": 0.001, "gat_heads": 4}
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_hyperparams(params)
        _wandb_mock.config.update.assert_called_once_with(params)

    def test_log_hyperparams_raises_if_not_initialized(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with pytest.raises(RuntimeError, match="init_run"):
            with patch.dict(sys.modules, {"wandb": _wandb_mock}):
                logger.log_hyperparams({"lr": 0.001})

    def test_log_hyperparams_empty_dict(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_hyperparams({})
        _wandb_mock.config.update.assert_called_once_with({})


# ---------------------------------------------------------------------------
# 5. ExperimentLogger.log_artifact
# ---------------------------------------------------------------------------

class TestLogArtifact:
    def _initialized_logger(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
        return logger, mod

    def test_log_artifact_raises_file_not_found(self, monkeypatch):
        logger, mod = self._initialized_logger(monkeypatch)
        with pytest.raises(FileNotFoundError):
            with patch.dict(sys.modules, {"wandb": _wandb_mock}):
                logger.log_artifact("/nonexistent/path/model.pt", artifact_type="model")

    def test_log_artifact_creates_artifact_with_file_stem_as_name(self, monkeypatch, tmp_path):
        logger, mod = self._initialized_logger(monkeypatch)
        fake_file = tmp_path / "my_model.pt"
        fake_file.write_bytes(b"fake model weights")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_artifact(str(fake_file), artifact_type="model")
        _wandb_mock.Artifact.assert_called_once()
        name_arg = _wandb_mock.Artifact.call_args.args[0] if _wandb_mock.Artifact.call_args.args else _wandb_mock.Artifact.call_args.kwargs.get("name")
        assert name_arg == "my_model"

    def test_log_artifact_uses_provided_name(self, monkeypatch, tmp_path):
        logger, mod = self._initialized_logger(monkeypatch)
        fake_file = tmp_path / "weights.bin"
        fake_file.write_bytes(b"data")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_artifact(str(fake_file), artifact_type="model", name="custom_name")
        name_arg = _wandb_mock.Artifact.call_args.args[0] if _wandb_mock.Artifact.call_args.args else _wandb_mock.Artifact.call_args.kwargs.get("name")
        assert name_arg == "custom_name"

    def test_log_artifact_sets_artifact_type(self, monkeypatch, tmp_path):
        logger, mod = self._initialized_logger(monkeypatch)
        fake_file = tmp_path / "data.csv"
        fake_file.write_text("a,b\n1,2")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_artifact(str(fake_file), artifact_type="dataset")
        type_arg = _wandb_mock.Artifact.call_args.kwargs.get("type") or (
            _wandb_mock.Artifact.call_args.args[1] if len(_wandb_mock.Artifact.call_args.args) > 1 else None
        )
        assert type_arg == "dataset"

    def test_log_artifact_adds_file_to_artifact(self, monkeypatch, tmp_path):
        logger, mod = self._initialized_logger(monkeypatch)
        fake_file = tmp_path / "artifact.pt"
        fake_file.write_bytes(b"x")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_artifact(str(fake_file), artifact_type="model")
        artifact_instance = _wandb_mock.Artifact.return_value
        artifact_instance.add_file.assert_called_once_with(str(fake_file))

    def test_log_artifact_logs_run_artifact(self, monkeypatch, tmp_path):
        logger, mod = self._initialized_logger(monkeypatch)
        fake_file = tmp_path / "model.pt"
        fake_file.write_bytes(b"x")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.log_artifact(str(fake_file), artifact_type="model")
        run = _wandb_mock.init.return_value
        run.log_artifact.assert_called_once()

    def test_log_artifact_raises_if_not_initialized(self, tmp_path):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        fake_file = tmp_path / "model.pt"
        fake_file.write_bytes(b"x")
        with pytest.raises(RuntimeError, match="init_run"):
            with patch.dict(sys.modules, {"wandb": _wandb_mock}):
                logger.log_artifact(str(fake_file), artifact_type="model")


# ---------------------------------------------------------------------------
# 6. ExperimentLogger.finish
# ---------------------------------------------------------------------------

class TestFinish:
    def test_finish_calls_wandb_finish(self, monkeypatch):
        monkeypatch.setenv("WANDB_API_KEY", "fake_key")
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.init_run(config={})
            logger.finish()
        _wandb_mock.finish.assert_called_once()

    def test_finish_is_noop_if_run_never_initialized(self):
        mod = _import_logger()
        logger = mod.ExperimentLogger(project="p")
        with patch.dict(sys.modules, {"wandb": _wandb_mock}):
            logger.finish()  # should not raise
        _wandb_mock.finish.assert_not_called()


# ---------------------------------------------------------------------------
# 7. compute_ece
# ---------------------------------------------------------------------------

class TestComputeEce:
    def _get_fn(self):
        mod = _import_logger()
        return mod.compute_ece

    def test_empty_input_returns_zero(self):
        fn = self._get_fn()
        result = fn(np.array([]), np.array([]))
        assert result == pytest.approx(0.0)

    def test_perfect_calibration_returns_zero(self):
        """If mean_pred == mean_true in every bin, ECE == 0."""
        fn = self._get_fn()
        # Predictions exactly match labels in each bin
        y_pred = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        y_true = np.array([0.0, 0.2, 0.5, 0.5, 1.0, 0.8])
        # This won't necessarily be 0 due to bin assignments; use a known 0 case:
        # All predictions are 0.5 and half the labels are 1 → mean_pred = mean_true = 0.5
        y_pred2 = np.full(100, 0.5)
        y_true2 = np.concatenate([np.ones(50), np.zeros(50)])
        result = fn(y_pred2, y_true2)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_worst_case_calibration(self):
        """Predict 1.0 for all negative examples → high ECE."""
        fn = self._get_fn()
        y_pred = np.ones(100)
        y_true = np.zeros(100)
        result = fn(y_pred, y_true)
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_shape_mismatch_raises_value_error(self):
        fn = self._get_fn()
        with pytest.raises(ValueError, match="[Ss]hape"):
            fn(np.array([0.5, 0.6]), np.array([0.0]))

    def test_returns_float(self):
        fn = self._get_fn()
        result = fn(np.array([0.3, 0.7]), np.array([0.0, 1.0]))
        assert isinstance(result, float)

    def test_ece_nonnegative(self):
        rng = np.random.default_rng(42)
        fn = self._get_fn()
        y_pred = rng.uniform(0, 1, 200)
        y_true = rng.integers(0, 2, 200).astype(float)
        assert fn(y_pred, y_true) >= 0.0

    def test_ece_at_most_one(self):
        rng = np.random.default_rng(0)
        fn = self._get_fn()
        y_pred = rng.uniform(0, 1, 200)
        y_true = rng.integers(0, 2, 200).astype(float)
        assert fn(y_pred, y_true) <= 1.0

    def test_n_bins_respected(self):
        """Using n_bins=1 puts everything in one bin; ECE = |mean_pred - mean_true|."""
        fn = self._get_fn()
        y_pred = np.array([0.4, 0.6])
        y_true = np.array([0.0, 1.0])
        ece_1bin = fn(y_pred, y_true, n_bins=1)
        # mean_pred = 0.5, mean_true = 0.5 → ECE = 0
        assert ece_1bin == pytest.approx(0.0, abs=1e-10)

    def test_ece_known_value(self):
        """
        All predictions in [0.9, 1.0) bin, y_true all 0:
        mean_pred ≈ 0.95, mean_true = 0.0 → ECE ≈ 0.95.
        """
        fn = self._get_fn()
        y_pred = np.full(10, 0.95)
        y_true = np.zeros(10)
        result = fn(y_pred, y_true, n_bins=10)
        assert result == pytest.approx(0.95, abs=1e-10)

    def test_single_element_input(self):
        fn = self._get_fn()
        result = fn(np.array([0.7]), np.array([1.0]))
        # |0.7 - 1.0| = 0.3, weight = 1/1 = 1 → ECE = 0.3
        assert result == pytest.approx(0.3, abs=1e-10)


# ---------------------------------------------------------------------------
# 8. make_sweep_config
# ---------------------------------------------------------------------------

class TestMakeSweepConfig:
    def _get_fn(self):
        mod = _import_logger()
        return mod.make_sweep_config

    def test_output_has_method_key(self):
        fn = self._get_fn()
        result = fn({"lr": [0.001, 0.01]})
        assert "method" in result

    def test_method_is_grid(self):
        fn = self._get_fn()
        result = fn({"lr": [0.001, 0.01]})
        assert result["method"] == "grid"

    def test_output_has_parameters_key(self):
        fn = self._get_fn()
        result = fn({"lr": [0.001]})
        assert "parameters" in result

    def test_each_param_wrapped_in_values(self):
        fn = self._get_fn()
        result = fn({"lr": [0.001, 0.01], "hidden_dim": [64, 128]})
        assert result["parameters"]["lr"] == {"values": [0.001, 0.01]}
        assert result["parameters"]["hidden_dim"] == {"values": [64, 128]}

    def test_empty_param_grid(self):
        fn = self._get_fn()
        result = fn({})
        assert result["method"] == "grid"
        assert result["parameters"] == {}

    def test_single_param(self):
        fn = self._get_fn()
        result = fn({"dropout": [0.1, 0.3, 0.5]})
        assert result["parameters"]["dropout"]["values"] == [0.1, 0.3, 0.5]

    def test_preserves_param_names(self):
        fn = self._get_fn()
        params = {"gat_heads": [2, 4, 8], "lstm_hidden": [32, 64]}
        result = fn(params)
        assert "gat_heads" in result["parameters"]
        assert "lstm_hidden" in result["parameters"]

    def test_returns_dict(self):
        fn = self._get_fn()
        result = fn({"lr": [0.01]})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 9. format_run_name
# ---------------------------------------------------------------------------

class TestFormatRunName:
    def _get_fn(self):
        mod = _import_logger()
        return mod.format_run_name

    def test_basic_format(self):
        fn = self._get_fn()
        assert fn(2024, "v1.0", "advi") == "mm_2024_v1.0_advi"

    def test_different_season(self):
        fn = self._get_fn()
        assert fn(2025, "v1.0", "nuts") == "mm_2025_v1.0_nuts"

    def test_different_model_version(self):
        fn = self._get_fn()
        assert fn(2024, "v2.3", "advi") == "mm_2024_v2.3_advi"

    def test_different_sampler(self):
        fn = self._get_fn()
        assert fn(2024, "v1.0", "svi") == "mm_2024_v1.0_svi"

    def test_returns_string(self):
        fn = self._get_fn()
        result = fn(2024, "v1.0", "advi")
        assert isinstance(result, str)

    def test_starts_with_mm(self):
        fn = self._get_fn()
        assert fn(2024, "v1.0", "advi").startswith("mm_")
