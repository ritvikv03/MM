"""
tests/pipeline/test_pipeline.py

Tests for src/pipeline/pipeline.py — all external calls are mocked.

No real torch, pymc, wandb, or data I/O is performed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.pipeline.config import PipelineConfig
from src.pipeline.pipeline import (
    build_gat_config,
    build_temporal_config,
    run_pipeline,
    run_training_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_cfg(**overrides) -> PipelineConfig:
    """Return a minimal valid PipelineConfig."""
    defaults = {
        "gat_hidden_dim": 64,
        "gat_num_heads": 4,
        "n_epochs": 3,
        "n_splits": 2,
        "test_size": 5,
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


# ---------------------------------------------------------------------------
# build_gat_config
# ---------------------------------------------------------------------------


class TestBuildGatConfig:
    def test_returns_dict(self):
        cfg = _valid_cfg()
        result = build_gat_config(cfg)
        assert isinstance(result, dict)

    def test_includes_node_in_features(self):
        cfg = _valid_cfg(node_in_features=15)
        result = build_gat_config(cfg)
        assert result["node_in_features"] == 15

    def test_includes_hidden_dim(self):
        cfg = _valid_cfg(gat_hidden_dim=128)
        result = build_gat_config(cfg)
        assert result["hidden_dim"] == 128

    def test_includes_num_heads(self):
        cfg = _valid_cfg(gat_num_heads=8)
        result = build_gat_config(cfg)
        assert result["num_heads"] == 8

    def test_includes_num_layers(self):
        cfg = _valid_cfg(gat_num_layers=3)
        result = build_gat_config(cfg)
        assert result["num_layers"] == 3

    def test_includes_dropout(self):
        cfg = _valid_cfg(gat_dropout=0.25)
        result = build_gat_config(cfg)
        assert result["dropout"] == 0.25

    def test_all_required_keys_present(self):
        cfg = _valid_cfg()
        result = build_gat_config(cfg)
        for key in ("node_in_features", "hidden_dim", "num_heads", "num_layers", "dropout"):
            assert key in result, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# build_temporal_config
# ---------------------------------------------------------------------------


class TestBuildTemporalConfig:
    def test_returns_dict(self):
        cfg = _valid_cfg()
        result = build_temporal_config(cfg)
        assert isinstance(result, dict)

    def test_includes_hidden_dim(self):
        cfg = _valid_cfg(temporal_hidden_dim=256)
        result = build_temporal_config(cfg)
        assert result["hidden_dim"] == 256

    def test_includes_num_layers(self):
        cfg = _valid_cfg(temporal_num_layers=3)
        result = build_temporal_config(cfg)
        assert result["num_layers"] == 3

    def test_includes_dropout(self):
        cfg = _valid_cfg(temporal_dropout=0.3)
        result = build_temporal_config(cfg)
        assert result["dropout"] == 0.3

    def test_includes_encoder_type(self):
        cfg = _valid_cfg(temporal_encoder_type="transformer")
        result = build_temporal_config(cfg)
        assert result["encoder_type"] == "transformer"

    def test_lstm_encoder_type(self):
        cfg = _valid_cfg(temporal_encoder_type="lstm")
        result = build_temporal_config(cfg)
        assert result["encoder_type"] == "lstm"

    def test_all_required_keys_present(self):
        cfg = _valid_cfg()
        result = build_temporal_config(cfg)
        for key in ("hidden_dim", "num_layers", "dropout", "encoder_type"):
            assert key in result, f"Missing key: {key!r}"


# ---------------------------------------------------------------------------
# run_training_loop
# ---------------------------------------------------------------------------


class TestRunTrainingLoop:
    def _make_mock_train_one_epoch(self, n_epochs: int):
        """Return a mock that produces plausible per-epoch metric dicts."""
        mock = MagicMock(
            side_effect=[
                {"brier_score": 0.25, "log_loss": 0.6, "accuracy": 0.7}
                for _ in range(n_epochs)
            ]
        )
        return mock

    def test_calls_train_one_epoch_n_epochs_times(self):
        n_epochs = 4
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        model = MagicMock()
        optimizer = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=model,
                optimizer=optimizer,
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        assert mock_train.call_count == n_epochs

    def test_returns_list_of_length_n_epochs(self):
        n_epochs = 5
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        model = MagicMock()
        optimizer = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=model,
                optimizer=optimizer,
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        assert isinstance(history, list)
        assert len(history) == n_epochs

    def test_all_wandb_keys_present_in_each_entry(self):
        n_epochs = 3
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        model = MagicMock()
        optimizer = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=model,
                optimizer=optimizer,
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        required_keys = {"brier_score", "log_loss", "clv_delta", "calibration_ece", "epoch"}
        for entry in history:
            assert required_keys.issubset(entry.keys()), (
                f"Missing keys {required_keys - entry.keys()} in epoch entry"
            )

    def test_clv_delta_defaults_to_zero(self):
        n_epochs = 2
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        model = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=model,
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        for entry in history:
            assert entry["clv_delta"] == 0.0

    def test_calibration_ece_defaults_to_zero(self):
        n_epochs = 2
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        model = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=model,
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        for entry in history:
            assert entry["calibration_ece"] == 0.0

    def test_epoch_index_in_each_entry(self):
        n_epochs = 3
        mock_train = self._make_mock_train_one_epoch(n_epochs)

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=MagicMock(),
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
            )

        epochs_found = [entry["epoch"] for entry in history]
        assert epochs_found == list(range(n_epochs))

    def test_logger_log_epoch_called_n_epochs_times(self):
        n_epochs = 4
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        mock_logger = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            run_training_loop(
                model=MagicMock(),
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
                logger=mock_logger,
            )

        assert mock_logger.log_epoch.call_count == n_epochs

    def test_no_logger_does_not_raise(self):
        n_epochs = 2
        mock_train = self._make_mock_train_one_epoch(n_epochs)

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            # No logger — should not raise
            history = run_training_loop(
                model=MagicMock(),
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
                logger=None,
            )
        assert len(history) == n_epochs

    def test_extra_metrics_fn_called_each_epoch(self):
        n_epochs = 3
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        extra_fn = MagicMock(return_value={"accuracy": 0.8})

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=MagicMock(),
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
                extra_metrics_fn=extra_fn,
            )

        assert extra_fn.call_count == n_epochs

    def test_extra_metrics_merged_into_history_entry(self):
        n_epochs = 2
        mock_train = self._make_mock_train_one_epoch(n_epochs)
        extra_fn = MagicMock(return_value={"accuracy": 0.9})

        with patch("src.pipeline.pipeline.train_one_epoch", mock_train):
            history = run_training_loop(
                model=MagicMock(),
                optimizer=MagicMock(),
                snapshots_list=[MagicMock()],
                game_home_idx=[0],
                game_away_idx=[1],
                labels=[1],
                n_epochs=n_epochs,
                extra_metrics_fn=extra_fn,
            )

        for entry in history:
            assert "accuracy" in entry


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def _stub_data_loader(self, cfg):
        """Minimal stub returning valid data dict."""
        return {
            "snapshots": [MagicMock()],
            "home_idx": [0],
            "away_idx": [1],
            "labels": [1],
            "game_records": [
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                {"y_true": 1, "clv": 0.0},
                {"y_true": 0, "clv": 0.0},
                # Need enough records: n_splits=2, test_size=5 → need ≥ 11 + gap
            ],
        }

    def _stub_model_factory(self, cfg):
        """Stub returning mock model and optimizer."""
        return MagicMock(), MagicMock()

    def test_validate_config_called_and_abort_on_invalid(self):
        """If validate_config raises, run_pipeline should propagate the error."""
        cfg = _valid_cfg()
        cfg.n_epochs = 0  # invalid — will trigger ValueError

        with pytest.raises(ValueError):
            run_pipeline(cfg)

    def test_data_loader_fn_called_once(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)
        data_loader = MagicMock(side_effect=self._stub_data_loader)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            run_pipeline(
                cfg,
                data_loader_fn=data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        data_loader.assert_called_once_with(cfg)

    def test_model_factory_fn_called_once(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)
        model_factory = MagicMock(side_effect=self._stub_model_factory)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=model_factory,
            )

        model_factory.assert_called_once_with(cfg)

    def test_returns_dict_with_all_required_keys(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        assert "training_history" in result
        assert "backtest_results" in result
        assert "backtest_summary" in result
        assert "config" in result

    def test_training_history_is_list(self):
        cfg = _valid_cfg(n_epochs=2, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        assert isinstance(result["training_history"], list)

    def test_config_key_is_dict_matching_to_wandb_config(self):
        from src.pipeline.config import to_wandb_config

        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        expected = to_wandb_config(cfg)
        assert result["config"] == expected

    def test_stub_path_all_none_injected_functions(self):
        """All-None injected functions should work (stub path, no data/model)."""
        cfg = _valid_cfg(n_epochs=0)
        # n_epochs=0 would fail validate... use 1 but None data_loader
        cfg2 = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)

        result = run_pipeline(
            cfg2,
            data_loader_fn=None,
            model_factory_fn=None,
        )

        assert "training_history" in result
        assert "backtest_results" in result
        assert "backtest_summary" in result
        assert "config" in result

    def test_bayesian_head_fn_called_if_provided(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)
        bayesian_fn = MagicMock(return_value=MagicMock())

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
                bayesian_head_fn=bayesian_fn,
            )

        bayesian_fn.assert_called_once()

    def test_logger_receives_final_backtest_summary(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)
        mock_logger = MagicMock()

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
                logger=mock_logger,
            )

        # Logger should have been called with backtest summary data
        assert mock_logger.log_epoch.call_count >= 1 or mock_logger.method_calls

    def test_backtest_results_is_list(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        assert isinstance(result["backtest_results"], list)

    def test_backtest_summary_is_dict(self):
        cfg = _valid_cfg(n_epochs=1, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        assert isinstance(result["backtest_summary"], dict)

    def test_validate_config_called_first(self):
        """validate_config must be invoked before any data loading."""
        cfg = _valid_cfg()
        data_loader = MagicMock(side_effect=self._stub_data_loader)

        with patch("src.pipeline.pipeline.validate_config", side_effect=ValueError("bad")):
            with pytest.raises(ValueError, match="bad"):
                run_pipeline(cfg, data_loader_fn=data_loader)

        # data_loader should NOT have been called since validation raised
        data_loader.assert_not_called()

    def test_training_history_length_equals_n_epochs_when_data_provided(self):
        n_epochs = 3
        cfg = _valid_cfg(n_epochs=n_epochs, n_splits=2, test_size=5)

        with patch("src.pipeline.pipeline.train_one_epoch") as mock_train:
            mock_train.return_value = {"brier_score": 0.25, "log_loss": 0.6}
            result = run_pipeline(
                cfg,
                data_loader_fn=self._stub_data_loader,
                model_factory_fn=self._stub_model_factory,
            )

        assert len(result["training_history"]) == n_epochs
