"""
tests/pipeline/test_config.py

Tests for src/pipeline/config.py — PipelineConfig dataclass and helpers.

All tests are pure-stdlib + dataclass; no torch/pymc/wandb required.
"""
from __future__ import annotations

import pytest

from src.pipeline.config import (
    PipelineConfig,
    config_from_dict,
    to_wandb_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# PipelineConfig — default values
# ---------------------------------------------------------------------------


class TestPipelineConfigDefaults:
    def test_season_default(self):
        cfg = PipelineConfig()
        assert cfg.season == 2024

    def test_sampler_default(self):
        cfg = PipelineConfig()
        assert cfg.sampler == "advi"

    def test_wandb_mode_default(self):
        cfg = PipelineConfig()
        assert cfg.wandb_mode == "disabled"

    def test_data_dir_default(self):
        cfg = PipelineConfig()
        assert cfg.data_dir == "data/raw"

    def test_cache_dir_default(self):
        cfg = PipelineConfig()
        assert cfg.cache_dir == "data/cache"

    def test_gat_hidden_dim_default(self):
        cfg = PipelineConfig()
        assert cfg.gat_hidden_dim == 64

    def test_gat_num_heads_default(self):
        cfg = PipelineConfig()
        assert cfg.gat_num_heads == 4

    def test_gat_num_layers_default(self):
        cfg = PipelineConfig()
        assert cfg.gat_num_layers == 2

    def test_temporal_encoder_type_default(self):
        cfg = PipelineConfig()
        assert cfg.temporal_encoder_type == "lstm"

    def test_n_epochs_default(self):
        cfg = PipelineConfig()
        assert cfg.n_epochs == 50

    def test_learning_rate_default(self):
        cfg = PipelineConfig()
        assert cfg.learning_rate == 0.001

    def test_device_default(self):
        cfg = PipelineConfig()
        assert cfg.device == "cpu"

    def test_wandb_project_default(self):
        cfg = PipelineConfig()
        assert cfg.wandb_project == "mm-stgnn"

    def test_wandb_entity_default(self):
        cfg = PipelineConfig()
        assert cfg.wandb_entity == ""

    def test_model_version_default(self):
        cfg = PipelineConfig()
        assert cfg.model_version == "v1.0"

    def test_n_splits_default(self):
        cfg = PipelineConfig()
        assert cfg.n_splits == 5

    def test_test_size_default(self):
        cfg = PipelineConfig()
        assert cfg.test_size == 20

    def test_advi_iterations_default(self):
        cfg = PipelineConfig()
        assert cfg.advi_iterations == 10_000

    def test_nuts_draws_default(self):
        cfg = PipelineConfig()
        assert cfg.nuts_draws == 500

    def test_random_seed_default(self):
        cfg = PipelineConfig()
        assert cfg.random_seed == 42

    def test_node_in_features_default(self):
        cfg = PipelineConfig()
        assert cfg.node_in_features == 12

    def test_edge_in_features_default(self):
        cfg = PipelineConfig()
        assert cfg.edge_in_features == 5

    def test_backtest_gap_default(self):
        cfg = PipelineConfig()
        assert cfg.backtest_gap == 1


# ---------------------------------------------------------------------------
# validate_config — invalid cases each raise ValueError
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def _make_valid(self) -> PipelineConfig:
        """Return a config that passes all validations."""
        # gat_hidden_dim=64, gat_num_heads=4 → 64 % 4 == 0 ✓
        return PipelineConfig()

    def test_valid_config_passes(self):
        cfg = self._make_valid()
        validate_config(cfg)  # should not raise

    def test_gat_hidden_dim_not_divisible_by_heads(self):
        cfg = self._make_valid()
        cfg.gat_hidden_dim = 65  # 65 % 4 != 0
        with pytest.raises(ValueError, match="gat_hidden_dim"):
            validate_config(cfg)

    def test_gat_hidden_dim_error_message_mentions_heads(self):
        cfg = self._make_valid()
        cfg.gat_hidden_dim = 65
        with pytest.raises(ValueError, match="gat_num_heads"):
            validate_config(cfg)

    def test_n_epochs_zero_raises(self):
        cfg = self._make_valid()
        cfg.n_epochs = 0
        with pytest.raises(ValueError, match="n_epochs"):
            validate_config(cfg)

    def test_n_epochs_negative_raises(self):
        cfg = self._make_valid()
        cfg.n_epochs = -5
        with pytest.raises(ValueError, match="n_epochs"):
            validate_config(cfg)

    def test_invalid_sampler_raises(self):
        cfg = self._make_valid()
        cfg.sampler = "mcmc_invalid"
        with pytest.raises(ValueError, match="sampler"):
            validate_config(cfg)

    def test_invalid_temporal_encoder_type_raises(self):
        cfg = self._make_valid()
        cfg.temporal_encoder_type = "rnn"
        with pytest.raises(ValueError, match="temporal_encoder_type"):
            validate_config(cfg)

    def test_n_splits_zero_raises(self):
        cfg = self._make_valid()
        cfg.n_splits = 0
        with pytest.raises(ValueError, match="n_splits"):
            validate_config(cfg)

    def test_n_splits_negative_raises(self):
        cfg = self._make_valid()
        cfg.n_splits = -1
        with pytest.raises(ValueError, match="n_splits"):
            validate_config(cfg)

    def test_test_size_zero_raises(self):
        cfg = self._make_valid()
        cfg.test_size = 0
        with pytest.raises(ValueError, match="test_size"):
            validate_config(cfg)

    def test_test_size_negative_raises(self):
        cfg = self._make_valid()
        cfg.test_size = -1
        with pytest.raises(ValueError, match="test_size"):
            validate_config(cfg)

    def test_sampler_nuts_is_valid(self):
        cfg = self._make_valid()
        cfg.sampler = "nuts"
        validate_config(cfg)  # should not raise

    def test_temporal_encoder_transformer_is_valid(self):
        cfg = self._make_valid()
        cfg.temporal_encoder_type = "transformer"
        validate_config(cfg)  # should not raise

    def test_n_splits_one_is_valid(self):
        cfg = self._make_valid()
        cfg.n_splits = 1
        validate_config(cfg)  # should not raise

    def test_test_size_one_is_valid(self):
        cfg = self._make_valid()
        cfg.test_size = 1
        validate_config(cfg)  # should not raise

    def test_n_epochs_one_is_valid(self):
        cfg = self._make_valid()
        cfg.n_epochs = 1
        validate_config(cfg)  # should not raise


# ---------------------------------------------------------------------------
# config_from_dict
# ---------------------------------------------------------------------------


class TestConfigFromDict:
    def test_known_keys_set_correctly(self):
        d = {"season": 2025, "gat_hidden_dim": 128, "gat_num_heads": 4}
        cfg = config_from_dict(d)
        assert cfg.season == 2025
        assert cfg.gat_hidden_dim == 128
        assert cfg.gat_num_heads == 4

    def test_unknown_keys_ignored(self):
        d = {"totally_unknown_key": "ignored_value", "another_key": 999}
        cfg = config_from_dict(d)
        # defaults should still be intact
        assert cfg.season == 2024
        assert cfg.sampler == "advi"

    def test_missing_keys_use_defaults(self):
        cfg = config_from_dict({})
        default = PipelineConfig()
        assert cfg.season == default.season
        assert cfg.n_epochs == default.n_epochs
        assert cfg.sampler == default.sampler

    def test_partial_dict_merges_with_defaults(self):
        d = {"n_epochs": 100}
        cfg = config_from_dict(d)
        assert cfg.n_epochs == 100
        assert cfg.learning_rate == PipelineConfig().learning_rate

    def test_returns_pipeline_config_instance(self):
        cfg = config_from_dict({})
        assert isinstance(cfg, PipelineConfig)

    def test_all_known_fields_can_be_set(self):
        d = {
            "season": 2023,
            "data_dir": "custom/raw",
            "cache_dir": "custom/cache",
            "node_in_features": 20,
            "edge_in_features": 8,
            "gat_hidden_dim": 128,
            "gat_num_heads": 8,
            "gat_num_layers": 3,
            "gat_dropout": 0.2,
            "temporal_hidden_dim": 256,
            "temporal_num_layers": 3,
            "temporal_dropout": 0.2,
            "temporal_encoder_type": "transformer",
            "learning_rate": 0.01,
            "n_epochs": 100,
            "device": "cuda",
            "sampler": "nuts",
            "advi_iterations": 5000,
            "nuts_draws": 200,
            "nuts_chains": 4,
            "nuts_tune": 100,
            "random_seed": 0,
            "n_splits": 3,
            "test_size": 10,
            "backtest_gap": 2,
            "wandb_project": "custom-project",
            "wandb_entity": "myteam",
            "wandb_mode": "offline",
            "model_version": "v2.0",
        }
        cfg = config_from_dict(d)
        assert cfg.season == 2023
        assert cfg.data_dir == "custom/raw"
        assert cfg.temporal_encoder_type == "transformer"
        assert cfg.sampler == "nuts"
        assert cfg.wandb_mode == "offline"

    def test_empty_string_ignored_not_none_comparison(self):
        """config_from_dict accepts empty string for wandb_entity."""
        cfg = config_from_dict({"wandb_entity": ""})
        assert cfg.wandb_entity == ""


# ---------------------------------------------------------------------------
# to_wandb_config
# ---------------------------------------------------------------------------


class TestToWandbConfig:
    def test_returns_dict(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert isinstance(result, dict)

    def test_excludes_wandb_project(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "wandb_project" not in result

    def test_excludes_wandb_entity(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "wandb_entity" not in result

    def test_excludes_wandb_mode(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "wandb_mode" not in result

    def test_includes_season(self):
        cfg = PipelineConfig(season=2025)
        result = to_wandb_config(cfg)
        assert result["season"] == 2025

    def test_includes_learning_rate(self):
        cfg = PipelineConfig(learning_rate=0.005)
        result = to_wandb_config(cfg)
        assert result["learning_rate"] == 0.005

    def test_includes_gat_hidden_dim(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "gat_hidden_dim" in result
        assert result["gat_hidden_dim"] == cfg.gat_hidden_dim

    def test_includes_n_epochs(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "n_epochs" in result

    def test_includes_sampler(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "sampler" in result
        assert result["sampler"] == "advi"

    def test_includes_model_version(self):
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        assert "model_version" in result

    def test_flat_dict_structure(self):
        """All values should be scalar (not nested dicts/lists)."""
        cfg = PipelineConfig()
        result = to_wandb_config(cfg)
        for k, v in result.items():
            assert not isinstance(v, dict), f"Key {k!r} has nested dict value"
