"""
tests/model/test_sentiment_encoder.py

Test suite for SentimentEncoder — lightweight attention-based encoder that
converts news/injury alert dicts into 8-dimensional latent vectors per team.

No HTTP calls; all tests pass pre-built alert dicts.
All torch / torch_geometric packages NOT required for these tests.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def encoder():
    from src.model.encoders.sentiment_encoder import SentimentEncoder
    return SentimentEncoder(latent_dim=8, recency_halflife_hours=24.0)


def _alert(team: str, keyword: str, hours_ago: float = 0.0) -> dict:
    """Build a minimal alert dict as produced by news_scraper.py."""
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    return {
        "team": team,
        "keyword": keyword,
        "text": f"{team} player with {keyword}",
        "source": "reddit",
        "url": "https://example.com",
        "timestamp": ts.isoformat(),
        "status": "REQUIRES_MANUAL_REVIEW",
    }


# ===========================================================================
# 1. Import & module-level checks
# ===========================================================================

class TestModuleImports:
    def test_severity_weights_importable(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS is not None

    def test_latent_dim_constant_importable(self):
        from src.model.encoders.sentiment_encoder import LATENT_DIM
        assert LATENT_DIM == 8

    def test_sentinel_encoder_importable(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        assert callable(SentimentEncoder)

    def test_package_init_exposes_encoder(self):
        from src.model.encoders import SentimentEncoder, SEVERITY_WEIGHTS
        assert callable(SentimentEncoder)
        assert isinstance(SEVERITY_WEIGHTS, dict)

    def test_no_bare_torch_import_in_module(self):
        """SentimentEncoder must not import torch at module level."""
        import ast, pathlib
        src = pathlib.Path(
            __file__
        ).parent.parent.parent / "src" / "model" / "encoders" / "sentiment_encoder.py"
        tree = ast.parse(src.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "torch", (
                        "Bare 'import torch' found at module level in sentiment_encoder.py"
                    )


# ===========================================================================
# 2. SEVERITY_WEIGHTS content
# ===========================================================================

class TestSeverityWeights:
    REQUIRED_KEYS = [
        "walking boot", "sprain", "suspension", "mri", "torn",
        "concussion", "out for season", "not at practice",
        "questionable", "doubtful", "out", "injured", "knee", "ankle",
    ]

    def test_has_all_required_keys(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        for key in self.REQUIRED_KEYS:
            assert key in SEVERITY_WEIGHTS, f"Missing key: {key!r}"

    def test_values_in_zero_one(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        for k, v in SEVERITY_WEIGHTS.items():
            assert 0.0 <= v <= 1.0, f"SEVERITY_WEIGHTS[{k!r}] = {v} outside [0, 1]"

    def test_torn_is_max_severity(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["torn"] == 1.0

    def test_out_for_season_is_max_severity(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["out for season"] == 1.0

    def test_suspension_higher_than_questionable(self):
        from src.model.encoders.sentiment_encoder import SEVERITY_WEIGHTS
        assert SEVERITY_WEIGHTS["suspension"] > SEVERITY_WEIGHTS["questionable"]


# ===========================================================================
# 3. SentimentEncoder construction
# ===========================================================================

class TestSentimentEncoderConstruction:
    def test_default_latent_dim(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder()
        assert enc.latent_dim == 8

    def test_custom_latent_dim(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder(latent_dim=16)
        assert enc.latent_dim == 16

    def test_default_halflife(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder()
        assert enc.recency_halflife_hours == 24.0

    def test_custom_halflife(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder(recency_halflife_hours=12.0)
        assert enc.recency_halflife_hours == 12.0


# ===========================================================================
# 4. encode_single_team — basic shape / zeros
# ===========================================================================

class TestEncodeSingleTeam:
    def test_returns_ndarray(self, encoder):
        result = encoder.encode_single_team("Duke", [])
        assert isinstance(result, np.ndarray)

    def test_returns_correct_shape(self, encoder):
        result = encoder.encode_single_team("Duke", [])
        assert result.shape == (8,)

    def test_unknown_team_returns_zeros(self, encoder):
        result = encoder.encode_single_team("UnknownTeam", [_alert("Duke", "sprain")])
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_no_alerts_returns_zeros(self, encoder):
        result = encoder.encode_single_team("Duke", [])
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_non_matching_alerts_returns_zeros(self, encoder):
        alerts = [_alert("Kansas", "sprain")]
        result = encoder.encode_single_team("Duke", alerts)
        np.testing.assert_array_equal(result, np.zeros(8))

    def test_matching_alert_returns_nonzero(self, encoder):
        alerts = [_alert("Duke", "sprain")]
        result = encoder.encode_single_team("Duke", alerts)
        assert result.sum() > 0.0

    def test_values_clipped_to_zero_one(self, encoder):
        """All output values must be in [0, 1]."""
        alerts = [_alert("Duke", kw) for kw in ["torn", "suspension", "concussion"]]
        result = encoder.encode_single_team("Duke", alerts)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_latent_dim_respected(self):
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder(latent_dim=16)
        result = enc.encode_single_team("Duke", [_alert("Duke", "sprain")])
        assert result.shape == (16,)


# ===========================================================================
# 5. Pressure channel semantics
# ===========================================================================

class TestPressureChannels:
    def test_higher_severity_keyword_gives_higher_pressure(self, encoder):
        """torn (1.0) should produce higher vec[0] than questionable (0.40)."""
        vec_torn = encoder.encode_single_team("Duke", [_alert("Duke", "torn")])
        vec_q    = encoder.encode_single_team("Duke", [_alert("Duke", "questionable")])
        assert vec_torn[0] > vec_q[0], (
            f"torn pressure {vec_torn[0]:.4f} should exceed questionable {vec_q[0]:.4f}"
        )

    def test_availability_flag_set_for_torn(self, encoder):
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "torn")])
        assert vec[4] == 1.0, f"vec[4] (availability flag) should be 1.0 for 'torn', got {vec[4]}"

    def test_availability_flag_set_for_out(self, encoder):
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "out")])
        assert vec[4] == 1.0

    def test_availability_flag_set_for_out_for_season(self, encoder):
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "out for season")])
        assert vec[4] == 1.0

    def test_availability_flag_zero_for_sprain(self, encoder):
        """'sprain' is not in the availability-flag trigger set."""
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "sprain")])
        assert vec[4] == 0.0

    def test_suspension_fraction_channel(self, encoder):
        """vec[5] should be 1.0 when the sole alert is a suspension."""
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "suspension")])
        assert vec[5] == 1.0, f"suspension fraction should be 1.0, got {vec[5]}"

    def test_suspension_fraction_zero_for_injury(self, encoder):
        """vec[5] should be 0.0 for a pure injury alert."""
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "sprain")])
        assert vec[5] == 0.0

    def test_pressure_x_availability_interaction(self, encoder):
        """vec[7] = pressure * vec[4] — should be 0 for sprain (vec[4]=0)."""
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "sprain")])
        assert vec[7] == 0.0

    def test_pressure_x_availability_nonzero_for_out(self, encoder):
        """vec[7] = pressure * vec[4] — should be positive for 'out'."""
        vec = encoder.encode_single_team("Duke", [_alert("Duke", "out")])
        assert vec[7] > 0.0


# ===========================================================================
# 6. Recency decay
# ===========================================================================

class TestRecencyDecay:
    def test_recent_alert_has_higher_weight_than_old(self, encoder):
        """An alert from 1h ago should produce higher pressure than 48h ago."""
        vec_recent = encoder.encode_single_team("Duke", [_alert("Duke", "sprain", hours_ago=1)])
        vec_old    = encoder.encode_single_team("Duke", [_alert("Duke", "sprain", hours_ago=48)])
        assert vec_recent[0] > vec_old[0], (
            "Recent alert should produce higher pressure than old alert"
        )

    def test_recency_decay_uses_halflife(self):
        """At t=halflife hours, weight should be ~0.5."""
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder(latent_dim=8, recency_halflife_hours=24.0)
        vec_hl = enc.encode_single_team("Duke", [_alert("Duke", "sprain", hours_ago=24)])
        vec_0  = enc.encode_single_team("Duke", [_alert("Duke", "sprain", hours_ago=0)])
        # At halflife the weight = exp(-1) ≈ 0.368 of the fresh weight
        # ratio should be roughly exp(-1) ≈ 0.368
        ratio = vec_hl[0] / vec_0[0]
        expected = math.exp(-1.0)
        assert abs(ratio - expected) < 0.05, (
            f"Recency ratio at halflife should be ~{expected:.3f}, got {ratio:.3f}"
        )

    def test_no_timestamp_uses_weight_one(self):
        """Alerts without timestamps should use recency weight = 1.0."""
        from src.model.encoders.sentiment_encoder import SentimentEncoder
        enc = SentimentEncoder()
        alert_no_ts = {
            "team": "Duke", "keyword": "sprain",
            "text": "Duke sprain", "source": "reddit", "url": "",
            "status": "REQUIRES_MANUAL_REVIEW",
            # No "timestamp" key
        }
        vec = enc.encode_single_team("Duke", [alert_no_ts])
        # With weight=1.0 and severity=0.60: pressure = 0.60
        assert abs(vec[0] - 0.60) < 0.01, f"Expected pressure ~0.60, got {vec[0]}"


# ===========================================================================
# 7. encode_alerts
# ===========================================================================

class TestEncodeAlerts:
    def test_returns_dict(self, encoder):
        alerts = [_alert("Duke", "sprain"), _alert("Kansas", "torn")]
        result = encoder.encode_alerts(alerts)
        assert isinstance(result, dict)

    def test_keys_are_team_names(self, encoder):
        alerts = [_alert("Duke", "sprain"), _alert("Kansas", "torn")]
        result = encoder.encode_alerts(alerts)
        assert "Duke" in result
        assert "Kansas" in result

    def test_values_are_ndarrays(self, encoder):
        alerts = [_alert("Duke", "sprain")]
        result = encoder.encode_alerts(alerts)
        assert isinstance(result["Duke"], np.ndarray)

    def test_empty_alerts_returns_empty_dict(self, encoder):
        result = encoder.encode_alerts([])
        assert result == {}

    def test_multiple_teams_independent(self, encoder):
        alerts = [_alert("Duke", "torn"), _alert("Kansas", "questionable")]
        result = encoder.encode_alerts(alerts)
        # Duke (torn=1.0) should have higher pressure than Kansas (questionable=0.40)
        assert result["Duke"][0] > result["Kansas"][0]

    def test_reference_time_parameter_accepted(self, encoder):
        """encode_alerts must accept optional reference_time without crashing."""
        from datetime import datetime, timezone
        ref = datetime.now(timezone.utc).isoformat()
        alerts = [_alert("Duke", "sprain")]
        result = encoder.encode_alerts(alerts, reference_time=ref)
        assert "Duke" in result


# ===========================================================================
# 8. encode_team_matrix (convenience function)
# ===========================================================================

class TestEncodeTeamMatrix:
    def test_importable(self):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        assert callable(encode_team_matrix)

    def test_returns_ndarray(self, encoder):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        teams = ["Duke", "Kansas"]
        alerts = [_alert("Duke", "sprain")]
        result = encode_team_matrix(teams, alerts)
        assert isinstance(result, np.ndarray)

    def test_shape_teams_x_latent_dim(self):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        teams = ["Duke", "Kansas", "Gonzaga"]
        alerts = [_alert("Duke", "sprain")]
        result = encode_team_matrix(teams, alerts)
        assert result.shape == (3, 8), f"Expected (3, 8), got {result.shape}"

    def test_unknown_team_row_is_zeros(self):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        teams = ["Duke", "Gonzaga"]
        alerts = [_alert("Duke", "sprain")]
        result = encode_team_matrix(teams, alerts)
        np.testing.assert_array_equal(result[1], np.zeros(8))  # Gonzaga has no alerts

    def test_custom_latent_dim_kwarg(self):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        teams = ["Duke"]
        alerts = [_alert("Duke", "sprain")]
        result = encode_team_matrix(teams, alerts, latent_dim=16)
        assert result.shape == (1, 16)

    def test_values_all_in_zero_one(self):
        from src.model.encoders.sentiment_encoder import encode_team_matrix
        teams = ["Duke", "Kansas"]
        alerts = [
            _alert("Duke", "torn"), _alert("Duke", "suspension"),
            _alert("Kansas", "concussion"),
        ]
        result = encode_team_matrix(teams, alerts)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
