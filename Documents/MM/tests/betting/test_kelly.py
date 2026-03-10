"""
Tests for src/betting/kelly.py

TDD RED phase — written before implementation.
Covers:
  - standalone conversion/utility functions
  - boundary and error conditions
  - BetRecord dataclass fields
  - BettingEngine.evaluate_bet (sharp filter, ev filter, stake clamping)
  - BettingEngine.evaluate_slate
  - BettingEngine.summary_stats (empty and non-empty)
"""

import math
import pytest

from src.betting.kelly import (
    american_to_decimal,
    decimal_to_prob,
    remove_vig_multiplicative,
    kelly_fraction,
    kelly_with_uncertainty,
    compute_clv,
    compute_ev,
    BetRecord,
    BettingEngine,
)


# ---------------------------------------------------------------------------
# american_to_decimal
# ---------------------------------------------------------------------------

class TestAmericanToDecimal:
    def test_positive_odds_plus_100(self):
        # +100 => (100/100) + 1 = 2.0
        assert american_to_decimal(100) == pytest.approx(2.0)

    def test_positive_odds_plus_200(self):
        # +200 => (200/100) + 1 = 3.0
        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_positive_odds_plus_150(self):
        # +150 => (150/100) + 1 = 2.5
        assert american_to_decimal(150) == pytest.approx(2.5)

    def test_negative_odds_minus_110(self):
        # -110 => (100/110) + 1 ≈ 1.9091
        assert american_to_decimal(-110) == pytest.approx(100 / 110 + 1, rel=1e-6)

    def test_negative_odds_minus_200(self):
        # -200 => (100/200) + 1 = 1.5
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_negative_odds_minus_300(self):
        # -300 => (100/300) + 1 ≈ 1.3333
        assert american_to_decimal(-300) == pytest.approx(100 / 300 + 1, rel=1e-6)

    def test_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            american_to_decimal(0)

    def test_large_positive(self):
        # +1000 => 11.0
        assert american_to_decimal(1000) == pytest.approx(11.0)

    def test_large_negative(self):
        # -1000 => 1.1
        assert american_to_decimal(-1000) == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# decimal_to_prob
# ---------------------------------------------------------------------------

class TestDecimalToProb:
    def test_decimal_2_gives_half(self):
        assert decimal_to_prob(2.0) == pytest.approx(0.5)

    def test_decimal_1_5(self):
        assert decimal_to_prob(1.5) == pytest.approx(1 / 1.5, rel=1e-9)

    def test_decimal_3_gives_one_third(self):
        assert decimal_to_prob(3.0) == pytest.approx(1 / 3, rel=1e-9)

    def test_exactly_one_raises(self):
        with pytest.raises(ValueError):
            decimal_to_prob(1.0)

    def test_less_than_one_raises(self):
        with pytest.raises(ValueError):
            decimal_to_prob(0.9)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            decimal_to_prob(0.0)


# ---------------------------------------------------------------------------
# remove_vig_multiplicative
# ---------------------------------------------------------------------------

class TestRemoveVigMultiplicative:
    def test_even_market_returns_half_half(self):
        # Both sides at 2.0 => after vig removal still 0.5 / 0.5
        home, away = remove_vig_multiplicative(2.0, 2.0)
        assert home == pytest.approx(0.5)
        assert away == pytest.approx(0.5)

    def test_probs_sum_to_one(self):
        home, away = remove_vig_multiplicative(1.909, 2.02)
        assert home + away == pytest.approx(1.0, rel=1e-9)

    def test_favorite_gets_higher_prob(self):
        # -110 / +100 market — home is favorite
        home_odds = american_to_decimal(-110)
        away_odds = american_to_decimal(100)
        home, away = remove_vig_multiplicative(home_odds, away_odds)
        assert home > away

    def test_known_values(self):
        # home_odds=1.8, away_odds=2.2
        # imp_home = 1/1.8, imp_away = 1/2.2
        # total = 1/1.8 + 1/2.2
        imp_home = 1 / 1.8
        imp_away = 1 / 2.2
        total = imp_home + imp_away
        expected_home = imp_home / total
        expected_away = imp_away / total
        h, a = remove_vig_multiplicative(1.8, 2.2)
        assert h == pytest.approx(expected_home, rel=1e-9)
        assert a == pytest.approx(expected_away, rel=1e-9)

    def test_home_odds_le_1_raises(self):
        with pytest.raises(ValueError):
            remove_vig_multiplicative(1.0, 2.0)

    def test_away_odds_le_1_raises(self):
        with pytest.raises(ValueError):
            remove_vig_multiplicative(2.0, 0.5)

    def test_both_odds_le_1_raises(self):
        with pytest.raises(ValueError):
            remove_vig_multiplicative(0.9, 0.8)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_zero_edge_returns_zero(self):
        # p=0.5, decimal=2.0 => b=1.0, kelly=(0.5*1 - 0.5)/1 = 0
        result = kelly_fraction(0.5, 2.0, fraction=0.25)
        assert result == pytest.approx(0.0)

    def test_positive_edge(self):
        # p=0.6, decimal=2.0, fraction=0.25
        # b=1.0, raw_kelly=(0.6-0.4)/1 = 0.2, stake=0.2*0.25=0.05
        result = kelly_fraction(0.6, 2.0, fraction=0.25)
        assert result == pytest.approx(0.05)

    def test_negative_edge_clamped_to_zero(self):
        # p=0.4, decimal=2.0 => raw_kelly = (0.4 - 0.6) / 1 = -0.2 => clamped to 0
        result = kelly_fraction(0.4, 2.0, fraction=0.25)
        assert result == pytest.approx(0.0)

    def test_high_probability_boundary(self):
        # p=0.999, decimal=1.5, fraction=0.25
        # b=0.5, kelly=(0.999*0.5 - 0.001)/0.5 = (0.4995-0.001)/0.5=0.4985/0.5=0.997
        # stake=0.997*0.25=0.24925
        b = 0.5
        raw = (0.999 * b - 0.001) / b
        expected = raw * 0.25
        assert kelly_fraction(0.999, 1.5, fraction=0.25) == pytest.approx(expected, rel=1e-6)

    def test_low_probability_boundary(self):
        # p=0.001, decimal=200.0 (huge underdog)
        # b=199.0, kelly=(0.001*199 - 0.999)/199
        b = 199.0
        raw = (0.001 * b - 0.999) / b
        expected = max(0.0, raw * 0.25)
        assert kelly_fraction(0.001, 200.0, fraction=0.25) == pytest.approx(expected, rel=1e-6)

    def test_near_minimum_odds(self):
        # decimal=1.01, p=0.99, fraction=0.5
        # b=0.01, kelly=(0.99*0.01 - 0.01)/0.01=(0.0099-0.01)/0.01=-0.01/0.01=-1 => 0
        result = kelly_fraction(0.99, 1.01, fraction=0.5)
        assert result == pytest.approx(0.0)

    def test_full_kelly_fraction_1(self):
        # fraction=1.0 should be allowed (full Kelly)
        result = kelly_fraction(0.6, 2.0, fraction=1.0)
        assert result == pytest.approx(0.2)

    def test_p_win_zero_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.0, 2.0)

    def test_p_win_one_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.0, 2.0)

    def test_p_win_negative_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(-0.1, 2.0)

    def test_p_win_above_one_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(1.1, 2.0)

    def test_fraction_zero_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 2.0, fraction=0.0)

    def test_fraction_above_one_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 2.0, fraction=1.1)

    def test_decimal_odds_le_1_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.6, 1.0)


# ---------------------------------------------------------------------------
# kelly_with_uncertainty
# ---------------------------------------------------------------------------

class TestKellyWithUncertainty:
    def test_zero_std_no_discount(self):
        # std=0 => discount factor = max(0, 1 - 0.5 * 0 * 10) = 1.0
        base = kelly_fraction(0.6, 2.0, fraction=0.25)
        result = kelly_with_uncertainty(0.6, 0.0, 2.0, fraction=0.25, uncertainty_discount=0.5)
        assert result == pytest.approx(base)

    def test_std_0_1_half_discount(self):
        # std=0.1 => discount = max(0, 1 - 0.5*0.1*10) = max(0, 0.5) = 0.5
        base = kelly_fraction(0.6, 2.0, fraction=0.25)
        result = kelly_with_uncertainty(0.6, 0.1, 2.0, fraction=0.25, uncertainty_discount=0.5)
        assert result == pytest.approx(base * 0.5, rel=1e-9)

    def test_large_std_clamps_to_zero(self):
        # std=0.3 => discount = max(0, 1 - 0.5*0.3*10) = max(0, -0.5) = 0
        result = kelly_with_uncertainty(0.6, 0.3, 2.0, fraction=0.25, uncertainty_discount=0.5)
        assert result == pytest.approx(0.0)

    def test_negative_std_raises(self):
        with pytest.raises(ValueError):
            kelly_with_uncertainty(0.6, -0.01, 2.0)

    def test_result_non_negative(self):
        result = kelly_with_uncertainty(0.55, 0.25, 2.1, fraction=0.25, uncertainty_discount=0.5)
        assert result >= 0.0

    def test_higher_uncertainty_reduces_stake(self):
        low_unc = kelly_with_uncertainty(0.6, 0.05, 2.0, fraction=0.25)
        high_unc = kelly_with_uncertainty(0.6, 0.15, 2.0, fraction=0.25)
        assert low_unc >= high_unc


# ---------------------------------------------------------------------------
# compute_clv
# ---------------------------------------------------------------------------

class TestComputeCLV:
    def test_positive_clv_when_bet_better(self):
        # bet at 0.52, closes at 0.50 => CLV = 0.02
        assert compute_clv(0.52, 0.50) == pytest.approx(0.02)

    def test_negative_clv_when_bet_worse(self):
        assert compute_clv(0.48, 0.50) == pytest.approx(-0.02)

    def test_zero_clv_exact_match(self):
        assert compute_clv(0.55, 0.55) == pytest.approx(0.0)

    def test_bet_prob_zero_raises(self):
        with pytest.raises(ValueError):
            compute_clv(0.0, 0.5)

    def test_bet_prob_one_raises(self):
        with pytest.raises(ValueError):
            compute_clv(1.0, 0.5)

    def test_close_prob_zero_raises(self):
        with pytest.raises(ValueError):
            compute_clv(0.5, 0.0)

    def test_close_prob_one_raises(self):
        with pytest.raises(ValueError):
            compute_clv(0.5, 1.0)

    def test_close_prob_negative_raises(self):
        with pytest.raises(ValueError):
            compute_clv(0.5, -0.1)


# ---------------------------------------------------------------------------
# compute_ev
# ---------------------------------------------------------------------------

class TestComputeEV:
    def test_even_market_zero_edge(self):
        # p=0.5, decimal=2.0 => EV = 0.5*1 - 0.5 = 0
        assert compute_ev(0.5, 2.0) == pytest.approx(0.0)

    def test_positive_ev(self):
        # p=0.6, decimal=2.0 => EV = 0.6 - 0.4 = 0.2
        assert compute_ev(0.6, 2.0) == pytest.approx(0.2)

    def test_negative_ev(self):
        # p=0.4, decimal=2.0 => EV = 0.4 - 0.6 = -0.2
        assert compute_ev(0.4, 2.0) == pytest.approx(-0.2)

    def test_p_win_zero_raises(self):
        with pytest.raises(ValueError):
            compute_ev(0.0, 2.0)

    def test_p_win_one_raises(self):
        with pytest.raises(ValueError):
            compute_ev(1.0, 2.0)

    def test_decimal_le_one_raises(self):
        with pytest.raises(ValueError):
            compute_ev(0.6, 1.0)

    def test_known_value_with_vig(self):
        # p=0.55, decimal=1.909
        expected = 0.55 * (1.909 - 1) - (1 - 0.55)
        assert compute_ev(0.55, 1.909) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# BetRecord dataclass
# ---------------------------------------------------------------------------

class TestBetRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            game_id="G001",
            bet_side="home",
            p_win_mean=0.6,
            p_win_std=0.05,
            open_american=-110,
            close_american=-115,
            kelly_stake=0.03,
            clv=0.01,
            ev=0.05,
            book="Pinnacle",
            sharp=True,
        )
        defaults.update(overrides)
        return BetRecord(**defaults)

    def test_default_sharp_true(self):
        rec = self._make_record()
        assert rec.sharp is True

    def test_fields_accessible(self):
        rec = self._make_record(game_id="TEST", bet_side="away", book="Circa")
        assert rec.game_id == "TEST"
        assert rec.bet_side == "away"
        assert rec.book == "Circa"

    def test_sharp_can_be_set_false(self):
        rec = self._make_record(sharp=False)
        assert rec.sharp is False

    def test_numeric_fields_stored_correctly(self):
        rec = self._make_record(p_win_mean=0.75, kelly_stake=0.04, clv=0.02, ev=0.10)
        assert rec.p_win_mean == pytest.approx(0.75)
        assert rec.kelly_stake == pytest.approx(0.04)
        assert rec.clv == pytest.approx(0.02)
        assert rec.ev == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# BettingEngine — construction
# ---------------------------------------------------------------------------

class TestBettingEngineInit:
    def test_default_parameters(self):
        engine = BettingEngine(bankroll=10000.0)
        assert engine.bankroll == pytest.approx(10000.0)
        assert engine.kelly_fraction == pytest.approx(0.25)
        assert engine.min_edge == pytest.approx(0.02)
        assert engine.max_stake_pct == pytest.approx(0.05)

    def test_sharp_books_frozenset(self):
        assert "Pinnacle" in BettingEngine.SHARP_BOOKS
        assert "Circa" in BettingEngine.SHARP_BOOKS
        assert "Bookmaker" in BettingEngine.SHARP_BOOKS
        assert "DraftKings" not in BettingEngine.SHARP_BOOKS

    def test_custom_parameters(self):
        engine = BettingEngine(bankroll=5000.0, kelly_fraction=0.5, min_edge=0.03, max_stake_pct=0.02)
        assert engine.bankroll == pytest.approx(5000.0)
        assert engine.kelly_fraction == pytest.approx(0.5)
        assert engine.min_edge == pytest.approx(0.03)
        assert engine.max_stake_pct == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# BettingEngine.evaluate_bet
# ---------------------------------------------------------------------------

class TestEvaluateBet:
    def setup_method(self):
        self.engine = BettingEngine(bankroll=10000.0, kelly_fraction=0.25, min_edge=0.02, max_stake_pct=0.05)

    def _call(self, **overrides):
        defaults = dict(
            game_id="G001",
            bet_side="home",
            p_win_mean=0.60,
            p_win_std=0.05,
            open_american=-110,
            close_american=-115,
            book="Pinnacle",
        )
        defaults.update(overrides)
        return self.engine.evaluate_bet(**defaults)

    def test_returns_none_for_non_sharp_book(self):
        result = self._call(book="DraftKings")
        assert result is None

    def test_returns_none_for_betmgm(self):
        result = self._call(book="BetMGM")
        assert result is None

    def test_returns_bet_record_for_pinnacle(self):
        result = self._call(book="Pinnacle")
        assert isinstance(result, BetRecord)

    def test_returns_bet_record_for_circa(self):
        result = self._call(book="Circa")
        assert isinstance(result, BetRecord)

    def test_returns_bet_record_for_bookmaker(self):
        result = self._call(book="Bookmaker")
        assert isinstance(result, BetRecord)

    def test_returns_none_when_ev_below_min_edge(self):
        # p=0.50, decimal=2.0 (even) => EV=0 < 0.02
        result = self._call(p_win_mean=0.50, open_american=100, close_american=100)
        assert result is None

    def test_stake_clamped_by_max_stake_pct(self):
        # Use a very large edge so raw kelly > max_stake_pct
        # p=0.99, -110 close => after vig removal, prob ~52%; p_win=0.99 is massive edge
        result = self._call(p_win_mean=0.99, p_win_std=0.01, close_american=-110)
        assert result is not None
        assert result.kelly_stake <= self.engine.max_stake_pct * self.engine.bankroll

    def test_stake_positive_for_positive_edge(self):
        result = self._call()
        if result is not None:
            assert result.kelly_stake > 0

    def test_sharp_flag_true_for_sharp_book(self):
        result = self._call(book="Pinnacle")
        if result is not None:
            assert result.sharp is True

    def test_game_id_propagated(self):
        result = self._call(game_id="GAME_XYZ", book="Pinnacle")
        if result is not None:
            assert result.game_id == "GAME_XYZ"

    def test_bet_side_propagated(self):
        result = self._call(bet_side="away", book="Pinnacle")
        if result is not None:
            assert result.bet_side == "away"

    def test_book_propagated(self):
        result = self._call(book="Circa")
        if result is not None:
            assert result.book == "Circa"

    def test_clv_field_is_float(self):
        result = self._call(book="Pinnacle")
        if result is not None:
            assert isinstance(result.clv, float)

    def test_ev_field_is_float(self):
        result = self._call(book="Pinnacle")
        if result is not None:
            assert isinstance(result.ev, float)


# ---------------------------------------------------------------------------
# BettingEngine.evaluate_slate
# ---------------------------------------------------------------------------

class TestEvaluateSlate:
    def setup_method(self):
        self.engine = BettingEngine(bankroll=10000.0, kelly_fraction=0.25, min_edge=0.02, max_stake_pct=0.05)

    def _sharp_game(self, game_id="G001", book="Pinnacle", p_win_mean=0.60):
        return dict(
            game_id=game_id,
            bet_side="home",
            p_win_mean=p_win_mean,
            p_win_std=0.05,
            open_american=-110,
            close_american=-115,
            book=book,
        )

    def test_empty_slate_returns_empty_list(self):
        assert self.engine.evaluate_slate([]) == []

    def test_non_sharp_books_filtered_out(self):
        games = [self._sharp_game(book="FanDuel"), self._sharp_game(book="DraftKings")]
        result = self.engine.evaluate_slate(games)
        assert result == []

    def test_mixed_books_returns_only_sharp(self):
        games = [
            self._sharp_game(game_id="G001", book="Pinnacle"),
            self._sharp_game(game_id="G002", book="FanDuel"),
        ]
        result = self.engine.evaluate_slate(games)
        sharp_ids = {r.game_id for r in result}
        assert "G002" not in sharp_ids

    def test_low_ev_games_excluded(self):
        # p=0.50, even market => EV=0
        games = [self._sharp_game(p_win_mean=0.50)]
        # Re-build with even odds so EV is definitely below threshold
        games[0]["open_american"] = 100
        games[0]["close_american"] = 100
        result = self.engine.evaluate_slate(games)
        assert result == []

    def test_returns_list_of_bet_records(self):
        games = [self._sharp_game(game_id="G001"), self._sharp_game(game_id="G002", book="Circa")]
        result = self.engine.evaluate_slate(games)
        for rec in result:
            assert isinstance(rec, BetRecord)


# ---------------------------------------------------------------------------
# BettingEngine.summary_stats
# ---------------------------------------------------------------------------

class TestSummaryStats:
    def setup_method(self):
        self.engine = BettingEngine(bankroll=10000.0)

    def _make_record(self, kelly_stake=100.0, ev=0.05, clv=0.02, sharp=True):
        return BetRecord(
            game_id="G001",
            bet_side="home",
            p_win_mean=0.6,
            p_win_std=0.05,
            open_american=-110,
            close_american=-115,
            kelly_stake=kelly_stake,
            clv=clv,
            ev=ev,
            book="Pinnacle",
            sharp=sharp,
        )

    def test_empty_records_returns_zeros(self):
        stats = self.engine.summary_stats([])
        assert stats["n_bets"] == 0
        assert stats["total_staked"] == pytest.approx(0.0)
        assert stats["mean_ev"] == pytest.approx(0.0)
        assert stats["mean_clv"] == pytest.approx(0.0)
        assert stats["mean_kelly"] == pytest.approx(0.0)
        assert stats["sharp_pct"] == pytest.approx(0.0)

    def test_single_record_stats(self):
        rec = self._make_record(kelly_stake=200.0, ev=0.10, clv=0.03, sharp=True)
        stats = self.engine.summary_stats([rec])
        assert stats["n_bets"] == 1
        assert stats["total_staked"] == pytest.approx(200.0)
        assert stats["mean_ev"] == pytest.approx(0.10)
        assert stats["mean_clv"] == pytest.approx(0.03)
        assert stats["mean_kelly"] == pytest.approx(200.0)
        assert stats["sharp_pct"] == pytest.approx(1.0)

    def test_all_keys_present(self):
        stats = self.engine.summary_stats([])
        assert set(stats.keys()) == {"n_bets", "total_staked", "mean_ev", "mean_clv", "mean_kelly", "sharp_pct"}

    def test_sharp_pct_mixed(self):
        records = [
            self._make_record(sharp=True),
            self._make_record(sharp=True),
            self._make_record(sharp=False),
        ]
        stats = self.engine.summary_stats(records)
        assert stats["sharp_pct"] == pytest.approx(2 / 3, rel=1e-9)

    def test_total_staked_sums_correctly(self):
        records = [self._make_record(kelly_stake=100.0), self._make_record(kelly_stake=250.0)]
        stats = self.engine.summary_stats(records)
        assert stats["total_staked"] == pytest.approx(350.0)

    def test_mean_ev_averages_correctly(self):
        records = [self._make_record(ev=0.10), self._make_record(ev=0.20)]
        stats = self.engine.summary_stats(records)
        assert stats["mean_ev"] == pytest.approx(0.15)

    def test_mean_clv_averages_correctly(self):
        records = [self._make_record(clv=0.01), self._make_record(clv=0.03)]
        stats = self.engine.summary_stats(records)
        assert stats["mean_clv"] == pytest.approx(0.02)

    def test_n_bets_count(self):
        records = [self._make_record() for _ in range(5)]
        stats = self.engine.summary_stats(records)
        assert stats["n_bets"] == 5

    def test_all_non_sharp_gives_zero_sharp_pct(self):
        records = [self._make_record(sharp=False), self._make_record(sharp=False)]
        stats = self.engine.summary_stats(records)
        assert stats["sharp_pct"] == pytest.approx(0.0)
