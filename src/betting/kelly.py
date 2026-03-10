"""
src/betting/kelly.py

Kelly Criterion stake sizing and CLV computation for the NCAA March Madness
ST-GNN project.

Design constraints (from CLAUDE.md):
  - Sharp books only: CLV is pegged to Pinnacle, Circa, or Bookmaker closing
    lines — never consensus or retail lines.
  - Fractional Kelly: fraction parameter must be < 1.0 in practice (<=1.0
    technically allowed as "full Kelly" but callers should pass < 1.0).
  - PIT integrity: all inputs are pre-assembled upstream; this module never
    fetches external data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Standalone conversion / utility functions
# ---------------------------------------------------------------------------

def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal odds.

    Parameters
    ----------
    american:
        American odds integer.  Must not be zero.

    Returns
    -------
    float
        Decimal odds (always > 1.0 for valid American odds).

    Raises
    ------
    ValueError
        If ``american == 0``.
    """
    if american == 0:
        raise ValueError("American odds cannot be zero.")
    if american > 0:
        return (american / 100) + 1.0
    # negative
    return (100 / abs(american)) + 1.0


def decimal_to_prob(decimal_odds: float) -> float:
    """Vig-inclusive implied probability from decimal odds.

    Parameters
    ----------
    decimal_odds:
        Decimal representation of odds.  Must be strictly greater than 1.0.

    Returns
    -------
    float
        Implied probability in (0, 1).

    Raises
    ------
    ValueError
        If ``decimal_odds <= 1.0``.
    """
    if decimal_odds <= 1.0:
        raise ValueError(
            f"decimal_odds must be > 1.0, got {decimal_odds}."
        )
    return 1.0 / decimal_odds


def remove_vig_multiplicative(
    home_odds: float,
    away_odds: float,
) -> tuple[float, float]:
    """Multiplicative vig removal — more accurate than additive for sharp books.

    Parameters
    ----------
    home_odds, away_odds:
        Decimal odds for each side.  Both must be strictly greater than 1.0.

    Returns
    -------
    tuple[float, float]
        ``(home_true_prob, away_true_prob)`` that sum to exactly 1.0.

    Raises
    ------
    ValueError
        If either odds value is <= 1.0.
    """
    if home_odds <= 1.0:
        raise ValueError(
            f"home_odds must be > 1.0, got {home_odds}."
        )
    if away_odds <= 1.0:
        raise ValueError(
            f"away_odds must be > 1.0, got {away_odds}."
        )
    home_imp = 1.0 / home_odds
    away_imp = 1.0 / away_odds
    total = home_imp + away_imp
    if total <= 0:
        raise ValueError(
            f"Sum of implied probabilities must be > 0, got {total}."
        )
    return (home_imp / total, away_imp / total)


def kelly_fraction(
    p_win: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly stake as a proportion of bankroll.

    Parameters
    ----------
    p_win:
        True win probability, strictly in (0, 1).
    decimal_odds:
        Decimal odds on offer, strictly > 1.0.
    fraction:
        Kelly multiplier in (0, 1].  Default 0.25 (quarter Kelly).

    Returns
    -------
    float
        Stake as fraction of bankroll, clamped to [0, fraction].

    Raises
    ------
    ValueError
        If any input is out of range.
    """
    if not (0.0 < p_win < 1.0):
        raise ValueError(
            f"p_win must be in (0, 1), got {p_win}."
        )
    if decimal_odds <= 1.0:
        raise ValueError(
            f"decimal_odds must be > 1.0, got {decimal_odds}."
        )
    if not (0.0 < fraction <= 1.0):
        raise ValueError(
            f"fraction must be in (0, 1], got {fraction}."
        )
    b = decimal_odds - 1.0          # net profit per unit staked
    raw_kelly = (p_win * b - (1.0 - p_win)) / b
    return max(0.0, raw_kelly * fraction)


def kelly_with_uncertainty(
    p_win_mean: float,
    p_win_std: float,
    decimal_odds: float,
    fraction: float = 0.25,
    uncertainty_discount: float = 0.5,
) -> float:
    """Kelly stake adjusted for posterior uncertainty.

    The discount factor shrinks the raw Kelly stake proportionally to the
    standard deviation of the posterior win-probability distribution, so that
    wider posteriors produce smaller (more conservative) stakes.

    Discount rule::

        discount = max(0.0, 1.0 - uncertainty_discount * p_win_std * 10)

    A ``p_win_std`` of 0.10 with ``uncertainty_discount=0.5`` halves the stake.

    Parameters
    ----------
    p_win_mean:
        Posterior mean win probability.
    p_win_std:
        Posterior standard deviation.  Must be >= 0.
    decimal_odds:
        Decimal odds, strictly > 1.0.
    fraction:
        Kelly multiplier passed through to :func:`kelly_fraction`.
    uncertainty_discount:
        Scaling coefficient for the std-based discount.  Default 0.5.

    Returns
    -------
    float
        Discounted Kelly stake fraction of bankroll.

    Raises
    ------
    ValueError
        If ``p_win_std < 0``, or if any value passed to :func:`kelly_fraction`
        is invalid.
    """
    if p_win_std < 0.0:
        raise ValueError(
            f"p_win_std must be >= 0, got {p_win_std}."
        )
    base_kelly = kelly_fraction(p_win_mean, decimal_odds, fraction)
    discount_factor = max(0.0, 1.0 - uncertainty_discount * p_win_std * 10.0)
    return base_kelly * discount_factor


def compute_clv(bet_prob: float, close_prob: float) -> float:
    """Closing Line Value in probability space (sharp books only).

    CLV = bet_prob - close_prob

    Positive CLV means you obtained a better price than the sharp closing line.

    Parameters
    ----------
    bet_prob:
        Implied probability at time of bet (from the opening / bet price).
        Must be in (0, 1).
    close_prob:
        Implied probability at sharp-book close.  Must be in (0, 1).

    Returns
    -------
    float
        CLV; positive = edge retained at close.

    Raises
    ------
    ValueError
        If either probability is outside (0, 1).
    """
    for name, val in (("bet_prob", bet_prob), ("close_prob", close_prob)):
        if not (0.0 < val < 1.0):
            raise ValueError(
                f"{name} must be in (0, 1), got {val}."
            )
    return bet_prob - close_prob


def compute_ev(p_win: float, decimal_odds: float) -> float:
    """Expected value per unit staked.

    EV = p_win * (decimal_odds - 1) - (1 - p_win)

    Parameters
    ----------
    p_win:
        True win probability, strictly in (0, 1).
    decimal_odds:
        Decimal odds, strictly > 1.0.

    Returns
    -------
    float
        Expected profit per unit staked (positive = +EV).

    Raises
    ------
    ValueError
        If ``p_win`` not in (0, 1) or ``decimal_odds <= 1.0``.
    """
    if not (0.0 < p_win < 1.0):
        raise ValueError(
            f"p_win must be in (0, 1), got {p_win}."
        )
    if decimal_odds <= 1.0:
        raise ValueError(
            f"decimal_odds must be > 1.0, got {decimal_odds}."
        )
    return p_win * (decimal_odds - 1.0) - (1.0 - p_win)


# ---------------------------------------------------------------------------
# BetRecord dataclass
# ---------------------------------------------------------------------------

@dataclass
class BetRecord:
    """Immutable record for a single evaluated bet.

    Attributes
    ----------
    game_id:
        Unique identifier for the game.
    bet_side:
        ``"home"`` or ``"away"``.
    p_win_mean:
        Posterior mean win probability from the Bayesian model.
    p_win_std:
        Posterior standard deviation of win probability.
    open_american:
        Opening line in American odds format.
    close_american:
        Closing line from the sharp book in American odds format.
    kelly_stake:
        Dollar amount to stake (already bankroll-scaled and clamped).
    clv:
        Closing Line Value in probability space.
    ev:
        Expected value per unit staked.
    book:
        Name of the book providing the line.
    sharp:
        True if ``book`` is a recognised sharp book
        (Pinnacle / Circa / Bookmaker).
    """

    game_id: str
    bet_side: str
    p_win_mean: float
    p_win_std: float
    open_american: int
    close_american: int
    kelly_stake: float
    clv: float
    ev: float
    book: str
    sharp: bool = True


# ---------------------------------------------------------------------------
# BettingEngine
# ---------------------------------------------------------------------------

class BettingEngine:
    """Evaluate bets and manage Kelly sizing against a bankroll.

    Only bets on sharp books (Pinnacle, Circa, Bookmaker) pass the filter.
    Bets below the ``min_edge`` EV threshold are rejected.
    Stakes are clamped to ``max_stake_pct`` of the bankroll.

    Parameters
    ----------
    bankroll:
        Current bankroll in dollars.
    kelly_fraction:
        Fractional Kelly multiplier passed to :func:`kelly_with_uncertainty`.
    min_edge:
        Minimum EV (per unit) required to place a bet.  Default 0.02.
    max_stake_pct:
        Hard cap on stake as a fraction of bankroll.  Default 0.05 (5 %).
    """

    SHARP_BOOKS: frozenset[str] = frozenset({"Pinnacle", "Circa", "Bookmaker"})

    def __init__(
        self,
        bankroll: float,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.02,
        max_stake_pct: float = 0.05,
    ) -> None:
        self.bankroll = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_stake_pct = max_stake_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_bet(
        self,
        game_id: str,
        bet_side: str,
        p_win_mean: float,
        p_win_std: float,
        open_american: int,
        close_american: int,
        book: str,
    ) -> BetRecord | None:
        """Evaluate a single candidate bet.

        Returns ``None`` if:
        - ``book`` is not in :attr:`SHARP_BOOKS`, or
        - the computed EV is below :attr:`min_edge`.

        The stake is computed via :func:`kelly_with_uncertainty` and clamped
        to ``max_stake_pct * bankroll``.

        Parameters
        ----------
        game_id:
            Unique game identifier.
        bet_side:
            ``"home"`` or ``"away"``.
        p_win_mean:
            Posterior mean win probability.
        p_win_std:
            Posterior std of win probability.
        open_american:
            Opening American odds.
        close_american:
            Closing American odds from the sharp book.
        book:
            Sportsbook name.

        Returns
        -------
        BetRecord or None
        """
        # Sharp-book filter
        if book not in self.SHARP_BOOKS:
            return None

        # Convert closing line to decimal and remove vig to get true probs
        open_dec = american_to_decimal(open_american)
        close_dec = american_to_decimal(close_american)

        # Compute bet-side implied probability from opening line (vig-inclusive)
        # and closing implied probability from close line (vig-inclusive).
        # We use the single-side decimal_to_prob here because we are measuring
        # CLV relative to the same side's closing price.
        bet_imp_prob = decimal_to_prob(open_dec)
        close_imp_prob = decimal_to_prob(close_dec)

        # CLV
        try:
            clv = compute_clv(bet_imp_prob, close_imp_prob)
        except ValueError:
            clv = 0.0

        # EV — evaluated at closing odds against posterior mean
        ev = compute_ev(p_win_mean, close_dec)

        # Edge filter
        if ev < self.min_edge:
            return None

        # Kelly stake (as proportion of bankroll)
        kelly_prop = kelly_with_uncertainty(
            p_win_mean=p_win_mean,
            p_win_std=p_win_std,
            decimal_odds=close_dec,
            fraction=self.kelly_fraction,
        )

        # Scale to dollars and apply hard cap
        raw_stake = kelly_prop * self.bankroll
        max_stake = self.max_stake_pct * self.bankroll
        stake = min(raw_stake, max_stake)

        return BetRecord(
            game_id=game_id,
            bet_side=bet_side,
            p_win_mean=p_win_mean,
            p_win_std=p_win_std,
            open_american=open_american,
            close_american=close_american,
            kelly_stake=stake,
            clv=clv,
            ev=ev,
            book=book,
            sharp=True,
        )

    def evaluate_slate(self, games: list[dict]) -> list[BetRecord]:
        """Evaluate a list of candidate bets.

        Parameters
        ----------
        games:
            List of dicts with keys: ``game_id``, ``bet_side``,
            ``p_win_mean``, ``p_win_std``, ``open_american``,
            ``close_american``, ``book``.

        Returns
        -------
        list[BetRecord]
            Only bets that pass the sharp-book and min-edge filters.
        """
        results: list[BetRecord] = []
        for g in games:
            record = self.evaluate_bet(
                game_id=g["game_id"],
                bet_side=g["bet_side"],
                p_win_mean=g["p_win_mean"],
                p_win_std=g["p_win_std"],
                open_american=g["open_american"],
                close_american=g["close_american"],
                book=g["book"],
            )
            if record is not None:
                results.append(record)
        return results

    def summary_stats(self, records: list[BetRecord]) -> dict:
        """Compute aggregate statistics over a list of BetRecords.

        Parameters
        ----------
        records:
            List of evaluated bet records.

        Returns
        -------
        dict with keys:
            ``n_bets``, ``total_staked``, ``mean_ev``, ``mean_clv``,
            ``mean_kelly``, ``sharp_pct``.
            All numeric values are 0.0 when ``records`` is empty.
        """
        if not records:
            return {
                "n_bets": 0,
                "total_staked": 0.0,
                "mean_ev": 0.0,
                "mean_clv": 0.0,
                "mean_kelly": 0.0,
                "sharp_pct": 0.0,
            }

        n = len(records)
        total_staked = sum(r.kelly_stake for r in records)
        mean_ev = sum(r.ev for r in records) / n
        mean_clv = sum(r.clv for r in records) / n
        mean_kelly = sum(r.kelly_stake for r in records) / n
        sharp_count = sum(1 for r in records if r.sharp)
        sharp_pct = sharp_count / n

        return {
            "n_bets": n,
            "total_staked": total_staked,
            "mean_ev": mean_ev,
            "mean_clv": mean_clv,
            "mean_kelly": mean_kelly,
            "sharp_pct": sharp_pct,
        }
