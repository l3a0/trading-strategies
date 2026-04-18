# pyright: reportUnknownMemberType=false
"""Unit tests for cc_backtest.py."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from cc_backtest import (
    bs_delta,
    bs_price,
    calc_rolling_volatility,
    find_strike_for_delta,
    normal_cdf,
    normal_pdf,
    run_cc_overlay,
)


# ====================
# normal_pdf
# ====================

class TestNormalPdf:
    def test_peak_at_zero(self) -> None:
        """PDF peaks at x=0 with value 1/sqrt(2π)."""
        assert normal_pdf(0) == pytest.approx(1.0 / math.sqrt(2 * math.pi), abs=1e-10)

    def test_symmetric(self) -> None:
        """PDF is symmetric: pdf(x) == pdf(-x)."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert normal_pdf(x) == pytest.approx(normal_pdf(-x), abs=1e-12)

    def test_tails_approach_zero(self) -> None:
        """PDF approaches 0 far from center."""
        assert normal_pdf(5.0) < 1e-5
        assert normal_pdf(-5.0) < 1e-5

    def test_known_values(self) -> None:
        """Check against known standard normal PDF values."""
        assert normal_pdf(1.0) == pytest.approx(0.24197, abs=1e-4)
        assert normal_pdf(2.0) == pytest.approx(0.05399, abs=1e-4)


# ====================
# normal_cdf
# ====================

class TestNormalCdf:
    def test_center(self) -> None:
        """CDF(0) = 0.5 by symmetry."""
        assert normal_cdf(0) == pytest.approx(0.5, abs=1e-7)

    def test_known_quantiles(self) -> None:
        """Check standard z-score quantiles."""
        assert normal_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
        assert normal_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)
        assert normal_cdf(1.0) == pytest.approx(0.8413, abs=1e-3)
        assert normal_cdf(-1.0) == pytest.approx(0.1587, abs=1e-3)

    def test_symmetry(self) -> None:
        """CDF(x) + CDF(-x) = 1."""
        for x in [0.5, 1.0, 1.5, 2.0, 3.0]:
            assert normal_cdf(x) + normal_cdf(-x) == pytest.approx(1.0, abs=1e-7)

    def test_monotonic(self) -> None:
        """CDF is strictly increasing."""
        xs = [-3, -2, -1, 0, 1, 2, 3]
        cdfs = [normal_cdf(x) for x in xs]
        for i in range(len(cdfs) - 1):
            assert cdfs[i] < cdfs[i + 1]

    def test_bounds(self) -> None:
        """CDF is bounded by [0, 1]."""
        assert 0 <= normal_cdf(-10) < 0.001
        assert 0.999 < normal_cdf(10) <= 1.0


# ====================
# bs_price
# ====================

class TestBsPrice:
    def test_put_call_parity(self) -> None:
        """Put-call parity: C - P = S - K*e^(-rT)."""
        S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, option_type='call')
        put = bs_price(S, K, T, r, sigma, option_type='put')
        expected = S - K * math.exp(-r * T)
        assert call - put == pytest.approx(expected, abs=1e-6)

    def test_put_call_parity_otm(self) -> None:
        """Put-call parity holds for OTM options too."""
        S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.04, 0.30
        call = bs_price(S, K, T, r, sigma, option_type='call')
        put = bs_price(S, K, T, r, sigma, option_type='put')
        expected = S - K * math.exp(-r * T)
        assert call - put == pytest.approx(expected, abs=1e-6)

    def test_call_positive(self) -> None:
        """Call price is always positive."""
        assert bs_price(100, 100, 0.25, 0.05, 0.20, option_type='call') > 0

    def test_put_positive(self) -> None:
        """Put price is always positive."""
        assert bs_price(100, 100, 0.25, 0.05, 0.20, option_type='put') > 0

    def test_higher_vol_higher_price(self) -> None:
        """Higher volatility produces higher option prices."""
        S, K, T, r = 100.0, 105.0, 0.25, 0.04
        low_vol = bs_price(S, K, T, r, 0.15, option_type='call')
        high_vol = bs_price(S, K, T, r, 0.40, option_type='call')
        assert high_vol > low_vol

    def test_longer_expiry_higher_price(self) -> None:
        """Longer time to expiry produces higher call price (no dividends)."""
        S, K, r, sigma = 100.0, 105.0, 0.04, 0.20
        short = bs_price(S, K, 0.1, r, sigma, option_type='call')
        long = bs_price(S, K, 1.0, r, sigma, option_type='call')
        assert long > short

    def test_deep_itm_call_near_intrinsic(self) -> None:
        """Deep ITM call ≈ S - K*e^(-rT) (intrinsic value)."""
        S, K, T, r, sigma = 200.0, 100.0, 0.25, 0.04, 0.20
        call = bs_price(S, K, T, r, sigma, option_type='call')
        intrinsic = S - K * math.exp(-r * T)
        assert call == pytest.approx(intrinsic, rel=0.01)

    def test_deep_otm_call_near_zero(self) -> None:
        """Deep OTM call is near zero."""
        call = bs_price(100, 200, 0.1, 0.04, 0.20, option_type='call')
        assert call < 0.01


# ====================
# bs_delta
# ====================

class TestBsDelta:
    def test_call_delta_range(self) -> None:
        """Call delta is between 0 and 1."""
        delta = bs_delta(100, 100, 0.25, 0.04, 0.20, option_type='call')
        assert 0 < delta < 1

    def test_put_delta_range(self) -> None:
        """Put delta is between -1 and 0."""
        delta = bs_delta(100, 100, 0.25, 0.04, 0.20, option_type='put')
        assert -1 < delta < 0

    def test_atm_call_delta_near_half(self) -> None:
        """ATM call delta ≈ 0.5 (above 0.5 due to drift term r + σ²/2)."""
        delta = bs_delta(100, 100, 0.25, 0.04, 0.20, option_type='call')
        assert delta == pytest.approx(0.5, abs=0.10)

    def test_put_call_delta_relationship(self) -> None:
        """Call delta - Put delta = 1."""
        S, K, T, r, sigma = 100.0, 105.0, 0.25, 0.04, 0.25
        call_d = bs_delta(S, K, T, r, sigma, option_type='call')
        put_d = bs_delta(S, K, T, r, sigma, option_type='put')
        assert call_d - put_d == pytest.approx(1.0, abs=1e-10)

    def test_deep_itm_call_delta_near_one(self) -> None:
        """Deep ITM call has delta near 1."""
        delta = bs_delta(200, 100, 0.25, 0.04, 0.20, option_type='call')
        assert delta > 0.99

    def test_deep_otm_call_delta_near_zero(self) -> None:
        """Deep OTM call has delta near 0."""
        delta = bs_delta(100, 200, 0.25, 0.04, 0.20, option_type='call')
        assert delta < 0.01

    def test_higher_strike_lower_call_delta(self) -> None:
        """Higher strike means lower call delta."""
        d1 = bs_delta(100, 100, 0.25, 0.04, 0.20, option_type='call')
        d2 = bs_delta(100, 110, 0.25, 0.04, 0.20, option_type='call')
        assert d1 > d2


# ====================
# find_strike_for_delta
# ====================

class TestFindStrikeForDelta:
    def test_call_strike_above_spot(self) -> None:
        """For OTM call (delta < 0.5), strike should be above spot."""
        strike = find_strike_for_delta(100, 30/252, 0.04, 0.20, 0.25, option_type='call')
        assert strike > 100

    def test_put_strike_below_spot(self) -> None:
        """For OTM put (delta > -0.5), strike should be below spot."""
        strike = find_strike_for_delta(100, 30/252, 0.04, 0.20, -0.25, option_type='put')
        assert strike < 100

    def test_returns_whole_dollar(self) -> None:
        """Strike should be a whole dollar amount."""
        strike = find_strike_for_delta(100, 30/252, 0.04, 0.20, 0.25, option_type='call')
        assert strike == int(strike)

    def test_delta_close_to_target(self) -> None:
        """The delta at the found strike should be close to the target."""
        S, T, r, sigma = 100.0, 30/252, 0.04, 0.20
        target = 0.25
        strike = find_strike_for_delta(S, T, r, sigma, target, option_type='call')
        actual_delta = bs_delta(S, strike, T, r, sigma, option_type='call')
        assert actual_delta == pytest.approx(target, abs=0.05)

    def test_lower_delta_higher_strike(self) -> None:
        """Lower target delta should produce a higher call strike (further OTM)."""
        S, T, r, sigma = 100.0, 30/252, 0.04, 0.20
        strike_25 = find_strike_for_delta(S, T, r, sigma, 0.25, option_type='call')
        strike_15 = find_strike_for_delta(S, T, r, sigma, 0.15, option_type='call')
        assert strike_15 >= strike_25


# ====================
# calc_rolling_volatility
# ====================

class TestCalcRollingVolatility:
    def test_output_length(self) -> None:
        """Output length = len(prices) - 1 (number of returns)."""
        prices = np.array([100.0 + i for i in range(50)], dtype=np.float64)
        vols = calc_rolling_volatility(prices, window=10)
        assert len(vols) == len(prices) - 1

    def test_nan_warmup(self) -> None:
        """First (window-1) values should be NaN."""
        prices = np.array([100.0 + i * 0.5 for i in range(50)])
        vols = calc_rolling_volatility(prices, window=10)
        for i in range(9):
            assert np.isnan(vols[i])
        assert not np.isnan(vols[9])

    def test_positive_volatility(self) -> None:
        """Volatility should be positive for non-constant prices."""
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
        vols = calc_rolling_volatility(prices, window=20)
        valid = vols[~np.isnan(vols)]
        assert all(v > 0 for v in valid)

    def test_constant_prices_zero_vol(self) -> None:
        """Constant prices produce zero volatility."""
        prices = np.full(50, 100.0)
        vols = calc_rolling_volatility(prices, window=10)
        valid = vols[~np.isnan(vols)]
        assert all(v == 0 for v in valid)


# ====================
# run_cc_overlay
# ====================

class TestRunCcOverlay:
    @pytest.fixture()
    def flat_market(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:  # pyright: ignore[reportUnknownParameterType]
        """A flat market: 500 days around $100 with realistic noise."""
        # Fake dates: 31 strings ("2020-01-01"..."2020-01-31") repeated 17×, sliced to 500.
        # run_cc_overlay treats them as opaque labels, so duplicates/non-trading days are fine.
        dates = [f"2020-01-{i:02d}" for i in range(1, 32)] * 17
        dates = dates[:500]
        # Simulate realistic flat market with ~15% annualized vol
        np.random.seed(0)
        daily_vol = 0.15 / math.sqrt(252)
        returns = np.random.normal(0, daily_vol, 499)
        prices = np.zeros(500)
        prices[0] = 100.0
        for i in range(1, 500):
            prices[i] = prices[i-1] * (1 + returns[i-1])
        return dates, prices

    @pytest.fixture()
    def rising_market(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:  # pyright: ignore[reportUnknownParameterType]
        """A steadily rising market over 500 days."""
        # Fake dates: 31 strings ("2020-01-01"..."2020-01-31") repeated 17×, sliced to 500.
        # run_cc_overlay treats them as opaque labels, so duplicates are fine.
        dates = [f"2020-01-{i:02d}" for i in range(1, 32)] * 17
        dates = dates[:500]
        np.random.seed(1)
        daily_returns = np.random.normal(0.001, 0.01, 499)
        prices = np.zeros(500)
        prices[0] = 50.0
        for i in range(1, 500):
            prices[i] = prices[i-1] * (1 + daily_returns[i-1])
        return dates, prices

    @pytest.fixture()
    def default_params(self) -> dict[str, float]:
        return {
            'call_delta': 0.25,
            'close_at_pct': 0.75,
            'dte': 21,
            'risk_free_rate': 0.045,
            'iv_multiplier': 1.3,
        }

    def test_returns_three_items(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """run_cc_overlay returns (summary, trades, daily_equity)."""
        dates, prices = rising_market
        result = run_cc_overlay(dates, prices, default_params)
        assert len(result) == 3

    def test_summary_keys(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Summary dict contains expected keys."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        expected_keys = {
            'capital', 'num_contracts', 'initial_stock_cost', 'cash',
            'final_equity', 'total_return_pct',
            'buy_hold_final', 'buy_hold_return_pct', 'excess_return_pct',
            'net_overlay_pnl', 'total_premium_collected', 'overlay_costs',
            'premium_retention_pct',
            'num_calls_sold', 'wins', 'losses', 'win_rate', 'max_drawdown_pct',
        }
        assert set(summary.keys()) == expected_keys

    def test_calls_sold_positive(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Should sell at least one call over 500 days."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert summary['num_calls_sold'] > 0

    def test_premium_collected_positive(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Total premium collected should be positive."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert summary['total_premium_collected'] > 0

    def test_wins_plus_losses_match_closed_trades(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """wins + losses = number of closed/expired trades."""
        dates, prices = rising_market
        summary, trades, _ = run_cc_overlay(dates, prices, default_params)
        closed = sum(1 for t in trades if t['action'] in ('close', 'close_itm', 'expiration'))
        assert summary['wins'] + summary['losses'] == closed

    def test_daily_equity_length(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Daily equity has one entry per day after warmup."""
        dates, prices = rising_market
        _, _, daily_equity = run_cc_overlay(dates, prices, default_params)
        # All days produce a daily_equity entry: the warmup uses a 0.20
        # vol fallback for day_idx < 3, so no days are skipped.
        # (Days where net_premium <= 0 would skip, but rising market has vol > 0.)
        assert len(daily_equity) == len(dates)

    def test_default_is_single_contract(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """With no `capital` param, default to 1 contract; capital = first price × 100."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert summary['num_contracts'] == 1
        assert summary['capital'] == pytest.approx(prices[0] * 100, abs=0.01)
        assert summary['cash'] == pytest.approx(0.0, abs=0.01)

    def test_flat_market_positive_premium(self, flat_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """In a flat market, CC overlay should collect positive premiums."""
        dates, prices = flat_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert summary['total_premium_collected'] > 0

    def test_max_drawdown_non_negative(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Max drawdown should be >= 0."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert summary['max_drawdown_pct'] >= 0

    def test_win_rate_bounded(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Win rate should be between 0 and 100."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        assert 0 <= summary['win_rate'] <= 100

    def test_equity_includes_stock_value(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Final equity should reflect stock appreciation, not just premiums."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        final_stock_value = prices[-1] * 100
        # Final equity should be in the neighborhood of stock value
        # (could be slightly above due to premiums or below due to assignment costs)
        assert summary['final_equity'] > final_stock_value * 0.5

    def test_buy_hold_matches_final_price(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """buy_hold_final = final price × shares + cash."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        shares = 100 * summary['num_contracts']
        expected = prices[-1] * shares + summary['cash']
        assert summary['buy_hold_final'] == pytest.approx(expected, abs=0.01)

    def test_excess_return_is_difference(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """excess_return_pct = total_return_pct - buy_hold_return_pct."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        expected = summary['total_return_pct'] - summary['buy_hold_return_pct']
        assert summary['excess_return_pct'] == pytest.approx(expected, abs=0.01)

    def test_net_overlay_pnl_equals_excess_dollars(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """net_overlay_pnl = final_equity - buy_hold_final (the dollar gap)."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        expected = summary['final_equity'] - summary['buy_hold_final']
        assert summary['net_overlay_pnl'] == pytest.approx(expected, abs=0.01)

    def test_overlay_costs_equation(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """gross_premium - overlay_costs = net_overlay_pnl."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        expected = summary['total_premium_collected'] - summary['overlay_costs']
        assert summary['net_overlay_pnl'] == pytest.approx(expected, abs=0.01)

    def test_premium_retention_pct(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """premium_retention_pct = net_overlay_pnl / gross_premium × 100."""
        dates, prices = rising_market
        summary, _, _ = run_cc_overlay(dates, prices, default_params)
        if summary['total_premium_collected'] > 0:
            expected = summary['net_overlay_pnl'] / summary['total_premium_collected'] * 100
            assert summary['premium_retention_pct'] == pytest.approx(expected, abs=0.1)

    def test_trade_actions_valid(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """All trade actions should be one of the expected values."""
        dates, prices = rising_market
        _, trades, _ = run_cc_overlay(dates, prices, default_params)
        valid_actions = {'sell', 'close', 'close_itm', 'expiration'}
        for trade in trades:
            assert trade['action'] in valid_actions

    def test_capital_sizes_into_whole_contracts(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """capital=$10K with $50 stock → 2 contracts ($10K stock + ~$0 cash)."""
        dates, prices = rising_market  # rising_market starts at $50
        params = {**default_params, 'capital': 10_000.0}
        summary, _, _ = run_cc_overlay(dates, prices, params)
        contract_cost = prices[0] * 100
        assert summary['num_contracts'] == int(10_000 // contract_cost)
        assert summary['initial_stock_cost'] == pytest.approx(
            summary['num_contracts'] * 100 * prices[0], abs=0.01
        )
        assert summary['cash'] == pytest.approx(10_000 - summary['initial_stock_cost'], abs=0.01)

    def test_capital_scales_premium_collected(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """N contracts collect ~N× the premium of 1 contract on the same path."""
        dates, prices = rising_market
        single, _, _ = run_cc_overlay(dates, prices, default_params)
        params = {**default_params, 'capital': 10_000.0}
        multi, _, _ = run_cc_overlay(dates, prices, params)
        ratio = multi['total_premium_collected'] / single['total_premium_collected']
        assert ratio == pytest.approx(multi['num_contracts'], rel=0.01)

    def test_capital_too_small_raises(self, rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]], default_params: dict[str, float]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """Capital below 1 contract should raise ValueError."""
        dates, prices = rising_market
        params = {**default_params, 'capital': 100.0}  # nowhere near 1 contract
        with pytest.raises(ValueError, match="insufficient for 1 contract"):
            run_cc_overlay(dates, prices, params)


# ====================
# Scenario tests: simulate specific trade flows and verify outcomes
# ====================

def _fake_dates(n: int) -> list[str]:
    """Generate n opaque date labels."""
    base = [f"2020-01-{i:02d}" for i in range(1, 32)]
    return (base * ((n // len(base)) + 1))[:n]


class TestScenarioFlatMarket:
    """Flat prices below strike → call expires OTM, full premium kept."""

    @pytest.fixture()
    def setup(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]], dict[str, float]]:  # pyright: ignore[reportUnknownParameterType]
        # 50 days of small oscillation around $100 to establish vol > 0
        # but never approach a 0.25Δ call strike (~$105+).
        prices = np.array(
            [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(50)],
            dtype=np.float64,
        )
        dates = _fake_dates(50)
        params: dict[str, float] = {
            'call_delta': 0.25,
            'close_at_pct': 0.75,
            'dte': 21,
            'risk_free_rate': 0.045,
            'iv_multiplier': 1.3,
        }
        return dates, prices, params

    def test_first_action_is_sell(self, setup: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]], dict[str, float]]) -> None:  # pyright: ignore[reportUnknownParameterType]
        dates, prices, params = setup
        _, trades, _ = run_cc_overlay(dates, prices, params)
        assert trades[0]['action'] == 'sell'

    def test_strike_is_above_spot(self, setup: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]], dict[str, float]]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """0.25Δ call should have strike above current price."""
        dates, prices, params = setup
        _, trades, _ = run_cc_overlay(dates, prices, params)
        first_sell = trades[0]
        assert first_sell['strike'] > first_sell['price']

    def test_premium_positive(self, setup: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]], dict[str, float]]) -> None:  # pyright: ignore[reportUnknownParameterType]
        dates, prices, params = setup
        _, trades, _ = run_cc_overlay(dates, prices, params)
        assert trades[0]['premium'] > 0

    def test_expiration_keeps_full_premium(self, setup: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]], dict[str, float]]) -> None:  # pyright: ignore[reportUnknownParameterType]
        """When call expires OTM, pnl = premium * 100 (no buyback cost)."""
        dates, prices, params = setup
        # Override close_at_pct to disable early profit-target close;
        # otherwise the call hits 75% profit before expiration in flat markets.
        params = {**params, 'close_at_pct': 1.0}
        _, trades, _ = run_cc_overlay(dates, prices, params)
        sell = trades[0]
        expiration = next(t for t in trades[1:] if t['action'] == 'expiration')
        # Premium kept fully → pnl = premium_per_share * 100
        expected_pnl = sell['premium'] * 100
        assert expiration['pnl'] == pytest.approx(expected_pnl, abs=1e-6)


class TestScenarioCalledAway:
    """Stock rallies past strike before expiration → assignment, lose upside."""

    def test_assignment_pnl(self) -> None:
        # Build prices: 22 flat days for warmup, then rally past strike
        # Day 0-21: prices around $100 with small variation (vol > 0)
        # Day 22 (expiration of first 21 DTE call sold on day 1): jump to $130
        warmup = [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(22)]
        # Days 22+: jump to a price clearly above any 0.25Δ strike
        rally = [130.0] * 25
        prices = np.array(warmup + rally, dtype=np.float64)
        dates = _fake_dates(len(prices))
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'iv_multiplier': 1.3,
        }
        _, trades, _ = run_cc_overlay(dates, prices, params)

        sell = trades[0]
        # First non-sell trade after the sell
        outcome = next(t for t in trades[1:] if t['action'] in ('expiration', 'close', 'close_itm'))

        if outcome['action'] == 'expiration':
            # Called away: pnl = (premium - (price - strike)) * 100
            expected = (sell['premium'] - (outcome['price'] - sell['strike'])) * 100
            assert outcome['pnl'] == pytest.approx(expected, abs=1e-6)
            # Stock rallied above strike, so we should be losing money on the overlay
            assert outcome['pnl'] < 0


class TestScenarioProfitTargetClose:
    """Sharp price drop after sell → call value collapses → close at target."""

    def test_close_at_profit_target(self) -> None:
        # 22 days of flat-ish prices to establish vol and sell a call
        warmup = [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(22)]
        # Then crash to $70 — way below strike, call value → ~0
        crash = [70.0] * 10
        prices = np.array(warmup + crash, dtype=np.float64)
        dates = _fake_dates(len(prices))
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'iv_multiplier': 1.3,
        }
        _, trades, _ = run_cc_overlay(dates, prices, params)

        sell = trades[0]
        close = next(t for t in trades[1:] if t['action'] == 'close')
        # Call should be near zero, so we capture nearly all the premium
        assert close['call_value'] < sell['premium'] * 0.25
        # Profit > 75% of premium
        assert close['profit_pct'] > 0.75


class TestScenarioMultipleCycles:
    """Long flat run should produce multiple sell-expire cycles."""

    def test_multiple_sells(self) -> None:
        # 100 days of variation around $100 → expect ~4-5 cycles (21 DTE each)
        prices = np.array(
            [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(100)],
            dtype=np.float64,
        )
        dates = _fake_dates(100)
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'iv_multiplier': 1.3,
        }
        summary, trades, _ = run_cc_overlay(dates, prices, params)

        sells = [t for t in trades if t['action'] == 'sell']
        closes_or_expirations = [t for t in trades if t['action'] in ('close', 'close_itm', 'expiration')]

        # Should have at least 3 cycles in 100 days with 21 DTE
        assert len(sells) >= 3
        # Sells and closes should be roughly balanced (within 1, since last position may be open)
        assert abs(len(sells) - len(closes_or_expirations)) <= 1
        # All trades resolved should be wins (flat market → premium kept each time)
        assert summary['win_rate'] == pytest.approx(100.0, abs=1e-6)


class TestScenarioPnlAccumulation:
    """Sum of individual trade pnls should equal final realized_pnl."""

    def test_pnl_sums_match(self) -> None:
        prices = np.array(
            [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(80)],
            dtype=np.float64,
        )
        dates = _fake_dates(80)
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'iv_multiplier': 1.3,
        }
        _, trades, _ = run_cc_overlay(dates, prices, params)

        # Sum pnls from trades that close a position (sell trades have pnl=0)
        total_pnl = sum(t['pnl'] for t in trades if t['action'] in ('close', 'close_itm', 'expiration'))
        # The last trade with realized_pnl should match this sum
        last_with_realized = [t for t in trades if t['action'] in ('close', 'close_itm', 'expiration')][-1]
        assert last_with_realized['realized_pnl'] == pytest.approx(total_pnl, abs=1e-6)


class TestScenarioEquityFinalState:
    """Final equity = stock value + cumulative overlay P&L."""

    def test_equity_decomposition(self) -> None:
        prices = np.array(
            [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(60)],
            dtype=np.float64,
        )
        dates = _fake_dates(60)
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'iv_multiplier': 1.3,
        }
        summary, trades, daily_equity = run_cc_overlay(dates, prices, params)

        final_price = float(prices[-1])
        final_stock_value = final_price * 100

        # Cumulative overlay P&L from closed trades
        realized = sum(t['pnl'] for t in trades if t['action'] in ('close', 'close_itm', 'expiration'))

        # Final equity = stock value + realized overlay P&L (no open position assumed)
        # Allow for unrealized P&L on any still-open position
        last_action = trades[-1]['action'] if trades else None
        if last_action == 'sell':
            # Position still open: equity includes unrealized P&L
            # Just verify equity is in the right ballpark
            assert summary['final_equity'] > final_stock_value * 0.9
        else:
            # No open position: equity should exactly equal stock + realized
            expected_equity = final_stock_value + realized
            assert summary['final_equity'] == pytest.approx(expected_equity, abs=0.01)

        # daily_equity[-1] equity should match summary
        assert daily_equity[-1]['equity'] == summary['final_equity']
