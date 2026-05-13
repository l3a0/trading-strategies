# pyright: reportUnknownMemberType=false
"""Unit tests for cc_backtest.py."""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Any

import numpy as np
import pytest

from cc_backtest import (
    bs_delta,
    bs_price,
    calc_rolling_volatility,
    classify_regime,
    compute_statistics,
    find_strike_for_delta,
    normal_cdf,
    normal_pdf,
    regime_analysis,
    run_cc_overlay,
    walk_forward_optimization,
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
            'risk_free_rate': 0.045,
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
            'risk_free_rate': 0.045,
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
            'risk_free_rate': 0.045,
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
            'risk_free_rate': 0.045,
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
            'risk_free_rate': 0.045,
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


# ====================
# compute_statistics
# ====================

def _build_daily_equity(
    equity_series: list[float],
    price_series: list[float],
) -> list[dict[str, Any]]:
    """Build a daily_equity payload in the shape compute_statistics expects."""
    dates = _fake_dates(len(equity_series))
    return [
        {'date': d, 'equity': e, 'price': p}
        for d, e, p in zip(dates, equity_series, price_series)
    ]


class TestComputeStatistics:
    """Statistical significance of the overlay vs. buy-and-hold."""

    def test_zero_excess_returns_give_zero_t_stat(self) -> None:
        """When overlay equity tracks buy-and-hold exactly, t-stat = 0."""
        # Flat prices @ $100 for 50 days. With num_contracts=1, cash=0,
        # bh_equity = 100 shares * $100 = $10,000 flat. Set overlay
        # equity to also be $10,000 flat → excess returns are all zero.
        prices = [100.0] * 50
        equity = [10_000.0] * 50
        daily_equity = _build_daily_equity(equity, prices)

        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        assert stats['t_stat_naive'] == pytest.approx(0.0, abs=1e-9)
        assert stats['t_stat_newey_west'] == pytest.approx(0.0, abs=1e-9)
        assert stats['ann_excess_return_pct'] == pytest.approx(0.0, abs=1e-9)
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False

    def test_constant_nonzero_excess_yields_zero_t_stat(self) -> None:
        """When excess returns are constant and non-zero, var = 0 → t_nw = 0.

        NOT redundant with test_zero_excess_returns_give_zero_t_stat: there,
        mean_e = 0 makes t_nw = 0 / se_nw = 0 regardless of se_nw, so the
        floor doesn't matter. Here mean_e ≠ 0, so se_nw is what determines
        the result. A previous implementation floored var_mean_nw at 1e-20,
        which gave se_nw = 1e-10 and t_nw = mean / 1e-10 (huge garbage).
        With the floor at 0, the se_nw > 0 guard correctly returns 0.
        """
        # Doubling each day → np.diff/equity[:-1] = 1.0 exactly (mul by 2
        # is bit-exact in float64), so excess = 1.0 every day → var_e = 0
        # exactly (no float noise). Flat prices → bh_ret = 0.
        n = 30
        prices = [100.0] * n
        equity = [float(2 ** i) for i in range(n)]
        daily_equity = _build_daily_equity(equity, prices)

        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        assert stats['t_stat_naive'] == pytest.approx(0.0, abs=1e-9)
        assert stats['t_stat_newey_west'] == pytest.approx(0.0, abs=1e-9)
        assert math.isfinite(stats['t_stat_newey_west'])

    def test_consistent_positive_excess_produces_positive_t_stat(self) -> None:
        """A consistent overlay advantage should produce a positive, large t-stat."""
        # Flat prices → bh_ret = 0 every day → excess = overlay_ret directly.
        # Build overlay equity that grows at a noisy positive rate.
        np.random.seed(42)
        n = 500
        prices = [100.0] * n
        daily_excess = 0.0005 + np.random.normal(0, 0.0003, n - 1)  # +5 bps/day noisy
        equity = [10_000.0]
        for r in daily_excess:
            equity.append(equity[-1] * (1 + r))
        daily_equity = _build_daily_equity(equity, prices)

        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        # With 500 days of +5 bps mean and 3 bps noise, t-stat should be very large
        assert stats['t_stat_naive'] > 10.0
        assert stats['t_stat_newey_west'] > 5.0  # NW slightly smaller due to any autocorrelation
        assert stats['ann_excess_return_pct'] > 10.0  # ~12.6% annualized
        assert stats['passes_t_2'] is True
        assert stats['passes_t_3'] is True

    def test_naive_t_stat_matches_formula(self) -> None:
        """t_naive should equal mean / (std / sqrt(n)) computed directly."""
        # Build a deterministic series with known mean and std
        np.random.seed(7)
        n = 252
        prices = [100.0] * n
        excess_returns = np.random.normal(0.0001, 0.001, n - 1)
        equity = [10_000.0]
        for r in excess_returns:
            equity.append(equity[-1] * (1 + r))
        daily_equity = _build_daily_equity(equity, prices)

        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        # Independently reconstruct excess returns and compute naive t-stat
        equity_arr = np.array(equity)
        overlay_ret = np.diff(equity_arr) / equity_arr[:-1]
        bh_ret = np.zeros_like(overlay_ret)  # flat prices
        excess = overlay_ret - bh_ret
        expected_t_naive = float(np.mean(excess)) / (float(np.std(excess, ddof=1)) / math.sqrt(len(excess)))

        assert stats['t_stat_naive'] == pytest.approx(expected_t_naive, abs=0.02)

    def test_passes_flags_reflect_newey_west_t_stat(self) -> None:
        """passes_t_2 and passes_t_3 should consistently reflect t_stat_newey_west."""
        np.random.seed(1)
        n = 1000
        prices = [100.0] * n
        excess = 0.001 + np.random.normal(0, 0.0002, n - 1)
        equity = [10_000.0]
        for r in excess:
            equity.append(equity[-1] * (1 + r))
        dail = _build_daily_equity(equity, prices)
        stats_strong = compute_statistics(dail, num_contracts=1, cash=0.0)

        assert stats_strong['passes_t_2'] == (abs(stats_strong['t_stat_newey_west']) > 2.0)
        assert stats_strong['passes_t_3'] == (abs(stats_strong['t_stat_newey_west']) > 3.0)

    def test_newey_west_lag_follows_andrews_rule(self) -> None:
        """NW lag should follow L = floor(4 * (n/100)^(2/9))."""
        n_equity = 2514
        prices = [100.0] * n_equity
        equity = [10_000.0 + i * 0.1 for i in range(n_equity)]
        daily_equity = _build_daily_equity(equity, prices)

        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        # n=2513 excess returns (one less than daily_equity length)
        n_returns = n_equity - 1
        expected_L = int(4 * (n_returns / 100) ** (2 / 9))
        assert stats['nw_lag'] == expected_L

    def test_raises_when_too_few_observations(self) -> None:
        """Need at least 2 daily returns (i.e., 3 equity points) for variance."""
        # Only 1 equity point → 0 returns → should raise
        daily_equity = _build_daily_equity([10_000.0], [100.0])
        with pytest.raises(ValueError, match="at least 2"):
            compute_statistics(daily_equity, num_contracts=1, cash=0.0)

    def test_includes_all_expected_keys(self) -> None:
        """Return dict should include all metrics the caller needs."""
        daily_equity = _build_daily_equity(
            [10_000.0 + i for i in range(100)],
            [100.0] * 100,
        )
        stats = compute_statistics(daily_equity, num_contracts=1, cash=0.0)

        expected_keys = {
            'n_days', 'years_of_data',
            'ann_excess_return_pct', 'ann_excess_vol_pct', 'sharpe_excess',
            't_stat_naive', 't_stat_newey_west', 'nw_lag',
            'passes_t_2', 'passes_t_3',
        }
        assert set(stats.keys()) == expected_keys

    def test_integrates_with_run_cc_overlay(self) -> None:
        """compute_statistics should consume run_cc_overlay output directly."""
        # 60 days of mild oscillation — generates real overlay activity
        prices = np.array(
            [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(60)],
            dtype=np.float64,
        )
        dates = _fake_dates(60)
        params: dict[str, float] = {
            'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
            'risk_free_rate': 0.045, 'capital': 100_000,
        }
        summary, _, daily_equity = run_cc_overlay(dates, prices, params)

        stats = compute_statistics(
            daily_equity,
            num_contracts=summary['num_contracts'],
            cash=summary['cash'],
        )

        # Basic sanity: all numeric outputs are finite
        assert math.isfinite(stats['t_stat_naive'])
        assert math.isfinite(stats['t_stat_newey_west'])
        assert math.isfinite(stats['sharpe_excess'])
        assert stats['n_days'] == len(daily_equity) - 1


# ====================
# Regression: bundled MSFT 10-year backtest
# ====================

# Resolve the CSV relative to this file so the test passes regardless of the
# directory pytest is invoked from.
_MSFT_CSV = os.path.join(os.path.dirname(__file__), 'msft_10yr_prices.csv')

# The parameters cc_backtest.py's __main__ uses and the tutorial documents.
_TUTORIAL_PARAMS: dict[str, float] = {
    'call_delta': 0.25,
    'close_at_pct': 0.75,
    'dte': 21,
    'risk_free_rate': 0.045,
    'capital': 100_000,
}

# Monte Carlo shuffle settings the tutorial reports its numbers for
# ("500 shuffles, seed=42").
_MC_SHUFFLES = 500
_MC_SEED = 42


def _load_msft_csv() -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
    """Mirror the CSV parser in cc_backtest.py's __main__ block."""
    dates: list[str] = []
    prices: list[float] = []
    with open(_MSFT_CSV) as f:
        for row in csv.reader(f):
            if not row or not row[0][:4].isdigit():
                continue
            dates.append(row[0])
            prices.append(float(row[1]))
    return dates, np.array(prices, dtype=np.float64)


class TestMsftTenYearRegression:
    """Pin the headline numbers the tutorial and README quote for the bundled
    MSFT data.

    These aren't "is the math correct" tests — TestRunCcOverlay and
    TestComputeStatistics cover correctness against synthetic fixtures. This
    class locks the *specific outputs* prose elsewhere in the repo cites, so an
    engine change that would silently move those numbers fails CI instead of
    leaving the docs quietly wrong.
    """

    @pytest.fixture(scope='class')
    def data(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
        return _load_msft_csv()

    @pytest.fixture(scope='class')
    def result(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dates, prices = data
        summary, _, daily_equity = run_cc_overlay(dates, prices, _TUTORIAL_PARAMS)
        stats = compute_statistics(
            daily_equity,
            num_contracts=summary['num_contracts'],
            cash=summary['cash'],
        )
        return summary, stats

    def test_capital_sizing(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """$100K sizes into 20 contracts: ~$95.6K stock + ~$4.4K cash."""
        summary, _ = result
        assert summary['num_contracts'] == 20
        assert summary['initial_stock_cost'] == pytest.approx(95_573.55, abs=0.5)
        assert summary['cash'] == pytest.approx(4_426.45, abs=0.5)

    def test_returns_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Buy-and-hold $746K (+646%) + net overlay $299K = overlay $1.045M (+945%)."""
        summary, _ = result
        assert summary['buy_hold_final'] == pytest.approx(746_166.44, abs=1.0)
        assert summary['buy_hold_return_pct'] == pytest.approx(646.17, abs=0.05)
        assert summary['net_overlay_pnl'] == pytest.approx(298_947.87, abs=1.0)
        assert summary['excess_return_pct'] == pytest.approx(298.95, abs=0.05)
        assert summary['final_equity'] == pytest.approx(1_045_114.31, abs=1.0)
        # The tutorial's headline "~945% total return on the bundled $100K config".
        assert summary['total_return_pct'] == pytest.approx(945.11, abs=0.05)

    def test_overlay_pnl_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """185 calls sold; ~$1.025M premium gross, ~$726K paid back in costs."""
        summary, _ = result
        assert summary['num_calls_sold'] == 185
        assert summary['total_premium_collected'] == pytest.approx(1_025_092.00, abs=5.0)
        assert summary['overlay_costs'] == pytest.approx(726_144.12, abs=5.0)

    def test_activity(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """~81% win rate, ~23% max drawdown."""
        summary, _ = result
        assert summary['win_rate'] == pytest.approx(81.0, abs=0.1)
        assert summary['max_drawdown_pct'] == pytest.approx(23.02, abs=0.05)

    def test_significance(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Sharpe 0.163, naive t=0.51, NW t=0.58 at L=8 — clears neither bar."""
        _, stats = result
        assert stats['n_days'] == 2514
        assert stats['years_of_data'] == pytest.approx(9.98, abs=0.005)
        assert stats['ann_excess_return_pct'] == pytest.approx(1.591, abs=0.001)
        assert stats['ann_excess_vol_pct'] == pytest.approx(9.79, abs=0.01)
        assert stats['sharpe_excess'] == pytest.approx(0.163, abs=0.001)
        assert stats['t_stat_naive'] == pytest.approx(0.51, abs=0.005)
        assert stats['t_stat_newey_west'] == pytest.approx(0.58, abs=0.005)
        assert stats['nw_lag'] == 8
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False

    @pytest.mark.parametrize(
        ('param', 'offsets_and_returns'),
        [
            # call_delta sweep: base 0.25 ± offset → total_return_pct.
            # Tutorial (rounded for display): -0.10:882%  -0.05:861%  base:945%
            #                                 +0.05:925%  +0.10:899%
            (
                'call_delta',
                [(-0.10, 881.53), (-0.05, 861.36), (0.0, 945.11),
                 (0.05, 925.27), (0.10, 898.97)],
            ),
            # close_at_pct sweep: base 0.75 ± offset → total_return_pct.
            # Tutorial: -0.20:882%  -0.10:984%  base:945%  +0.10:965%  +0.20:895%
            (
                'close_at_pct',
                [(-0.20, 882.29), (-0.10, 984.25), (0.0, 945.11),
                 (0.10, 965.33), (0.20, 895.26)],
            ),
        ],
    )
    def test_sensitivity_perturbations(
        self,
        data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],
        param: str,
        offsets_and_returns: list[tuple[float, float]],
    ) -> None:
        """Perturbing one param at a time reproduces the tutorial's sweep numbers.

        Sensitivity analysis: for each parameter, vary it by a fixed
        offset from base and measure impact on total return. *High*
        sensitivity (large swings under small perturbations) suggests
        overfitting — the optimum is a knife edge rather than a plateau.
        A robust strategy should stay in a similar range across the sweep.

        Each variant runs the full overlay with all params held fixed
        except the one being perturbed:
          - call_delta sweep at ±0.05 / ±0.10 from base=0.25
          - close_at_pct sweep at ±0.10 / ±0.20 from base=0.75

        A real sensitivity helper would also skip invalid parameter
        values (negative call_delta, non-positive dte, close_at_pct ≤ 0
        or > 1) before running the backtest; none of the sweeps below
        hit those edges, so this test doesn't bother.

        These pin both the individual returns and the "robust" verdict:
        the worst drop from base stays single-digit-percent of the base
        return — the "Swing" interpretation in the tutorial's example
        output ("robust" if the swing is small relative to base).
        """
        dates, prices = data
        base = _TUTORIAL_PARAMS[param]
        returns: list[float] = []
        for offset, expected in offsets_and_returns:
            # Hold all params fixed except the one being perturbed; only
            # `param` shifts by `offset` from its base value. The expected
            # total_return_pct is pinned per offset so a regression here
            # surfaces as a test failure rather than a silent drift in
            # the tutorial's worked example.
            test_params = {**_TUTORIAL_PARAMS, param: base + offset}
            summary, _, _ = run_cc_overlay(dates, prices, test_params)
            assert summary['total_return_pct'] == pytest.approx(expected, abs=0.5)
            returns.append(summary['total_return_pct'])
        base_return = next(
            r for (off, _), r in zip(offsets_and_returns, returns) if off == 0.0
        )
        # Worst drop from base, as a percentage of base. Single-digit-%
        # means the strategy isn't fragile to this parameter — the
        # "robust" verdict in the tutorial. Double-digit % drops would
        # indicate the chosen value is a knife-edge optimum.
        worst_drop_pct = (base_return - min(returns)) / base_return * 100
        assert worst_drop_pct < 10.0  # "robust": single-digit-percent drop

    def test_monte_carlo_shuffle(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Reproduce the tutorial's Monte Carlo shuffle (500 paths, seed=42).

        Monte Carlo randomization test by shuffling daily returns.

        Algorithm:
            1. Calculate daily returns from actual prices.
            2. Run real backtest (baseline).
            3. For each shuffle: randomize return order, rebuild prices,
               backtest.
            4. Calculate percentile of real return vs the MC distribution.

        Why this works: real prices have a specific *order* — trends,
        mean reversion, volatility clusters. Shuffling destroys that
        order while keeping the exact same set of daily returns (same
        mean, same volatility, same distribution). If the strategy
        profits on both real and shuffled paths, it's capturing
        statistical *properties* of the returns and those survive
        shuffling. If it only works on the real path, it was exploiting
        the specific *sequence* — overfitting or luck.

        On the bundled MSFT data the real ordered path beats every
        shuffled path (percentile 100), with mc_mean ~654% and the best
        shuffled path ~934% — the overlay exploits real price structure,
        not just the return distribution. This is the slowest test in
        the suite (~500 backtests, a couple of seconds).
        """
        dates, prices = data

        # Run baseline (real backtest) for comparison.
        real_summary, _, _ = run_cc_overlay(dates, prices, _TUTORIAL_PARAMS)
        real_return = real_summary['total_return_pct']

        # Calculate daily returns from the real price series.
        daily_returns = [
            float((prices[i] - prices[i - 1]) / prices[i - 1])
            for i in range(1, len(prices))
        ]

        rng = random.Random(_MC_SEED)
        mc_returns: list[float] = []

        for _ in range(_MC_SHUFFLES):
            # Shuffle returns (preserves distribution, changes sequence).
            shuffled = daily_returns.copy()
            rng.shuffle(shuffled)

            # Rebuild a price series from the shuffled returns:
            # start at the original first price, then chain-multiply
            # each return. `synthetic[-1]` grabs the last price in the
            # list so far, so each new price builds on the previous one
            # (just like real prices). `(1 + ret)` converts a return
            # into a price multiplier:
            #   ret=+0.02 → 1.02 (up 2%)
            #   ret=-0.01 → 0.99 (down 1%)
            #   ret=  0   → 1.00 (flat)
            # e.g., price[0]=100, returns=[+2%, -1%, +3%]
            #   → 100 → 100*1.02=102 → 102*0.99=100.98 → 100.98*1.03=104.01
            # Same set of daily moves, different order → different price path.
            synthetic = [float(prices[0])]
            for ret in shuffled:
                synthetic.append(synthetic[-1] * (1 + ret))

            # Run backtest on the synthetic prices.
            # Some shuffled paths can blow up inside the backtest —
            # common causes:
            #   - Log of zero/negative price: large negative returns
            #     can compound a small price to zero or below, crashing
            #     np.log() in the volatility calculation.
            #   - Division by zero: a flat price stretch → stdev=0 →
            #     Black-Scholes divides by volatility.
            #   - Black-Scholes edge cases: extreme strikes or near-zero
            #     time to expiry produce NaN/Inf in option-pricing math.
            # A few failed shuffles out of hundreds don't affect the
            # distribution, so we skip them and keep going. With seed=42
            # on this data, none of the 500 shuffles blow up.
            try:
                mc_summary, _, _ = run_cc_overlay(
                    dates, np.array(synthetic, dtype=np.float64), _TUTORIAL_PARAMS
                )
                mc_returns.append(mc_summary['total_return_pct'])
            except Exception:
                continue

        assert len(mc_returns) == _MC_SHUFFLES  # no path blew up at this seed

        # Percentile: what % of random shuffles did our real strategy beat?
        #
        # Step 1: count how many MC returns are worse than the real return.
        #   e.g., real_return=945, mc_returns=[800, 900, 1100, 700, 850]
        #   worse = 4 (we beat 800, 900, 700, 850 — all except 1100)
        #
        # Step 2: convert to a percentile.
        #   percentile = 100 * 4 / 5 = 80
        #   → "Our strategy beat 80% of random shuffles"
        #
        # High percentile (80+) = strategy is genuinely good, not lucky.
        # Low percentile (~30) = random ordering does just as well,
        #   suggesting returns came from the market, not the strategy.
        worse = sum(1 for r in mc_returns if r < real_return)
        percentile = int(100 * worse / len(mc_returns))
        mc_mean = sum(mc_returns) / len(mc_returns)

        assert real_return == pytest.approx(945.11, abs=0.05)
        assert percentile == 100  # real path beats every shuffle
        assert mc_mean == pytest.approx(654.0, abs=2.0)
        assert max(mc_returns) == pytest.approx(934.0, abs=2.0)

    def test_classify_regime_thresholds(self) -> None:
        """classify_regime returns the right label at each band edge.

        Flat history at 100 for 200 days → SMA = 100. With the
        default ±5% threshold:
          - price > 105   → 'bull'
          - price < 95    → 'bear'
          - 95 ≤ p ≤ 105  → 'sideways'
          - <200 prices   → 'unknown'

        The edge case 'price exactly equal to threshold' stays
        sideways (strict inequalities for bull/bear).
        """
        base = [100.0] * 200

        assert classify_regime(base + [106.0]) == 'bull'
        assert classify_regime(base + [94.0]) == 'bear'
        assert classify_regime(base + [100.0]) == 'sideways'
        # Boundary: strict inequalities, so equal-to-threshold stays sideways
        assert classify_regime(base + [105.0]) == 'sideways'
        assert classify_regime(base + [95.0]) == 'sideways'
        # Insufficient history
        assert classify_regime([100.0] * 50) == 'unknown'
        assert classify_regime([]) == 'unknown'

    def test_regime_analysis(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Aggregate overlay PnL by bull/bear/sideways/unknown regime.

        Classifies each day with a trailing-200-day SMA at ±5% bands;
        the first 199 days are 'unknown' because the SMA needs 200
        observations to compute. Each closed trade's pnl is bucketed
        into the regime active on its close date — no future peeking.

        Empirical observation on the bundled MSFT data: most of the
        overlay's premium income comes from days the SMA classifies
        as *bear* or *sideways*, not bull. Bull days dominate the
        day count (1,690 of 2,515) but contribute only ~$18K of
        trade pnl; bear days are ~280 but contribute ~$152K, because
        premium is richest where volatility is highest.
        """
        dates, prices = data
        _, trades, _ = run_cc_overlay(dates, prices, _TUTORIAL_PARAMS)
        result = regime_analysis(dates, prices, trades)

        # Day counts: SMA200 with ±5% bands on MSFT 2016-2026 produces
        # this exact split. Total equals len(prices) by construction.
        # The first `window` (200) days are 'unknown' because the SMA needs
        # 200 prior observations to compute (we classify using prices[:i],
        # so day 200 is the first one with enough history).
        assert result['bull']['days'] == 1690
        assert result['bear']['days'] == 279
        assert result['sideways']['days'] == 346
        assert result['unknown']['days'] == 200
        total_days = sum(result[r]['days'] for r in result)
        assert total_days == len(prices)

        # Per-regime trade PnL — the tutorial's headline empirical claim.
        assert result['bull']['total_pnl'] == pytest.approx(57976.42, abs=5.0)
        assert result['bear']['total_pnl'] == pytest.approx(96619.30, abs=5.0)
        assert result['sideways']['total_pnl'] == pytest.approx(139165.45, abs=5.0)
        assert result['unknown']['total_pnl'] == pytest.approx(5456.15, abs=5.0)

        # Bear and sideways' per-day averages dwarf bull's — premium
        # is richest in volatile and choppy regimes. Specifically:
        # ~$346/day in bear and ~$402/day in sideways vs ~$34/day
        # in bull, i.e. ~10× higher.
        assert result['bear']['avg_pnl_per_day'] > 8 * result['bull']['avg_pnl_per_day']
        assert result['sideways']['avg_pnl_per_day'] > 8 * result['bull']['avg_pnl_per_day']

    def test_walk_forward_optimization(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin the walk-forward result on MSFT 2016-2026.

        With the tutorial's standard 3×3×3 grid (call_delta ∈
        {0.15, 0.20, 0.25}, dte ∈ {21, 30, 45}, close_at_pct ∈
        {0.50, 0.75, 1.00}), a 2-year train / 6-month test / 6-month
        roll schedule produces 15 walk-forward periods over the
        2018-04 → 2025-10 out-of-sample span.

        Empirical observations pinned here:
          - The familiar __main__ defaults (0.25Δ, 21 DTE) win the
            most periods, but **close_at_pct=0.50 wins more periods
            than the 0.75 default** — closing earlier frees capital
            and skips the last sliver of theta that gamma often
            eats.
          - Cumulative OOS compound return (per-period 6mo returns
            chained) is ~510% over 7.5 years — substantially less
            than the ~582% fixed-params return over the same span.
            That gap is the cost of not having hindsight; the
            walk-forward number is the return you'd have actually
            achieved running this strategy in real time.

        Total runtime is a couple of seconds (15 windows × 27
        combos = 405 train backtests on 504-day windows).
        """
        from collections import Counter

        dates, prices = data
        param_grid: dict[str, list[float]] = {
            'call_delta': [0.15, 0.20, 0.25],
            'dte': [21, 30, 45],
            'close_at_pct': [0.50, 0.75, 1.00],
        }
        oos_equity, records = walk_forward_optimization(dates, prices, param_grid)

        # Window structure: 2y train, 6mo test, 6mo roll → 15 periods on
        # a 10y MSFT dataset starting 2016-04.
        assert len(records) == 15
        assert len(oos_equity) == 1887  # daily OOS equity points across all periods

        # train_end == test_start by construction (half-open intervals).
        for r in records:
            assert r['train_end'] == r['test_start']

        # First and last period bounds.
        assert records[0]['test_start'] == '2018-04-11'
        assert records[0]['test_end'] == '2018-10-11'
        assert records[-1]['test_start'] == '2025-04-11'
        assert records[-1]['test_end'] == '2025-10-11'

        # Most-chosen params: 0.25Δ and 21 DTE win consistently;
        # close_at_pct=0.50 wins more periods than the 0.75 default.
        delta_counts = Counter(r['best_params']['call_delta'] for r in records)
        dte_counts = Counter(r['best_params']['dte'] for r in records)
        close_counts = Counter(r['best_params']['close_at_pct'] for r in records)
        assert delta_counts[0.25] == 13
        assert delta_counts[0.20] == 2
        assert delta_counts[0.15] == 0
        assert dte_counts[21] == 10
        assert dte_counts[30] == 4
        assert dte_counts[45] == 1
        assert close_counts[0.50] == 8
        assert close_counts[0.75] == 6
        assert close_counts[1.00] == 1

        # Cumulative OOS compound return: chain per-period 6mo returns.
        cumulative = 1.0
        for r in records:
            period_eq = [
                d for d in oos_equity
                if r['test_start'] <= d['date'] < r['test_end']
            ]
            assert period_eq, f"no OOS equity for period {r['test_start']}"
            period_ret = (period_eq[-1]['equity'] - period_eq[0]['equity']) / period_eq[0]['equity']
            cumulative *= (1.0 + period_ret)
        cumulative_pct = (cumulative - 1.0) * 100
        # Pinned around ~510%, allow a few pp of slack for floating-point
        # variation in the run-to-run results.
        assert cumulative_pct == pytest.approx(510.0, abs=5.0)
