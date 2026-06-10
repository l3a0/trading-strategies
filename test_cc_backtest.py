# pyright: reportUnknownMemberType=false
"""Unit tests for cc_backtest.py."""

from __future__ import annotations

import csv
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cc_backtest import (
    _param_combinations,
    bs_delta,
    bs_price,
    calc_rolling_volatility,
    classify_regime,
    compute_statistics,
    degrees_of_freedom,
    find_strike_for_delta,
    monte_carlo_shuffle,
    normal_cdf,
    normal_pdf,
    regime_analysis,
    run_cc_overlay,
    sensitivity_analysis,
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

        # daily_equity's final row equity should match summary
        assert daily_equity['equity'].iloc[-1] == summary['final_equity']


# ====================
# compute_statistics
# ====================

def _build_daily_equity(
    equity_series: list[float],
    price_series: list[float],
) -> pd.DataFrame:
    """Build a daily_equity payload in the shape compute_statistics expects."""
    dates = _fake_dates(len(equity_series))
    return pd.DataFrame(
        {'date': dates, 'equity': equity_series, 'price': price_series}
    )


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
# Risk-managed (delta-hedged) covered call — Israelov & Nielsen (2015)
# ====================


class TestRiskManagedCoveredCall:
    """The `delta_hedge=True` mode adds extra long stock each day to keep the portfolio's net
    delta pinned at the buy-and-hold equivalent, stripping out the equity-timing exposure.
    Conceptually it should:
      - leave the trade flow unchanged (same calls sold, same premium collected),
      - shrink excess-return variance materially (the variance source is exactly what we
        hedged out), and
      - reduce to the naive backtest when the flag is False or absent.
    """

    @pytest.fixture()
    def rising_market(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:  # pyright: ignore[reportUnknownParameterType]
        """A steadily rising market over 500 days (mirrors TestRunCcOverlay's fixture)."""
        dates = _fake_dates(500)
        np.random.seed(1)
        daily_returns = np.random.normal(0.001, 0.01, 499)
        prices = np.zeros(500)
        prices[0] = 50.0
        for i in range(1, 500):
            prices[i] = prices[i - 1] * (1 + daily_returns[i - 1])
        return dates, prices

    @pytest.fixture()
    def base_params(self) -> dict[str, float]:
        return {
            'call_delta': 0.25,
            'close_at_pct': 0.75,
            'dte': 21,
            'risk_free_rate': 0.045,
            'capital': 100_000,
        }

    def test_default_is_naive(
        self,
        rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],  # pyright: ignore[reportUnknownParameterType]
        base_params: dict[str, float],
    ) -> None:
        """Omitting the flag must produce the exact naive trajectory — pure backwards compat."""
        dates, prices = rising_market
        no_flag, _, eq_no_flag = run_cc_overlay(dates, prices, base_params)
        explicit_off, _, eq_off = run_cc_overlay(dates, prices, {**base_params, 'delta_hedge': 0.0})
        assert no_flag['final_equity'] == pytest.approx(explicit_off['final_equity'], abs=0.01)
        assert no_flag['total_premium_collected'] == pytest.approx(
            explicit_off['total_premium_collected'], abs=0.01
        )
        # And the daily equity curves must match day for day.
        assert (eq_no_flag['equity'].to_numpy() == eq_off['equity'].to_numpy()).all()

    def test_trade_flow_identical_to_naive(
        self,
        rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],  # pyright: ignore[reportUnknownParameterType]
        base_params: dict[str, float],
    ) -> None:
        """Hedging shares does NOT change which calls get sold or when they close — only the
        equity curve. Same calls sold, same gross premium, same closes/expirations."""
        dates, prices = rising_market
        naive_summary, naive_trades, _ = run_cc_overlay(dates, prices, base_params)
        hedge_summary, hedge_trades, _ = run_cc_overlay(
            dates, prices, {**base_params, 'delta_hedge': 1.0}
        )

        assert hedge_summary['num_calls_sold'] == naive_summary['num_calls_sold']
        assert hedge_summary['total_premium_collected'] == pytest.approx(
            naive_summary['total_premium_collected'], abs=0.01
        )
        # Trade sequence (actions, strikes, dates) must match — the call leg is unchanged.
        naive_actions = [(t['date'], t['action'], t.get('strike')) for t in naive_trades]
        hedge_actions = [(t['date'], t['action'], t.get('strike')) for t in hedge_trades]
        assert naive_actions == hedge_actions

    def test_summary_cash_is_initial_not_working(
        self,
        rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],  # pyright: ignore[reportUnknownParameterType]
        base_params: dict[str, float],
    ) -> None:
        """summary['cash'] must report the *initial* idle cash even under delta_hedge=True so
        compute_statistics' buy-and-hold reconstruction (shares × prices + cash) stays correct."""
        dates, prices = rising_market
        naive_summary, _, _ = run_cc_overlay(dates, prices, base_params)
        hedge_summary, _, _ = run_cc_overlay(dates, prices, {**base_params, 'delta_hedge': 1.0})
        # Both should report the same initial leftover cash.
        assert hedge_summary['cash'] == pytest.approx(naive_summary['cash'], abs=0.01)

    def test_excess_variance_shrinks_under_hedge(
        self,
        rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],  # pyright: ignore[reportUnknownParameterType]
        base_params: dict[str, float],
    ) -> None:
        """The whole point: hedging out the call's delta should cut excess-return variance.
        If this test fails the hedge isn't actually pinning net delta."""
        dates, prices = rising_market
        naive_summary, _, naive_eq = run_cc_overlay(dates, prices, base_params)
        hedge_summary, _, hedge_eq = run_cc_overlay(
            dates, prices, {**base_params, 'delta_hedge': 1.0}
        )
        naive_stats = compute_statistics(
            naive_eq, num_contracts=naive_summary['num_contracts'], cash=naive_summary['cash']
        )
        hedge_stats = compute_statistics(
            hedge_eq, num_contracts=hedge_summary['num_contracts'], cash=hedge_summary['cash']
        )
        # Excess vol should drop substantially — the variance source we hedged away is large
        # relative to whatever variance remains (theta plus residual gamma).
        assert hedge_stats['ann_excess_vol_pct'] < 0.75 * naive_stats['ann_excess_vol_pct']

    def test_hedge_curve_differs_from_naive(
        self,
        rising_market: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],  # pyright: ignore[reportUnknownParameterType]
        base_params: dict[str, float],
    ) -> None:
        """A whole-market backtest with delta_hedge=True must produce a different equity curve
        than the naive run — proves the hedge actually moved money."""
        dates, prices = rising_market
        _, _, naive_eq = run_cc_overlay(dates, prices, base_params)
        _, _, hedge_eq = run_cc_overlay(dates, prices, {**base_params, 'delta_hedge': 1.0})

        # Curves should differ on most days. Tolerate a few coincidences (e.g. days with no
        # open position) but require substantial divergence overall.
        diffs = np.abs(naive_eq['equity'].to_numpy() - hedge_eq['equity'].to_numpy())
        assert (diffs > 0.01).sum() > 0.5 * len(diffs)


class TestMsftRiskManagedRegression:
    """Pin the headline numbers the tutorial quotes for the risk-managed MSFT backtest.

    Locks the side-by-side comparison that appears in Part 5's risk-managed subsection so any
    engine change that would silently shift those numbers surfaces in CI.
    """

    @pytest.fixture(scope='class')
    def data(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
        return _load_msft_csv()

    @pytest.fixture(scope='class')
    def result(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dates, prices = data
        hedge_params: dict[str, float] = {**_TUTORIAL_PARAMS, 'delta_hedge': 1.0}
        summary, _, daily_equity = run_cc_overlay(dates, prices, hedge_params)
        stats = compute_statistics(
            daily_equity, num_contracts=summary['num_contracts'], cash=summary['cash']
        )
        return summary, stats

    def test_capital_sizing_matches_naive(
        self, result: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Base position is unchanged by hedging: same $100K → 20 contracts, same idle cash.

        The hedge adds *extra* shares funded by the working cash account; the buy-and-hold
        baseline (base shares + initial cash) is identical to the naive run, which is why
        summary['cash'] still reports the initial $4,426.45 leftover.
        """
        summary, _ = result
        assert summary['num_contracts'] == 20
        assert summary['initial_stock_cost'] == pytest.approx(95_573.55, abs=0.5)
        assert summary['cash'] == pytest.approx(4_426.45, abs=0.5)
        assert summary['buy_hold_final'] == pytest.approx(746_166.44, abs=1.0)
        assert summary['buy_hold_return_pct'] == pytest.approx(646.17, abs=0.05)

    def test_returns_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Hedged dollar uplift: net overlay $304K (vs naive $268K) → overlay $1.050M (+950%).

        Same buy-and-hold $746K base, but stripping the equity-timing wiggle lifts net
        overlay P&L from $268,424.87 (naive) to $303,717.73 and the overlay's total return
        from +914.59% to +949.88%.
        """
        summary, _ = result
        assert summary['net_overlay_pnl'] == pytest.approx(303_717.73, abs=2.0)
        assert summary['excess_return_pct'] == pytest.approx(303.72, abs=0.05)
        assert summary['final_equity'] == pytest.approx(1_049_884.17, abs=2.0)
        assert summary['total_return_pct'] == pytest.approx(949.88, abs=0.05)

    def test_overlay_pnl_breakdown(
        self, result: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Gross premium is unchanged ($998K) but net costs fall to ~$695K (naive ~$730K).

        Same calls sold means identical gross premium; the hedge captures the call's delta
        so less of that premium is paid back through the equity-timing exposure, lifting
        premium retention from 26.9% (naive) to 30.4%.
        """
        summary, _ = result
        assert summary['total_premium_collected'] == pytest.approx(998_518.91, abs=5.0)
        assert summary['overlay_costs'] == pytest.approx(694_801.18, abs=5.0)
        assert summary['premium_retention_pct'] == pytest.approx(30.4, abs=0.1)

    def test_activity(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """~81% win rate (unchanged — same trade outcomes), ~30% max drawdown.

        Max drawdown is *higher* than the naive 22.86%: the hedge holds extra long stock
        funded by a negative cash balance, so the levered position drops harder in
        selloffs even though excess-return *vol* is lower (drawdown is a path statistic on
        total equity, not a dispersion measure on excess returns).
        """
        summary, _ = result
        assert summary['num_calls_sold'] == 181
        assert summary['win_rate'] == pytest.approx(81.1, abs=0.1)
        assert summary['max_drawdown_pct'] == pytest.approx(30.25, abs=0.05)

    def test_significance_uplift(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Risk-managed: Sharpe 0.462, NW t-stat 1.63 vs. naive's 0.126 / 0.46.

        Removing the equity-timing wiggle ~halves excess vol (9.90% → 5.39%) and ~doubles
        annualized excess return (+1.249% → +2.492%), pushing the t-stat from 0.46 → 1.63 —
        roughly 3.5× — without changing which calls were sold. Still below the t=2 bar
        because single-stock VRP on MSFT is structurally weak (see tutorial Part 5).
        """
        _, stats = result
        assert stats['ann_excess_return_pct'] == pytest.approx(2.492, abs=0.005)
        assert stats['ann_excess_vol_pct'] == pytest.approx(5.39, abs=0.02)
        assert stats['sharpe_excess'] == pytest.approx(0.462, abs=0.005)
        assert stats['t_stat_newey_west'] == pytest.approx(1.63, abs=0.02)
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False


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


# Bundled QQQ 10-year data — same yfinance CSV format as the MSFT file.
# Pins the headline figures the QQQ blog post (blog/05) quotes, the same way
# the MSFT classes pin the tutorial's. The CSV ships in the repo so CI
# reproduces the numbers without a network call.
_QQQ_CSV = os.path.join(os.path.dirname(__file__), 'qqq_10yr_prices.csv')


def _load_qqq_csv() -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
    """Parse the bundled QQQ CSV (same parser as _load_msft_csv)."""
    dates: list[str] = []
    prices: list[float] = []
    with open(_QQQ_CSV) as f:
        for row in csv.reader(f):
            if not row or not row[0][:4].isdigit():
                continue
            dates.append(row[0])
            prices.append(float(row[1]))
    return dates, np.array(prices, dtype=np.float64)


class TestQqqTenYearRegression:
    """Pin the headline numbers the QQQ blog post (blog/05) quotes for the
    naive overlay.

    QQQ counterpart to TestMsftTenYearRegression: locks the specific outputs
    the post cites so an engine change that would silently move them fails CI
    instead of leaving the post quietly wrong. Same _TUTORIAL_PARAMS, the
    bundled qqq_10yr_prices.csv (2016-06 → 2026-06).
    """

    @pytest.fixture(scope='class')
    def data(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
        return _load_qqq_csv()

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
        """$100K sizes into 9 QQQ contracts: ~$92.6K stock + ~$7.4K cash."""
        summary, _ = result
        assert summary['num_contracts'] == 9
        assert summary['initial_stock_cost'] == pytest.approx(92_622.08, abs=0.5)
        assert summary['cash'] == pytest.approx(7_377.92, abs=0.5)

    def test_returns_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Buy-and-hold $642K (+542%) + net overlay $104K = overlay $745K (+645%)."""
        summary, _ = result
        assert summary['buy_hold_final'] == pytest.approx(641_931.92, abs=1.0)
        assert summary['buy_hold_return_pct'] == pytest.approx(541.93, abs=0.05)
        assert summary['net_overlay_pnl'] == pytest.approx(103_522.06, abs=1.0)
        assert summary['excess_return_pct'] == pytest.approx(103.52, abs=0.05)
        assert summary['final_equity'] == pytest.approx(745_453.98, abs=1.0)
        assert summary['total_return_pct'] == pytest.approx(645.45, abs=0.05)

    def test_overlay_pnl_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """182 calls sold; ~$493K premium gross, ~$390K paid back, 21.0% retained."""
        summary, _ = result
        assert summary['num_calls_sold'] == 182
        assert summary['total_premium_collected'] == pytest.approx(493_471.64, abs=5.0)
        assert summary['overlay_costs'] == pytest.approx(389_949.58, abs=5.0)
        assert summary['premium_retention_pct'] == pytest.approx(21.0, abs=0.1)

    def test_activity(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """~77.5% win rate, ~22% max drawdown."""
        summary, _ = result
        assert summary['win_rate'] == pytest.approx(77.5, abs=0.1)
        assert summary['max_drawdown_pct'] == pytest.approx(21.96, abs=0.05)

    def test_significance(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """The post's headline: Sharpe 0.027, naive t=0.09, NW t=0.10 at L=8.

        Even weaker than MSFT's 0.46 — the naive QQQ overlay's excess over
        buy-and-hold is statistically indistinguishable from zero.
        """
        _, stats = result
        assert stats['n_days'] == 2514
        assert stats['years_of_data'] == pytest.approx(9.98, abs=0.005)
        assert stats['ann_excess_return_pct'] == pytest.approx(0.224, abs=0.002)
        assert stats['ann_excess_vol_pct'] == pytest.approx(8.27, abs=0.02)
        assert stats['sharpe_excess'] == pytest.approx(0.027, abs=0.002)
        assert stats['t_stat_naive'] == pytest.approx(0.09, abs=0.005)
        assert stats['t_stat_newey_west'] == pytest.approx(0.10, abs=0.005)
        assert stats['nw_lag'] == 8
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False

    def test_monte_carlo_shuffle(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin monte_carlo_shuffle on QQQ (500 paths, seed=42).

        The real ordered path (+645%) lands at the 80th percentile of the
        shuffled distribution (mean +597%, best +821%) — suggestive of some
        real price structure, but well short of MSFT's percentile-100 result.
        """
        dates, prices = data
        mc = monte_carlo_shuffle(
            dates, prices, _TUTORIAL_PARAMS,
            n_shuffles=_MC_SHUFFLES, seed=_MC_SEED,
        )
        assert mc['n_completed'] == _MC_SHUFFLES
        assert mc['real_return'] == pytest.approx(645.45, abs=0.05)
        assert mc['percentile'] == 80
        assert mc['mc_mean'] == pytest.approx(596.6, abs=2.0)
        assert mc['mc_max'] == pytest.approx(821.3, abs=2.0)


class TestQqqRiskManagedRegression:
    """Pin the risk-managed (delta-hedged) QQQ numbers the blog post quotes.

    QQQ counterpart to TestMsftRiskManagedRegression. Hedging the equity-timing
    wiggle lifts the Sharpe of excess from 0.027 (naive) to 0.405 and the
    Newey-West t-stat from 0.10 to 1.58 — a far larger proportional jump than
    MSFT's, because QQQ's naive signal is thinner to begin with. Still short of
    the t=2 bar: one index ETF over one decade can't clear it either.
    """

    @pytest.fixture(scope='class')
    def data(self) -> tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]:
        return _load_qqq_csv()

    @pytest.fixture(scope='class')
    def result(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        dates, prices = data
        hedge_params: dict[str, float] = {**_TUTORIAL_PARAMS, 'delta_hedge': 1.0}
        summary, _, daily_equity = run_cc_overlay(dates, prices, hedge_params)
        stats = compute_statistics(
            daily_equity, num_contracts=summary['num_contracts'], cash=summary['cash']
        )
        return summary, stats

    def test_returns_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Hedged uplift: net overlay $217K (vs naive $104K) → overlay $859K (+759%).

        Same 182 calls, same $493K gross premium, but capturing the call's
        delta roughly doubles net overlay P&L and lifts premium retention from
        21.0% (naive) to 44.0%. Max drawdown rises to ~27% (the hedge holds
        extra long stock funded by negative cash, so it drops harder in
        selloffs even though excess-return vol is lower).
        """
        summary, _ = result
        assert summary['num_contracts'] == 9
        assert summary['net_overlay_pnl'] == pytest.approx(217_118.92, abs=2.0)
        assert summary['total_return_pct'] == pytest.approx(759.05, abs=0.05)
        assert summary['total_premium_collected'] == pytest.approx(493_471.64, abs=5.0)
        assert summary['premium_retention_pct'] == pytest.approx(44.0, abs=0.1)
        assert summary['max_drawdown_pct'] == pytest.approx(26.59, abs=0.05)

    def test_significance_uplift(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Risk-managed: Sharpe 0.405, NW t-stat 1.58 vs. naive's 0.027 / 0.10.

        Stripping the equity-timing wiggle cuts excess vol (8.27% → 5.19%) and
        ~9× the annualized excess return (+0.224% → +2.099%), pushing the NW
        t-stat from 0.10 to 1.58 — roughly a 15× Sharpe jump. Still below t=2:
        a single index ETF over one decade can't clear the bar, same lesson as
        MSFT (see blog/04 and the post's breadth argument).
        """
        _, stats = result
        assert stats['ann_excess_return_pct'] == pytest.approx(2.099, abs=0.005)
        assert stats['ann_excess_vol_pct'] == pytest.approx(5.19, abs=0.02)
        assert stats['sharpe_excess'] == pytest.approx(0.405, abs=0.005)
        assert stats['t_stat_newey_west'] == pytest.approx(1.58, abs=0.02)
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False


class TestDegreesOfFreedom:
    """Pardo-style degrees-of-freedom validation (degrees_of_freedom).

    Two independent checks: (A) the bar-level "% degrees of freedom
    remaining" — Pardo's formal formula, which passes comfortably here —
    and (B) the ~30-trade sample-size floor, which is the binding
    constraint for a held-position overlay. The data-backed numbers
    (median grid trade count, per-window winners) live in
    TestMsftTenYearRegression; these lock the pure-function arithmetic.
    """

    def test_standard_in_sample_window(self) -> None:
        # Default 3-year in-sample window: 756 bars − 3 free params − 30-bar lookback.
        dof = degrees_of_freedom(756, n_parameters=3, indicator_lookback=30)
        assert dof['consumed'] == 33
        assert dof['remaining'] == 723
        assert dof['pct_remaining'] == 0.9563  # 723/756, rounded to 4 dp
        assert dof['passes_dof'] is True        # 95.63% clears the 90% floor
        assert dof['passes_trades'] is None     # no n_trades supplied

    def test_trade_count_floor(self) -> None:
        # Check (B): the conventional 30-trade floor.
        assert degrees_of_freedom(504, 3, 30, n_trades=24)['passes_trades'] is False
        assert degrees_of_freedom(504, 3, 30, n_trades=30)['passes_trades'] is True
        assert degrees_of_freedom(504, 3, 30, n_trades=50)['passes_trades'] is True

    def test_bar_level_threshold(self) -> None:
        # A tight 200-bar window: 167/200 = 83.5% < 90% → fails check (A).
        tight = degrees_of_freedom(200, 3, 30)
        assert tight['pct_remaining'] == 0.835
        assert tight['passes_dof'] is False
        # quantstrat's stricter 95% bar fails a 2-year (504-bar) window
        # (93.45% < 95%); the default 3-year window (95.63%) clears even that.
        assert degrees_of_freedom(504, 3, 30, min_pct_remaining=0.95)['passes_dof'] is False
        assert degrees_of_freedom(756, 3, 30, min_pct_remaining=0.95)['passes_dof'] is True

    def test_full_sample_passes_both(self) -> None:
        # Full 10y single run (2515 bars, 181 trades) passes both checks —
        # which is exactly why the per-window walk-forward view, not the
        # full-sample view, is the honest granularity for the DOF question.
        dof = degrees_of_freedom(2515, 3, 30, n_trades=181)
        assert dof['passes_dof'] is True
        assert dof['passes_trades'] is True


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
        """Buy-and-hold $746K (+646%) + net overlay $268K = overlay $1.015M (+915%)."""
        summary, _ = result
        assert summary['buy_hold_final'] == pytest.approx(746_166.44, abs=1.0)
        assert summary['buy_hold_return_pct'] == pytest.approx(646.17, abs=0.05)
        assert summary['net_overlay_pnl'] == pytest.approx(268_424.87, abs=1.0)
        assert summary['excess_return_pct'] == pytest.approx(268.42, abs=0.05)
        assert summary['final_equity'] == pytest.approx(1_014_591.31, abs=1.0)
        # The tutorial's headline "~915% total return on the bundled $100K config".
        assert summary['total_return_pct'] == pytest.approx(914.59, abs=0.05)

    def test_overlay_pnl_breakdown(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """181 calls sold; ~$999K premium gross, ~$730K paid back in costs."""
        summary, _ = result
        assert summary['num_calls_sold'] == 181
        assert summary['total_premium_collected'] == pytest.approx(998_518.91, abs=5.0)
        assert summary['overlay_costs'] == pytest.approx(730_094.04, abs=5.0)

    def test_activity(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """~81% win rate, ~23% max drawdown."""
        summary, _ = result
        assert summary['win_rate'] == pytest.approx(81.1, abs=0.1)
        assert summary['max_drawdown_pct'] == pytest.approx(22.86, abs=0.05)

    def test_significance(self, result: tuple[dict[str, Any], dict[str, Any]]) -> None:
        """Sharpe 0.126, naive t=0.40, NW t=0.46 at L=8 — clears neither bar."""
        _, stats = result
        assert stats['n_days'] == 2514
        assert stats['years_of_data'] == pytest.approx(9.98, abs=0.005)
        assert stats['ann_excess_return_pct'] == pytest.approx(1.249, abs=0.001)
        assert stats['ann_excess_vol_pct'] == pytest.approx(9.90, abs=0.01)
        assert stats['sharpe_excess'] == pytest.approx(0.126, abs=0.001)
        assert stats['t_stat_naive'] == pytest.approx(0.40, abs=0.005)
        assert stats['t_stat_newey_west'] == pytest.approx(0.46, abs=0.005)
        assert stats['nw_lag'] == 8
        assert stats['passes_t_2'] is False
        assert stats['passes_t_3'] is False

    @pytest.mark.parametrize(
        ('param', 'offsets_and_returns'),
        [
            # call_delta sweep: base 0.25 ± offset → total_return_pct.
            # Tutorial (rounded for display): -0.10:837%  -0.05:827%  base:915%
            #                                 +0.05:900%  +0.10:904%
            (
                'call_delta',
                [(-0.10, 836.93), (-0.05, 827.23), (0.0, 914.59),
                 (0.05, 900.20), (0.10, 903.82)],
            ),
            # close_at_pct sweep: base 0.75 ± offset → total_return_pct.
            # Tutorial: -0.20:946%  -0.10:956%  base:915%  +0.10:857%  +0.20:902%
            (
                'close_at_pct',
                [(-0.20, 945.77), (-0.10, 956.38), (0.0, 914.59),
                 (0.10, 856.57), (0.20, 901.52)],
            ),
        ],
    )
    def test_sensitivity_perturbations(
        self,
        data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]],
        param: str,
        offsets_and_returns: list[tuple[float, float]],
    ) -> None:
        """Pin cc_backtest.sensitivity_analysis on MSFT 2016-2026.

        The perturb-one-param-at-a-time sweep and its rationale now live
        in ``cc_backtest.sensitivity_analysis`` (the notebook companion
        calls the same function, so the two can't drift). This test pins
        its outputs at the tutorial's settings:
          - call_delta sweep at ±0.05 / ±0.10 from base=0.25
          - close_at_pct sweep at ±0.10 / ±0.20 from base=0.75

        It pins both the individual returns per offset and the "robust"
        verdict: the worst drop from base stays single-digit-percent of
        the base return — the "Swing" interpretation in the tutorial's
        example output ("robust" if the swing is small relative to base).
        Double-digit-% drops would indicate the chosen value is a
        knife-edge optimum (overfitting) rather than a plateau.
        """
        dates, prices = data
        offsets = tuple(off for off, _ in offsets_and_returns)
        result = sensitivity_analysis(
            dates, prices, _TUTORIAL_PARAMS, sweeps=((param, offsets),)
        )[param]

        # Each offset's total_return_pct is pinned so a regression here
        # surfaces as a test failure rather than a silent drift in the
        # tutorial's worked example.
        for (offset, expected), (got_off, got_ret) in zip(
            offsets_and_returns, result['returns']
        ):
            assert got_off == offset
            assert got_ret == pytest.approx(expected, abs=0.5)

        # Worst drop from base, as a percentage of base. Single-digit-%
        # means the strategy isn't fragile to this parameter — the
        # "robust" verdict in the tutorial.
        assert result['worst_drop_pct'] < 10.0  # single-digit-% drop

    def test_monte_carlo_shuffle(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin cc_backtest.monte_carlo_shuffle on MSFT 2016-2026.

        The shuffle/rebuild/re-backtest algorithm and its rationale now
        live in ``cc_backtest.monte_carlo_shuffle`` (the notebook
        companion calls the same function, so the two can't drift). This
        test pins its outputs at the tutorial's settings: 500 paths,
        seed=42, the standard params.

        On the bundled MSFT data the real ordered path beats every
        shuffled path (percentile 100), with mc_mean ~657% and the best
        shuffled path ~870% — the overlay exploits real price structure,
        not just the return distribution. This is the slowest test in
        the suite (~500 backtests, a couple of seconds).
        """
        dates, prices = data
        mc = monte_carlo_shuffle(
            dates, prices, _TUTORIAL_PARAMS,
            n_shuffles=_MC_SHUFFLES, seed=_MC_SEED,
        )

        assert mc['n_completed'] == _MC_SHUFFLES  # no path blew up at seed=42
        assert mc['real_return'] == pytest.approx(914.59, abs=0.05)
        assert mc['percentile'] == 100  # real path beats every shuffle
        assert mc['mc_mean'] == pytest.approx(656.8, abs=2.0)
        assert mc['mc_max'] == pytest.approx(870.1, abs=2.0)

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

        classify_regime returns a Series of per-index labels; the
        scalar "regime at the end" is `.iloc[-1]`.
        """
        base = [100.0] * 200

        assert classify_regime(base + [106.0]).iloc[-1] == 'bull'
        assert classify_regime(base + [94.0]).iloc[-1] == 'bear'
        assert classify_regime(base + [100.0]).iloc[-1] == 'sideways'
        # Boundary: strict inequalities, so equal-to-threshold stays sideways
        assert classify_regime(base + [105.0]).iloc[-1] == 'sideways'
        assert classify_regime(base + [95.0]).iloc[-1] == 'sideways'
        # Insufficient history: SMA undefined, label stays 'unknown'
        assert classify_regime([100.0] * 50).iloc[-1] == 'unknown'
        # Empty input: empty Series, no label to take
        assert classify_regime([]).empty

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
        day count (1,690 of 2,515) but contribute only ~$39K of
        trade pnl; bear days are ~280 but contribute ~$85K, because
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
        assert result['bull']['total_pnl'] == pytest.approx(38916.76, abs=5.0)
        assert result['bear']['total_pnl'] == pytest.approx(84616.05, abs=5.0)
        assert result['sideways']['total_pnl'] == pytest.approx(139031.86, abs=5.0)
        assert result['unknown']['total_pnl'] == pytest.approx(7915.91, abs=5.0)

        # Bear and sideways' per-day averages dwarf bull's — premium
        # is richest in volatile and choppy regimes. Specifically:
        # ~$303/day in bear and ~$402/day in sideways vs ~$23/day
        # in bull, i.e. ~10× higher.
        assert result['bear']['avg_pnl_per_day'] > 8 * result['bull']['avg_pnl_per_day']
        assert result['sideways']['avg_pnl_per_day'] > 8 * result['bull']['avg_pnl_per_day']

    def test_walk_forward_optimization(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin the walk-forward result on MSFT 2016-2026 (default 3-year train).

        With the tutorial's standard 3×3×3 grid (call_delta ∈
        {0.15, 0.20, 0.25}, dte ∈ {21, 30, 45}, close_at_pct ∈
        {0.50, 0.75, 1.00}) and the default 3-year train / 6-month test /
        6-month roll schedule, the engine produces 13 walk-forward periods
        over the 2019-04 → 2025-10 out-of-sample span. The window is 3 years
        specifically so every period clears Pardo's ~30-trade sample-size
        floor (the 2-year contrast is pinned in
        test_degrees_of_freedom_two_year_contrast).

        Empirical observations pinned here:
          - call_delta locks to 0.25 in all 13 periods; dte favors 21
            (9 of 13, the rest 30); close_at_pct splits 0.75 (7) / 0.50 (6)
            — the earlier-close setting competes once the window is longer.
          - Cumulative OOS compound return (per-period 6mo returns chained)
            is ~324% over the 6.5-year span, vs the ~378% fixed-defaults
            return over the same span. Both halves are pinned, so the
            324-vs-378 comparison the tutorial and blog draw is CI-verified.
            The gap is the cost of not having hindsight.
          - Same-span buy-and-hold is ~317%, so the honest walk-forward edge
            over buy-and-hold is only ~7 pp over 6.5 years — even thinner
            than the 2-year window's ~16 pp. Pinned so prose stays honest.
          - Convention note: these three pins deliberately mix accounting.
            The ~324% chains per-window returns with capital restarting at
            $100K each window; the ~378% / ~317% legs are single continuous
            runs over the span. On one consistent convention the gaps shift
            (fixed-vs-WF ~47-52 pp; WF-vs-BH ~+19 pp all-chained, ~+44 pp
            carrying capital forward) without changing the ordering. Prose
            on both surfaces (tutorial Part 4, blog Post 3) carries the
            matching caveat and defers the edge verdict to the Newey-West
            t-stat in test_significance, which no endpoint convention
            touches. Keep the pins on the published mixed convention; if
            re-pinning, update the caveat everywhere.

        Runtime is a few seconds (13 windows × 27 combos = 351 train
        backtests on 756-day windows).
        """
        from collections import Counter

        dates, prices = data
        param_grid: dict[str, list[float]] = {
            'call_delta': [0.15, 0.20, 0.25],
            'dte': [21, 30, 45],
            'close_at_pct': [0.50, 0.75, 1.00],
        }
        oos_equity, records = walk_forward_optimization(dates, prices, param_grid)

        # Window structure: 3y train, 6mo test, 6mo roll → 13 periods on
        # a 10y MSFT dataset starting 2016-04.
        assert len(records) == 13
        assert len(oos_equity) == 1635  # daily OOS equity points across all periods

        # train_end == test_start by construction (half-open intervals).
        for r in records:
            assert r['train_end'] == r['test_start']

        # First and last period bounds. First test starts 3 years in (2019-04),
        # one year later than the 2-year window's 2018-04.
        assert records[0]['test_start'] == '2019-04-11'
        assert records[0]['test_end'] == '2019-10-11'
        assert records[-1]['test_start'] == '2025-04-11'
        assert records[-1]['test_end'] == '2025-10-11'

        # Most-chosen params. call_delta pins to 0.25 in every period; dte
        # favors the monthly 21; close_at_pct is split between 0.75 and the
        # earlier-profit 0.50, with 45 DTE and the hold-to-expiry 1.00 never
        # winning at this window length.
        delta_counts = Counter(r['best_params']['call_delta'] for r in records)
        dte_counts = Counter(r['best_params']['dte'] for r in records)
        close_counts = Counter(r['best_params']['close_at_pct'] for r in records)
        assert delta_counts[0.25] == 13
        assert delta_counts[0.20] == 0
        assert delta_counts[0.15] == 0
        assert dte_counts[21] == 9
        assert dte_counts[30] == 4
        assert dte_counts[45] == 0
        assert close_counts[0.75] == 7
        assert close_counts[0.50] == 6
        assert close_counts[1.00] == 0

        # Pardo "How Many Trades?" sample-size check (feeds degrees_of_freedom,
        # check B): the 3-year window lifts EVERY period past the 30-trade
        # floor (median 54). That is the whole reason the window is 3 years —
        # contrast the 2-year window's 7-of-15-below in
        # test_degrees_of_freedom_two_year_contrast. Pins fig13's right panel.
        n_trades = [r['n_trades'] for r in records]
        assert all(isinstance(t, int) and t >= 30 for t in n_trades)
        assert sorted(n_trades)[len(n_trades) // 2] == 54  # median

        # Cumulative OOS compound return: chain per-period 6mo returns.
        cumulative = 1.0
        for r in records:
            in_period = (oos_equity['date'] >= r['test_start']) & (oos_equity['date'] < r['test_end'])
            period_eq: pd.Series[float] = oos_equity.loc[in_period, 'equity']  # type: ignore[assignment]
            assert not period_eq.empty, f"no OOS equity for period {r['test_start']}"
            period_ret = (period_eq.iloc[-1] - period_eq.iloc[0]) / period_eq.iloc[0]
            cumulative *= (1.0 + period_ret)
        cumulative_pct = (cumulative - 1.0) * 100
        # Pinned around ~324%, allow a few pp of slack for floating-point
        # variation in the run-to-run results.
        assert cumulative_pct == pytest.approx(324.0, abs=5.0)

        # The other half of the comparison the tutorial (Part 4) and the blog
        # series quote: the fixed __main__ defaults run over the *same* OOS
        # span. Pin it so the 324-vs-378 number can't drift silently across
        # the three prose surfaces. Derive the span from the asserted OOS
        # bounds (2019-04-11 → 2025-10-11) rather than hardcoding.
        oos_lo, oos_hi = records[0]['test_start'], records[-1]['test_end']
        span_idx = [i for i, d in enumerate(dates) if oos_lo <= d < oos_hi]
        span_dates = dates[span_idx[0]:span_idx[-1] + 1]
        span_prices = prices[span_idx[0]:span_idx[-1] + 1]
        assert len(span_dates) == len(oos_equity)  # same window as the OOS curve
        fixed_summary, _, _ = run_cc_overlay(span_dates, span_prices, _TUTORIAL_PARAMS)
        # Deterministic single run_cc_overlay call — pin to the same
        # precision as the other headline total_return_pct regressions.
        assert fixed_summary['total_return_pct'] == pytest.approx(378.17, abs=0.05)
        # Same-span buy-and-hold baseline (~317%). The honest walk-forward edge
        # over buy-and-hold is ~7 pp over 6.5 years on this mixed convention
        # (~19-44 pp on consistent ones — see the docstring's convention note).
        # Pinned so that framing is CI-verified wherever prose uses it.
        assert fixed_summary['buy_hold_return_pct'] == pytest.approx(316.83, abs=0.05)

    def test_degrees_of_freedom_first_window(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin the Pardo degrees-of-freedom numbers the __main__ report and
        the Part 4 tutorial prose cite for the default (first 3-year)
        in-sample window.

        Both of Pardo's checks pass at 3 years: the bar-level % comfortably
        (95.6%), and — the binding one — the trade count, whose grid median of
        36 clears the ~30 floor. That the trade count passes here but not at
        2 years (test_degrees_of_freedom_two_year_contrast) is exactly why the
        engine defaults to a 3-year window.
        """
        dates, prices = data
        train_cut = pd.to_datetime(dates[0]) + pd.DateOffset(years=3)
        is_dates = [d for d in dates if pd.to_datetime(d) < train_cut]
        is_prices = prices[:len(is_dates)]
        assert len(is_dates) == 756  # 3 years × 252 trading days

        # Bar-level degrees of freedom (check A): 756 − 3 params − 30 lookback.
        dof = degrees_of_freedom(len(is_dates), n_parameters=3, indicator_lookback=30)
        assert dof['consumed'] == 33
        assert dof['remaining'] == 723
        assert dof['pct_remaining'] == 0.9563
        assert dof['passes_dof'] is True

        # Trade count across the 27-combo grid on that window (check B):
        # median 36, range 17–73 — the figures the __main__ DOF block prints.
        grid = {'call_delta': [0.15, 0.20, 0.25], 'dte': [21, 30, 45],
                'close_at_pct': [0.50, 0.75, 1.00]}
        base = {'risk_free_rate': 0.045, 'capital': 100_000}
        counts = sorted(
            int(run_cc_overlay(is_dates, is_prices, {**base, **c})[0]['num_calls_sold'])
            for c in _param_combinations(grid)
        )
        assert counts[0] == 17
        assert counts[-1] == 73
        assert counts[len(counts) // 2] == 36  # median
        # Wiring the median into the sample-size check clears it (36 >= 30).
        assert degrees_of_freedom(756, 3, 30, n_trades=counts[len(counts) // 2])['passes_trades'] is True

    def test_degrees_of_freedom_two_year_contrast(
        self, data: tuple[list[str], np.ndarray[Any, np.dtype[np.float64]]]
    ) -> None:
        """Pin the 2-year 'before' that justifies the 3-year default window.

        A 2-year window is what the engine does NOT use, precisely because its
        trade count strains Pardo's floor: across the 15 walk-forward periods
        the winning fit lands below 30 trades in 7 of them (median 30), and the
        first window's grid median is just 24 (range 12–50) — below the floor —
        even though the bar-level check passes (93.5%). fig13's left panel and
        the Part 4 "why 3 years" prose cite these; the 3-year side is pinned in
        test_walk_forward_optimization and test_degrees_of_freedom_first_window.
        """
        dates, prices = data
        grid = {'call_delta': [0.15, 0.20, 0.25], 'dte': [21, 30, 45],
                'close_at_pct': [0.50, 0.75, 1.00]}
        base = {'risk_free_rate': 0.045, 'capital': 100_000}

        # Per-window winning-fit trade counts at the 2-year window.
        _, recs = walk_forward_optimization(dates, prices, grid, train_years=2)
        n_trades = sorted(r['n_trades'] for r in recs)
        assert len(recs) == 15
        assert sum(1 for t in n_trades if t < 30) == 7    # 7 of 15 below the floor
        assert n_trades[len(n_trades) // 2] == 30          # median right on it

        # First 2-year window: bar-level passes (93.5%), grid trade median strains.
        train_cut = pd.to_datetime(dates[0]) + pd.DateOffset(years=2)
        is_dates = [d for d in dates if pd.to_datetime(d) < train_cut]
        is_prices = prices[:len(is_dates)]
        assert len(is_dates) == 504
        assert degrees_of_freedom(504, 3, 30)['pct_remaining'] == 0.9345
        counts = sorted(
            int(run_cc_overlay(is_dates, is_prices, {**base, **c})[0]['num_calls_sold'])
            for c in _param_combinations(grid)
        )
        assert counts[0] == 12
        assert counts[-1] == 50
        assert counts[len(counts) // 2] == 24             # below the 30 floor
        assert degrees_of_freedom(504, 3, 30, n_trades=24)['passes_trades'] is False
