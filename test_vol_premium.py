"""Tests for the delta-neutral short-volatility VRP engine (vol_premium.py).

Two layers, mirroring the repo's pattern:
  - TestShortVolMechanics / TestShortVolStatistics / TestShortVolCompletion:
    ALWAYS-RUN synthetic checks of the delta-neutral logic, the market-neutral
    significance helper, and the completion (rf accrual, hedge cost, the rf-base
    netting the audit fixed). These pin the MECHANISM and must hold regardless of
    the real data.
  - TestSpyShortVolRegression: DATASET-GATED. Pins the completed, audited SPY
    headline (0.25-delta, full 2010-12 -> 2026-06 span): the rate-invariant
    Bakshi-Kapadia delta-hedged premium is +2.54 (Sharpe 0.52) and survives
    SPY's realistic transaction costs (+2.42 at ~0.2bp). Caveats in docs/vol_premium.md.
"""

from __future__ import annotations

import math
import os
from typing import Any

import pandas as pd
import pytest

from vol_premium import run_real_short_vol_overlay, short_vol_statistics

_SPY_DAILIES = os.path.join(os.path.dirname(__file__), 'spy_option_dailies.csv')
_HAVE_SPY = os.path.exists(_SPY_DAILIES) or os.path.exists(_SPY_DAILIES + '.gz')


def _scenario(
    price_path: list[tuple[str, float]],
    option_path: list[tuple[float, float, float, float]],
    strike: float,
    cid: str = 'OPT',
) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
    """Build a one-cycle synthetic (dates, prices, store). The last date is the
    expiration. option_path[i] = (bid, ask, mid, delta) for day i."""
    exp = price_path[-1][0]
    dte0 = len(price_path) - 1
    store: dict[str, dict[str, Any]] = {}
    dates: list[str] = []
    prices: list[float] = []
    for i, ((date, px), (bid, ask, mid, delta)) in enumerate(zip(price_path, option_path)):
        dates.append(date)
        prices.append(px)
        cand = (dte0 - i, delta, bid, ask, mid, exp, strike, cid)
        store[date] = {'candidates': [cand], 'marks': {cid: (bid, ask, mid, delta)}}
    return dates, prices, store


class TestShortVolMechanics:
    def test_flat_market_harvests_premium(self) -> None:
        """Stock dead flat, call decays to worthless: the delta-neutral short
        keeps ~the whole premium (the VRP when realized vol is 0)."""
        days = [f'2020-01-0{i+1}' for i in range(6)]
        price_path = [(d, 100.0) for d in days]
        option_path = [(2.0, 2.1, 2.05, 0.50), (1.55, 1.65, 1.60, 0.45),
                       (1.05, 1.15, 1.10, 0.35), (0.55, 0.65, 0.60, 0.20),
                       (0.15, 0.25, 0.20, 0.08), (0.0, 0.0, 0.0, 0.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=102.0)
        summary, _, _ = run_real_short_vol_overlay(
            dates, prices, store, {'target_delta': 0.50, 'capital': 100_000, 'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        assert summary['num_calls_sold'] == 1
        # Flat market => hedge trades all execute at 100, no hedge P&L; net ~ premium.
        assert summary['net_pnl'] > 0
        assert summary['net_pnl'] == pytest.approx(summary['total_premium_collected'], rel=0.02)

    def test_hedge_offsets_direction(self) -> None:
        """Stock trends up through the strike: the long hedge captures the rise,
        so the delta-neutral net loss is a small gamma cost, NOT the full naked
        assignment loss. This is the property the covered-call hedge-to-B&H lacks."""
        days = [f'2020-02-0{i+1}' for i in range(5)]
        price_path = [('2020-02-01', 100.0), ('2020-02-02', 102.0),
                      ('2020-02-03', 105.0), ('2020-02-04', 108.0),
                      ('2020-02-05', 110.0)]
        option_path = [(2.0, 2.1, 2.05, 0.50), (3.0, 3.1, 3.05, 0.62),
                       (4.5, 4.6, 4.55, 0.80), (6.4, 6.6, 6.50, 0.95),
                       (8.0, 8.1, 8.05, 1.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=102.0)
        summary, _, _ = run_real_short_vol_overlay(
            dates, prices, store, {'target_delta': 0.50, 'capital': 100_000, 'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        shares = summary['num_contracts'] * 100
        premium_per_share = summary['total_premium_collected'] / shares
        naked_loss = (premium_per_share - (110.0 - 102.0)) * shares  # ~ -6/share, large
        assert naked_loss < 0
        # Hedged loss is a fraction of the naked assignment loss (the hedge worked).
        assert abs(summary['net_pnl']) < 0.5 * abs(naked_loss)

    def test_no_chain_no_trade(self) -> None:
        """Empty store on every date => no entry, flat equity at capital."""
        dates = ['2021-03-01', '2021-03-02', '2021-03-03']
        prices = [100.0, 101.0, 99.0]
        store: dict[str, dict[str, Any]] = {}
        summary, trades, eq = run_real_short_vol_overlay(
            dates, prices, store, {'capital': 100_000, 'risk_free_rate': 0.0})
        assert summary['num_calls_sold'] == 0
        assert trades == []
        assert summary['net_pnl'] == pytest.approx(0.0, abs=0.01)


class TestShortVolStatistics:
    def test_rising_equity_positive_t(self) -> None:
        """A steadily rising equity curve => positive, finite Newey-West t."""
        eq = pd.DataFrame({'equity': [100_000 + 50 * i for i in range(300)],
                           'price': [100.0] * 300})
        st = short_vol_statistics(eq, 100_000, rf=0.0)
        assert st['t_stat_newey_west'] > 0
        assert st['n_days'] == 299
        assert 'passes_t_2' in st

    def test_flat_equity_zero_t(self) -> None:
        """A flat equity curve => ~zero t, does not pass the t=2 bar."""
        eq = pd.DataFrame({'equity': [100_000.0] * 200, 'price': [100.0] * 200})
        st = short_vol_statistics(eq, 100_000, rf=0.0)
        assert st['t_stat_newey_west'] == pytest.approx(0.0, abs=1e-9)
        assert st['passes_t_2'] is False

    def test_flat_rf_fallback_columnless_curve(self) -> None:
        """The legacy flat-rf FALLBACK, for a hand-built curve with NO rf_credit
        column. A synthetic equity that grows by exactly rf-on-capital each day
        nets to ~zero excess (t~0) under the flat fallback. This is NOT the engine
        path: the engine records the ACTUAL per-day rf credit (cash base) and the
        helper nets that instead (test_excess_nets_actual_rf_with_open_position).
        Flat-on-capital is only a convenience for column-less curves — on real
        runs it MIS-benchmarks, removing rf on the capital ($100K) rather than the
        smaller cash base (~$68K) the engine actually credited."""
        cap, rf = 100_000.0, 0.045
        daily = cap * rf / 252  # exactly the T-bill dollar earned per day on capital
        eq = pd.DataFrame({'equity': [cap + daily * i for i in range(400)],
                           'price': [100.0] * 400})
        st = short_vol_statistics(eq, cap, rf=rf)  # no rf_credit column -> flat fallback
        assert st['t_stat_newey_west'] == pytest.approx(0.0, abs=1e-6)
        assert st['ann_excess_return_pct'] == pytest.approx(0.0, abs=1e-6)
        assert st['passes_t_2'] is False


class TestShortVolCompletion:
    """The two omissions the buy-and-hold comparison exposed, now modeled and
    tested: rf earned on idle collateral, and the share-hedge half-spread cost."""

    def test_rf_accrues_on_idle_cash(self) -> None:
        """No position ever opens (empty store): equity compounds at the
        risk-free rate, and the vol alpha (net of rf) is ~0."""
        dates = [f'2020-{i // 28 + 1:02d}-{i % 28 + 1:02d}' for i in range(252)]
        prices = [100.0] * len(dates)
        store: dict[str, Any] = {}
        s, _, _ = run_real_short_vol_overlay(
            dates, prices, store,
            {'capital': 100_000, 'risk_free_rate': 0.045, 'hedge_cost_bps': 0.0})
        assert s['num_calls_sold'] == 0
        assert s['interest_earned'] > 0
        assert s['alpha_vs_cash'] == pytest.approx(0.0, abs=0.01)
        expected = 100_000 * ((1 + 0.045 / 252) ** (len(dates) - 1) - 1)
        assert s['net_pnl'] == pytest.approx(expected, rel=1e-6)

    def test_hedge_cost_reduces_pnl(self) -> None:
        """A nonzero hedge_cost_bps charges the half-spread on each rebalance and
        lowers net P&L by exactly the accumulated cost (commission-free shares)."""
        days = [f'2020-01-0{i + 1}' for i in range(6)]
        price_path = [(d, 100.0) for d in days]
        option_path = [(2.0, 2.1, 2.05, 0.50), (1.55, 1.65, 1.60, 0.45),
                       (1.05, 1.15, 1.10, 0.35), (0.55, 0.65, 0.60, 0.20),
                       (0.15, 0.25, 0.20, 0.08), (0.0, 0.0, 0.0, 0.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=102.0)
        base = {'target_delta': 0.50, 'capital': 100_000, 'risk_free_rate': 0.0}
        free, _, _ = run_real_short_vol_overlay(dates, prices, store, {**base, 'hedge_cost_bps': 0.0})
        costed, _, _ = run_real_short_vol_overlay(dates, prices, store, {**base, 'hedge_cost_bps': 50.0})
        assert costed['total_hedge_cost'] > 0
        assert free['net_pnl'] - costed['net_pnl'] == pytest.approx(costed['total_hedge_cost'], rel=1e-6)

    def test_decomposition_identity(self) -> None:
        """Audit invariant: net_pnl == rf interest + vol alpha, and equals
        final_equity − capital (the single conservation law)."""
        price_path = [('2020-02-01', 100.0), ('2020-02-02', 102.0),
                      ('2020-02-03', 105.0), ('2020-02-04', 108.0),
                      ('2020-02-05', 110.0)]
        option_path = [(2.0, 2.1, 2.05, 0.50), (3.0, 3.1, 3.05, 0.62),
                       (4.5, 4.6, 4.55, 0.80), (6.4, 6.6, 6.50, 0.95),
                       (8.0, 8.1, 8.05, 1.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=102.0)
        s, _, _ = run_real_short_vol_overlay(
            dates, prices, store, {'target_delta': 0.50, 'capital': 100_000})
        assert s['net_pnl'] == pytest.approx(s['alpha_vs_cash'] + s['interest_earned'], abs=0.01)
        assert s['final_equity'] - s['capital'] == pytest.approx(s['net_pnl'], abs=0.01)

    def test_excess_nets_actual_rf_with_open_position(self) -> None:
        """The path the other 9 tests miss: rf > 0 AND a hedged position open
        across the rf-accruing days. The engine credits rf on CASH (not capital),
        so short_vol_statistics must net the ACTUAL recorded credit. Its excess
        then equals the rf-netted vol-P&L — identical to the SAME run with rf=0
        (rf cancels, rate-invariant). A flat rf/252 on capital would NOT cancel
        (cash != capital): that base mismatch is exactly the bug this guards.

        (Every other test either passes rf=0.0 to the helper, or opens no position
        on the rf>0 days, so none exercises the netting on a real cash path.)"""
        days = [d.strftime('%Y-%m-%d') for d in pd.bdate_range('2020-01-02', periods=31)]
        strike = 100.0
        price_path, option_path = [], []
        for i, d in enumerate(days):
            px = 100.0 + 2.0 * math.sin(i)              # deterministic mild oscillation
            delta = min(max(0.5 + (px - 100.0) * 0.08, 0.05), 0.95)
            tv = 2.0 * (1 - i / (len(days) - 1))        # time value -> 0 at expiry
            mid = max(0.0, px - strike) + tv
            price_path.append((d, px))
            option_path.append((max(0.0, mid - 0.05), mid + 0.05, mid, delta))
        dates, prices, store = _scenario(price_path, option_path, strike=strike)
        base = {'target_delta': 0.50, 'dte': 30, 'capital': 100_000, 'hedge_cost_bps': 0.0}
        s_rf, _, eq_rf = run_real_short_vol_overlay(dates, prices, store, {**base, 'risk_free_rate': 0.045})
        s_0, _, eq_0 = run_real_short_vol_overlay(dates, prices, store, {**base, 'risk_free_rate': 0.0})
        # Preconditions: this run actually exercises the missed path.
        assert s_rf['num_calls_sold'] >= 1
        assert s_rf['interest_earned'] != 0.0
        assert 'rf_credit' in eq_rf.columns and eq_rf['rf_credit'].abs().sum() > 0
        st_rf = short_vol_statistics(eq_rf, s_rf['capital'])  # nets the recorded rf_credit
        st_0 = short_vol_statistics(eq_0, s_0['capital'])     # rf=0 reference vol-P&L
        # rf cancels: the netted excess equals the rf=0 vol-P&L (day-mean and t-stat).
        assert st_rf['mean_daily_excess_dollars'] == pytest.approx(st_0['mean_daily_excess_dollars'], abs=0.05)
        assert st_rf['t_stat_newey_west'] == pytest.approx(st_0['t_stat_newey_west'], abs=0.02)
        # Conservation: the summed excess is exactly the equity growth (from the
        # day-0 entry baseline) net of the rf the engine credited — the rf is
        # removed on the SAME base it was earned on, so it cancels to the vol-P&L.
        eqv = eq_rf['equity'].to_numpy(float)
        expected = (eqv[-1] - eqv[0]) - s_rf['interest_earned']
        assert st_rf['mean_daily_excess_dollars'] * st_rf['n_days'] == pytest.approx(expected, abs=0.5)
        # The buggy flat-rf-on-capital benchmark does NOT cancel (cash != capital).
        flat = short_vol_statistics(eq_rf.drop(columns=['rf_credit']), s_rf['capital'], rf=0.045)
        assert abs(flat['mean_daily_excess_dollars'] - st_rf['mean_daily_excess_dollars']) > 1.0


@pytest.mark.skipif(not _HAVE_SPY, reason='needs spy_option_dailies.csv or its .gz twin')
class TestSpyShortVolRegression:
    """Pin the delta-neutral short-CALL VRP on real SPY chains — the completed,
    audited headline. 0.25-delta, 30 DTE, hold-to-expiry, full 2010-12 -> 2026-06
    span (CHAIN_CLEAN_START['SPY']); rf credited on the cash collateral;
    Schwab-aware hedge cost (commission-free shares, half-spread). Significance is
    the rate-invariant Bakshi-Kapadia delta-hedged-gain measure (rf netted on the
    cash base it was earned on — see short_vol_statistics).

    Verdict: the gross delta-hedged premium is POSITIVE and MARGINALLY SIGNIFICANT
    (Newey-West t +2.54, Sharpe 0.52, +$36.5K vol-P&L) and SURVIVES SPY's realistic
    transaction costs — at SPY's ~penny half-spread (~0.2bp) it is +2.42, at 0.5bp
    +2.25 (both clear t=2); it slips to +1.97 only at a conservative 1bp and goes
    negative by 5bp. Rate-invariant (rf=0 and rf=4.5% give the same t). The covered
    call buried this under equity beta + a buy-and-hold hedge (cf.
    TestMsftRealRiskManagedRegression / TestQqqRealRiskManagedRegression, t ~0).
    Caveats (docs/vol_premium.md): gross of the hedge's capital financing, single
    index, daily-close hedging understates the short-gamma tail, +0.26 correlation
    to SPY. Adversarially audited (engine bookkeeping clean; one benchmark-base bug
    found and fixed).
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_DAILIES, start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': 0.25, 'dte': 30, 'capital': 100_000,
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        """8 contracts, 175 calls over 2010-12 -> 2026-06; net +$73.0K = +$36.5K rf
        interest + $36.5K vol premium (frictionless)."""
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 8
        assert s['num_calls_sold'] == 175
        assert s['win_rate'] == pytest.approx(65.5, abs=0.1)
        assert s['net_pnl'] == pytest.approx(72_999.90, abs=1.5)
        assert s['interest_earned'] == pytest.approx(36_504.76, abs=1.5)
        assert s['alpha_vs_cash'] == pytest.approx(36_495.14, abs=1.5)
        assert s['max_drawdown_pct'] == pytest.approx(4.09, abs=0.05)
        assert s['net_pnl'] == pytest.approx(s['alpha_vs_cash'] + s['interest_earned'], abs=0.01)

    def test_gross_premium_significant(self, market: Any) -> None:
        """NW t +2.54 (Sharpe 0.52): the call-wing delta-hedged premium clears
        t=2 where the covered call showed ~0."""
        _, st = self._run(market, 0.0)
        assert st['t_stat_newey_west'] == pytest.approx(2.54, abs=0.02)
        assert st['sharpe'] == pytest.approx(0.52, abs=0.005)
        assert st['ann_excess_return_pct'] == pytest.approx(2.36, abs=0.02)
        assert st['nw_lag'] == 9
        assert st['passes_t_2'] is True

    def test_rate_invariant(self, market: Any) -> None:
        """The verdict nets rf on the cash base it was earned on, so it is the
        same whether the engine charged rf=0 or rf=4.5% (the audited fix)."""
        _, st0 = self._run(market, 0.0, rf=0.0)
        _, st45 = self._run(market, 0.0, rf=0.045)
        assert st0['t_stat_newey_west'] == pytest.approx(st45['t_stat_newey_west'], abs=0.01)
        assert st45['t_stat_newey_west'] == pytest.approx(2.54, abs=0.02)

    def test_survives_realistic_cost_but_not_wide(self, market: Any) -> None:
        """At SPY's ~penny half-spread (~0.2bp) the premium clears t=2 (+2.42);
        it slips below only at a conservative 1bp (+1.97) and goes negative by an
        unrealistic 5bp (−0.35)."""
        _, st02 = self._run(market, 0.2)
        assert st02['t_stat_newey_west'] == pytest.approx(2.42, abs=0.02)
        assert st02['passes_t_2'] is True
        s1, st1 = self._run(market, 1.0)
        assert s1['total_hedge_cost'] == pytest.approx(8_298.81, abs=5.0)
        assert st1['t_stat_newey_west'] == pytest.approx(1.97, abs=0.02)
        assert st1['passes_t_2'] is False
        _, st5 = self._run(market, 5.0)
        assert st5['t_stat_newey_west'] == pytest.approx(-0.35, abs=0.02)
        assert st5['passes_t_2'] is False
