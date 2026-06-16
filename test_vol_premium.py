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

from vol_premium import (
    run_real_short_vol_overlay,
    run_real_straddle_overlay,
    select_put_entry,
    select_straddle,
    short_vol_statistics,
)

_SPY_DAILIES = os.path.join(os.path.dirname(__file__), 'spy_option_dailies.csv')
_HAVE_SPY = os.path.exists(_SPY_DAILIES) or os.path.exists(_SPY_DAILIES + '.gz')
_SPY_PUTS = os.path.join(os.path.dirname(__file__), 'spy_option_dailies_puts.csv')
_HAVE_SPY_PUTS = os.path.exists(_SPY_PUTS) or os.path.exists(_SPY_PUTS + '.gz')
_IWM_DAILIES = os.path.join(os.path.dirname(__file__), 'iwm_option_dailies.csv')
_HAVE_IWM = os.path.exists(_IWM_DAILIES) or os.path.exists(_IWM_DAILIES + '.gz')


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

    def test_summed_excess_omits_day0_entry_spread(self) -> None:
        """short_vol_statistics' summed daily excess and summary['alpha_vs_cash']
        are NOT identical — they differ by the day-0 entry-spread mark.

        np.diff(eq) starts from eq[0], which is ALREADY struck at the entry bid/ask
        mid (the short was sold at the bid, day 0 is marked at the mid, less
        commission and the day-0 hedge half-spread), so the summed-excess series
        omits that single day-0 cost that alpha_vs_cash carries. The gap is exactly
        eq[0] - capital — ONE entry spread no matter how many cycles run, since only
        the first entry predates a diff — and because it OMITS a cost, the summed
        excess slightly flatters the vol-P&L, never deflates it. The earlier 1-cycle
        checks compared the summed excess to the eq[0] baseline (the correct one), so
        none pinned this gap to alpha_vs_cash directly; a multi-cycle run makes the
        single-spread offset unambiguous."""
        # Three back-to-back call cycles, each 6 trading days with its own
        # expiration, so the engine re-enters twice after the day-0 cycle.
        dates: list[str] = []
        prices: list[float] = []
        store: dict[str, dict[str, Any]] = {}
        base = pd.Timestamp('2020-01-02')
        idx = 0
        for c in range(3):
            cid = f'OPT{c}'
            cyc = [(base + pd.Timedelta(days=idx + k)).strftime('%Y-%m-%d') for k in range(6)]
            exp = cyc[-1]
            for k, d in enumerate(cyc):
                frac = 1 - k / 5
                mid = round(2.0 * frac, 2)
                bid, ask = round(max(0.0, mid - 0.05), 2), round(mid + 0.05, 2)
                delta = round(max(0.06, 0.50 * frac + 0.06), 2)  # stays in (0.05, 0.60)
                store[d] = {'candidates': [(5 - k, delta, bid, ask, mid, exp, 102.0, cid)],
                            'marks': {cid: (bid, ask, mid, delta)}}
                dates.append(d)
                prices.append(100.0 + 0.5 * k)
            idx += 6
        params = {'target_delta': 0.50, 'dte': 5, 'capital': 100_000,
                  'risk_free_rate': 0.045, 'hedge_cost_bps': 1.0}
        s, _, eq = run_real_short_vol_overlay(dates, prices, store, params)
        assert s['num_calls_sold'] >= 2  # genuinely multi-cycle
        st = short_vol_statistics(eq, s['capital'], rf=0.045)

        eqv = eq['equity'].to_numpy(float)
        # What short_vol_statistics actually conserves: equity growth from the day-0
        # mark, net of the engine's recorded rf — NOT growth from capital.
        summed_excess = (eqv[-1] - eqv[0]) - s['interest_earned']
        assert st['mean_daily_excess_dollars'] * st['n_days'] == pytest.approx(summed_excess, abs=0.5)
        # The gap to alpha_vs_cash is exactly the day-0 entry-spread mark.
        day0_mark = eqv[0] - s['capital']
        assert s['alpha_vs_cash'] - summed_excess == pytest.approx(day0_mark, abs=0.02)
        # It is a real cost (sold at the bid, marked at the mid + fees), not rounding
        # noise — so the gap genuinely exists and the summed excess flatters.
        assert day0_mark < 0.0
        assert abs(day0_mark) > 1.0


class TestShortPutMechanics:
    """The §9 put-leg engine capability (synthetic, always-run): the put mirror
    of TestShortVolMechanics. select_put_entry picks the nearest negative-delta
    put; a flat market harvests the premium; the SHORT-stock hedge offsets a
    drop. Pins the mechanism before the real put fetch (docs/prereg_vol_premium.md)
    — no real put data exists yet, so the registered run stays gated."""

    @staticmethod
    def _put_cand(dte: int, delta: float, bid: float = 1.0) -> tuple[Any, ...]:
        return (dte, delta, bid, bid + 0.1, bid + 0.05, '2024-01-19', 100.0, 'P')

    def test_select_put_entry_nearest_negative_delta(self) -> None:
        day = {'candidates': [
            self._put_cand(30, -0.10), self._put_cand(30, -0.28),
            self._put_cand(30, -0.45), self._put_cand(30, 0.30),  # a call: ignored
            self._put_cand(30, -0.02),  # outside the -0.05 band: ignored
        ], 'marks': {}}
        pick = select_put_entry(day, 30, -0.25)
        assert pick is not None
        assert pick[1] == pytest.approx(-0.28)  # nearest to -0.25 among in-band puts

    def test_put_flat_market_harvests_premium(self) -> None:
        """OTM put (strike 95 < spot 100), flat market, decays to worthless: the
        delta-neutral short put keeps ~the whole premium."""
        days = [f'2020-03-0{i + 1}' for i in range(6)]
        price_path = [(d, 100.0) for d in days]
        option_path = [(2.0, 2.1, 2.05, -0.30), (1.55, 1.65, 1.60, -0.24),
                       (1.05, 1.15, 1.10, -0.17), (0.55, 0.65, 0.60, -0.10),
                       (0.15, 0.25, 0.20, -0.04), (0.0, 0.0, 0.0, 0.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=95.0)
        s, _, _ = run_real_short_vol_overlay(
            dates, prices, store,
            {'option_type': 'put', 'target_delta': -0.30, 'capital': 100_000,
             'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        assert s['num_calls_sold'] == 1
        assert s['net_pnl'] > 0
        assert s['net_pnl'] == pytest.approx(s['total_premium_collected'], rel=0.02)

    def test_put_hedge_offsets_a_drop(self) -> None:
        """Stock falls through the strike: the SHORT-stock hedge gains as the
        stock drops, so the delta-neutral net loss is a small gamma cost, NOT the
        full naked put-assignment loss — the put mirror of the call hedge test."""
        price_path = [('2020-04-01', 100.0), ('2020-04-02', 98.0),
                      ('2020-04-03', 95.0), ('2020-04-04', 92.0),
                      ('2020-04-05', 90.0)]
        option_path = [(2.0, 2.1, 2.05, -0.50), (3.0, 3.1, 3.05, -0.62),
                       (4.5, 4.6, 4.55, -0.80), (6.4, 6.6, 6.50, -0.95),
                       (8.0, 8.1, 8.05, -1.0)]
        dates, prices, store = _scenario(price_path, option_path, strike=102.0)
        s, _, _ = run_real_short_vol_overlay(
            dates, prices, store,
            {'option_type': 'put', 'target_delta': -0.50, 'capital': 100_000,
             'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        shares = s['num_contracts'] * 100
        premium_per_share = s['total_premium_collected'] / shares
        naked_loss = (premium_per_share - (102.0 - 90.0)) * shares  # ~ -10/share, large
        assert naked_loss < 0
        assert abs(s['net_pnl']) < 0.5 * abs(naked_loss)


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


@pytest.mark.skipif(not _HAVE_SPY_PUTS, reason='needs spy_option_dailies_puts.csv or its .gz twin')
class TestSpyShortPutRegression:
    """Pin the REGISTERED put-side VRP result on real SPY chains (docs/prereg_vol_premium.md,
    registered at PR #23's merge commit; analysis run_registered_vrp.py). A daily
    delta-neutral short PUT at target delta -0.25, 30 DTE, hold-to-expiry, sold at the
    bid, hedged with SHORT stock; full 2010-12 -> 2026-06 span (CHAIN_CLEAN_START['SPY'])
    -- the exact mirror of the pinned 0.25-delta CALL wing (TestSpyShortVolRegression,
    +2.54), only the wing flipped (prereg §2.3).

    REGISTERED VERDICT (prereg §5/§6, row 4): NULL. The put-wing delta-hedged gain is
    INSIGNIFICANT even gross (Newey-West t +0.20) and net of the 0.5bp headline cost
    (+0.09) -- far below the t=2 bar and below the +2.54 call wing, so the §1.3 mechanism
    clause is NOT met. The short-put (crash-insurance) book harvests premium in calm years
    and gives it back in vol events (2018 -$8.1K, 2022 -$6.7K, 2025 -$6.4K), netting ~0;
    its 13.3% drawdown vs the call's 4.1% is the skew tail. Adversarially verified (5
    independent lenses, 0 refutations): rate-invariant, delta-neutral (corr -0.06 to SPY),
    economically coherent, correct mechanics. Consistent with Dew-Becker & Giglio (2025)
    post-2010 decline, now on the put wing.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_PUTS, start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': -0.25, 'dte': 30, 'capital': 100_000, 'option_type': 'put',
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        """175 put cycles (8 contracts), 88.5% win, $364,106 premium collected;
        frictionless net +$155.9K = +$150.7K rf interest + only +$5.2K vol premium."""
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 8
        assert s['num_calls_sold'] == 175
        assert s['win_rate'] == pytest.approx(88.5, abs=0.1)
        assert s['total_premium_collected'] == pytest.approx(364_106.0, abs=2.0)
        assert s['alpha_vs_cash'] == pytest.approx(5_186.86, abs=2.0)
        assert s['interest_earned'] == pytest.approx(150_666.32, abs=2.0)
        assert s['net_pnl'] == pytest.approx(155_853.18, abs=2.0)
        assert s['max_drawdown_pct'] == pytest.approx(13.18, abs=0.05)
        assert s['net_pnl'] == pytest.approx(s['alpha_vs_cash'] + s['interest_earned'], abs=0.01)

    def test_registered_verdict_null(self, market: Any) -> None:
        """H1 FAILS (prereg §5.1): the put-wing gross t is +0.20 and the net-0.5bp t
        is +0.09 -- both far below 2, and below the +2.54 call wing, so §1.3's mechanism
        clause is not met. passes_t_2 is False at the verdict cost."""
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(0.20, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(0.09, abs=0.02)
        assert st5['sharpe'] == pytest.approx(0.014, abs=0.005)
        assert st5['nw_lag'] == 9
        assert st5['passes_t_2'] is False
        assert st5['t_stat_newey_west'] < 2.54  # §1.3 mechanism clause not met

    def test_cost_curve(self, market: Any) -> None:
        """The put wing never clears t=2: +0.20 gross -> +0.16 (0.2bp) -> +0.09
        (0.5bp headline) -> -0.02 (1bp). It is null before costs even bite."""
        assert self._run(market, 0.2)[1]['t_stat_newey_west'] == pytest.approx(0.16, abs=0.02)
        s1, st1 = self._run(market, 1.0)
        assert st1['t_stat_newey_west'] == pytest.approx(-0.02, abs=0.02)
        assert s1['total_hedge_cost'] == pytest.approx(5_830.11, abs=5.0)
        assert st1['passes_t_2'] is False

    def test_rate_invariant(self, market: Any) -> None:
        """The verdict nets rf on the cash base it was earned on, so the put t is the
        same at rf=0 and rf=4.5% (the audited measure, guarding the null)."""
        _, st0 = self._run(market, 0.5, rf=0.0)
        _, st45 = self._run(market, 0.5, rf=0.045)
        assert st0['t_stat_newey_west'] == pytest.approx(st45['t_stat_newey_west'], abs=0.01)
        assert st45['t_stat_newey_west'] == pytest.approx(0.09, abs=0.02)


@pytest.mark.skipif(not _HAVE_IWM, reason='needs iwm_option_dailies.csv or its .gz twin')
class TestIwmShortPutRegression:
    """Pin the REGISTERED out-of-sample confirmation arm (prereg §5.2): the same
    -0.25-delta short put on real IWM (Russell 2000) chains, an index this project had
    never run -- so its underlying is genuinely naive. Span 2010-12-01 -> 2026-06-05
    (CHAIN_CLEAN_START['IWM'], validated clean from row one).

    DOES NOT CONFIRM (§5.2): IWM's put-wing gain is larger than SPY's (+$25.1K gross vol
    P&L) but still INSIGNIFICANT -- gross t +1.00, net-0.5bp +0.91, below the t=2 bar.
    With SPY also null, the §6 conjunction is not met: not a confirmed put-wing VRP. Same
    short-gamma signature (delta-neutral, corr -0.12 to IWM; 13.4% drawdown).
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_IWM_DAILIES, start=CHAIN_CLEAN_START['IWM'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('IWM', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        # identical instrument + cost rule to SPY (prereg §5.2)
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': -0.25, 'dte': 30, 'capital': 100_000, 'option_type': 'put',
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        """169 put cycles (13 contracts), 86.3% win, $361,467 premium; frictionless net
        +$181.3K = +$156.2K rf interest + +$25.1K vol premium."""
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 13
        assert s['num_calls_sold'] == 169
        assert s['win_rate'] == pytest.approx(86.3, abs=0.1)
        assert s['total_premium_collected'] == pytest.approx(361_466.95, abs=2.0)
        assert s['alpha_vs_cash'] == pytest.approx(25_126.99, abs=2.0)
        assert s['net_pnl'] == pytest.approx(181_284.80, abs=2.0)
        assert s['max_drawdown_pct'] == pytest.approx(13.30, abs=0.05)

    def test_does_not_confirm(self, market: Any) -> None:
        """§5.2: IWM confirms only if gross AND net-0.5bp t > 2. Both fail (+1.00 gross,
        +0.91 net) -- larger than SPY's null but still insignificant."""
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.00, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(0.91, abs=0.02)
        assert st5['sharpe'] == pytest.approx(0.129, abs=0.005)
        assert st5['nw_lag'] == 9
        assert st5['passes_t_2'] is False

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        """Cost curve +1.00 -> +0.91 (0.5bp) -> +0.81 (1bp), never near 2; rate-invariant
        (same t at rf=0 and 4.5%)."""
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(0.81, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(0.91, abs=0.02)


def _straddle_scenario(
    price_path: list[tuple[str, float]],
    call_path: list[tuple[float, float, float, float]],
    put_path: list[tuple[float, float, float, float]],
    call_strike: float,
    put_strike: float,
) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
    """One-cycle two-leg synthetic (dates, prices, store). Last date = expiration.
    call_path[i] / put_path[i] = (bid, ask, mid, delta) for day i."""
    exp = price_path[-1][0]
    dte0 = len(price_path) - 1
    store: dict[str, dict[str, Any]] = {}
    dates: list[str] = []
    prices: list[float] = []
    for i, ((date, px), c, p) in enumerate(zip(price_path, call_path, put_path)):
        dates.append(date)
        prices.append(px)
        cc = (dte0 - i, c[3], c[0], c[1], c[2], exp, call_strike, 'C')
        pc = (dte0 - i, p[3], p[0], p[1], p[2], exp, put_strike, 'P')
        store[date] = {'candidates': [cc, pc],
                       'marks': {'C': (c[0], c[1], c[2], c[3]), 'P': (p[0], p[1], p[2], p[3])}}
    return dates, prices, store


class TestStraddleMechanics:
    """Synthetic, always-run checks of the two-leg run_real_straddle_overlay: both
    premiums collected, the combined-delta hedge offsets a move, and BOTH legs settle
    at expiry. Pin the §7 straddle MECHANISM regardless of real data. select_straddle
    picks both legs at one expiration (a true straddle, not a diagonal)."""

    def test_selects_both_legs_same_expiry(self) -> None:
        d, _, store = _straddle_scenario(
            [('2020-01-01', 100.0), ('2020-01-31', 100.0)],
            [(2.0, 2.1, 2.05, 0.50), (0.0, 0.0, 0.0, 0.0)],
            [(2.0, 2.1, 2.05, -0.50), (0.0, 0.0, 0.0, 0.0)], 100.0, 100.0)
        pick = select_straddle(store[d[0]], 30, 0.50, -0.50)
        assert pick is not None
        call, put = pick
        assert call[1] > 0 and put[1] < 0 and call[5] == put[5]  # opposite signs, one expiry

    def test_flat_market_harvests_both_premiums(self) -> None:
        """Stock dead flat at the strike; both legs decay to worthless. The
        delta-neutral short straddle (combined delta ~0 -> ~0 hedge) keeps ~the whole
        two-leg premium."""
        days = [f'2020-01-0{i + 1}' for i in range(6)]
        price_path = [(d, 100.0) for d in days]
        call_path = [(2.0, 2.1, 2.05, 0.50), (1.55, 1.65, 1.60, 0.45),
                     (1.05, 1.15, 1.10, 0.35), (0.55, 0.65, 0.60, 0.20),
                     (0.15, 0.25, 0.20, 0.08), (0.0, 0.0, 0.0, 0.0)]
        put_path = [(2.0, 2.1, 2.05, -0.50), (1.55, 1.65, 1.60, -0.45),
                    (1.05, 1.15, 1.10, -0.35), (0.55, 0.65, 0.60, -0.20),
                    (0.15, 0.25, 0.20, -0.08), (0.0, 0.0, 0.0, 0.0)]
        dates, prices, store = _straddle_scenario(price_path, call_path, put_path, 100.0, 100.0)
        s, _, _ = run_real_straddle_overlay(
            dates, prices, store,
            {'call_delta': 0.50, 'put_delta': -0.50, 'capital': 100_000,
             'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        assert s['num_straddles_sold'] == 1
        assert s['net_pnl'] > 0
        assert s['net_pnl'] == pytest.approx(s['total_premium_collected'], rel=0.02)

    def test_hedge_offsets_an_up_move(self) -> None:
        """Stock trends up through the strikes: the combined delta -> +1, the hedge
        goes LONG and captures the rise, so the net loss is a small gamma cost, not
        the full naked-straddle assignment loss (call finishes ITM by 10)."""
        days = ['2020-02-0' + str(i + 1) for i in range(5)]
        price_path = [(days[0], 100.0), (days[1], 102.0), (days[2], 105.0),
                      (days[3], 108.0), (days[4], 110.0)]
        call_path = [(2.0, 2.1, 2.05, 0.50), (3.0, 3.1, 3.05, 0.62),
                     (4.5, 4.6, 4.55, 0.80), (6.4, 6.6, 6.50, 0.95), (8.0, 8.1, 8.05, 1.0)]
        put_path = [(2.0, 2.1, 2.05, -0.50), (1.2, 1.3, 1.25, -0.35),
                    (0.5, 0.6, 0.55, -0.18), (0.1, 0.2, 0.15, -0.05), (0.0, 0.05, 0.02, -0.01)]
        dates, prices, store = _straddle_scenario(price_path, call_path, put_path, 100.0, 100.0)
        s, _, _ = run_real_straddle_overlay(
            dates, prices, store,
            {'call_delta': 0.50, 'put_delta': -0.50, 'capital': 100_000,
             'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        shares = s['num_contracts'] * 100
        prem_per_share = s['total_premium_collected'] / shares  # ~4 (both legs)
        naked = (prem_per_share - (110.0 - 100.0)) * shares  # call ITM by 10 => big loss
        assert naked < 0
        assert abs(s['net_pnl']) < 0.5 * abs(naked)

    def test_put_leg_settles_on_a_down_move(self) -> None:
        """Stock falls below the strikes: the put leg finishes ITM and IS paid (the
        combined delta -> -1, hedge SHORT). The hedged net loss is a fraction of the
        naked straddle's, confirming the second leg settles and hedges correctly."""
        days = ['2020-03-0' + str(i + 1) for i in range(5)]
        price_path = [(days[0], 100.0), (days[1], 97.0), (days[2], 94.0),
                      (days[3], 92.0), (days[4], 90.0)]
        call_path = [(2.0, 2.1, 2.05, 0.50), (1.0, 1.1, 1.05, 0.35),
                     (0.4, 0.5, 0.45, 0.18), (0.1, 0.2, 0.15, 0.06), (0.0, 0.05, 0.02, 0.01)]
        put_path = [(2.0, 2.1, 2.05, -0.50), (3.2, 3.3, 3.25, -0.66),
                    (5.0, 5.1, 5.05, -0.84), (7.0, 7.1, 7.05, -0.95), (10.0, 10.1, 10.05, -1.0)]
        dates, prices, store = _straddle_scenario(price_path, call_path, put_path, 100.0, 100.0)
        s, _, _ = run_real_straddle_overlay(
            dates, prices, store,
            {'call_delta': 0.50, 'put_delta': -0.50, 'capital': 100_000,
             'risk_free_rate': 0.0, 'hedge_cost_bps': 0.0})
        shares = s['num_contracts'] * 100
        prem_per_share = s['total_premium_collected'] / shares
        naked = (prem_per_share - (100.0 - 90.0)) * shares  # put ITM by 10
        assert naked < 0
        assert abs(s['net_pnl']) < 0.5 * abs(naked)


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz)')
class TestSpyStraddleSecondary:
    """Pin the §7 ATM-straddle SECONDARY on real SPY chains (calls merged with puts).
    REPORTED, NEVER PROMOTED (prereg §7): a secondary that cannot change the §5 primary
    (short-put) verdict. Span 2010-12-01 -> 2026-06-05.

    Result: the full variance harvester (short ~0.50d call + short ~-0.50d put, same
    expiry, hold-to-expiry, net-delta hedged) clears MORE than the put wing alone
    (gross Newey-West t +0.90, net-0.5bp +0.72) but still does NOT reach t=2 -- a
    richer null. Rate-invariant, delta-neutral (corr -0.03 to SPY), 16.9% drawdown
    (short both wings); 2022's grinding bear is the biggest single drag (-$30.5K).
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_DAILIES, extra_paths=[_SPY_PUTS], start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_straddle_overlay(
            dates, prices, store,
            {'dte': 30, 'capital': 100_000, 'call_delta': 0.50, 'put_delta': -0.50,
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        """175 straddles (8 contracts), 55.2% win, $1.54M two-leg premium;
        frictionless vol-P&L +$39.1K, 16.6% drawdown."""
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 8
        assert s['num_straddles_sold'] == 175
        assert s['win_rate'] == pytest.approx(55.2, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(39_070.76, abs=3.0)
        assert s['max_drawdown_pct'] == pytest.approx(16.55, abs=0.05)

    def test_secondary_null(self, market: Any) -> None:
        """Richer than the put wing alone but still null: gross t +0.90, net-0.5bp
        +0.72 -- never clears t=2. Cannot change the §5 verdict (§7)."""
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(0.90, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(0.72, abs=0.02)
        assert st5['sharpe'] == pytest.approx(0.128, abs=0.005)
        assert st5['nw_lag'] == 9
        assert st5['passes_t_2'] is False

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        """+0.90 gross -> +0.72 (0.5bp) -> +0.54 (1bp), never near 2; rate-invariant."""
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(0.54, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(0.72, abs=0.02)


@pytest.mark.skipif(not _HAVE_IWM, reason='needs iwm_option_dailies.csv or its .gz twin')
class TestIwmStraddleSecondary:
    """Pin the §7 ATM-straddle SECONDARY on real IWM chains (both wings in one file).
    REPORTED, NEVER PROMOTED. Span 2010-12-01 -> 2026-06-05.

    Result: IWM's straddle is the strongest variance harvester of the set (gross t
    +1.42, net-0.5bp +1.28, +$62.9K gross vol-P&L) but STILL does not reach t=2.
    Delta-neutral (corr -0.01), 24.7% drawdown; 2021 is the big harvest (+$36.3K).
    Reinforces the primary null: even the full strip, on the naive index, isn't
    significant net of cost over this span.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_IWM_DAILIES, start=CHAIN_CLEAN_START['IWM'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('IWM', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_straddle_overlay(
            dates, prices, store,
            {'dte': 30, 'capital': 100_000, 'call_delta': 0.50, 'put_delta': -0.50,
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        """169 straddles (13 contracts), 64.9% win, $1.61M premium; frictionless
        vol-P&L +$62.9K, 24.2% drawdown."""
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 13
        assert s['num_straddles_sold'] == 169
        assert s['win_rate'] == pytest.approx(64.9, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(62_862.08, abs=3.0)
        assert s['max_drawdown_pct'] == pytest.approx(24.18, abs=0.05)

    def test_secondary_null(self, market: Any) -> None:
        """The strongest harvester of the set, still null: gross t +1.42, net-0.5bp
        +1.28 -- below t=2. Cannot change the §5 verdict (§7)."""
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.42, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(1.28, abs=0.02)
        assert st5['sharpe'] == pytest.approx(0.251, abs=0.005)
        assert st5['nw_lag'] == 9
        assert st5['passes_t_2'] is False

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        """+1.42 gross -> +1.28 (0.5bp) -> +1.15 (1bp), never reaching 2; rate-invariant."""
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(1.15, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.28, abs=0.02)
