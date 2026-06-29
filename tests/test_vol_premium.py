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
from common.paths import DATA_DIR
from typing import Any

import pandas as pd
import pytest

from search.edge_search import STRUCTURE_GRAMMAR
from realchains.vol_premium import (
    STRUCTURE_SPECS,
    _leg_intrinsic,
    bs_gamma,
    bs_price,
    bs_vega,
    implied_vol,
    run_real_calendar_overlay,
    run_real_credit_spread_overlay,
    run_real_iron_condor_overlay,
    run_real_risk_reversal_overlay,
    run_real_short_vol_overlay,
    run_real_straddle_overlay,
    run_real_strangle_overlay,
    select_calendar,
    select_credit_spread,
    select_iron_condor,
    select_put_entry,
    select_straddle,
    short_vol_statistics,
    structure_greek_signature,
)

_SPY_DAILIES = os.path.join(DATA_DIR, 'spy_option_dailies.csv')
_HAVE_SPY = os.path.exists(_SPY_DAILIES) or os.path.exists(_SPY_DAILIES + '.gz')
_SPY_PUTS = os.path.join(DATA_DIR, 'spy_option_dailies_puts.csv')
_HAVE_SPY_PUTS = os.path.exists(_SPY_PUTS) or os.path.exists(_SPY_PUTS + '.gz')
_IWM_DAILIES = os.path.join(DATA_DIR, 'iwm_option_dailies.csv')
_HAVE_IWM = os.path.exists(_IWM_DAILIES) or os.path.exists(_IWM_DAILIES + '.gz')
_NVDA_DAILIES = os.path.join(DATA_DIR, 'nvda_option_dailies.csv')
_HAVE_NVDA = os.path.exists(_NVDA_DAILIES) or os.path.exists(_NVDA_DAILIES + '.gz')


def _have(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(path + '.gz')


# MSFT / QQQ daily CALLS (canonical + backfill) — for the call-wing VRP cross-section.
# The pinned spans start at 2010-05 (MSFT) / 2011-03 (QQQ), so both files are required.
_MSFT_DAILIES = os.path.join(DATA_DIR, 'msft_option_dailies.csv')
_MSFT_BACKFILL = os.path.join(DATA_DIR, 'msft_option_dailies_2008_2016.csv')
_HAVE_MSFT = _have(_MSFT_DAILIES) and _have(_MSFT_BACKFILL)
_QQQ_DAILIES = os.path.join(DATA_DIR, 'qqq_option_dailies.csv')
_QQQ_BACKFILL = os.path.join(DATA_DIR, 'qqq_option_dailies_2011_2016.csv')
_HAVE_QQQ = _have(_QQQ_DAILIES) and _have(_QQQ_BACKFILL)
# MSFT / QQQ PUT wings — for the put + straddle cross-section extensions.
_MSFT_PUTS = os.path.join(DATA_DIR, 'msft_option_dailies_puts.csv')
_HAVE_MSFT_PUTS = _have(_MSFT_PUTS)
_QQQ_PUTS = os.path.join(DATA_DIR, 'qqq_option_dailies_puts.csv')
_HAVE_QQQ_PUTS = _have(_QQQ_PUTS)


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
    span (REGISTERED_CLEAN_START['SPY'] — frozen: this +2.54 is the committed
    benchmark of the registered put-side prereg's mechanism clause, so it stays on
    the as-registered span even though the live SPY hygiene boundary moved to
    2010-05-17); rf credited on the cash collateral;
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
        from realchains.real_cc_backtest import REGISTERED_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_DAILIES, start=REGISTERED_CLEAN_START['SPY'])  # registration-frozen benchmark
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
    bid, hedged with SHORT stock; full 2010-12 -> 2026-06 span (REGISTERED_CLEAN_START['SPY'],
    registration-frozen) -- the exact mirror of the pinned 0.25-delta CALL wing (TestSpyShortVolRegression,
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
        from realchains.real_cc_backtest import REGISTERED_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_PUTS, start=REGISTERED_CLEAN_START['SPY'])  # registration-frozen span
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
        from realchains.real_cc_backtest import REGISTERED_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_IWM_DAILIES, start=REGISTERED_CLEAN_START['IWM'])  # registration-frozen span
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
        from realchains.real_cc_backtest import REGISTERED_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_DAILIES, extra_paths=[_SPY_PUTS], start=REGISTERED_CLEAN_START['SPY'])  # registration-frozen span
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
        from realchains.real_cc_backtest import REGISTERED_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_IWM_DAILIES, start=REGISTERED_CLEAN_START['IWM'])  # registration-frozen span
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


# ---------------------------------------------------------------------------
# EXPLORATORY call-wing VRP cross-section (NOT registered). Extends the pinned
# SPY call wing (TestSpyShortVolRegression, +2.54 gross / +2.25 net-0.5bp) to the
# other tickers that already carry calls -- no new data, just measurement. The
# finding: the call-wing delta-hedged premium is an INDEX, COST-FRAGILE phenomenon.
# SPY clears the bar to 0.5bp; QQQ is gross-significant but dies at cost; IWM is
# null; the single name (MSFT) LOSES with a catastrophic drawdown. Pinned so the
# cross-section isn't re-derived; exploratory, not a registered verdict.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_QQQ, reason='needs qqq_option_dailies.csv + its 2011_2016 backfill (or .gz)')
class TestQqqShortVolRegression:
    """EXPLORATORY: the delta-neutral short 0.25d CALL on real QQQ chains (canonical
    + 2011_2016 backfill), 2011-03-23 -> 2026-06-05. The closest thing to a partial
    replication of the SPY call wing: gross Newey-West t +2.07 (clears 2), but it
    dies at the 0.5bp headline cost (+1.88) -- the SPY signal (+2.54 -> +2.25) is
    stronger and cost-surviving; QQQ's is gross-only. Rate-invariant, delta-neutral
    (corr +0.15). Not registered, cannot be promoted."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_QQQ_DAILIES, extra_paths=[_QQQ_BACKFILL], start=CHAIN_CLEAN_START.get('QQQ'))
        days = sorted(store)
        dates, prices = load_unadjusted_prices('QQQ', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': 0.25, 'dte': 30, 'capital': 100_000, 'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 17
        assert s['num_calls_sold'] == 166
        assert s['win_rate'] == pytest.approx(68.5, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(69_381.23, abs=3.0)
        assert s['max_drawdown_pct'] == pytest.approx(13.73, abs=0.05)

    def test_gross_significant_but_dies_at_cost(self, market: Any) -> None:
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(2.07, abs=0.02)
        assert st0['passes_t_2'] is True   # gross clears the bar
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(1.88, abs=0.02)
        assert st5['passes_t_2'] is False  # but dies at the 0.5bp headline cost
        assert st5['sharpe'] == pytest.approx(0.362, abs=0.005)

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(1.70, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.88, abs=0.02)


@pytest.mark.skipif(not _HAVE_IWM, reason='needs iwm_option_dailies.csv or its .gz twin')
class TestIwmShortVolRegression:
    """EXPLORATORY: the delta-neutral short 0.25d CALL on real IWM chains (both-wing
    file; select_entry takes the calls), 2010-12-01 -> 2026-06-05. Null: gross
    Newey-West t +1.37, net-0.5bp +1.18 -- never clears 2. Delta-neutral (corr +0.15),
    small 7% drawdown. The call-wing premium is absent on the small-cap index. Not
    registered, cannot be promoted."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_IWM_DAILIES, start=CHAIN_CLEAN_START['IWM'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('IWM', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': 0.25, 'dte': 30, 'capital': 100_000, 'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 13
        assert s['num_calls_sold'] == 169
        assert s['win_rate'] == pytest.approx(78.0, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(19_777.18, abs=3.0)
        assert s['max_drawdown_pct'] == pytest.approx(6.83, abs=0.05)

    def test_null(self, market: Any) -> None:
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.37, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(1.18, abs=0.02)
        assert st5['sharpe'] == pytest.approx(0.257, abs=0.005)
        assert st5['passes_t_2'] is False

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(0.98, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(1.18, abs=0.02)


@pytest.mark.skipif(not _HAVE_MSFT, reason='needs msft_option_dailies.csv + its 2008_2016 backfill (or .gz)')
class TestMsftShortVolRegression:
    """EXPLORATORY: the delta-neutral short 0.25d CALL on the single name MSFT
    (canonical + 2008_2016 backfill), 2010-05-10 -> 2026-04-10. The single-name
    disaster: it LOSES (gross Newey-West t -0.26, net-0.5bp -0.37, net P&L -$58K) with
    a catastrophic 74.6% drawdown (equity peaked $114K, troughed $29K) as MSFT ran
    12.8x. Delta-neutral (corr +0.20) -- the loss is genuine short-gamma bleed on a
    violently-trending single name, not a hedge bug (the same frozen engine gives the
    sane SPY +2.54). Confirms the call-wing premium is an INDEX phenomenon, destructive
    on single names. Not registered."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_MSFT_DAILIES, extra_paths=[_MSFT_BACKFILL], start=CHAIN_CLEAN_START['MSFT'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-04-11')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_short_vol_overlay(
            dates, prices, store,
            {'target_delta': 0.25, 'dte': 30, 'capital': 100_000, 'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline_is_a_loss(self, market: Any) -> None:
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 34
        assert s['num_calls_sold'] == 172
        assert s['win_rate'] == pytest.approx(72.5, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(-18_202.17, abs=5.0)  # NEGATIVE
        assert s['net_pnl'] == pytest.approx(-48_198.61, abs=5.0)        # loses money
        assert s['max_drawdown_pct'] == pytest.approx(68.39, abs=0.1)    # catastrophic

    def test_null_negative(self, market: Any) -> None:
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-0.26, abs=0.02)
        assert st0['passes_t_2'] is False
        s5, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(-0.37, abs=0.02)
        assert s5['max_drawdown_pct'] == pytest.approx(74.58, abs=0.1)
        assert s5['alpha_vs_cash'] == pytest.approx(-26_086.06, abs=5.0)

    def test_cost_curve_and_rate_invariant(self, market: Any) -> None:
        assert self._run(market, 1.0)[1]['t_stat_newey_west'] == pytest.approx(-0.48, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-0.37, abs=0.02)


def _condor_scenario(
    price_path: list[tuple[str, float]],
    legs: dict[str, tuple[float, list[tuple[float, float, float, float]]]],
) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
    """One-cycle 4-leg synthetic. legs maps contractID -> (strike, path) where
    path[i] = (bid, ask, mid, delta) per day. Last date = expiration."""
    exp = price_path[-1][0]
    dte0 = len(price_path) - 1
    store: dict[str, dict[str, Any]] = {}
    dates: list[str] = []
    prices: list[float] = []
    for i, (date, px) in enumerate(price_path):
        dates.append(date)
        prices.append(px)
        cands, marks = [], {}
        for cid, (strike, path) in legs.items():
            b, a, m, dl = path[i]
            cands.append((dte0 - i, dl, b, a, m, exp, strike, cid))
            marks[cid] = (b, a, m, dl)
        store[date] = {'candidates': cands, 'marks': marks}
    return dates, prices, store


def _flat_condor_legs() -> dict[str, tuple[float, list[tuple[float, float, float, float]]]]:
    """A 6-day cycle: short 25d strangle (95 put / 105 call) + 10d wings (90 / 110),
    all decaying to worthless. Entry-day deltas put select_iron_condor on the right
    strikes."""
    decay = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    wdecay = [0.4, 0.32, 0.24, 0.16, 0.08, 0.0]
    sc = [(round(v, 2), round(v + 0.1, 2), round(v + 0.05, 2), d) for v, d in zip(decay, [0.25, 0.22, 0.18, 0.12, 0.06, 0.0])]
    lc = [(round(v, 2), round(v + 0.1, 2), round(v + 0.05, 2), d) for v, d in zip(wdecay, [0.10, 0.08, 0.06, 0.04, 0.02, 0.0])]
    sp = [(round(v, 2), round(v + 0.1, 2), round(v + 0.05, 2), d) for v, d in zip(decay, [-0.25, -0.22, -0.18, -0.12, -0.06, 0.0])]
    lp = [(round(v, 2), round(v + 0.1, 2), round(v + 0.05, 2), d) for v, d in zip(wdecay, [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0])]
    return {'SC': (105.0, sc), 'LC': (110.0, lc), 'SP': (95.0, sp), 'LP': (90.0, lp)}


class TestIronCondorMechanics:
    """Synthetic, always-run checks of the four-leg run_real_iron_condor_overlay:
    correct leg selection, the net credit kept when price finishes inside the short
    strikes, and the loss CAPPED by the long wing on a breach. Pin the §-exploratory
    iron-condor MECHANISM regardless of real data."""

    def test_selects_four_ordered_legs(self) -> None:
        days = [f'2020-01-0{i + 1}' for i in range(6)]
        dates, _, store = _condor_scenario([(d, 100.0) for d in days], _flat_condor_legs())
        pick = select_iron_condor(store[dates[0]], 30, 0.25, 0.10)
        assert pick is not None
        sc, lc, sp, lp = pick
        assert lp[6] < sp[6] < sc[6] < lc[6]          # long put < short put < short call < long call
        assert sc[1] > 0 and lc[1] > 0 and sp[1] < 0 and lp[1] < 0
        assert abs(sc[1]) > abs(lc[1]) and abs(sp[1]) > abs(lp[1])  # shorts nearer the money

    def test_inside_strikes_keeps_net_credit(self) -> None:
        """Price finishes between the short strikes; all four legs expire worthless,
        so the condor keeps ~its whole net credit (the win case)."""
        days = [f'2020-02-0{i + 1}' for i in range(6)]
        dates, prices, store = _condor_scenario([(d, 100.0) for d in days], _flat_condor_legs())
        s, _, _ = run_real_iron_condor_overlay(
            dates, prices, store,
            {'short_delta': 0.25, 'wing_delta': 0.10, 'capital': 100_000, 'risk_free_rate': 0.0})
        assert s['num_condors_sold'] == 1
        assert s['net_pnl'] > 0
        assert s['net_pnl'] == pytest.approx(s['total_premium_collected'], rel=0.02)

    def test_loss_is_capped_by_the_wing(self) -> None:
        """Price crashes through the long put: the loss is CAPPED at (short_put −
        long_put width) − net credit, a fraction of the naked short put's loss."""
        days = [f'2020-03-0{i + 1}' for i in range(6)]
        price_path = [(days[0], 100.0)] + [(days[i], 100.0 - 4 * i) for i in range(1, 6)]  # -> 80
        dates, prices, store = _condor_scenario(price_path, _flat_condor_legs())
        s, _, _ = run_real_iron_condor_overlay(
            dates, prices, store,
            {'short_delta': 0.25, 'wing_delta': 0.10, 'capital': 100_000, 'risk_free_rate': 0.0})
        shares = s['num_contracts'] * 100
        credit_ps = s['total_premium_collected'] / shares
        spread_w = 95.0 - 90.0  # short put 95, long put 90
        capped_loss = (credit_ps - spread_w) * shares          # ~ -(5 - credit)*shares
        naked_loss = (credit_ps - (95.0 - 80.0)) * shares       # short put ITM by 15, no wing
        assert s['net_pnl'] == pytest.approx(capped_loss, rel=0.05)
        assert s['net_pnl'] < 0 and abs(s['net_pnl']) < abs(naked_loss)  # wing capped it


class TestCreditSpreadMechanics:
    """Synthetic, always-run checks of the two-leg run_real_credit_spread_overlay
    (widening 3, the first CARRY structure — the put half of the iron condor): correct
    two-put-leg selection, the net credit kept when price finishes above the short
    strike, and the loss CAPPED by the long wing on a breach. Reuses the condor
    scenario (its SP=95/0.25d + LP=90/0.10d are exactly the credit spread's two legs)."""

    def test_selects_two_ordered_put_legs(self) -> None:
        days = [f'2020-01-0{i + 1}' for i in range(6)]
        dates, _, store = _condor_scenario([(d, 100.0) for d in days], _flat_condor_legs())
        pick = select_credit_spread(store[dates[0]], 30, 0.25, 0.10)
        assert pick is not None
        sp, lp = pick
        assert lp[6] < sp[6]                          # long-put wing strike below the short put
        assert sp[1] < 0 and lp[1] < 0                # both puts (negative vendor delta)
        assert abs(sp[1]) > abs(lp[1])               # the short sits nearer the money

    def test_above_short_strike_keeps_net_credit(self) -> None:
        """Price stays above the short strike; both puts expire worthless, so the
        spread keeps ~its whole net credit (the win case)."""
        days = [f'2020-02-0{i + 1}' for i in range(6)]
        dates, prices, store = _condor_scenario([(d, 100.0) for d in days], _flat_condor_legs())
        s, trades, _ = run_real_credit_spread_overlay(
            dates, prices, store,
            {'short_delta': 0.25, 'wing_delta': 0.10, 'capital': 100_000, 'risk_free_rate': 0.0})
        assert len(trades) > 0
        assert s['num_credit_spreads_sold'] == 1
        assert s['net_pnl'] > 0
        assert s['net_pnl'] == pytest.approx(s['total_premium_collected'], rel=0.02)

    def test_loss_is_bounded_by_the_wing(self) -> None:
        """Price crashes through the long put: the loss is BOUNDED — the long-put wing
        caps the option-leg payoff at the (short − long) width, so the realized loss
        stays a fraction of the naked short put's. Unlike the static iron condor the
        credit spread is combined-DELTA-HEDGED (the campaign config), so the hedge
        offsets part of the directional loss too — the exact loss isn't the unhedged
        width formula, but it is well inside the naked-short-put loss either way."""
        days = [f'2020-03-0{i + 1}' for i in range(6)]
        price_path = [(days[0], 100.0)] + [(days[i], 100.0 - 4 * i) for i in range(1, 6)]  # -> 80
        dates, prices, store = _condor_scenario(price_path, _flat_condor_legs())
        s, _, _ = run_real_credit_spread_overlay(
            dates, prices, store,
            {'short_delta': 0.25, 'wing_delta': 0.10, 'capital': 100_000, 'risk_free_rate': 0.0})
        shares = s['num_contracts'] * 100
        credit_ps = s['total_premium_collected'] / shares
        spread_w = 95.0 - 90.0  # short put 95 / long put 90 -> max option-leg loss = width − credit
        worst_spread_loss = (credit_ps - spread_w) * shares     # the defined-risk floor on the legs
        naked_loss = (credit_ps - (95.0 - 80.0)) * shares       # short put ITM by 15, no wing
        # the wing caps the structure: the realized loss is far smaller than the naked short put's
        assert s['net_pnl'] < 0 and abs(s['net_pnl']) < abs(naked_loss)
        # and no worse than ~the defined-risk floor (a small slack for hedge cost on the synthetic)
        assert s['net_pnl'] >= worst_spread_loss - abs(0.10 * worst_spread_loss)


class TestCalendarMechanics:
    """Synthetic, always-run checks of select_calendar — the only selector that
    forces a SECOND, later expiration (widening 4, the TERM family). Pin the
    selection CONTRACT regardless of real data: the far leg is matched by the near
    leg's exact STRIKE on a strictly-later expiration, and only when that expiry
    clears the `min_gap_dte` term-separation floor. The staggered-settlement engine
    path these legs drive is exercised end-to-end by the dataset-gated equivalence
    and campaign tests; here we pin the selector in isolation."""

    @staticmethod
    def _cand(dte: int, delta: float, strike: float, exp: str,
              bid: float = 1.0) -> tuple[Any, ...]:
        # candidate tuple: (dte, delta, bid, ask, mid, expiration, strike, contractID)
        return (dte, delta, bid, bid + 0.2, bid + 0.1, exp, strike, f'C{exp}-{strike}')

    def test_picks_far_leg_at_same_strike_later_expiry(self) -> None:
        """The near leg is an ~ATM call (select_entry, ~0.50 delta); the far leg is
        the SAME-strike call on a later expiration ≥ min_gap_dte beyond the near."""
        day = {'candidates': [
            self._cand(30, 0.50, 100.0, '2024-02-16'),   # the near ATM call (strike 100)
            self._cand(30, 0.30, 105.0, '2024-02-16'),   # near, wrong strike: not the far match
            self._cand(60, 0.55, 100.0, '2024-03-15'),   # the far same-strike call, +30 DTE
        ], 'marks': {}}
        pick = select_calendar(day, near_dte=30, far_dte=60, target_delta=0.50)
        assert pick is not None
        near, far = pick
        assert near[6] == far[6] == 100.0          # SAME strike (a true calendar)
        assert far[5] > near[5]                     # far expiry strictly later
        assert far[0] - near[0] >= 30               # clears the min_gap_dte floor

    def test_none_when_far_leg_too_close(self) -> None:
        """A far call at the same strike but only a few DTE past the near reads
        vega-neutral, not the long-vega calendar — the min_gap_dte floor rejects it."""
        day = {'candidates': [
            self._cand(30, 0.50, 100.0, '2024-02-16'),   # near ATM call
            self._cand(40, 0.52, 100.0, '2024-02-23'),   # same strike but only +10 DTE
        ], 'marks': {}}
        assert select_calendar(day, near_dte=30, far_dte=60, target_delta=0.50,
                               min_gap_dte=30) is None

    def test_none_when_no_far_strike_listed(self) -> None:
        """No later expiration carries the near leg's exact strike (MSFT's real
        failure mode) — the same-strike calendar can't be built, so None."""
        day = {'candidates': [
            self._cand(30, 0.50, 100.0, '2024-02-16'),   # near ATM call at strike 100
            self._cand(60, 0.55, 105.0, '2024-03-15'),   # far call, DIFFERENT strike (105)
        ], 'marks': {}}
        assert select_calendar(day, near_dte=30, far_dte=60, target_delta=0.50) is None


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz)')
class TestSpyIronCondorExploratory:
    """EXPLORATORY (not registered, not even a prereg secondary): a daily short IRON
    CONDOR on real SPY chains (calls merged with puts), 25d shorts / 10d wings, 30 DTE,
    hold-to-expiry, NO stock hedge -- the defined-risk retail structure.

    Verdict: it LOSES vs cash. At realistic bid/ask fills the excess-over-cash is
    -$47.6K (Newey-West t -1.08, Sharpe -0.21); even frictionless (mid) it is -0.89.
    Total P&L is +$54.5K, but that is ENTIRELY rf interest on idle collateral -- the
    condor itself underperformed T-bills. The long wings DID cap the per-event tail
    (17.1% max drawdown vs the naked single name's 74.6%), but the structure still
    bled: thin OTM premium minus the wing cost minus four legs of bid/ask minus the
    unhedged directional losses. Worse than the delta-hedged SPY straddle (+0.72) and
    every wing. Rate-invariant. Pinned so the exploration isn't re-derived.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_SPY_DAILIES, extra_paths=[_SPY_PUTS], start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, fill: str, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_iron_condor_overlay(
            dates, prices, store,
            {'dte': 30, 'capital': 100_000, 'short_delta': 0.25, 'wing_delta': 0.10,
             'fill': fill, 'risk_free_rate': rf})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_headline(self, market: Any) -> None:
        s, _ = self._run(market, 'bid_ask')
        assert s['num_condors_sold'] == 175
        assert s['win_rate'] == pytest.approx(59.8, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(-47_600.01, abs=10.0)  # loses vs cash
        assert s['net_pnl'] == pytest.approx(54_526.88, abs=10.0)         # positive ONLY via rf
        assert s['max_drawdown_pct'] == pytest.approx(17.07, abs=0.1)

    def test_loses_vs_cash(self, market: Any) -> None:
        _, st = self._run(market, 'bid_ask')
        assert st['t_stat_newey_west'] == pytest.approx(-1.08, abs=0.02)
        assert st['sharpe'] == pytest.approx(-0.207, abs=0.005)
        assert st['nw_lag'] == 9
        assert st['passes_t_2'] is False
        _, st_mid = self._run(market, 'mid')
        assert st_mid['t_stat_newey_west'] == pytest.approx(-0.89, abs=0.02)  # negative even frictionless

    def test_rate_invariant(self, market: Any) -> None:
        _, st0 = self._run(market, 'bid_ask', rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-1.08, abs=0.02)


# ---------------------------------------------------------------------------
# EXPLORATORY put + straddle cross-section on MSFT/QQQ (NOT registered). Extends
# the registered SPY/IWM put (TestSpy/IwmShortPutRegression) and §7 straddle
# (TestSpy/IwmStraddleSecondary) to the two tickers whose puts were fetched later.
# Completes the 4-for-4 picture: only the SPY call wing clears t=2 net of cost;
# the put wing is null-to-negative everywhere, and the single-name (MSFT) straddle
# is an outright blow-up. Pinned so the cross-section isn't re-derived.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_MSFT_PUTS, reason='needs msft_option_dailies_puts.csv or its .gz twin')
class TestMsftShortPutExploratory:
    """EXPLORATORY: short -0.25d PUT on real MSFT chains, 2010-05-10 -> 2026-04-10.
    NEGATIVE: gross Newey-West t -0.75, net-0.5bp -0.84 (vol-P&L -$58K -> -$65K), 28%
    drawdown. The single-name put wing loses; rate-invariant. Not registered."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_MSFT_PUTS, start=CHAIN_CLEAN_START['MSFT'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-04-11')
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
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 34
        assert s['num_calls_sold'] == 172
        assert s['win_rate'] == pytest.approx(88.3, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(-58_419.28, abs=5.0)
        assert s['max_drawdown_pct'] == pytest.approx(27.71, abs=0.1)

    def test_negative_null(self, market: Any) -> None:
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-0.75, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(-0.84, abs=0.02)
        assert st5['sharpe'] == pytest.approx(-0.186, abs=0.005)
        _, st0rf = self._run(market, 0.5, rf=0.0)
        assert st0rf['t_stat_newey_west'] == pytest.approx(-0.84, abs=0.02)  # rate-invariant


@pytest.mark.skipif(not _HAVE_QQQ_PUTS, reason='needs qqq_option_dailies_puts.csv or its .gz twin')
class TestQqqShortPutExploratory:
    """EXPLORATORY: short -0.25d PUT on real QQQ chains, 2011-03-23 -> 2026-06-05.
    NEGATIVE: gross Newey-West t -0.92, net-0.5bp -1.00 (vol-P&L -$48K -> -$52K), 21%
    drawdown. Even on the index, the put wing loses on QQQ. Not registered."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_QQQ_PUTS, start=CHAIN_CLEAN_START.get('QQQ'))
        days = sorted(store)
        dates, prices = load_unadjusted_prices('QQQ', days[0], '2026-06-06')
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
        s, _ = self._run(market, 0.0)
        assert s['num_contracts'] == 17
        assert s['num_calls_sold'] == 166
        assert s['win_rate'] == pytest.approx(87.9, abs=0.1)
        assert s['alpha_vs_cash'] == pytest.approx(-47_745.40, abs=5.0)
        assert s['max_drawdown_pct'] == pytest.approx(20.92, abs=0.1)

    def test_negative_null(self, market: Any) -> None:
        _, st0 = self._run(market, 0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-0.92, abs=0.02)
        assert st0['passes_t_2'] is False
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(-1.00, abs=0.02)
        assert st5['nw_lag'] == 8
        _, st0rf = self._run(market, 0.5, rf=0.0)
        assert st0rf['t_stat_newey_west'] == pytest.approx(-1.00, abs=0.02)  # rate-invariant


@pytest.mark.skipif(not (_HAVE_MSFT and _HAVE_MSFT_PUTS),
                    reason='needs msft calls + backfill + puts (or .gz)')
class TestMsftStraddleExploratory:
    """EXPLORATORY: ATM short STRADDLE on real MSFT chains (calls + backfill + puts
    merged), 2010-05-10 -> 2026-04-10. The single-name BLOW-UP: gross Newey-West t
    -1.26, net-0.5bp -1.36, vol-P&L -$206K, and TOTAL net P&L is also negative
    (-$145K incl. rf) -- a 156.9% max drawdown means the account goes NEGATIVE (no
    modeled margin call). Short both wings on a stock that ran 12.8x, with the hedge
    chasing and the notional ballooning (fixed-contract sizing). The extreme of the
    single-name short-vol cautionary tale. Not registered."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_MSFT_DAILIES, extra_paths=[_MSFT_BACKFILL, _MSFT_PUTS],
                                 start=CHAIN_CLEAN_START['MSFT'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-04-11')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_straddle_overlay(
            dates, prices, store,
            {'dte': 30, 'capital': 100_000, 'call_delta': 0.50, 'put_delta': -0.50,
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_blowup(self, market: Any) -> None:
        s, st = self._run(market, 0.0)
        assert s['num_straddles_sold'] == 172
        assert st['t_stat_newey_west'] == pytest.approx(-1.26, abs=0.02)
        assert st['passes_t_2'] is False
        assert s['alpha_vs_cash'] == pytest.approx(-206_419.91, abs=20.0)
        assert s['net_pnl'] < 0                              # even total P&L is negative
        assert s['max_drawdown_pct'] == pytest.approx(156.86, abs=0.5)  # account goes negative

    def test_net_of_cost_and_rate_invariant(self, market: Any) -> None:
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(-1.36, abs=0.02)
        _, st0 = self._run(market, 0.5, rf=0.0)
        assert st0['t_stat_newey_west'] == pytest.approx(-1.36, abs=0.02)


@pytest.mark.skipif(not (_HAVE_QQQ and _HAVE_QQQ_PUTS),
                    reason='needs qqq calls + backfill + puts (or .gz)')
class TestQqqStraddleExploratory:
    """EXPLORATORY: ATM short STRADDLE on real QQQ chains (calls + backfill + puts
    merged), 2011-03-23 -> 2026-06-05. Null: gross Newey-West t +0.33, net-0.5bp
    +0.21 (vol-P&L +$31K -> +$19K), 53% drawdown. Positive but nowhere near t=2 --
    the index straddle harvests a little but does not clear the bar. Not registered."""

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, Any]]:
        from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
        store = load_chain_store(_QQQ_DAILIES, extra_paths=[_QQQ_BACKFILL, _QQQ_PUTS],
                                 start=CHAIN_CLEAN_START.get('QQQ'))
        days = sorted(store)
        dates, prices = load_unadjusted_prices('QQQ', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def _run(self, market: Any, bps: float, rf: float = 0.045) -> tuple[Any, Any]:
        dates, prices, store = market
        s, _, eq = run_real_straddle_overlay(
            dates, prices, store,
            {'dte': 30, 'capital': 100_000, 'call_delta': 0.50, 'put_delta': -0.50,
             'risk_free_rate': rf, 'hedge_cost_bps': bps})
        return s, short_vol_statistics(eq, s['capital'], rf=rf)

    def test_null(self, market: Any) -> None:
        s, st0 = self._run(market, 0.0)
        assert s['num_straddles_sold'] == 166
        assert st0['t_stat_newey_west'] == pytest.approx(0.33, abs=0.02)
        assert st0['passes_t_2'] is False
        assert s['alpha_vs_cash'] == pytest.approx(31_251.49, abs=20.0)
        assert s['max_drawdown_pct'] == pytest.approx(53.15, abs=0.5)
        _, st5 = self._run(market, 0.5)
        assert st5['t_stat_newey_west'] == pytest.approx(0.21, abs=0.02)
        assert st5['nw_lag'] == 8


# --------------------------------------------------------------------------- #
# Generic multi-leg structure engine (Ring 1 / Stage A of the "big idea desk")
# --------------------------------------------------------------------------- #
class TestGenericStructureEngineSpecs:
    """ALWAYS-RUN: the generic engine's structure specs + leg math. Post-Stage-B the three named
    overlays are thin DELEGATES to run_real_structure_overlay under STRUCTURE_SPECS; the
    dataset-gated TestGenericStructureEngineEquivalence pins each delegate enters + emits its
    complete rich summary (the byte-identical numbers carry through the registered regressions)."""

    def test_specs_are_well_formed(self) -> None:
        assert set(STRUCTURE_SPECS) == {'short_vol', 'straddle', 'iron_condor', 'strangle',
                                        'risk_reversal', 'credit_spread', 'calendar'}
        for name, spec in STRUCTURE_SPECS.items():
            assert callable(spec['select'])
            assert spec['entry_guard'] in ('each_short_positive', 'net_positive')
            assert spec['hedge_mode'] in ('per_leg_sign', 'combined', 'none')
            assert spec['management'] in ('hold', 'early_close_single')
            assert isinstance(spec['defaults'], dict)
        # the one per-overlay default that differs from the generic's own (1.0): the
        # straddle's frozen hedge_cost_bps is 0.5 — getting this wrong double-charges its
        # hedge (the bug Stage A's equivalence pass caught on GLD/XLE/EEM/NVDA).
        assert STRUCTURE_SPECS['straddle']['defaults'] == {'hedge_cost_bps': 0.5}
        assert STRUCTURE_SPECS['short_vol']['defaults'] == {}
        assert STRUCTURE_SPECS['iron_condor']['hedge_mode'] == 'none'

    def test_leg_intrinsic(self) -> None:
        call = {'right': 'call', 'strike': 100.0}
        put = {'right': 'put', 'strike': 100.0}
        assert _leg_intrinsic(call, 110.0) == 10.0 and _leg_intrinsic(call, 90.0) == 0.0
        assert _leg_intrinsic(put, 90.0) == 10.0 and _leg_intrinsic(put, 110.0) == 0.0


def _equiv_market(ticker: str, path: str, extra_paths=()):
    from realchains.real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
    store = load_chain_store(path, extra_paths=extra_paths, start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    return [d for d, _ in pairs], [p for _, p in pairs], store


@pytest.fixture(scope='module')
def spy_merged_market():
    # SPY calls + the separate puts file merged at load (extra_paths), the same way
    # run_registered_vrp loads the SPY straddle — so the put-leg straddle/iron-condor trade
    # on the canonical ticker rather than falling back to NVDA. Loaded once for the module.
    return _equiv_market('SPY', _SPY_DAILIES, extra_paths=[_SPY_PUTS])


_NAMED_OVERLAY = {'short_vol': run_real_short_vol_overlay,    # post-Stage-B: thin delegates
                  'straddle': run_real_straddle_overlay,     # to run_structure_via_spec
                  'iron_condor': run_real_iron_condor_overlay,
                  'strangle': run_real_strangle_overlay,     # widening 1 (the OTM straddle)
                  'risk_reversal': run_real_risk_reversal_overlay,  # widening 2 (SKEW)
                  'credit_spread': run_real_credit_spread_overlay,  # widening 3 (CARRY)
                  'calendar': run_real_calendar_overlay}     # widening 4 (TERM, two expirations)
_STRUCT_PARAMS = {'short_vol': {'target_delta': 0.25, 'dte': 30},
                  'straddle': {'dte': 30},
                  'iron_condor': {'dte': 30, 'short_delta': 0.25, 'wing_delta': 0.10},
                  'strangle': {'dte': 30, 'short_delta': 0.25},
                  'risk_reversal': {'dte': 30, 'short_delta': 0.25},
                  'credit_spread': {'dte': 30, 'short_delta': 0.25, 'wing_delta': 0.10},
                  'calendar': {'near_dte': 30, 'far_dte': 60}}

# The EXACT rich summary field set each named overlay produces — an INDEPENDENT reference (not the
# engine), so a dropped/renamed field fails the check. Per-overlay: short-vol echoes target_delta +
# carries the hedge fields; straddle drops target_delta; the static iron-condor drops the hedge
# fields too; each its own num_*_sold key.
_OVERLAY_SUMMARY_KEYS = {
    'short_vol': {'capital', 'num_contracts', 'target_delta', 'final_equity', 'net_pnl',
                  'alpha_vs_cash', 'interest_earned', 'total_premium_collected', 'total_hedge_cost',
                  'hedge_cost_bps', 'num_calls_sold', 'wins', 'losses', 'win_rate',
                  'max_drawdown_pct', 'risk_free_rate', 'cash'},
    'straddle': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                 'interest_earned', 'total_premium_collected', 'total_hedge_cost', 'hedge_cost_bps',
                 'num_straddles_sold', 'wins', 'losses', 'win_rate', 'max_drawdown_pct',
                 'risk_free_rate', 'cash'},
    'iron_condor': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                    'interest_earned', 'total_premium_collected', 'num_condors_sold', 'wins',
                    'losses', 'win_rate', 'max_drawdown_pct', 'risk_free_rate', 'cash'},
    'strangle': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                 'interest_earned', 'total_premium_collected', 'total_hedge_cost', 'hedge_cost_bps',
                 'num_strangles_sold', 'wins', 'losses', 'win_rate', 'max_drawdown_pct',
                 'risk_free_rate', 'cash'},   # same shape as the straddle (its OTM cousin)
    'risk_reversal': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                      'interest_earned', 'total_premium_collected', 'total_hedge_cost',
                      'hedge_cost_bps', 'num_risk_reversals_sold', 'wins', 'losses', 'win_rate',
                      'max_drawdown_pct', 'risk_free_rate', 'cash'},   # hedged 2-leg, SKEW
    'credit_spread': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                      'interest_earned', 'total_premium_collected', 'total_hedge_cost',
                      'hedge_cost_bps', 'num_credit_spreads_sold', 'wins', 'losses', 'win_rate',
                      'max_drawdown_pct', 'risk_free_rate', 'cash'},   # hedged 2-leg put, CARRY
    'calendar': {'capital', 'num_contracts', 'final_equity', 'net_pnl', 'alpha_vs_cash',
                 'interest_earned', 'total_premium_collected', 'total_hedge_cost', 'hedge_cost_bps',
                 'num_calendars_sold', 'wins', 'losses', 'win_rate', 'max_drawdown_pct',
                 'risk_free_rate', 'cash'},   # hedged TWO-expiration structure (TERM)
}


def _assert_engine_equivalent(market, name: str, *, must_trade: bool = True) -> None:
    """Post-Stage-B: each named overlay (run_real_*_overlay) is now a thin DELEGATE to the single
    generic engine via run_structure_via_spec — there is no separate frozen body left to compare to.
    This pins that the delegate ENTERS (`must_trade`, keyed off the trade list — rf-credit drift
    makes equity move even with zero trades, so equity.nunique() can't tell a real run from a
    non-trading one) and produces its COMPLETE rich summary: the exact per-overlay field set,
    checked against the INDEPENDENT _OVERLAY_SUMMARY_KEYS reference so a dropped/renamed field fails
    here. The byte-identical-to-frozen NUMERIC proof was the swap's gate (PR #64, consumed); the
    registered/exploratory regressions carry the t-stat/equity VALUES forward through the delegates."""
    dates, prices, store = market
    params = {**_STRUCT_PARAMS[name], 'capital': 100_000}
    s, trades, eq = _NAMED_OVERLAY[name](dates, prices, store, params)
    if must_trade:
        assert len(trades) > 0, f'{name} never traded on this store'
    assert set(s) == _OVERLAY_SUMMARY_KEYS[name], \
        f'{name} summary keys drifted: {set(s) ^ _OVERLAY_SUMMARY_KEYS[name]}'
    # the delegate yields a real equity series the downstream HAC-t can consume
    assert short_vol_statistics(eq, s['capital'], rf=s['risk_free_rate'])['t_stat_newey_west'] is not None


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz twins)')
class TestGenericStructureEngineEquivalence:
    """DATASET-GATED: post-Stage-B, the three named overlays are thin DELEGATES to the single
    generic engine (run_structure_via_spec). This pins that each delegate ENTERS and produces its
    COMPLETE per-overlay rich summary on real chains. All three run on SPY: the canonical
    spy_option_dailies.csv is CALLS-ONLY, so the put-leg straddle/iron-condor need the separate
    spy_option_dailies_puts.csv MERGED at load (the `spy_merged_market` fixture, the same way
    run_registered_vrp loads the SPY straddle); the `must_trade` guard turns a missing-puts vacuity
    into a failure rather than a false pass. The byte-identical-to-the-old-frozen-bodies NUMBERS are
    pinned by the registered/exploratory regressions (which now run through these delegates)."""

    def test_short_vol_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'short_vol')

    def test_straddle_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'straddle')

    def test_iron_condor_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'iron_condor')

    def test_strangle_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'strangle')   # widening 1 (the OTM straddle)

    def test_risk_reversal_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'risk_reversal')  # widening 2 (SKEW, mixed-sign)

    def test_credit_spread_summary_complete(self, spy_merged_market) -> None:
        _assert_engine_equivalent(spy_merged_market, 'credit_spread')  # widening 3 (CARRY, put-leg)

    def test_calendar_summary_complete(self, spy_merged_market) -> None:
        # widening 4 (TERM): the first TWO-expiration structure — exercises the engine's staggered
        # settlement (the near leg settles while the far leg lives on) and the multi-exp signature.
        _assert_engine_equivalent(spy_merged_market, 'calendar')

    @pytest.mark.skipif(not _HAVE_NVDA, reason='needs nvda_option_dailies.csv or its .gz twin')
    def test_iron_condor_summary_complete_nvda(self) -> None:
        # the iron-condor delegate on a second ticker (NVDA): enters + emits the complete summary.
        _assert_engine_equivalent(_equiv_market('NVDA', _NVDA_DAILIES), 'iron_condor')


# --------------------------------------------------------------------------- #
# Signature-vs-engine consistency check (Black-Scholes greeks)
# --------------------------------------------------------------------------- #
class TestGreeks:
    """ALWAYS-RUN: the BS greek primitives + structure_greek_signature. These derive a
    structure's net-greek signature from real entry legs — the math the dataset-gated
    consistency check rides on."""

    def test_bs_price_and_iv_roundtrip(self) -> None:
        S, K, t, r, sig = 100.0, 100.0, 0.25, 0.045, 0.20
        price = bs_price('call', S, K, t, r, sig)
        assert price == pytest.approx(4.5498, abs=1e-3)          # ATM call, known value
        assert implied_vol('call', price, S, K, t, r) == pytest.approx(sig, abs=1e-5)
        # put via the same machinery; IV round-trips too
        pput = bs_price('put', S, K, t, r, sig)
        assert implied_vol('put', pput, S, K, t, r) == pytest.approx(sig, abs=1e-5)

    def test_gamma_vega_positive_and_degenerate(self) -> None:
        S, K, t, r, sig = 100.0, 105.0, 0.25, 0.045, 0.25
        assert bs_gamma(S, K, t, r, sig) > 0 and bs_vega(S, K, t, r, sig) > 0
        # degenerate inputs -> 0, intrinsic
        assert bs_gamma(S, K, 0.0, r, sig) == 0.0 and bs_vega(S, K, t, r, 0.0) == 0.0
        assert bs_price('call', S, K, 0.0, r, sig) == max(0.0, S - K)

    def test_gamma_vega_known_magnitudes(self) -> None:
        # exact ATM BS gamma/vega (S=K=100, t=0.25, r=0.045, sigma=0.20) — the consistency check
        # reads only the net-greek SIGN, so this magnitude pin is what locks the normalization
        # (a missing sqrt(t) or a misplaced sigma would keep the sign and slip through otherwise).
        S, K, t, r, sig = 100.0, 100.0, 0.25, 0.045, 0.20
        assert bs_gamma(S, K, t, r, sig) == pytest.approx(0.039371, abs=1e-6)
        assert bs_vega(S, K, t, r, sig) == pytest.approx(19.685481, abs=1e-5)

    def test_iv_none_below_intrinsic(self) -> None:
        # a mark at/below intrinsic (or t<=0) has no implied vol
        assert implied_vol('call', 0.0, 100.0, 90.0, 0.25, 0.045) is None   # < 10 intrinsic
        assert implied_vol('call', 5.0, 100.0, 90.0, 0.0, 0.045) is None    # t=0

    def test_iv_none_on_degenerate_marks(self) -> None:
        # a NaN mark must return None, not leak past the guards to a fabricated vol of hi=5.0
        assert implied_vol('call', float('nan'), 100.0, 100.0, 0.25, 0.045) is None
        assert implied_vol('call', 5.0, float('nan'), 100.0, 0.25, 0.045) is None
        # a price whose extrinsic value sits below the price tolerance -> no reliable IV (None),
        # rather than a junk root from the absolute-residual stopping test
        deep = 10.0 + 1e-12                       # K=90 call, spot 100 -> intrinsic 10
        assert implied_vol('call', deep, 100.0, 90.0, 0.25, 0.045) is None

    def test_structure_greek_signature_synthetic(self) -> None:
        S, yrs = 100.0, 0.1
        short_straddle = [
            {'sign': -1, 'right': 'call', 'strike': 100.0, 'mid': 5.0, 'delta': 0.50,
             'expiration': 'E', 'contract': 'c'},
            {'sign': -1, 'right': 'put', 'strike': 100.0, 'mid': 5.0, 'delta': -0.50,
             'expiration': 'E', 'contract': 'p'},
        ]
        assert structure_greek_signature(short_straddle, S, yrs) == {
            'legs': 2, 'expirations': 1, 'net_vega': 'short', 'net_delta': 'neutral',
            'net_skew': 'flat'}                  # all-short -> no short-vs-long IV asymmetry
        # a single LONG option is long vega and (delta +0.5) long direction
        long_call = [{'sign': +1, 'right': 'call', 'strike': 100.0, 'mid': 5.0, 'delta': 0.50,
                      'expiration': 'E', 'contract': 'c'}]
        s = structure_greek_signature(long_call, S, yrs)
        assert s['net_vega'] == 'long' and s['net_delta'] == 'long' and s['net_skew'] == 'flat'
        # a risk reversal — SHORT the rich put (mid 8) + LONG the cheap call (mid 3): net_skew reads
        # the short-vs-long IV asymmetry ('short_rich'), net_delta is long, net_vega offsets to neutral
        rr = [{'sign': -1, 'right': 'put', 'strike': 95.0, 'mid': 8.0, 'delta': -0.25,
               'expiration': 'E', 'contract': 'p'},
              {'sign': +1, 'right': 'call', 'strike': 105.0, 'mid': 3.0, 'delta': 0.25,
               'expiration': 'E', 'contract': 'c'}]
        rs = structure_greek_signature(rr, S, yrs)
        assert rs['net_skew'] == 'short_rich' and rs['net_delta'] == 'long'
        # distinct expirations are counted (a calendar would be expirations=2)
        cal = [{'sign': -1, 'right': 'call', 'strike': 100.0, 'mid': 3.0, 'delta': 0.50,
                'expiration': 'E1', 'contract': 'a'},
               {'sign': +1, 'right': 'call', 'strike': 100.0, 'mid': 5.0, 'delta': 0.50,
                'expiration': 'E2', 'contract': 'b'}]
        assert structure_greek_signature(cal, S, yrs)['expirations'] == 2

    def test_structure_greek_signature_raises_on_uninvertible_leg(self) -> None:
        bad = [{'sign': -1, 'right': 'call', 'strike': 90.0, 'mid': 0.0,  # mid < 10 intrinsic
                'expiration': 'E', 'contract': 'x'}]
        with pytest.raises(ValueError, match='could not imply vol'):
            structure_greek_signature(bad, 100.0, 0.1)


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz twins)')
class TestGrammarSignatureMatchesEngine:
    """DATASET-GATED: the grammar's DECLARED economic signature (edge_search.STRUCTURE_GRAMMAR) is
    VERIFIED against the engine's actual entry legs — the consistency check that turns the typing
    from a label into an enforcement. For each structure, run its selector on SPY (calls + the
    separate puts file merged at load, so the put-leg straddle/iron-condor/risk-reversal/credit-spread
    trade), back the IV out of each leg's mid, compute the three robust axes (net_vega, net_delta,
    net_skew), and assert the engine-derived {legs, expirations, net_vega, net_delta, net_skew}
    matches the declared signature. A future overlay that DECLARES short vega while the engine runs
    something long-vega — or a skew structure that declares short_rich while the engine longs the rich
    wing — fails here. (This is exactly the check that corrected the credit spread's declared net_skew
    from short_rich to long_rich: its long OTM put wing sits on the steep part of the put skew, so it
    is the richer leg — the same long_rich read as the iron condor.)

    The calendar (widening 4, TERM) is the first TWO-expiration structure: `entry_date` is passed so
    structure_greek_signature backs each leg's IV out at its OWN tenor (the near and far calls live on
    different clocks), which is what makes net_vega='long' the engine's real signature. Passing
    entry_date is byte-identical for the single-expiration structures (all their legs share one tenor)."""

    def test_each_overlay_signature_matches_engine(self, spy_merged_market) -> None:
        dates, prices, store = spy_merged_market
        for name, spec in STRUCTURE_SPECS.items():
            derived = None
            for i, d in enumerate(dates):
                day = store.get(d)
                if day is None:
                    continue
                legs = spec['select'](day, _STRUCT_PARAMS[name])
                if not legs:
                    continue
                years = (pd.Timestamp(legs[0]['expiration']) - pd.Timestamp(d)).days / 365.0
                try:
                    # entry_date drives per-leg tenor (the calendar's near/far legs differ);
                    # single-expiration structures are unaffected (one shared tenor).
                    derived = structure_greek_signature(legs, prices[i], years, entry_date=d)
                    break
                except ValueError:
                    continue   # a stale-mark entry; try the next
            assert derived is not None, f'{name}: found no clean SPY entry to verify'
            declared = STRUCTURE_GRAMMAR[name].signature
            for k in ('legs', 'expirations', 'net_vega', 'net_delta', 'net_skew'):
                assert derived[k] == declared[k], (
                    f'{name}.{k}: engine-derived {derived[k]!r} != declared {declared[k]!r}')
