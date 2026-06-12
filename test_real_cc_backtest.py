"""Tests for real_cc_backtest.py — the real-option-chain overlay adapter.

Two layers:

- Pure-logic unit tests (always run, including CI): entry selection, the
  chain store's era clip and mark clamp, and fill-model arithmetic against
  synthetic chain slices.
- TestQqqRealChainRegression / TestMsftRealChainRegression: pin the headline
  numbers of the real-premium runs. Each requires its {ticker}_option_dailies
  CSV (or .gz twin), which is too large for git history and ships as a
  release asset (data-2026-06): CI downloads and checksum-verifies them
  before pytest, fetch_option_data.sh does the same locally, and each class
  skips gracefully when its file is absent. They lock the adversarially-
  reviewed results: the covered call that the proxy engine scores at +$120K
  net overlay P&L on QQQ loses $157K on real premiums (Newey-West t = -1.78),
  and the proxy's +$270K on MSFT — the series' published $268K headline —
  loses $184K (Newey-West t = -1.73).
- TestMsftRealWalkForwardRegression: pins the walk-forward optimization on
  the MSFT chains (walk_forward_real.py, 4-year train, bid/ask fills) —
  the optimizer's minimum-engagement drift and the chained OOS scoreboard.
  Same dataset/skip mechanics as the MSFT class above.
- TestMsftExtendedSpanRegression: pins the 16-year run (2010-05 -> 2026-04)
  on the merged canonical + backfill chains — the 2010-2013 sideways era
  does not rescue the overlay (net -$382K, NW t = -1.28, vs the proxy's
  +$533K on the same series). The 2008 -> mid-2010 placeholder-greeks era
  is EXCLUDED at load time (CHAIN_CLEAN_START): its entry band is vendor
  junk, so the GFC itself is untestable on these chains. Additionally
  requires msft_option_dailies_2008_2016.csv[.gz] (same release/CI
  mechanics).
- TestMsftStopLossRegression: pins the stop_loss_mult variant (buy back
  the short call at a multiple of the premium collected) — the stop makes
  the loss WORSE at every level and monotonically worse as it tightens
  (whipsaw on a trending stock), on both the 10- and 16-year spans.
- TestSpyRealWalkForwardRegression: pins the SPY walk-forward on real
  chains (2010-12 -> 2026-06, 23 windows, 4-year train): tuned overlay
  +143% chained OOS vs +171% buy-and-hold, beating it in 10/23 windows.
  Requires spy_option_dailies.csv[.gz] (same release/CI mechanics).
- TestQqqExtendedWalkForwardRegression: pins the QQQ 15-year walk-forward
  on the merged canonical + 2011-2016 backfill chains (22 windows): tuned
  overlay +283% chained OOS vs +418% buy-and-hold, beating it in only
  5/22 windows. Completes the three-ticker matrix: 22/68 real-chain
  windows beat buy-and-hold vs 55/68 on the proxy.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from cc_backtest import compute_statistics, run_cc_overlay
from real_cc_backtest import (
    CHAIN_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
    select_entry,
)
from walk_forward_real import FIXED_PARAMS, PARAM_GRID, _chain, walk_forward_real

_DAILIES = os.path.join(os.path.dirname(__file__), 'qqq_option_dailies.csv')
_UNADJ = os.path.join(os.path.dirname(__file__), 'qqq_10yr_prices_unadjusted.csv')
_HAVE_DAILIES = os.path.exists(_DAILIES) or os.path.exists(_DAILIES + '.gz')

_MSFT_DAILIES = os.path.join(os.path.dirname(__file__), 'msft_option_dailies.csv')
_HAVE_MSFT_DAILIES = (os.path.exists(_MSFT_DAILIES)
                      or os.path.exists(_MSFT_DAILIES + '.gz'))

_MSFT_BACKFILL = os.path.join(os.path.dirname(__file__),
                              'msft_option_dailies_2008_2016.csv')
_HAVE_MSFT_BACKFILL = (os.path.exists(_MSFT_BACKFILL)
                       or os.path.exists(_MSFT_BACKFILL + '.gz'))

_SPY_DAILIES = os.path.join(os.path.dirname(__file__), 'spy_option_dailies.csv')
_HAVE_SPY_DAILIES = (os.path.exists(_SPY_DAILIES)
                     or os.path.exists(_SPY_DAILIES + '.gz'))

_QQQ_BACKFILL = os.path.join(os.path.dirname(__file__),
                             'qqq_option_dailies_2011_2016.csv')
_HAVE_QQQ_BACKFILL = (os.path.exists(_QQQ_BACKFILL)
                      or os.path.exists(_QQQ_BACKFILL + '.gz'))

_PARAMS: dict[str, float] = {
    'call_delta': 0.25,
    'close_at_pct': 0.75,
    'dte': 30,  # calendar-day parity with the engine's 21 trading days (21/252*365)
    'risk_free_rate': 0.045,
    'capital': 100_000,
}


def _cand(dte: int, delta: float, bid: float = 1.0, ask: float = 1.1,
          mid: float = 1.05, exp: str = '2024-01-19', strike: float = 100.0,
          cid: str = 'X') -> tuple[int, float, float, float, float, str, float, str]:
    return (dte, delta, bid, ask, mid, exp, strike, cid)


class TestSelectEntry:
    """select_entry: nearest-DTE expiration first, then nearest delta within it."""

    def test_nearest_dte_cohort_wins(self) -> None:
        day = {'candidates': [
            _cand(7, 0.25, cid='wk'),
            _cand(28, 0.30, cid='m28a'),
            _cand(28, 0.24, cid='m28b'),
            _cand(56, 0.25, cid='m56'),
        ]}
        pick = select_entry(day, target_dte=30, target_delta=0.25)
        assert pick is not None
        assert pick[7] == 'm28b'  # 28 is nearest 30; delta 0.24 nearest 0.25

    def test_delta_band_filter(self) -> None:
        # Deltas at/outside (0.05, 0.60) are not sellable candidates.
        day = {'candidates': [
            _cand(30, 0.95, cid='deep'),
            _cand(30, 0.02, cid='tail'),
            _cand(30, 0.59, cid='ok'),
        ]}
        pick = select_entry(day, 30, 0.25)
        assert pick is not None
        assert pick[7] == 'ok'

    def test_zero_bid_excluded(self) -> None:
        day = {'candidates': [_cand(30, 0.25, bid=0.0, cid='dead')]}
        assert select_entry(day, 30, 0.25) is None

    def test_empty_day(self) -> None:
        assert select_entry({'candidates': []}, 30, 0.25) is None


class TestChainStoreEraClip:
    """load_chain_store: the placeholder-greeks era is excluded, not repaired.

    The 2008 -> mid/late-2010 era of the Alpha Vantage dailies carries
    vendor placeholder greeks inside the entry band (IVs on the quantized
    lattice {0.01488, 0.02463, ...}, deltas that jump 0.505 -> 0.087
    between adjacent strikes, marks like 0.01 on a 10.15/10.35 quote). No
    delta-targeted entry can trade that data, so runs drop the era at load
    time via `start` (per-ticker boundaries in CHAIN_CLEAN_START). Two
    row-level alternatives were evaluated and set aside: quarantining
    mark-outside-quote rows from entry is byte-identical to the clip on
    every pinned surface (nothing defective survives the boundary), and an
    IV < 0.05 filter falsely flags legitimate low-vol rows (SPY 2017). The
    modern files' small tail of out-of-band marks (0.05-0.14% of rows)
    keeps the midpoint clamp.
    """

    HEADER = ('date,expiration,dte,strike,bid,ask,mark,last,volume,'
              'open_interest,implied_volatility,delta,contractID')

    def _store_from(self, tmp_path: Path, rows: list[str],
                    start: str | None = None) -> dict[str, dict[str, Any]]:
        p = tmp_path / 'dailies.csv'
        p.write_text('\n'.join([self.HEADER, *rows]) + '\n')
        return load_chain_store(str(p), start=start)

    def test_clean_row_is_candidate(self, tmp_path: Path) -> None:
        store = self._store_from(tmp_path, [
            '2024-01-02,2024-02-02,31,110.0,2.00,2.20,2.10,0,0,0,0.21000,0.25000,OK',
        ])
        day = store['2024-01-02']
        assert [c[7] for c in day['candidates']] == ['OK']
        assert day['marks']['OK'] == (2.00, 2.20, 2.10, 0.25)

    def test_start_clips_era_rows_entirely(self, tmp_path: Path) -> None:
        # The 2008-2010 defect verbatim: mark 0.01 on a 10.15/10.35 quote,
        # lattice IV, placeholder delta. With `start` at the clean boundary
        # the row vanishes from candidates AND marks.
        store = self._store_from(tmp_path, [
            '2008-01-30,2008-03-22,52,32.5,10.15,10.35,0.01,0,0,0,0.01488,0.20567,BAD',
            '2024-01-02,2024-02-02,31,110.0,2.00,2.20,2.10,0,0,0,0.21000,0.25000,OK',
        ], start=CHAIN_CLEAN_START['MSFT'])
        assert '2008-01-30' not in store
        assert [c[7] for c in store['2024-01-02']['candidates']] == ['OK']

    def test_out_of_band_mark_clamped_to_midpoint(self, tmp_path: Path) -> None:
        # Without a clip the row stays a candidate but its mark is repaired
        # to the quote midpoint — the modern files' 0.05-0.14% tail.
        store = self._store_from(tmp_path, [
            '2024-01-03,2024-02-02,30,110.0,2.00,2.20,9.99,0,0,0,0.21000,0.25000,OOB',
        ])
        day = store['2024-01-03']
        assert [c[7] for c in day['candidates']] == ['OOB']
        assert day['marks']['OOB'] == (2.00, 2.20, pytest.approx(2.10), 0.25)

    def test_lattice_iv_with_clean_mark_stays(self, tmp_path: Path) -> None:
        # SPY 2017-09-13 verbatim: lattice IV (0.03439) but a sane delta and
        # an in-band mark — legitimate low-vol data, deliberately kept. An
        # IV < 0.05 row filter was considered and rejected over rows like
        # this one (it would flip entry selection on 8 clean 2017 days).
        store = self._store_from(tmp_path, [
            '2017-09-13,2017-09-20,7,251.0,0.22,0.24,0.23,0,0,0,0.03439,0.25901,LAT',
        ])
        assert [c[7] for c in store['2017-09-13']['candidates']] == ['LAT']

    def test_blank_iv_is_fine(self, tmp_path: Path) -> None:
        # The loader never reads the IV column; a blank one must not matter.
        store = self._store_from(tmp_path, [
            '2024-01-02,2024-02-02,31,110.0,2.00,2.20,2.10,0,0,0,,0.25000,NOIV',
        ])
        assert [c[7] for c in store['2024-01-02']['candidates']] == ['NOIV']


class TestFillModel:
    """Bid/ask vs mid fills on a minimal two-day synthetic market."""

    @staticmethod
    def _store() -> dict[str, dict[str, Any]]:
        # Day 1: sell the 30-DTE 0.25-delta call (bid 2.00 / ask 2.20 / mid 2.10).
        # Day 2: option collapses (bid 0.10 / ask 0.30 / mid 0.20) -> profit target.
        c1 = _cand(30, 0.25, bid=2.00, ask=2.20, mid=2.10, exp='2099-01-01',
                   strike=110.0, cid='C')
        return {
            '2024-01-02': {'candidates': [c1], 'marks': {'C': (2.00, 2.20, 2.10, 0.25)}},
            '2024-01-03': {'candidates': [], 'marks': {'C': (0.10, 0.30, 0.20, 0.05)}},
        }

    def test_bid_ask_fills(self) -> None:
        s, trades, _ = run_real_cc_overlay(
            ['2024-01-02', '2024-01-03'], [100.0, 100.0], self._store(),
            {**_PARAMS, 'capital': 10_000},
        )
        sell = next(t for t in trades if t['action'] == 'sell')
        close = next(t for t in trades if t['action'] == 'close')
        assert sell['premium'] == pytest.approx(2.00 - 0.0065)  # bid less commission
        # Buyback at ask + commission: pnl per share = (2.00-0.0065) - (0.30+0.0065).
        assert close['pnl'] == pytest.approx((1.9935 - 0.3065) * 100, abs=1e-6)
        assert s['num_calls_sold'] == 1
        assert s['win_rate'] == 100.0

    def test_mid_fills(self) -> None:
        _, trades, _ = run_real_cc_overlay(
            ['2024-01-02', '2024-01-03'], [100.0, 100.0], self._store(),
            {**_PARAMS, 'capital': 10_000, 'fill': 'mid'},
        )
        sell = next(t for t in trades if t['action'] == 'sell')
        close = next(t for t in trades if t['action'] == 'close')
        assert sell['premium'] == pytest.approx(2.10 - 0.0065)  # mid less commission
        assert close['pnl'] == pytest.approx((2.0935 - 0.2065) * 100, abs=1e-6)


@pytest.mark.skipif(
    not _HAVE_DAILIES,
    reason='needs qqq_option_dailies.csv or its committed .gz twin',
)
class TestQqqRealChainRegression:
    """Pin the real-premium QQQ run (2016-06 -> 2026-06, bid/ask fills).

    These numbers were adversarially verified by an 11-agent review
    (accounting identities to the penny, parity vs run_cc_overlay, look-ahead
    sweep, data-quality checks) before being pinned. The proxy comparison run
    uses the same unadjusted close series, so the real-vs-proxy gap isolates
    the option-pricing source: the proxy's +$120K net overlay P&L becomes a
    -$157K loss on real premiums.

    Data source: qqq_option_dailies.csv.gz lives on the data-2026-06 release
    (too large for git history); CI downloads and checksum-verifies it before
    pytest, and fetch_option_data.sh does the same locally. With neither the
    .gz nor the raw CSV present, these pins skip rather than fail.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        store = load_chain_store(_DAILIES)
        days = sorted(store)
        dates, prices = load_unadjusted_prices('QQQ', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    @pytest.fixture(scope='class')
    def real(self, market) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
        dates, prices, store = market
        return run_real_cc_overlay(dates, prices, store, _PARAMS)

    def test_headline(self, real) -> None:
        """Net overlay -$156.6K: the sign-flip result, exact to the review."""
        s, _, _ = real
        assert s['num_contracts'] == 9
        assert s['net_overlay_pnl'] == pytest.approx(-156_628.35, abs=1.0)
        assert s['total_premium_collected'] == pytest.approx(431_822.70, abs=1.0)
        assert s['final_equity'] == pytest.approx(478_511.65, abs=1.0)
        assert s['buy_hold_final'] == pytest.approx(635_140.00, abs=1.0)
        assert s['total_return_pct'] == pytest.approx(378.51, abs=0.05)
        assert s['buy_hold_return_pct'] == pytest.approx(535.14, abs=0.05)
        assert s['max_drawdown_pct'] == pytest.approx(38.22, abs=0.05)

    def test_activity(self, real) -> None:
        """198 calls; 122 profit-target closes, 71 deep-ITM closes, 5 expirations.

        The bucket P&L totals pin the loss anatomy quoted in prose: the
        profit-target wins (+$234K) are overwhelmed by deep-ITM forced
        buybacks (-$398K).
        """
        s, trades, _ = real
        assert s['num_calls_sold'] == 198
        assert s['wins'] == 127
        assert s['losses'] == 71
        assert s['win_rate'] == pytest.approx(64.1, abs=0.1)
        actions = [t['action'] for t in trades]
        assert actions.count('close') == 122
        assert actions.count('close_itm') == 71
        assert actions.count('expiration') == 5
        pnl = {a: sum(t['pnl'] for t in trades if t['action'] == a)
               for a in ('close', 'close_itm', 'expiration')}
        assert pnl['close'] == pytest.approx(233_607.60, abs=1.0)
        assert pnl['close_itm'] == pytest.approx(-398_369.70, abs=1.0)
        assert pnl['expiration'] == pytest.approx(8_133.75, abs=1.0)

    def test_significance(self, real) -> None:
        """NW t = -1.78: the real overlay's harm is near-significant."""
        s, _, eq = real
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert st['t_stat_newey_west'] == pytest.approx(-1.78, abs=0.02)
        assert st['sharpe_excess'] == pytest.approx(-0.53, abs=0.01)
        assert st['passes_t_2'] is False

    def test_mid_fill_variant(self, market) -> None:
        """Mid fills recover only ~$12K of the loss: spread is not the driver.

        Re-pinned -144,744.30 -> -144,735.30 (a $9 shift) when the loader
        gained the mark sanity clamp: ~0.14% of QQQ rows carried marks
        outside [bid, ask], and mid fills trade at the mark. The bid/ask
        pins were unaffected.
        """
        dates, prices, store = market
        s, _, eq = run_real_cc_overlay(dates, prices, store, {**_PARAMS, 'fill': 'mid'})
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(-144_735.30, abs=1.0)
        assert s['num_calls_sold'] == 210
        assert st['t_stat_newey_west'] == pytest.approx(-1.76, abs=0.02)

    def test_proxy_same_series(self, market) -> None:
        """The proxy engine on the identical unadjusted series: +$120K, t +0.14.

        Pinned beside the real run so the $277K real-vs-proxy swing is
        CI-verified end to end (locally) from one data lineage.
        """
        dates, prices, _ = market
        import numpy as np
        s, _, eq = run_cc_overlay(dates, np.array(prices),
                                  {**_PARAMS, 'dte': 21})  # engine dte is trading days
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(120_216.68, abs=1.0)
        assert st['t_stat_newey_west'] == pytest.approx(0.14, abs=0.02)


@pytest.mark.skipif(
    not _HAVE_MSFT_DAILIES,
    reason='needs msft_option_dailies.csv or its committed .gz twin',
)
class TestMsftRealChainRegression:
    """Pin the real-premium MSFT run (2016-04 -> 2026-04, bid/ask fills).

    This is the reckoning for the series' published headline: on the
    unadjusted close series the chains require, the proxy engine reports
    +$269,948 net overlay P&L and real premiums produce -$183,552. (The
    published $268,424.87 is the same engine on the dividend-adjusted series
    with 20 contracts, NW t=+0.46; the dollar proximity to the unadjusted
    twin is partly coincidental — fewer, pricier shares offset a larger
    per-share P&L — but the verdict is identical.) Same loss anatomy as QQQ
    — profit-target wins overwhelmed by deep-ITM forced buybacks — with one
    difference: switching both legs to mid fills recovers ~$108K on MSFT
    (single-name spreads are wide), where it recovered only ~$12K on QQQ.
    Either way the overlay shows no edge: NW t = -1.73 at the bid/ask
    (short of the t=2 bar the repo holds every result to), -0.90 at mid.

    Data source: msft_option_dailies.csv.gz lives on the data-2026-06 release
    (too large for git history); CI downloads and checksum-verifies it before
    pytest, and fetch_option_data.sh does the same locally. With neither the
    .gz nor the raw CSV present, these pins skip rather than fail.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        store = load_chain_store(_MSFT_DAILIES)
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    @pytest.fixture(scope='class')
    def real(self, market) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
        dates, prices, store = market
        return run_real_cc_overlay(dates, prices, store, _PARAMS)

    def test_headline(self, real) -> None:
        """Net overlay -$183.6K: the published +$268K headline, sign-flipped."""
        s, _, _ = real
        assert s['num_contracts'] == 18
        assert s['net_overlay_pnl'] == pytest.approx(-183_552.34, abs=1.0)
        assert s['total_premium_collected'] == pytest.approx(729_054.90, abs=1.0)
        assert s['final_equity'] == pytest.approx(486_255.65, abs=1.0)
        assert s['buy_hold_final'] == pytest.approx(669_807.99, abs=1.0)
        assert s['total_return_pct'] == pytest.approx(386.26, abs=0.05)
        assert s['buy_hold_return_pct'] == pytest.approx(569.81, abs=0.05)
        assert s['max_drawdown_pct'] == pytest.approx(41.00, abs=0.05)

    def test_activity(self, real) -> None:
        """183 calls; 122 profit-target closes, 54 deep-ITM closes, 6 expirations.

        Same loss anatomy as QQQ, pinned in dollars: profit-target wins
        (+$429K) overwhelmed by deep-ITM forced buybacks (-$611K).
        """
        s, trades, _ = real
        assert s['num_calls_sold'] == 183
        assert s['wins'] == 124
        assert s['losses'] == 58
        assert s['win_rate'] == pytest.approx(68.1, abs=0.1)
        actions = [t['action'] for t in trades]
        assert actions.count('close') == 122
        assert actions.count('close_itm') == 54
        assert actions.count('expiration') == 6
        pnl = {a: sum(t['pnl'] for t in trades if t['action'] == a)
               for a in ('close', 'close_itm', 'expiration')}
        assert pnl['close'] == pytest.approx(429_037.20, abs=1.0)
        assert pnl['close_itm'] == pytest.approx(-611_301.60, abs=1.0)
        assert pnl['expiration'] == pytest.approx(-736.23, abs=1.0)

    def test_significance(self, real) -> None:
        """NW t = -1.73: suggestive of harm under worst-case fills, but short
        of the t=2 bar this repo holds every result to (and -0.90 at mid —
        noise). The defensible claim is 'no edge, sign now negative'."""
        s, _, eq = real
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert st['t_stat_newey_west'] == pytest.approx(-1.73, abs=0.02)
        assert st['sharpe_excess'] == pytest.approx(-0.49, abs=0.01)
        assert st['passes_t_2'] is False

    def test_mid_fill_variant(self, market) -> None:
        """Mid fills recover ~$108K but still lose: spread hurts, economics decide."""
        dates, prices, store = market
        s, _, eq = run_real_cc_overlay(dates, prices, store, {**_PARAMS, 'fill': 'mid'})
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(-75_988.84, abs=1.0)
        assert s['num_calls_sold'] == 195
        assert st['t_stat_newey_west'] == pytest.approx(-0.90, abs=0.02)

    def test_proxy_same_series(self, market) -> None:
        """The proxy engine on the identical unadjusted series: +$270K, t +0.58.

        Pinned beside the real run so the $454K real-vs-proxy swing is
        CI-verified end to end from one data lineage — and so the published
        $268,424.87 (adjusted series, README/blog) has a CI-checked twin on
        the unadjusted series the real chains require.
        """
        dates, prices, _ = market
        import numpy as np
        s, _, eq = run_cc_overlay(dates, np.array(prices),
                                  {**_PARAMS, 'dte': 21})  # engine dte is trading days
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(269_948.12, abs=1.0)
        assert st['t_stat_newey_west'] == pytest.approx(0.58, abs=0.02)


@pytest.mark.skipif(
    not _HAVE_MSFT_DAILIES,
    reason='needs msft_option_dailies.csv or its committed .gz twin',
)
class TestMsftRealWalkForwardRegression:
    """Pin the walk-forward optimization on real MSFT chains (4y train, bid/ask).

    walk_forward_real mirrors walk_forward_optimization's window arithmetic
    (6-month test/roll, in-sample-Sharpe selection) with run_real_cc_overlay
    as the engine and a 21/30/45 CALENDAR-day grid. The 4-year train window
    is the smallest integer size that clears Pardo's ~30-trade sample-size
    floor for every grid combo: the binding corner — a 45-day call held to
    expiration — completes at most ~24 cycles in 3 years (3*365/45) but ~32
    in 4; empirically the leanest 4-year fit runs 33 trades. The price is
    two fewer OOS windows than the 3-year default (13 -> 11).

    What this locks (2020-04 -> 2025-10 OOS span, per-window $100K restarts,
    the same convention for all three curves):

    - The optimizer's minimum-engagement drift on real quotes:
      close_at_pct=1.00 (never pay the ask to take profit early) wins all
      11 windows, and (0.15 delta, 21-day) is the modal triple.
    - The scoreboard: walk-forward picks chain to +185.0% OOS vs +114.3%
      for the fixed published defaults and +184.8% for buy-and-hold — a
      dead heat with holding the stock (and the heat does not survive mid
      fills, which are not pinned here). Tuning on real chains is damage
      control, not edge.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        store = load_chain_store(_MSFT_DAILIES)
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    @pytest.fixture(scope='class')
    def records(self, market) -> list[dict[str, Any]]:
        dates, prices, store = market
        return walk_forward_real(dates, prices, store, PARAM_GRID,
                                 fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask'},
                                 train_years=4)

    def test_window_layout(self, records) -> None:
        """11 non-overlapping 6-month test windows, 2020-04 -> 2025-10."""
        assert len(records) == 11
        assert records[0]['train_start'] == '2016-04-11'
        assert records[0]['train_end'] == '2020-04-11'
        assert records[0]['test_start'] == '2020-04-11'
        assert records[-1]['test_start'] == '2025-04-11'
        assert records[-1]['test_end'] == '2025-10-11'

    def test_optimizer_choices(self, records) -> None:
        """close=1.00 in 11/11 windows; (0.15, 21) modal — minimum engagement."""
        triples = [(r['best_params']['call_delta'], int(r['best_params']['dte']),
                    r['best_params']['close_at_pct']) for r in records]
        assert triples == [
            (0.15, 21, 1.0), (0.15, 21, 1.0), (0.15, 21, 1.0), (0.15, 21, 1.0),
            (0.15, 45, 1.0), (0.20, 30, 1.0), (0.15, 21, 1.0), (0.15, 21, 1.0),
            (0.20, 30, 1.0), (0.20, 30, 1.0), (0.20, 30, 1.0),
        ]
        assert Counter(t[:2] for t in triples) == {(0.15, 21): 6, (0.20, 30): 4,
                                                   (0.15, 45): 1}
        assert records[0]['train_sharpe'] == pytest.approx(1.133, abs=5e-4)
        assert records[-1]['train_sharpe'] == pytest.approx(0.598, abs=5e-4)

    def test_pardo_floor(self, records) -> None:
        """Every grid combo clears 30 IS trades in every window (leanest: 33)."""
        assert all(r['n_below_30'] == 0 for r in records)
        assert min(r['min_grid_trades'] for r in records) == 33
        assert [r['n_trades'] for r in records] == [69, 69, 68, 69, 33, 46,
                                                    70, 70, 46, 46, 46]

    def test_oos_scoreboard(self, records) -> None:
        """WF +185.0% vs fixed +114.3% vs B&H +184.8%, chained per-window restarts."""
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(184.97, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(114.27, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(184.82, abs=0.01)
        assert sum(r['oos_return_pct'] > r['fixed_return_pct'] for r in records) == 9
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 6


@pytest.mark.skipif(
    not (_HAVE_MSFT_DAILIES and _HAVE_MSFT_BACKFILL),
    reason='needs msft_option_dailies.csv and msft_option_dailies_2008_2016.csv '
           '(or their .gz twins)',
)
class TestMsftExtendedSpanRegression:
    """Pin the 16-year MSFT run (2010-05 -> 2026-04) on merged real chains.

    The 2008-2016 backfill was bought to answer one question: was the 'no
    edge' verdict an artifact of testing a covered call on a 10x bull run?
    The honest answer after the data audit: the GFC itself is UNTESTABLE
    on these chains — the 2008 -> mid-2010 era carries vendor placeholder
    greeks inside the entry band (entries there would have run a median
    vendor delta of 0.074 against the 0.25 target), so runs exclude the
    era at load time (CHAIN_CLEAN_START['MSFT'] = '2010-05-10', the first
    trading day past the last in-band placeholder row). What the span CAN
    test — the 2010-2013 sideways era the strategy is supposedly for,
    plus the full 2010s bull — still says no: the overlay loses MORE than
    on the 10-year span (-$382K vs -$184K), premium retention is negative,
    and the same proxy engine on the same series reports +$533K. The
    walk-forward (23 windows, 4-year train, every grid fit above the Pardo
    floor) chains to +521% OOS vs +913% for buy-and-hold.

    These runs still exercise the era-specific engine paths: pre-Feb-2015
    Saturday-dated expirations settle against the prior Friday close, and
    the loader's midpoint clamp repairs the modern files' small tail of
    out-of-band marks. The backfill ships on the same data-2026-06 release
    with the same checksum/CI mechanics as the canonical datasets; the
    class skips when either file is absent.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        store = load_chain_store(_MSFT_DAILIES, [_MSFT_BACKFILL],
                                 start=CHAIN_CLEAN_START['MSFT'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    @pytest.fixture(scope='class')
    def real(self, market) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
        dates, prices, store = market
        return run_real_cc_overlay(dates, prices, store, _PARAMS)

    def test_span(self, market) -> None:
        """4,005 trading days, 2010-05-10 -> 2026-04-10; era rows clipped."""
        dates, _, store = market
        assert (dates[0], dates[-1], len(dates)) == ('2010-05-10', '2026-04-10', 4005)
        assert len(store) == 4000  # chain days (a handful of price days lack chains)

    def test_overlay_headline(self, real) -> None:
        """Net overlay -$382K over 16 years: more history, bigger loss."""
        s, _, _ = real
        assert s['num_contracts'] == 34
        assert s['net_overlay_pnl'] == pytest.approx(-382_209.36, abs=1.0)
        assert s['total_premium_collected'] == pytest.approx(1_481_714.90, abs=1.0)
        assert s['final_equity'] == pytest.approx(880_352.62, abs=1.0)
        assert s['total_return_pct'] == pytest.approx(780.35, abs=0.05)
        assert s['buy_hold_return_pct'] == pytest.approx(1162.56, abs=0.05)
        assert s['premium_retention_pct'] == pytest.approx(-25.8, abs=0.1)
        assert s['max_drawdown_pct'] == pytest.approx(42.70, abs=0.05)

    def test_activity(self, real) -> None:
        """291 calls; the 8 expirations include Saturday-settled pre-2015 cycles."""
        s, trades, _ = real
        assert s['num_calls_sold'] == 291
        assert (s['wins'], s['losses']) == (204, 86)
        actions = [t['action'] for t in trades]
        assert actions.count('close') == 200
        assert actions.count('close_itm') == 82
        assert actions.count('expiration') == 8

    def test_significance(self, real) -> None:
        """NW t = -1.28: no edge, sign negative, 16 years of data."""
        s, _, eq = real
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert st['t_stat_newey_west'] == pytest.approx(-1.28, abs=0.02)
        assert st['sharpe_excess'] == pytest.approx(-0.303, abs=0.005)
        assert st['passes_t_2'] is False

    def test_proxy_same_series(self, real, market) -> None:
        """The proxy on the identical 16-year series: +$533K — a $915K swing."""
        dates, prices, _ = market
        import numpy as np
        s, _, eq = run_cc_overlay(dates, np.array(prices),
                                  {**_PARAMS, 'dte': 21})  # engine dte is trading days
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(533_159.94, abs=1.0)
        assert s['num_calls_sold'] == 301
        assert st['t_stat_newey_west'] == pytest.approx(0.20, abs=0.02)

    def test_walk_forward(self, market) -> None:
        """23 windows, all Pardo floors clear; WF +521% vs B&H +913% chained."""
        dates, prices, store = market
        records = walk_forward_real(dates, prices, store, PARAM_GRID,
                                    fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask'},
                                    train_years=4)
        assert len(records) == 23
        assert records[0]['train_start'] == '2010-05-10'
        assert records[0]['test_start'] == '2014-05-10'
        assert records[-1]['test_end'] == '2025-11-10'
        assert all(r['n_below_30'] == 0 for r in records)
        assert min(r['n_trades'] for r in records) == 33
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(520.95, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(464.66, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(912.55, abs=0.01)
        assert sum(r['oos_return_pct'] > r['fixed_return_pct'] for r in records) == 13
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 7
        deltas = Counter(r['best_params']['call_delta'] for r in records)
        closes = Counter(r['best_params']['close_at_pct'] for r in records)
        assert deltas == {0.15: 14, 0.20: 9}
        assert closes == {1.00: 17, 0.75: 6}


@pytest.mark.skipif(
    not (_HAVE_MSFT_DAILIES and _HAVE_MSFT_BACKFILL),
    reason='needs msft_option_dailies.csv and msft_option_dailies_2008_2016.csv '
           '(or their .gz twins)',
)
class TestMsftStopLossRegression:
    """Pin the stop-loss variant: a 2x-premium stop makes the loss WORSE.

    stop_loss_mult buys the short call back when its ask reaches that
    multiple of the net premium collected (the classic "stop at 2x entry"),
    evaluated daily at the close like the engine's other exit rules. On a
    relentlessly trending stock the stop is whipsaw machinery: a 0.25-delta
    call doubles on a moderate rally, so the stop fires constantly (118
    times in ten years vs the baseline's 54 deep-ITM closes), each firing
    locks in ~1x premium plus a spread crossing, and the engine re-sells
    into the same rally. Tightening the stop is monotonically worse, and no
    level improves on the no-stop baseline (-$183,552 over 10 years,
    -$382,209 over the 16-year span — pinned in the headline classes
    above). Max
    drawdown RISES with the stop (the drawdown driver is the stock leg,
    not the short call), and no variant moves the NW t-stat off "no edge."

    Convention caveat carried with the pin: this is a stop-MARKET on daily
    closes — intraday touches would fire even more often, so these numbers
    flatter the stop if anything.
    """

    @pytest.fixture(scope='class')
    def market(self) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        store = load_chain_store(_MSFT_DAILIES, [_MSFT_BACKFILL],
                                 start=CHAIN_CLEAN_START['MSFT'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return [d for d, _ in pairs], [p for _, p in pairs], store

    @staticmethod
    def _ten_year(market) -> tuple[list[str], list[float], dict[str, dict[str, Any]]]:
        dates, prices, store = market
        pairs = [(d, p) for d, p in zip(dates, prices) if d >= '2016-04-11']
        return [d for d, _ in pairs], [p for _, p in pairs], store

    def test_ten_year_stop_2x(self, market) -> None:
        """10y, 2x stop: -$252K vs -$184K baseline; 118 stops preempt deep-ITM."""
        dates, prices, store = self._ten_year(market)
        s, trades, eq = run_real_cc_overlay(dates, prices, store,
                                            {**_PARAMS, 'stop_loss_mult': 2.0})
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(-251_775.94, abs=1.0)
        assert s['total_premium_collected'] == pytest.approx(1_000_234.80, abs=1.0)
        assert s['num_calls_sold'] == 256
        actions = [t['action'] for t in trades]
        assert actions.count('close') == 131
        assert actions.count('close_stop') == 118
        assert actions.count('close_itm') == 2
        assert actions.count('expiration') == 4
        assert (s['wins'], s['losses']) == (134, 121)
        assert s['max_drawdown_pct'] == pytest.approx(50.74, abs=0.05)
        assert st['t_stat_newey_west'] == pytest.approx(-1.58, abs=0.02)
        assert st['passes_t_2'] is False

    def test_sixteen_year_stop_2x(self, market) -> None:
        """16y, 2x stop: -$514K vs -$382K baseline. More history, same lesson."""
        dates, prices, store = market
        s, trades, eq = run_real_cc_overlay(dates, prices, store,
                                            {**_PARAMS, 'stop_loss_mult': 2.0})
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(-514_187.18, abs=1.0)
        assert s['num_calls_sold'] == 388
        assert [t['action'] for t in trades].count('close_stop') == 168
        assert st['t_stat_newey_west'] == pytest.approx(-1.14, abs=0.02)
        assert st['passes_t_2'] is False

    def test_tightening_is_monotonically_worse(self, market) -> None:
        """10y P&L ordering: 1.5x < 2x < 3x < no-stop baseline."""
        dates, prices, store = self._ten_year(market)
        pnl = {}
        for mult in (1.5, 3.0):
            s, _, _ = run_real_cc_overlay(dates, prices, store,
                                          {**_PARAMS, 'stop_loss_mult': mult})
            pnl[mult] = s['net_overlay_pnl']
        assert pnl[1.5] == pytest.approx(-374_845.50, abs=1.0)
        assert pnl[3.0] == pytest.approx(-232_950.60, abs=1.0)
        # baseline -183,552.34 is pinned by TestMsftRealChainRegression
        assert pnl[1.5] < -251_775.94 < pnl[3.0] < -183_552.34

    def test_walk_forward_cannot_tune_around_the_stop(self, market) -> None:
        """WF with the 2x stop chains to +154.53% vs +184.97% without (10y, 4y train).

        Given the chance to re-tune in every training window, the optimizer
        retreats to the grid's minimum-engagement corner — delta 0.15 and
        close_at_pct 1.00 in 11/11 windows — and still can't make the stop
        pay: beats-B&H windows drop from 6/11 to 3/11. (On the 16-year span
        the unpinned comparison is +593.89% with the stop vs +520.95%
        without, close=1.00 in 23/23 — the tuned stop variant noses ahead
        there, but both trail buy-and-hold's +913% badly, and the
        fixed-rule stop pinned above is still strictly worse than its
        baseline; not pinned here to keep the suite fast.)
        """
        dates, prices, store = self._ten_year(market)
        records = walk_forward_real(dates, prices, store, PARAM_GRID,
                                    fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask',
                                                  'stop_loss_mult': 2.0},
                                    train_years=4)
        assert len(records) == 11
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(154.53, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(110.61, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(184.82, abs=0.01)
        assert Counter(r['best_params']['call_delta'] for r in records) == {0.15: 11}
        assert Counter(r['best_params']['close_at_pct'] for r in records) == {1.00: 11}
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 3


@pytest.mark.skipif(
    not _HAVE_SPY_DAILIES,
    reason='needs spy_option_dailies.csv or its committed .gz twin',
)
class TestSpyRealWalkForwardRegression:
    """Pin the SPY walk-forward on real chains (2010-12 -> 2026-06, 4y train).

    Third underlying, same verdict: across 23 half-year test windows
    (2014-12 -> 2026-06) the tuned covered call chains to +143% OOS vs
    +171% for buy-and-hold, beating it in 10/23 windows — the closest race
    in the matrix, and still a loss. The 2008 -> late-2010 placeholder-
    greeks era is excluded at load time (CHAIN_CLEAN_START['SPY'] =
    '2010-12-01'; the in-band vendor-junk signature straggles through
    2010-11 on SPY). SPY-specific wrinkle worth keeping pinned: the
    optimizer is LESS extreme here than on MSFT — 0.25 delta wins 9/23
    windows (index premiums are lean but spreads are tight, so moderate
    deltas punish less) — while close=1.00 still dominates at 21/23.
    Every grid fit clears the Pardo 30-trade floor in every window
    (leanest 32).

    The proxy twin (deliberately NOT pinned, per review scope) claims the
    opposite on the same span: +365% chained, beating B&H in 22/23
    windows — the starkest real-vs-proxy inversion in the repo. If that
    comparison is ever published, pin it first.

    Data: spy_option_dailies.csv.gz ships on the data-2026-06 release
    (checksum-verified by CI); the unadjusted close series is committed in
    git like the other tickers'.
    """

    @pytest.fixture(scope='class')
    def records(self) -> list[dict[str, Any]]:
        store = load_chain_store(_SPY_DAILIES, start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return walk_forward_real([d for d, _ in pairs], [p for _, p in pairs], store,
                                 PARAM_GRID,
                                 fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask'},
                                 train_years=4)

    def test_window_layout(self, records) -> None:
        """23 non-overlapping 6-month test windows, 2014-12 -> 2026-06."""
        assert len(records) == 23
        assert records[0]['train_start'] == '2010-12-01'
        assert records[0]['test_start'] == '2014-12-01'
        assert records[-1]['test_start'] == '2025-12-01'
        assert records[-1]['test_end'] == '2026-06-01'

    def test_optimizer_choices(self, records) -> None:
        """Delta is less extreme than MSFT's (0.25 wins 9/23); close=1.00 still rules."""
        assert Counter(r['best_params']['call_delta'] for r in records) == \
            {0.15: 11, 0.20: 3, 0.25: 9}
        assert Counter(int(r['best_params']['dte']) for r in records) == \
            {21: 11, 30: 7, 45: 5}
        assert Counter(r['best_params']['close_at_pct'] for r in records) == \
            {1.00: 21, 0.50: 1, 0.75: 1}
        assert records[0]['best_params'] == {'call_delta': 0.20, 'dte': 21,
                                             'close_at_pct': 1.00}
        assert records[0]['train_sharpe'] == pytest.approx(1.026, abs=5e-4)
        assert records[-1]['train_sharpe'] == pytest.approx(0.749, abs=5e-4)

    def test_pardo_floor(self, records) -> None:
        """Every grid combo clears 30 IS trades in every window (leanest: 32)."""
        assert all(r['n_below_30'] == 0 for r in records)
        assert min(r['min_grid_trades'] for r in records) == 32
        assert min(r['n_trades'] for r in records) == 34

    def test_oos_scoreboard(self, records) -> None:
        """WF +143.0% vs fixed +95.8% vs B&H +170.9%; beats B&H in 10/23."""
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(143.01, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(95.78, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(170.87, abs=0.01)
        assert sum(r['oos_return_pct'] > r['fixed_return_pct'] for r in records) == 19
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 10


# The 2026-06 grid-expansion experiment: every edge the 27-combo PARAM_GRID
# was pinned against, widened one step — delta down to 0.10 and up to 0.30,
# dte down to 14. The non-expansions are principled: close_at_pct=1.00 is
# hold-to-expiry (nothing beyond it), dte=60 would put the 4y-train roll
# count (~24) under the Pardo 30-trade floor, and delta 0.05 sits on
# select_entry's band edge. Test-local on purpose: widening the production
# PARAM_GRID would silently re-pin every walk-forward regression.
EXPANDED_PARAM_GRID: dict[str, list[float]] = {
    'call_delta': [0.10, 0.15, 0.20, 0.25, 0.30],
    'dte': [14, 21, 30, 45],  # CALENDAR days, like PARAM_GRID
    'close_at_pct': [0.50, 0.75, 1.00],
}


@pytest.mark.skipif(
    not _HAVE_SPY_DAILIES,
    reason='needs spy_option_dailies.csv or its committed .gz twin',
)
class TestSpyExpandedGridRegression:
    """Pin the SPY walk-forward on the 60-combo expanded grid: nothing changes.

    Doubling the search space (27 -> 60 combos, EXPANDED_PARAM_GRID above)
    along every edge the baseline grid was pinned against moves the chained
    OOS +143.01% -> +143.62% — same 10/23 windows over buy-and-hold's
    +170.87%, same verdict. The optimizer sprints to the new edges (the new
    dte=14 is instantly modal at 10/23; the new deltas 0.10/0.30 together
    take 11/23) while the delta distribution flattens from the baseline's
    bimodal edge-pinning to near-uniform — so the edge-pinning in the
    27-grid pin was noise-chasing, not a truncated optimum. Selection, not
    menu, is the binding constraint: in the experiment's per-combo
    train/test matrix (measured 2026-06, NOT pinned — re-derive before
    publishing) mean Spearman(IS Sharpe, OOS return) was +0.13, the IS
    pick landed at the 59th OOS percentile of its 60-combo field, and the
    hindsight oracle rose +220% -> +250% chained (20/23 over B&H) while
    the realizable pick gained 0.6pp — grid expansion widens the ceiling,
    not the floor you can stand on. close=1.00 stays dominant (18/23), the
    one stable parameter across both grids. The lever this argues for is
    Pardo's cut-parameters, not a wider menu.

    Setup is otherwise identical to TestSpyRealWalkForwardRegression:
    4y train, bid/ask fills, CHAIN_CLEAN_START['SPY'] era clip, 23
    half-year test windows 2014-12 -> 2026-06. Every grid combo clears
    the Pardo 30-trade floor in every window (leanest 32, unchanged from
    the baseline grid).
    """

    @pytest.fixture(scope='class')
    def records(self) -> list[dict[str, Any]]:
        store = load_chain_store(_SPY_DAILIES, start=CHAIN_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return walk_forward_real([d for d, _ in pairs], [p for _, p in pairs], store,
                                 EXPANDED_PARAM_GRID,
                                 fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask'},
                                 train_years=4)

    def test_window_layout(self, records) -> None:
        """Same 23 windows as the baseline pin — the grid doesn't move them."""
        assert len(records) == 23
        assert records[0]['train_start'] == '2010-12-01'
        assert records[0]['test_start'] == '2014-12-01'
        assert records[-1]['test_start'] == '2025-12-01'
        assert records[-1]['test_end'] == '2026-06-01'

    def test_optimizer_choices(self, records) -> None:
        """New edges win immediately (dte=14 modal, 0.10/0.30 take 11/23) — for nothing."""
        assert Counter(r['best_params']['call_delta'] for r in records) == \
            {0.10: 5, 0.15: 4, 0.20: 3, 0.25: 5, 0.30: 6}
        assert Counter(int(r['best_params']['dte']) for r in records) == \
            {14: 10, 21: 3, 30: 6, 45: 4}
        assert Counter(r['best_params']['close_at_pct'] for r in records) == \
            {1.00: 18, 0.75: 5}
        assert records[0]['best_params'] == {'call_delta': 0.20, 'dte': 14,
                                             'close_at_pct': 0.75}
        assert records[0]['train_sharpe'] == pytest.approx(1.056, abs=5e-4)
        assert records[-1]['best_params'] == {'call_delta': 0.30, 'dte': 30,
                                              'close_at_pct': 0.75}
        assert records[-1]['train_sharpe'] == pytest.approx(0.757, abs=5e-4)

    def test_pardo_floor(self, records) -> None:
        """All 60 combos clear 30 IS trades in every window (leanest: 32)."""
        assert all(r['n_below_30'] == 0 for r in records)
        assert min(r['min_grid_trades'] for r in records) == 32
        assert min(r['n_trades'] for r in records) == 37

    def test_oos_scoreboard(self, records) -> None:
        """WF +143.6% vs fixed +95.8% vs B&H +170.9% — 0.6pp for doubling the grid."""
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(143.62, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(95.78, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(170.87, abs=0.01)
        assert sum(r['oos_return_pct'] > r['fixed_return_pct'] for r in records) == 18
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 10


@pytest.mark.skipif(
    not (_HAVE_DAILIES and _HAVE_QQQ_BACKFILL),
    reason='needs qqq_option_dailies.csv and qqq_option_dailies_2011_2016.csv '
           '(or their .gz twins)',
)
class TestQqqExtendedWalkForwardRegression:
    """Pin the QQQ 15-year walk-forward on merged real chains (4y train, bid/ask).

    Third underlying, same verdict, completing the matrix: across 22
    half-year test windows (2015-03 -> 2026-03) the tuned covered call
    chains to +283% OOS vs +418% buy-and-hold, beating it in only 5/22
    windows. Cross-ticker tally at this pin: 22/68 real-chain windows beat
    buy-and-hold (MSFT 7/23, SPY 10/23, QQQ 5/22) vs 55/68 for the proxy
    on the same spans. QQQ's quirk: the optimizer prefers the 45-day
    cycle more than any other ticker (10/22 windows, mostly 2016-2019)
    before converging on low-delta/short-cycle post-2020; close=1.00 still
    dominates at 19/22. Every grid fit clears the Pardo floor in every
    window (leanest 34).

    The proxy twin (not pinned, per the SPY precedent): +749.72% chained,
    beats B&H 17/22, maximum-engagement picks. Pin before publishing.

    Data: requires both the canonical QQQ dailies and the 2011-2016
    backfill (which begins at the QQQQ->QQQ ticker rename; the QQQQ era
    carries placeholder vendor greeks and is excluded — see CLAUDE.md's
    pipeline era-gotchas). The unadjusted close series extends to 2011 in
    git; the canonical-span QQQ pins are unaffected (they clip to their
    own store's span).
    """

    @pytest.fixture(scope='class')
    def records(self) -> list[dict[str, Any]]:
        store = load_chain_store(_DAILIES, [_QQQ_BACKFILL])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('QQQ', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        return walk_forward_real([d for d, _ in pairs], [p for _, p in pairs], store,
                                 PARAM_GRID,
                                 fixed_params={**FIXED_PARAMS, 'fill': 'bid_ask'},
                                 train_years=4)

    def test_window_layout(self, records) -> None:
        """22 non-overlapping 6-month test windows, 2015-03 -> 2026-03."""
        assert len(records) == 22
        assert records[0]['train_start'] == '2011-03-23'
        assert records[0]['test_start'] == '2015-03-23'
        assert records[-1]['test_start'] == '2025-09-23'
        assert records[-1]['test_end'] == '2026-03-23'

    def test_optimizer_choices(self, records) -> None:
        """The 45-day cycle wins a plurality (10/22) — a QQQ-specific quirk."""
        assert Counter(r['best_params']['call_delta'] for r in records) == \
            {0.15: 10, 0.20: 6, 0.25: 6}
        assert Counter(int(r['best_params']['dte']) for r in records) == \
            {45: 10, 21: 6, 30: 6}
        assert Counter(r['best_params']['close_at_pct'] for r in records) == \
            {1.00: 19, 0.50: 2, 0.75: 1}
        assert records[0]['train_sharpe'] == pytest.approx(1.201, abs=5e-4)
        assert records[-1]['train_sharpe'] == pytest.approx(0.672, abs=5e-4)

    def test_pardo_floor(self, records) -> None:
        """Every grid combo clears 30 IS trades in every window (leanest: 34)."""
        assert all(r['n_below_30'] == 0 for r in records)
        assert min(r['n_trades'] for r in records) == 34

    def test_oos_scoreboard(self, records) -> None:
        """WF +283.2% vs fixed +208.8% vs B&H +418.0%; beats B&H in 5/22."""
        assert _chain([r['oos_return_pct'] for r in records]) == pytest.approx(283.25, abs=0.01)
        assert _chain([r['fixed_return_pct'] for r in records]) == pytest.approx(208.75, abs=0.01)
        assert _chain([r['bh_return_pct'] for r in records]) == pytest.approx(417.96, abs=0.01)
        assert sum(r['oos_return_pct'] > r['fixed_return_pct'] for r in records) == 15
        assert sum(r['oos_return_pct'] > r['bh_return_pct'] for r in records) == 5
