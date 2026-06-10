"""Tests for real_cc_backtest.py — the real-option-chain overlay adapter.

Two layers:

- Pure-logic unit tests (always run, including CI): entry selection and
  fill-model arithmetic against synthetic chain slices.
- TestQqqRealChainRegression: pins the headline numbers of the real-premium
  QQQ run. Requires qqq_option_dailies.csv (or its .gz twin), which is too
  large for git history and ships as a release asset (data-2026-06): CI
  downloads and checksum-verifies it before pytest, fetch_option_data.sh
  does the same locally, and the class skips gracefully when neither file
  is present. It locks the adversarially-reviewed result: the covered call
  that the proxy engine scores at +$120K net overlay P&L loses $157K on
  real premiums (Newey-West t = -1.78).
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from cc_backtest import compute_statistics, run_cc_overlay
from real_cc_backtest import (
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
    select_entry,
)

_DAILIES = os.path.join(os.path.dirname(__file__), 'qqq_option_dailies.csv')
_UNADJ = os.path.join(os.path.dirname(__file__), 'qqq_10yr_prices_unadjusted.csv')
_HAVE_DAILIES = os.path.exists(_DAILIES) or os.path.exists(_DAILIES + '.gz')

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
        """198 calls; 122 profit-target closes, 71 deep-ITM closes, 5 expirations."""
        s, trades, _ = real
        assert s['num_calls_sold'] == 198
        assert s['wins'] == 127
        assert s['losses'] == 71
        assert s['win_rate'] == pytest.approx(64.1, abs=0.1)
        actions = [t['action'] for t in trades]
        assert actions.count('close') == 122
        assert actions.count('close_itm') == 71
        assert actions.count('expiration') == 5

    def test_significance(self, real) -> None:
        """NW t = -1.78: the real overlay's harm is near-significant."""
        s, _, eq = real
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert st['t_stat_newey_west'] == pytest.approx(-1.78, abs=0.02)
        assert st['sharpe_excess'] == pytest.approx(-0.53, abs=0.01)
        assert st['passes_t_2'] is False

    def test_mid_fill_variant(self, market) -> None:
        """Mid fills recover only ~$12K of the loss: spread is not the driver."""
        dates, prices, store = market
        s, _, eq = run_real_cc_overlay(dates, prices, store, {**_PARAMS, 'fill': 'mid'})
        st = compute_statistics(eq, num_contracts=s['num_contracts'], cash=s['cash'])
        assert s['net_overlay_pnl'] == pytest.approx(-144_744.30, abs=1.0)
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
