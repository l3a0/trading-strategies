"""Gap A — the trade-level R-multiple ledger (common/trade_ledger.py).

Two layers, per the repo pattern (docs/van_tharp_gap_a.md):

- ``TestTradeLedgerMechanics`` — always-run synthetic layer: hand-built event
  streams, every assertion against a hand-derived value. Covers the reducer
  (pairing, the three R bases + the mixed-sign floor, MAE finalization,
  settle_leg skipping, dangling-entry dropping) and the statistics
  (expectancy, SQN, ``r_newey_west_t`` against a by-hand Bartlett
  computation, the ex-post ``avg_loss_1r`` normalizer).
- ``TestTradeLedgerRegression`` — dataset-gated: pins the ledger statistics
  of two already-pinned real overlays (the MSFT covered call of
  ``TestMsftRealChainRegression`` and the SPY short-vol overlay of
  ``TestSpyShortVolRegression``). EXPLORATORY numbers (kill-or-justify, never
  a registered verdict); the daily Newey-West t remains the significance
  authority — these pins exist so the measurement, once made, stays made.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.trade_ledger import (
    TradeRecord,
    build_trade_ledger,
    ledger_statistics,
)

_DATA = Path(__file__).resolve().parent.parent / 'data'
_MSFT_DAILIES = _DATA / 'msft_option_dailies.csv'
_SPY_DAILIES = _DATA / 'spy_option_dailies.csv'


def _have(base: Path) -> bool:
    return base.exists() or base.with_suffix('.csv.gz').exists()


class TestTradeLedgerMechanics:
    """Always-run synthetic layer — no datasets, hand-derived values only."""

    def test_premium_collected_basis(self) -> None:
        """CC sell/expiration pair: R = premium × shares; win; MAE from event."""
        trades = [
            {'date': '2020-01-02', 'action': 'sell', 'premium': 2.0, 'strike': 100.0, 'pnl': 0},
            {'date': '2020-02-01', 'action': 'expiration', 'pnl': 700.0, 'mae': -300.0},
        ]
        recs = build_trade_ledger(trades, strategy='covered_call', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        assert len(recs) == 1
        r = recs[0]
        assert r.initial_risk == 200.0          # 2.0/share × 100 shares
        assert r.r_multiple == pytest.approx(3.5)
        assert r.mae == -300.0 and r.mae_r == pytest.approx(-1.5)
        assert r.outcome == 'win'
        assert r.risk_basis == 'premium_collected'
        assert (r.entry_date, r.close_date) == ('2020-01-02', '2020-02-01')

    def test_defined_max_loss_credit_spread(self) -> None:
        """Bull put spread: R/share = width − net credit = 5 − 0.90 = 4.10."""
        trades = [
            {'date': '2020-01-02', 'action': 'enter', 'legs': 2, 'credit': 0.90,
             'legs_detail': [
                 {'sign': -1, 'right': 'put', 'strike': 100.0, 'entry_net': 1.30, 'expiration': '2020-02-01'},
                 {'sign': 1, 'right': 'put', 'strike': 95.0, 'entry_net': 0.40, 'expiration': '2020-02-01'},
             ]},
            {'date': '2020-02-01', 'action': 'settle', 'pnl': 90.0, 'mae': -120.0},
        ]
        recs = build_trade_ledger(trades, strategy='credit_spread', ticker='T',
                                  shares=100, risk_basis='defined_max_loss')
        r = recs[0]
        assert r.initial_risk == pytest.approx(410.0)
        assert r.r_multiple == pytest.approx(90.0 / 410.0, abs=1e-4)
        assert r.risk_basis == 'defined_max_loss'

    def test_defined_max_loss_iron_condor_max_width(self) -> None:
        """IC with asymmetric wings: R uses the WIDER wing (10) minus net credit."""
        legs = [
            {'sign': -1, 'right': 'call', 'strike': 110.0, 'entry_net': 1.00, 'expiration': 'e'},
            {'sign': 1, 'right': 'call', 'strike': 115.0, 'entry_net': 0.30, 'expiration': 'e'},
            {'sign': -1, 'right': 'put', 'strike': 90.0, 'entry_net': 1.10, 'expiration': 'e'},
            {'sign': 1, 'right': 'put', 'strike': 80.0, 'entry_net': 0.20, 'expiration': 'e'},
        ]
        net = 1.00 - 0.30 + 1.10 - 0.20          # 1.60
        trades = [
            {'date': 'd1', 'action': 'enter', 'legs': 4, 'credit': net, 'legs_detail': legs},
            {'date': 'd2', 'action': 'settle', 'pnl': 160.0},
        ]
        recs = build_trade_ledger(trades, strategy='iron_condor', ticker='T',
                                  shares=100, risk_basis='defined_max_loss')
        assert recs[0].initial_risk == pytest.approx((10.0 - net) * 100)

    def test_stop_distance_basis(self) -> None:
        """Stopped CC at 2× premium: R = (mult−1) × premium × shares = one premium."""
        trades = [
            {'date': 'd1', 'action': 'sell', 'premium': 2.0, 'strike': 100.0, 'pnl': 0},
            {'date': 'd2', 'action': 'close_stop', 'pnl': -210.0, 'mae': -215.0},
        ]
        recs = build_trade_ledger(trades, strategy='covered_call', ticker='T',
                                  shares=100, risk_basis='stop_distance', stop_loss_mult=2.0)
        r = recs[0]
        assert r.initial_risk == pytest.approx(200.0)
        assert r.r_multiple == pytest.approx(-1.05)
        assert r.outcome == 'loss'

    def test_mixed_sign_premium_floor(self) -> None:
        """Risk-reversal-shaped entry (net ≈ 0): R floors at the gross short
        premium and the basis string records the normalization."""
        trades = [
            {'date': 'd1', 'action': 'enter', 'legs': 2, 'credit': 0.05,
             'legs_detail': [
                 {'sign': -1, 'right': 'put', 'strike': 95.0, 'entry_net': 1.50, 'expiration': 'e'},
                 {'sign': 1, 'right': 'call', 'strike': 105.0, 'entry_net': 1.45, 'expiration': 'e'},
             ]},
            {'date': 'd2', 'action': 'settle', 'pnl': 30.0},
        ]
        recs = build_trade_ledger(trades, strategy='risk_reversal', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        r = recs[0]
        assert r.initial_risk == pytest.approx(150.0)   # gross short 1.50, not net 0.05
        assert r.risk_basis == 'premium_collected_abs'

    def test_all_short_floor_survives_float_noise(self) -> None:
        """1-ulp regression: the event credit is round(x, 4) but legs_detail
        carries the unrounded entry_net (e.g. 2.345 − 0.0065 =
        2.3385000000000002 vs credit 2.3385). The floor decision compares
        within the 4dp quantum, so a pure all-short structure stays
        'premium_collected' — exact equality mislabelled 25% of the SPY
        short-vol ledger as 'premium_collected_abs'."""
        entry_net = 2.345 - 0.0065               # 2.3385000000000002 (binary)
        trades = [
            {'date': 'd1', 'action': 'enter', 'legs': 1, 'credit': round(entry_net, 4),
             'legs_detail': [
                 {'sign': -1, 'right': 'call', 'strike': 100.0, 'entry_net': entry_net,
                  'expiration': 'e'},
             ]},
            {'date': 'd2', 'action': 'settle', 'pnl': 100.0},
        ]
        recs = build_trade_ledger(trades, strategy='short_vol', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        assert recs[0].risk_basis == 'premium_collected'
        assert recs[0].initial_risk == pytest.approx(233.85)

    def test_all_short_floor_is_noop(self) -> None:
        """Straddle-shaped entry (all-short): gross short == net credit, so the
        floor changes nothing and the basis stays 'premium_collected'."""
        trades = [
            {'date': 'd1', 'action': 'enter', 'legs': 2, 'credit': 3.0,
             'legs_detail': [
                 {'sign': -1, 'right': 'call', 'strike': 100.0, 'entry_net': 1.6, 'expiration': 'e'},
                 {'sign': -1, 'right': 'put', 'strike': 100.0, 'entry_net': 1.4, 'expiration': 'e'},
             ]},
            {'date': 'd2', 'action': 'settle', 'pnl': -450.0, 'mae': -500.0},
        ]
        recs = build_trade_ledger(trades, strategy='short_straddle', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        r = recs[0]
        assert r.initial_risk == pytest.approx(300.0)
        assert r.risk_basis == 'premium_collected'
        assert r.r_multiple == pytest.approx(-1.5)      # the fat tail reads past −1R

    def test_mae_finalized_with_pnl(self) -> None:
        """A loser whose settle P&L is worse than any daily mark: final MAE
        includes where the trade ended (min of event mae and pnl)."""
        trades = [
            {'date': 'd1', 'action': 'sell', 'premium': 1.0, 'strike': 100.0, 'pnl': 0},
            {'date': 'd2', 'action': 'expiration', 'pnl': -800.0, 'mae': -350.0},
        ]
        recs = build_trade_ledger(trades, strategy='covered_call', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        assert recs[0].mae == -800.0

    def test_missing_mae_degrades_to_pnl_floor(self) -> None:
        """An event stream without 'mae' (pre-A2 caller) yields min(pnl, 0)."""
        trades = [
            {'date': 'd1', 'action': 'sell', 'premium': 1.0, 'strike': 100.0, 'pnl': 0},
            {'date': 'd2', 'action': 'expiration', 'pnl': 100.0},
        ]
        recs = build_trade_ledger(trades, strategy='covered_call', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        assert recs[0].mae == 0.0

    def test_settle_leg_skipped_and_dangling_entry_dropped(self) -> None:
        """Calendar-style stream: settle_leg is informational (folded into the
        final settle) and a trailing open entry produces no record."""
        trades = [
            {'date': 'd1', 'action': 'enter', 'legs': 2, 'credit': -1.2,
             'legs_detail': [
                 {'sign': -1, 'right': 'call', 'strike': 100.0, 'entry_net': 2.0, 'expiration': 'near'},
                 {'sign': 1, 'right': 'call', 'strike': 100.0, 'entry_net': 3.2, 'expiration': 'far'},
             ]},
            {'date': 'd2', 'action': 'settle_leg', 'right': 'call', 'strike': 100.0,
             'expiration': 'near', 'pnl': -50.0},
            {'date': 'd3', 'action': 'settle', 'pnl': 40.0, 'mae': -90.0},
            {'date': 'd4', 'action': 'enter', 'legs': 2, 'credit': -1.0,
             'legs_detail': [
                 {'sign': -1, 'right': 'call', 'strike': 100.0, 'entry_net': 2.0, 'expiration': 'near'},
                 {'sign': 1, 'right': 'call', 'strike': 100.0, 'entry_net': 3.0, 'expiration': 'far'},
             ]},
        ]
        recs = build_trade_ledger(trades, strategy='calendar', ticker='T',
                                  shares=100, risk_basis='premium_collected')
        assert len(recs) == 1                    # one completed cycle; dangler dropped
        r = recs[0]
        assert (r.entry_date, r.close_date) == ('d1', 'd3')
        # net is a DEBIT (−1.2): floor picks max(1.2, gross short 2.0) = 2.0
        assert r.initial_risk == pytest.approx(200.0)
        assert r.risk_basis == 'premium_collected_abs'

    def test_ledger_statistics_hand_computed(self) -> None:
        """r = [1.0, 0.5, −0.5, 1.0]: mean 0.5, sample std √0.5, SQN = √4·0.5/√0.5
        = √2; NW at L=1 gives t = √3 (γ₁ = −1/6, S = 0.5 − 1/6 = 1/3)."""
        recs = [
            TradeRecord('s', 'T', 'd1', 'd2', pnl=p, risk_basis='premium_collected',
                        initial_risk=100.0, r_multiple=p / 100.0, mae=min(p, 0.0),
                        mae_r=min(p, 0.0) / 100.0, outcome='win' if p >= 0 else 'loss')
            for p in (100.0, 50.0, -50.0, 100.0)
        ]
        stats = ledger_statistics(recs)
        assert stats['n'] == 4
        assert stats['expectancy_r'] == pytest.approx(0.5)
        assert stats['sqn'] == pytest.approx(math.sqrt(2), abs=1e-3)
        assert stats['r_newey_west_t'] == pytest.approx(math.sqrt(3), abs=1e-3)
        assert stats['win_rate'] == 75.0
        assert stats['avg_win_r'] == pytest.approx((1.0 + 0.5 + 1.0) / 3, abs=1e-4)
        assert stats['avg_loss_r'] == pytest.approx(-0.5)
        assert stats['mae_r_distribution']['worst'] == pytest.approx(-0.5)

    def test_avg_loss_1r_normalizer(self) -> None:
        """Ex-post Tharp fallback: 1R := mean |losing pnl| = 50, so r = pnl/50 —
        expectancy doubles vs the declared 100-dollar basis."""
        recs = [
            TradeRecord('s', 'T', 'd1', 'd2', pnl=p, risk_basis='premium_collected',
                        initial_risk=100.0, r_multiple=p / 100.0, mae=min(p, 0.0),
                        mae_r=min(p, 0.0) / 100.0, outcome='win' if p >= 0 else 'loss')
            for p in (100.0, 50.0, -50.0, 100.0)
        ]
        stats = ledger_statistics(recs, r_normalizer='avg_loss_1r')
        assert stats['r_normalizer'] == 'avg_loss_1r'
        assert stats['expectancy_r'] == pytest.approx(1.0)   # mean pnl 50 / 1R 50
        # no losers -> falls back to declared, and says so
        winners = [r for r in recs if r.outcome == 'win']
        fallback = ledger_statistics(winners, r_normalizer='avg_loss_1r')
        assert fallback['r_normalizer'] == 'declared'

    def test_guards(self) -> None:
        """Loud failures: unknown basis, stop basis without a mult, defined-risk
        without legs_detail."""
        sell = {'date': 'd1', 'action': 'sell', 'premium': 2.0, 'strike': 100.0, 'pnl': 0}
        term = {'date': 'd2', 'action': 'expiration', 'pnl': 0.0}
        with pytest.raises(ValueError, match='unknown risk_basis'):
            build_trade_ledger([sell, term], strategy='s', ticker='T',
                               shares=100, risk_basis='notional')
        with pytest.raises(ValueError, match='stop_loss_mult'):
            build_trade_ledger([sell, term], strategy='s', ticker='T',
                               shares=100, risk_basis='stop_distance')
        enter_no_legs = {'date': 'd1', 'action': 'enter', 'legs': 2, 'credit': 1.0}
        with pytest.raises(ValueError, match='legs_detail'):
            build_trade_ledger([enter_no_legs, term], strategy='s', ticker='T',
                               shares=100, risk_basis='defined_max_loss')


@pytest.mark.skipif(not (_have(_MSFT_DAILIES) and _have(_SPY_DAILIES)),
                    reason='needs msft/spy option dailies (data-2026-06 release)')
class TestTradeLedgerRegression:
    """Pin the ledger statistics of two already-pinned real overlays.

    EXPLORATORY (docs/van_tharp_gap_a.md): these pins settle the measurement,
    they do not promote it — the daily Newey-West t stays the significance
    authority. Runs mirror TestMsftRealChainRegression (real CC, bid/ask,
    canonical span) and TestSpyShortVolRegression (0.25Δ/30-DTE short call,
    REGISTERED_CLEAN_START['SPY'] span, frictionless hedge).

    Basis caveat: trade-event pnl is the RAW OPTION-CYCLE P&L (premium vs
    settlement/buyback) — no hedge P&L, no rf credit. So SPY's negative
    per-cycle expectancy here does NOT contradict TestSpyShortVolRegression's
    +2.54 daily Newey-West t: that headline is the delta-hedged-gain measure
    (hedge netted daily, rf on the cash base). Same overlay, different
    measurement objects. The first real ledger output is the Van Tharp
    win-rate-vs-expectancy flip itself: both overlays win ~two-thirds of the
    time with negative per-trade expectancy and a fat left MAE tail.
    """

    @pytest.fixture(scope='class')
    def msft_ledger(self) -> list[TradeRecord]:
        from realchains.real_cc_backtest import (
            load_chain_store,
            load_unadjusted_prices,
            run_real_cc_overlay,
        )
        store = load_chain_store(str(_MSFT_DAILIES))
        days = sorted(store)
        dates, prices = load_unadjusted_prices('MSFT', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        s, trades, _ = run_real_cc_overlay(
            [d for d, _ in pairs], [p for _, p in pairs], store,
            {'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 30,
             'risk_free_rate': 0.045, 'capital': 100_000},
        )
        return build_trade_ledger(trades, strategy='covered_call', ticker='MSFT',
                                  shares=100 * s['num_contracts'],
                                  risk_basis='premium_collected')

    @pytest.fixture(scope='class')
    def spy_ledger(self) -> list[TradeRecord]:
        from realchains.real_cc_backtest import (
            REGISTERED_CLEAN_START,
            load_chain_store,
            load_unadjusted_prices,
        )
        from realchains.vol_premium import run_real_short_vol_overlay
        store = load_chain_store(str(_SPY_DAILIES), start=REGISTERED_CLEAN_START['SPY'])
        days = sorted(store)
        dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
        pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
        s, trades, _ = run_real_short_vol_overlay(
            [d for d, _ in pairs], [p for _, p in pairs], store,
            {'target_delta': 0.25, 'dte': 30, 'capital': 100_000,
             'risk_free_rate': 0.045, 'hedge_cost_bps': 0.0},
        )
        return build_trade_ledger(trades, strategy='short_vol', ticker='SPY',
                                  shares=100 * s['num_contracts'],
                                  risk_basis='premium_collected')

    def test_msft_cc_ledger(self, msft_ledger: list[TradeRecord]) -> None:
        """182 completed cycles: 68.1% win rate, −0.39R expectancy — the flip."""
        stats = ledger_statistics(msft_ledger)
        assert stats['n'] == 182
        assert stats['expectancy_r'] == pytest.approx(-0.3901, abs=0.005)
        assert stats['sqn'] == pytest.approx(-2.70, abs=0.01)
        assert stats['r_newey_west_t'] == pytest.approx(-2.797, abs=0.01)
        assert stats['win_rate'] == pytest.approx(68.1, abs=0.1)
        assert stats['avg_loss_r'] == pytest.approx(-2.9788, abs=0.005)
        assert stats['mae_r_distribution']['worst'] == pytest.approx(-7.4156, abs=0.05)

    def test_spy_short_vol_ledger(self, spy_ledger: list[TradeRecord]) -> None:
        """174 completed cycles (the 175th is an open dangler, dropped): 65.5%
        win rate, −0.54R expectancy, worst MAE −11.4R — the fat left tail."""
        stats = ledger_statistics(spy_ledger)
        assert stats['n'] == 174
        assert stats['expectancy_r'] == pytest.approx(-0.5407, abs=0.005)
        assert stats['sqn'] == pytest.approx(-2.815, abs=0.01)
        assert stats['r_newey_west_t'] == pytest.approx(-3.224, abs=0.01)
        assert stats['win_rate'] == pytest.approx(65.5, abs=0.1)
        assert stats['avg_loss_r'] == pytest.approx(-3.3769, abs=0.005)
        assert stats['mae_r_distribution']['worst'] == pytest.approx(-11.4131, abs=0.05)
