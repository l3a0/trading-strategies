"""Tests for realchains/cc_r_experiment.py (docs/spy_cc_r_experiment_plan.md).

Two layers, the house pattern:

- An always-run synthetic layer for the accounting machinery — above all the
  excess-path cycle attribution (the Book-H seam fix, plan §3), whose
  conservation identity (summed cycle P&L + tail residual == final excess,
  flat excess outside windows) is what lets the hedged book's R include the
  hedge share P&L the engine's event stream never records.
- A dataset-gated `TestSpyCcRExperiment` regression pinning the decisive
  numbers of the one measurement run (added with the run, per plan §11).

Epistemic status: EXPLORATORY (Gap E precedent) — kill-or-justify, no idea-
ledger rows, no e-value spent; the daily Newey-West t stays the significance
authority.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from common.paths import data_path
from common.trade_ledger import TradeRecord
from realchains.cc_r_experiment import (
    BASE_PARAMS,
    CC_R_SEED,
    attribute_cycles,
    exit_grid,
    open_tail_entry,
    overlay_excess,
    overwrite_ratio_curve,
    paired_direction_bill,
    regime_spread,
    run_experiment,
)

_SPY_DAILIES = data_path('spy_option_dailies.csv')
_HAVE_SPY = os.path.exists(_SPY_DAILIES) or os.path.exists(_SPY_DAILIES + '.gz')


def _rec(entry: str, close: str, pnl: float, risk: float) -> TradeRecord:
    return TradeRecord(
        strategy='covered_call', ticker='SPY',
        entry_date=entry, close_date=close, pnl=pnl,
        risk_basis='premium_collected', initial_risk=risk,
        r_multiple=round(pnl / risk, 4), mae=0.0, mae_r=0.0,
        outcome='win' if pnl >= 0 else 'loss',
    )


DATES = [f'2020-01-{d:02d}' for d in range(1, 11)]   # d1..d10, weekday-agnostic


class TestAttributeCycles:
    """The seam fix's arithmetic on hand-built excess paths."""

    def test_two_cycles_conserve_and_recompute(self) -> None:
        """Windowed deltas reproduce cycle P&L, MAE is the windowed running
        minimum, and the conservation identity holds with a flat tail."""
        #        d1  d2   d3   d4  d5  d6  d7  d8  d9  d10
        excess = [0., -5., -12., 30., 30., 28., 10., 45., 45., 45.]
        records = [_rec(DATES[1], DATES[3], 30.0, 100.0),   # d2 -> d4
                   _rec(DATES[5], DATES[7], 15.0, 50.0)]    # d6 -> d8
        adj, gap_drift, tail, raw_sum = attribute_cycles(records, DATES, excess)
        assert gap_drift == 0.0
        assert tail == 0.0
        assert [r.pnl for r in adj] == [30.0, 15.0]
        assert adj[0].r_multiple == pytest.approx(0.30)
        assert adj[1].r_multiple == pytest.approx(0.30)
        # MAE: cycle 1 dips to -12 from base 0; cycle 2 dips to 10 from base 30.
        assert adj[0].mae == -12.0 and adj[0].mae_r == pytest.approx(-0.12)
        assert adj[1].mae == -20.0 and adj[1].mae_r == pytest.approx(-0.40)
        assert raw_sum + tail == pytest.approx(excess[-1])

    def test_hedged_style_divergence_is_measured(self) -> None:
        """When the excess path carries P&L the event stream didn't (the
        hedge), the attributed pnl differs from the record's — that
        difference is the measurement, and outcome/r flip with it."""
        excess = [0., 0., -80., -60., -60., -60., -60., -60., -60., -60.]
        records = [_rec(DATES[1], DATES[3], 40.0, 100.0)]   # engine said +40
        adj, gap_drift, _, _ = attribute_cycles(records, DATES, excess)
        assert gap_drift == 0.0
        assert adj[0].pnl == -60.0
        assert adj[0].r_multiple == pytest.approx(-0.60)
        assert adj[0].outcome == 'loss'
        assert adj[0].initial_risk == 100.0        # risk basis never changes

    def test_gap_drift_detected(self) -> None:
        """Excess moving OUTSIDE every cycle window breaks the flat-between-
        cycles premise and must surface as gap_drift."""
        excess = [0., 0., 10., 10., 17., 17., 17., 17., 17., 17.]  # d5 leaks +7
        records = [_rec(DATES[1], DATES[2], 10.0, 100.0)]
        _, gap_drift, _, _ = attribute_cycles(records, DATES, excess)
        assert gap_drift == pytest.approx(7.0)

    def test_open_tail_residual_excluded_from_drift(self) -> None:
        """A still-open final position's excess belongs to tail_residual,
        not gap_drift, and completes the conservation identity."""
        excess = [0., 0., 25., 25., 25., 25., 25., 25., 20., 12.]
        records = [_rec(DATES[1], DATES[2], 25.0, 100.0)]
        adj, gap_drift, tail, raw_sum = attribute_cycles(
            records, DATES, excess, open_entry_date=DATES[8])
        assert gap_drift == 0.0
        assert tail == pytest.approx(-13.0)        # 12 - 25
        assert raw_sum + tail == pytest.approx(excess[-1])

    def test_entry_on_first_day_uses_zero_base(self) -> None:
        excess = [-3., 8., 8., 8., 8., 8., 8., 8., 8., 8.]
        records = [_rec(DATES[0], DATES[1], 8.0, 10.0)]
        adj, gap_drift, _, _ = attribute_cycles(records, DATES, excess)
        assert gap_drift == 0.0
        assert adj[0].pnl == 8.0
        assert adj[0].mae == -3.0                  # the entry-day dip counts


class TestPairedDirectionBill:
    def test_bill_is_u_minus_h(self) -> None:
        """Plan §3's frozen convention: the bill is R_U − R_H (negative mean
        = Book U underperforms its hedged twin); hedge_pnl_dollars is the
        separately-labeled hedge flow, sum(H − U)."""
        u = [_rec(DATES[1], DATES[3], 10.0, 100.0),
             _rec(DATES[5], DATES[7], -50.0, 100.0)]
        h = [_rec(DATES[1], DATES[3], 20.0, 100.0),
             _rec(DATES[5], DATES[7], -10.0, 100.0)]
        p = paired_direction_bill(u, h)
        assert p['n'] == 2
        assert p['mean_r'] == pytest.approx(-0.25)     # (-0.1 + -0.4) / 2
        assert p['median_r'] == pytest.approx(-0.25)
        assert p['hedge_pnl_dollars'] == pytest.approx(50.0)
        assert p['share_positive'] == 0.0              # U never beat H here

    def test_empty_books_report_zero(self) -> None:
        p = paired_direction_bill([], [])
        assert p == {'n': 0, 'mean_r': 0.0, 'median_r': 0.0,
                     'hedge_pnl_dollars': 0.0, 'share_positive': 0.0}

    def test_cycle_mismatch_raises(self) -> None:
        u = [_rec(DATES[1], DATES[3], 10.0, 100.0)]
        h = [_rec(DATES[2], DATES[3], 10.0, 100.0)]
        with pytest.raises(ValueError, match='cycle mismatch'):
            paired_direction_bill(u, h)
        with pytest.raises(ValueError, match='cycle counts differ'):
            paired_direction_bill(u, u + u)


class TestGridAndHelpers:
    def test_exit_grid_is_3x3_crossed_with_deep_itm(self) -> None:
        """The plan §4 3x3 crossed with the 2026-07-19 deep-ITM widening:
        managed cells first (original pin order preserved), then _noitm."""
        grid = exit_grid()
        assert len(grid) == 18
        assert grid[0] == (1.0, None, True)         # row-major, managed first
        assert (0.75, None, True) in grid           # the published baseline
        assert grid[9] == (1.0, None, False)        # the true hold-to-expiry
        assert {c for c, _, _ in grid} == {1.0, 0.75, 0.50}
        assert {s for _, s, _ in grid} == {None, 2.0, 1.5}
        assert [m for _, _, m in grid] == [True] * 9 + [False] * 9

    def test_baseline_params_and_seed_are_the_plan(self) -> None:
        assert BASE_PARAMS['call_delta'] == 0.25
        assert BASE_PARAMS['dte'] == 30
        assert BASE_PARAMS['close_at_pct'] == 0.75
        assert CC_R_SEED == 20260719

    def test_open_tail_entry(self) -> None:
        """Sell events carry 'pnl': 0 in the REAL engine schema
        (real_cc_backtest.py's sell_record) — the synthetic events must too,
        or a branch-order regression in open_tail_entry (testing 'pnl' before
        action == 'sell') would pass here while breaking on real data."""
        assert open_tail_entry([]) is None
        closed = [{'action': 'sell', 'date': 'a', 'pnl': 0},
                  {'action': 'close', 'date': 'b', 'pnl': 1.0}]
        assert open_tail_entry(closed) is None
        assert open_tail_entry(
            closed + [{'action': 'sell', 'date': 'c', 'pnl': 0}]) == 'c'

    def test_regime_spread_floor_and_unknown(self) -> None:
        cells = {
            'bull_quiet': {'expectancy_r': -0.5, 'meets_floor': True},
            'bear_volatile': {'expectancy_r': 0.3, 'meets_floor': True},
            'sideways_quiet': {'expectancy_r': 9.9, 'meets_floor': False},
            'unknown': {'expectancy_r': -9.9, 'meets_floor': True},
        }
        assert regime_spread(cells) == pytest.approx(0.8)
        assert regime_spread({'bull_quiet': cells['bull_quiet']}) == 0.0


class TestOverwriteRatioCurve:
    def test_blend_and_convergence(self) -> None:
        eq = pd.DataFrame({'date': DATES[:3],
                           'equity': [1000.0, 1050.0, 1080.0],
                           'price': [100.0, 110.0, 112.0]})
        shares, cash, capital = 10, 0.0, 1000.0
        # bh = [1000, 1100, 1120]; excess = [0, -50, -40]
        assert overlay_excess(eq, shares, cash) == pytest.approx([0.0, -50.0, -40.0])
        out = overwrite_ratio_curve(eq, shares, cash, capital,
                                    ratios=(1.0, 0.5, 0.0))
        assert out[1.0]['final_equity'] == pytest.approx(1080.0)
        assert out[0.5]['final_equity'] == pytest.approx(1100.0)
        assert out[0.0]['final_equity'] == pytest.approx(1120.0)   # buy-and-hold
        # rho=1 path [1000,1050,1080]: peak 1050 -> dd 0 at end... peak=1050,
        # trough at 1050? path rises after 1050 -> max_dd 0; rho=0 path rises
        # monotonically -> 0.
        assert out[0.0]['max_drawdown_pct'] == 0.0

    def test_drawdown_uses_capital_floor(self) -> None:
        eq = pd.DataFrame({'date': DATES[:2],
                           'equity': [900.0, 950.0],
                           'price': [100.0, 100.0]})
        out = overwrite_ratio_curve(eq, 10, 0.0, 1000.0, ratios=(1.0,))
        # peak floored at capital 1000 -> dd = (1000-900)/1000 = 10%
        assert out[1.0]['max_drawdown_pct'] == pytest.approx(10.0)


@pytest.mark.skipif(not _HAVE_SPY, reason='needs spy_option_dailies.csv or its .gz twin')
class TestSpyCcRExperiment:
    """The one measurement run of docs/spy_cc_r_experiment_plan.md, pinned.

    EXPLORATORY (Gap E precedent) — kill-or-justify, no idea-ledger rows, no
    e-value spent; the daily Newey-West t stays the sole significance
    authority. Headline verdicts against the plan's seven committed priors:

    - The decomposition closes exactly: Book U's -$124,197 net overlay ==
      Book H's -$1,401 minus the hedge's +$122,796 — the SPY covered call's
      loss is ~entirely the direction bill; the insurance book is breakeven
      over sixteen years.
    - SPY's missing risk-managed cell: daily NW t +0.41 (live) / +0.42
      (frozen arm) — joins MSFT -0.23 and QQQ +0.18 near zero. The
      exploratory +2.54 short-vol premium does NOT carry into the CC frame
      at the baseline exit (at hold it recovers only to +0.79; the rest is
      estimator/exit-cadence convention, itemized in the log entry).
    - Prior 2 CONTRADICTED in R, upheld in dollars: stops IMPROVE Book U's
      per-cycle expectancy (-0.30R -> -0.16R at 1.5x) while cycle count
      balloons 325 -> 573, dollars stay ~flat, and the trade-order t
      WORSENS (-3.49 -> -4.75) — the MSFT whipsaw verdict does not
      replicate per-cycle on SPY.
    - The §8 escalation bar FIRES on the hedged hold family — four cells
      after the deep-ITM widening (managed close1.0 +0.1186R at trade-t
      +2.26 and its 2x-stop twin at +2.57; unmanaged _noitm +0.1514R at
      +2.31 and its 2x-stop twin at +2.16) —
      right-signed, above the pre-committed bar, so they escalate to a
      human-signed registration proposal per the plan. The daily authority
      reads +0.79 / +1.08 there — NOT significant — and the cells are the
      known SPY call-wing premium family in CC clothes; escalation is a
      proposal gate, never a finding.
    - Prior 7 CONFIRMED: hedging flattens Tharp's regime structure (floor-
      cell spread 0.68R -> 0.24R), and Book U's death cell is bull_quiet
      (-0.617R on 169 cycles — the QQQ pattern replicated on SPY) while the
      hedge flips that same cell positive (+0.150R).

    Determinism note: the engine paths are deterministic and the sizing
    replay is seeded (CC_R_SEED), so these pins are exact up to float noise;
    tolerances are formatting-width, not uncertainty.
    """

    @pytest.fixture(scope='class')
    def live(self) -> dict:
        return run_experiment('live')

    @pytest.fixture(scope='class')
    def registered(self) -> dict:
        return run_experiment('registered')

    def test_spans(self, live, registered) -> None:
        assert live['span'] == ['2010-05-17', '2026-06-05']
        assert live['n_days'] == 4039
        assert registered['span'] == ['2010-12-01', '2026-06-05']
        assert registered['n_days'] == 3901

    def test_book_u_baseline(self, live) -> None:
        c = live['cells']['U:close0.75_stopNone']
        assert c['n'] == 325
        assert c['expectancy_r'] == pytest.approx(-0.3049, abs=0.001)
        assert c['win_rate'] == pytest.approx(65.2, abs=0.1)
        assert c['worst_r'] == pytest.approx(-6.582, abs=0.005)
        assert c['r_newey_west_t'] == pytest.approx(-3.488, abs=0.01)
        assert c['daily_nw_t'] == pytest.approx(-1.78, abs=0.01)
        assert c['net_overlay_pnl'] == pytest.approx(-124_196.82, abs=5.0)
        assert c['sqn'] == pytest.approx(-3.24, abs=0.01)
        assert c['mae_r']['median'] == pytest.approx(-0.8145, abs=0.001)

    def test_book_h_baseline(self, live) -> None:
        c = live['cells']['H:close0.75_stopNone']
        assert c['n'] == 325
        assert c['expectancy_r'] == pytest.approx(-0.0001, abs=0.001)
        assert c['win_rate'] == pytest.approx(51.4, abs=0.1)
        assert c['worst_r'] == pytest.approx(-2.3855, abs=0.005)
        assert c['daily_nw_t'] == pytest.approx(0.41, abs=0.01)
        assert c['net_overlay_pnl'] == pytest.approx(-1_400.94, abs=5.0)
        assert c['mae_r']['median'] == pytest.approx(-0.2282, abs=0.001)

    def test_decomposition_identity(self, live) -> None:
        """U's loss == H's ~zero minus the hedge P&L — the direction bill
        (plan §3's frozen R_U − R_H), closing to the dollar across
        independently-measured books. The bill is POSITIVE in 62% of cycles
        (the hedge pays buy-high/sell-low whipsaw in chop; the plain book
        wins small) and deeply negative in the rally cycles (the hedge is
        long-on-average — delta 0..1, never short — so it collects the drift
        that dwarfs the toll on a \\~6x span) — netting −0.30R per cycle.
        The hedge P&L is the bill from the repayment side, not a trading
        profit: it gains what the call's direction exposure loses."""
        u = live['cells']['U:close0.75_stopNone']['net_overlay_pnl']
        h = live['cells']['H:close0.75_stopNone']['net_overlay_pnl']
        p = live['paired']
        assert p['hedge_pnl_dollars'] == pytest.approx(122_795.85, abs=5.0)
        assert u == pytest.approx(h - p['hedge_pnl_dollars'], abs=1.0)
        assert p['mean_r'] == pytest.approx(-0.3048, abs=0.001)
        assert p['median_r'] == pytest.approx(0.5769, abs=0.001)
        assert p['share_positive'] == pytest.approx(0.6185, abs=0.002)

    def test_conservation_invariants(self, live) -> None:
        """Conservation from the unrounded window deltas: exact to the cent.
        gap_drift == 0.0 here records that the cycles TILE the live span
        (no outside-window days exist) — the flatness rail is exercised by
        the synthetic layer; the live rail on real data is the per-cycle
        Book-U identity asserted inside run_cell on every unhedged cell."""
        for key in ('U:close0.75_stopNone', 'H:close0.75_stopNone'):
            c = live['cells'][key]
            assert c['gap_drift'] == 0.0
            assert c['tail_residual'] == 0.0
            assert c['conservation_sum'] == pytest.approx(c['excess_final'], abs=0.01)

    def test_exit_grid_expectancies(self, live) -> None:
        expected_u = {
            'close1_stopNone': -0.2738, 'close1_stop2': -0.2031,
            'close1_stop1.5': -0.1551, 'close0.75_stopNone': -0.3049,
            'close0.75_stop2': -0.2131, 'close0.75_stop1.5': -0.1563,
            'close0.5_stopNone': -0.2960, 'close0.5_stop2': -0.1732,
            'close0.5_stop1.5': -0.1329,
        }
        expected_h = {
            'close1_stopNone': 0.1186, 'close1_stop2': 0.0961,
            'close1_stop1.5': 0.0468, 'close0.75_stopNone': -0.0001,
            'close0.75_stop2': -0.0196, 'close0.75_stop1.5': -0.0283,
            'close0.5_stopNone': -0.0175, 'close0.5_stop2': -0.0342,
            'close0.5_stop1.5': -0.0425,
        }
        for cell_key, exp in expected_u.items():
            assert live['cells'][f'U:{cell_key}']['expectancy_r'] == \
                pytest.approx(exp, abs=0.001), cell_key
        for cell_key, exp in expected_h.items():
            assert live['cells'][f'H:{cell_key}']['expectancy_r'] == \
                pytest.approx(exp, abs=0.001), cell_key
        # Every U cell negative (prior 1); stops IMPROVE U per-cycle R while
        # the trade-order t worsens (prior 2's contradiction, both halves).
        assert all(v < 0 for v in expected_u.values())
        assert live['cells']['U:close0.75_stop1.5']['r_newey_west_t'] < \
            live['cells']['U:close0.75_stopNone']['r_newey_west_t']

    def test_escalation_cells(self, live) -> None:
        """The original two above-bar cells (the _noitm twins joined them
        in the deep-ITM widening; see test_deep_itm_knob_book_h_and_the_ladder)
        — all escalate to ONE human-signed registration proposal, nothing
        more: the daily authority stays below 2 on every member."""
        hold = live['cells']['H:close1_stopNone']
        stop2 = live['cells']['H:close1_stop2']
        assert hold['expectancy_r'] == pytest.approx(0.1186, abs=0.001)
        assert hold['r_newey_west_t'] == pytest.approx(2.263, abs=0.01)
        assert stop2['expectancy_r'] == pytest.approx(0.0961, abs=0.001)
        assert stop2['r_newey_west_t'] == pytest.approx(2.568, abs=0.01)
        assert hold['daily_nw_t'] == pytest.approx(0.79, abs=0.01)
        assert stop2['daily_nw_t'] == pytest.approx(1.08, abs=0.01)
        assert hold['daily_nw_t'] < 2 and stop2['daily_nw_t'] < 2

    def test_tightest_target_owns_the_worst_tail(self, live) -> None:
        c = live['cells']['U:close0.5_stopNone']
        assert c['worst_r'] == pytest.approx(-15.8397, abs=0.01)

    def test_sizing_battery(self, live) -> None:
        s = live['sizing']
        assert s['U']['kelly'] == 0.0 and s['H']['kelly'] == 0.0
        assert s['U']['sweep'][0.01]['p_ruin'] == pytest.approx(0.9087, abs=0.001)
        assert s['U']['sweep'][0.02]['p_ruin'] == pytest.approx(0.9973, abs=0.001)
        assert s['H']['sweep'][0.02]['p_ruin'] == pytest.approx(0.0052, abs=0.001)
        assert s['H']['sweep'][0.02]['p_ruin_25dd'] == pytest.approx(0.2569, abs=0.001)

    def test_overwrite_dial(self, live) -> None:
        o = live['overwrite']
        assert o[1.0]['final_equity'] == pytest.approx(474_683.17, abs=5.0)
        assert o[0.5]['final_equity'] == pytest.approx(536_781.58, abs=5.0)
        assert o[0.25]['final_equity'] == pytest.approx(567_830.79, abs=5.0)
        # Monotone toward buy-and-hold: smaller overwrite, more final equity.
        assert o[1.0]['final_equity'] < o[0.5]['final_equity'] < o[0.25]['final_equity']
        # Book H's dial (plan §5 applies the battery to each book): H sits a
        # whisker under buy-and-hold at every ratio — nothing to dial.
        oh = live['overwrite_h']
        assert oh[1.0]['final_equity'] == pytest.approx(597_479.05, abs=5.0)
        assert oh[0.25]['final_equity'] == pytest.approx(598_529.76, abs=5.0)
        assert oh[1.0]['max_drawdown_pct'] == pytest.approx(37.50, abs=0.05)

    def test_regime_flattening(self, live) -> None:
        r = live['regime']
        assert r['U']['spread'] == pytest.approx(0.6807, abs=0.005)
        assert r['H']['spread'] == pytest.approx(0.2380, abs=0.005)
        assert r['H']['spread'] < r['U']['spread']          # prior 7
        u_bq = r['U']['cells']['bull_quiet']
        h_bq = r['H']['cells']['bull_quiet']
        assert u_bq['n'] == 169 and h_bq['n'] == 169
        assert u_bq['expectancy_r'] == pytest.approx(-0.6174, abs=0.001)
        assert h_bq['expectancy_r'] == pytest.approx(0.1501, abs=0.001)

    def test_deep_itm_knob_grid_shape(self, live, registered) -> None:
        """The 2026-07-19 widening: 18 cells per book live (the 3x3 crossed
        with manage_deep_itm), managed keys unchanged; the registered arm
        carries the baseline managed pair plus the true-hold _noitm pair."""
        assert len(live['cells']) == 36
        assert len(registered['cells']) == 4
        assert sum('_noitm' in k for k in live['cells']) == 18
        assert 'U:close1_stopNone_noitm' in registered['cells']

    def test_cross_engine_identity_at_true_hold(self, registered) -> None:
        """With BOTH exits off (no profit-take, no deep-ITM buyback), the CC
        engine's option cycles reproduce the structure engine's short-vol
        book on the same frozen span EXACTLY: n = 174 cycles at -0.5407R —
        the Gap E pinned baseline (TestSpyExitVariantExploration). Two
        independent engines, one book, once the conventions match: the
        +2.54-vs-CC-frame gap is proven to be wrapper, not measurement."""
        u = registered['cells']['U:close1_stopNone_noitm']
        assert u['n'] == 174
        assert u['expectancy_r'] == pytest.approx(-0.5407, abs=0.001)

    def test_deep_itm_knob_book_u(self, live) -> None:
        """Book U without the forced buyback: regime-dependent dollars (at
        hold it was PROTECTIVE — a de facto directional stop; at the 75%
        target its removal nets +$37K), but the tail always fattens (worst R
        -6.58 managed -> -10.98/-17.25/-23.62 across no-stop _noitm cells),
        and every one of the 18 U cells stays negative — prior 1 extends
        across the whole widened grid."""
        hold = live['cells']['U:close1_stopNone_noitm']
        assert hold['expectancy_r'] == pytest.approx(-0.5307, abs=0.001)
        assert hold['worst_r'] == pytest.approx(-10.9839, abs=0.01)
        assert hold['net_overlay_pnl'] == pytest.approx(-168_285.25, abs=5.0)
        base = live['cells']['U:close0.75_stopNone_noitm']
        assert base['net_overlay_pnl'] == pytest.approx(-87_280.77, abs=5.0)
        assert base['worst_r'] == pytest.approx(-17.2542, abs=0.01)
        assert live['cells']['U:close0.5_stopNone_noitm']['worst_r'] == \
            pytest.approx(-23.6176, abs=0.01)
        assert all(c['expectancy_r'] < 0
                   for k, c in live['cells'].items() if k.startswith('U'))

    def test_deep_itm_knob_book_h_and_the_ladder(self, live, registered) -> None:
        """The hedged book prefers NO deep-ITM close at hold (+$39.8K vs
        +$21.4K managed; the hedge already absorbs direction, so the forced
        buyback only paid spread and timing). This completes the exit-
        convention ladder toward the +2.54 short-vol convention: daily NW t
        +0.41 (baseline) -> +0.79 (hold, managed) -> +1.19 live / +1.25
        frozen (true hold) — still below 2 everywhere, with the residual gap
        to +2.54/+2.25 now attributable to yardstick (rf netting), the
        hedge fee, and uncredited hedge dividends. Trade-order t +2.31
        (live) / +2.08 (frozen) joins the above-bar §8 family — same single
        registration proposal, still junior-judge only."""
        hold = live['cells']['H:close1_stopNone_noitm']
        assert hold['expectancy_r'] == pytest.approx(0.1514, abs=0.001)
        assert hold['r_newey_west_t'] == pytest.approx(2.311, abs=0.01)
        assert hold['daily_nw_t'] == pytest.approx(1.19, abs=0.01)
        assert hold['net_overlay_pnl'] == pytest.approx(39_806.88, abs=5.0)
        reg = registered['cells']['H:close1_stopNone_noitm']
        assert reg['daily_nw_t'] == pytest.approx(1.25, abs=0.01)
        assert reg['r_newey_west_t'] == pytest.approx(2.077, abs=0.01)
        assert reg['expectancy_r'] == pytest.approx(0.1235, abs=0.001)
        assert hold['daily_nw_t'] < 2 and reg['daily_nw_t'] < 2

    def test_registered_arm(self, registered) -> None:
        u = registered['cells']['U:close0.75_stopNone']
        h = registered['cells']['H:close0.75_stopNone']
        assert u['n'] == 314 and h['n'] == 314
        assert u['expectancy_r'] == pytest.approx(-0.3107, abs=0.001)
        assert u['daily_nw_t'] == pytest.approx(-1.77, abs=0.01)
        assert h['expectancy_r'] == pytest.approx(-0.0023, abs=0.001)
        assert h['daily_nw_t'] == pytest.approx(0.42, abs=0.01)
        assert registered['paired']['hedge_pnl_dollars'] == \
            pytest.approx(122_721.94, abs=5.0)
        assert registered['paired']['mean_r'] == pytest.approx(-0.3084, abs=0.001)


_NVDA = data_path('nvda_option_dailies.csv')
_GLD = data_path('gld_option_dailies.csv')
_HAVE_CODA = all(os.path.exists(p) or os.path.exists(p + '.gz')
                 for p in (_NVDA, _GLD))


@pytest.mark.skipif(not _HAVE_CODA,
                    reason='needs nvda/gld option dailies (or .gz twins)')
class TestNvdaGldTrueHoldExploration:
    """The wing story's coda, pinned (2026-07-20, owner-directed): the
    true-hold delta-hedged call-selling book (close_at_pct 1.0, no stop,
    manage_deep_itm False — the SPY flagship's exact configuration) run on
    the two backwards-sign tickers from the wing diagnostic. The prediction
    the wing table invites — fattest calibration premium, best book — is
    demolished in opposite directions:

    - NVDA (+10.5 vol points of wing premium, the table's fattest):
      **bankruptcy**. Net overlay −$6.5M on $100K; final equity NEGATIVE
      (−$5.1M); max drawdown 185% — the account hit zero mid-span and the
      zero-interest, never-margin-called hedge financing let the simulation
      sail on through; the real book simply dies. Worst cycle −43.5R. The
      unit trap made flesh: the diagnostic prices the wing against
      UPSIDE-ONLY movement, the hedged seller pays TOTAL variance plus the
      overnight gaps a daily close-hedge cannot touch, and NVDA is the
      gappiest large stock alive. Both are true at once: the wing is
      overpriced as insurance AND ruinous to sell hedged.
    - GLD (+5.1 vol points): the best per-cycle hedged book in the
      program — +0.188R, 65.3% wins, trade-order t **+3.10** (the highest
      junior-judge score measured) — and the daily authority reads +0.43.
      The third member of the tidy-endpoints/noisy-journey family, with
      every usual asterisk: one ticker, spent data, outside any
      pre-committed grid, so no escalation machinery applies.

    EXPLORATORY, chat-level run formalized — kill-or-justify, no idea-ledger
    rows, no e-value; the daily NW t stays the authority. Any GLD follow-up
    is a NEW registration, never a revival of this pin. CI note: two more
    store loads in the cc-r bucket (~3-5 min cold).
    """

    @pytest.fixture(scope='class')
    def books(self) -> dict:
        from common.trade_ledger import ledger_statistics
        from engine.cc_backtest import compute_statistics
        from realchains.cc_r_experiment import (
            attribute_cycles, open_tail_entry, overlay_excess)
        from realchains.real_cc_backtest import (
            CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices,
            run_real_cc_overlay)
        from common.trade_ledger import build_trade_ledger
        out = {}
        for t in ('NVDA', 'GLD'):
            store = load_chain_store(data_path(f'{t.lower()}_option_dailies.csv'),
                                     start=CHAIN_CLEAN_START.get(t))
            days = sorted(store)
            pd_, px = load_unadjusted_prices(t, days[0], '2026-06-06')
            pairs = [(d, p) for d, p in zip(pd_, px) if days[0] <= d <= days[-1]]
            dates = [d for d, _ in pairs]
            prices = [p for _, p in pairs]
            summary, trades, eq = run_real_cc_overlay(
                dates, prices, store,
                {'call_delta': 0.25, 'dte': 30, 'close_at_pct': 1.0,
                 'capital': 100_000, 'delta_hedge': True,
                 'manage_deep_itm': False})
            shares = summary['num_contracts'] * 100
            native = build_trade_ledger(trades, strategy='covered_call',
                                        ticker=t, shares=shares,
                                        risk_basis='premium_collected')
            excess = overlay_excess(eq, shares, summary['cash'])
            adj, gap, tail, raw = attribute_cycles(
                native, list(eq['date']), excess,
                open_entry_date=open_tail_entry(trades))
            st = ledger_statistics(adj)
            daily = compute_statistics(eq, num_contracts=summary['num_contracts'],
                                       cash=summary['cash'])
            out[t] = {'summary': summary, 'ledger': st, 'records': adj,
                      'daily_t': daily['t_stat_newey_west'],
                      'gap_drift': gap, 'conservation': raw + tail,
                      'excess_final': excess[-1]}
            del store
        return out

    def test_nvda_bankruptcy(self, books) -> None:
        n = books['NVDA']
        assert n['ledger']['n'] == 166
        assert n['ledger']['expectancy_r'] == pytest.approx(-0.2907, abs=0.002)
        assert min(r.r_multiple for r in n['records']) == \
            pytest.approx(-43.5378, abs=0.05)
        assert n['summary']['net_overlay_pnl'] == pytest.approx(-6_505_498.38,
                                                                abs=100.0)
        assert n['summary']['final_equity'] < 0          # the account DIED
        assert n['summary']['max_drawdown_pct'] > 100    # ruin mid-span
        assert n['daily_t'] == pytest.approx(-1.28, abs=0.02)

    def test_gld_best_junior_worst_senior(self, books) -> None:
        g = books['GLD']
        assert g['ledger']['n'] == 167
        assert g['ledger']['expectancy_r'] == pytest.approx(0.1879, abs=0.002)
        assert g['ledger']['win_rate'] == pytest.approx(65.3, abs=0.1)
        assert g['ledger']['r_newey_west_t'] == pytest.approx(3.096, abs=0.02)
        assert g['daily_t'] == pytest.approx(0.43, abs=0.02)
        assert g['daily_t'] < 2                          # the authority shrugs
        assert g['summary']['net_overlay_pnl'] == pytest.approx(20_219.34,
                                                                abs=5.0)

    def test_conservation_holds_even_through_ruin(self, books) -> None:
        """The attribution identity survives a book that goes bust: cycle
        sums still telescope to the final excess to the cent."""
        for t in ('NVDA', 'GLD'):
            b = books[t]
            assert b['conservation'] == pytest.approx(b['excess_final'],
                                                      abs=0.05)
            assert b['gap_drift'] < 0.02
