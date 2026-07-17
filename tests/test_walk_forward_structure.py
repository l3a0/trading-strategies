"""Always-run synthetic tests for the put-credit-spread walk-forward driver.

These pin the MACHINERY of docs/prereg_put_credit_spread.md (sections 3, 5)
per docs/put_spread_analysis_plan.md — lattice enumeration, selection rule,
stitching/seam accounting, the dte21 guard, the driver-side hedge override
(plan D1, asserted byte-identical to run_real_credit_spread_overlay), the
arm-E jitter selector, and the section-7.3 companions. Every assertion is
hand-derived on synthetic stores; NO dataset-gated test lives here — result
pins belong to the results PR, and no real-data run may precede the
analysis-code merge (prereg section 10 ordering).
"""
from __future__ import annotations

import math
import random
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest

from common.stats import newey_west_summary
from realchains.real_cc_backtest import COMMISSION_PER_SHARE
from realchains.vol_premium import STRUCTURE_SPECS, run_real_structure_overlay
from realchains.walk_forward_structure import (
    CENTRAL_CELL,
    EXIT_VARIANTS,
    Cell,
    day0_omission,
    enumerate_joint_cells,
    excess_stream,
    jitter_select_factory,
    loyo_nw,
    replay_records,
    run_cell,
    seam_charge,
    sharpe_unrounded,
    stationary_bootstrap,
    stitch_records,
    verdict_stats,
    walk_forward_structure,
)


# --- synthetic fixtures ------------------------------------------------------

def _spread_market() -> tuple[list[str], list[float], dict[str, Any]]:
    """A one-cycle two-put market the real selector can trade: entry day
    candidates at -0.25/-0.20 deltas, mid-cycle marks, expiry-day settle."""
    dates = ['2020-01-02', '2020-01-03', '2020-01-06', '2020-02-21']
    prices = [100.0, 101.0, 100.0, 102.0]
    exp = '2020-02-21'
    short_cand = (50, -0.25, 1.0, 1.2, 1.1, exp, 100.0, 'P100')
    wing_cand = (50, -0.20, 0.5, 0.7, 0.6, exp, 95.0, 'P95')
    marks_mid = {'P100': (0.9, 1.1, 1.0, -0.24), 'P95': (0.4, 0.6, 0.5, -0.19)}
    store = {
        '2020-01-02': {'candidates': [short_cand, wing_cand], 'marks': dict(marks_mid)},
        '2020-01-03': {'candidates': [], 'marks': dict(marks_mid)},
        '2020-01-06': {'candidates': [], 'marks': dict(marks_mid)},
        '2020-02-21': {'candidates': [], 'marks': dict(marks_mid)},
    }
    return dates, prices, store


def _stub_run_cell(specs: dict[str, tuple[int, float, float]]):
    """Injectable engine stub (the driver's documented test seam): per cell
    key, (entry_count, mean daily P&L, wobble) — equity alternates
    mean+wobble / mean-wobble so the unrounded Sharpe is mean/wobble-driven."""

    def stub(
        dates: list[str], prices: list[float], store: dict, cell: Cell,
        *, hedged: bool = True, select: Optional[Any] = None,
        extra_params: Optional[dict] = None,
    ):
        spec = specs[cell.key()]
        if spec == 'raise-value':
            raise ValueError('capital insufficient for one contract')
        if spec == 'raise-runtime':
            raise RuntimeError('genuine engine bug')
        n, mean, wobble = spec
        eq = [100_000.0]
        for i in range(1, len(dates)):
            eq.append(eq[-1] + mean + (wobble if i % 2 else -wobble))
        df = pd.DataFrame({
            'date': dates, 'equity': eq, 'price': prices,
            'rf_credit': [0.0] * len(dates),
        })
        summary = {'capital': 100_000.0, 'num_contracts': 1,
                   'num_credit_spreads_sold': n}
        return summary, [], df

    return stub


def _bdates(start: str, end: str) -> list[str]:
    return [d.strftime('%Y-%m-%d') for d in pd.bdate_range(start, end)]


# --- the lattice -------------------------------------------------------------

class TestPutSpreadLattice:
    def test_sixty_nine_cells_in_frozen_order(self):
        cells = enumerate_joint_cells()
        assert len(cells) == 69  # 9 x 8 - 3 (Amendment 1)
        assert cells[0] == Cell(21, 0.20, 0.15, 'hold')
        # 21-DTE entries carry 7 exits (dte21 excluded), others all 8
        per_entry: dict[tuple, int] = {}
        for c in cells:
            per_entry[(c.dte, c.short_delta)] = per_entry.get(
                (c.dte, c.short_delta), 0) + 1
        assert all(
            n == (7 if dte == 21 else 8) for (dte, _), n in per_entry.items()
        )
        assert len(per_entry) == 9

    def test_wing_is_derived(self):
        for c in enumerate_joint_cells():
            assert c.wing_delta == pytest.approx(c.short_delta - 0.05)

    def test_exit_variants_frozen(self):
        names = [n for n, _ in EXIT_VARIANTS]
        assert names == ['hold', 'target50', 'target75', 'stop2x', 'stop3x',
                         'dte21', 'bracket', 'bracket75']
        params = dict(EXIT_VARIANTS)
        assert params['bracket75'] == {'close_at_pct': 0.75,
                                       'stop_loss_mult': 1.5}  # Amendment 1
        assert params['bracket'] == {'close_at_pct': 0.50,
                                     'stop_loss_mult': 2.0}

    def test_cell_params(self):
        p = Cell(30, 0.25, 0.20, 'bracket75').params()
        assert p['dte'] == 30 and p['short_delta'] == 0.25
        assert p['wing_delta'] == 0.20 and p['capital'] == 100_000
        assert p['risk_free_rate'] == 0.045
        assert p['close_at_pct'] == 0.75 and p['stop_loss_mult'] == 1.5
        assert 'close_at_pct' not in Cell(30, 0.25, 0.20, 'hold').params()
        assert CENTRAL_CELL == Cell(30, 0.25, 0.20, 'hold')


# --- selection ---------------------------------------------------------------

class TestSpreadWfSelection:
    DATES = _bdates('2020-01-01', '2022-06-30')
    PRICES = [100.0] * len(DATES)
    CELLS = [Cell(30, 0.25, 0.20, 'hold'), Cell(30, 0.30, 0.25, 'hold'),
             Cell(45, 0.25, 0.20, 'hold')]

    def _run(self, specs):
        return walk_forward_structure(
            self.DATES, self.PRICES, {}, cells=self.CELLS,
            train_years=1, test_months=3, roll_months=3, min_trades=30,
            run_cell_fn=_stub_run_cell(specs),
        )

    def test_highest_sharpe_wins(self):
        recs = self._run({
            self.CELLS[0].key(): (40, 5.0, 50.0),   # sharpe ~ 1.59
            self.CELLS[1].key(): (40, 10.0, 20.0),  # sharpe ~ 7.9 — wins
            self.CELLS[2].key(): (40, 1.0, 100.0),
        })
        assert len(recs) == 5
        assert all(r['winner'] == self.CELLS[1] for r in recs)
        assert all(r['n_trades'] == 40 for r in recs)

    def test_floor_disqualifies_on_entry_count(self):
        recs = self._run({
            self.CELLS[0].key(): (29, 10.0, 1.0),   # best Sharpe, under floor
            self.CELLS[1].key(): (30, 5.0, 50.0),   # qualifies — wins
            self.CELLS[2].key(): (31, 1.0, 100.0),
        })
        assert all(r['winner'] == self.CELLS[1] for r in recs)
        assert all(r['n_below_30'] == 1 for r in recs)
        assert all(r['min_grid_trades'] == 29 for r in recs)

    def test_tie_breaks_by_lattice_order(self):
        recs = self._run({
            self.CELLS[0].key(): (40, 5.0, 25.0),
            self.CELLS[1].key(): (40, 5.0, 25.0),   # identical — earlier wins
            self.CELLS[2].key(): (40, 1.0, 100.0),
        })
        assert all(r['winner'] == self.CELLS[0] for r in recs)

    def test_no_winner_window_trades_nothing(self):
        recs = self._run({c.key(): (10, 5.0, 25.0) for c in self.CELLS})
        assert all(r['winner'] is None for r in recs)
        for r in recs:
            assert np.all(r['oos_excess'] == 0.0)
            assert len(r['oos_excess']) == len(r['oos_dates'])
            assert r['deployed_notional'] == 0.0
            assert r['n_below_30'] == len(self.CELLS)

    def test_valueerror_cell_is_recorded_skip_not_silent(self):
        recs = self._run({
            self.CELLS[0].key(): 'raise-value',
            self.CELLS[1].key(): (40, 5.0, 50.0),
            self.CELLS[2].key(): (40, 1.0, 100.0),
        })
        assert all(r['winner'] == self.CELLS[1] for r in recs)
        for r in recs:
            assert len(r['failed_cells']) == 1
            assert self.CELLS[0].key() in r['failed_cells'][0]

    def test_unknown_exception_propagates(self):
        with pytest.raises(RuntimeError, match='genuine engine bug'):
            self._run({
                self.CELLS[0].key(): 'raise-runtime',
                self.CELLS[1].key(): (40, 5.0, 50.0),
                self.CELLS[2].key(): (40, 1.0, 100.0),
            })

    def test_selection_metric_is_unrounded_sharpe(self):
        excess = np.array([0.001, -0.0005, 0.001, -0.0005])
        expected = (np.mean(excess) / np.std(excess, ddof=1)) * math.sqrt(252)
        assert sharpe_unrounded(excess) == pytest.approx(expected, rel=1e-12)
        assert sharpe_unrounded(np.zeros(0)) == -math.inf
        assert sharpe_unrounded(np.zeros(5)) == 0.0


# --- stitching and seam accounting -------------------------------------------

class TestSpreadWfStitching:
    LEGS = [
        {'sign': -1, 'right': 'put', 'strike': 100.0, 'entry_net': 0.9935,
         'expiration': '2020-02-01'},
        {'sign': 1, 'right': 'put', 'strike': 95.0, 'entry_net': 0.7065,
         'expiration': '2020-02-01'},
    ]
    STORE = {
        '2020-01-02': {
            'candidates': [
                (30, -0.25, 1.0, 1.2, 1.1, '2020-02-01', 100.0, 'A'),
                (30, -0.20, 0.5, 0.7, 0.6, '2020-02-01', 95.0, 'B'),
            ],
            'marks': {},
        },
    }

    def test_seam_charge_hand_derived(self):
        trades = [{'action': 'enter', 'date': '2020-01-02',
                   'legs_detail': self.LEGS}]
        # short buys back at ask: (1.2-1.1)+c; long sells at bid: (0.6-0.5)+c
        expected = ((0.1 + COMMISSION_PER_SHARE) * 2) * 1000
        got = seam_charge(self.STORE, ['2020-01-02', '2020-01-03'],
                          trades, 1000)
        assert got == pytest.approx(expected)  # 2020-01-03 unquoted: walk back

    def test_settle_leg_fails_loudly(self):
        trades = [
            {'action': 'enter', 'date': '2020-01-02', 'legs_detail': self.LEGS},
            {'action': 'settle_leg', 'date': '2020-01-15', 'pnl': 0.0},
        ]
        with pytest.raises(ValueError, match='single-expiration'):
            seam_charge(self.STORE, ['2020-01-02'], trades, 1000)

    def test_seam_charge_zero_when_settled(self):
        trades = [
            {'action': 'enter', 'date': '2020-01-02', 'legs_detail': self.LEGS},
            {'action': 'settle', 'date': '2020-02-01', 'pnl': 0.0, 'mae': 0.0},
        ]
        assert seam_charge(self.STORE, ['2020-01-02'], trades, 1000) == 0.0

    def test_day0_omission_only_on_first_day_entry(self):
        trades = [{'action': 'enter', 'date': '2020-01-02',
                   'legs_detail': self.LEGS}]
        # short sold at bid, marked mid: (1.1-1.0)+c; long at ask: (0.7-0.6)+c
        expected = ((0.1 + COMMISSION_PER_SHARE) * 2) * 1000
        assert day0_omission(self.STORE, ['2020-01-02', '2020-01-03'],
                             trades, 1000) == pytest.approx(expected)
        assert day0_omission(self.STORE, ['2020-01-01', '2020-01-02'],
                             trades, 1000) == 0.0

    def test_stitch_concatenates_in_window_order(self):
        recs = [
            {'oos_excess': np.array([0.1, 0.2]), 'oos_dates': ['a', 'b']},
            {'oos_excess': np.zeros(3), 'oos_dates': ['c', 'd', 'e']},
        ]
        stitched, dates = stitch_records(recs)
        assert list(stitched) == pytest.approx([0.1, 0.2, 0.0, 0.0, 0.0])
        assert dates == ['a', 'b', 'c', 'd', 'e']

    def test_verdict_stats_p_convention(self):
        stats = verdict_stats(np.array([0.0, 0.0, 0.0, 0.0]))
        assert stats['t_newey_west'] == 0.0
        assert stats['one_sided_p'] == pytest.approx(0.5)


# --- the dte21 guard ---------------------------------------------------------

class TestDte21Guard:
    def _day(self, short_dte: int) -> dict[str, Any]:
        exp = '2020-02-01'
        return {
            'candidates': [
                (short_dte, -0.25, 1.0, 1.2, 1.1, exp, 100.0, 'P100'),
                (short_dte, -0.20, 0.5, 0.7, 0.6, exp, 95.0, 'P95'),
            ],
            'marks': {},
        }

    def test_skips_actual_dte_at_or_below_22(self):
        from realchains.walk_forward_structure import _cell_select
        cell = Cell(30, 0.25, 0.20, 'dte21')
        sel = _cell_select(cell)
        assert sel(self._day(22), cell.params()) is None
        assert sel(self._day(20), cell.params()) is None

    def test_delegates_byte_identically_above_22(self):
        from realchains.walk_forward_structure import _cell_select
        cell = Cell(30, 0.25, 0.20, 'dte21')
        base = STRUCTURE_SPECS['credit_spread']['select']
        day = self._day(23)
        assert _cell_select(cell)(day, cell.params()) == base(day, cell.params())

    def test_other_exits_use_spec_selector(self):
        from realchains.walk_forward_structure import _cell_select
        assert (_cell_select(Cell(30, 0.25, 0.20, 'hold'))
                is STRUCTURE_SPECS['credit_spread']['select'])


# --- the driver-side hedge override (plan D1) --------------------------------

class TestHedgeOverrideEquivalence:
    def test_hedged_path_matches_committed_delegate(self):
        from realchains.vol_premium import run_real_credit_spread_overlay
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        s1, t1, eq1 = run_cell(dates, prices, store, cell, hedged=True)
        s2, t2, eq2 = run_real_credit_spread_overlay(
            dates, prices, store, cell.params())
        assert s1 == s2
        assert t1 == t2
        pd.testing.assert_frame_equal(eq1, eq2)
        assert s1['num_credit_spreads_sold'] == 1  # the market actually trades

    def test_unhedged_path_is_engine_hedge_mode_none(self):
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        s1, t1, eq1 = run_cell(dates, prices, store, cell, hedged=False)
        spec = STRUCTURE_SPECS['credit_spread']
        merged = {**spec['defaults'], **cell.params()}
        q, t2, eq2 = run_real_structure_overlay(
            dates, prices, store, merged, select=spec['select'],
            entry_guard=spec['entry_guard'], hedge_mode='none',
            management=spec['management'])
        s2 = spec['summary'](q, merged)
        assert s1 == s2
        assert t1 == t2
        pd.testing.assert_frame_equal(eq1, eq2)
        assert s1['total_hedge_cost'] == 0.0  # no stock ever trades

    def test_hedged_and_unhedged_diverge(self):
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        _, _, eq_h = run_cell(dates, prices, store, cell, hedged=True)
        _, _, eq_u = run_cell(dates, prices, store, cell, hedged=False)
        assert not eq_h['equity'].equals(eq_u['equity'])

    def test_extra_params_merge_last_cost_curve_seam(self):
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        s_default, _, _ = run_cell(dates, prices, store, cell, hedged=True)
        s_free, _, _ = run_cell(dates, prices, store, cell, hedged=True,
                                extra_params={'hedge_cost_bps': 0.0})
        assert s_default['hedge_cost_bps'] == 0.5  # the spec default held
        assert s_free['hedge_cost_bps'] == 0.0     # the override won
        assert s_free['total_hedge_cost'] == 0.0
        # None leaves the default path byte-identical
        s_none, _, _ = run_cell(dates, prices, store, cell, hedged=True,
                                extra_params=None)
        assert s_none == s_default


# --- arm E jitter mechanics --------------------------------------------------

class TestSpreadJitterMechanics:
    def test_k0_delegates_immediately(self):
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        factory = jitter_select_factory(random.Random(1), k=0)
        sel = factory(cell)
        day = store['2020-01-02']
        base = STRUCTURE_SPECS['credit_spread']['select']
        assert sel(day, cell.params()) == base(day, cell.params())

    def test_wait_counts_emission_keyed(self):
        cell = Cell(30, 0.25, 0.20, 'hold')
        seed = 7
        j = random.Random(seed).randint(0, 10)
        factory = jitter_select_factory(random.Random(seed), k=10)
        sel = factory(cell)
        day = _spread_market()[2]['2020-01-02']
        for _ in range(j):
            assert sel(day, cell.params()) is None  # waiting
        picked = sel(day, cell.params())
        assert picked is not None  # delegates after J chain-days
        # emission re-arms: the very next flat call draws a fresh J
        rng_tail = random.Random(seed)
        rng_tail.randint(0, 10)
        j2 = rng_tail.randint(0, 10)
        if j2 > 0:
            assert sel(day, cell.params()) is None

    def test_same_seed_same_career(self):
        dates, prices, store = _spread_market()
        cell = Cell(30, 0.25, 0.20, 'hold')
        outs = []
        for _ in range(2):
            factory = jitter_select_factory(random.Random(20260717), k=3)
            sel = factory(cell)
            outs.append([sel(store['2020-01-02'], cell.params()) is None
                         for _ in range(5)])
        assert outs[0] == outs[1]

    def test_replay_alignment_guard(self):
        dates = _bdates('2020-01-01', '2021-12-31')
        prices = [100.0] * len(dates)
        cell = Cell(30, 0.25, 0.20, 'hold')
        stub = _stub_run_cell({cell.key(): (40, 5.0, 25.0)})
        recs = walk_forward_structure(
            dates, prices, {}, cells=[cell], train_years=1, test_months=3,
            roll_months=3, run_cell_fn=stub)
        replayed = replay_records(
            recs, dates, prices, {}, hedged=False, run_cell_fn=stub,
            train_years=1, test_months=3, roll_months=3)
        assert len(replayed) == len(recs)
        assert all(r['winner'] == cell for r in replayed)
        with pytest.raises(ValueError):
            replay_records(recs[:-1], dates, prices, {}, hedged=False,
                           run_cell_fn=stub, train_years=1, test_months=3,
                           roll_months=3)


# --- section-7.3 companions --------------------------------------------------

class TestStationaryBootstrap:
    def test_deterministic_under_seed(self):
        x = np.sin(np.arange(60)) * 0.01 + 0.002
        a = stationary_bootstrap(x, block=5, n_boot=200, seed=42)
        b = stationary_bootstrap(x, block=5, n_boot=200, seed=42)
        assert a == b

    def test_add_one_p_bounds(self):
        up = stationary_bootstrap(np.full(50, 0.001), block=5, n_boot=200,
                                  seed=1)
        assert up['p_boot'] == pytest.approx(1 / 201)  # never <= 0
        centered = stationary_bootstrap(
            np.tile([0.001, -0.001], 30), block=2, n_boot=400, seed=2)
        assert 0.2 < centered['p_boot'] < 0.8

    def test_degenerate_series(self):
        assert stationary_bootstrap(np.zeros(1), n_boot=10)['p_boot'] == 1.0


class TestLoyoStream:
    def test_hand_computed_year_drop(self):
        dates = (['2020-01-%02d' % d for d in range(1, 11)]
                 + ['2021-01-%02d' % d for d in range(1, 11)])
        x = np.array([0.01] * 10 + [-0.01] * 10)
        out = loyo_nw(x, dates)
        assert set(out) == {'2020', '2021'}
        assert out['2020'] == pytest.approx(
            newey_west_summary(x[10:]).t_newey_west)
        assert out['2021'] == pytest.approx(
            newey_west_summary(x[:10]).t_newey_west)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            loyo_nw(np.zeros(3), ['2020-01-01'])

    def test_excess_stream_rf_netting(self):
        eq = pd.DataFrame({
            'date': ['d0', 'd1', 'd2'],
            'equity': [100_000.0, 100_100.0, 100_150.0],
            'price': [100.0, 100.0, 100.0],
            'rf_credit': [0.0, 10.0, 10.0],
        })
        got = excess_stream(eq, 100_000.0)
        assert got == pytest.approx([(100 - 10) / 100_000,
                                     (50 - 10) / 100_000])
