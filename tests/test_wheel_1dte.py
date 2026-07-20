"""Tests for realchains/wheel_1dte.py — the frozen QQQ 1-DTE wheel build.

Always-run synthetic layer (plan §13): state transitions, the basis rule
binding and inert, gate application with fallback, settlement order and tie
handling, the affordability clamp, and conservation. The dataset-gated
``TestQqqWheel1dteExploration`` pins the run's decisive numbers and requires
both chain stores AND the local intraday archive (skips in CI until that
archive is published — a separate, human-gated decision).
"""

from __future__ import annotations

import os

import pytest

from common.paths import data_path
from realchains.wheel_1dte import (
    CAPITAL,
    FEE_PER_CONTRACT,
    PRIMARY_CELL,
    build_wheel_index,
    decomposition_companion,
    load_gate_signals,
    overnight_summary,
    rotation_summary,
    run_wheel,
    sizing_battery,
)

# ---------------------------------------------------------------- synthetic

# Consecutive real weekdays (Tue 2024-01-02 .. Thu 2024-01-11) so the
# calendar-day interest/weekend arithmetic is exercised honestly.
D = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08',
     '2024-01-09', '2024-01-10', '2024-01-11']


def up_signals(dates):
    return {d: {'up_355': True, 'up_cc': True, 'fallback': False, 'disagree': False}
            for d in dates}


def idx(dates, **by_day):
    """{date: {'puts': [...], 'calls': [...]}} for dates[1:-1]; rows are
    (delta, strike, bid, mid). by_day keys are i1/i2/... = dates[1]/dates[2]/…"""
    out = {}
    for i, d in enumerate(dates[1:-1], start=1):
        out[d] = by_day.get(f'i{i}', {'puts': [], 'calls': []})
    return out


def cell(**over):
    return {**PRIMARY_CELL, **over}


class TestWorthlessPut:
    def test_premium_kept_and_conserved(self):
        dates, closes = D[:4], [400.0, 400.0, 401.0, 402.0]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.1)], 'calls': []})
        run = run_wheel(dates, closes, market, up_signals(dates), **cell())
        # entry night: equity dips by exactly the fee
        assert run['daily'][0]['equity'] == pytest.approx(CAPITAL - FEE_PER_CONTRACT)
        # expiry: full premium kept (mid fill), fee netted
        assert run['summary']['final_equity'] == pytest.approx(CAPITAL + 110.0 - 0.65)
        assert run['totals']['expired_puts'] == 1 and run['totals']['assignments'] == 0
        # per-overnight R: kept-whole-premium minus the fee
        (rec,) = run['records']
        assert rec.r_multiple == pytest.approx((110.0 - 0.65) / 110.0, abs=1e-4)
        assert run['record_sides'] == ['put']

    def test_bid_fill_floor(self):
        dates, closes = D[:4], [400.0, 400.0, 401.0, 402.0]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.1)], 'calls': []})
        run = run_wheel(dates, closes, market, up_signals(dates), **cell(), fill='bid')
        assert run['summary']['final_equity'] == pytest.approx(CAPITAL + 100.0 - 0.65)


class TestSettlementTies:
    def test_put_assigned_exactly_when_below(self):
        dates = D[:4]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        # close == strike: expires worthless (the frozen tie rule)
        run = run_wheel(dates, [400.0, 400.0, 396.0, 400.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['assignments'] == 0
        # one cent below: assigned
        run = run_wheel(dates, [400.0, 400.0, 395.99, 400.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['assignments'] == 1
        assert run['open_rotation'] is not None      # still holding at window end

    def test_call_away_exactly_when_above(self):
        dates = D[:5]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 398.0, 0.8, 0.8)]},
        )
        # D3 close == call strike: not called away
        run = run_wheel(dates, [400.0, 400.0, 390.0, 398.0, 398.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['call_aways'] == 0 and run['open_rotation'] is not None
        # strictly above: called away, rotation closes at the strike
        run = run_wheel(dates, [400.0, 400.0, 390.0, 399.0, 399.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['call_aways'] == 1 and run['open_rotation'] is None
        (rot,) = run['rotations']
        assert rot['exit_price'] == 398.0 and rot['exit_reason'] == 'called_away'


class TestRotationLedger:
    def _rally_run(self):
        """Assigned at 396, one call at 398, called away while QQQ rockets:
        raw column green, gap column red — the §7 disagreement case."""
        dates = D[:4]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 398.0, 0.8, 0.8)]},
        )
        return run_wheel(dates, [400.0, 400.0, 390.0, 420.0], market,
                         up_signals(dates), **cell())

    def test_raw_column_practitioner_accounting(self):
        run = self._rally_run()
        (rot,) = run['rotations']
        # put premium (100 - .65) + call premium (80 - .65) + (398-396)*100
        assert rot['raw_pnl'] == pytest.approx(99.35 + 79.35 + 200.0)
        assert rot['underwater_days'] == 1 and rot['holding_days'] == 1
        assert run['summary']['final_equity'] == pytest.approx(CAPITAL + 378.70)

    def test_gap_column_disagrees(self):
        run = self._rally_run()
        (rot,) = run['rotations']
        # comparator: 100 shares from 400 -> 420 (+$2000); wheel made $378.70
        assert rot['gap_pnl'] == pytest.approx(378.70 - 2000.0)
        s = rotation_summary(run['rotations'])
        assert s['raw_win_rate'] == 100.0 and s['gap_win_rate'] == 0.0
        assert s['rescue_share'] == 100.0

    def test_decomposition_companion_hand_derived(self):
        run = self._rally_run()
        dates, closes = D[:4], [400.0, 400.0, 390.0, 420.0]
        comp = decomposition_companion(run, dates, closes)
        # put night: short -0.20d put -> +20 share-equivalents; hedge shorts 20
        # shares over 400 -> 390: +$200. call night: stock 100 + short 0.20d
        # call -> +80; hedge shorts 80 over 390 -> 420: -$2400.
        assert comp['unhedged_total'] == pytest.approx(378.70)
        assert comp['hedged_total'] == pytest.approx(378.70 + 200.0 - 2400.0)
        assert comp['direction_bill'] == pytest.approx(2200.0)

    def test_conservation_across_mixed_path(self):
        # worthless put, then an assigned rotation held to window end:
        # the engine's internal conservation assert must hold and the final
        # equity must equal the hand-built sum.
        dates = D[:5]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [(-0.20, 394.0, 1.2, 1.2)], 'calls': []},
        )
        run = run_wheel(dates, [400.0, 400.0, 398.0, 380.0, 385.0], market,
                        up_signals(dates), **cell())
        # D2: put expires (+99.35); D2 night: sell 394 put; D3: assigned at 394
        # (close 380); D4: no call available; window ends holding.
        hand = 99.35 + 119.35 + (385.0 - 394.0) * 100
        assert run['summary']['final_equity'] - CAPITAL == pytest.approx(hand)
        assert run['open_rotation']['raw_pnl_marked'] == pytest.approx(119.35 - 900.0)


class TestTwoRotations:
    """The wheel actually wheeling: called away, re-entered, assigned again,
    called away — pinning the second rotation's gap-column base (the
    base_i >= 0 branch) and the two-ledger split on one path."""

    def _run(self):
        dates = D
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 398.0, 0.8, 0.8)]},
            i4={'puts': [(-0.20, 394.0, 1.2, 1.2)], 'calls': []},
            i6={'puts': [], 'calls': [(0.20, 396.0, 0.5, 0.5)]},
        )
        closes = [400.0, 400.0, 390.0, 399.0, 398.0, 380.0, 385.0, 397.0]
        return run_wheel(dates, closes, market, up_signals(dates), **cell())

    def test_two_rotations_close_with_hand_derived_columns(self):
        run = self._run()
        r1, r2 = run['rotations']
        assert (r1['raw_pnl'], r2['raw_pnl']) == (pytest.approx(378.70),
                                                  pytest.approx(368.70))
        # gap base for rotation 2 sits at the entry close (plus the entry
        # fee), NOT the pre-entry session: hand-derived from both books
        assert r1['gap_pnl'] == pytest.approx(478.70)
        assert r2['gap_pnl'] == pytest.approx(468.70)
        assert run['summary']['final_equity'] == pytest.approx(100747.40)

    def test_two_ledgers_disagree_by_construction(self):
        # every overnight trade on this path LOSES (assignments and
        # called-away intrinsic exceed each premium), yet every rotation
        # ends green — the §7 split in one fixture
        run = self._run()
        ov = overnight_summary(run['records'], run['record_sides'])
        assert ov['puts']['n'] == 2 and ov['puts']['win_rate'] == 0.0
        assert ov['calls']['n'] == 2 and ov['calls']['win_rate'] == 0.0
        s = rotation_summary(run['rotations'])
        assert s['n'] == 2 and s['raw_win_rate'] == 100.0


class TestBasisRule:
    def _held(self, calls, basis_rule=True, closes=None):
        dates = D[:5]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': calls},
        )
        return run_wheel(dates, closes or [400.0, 400.0, 390.0, 395.0, 395.0],
                         market, up_signals(dates), **cell(basis_rule=basis_rule))

    def test_binding_blocks_below_basis(self):
        run = self._held([(0.20, 390.0, 0.9, 0.9)])       # only strike < basis 396
        assert run['totals']['call_sales'] == 0
        assert run['diag']['basis_rule_binding_days'] == 1

    def test_ablation_sells_and_realizes_loss(self):
        run = self._held([(0.20, 390.0, 0.9, 0.9)], basis_rule=False)
        assert run['totals']['call_sales'] == 1 and run['totals']['call_aways'] == 1
        (rot,) = run['rotations']
        assert rot['share_pnl'] == pytest.approx((390.0 - 396.0) * 100)

    def test_inert_when_floor_below_strikes(self):
        run = self._held([(0.20, 398.0, 0.8, 0.8), (0.22, 397.0, 0.9, 0.9)])
        assert run['totals']['call_sales'] == 1
        assert run['diag']['basis_rule_binding_days'] == 0

    def test_binding_picks_closest_qualifying_above(self):
        # unconstrained pick would be 394 (nearest 0.20); rule takes 397
        run = self._held([(0.20, 394.0, 0.9, 0.9), (0.17, 397.0, 0.7, 0.7)])
        assert run['totals']['call_sales'] == 1
        call_entry = next(t for t in run['trades'] if t.get('side') == 'call')
        assert call_entry['strike'] == 397.0
        assert run['diag']['basis_rule_binding_days'] == 1

    def test_adjusted_basis_ratchets_the_floor(self):
        # assigned at 396 with a $1.00 gross put premium: adjusted floor
        # starts at 395.0 (strike minus GROSS premiums, §8 verbatim), so the
        # 395.5 call sells under 'adjusted' and is blocked under 'strike';
        # after its $0.80 premium the floor ratchets to 394.2, unlocking the
        # 394.5 call the next night.
        dates = D[:6]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 395.5, 0.8, 0.8)]},
            i3={'puts': [], 'calls': [(0.20, 394.5, 0.7, 0.7)]},
        )
        closes = [400.0, 400.0, 390.0, 391.0, 392.0, 393.0]
        strike_v = run_wheel(dates, closes, market, up_signals(dates), **cell())
        assert strike_v['totals']['call_sales'] == 0
        adj = run_wheel(dates, closes, market, up_signals(dates),
                        **cell(), basis_variant='adjusted')
        assert adj['totals']['call_sales'] == 2
        second = [t for t in adj['trades'] if t.get('side') == 'call'][1]
        assert second['strike'] == 394.5


class TestGate:
    def test_blocks_down_day_and_ablation_sells(self):
        dates = D[:4]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        sig = up_signals(dates)
        sig[dates[1]] = {'up_355': False, 'up_cc': True, 'fallback': False, 'disagree': True}
        gated = run_wheel(dates, [400.0] * 4, market, sig, **cell())
        assert gated['totals']['put_sales'] == 0 and gated['diag']['gate_blocked'] == 1
        cc = run_wheel(dates, [400.0] * 4, market, sig, **cell(gate='cc'))
        assert cc['totals']['put_sales'] == 1
        off = run_wheel(dates, [400.0] * 4, market, sig, **cell(gate=None))
        assert off['totals']['put_sales'] == 1

    def test_missing_signal_blocks_and_counts(self):
        dates = D[:4]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0] * 4, market, {}, **cell())
        assert run['totals']['put_sales'] == 0
        # counted on every cash day the gate found no signal (both entry days)
        assert run['diag']['no_signal_days'] == 2

    def test_gate_never_touches_call_entries(self):
        # §4: the gate applies to put entries ONLY. Down-day (and missing)
        # signals on the call night must not block the call sale.
        dates = D[:4]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 398.0, 0.8, 0.8)]},
        )
        for call_night_signal in (
            {'up_355': False, 'up_cc': False, 'fallback': False, 'disagree': False},
            None,
        ):
            sig = up_signals(dates)
            if call_night_signal is None:
                del sig[dates[2]]
            else:
                sig[dates[2]] = call_night_signal
            run = run_wheel(dates, [400.0, 400.0, 390.0, 391.0], market, sig, **cell())
            assert run['totals']['call_sales'] == 1
            assert run['diag']['gate_blocked'] == 0

    def test_diag_counters_flow_through_run(self):
        dates = D[:5]
        market = idx(dates)
        sig = up_signals(dates)
        sig[dates[1]] = {'up_355': True, 'up_cc': True, 'fallback': True, 'disagree': False}
        sig[dates[2]] = {'up_355': False, 'up_cc': True, 'fallback': False, 'disagree': True}
        run = run_wheel(dates, [400.0] * 5, market, sig, **cell())
        assert run['diag']['signal_days'] == 4
        assert run['diag']['fallback_days'] == 1
        assert run['diag']['disagree_days'] == 1


class TestGateSignals:
    """load_gate_signals against a synthetic intraday archive (tmp file)."""

    def _write(self, tmp_path, rows):
        p = tmp_path / 'intraday.csv'
        with open(p, 'w') as f:
            f.write('timestamp,open,high,low,close,volume\n')
            for ts, close in rows:
                f.write(f'{ts},0,0,0,{close},100\n')
        return str(p)

    def test_normal_day_and_disagreement(self, tmp_path):
        dates = ['2024-11-26', '2024-11-27']
        closes = [500.0, 502.0]                      # official close UP
        p = self._write(tmp_path, [('2024-11-27 15:54:00', 499.0)])  # 3:55 signal DOWN
        sig = load_gate_signals(dates, closes, p)['2024-11-27']
        assert sig == {'up_355': False, 'up_cc': True, 'fallback': False, 'disagree': True}

    def test_half_day_uses_1255_and_ignores_afterhours(self, tmp_path):
        # 2024-11-29 is on the static half-day calendar: the 12:54 bar rules,
        # the 15:50 after-hours print (which would say UP) must be ignored.
        dates = ['2024-11-27', '2024-11-29']
        closes = [502.0, 501.0]
        p = self._write(tmp_path, [('2024-11-29 12:54:00', 500.0),
                                   ('2024-11-29 15:50:00', 600.0)])
        sig = load_gate_signals(dates, closes, p)['2024-11-29']
        assert sig['up_355'] is False and sig['fallback'] is False

    def test_missing_window_falls_back(self, tmp_path):
        dates = ['2024-12-02', '2024-12-03']
        closes = [501.0, 505.0]
        p = self._write(tmp_path, [('2024-12-03 12:00:00', 502.0)])  # outside 15:40-15:55
        sig = load_gate_signals(dates, closes, p)['2024-12-03']
        assert sig['fallback'] is True and sig['up_355'] is True     # cc sign: 505 > 501

    def test_scale_mismatch_falls_back(self, tmp_path):
        dates = ['2024-12-03', '2024-12-04']
        closes = [505.0, 506.0]
        p = self._write(tmp_path, [('2024-12-04 15:55:00', 300.0)])  # >5% off the close
        sig = load_gate_signals(dates, closes, p)['2024-12-04']
        assert sig['fallback'] is True and sig['up_355'] is True

    def test_last_bar_in_window_wins(self, tmp_path):
        # two bars inside [15:40, 15:55]: the later one (down) is the signal
        dates = ['2024-12-04', '2024-12-05']
        closes = [506.0, 507.0]
        p = self._write(tmp_path, [('2024-12-05 15:41:00', 510.0),
                                   ('2024-12-05 15:54:00', 505.0)])
        sig = load_gate_signals(dates, closes, p)['2024-12-05']
        assert sig['up_355'] is False and sig['fallback'] is False


class TestClampAndSizing:
    def test_affordability_clamp_partial_and_zero(self):
        dates = D[:4]
        # strike x 100 x 2 = $104K exceeds the $100K cash: clamps 2 -> 1
        market = idx(dates, i1={'puts': [(-0.20, 520.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 521.0, 522.0], market,
                        up_signals(dates), **cell(contracts=2))
        assert run['trades'][0]['n'] == 1 and run['diag']['clamped_days'] == 1
        market = idx(dates, i1={'puts': [(-0.20, 1100.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 401.0, 402.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['put_sales'] == 0 and run['diag']['clamp_zero_days'] == 1

    def test_sizing_battery_all_green_bag_reports_unbounded_kelly(self):
        rotations = [{'raw_pnl': 200.0}, {'raw_pnl': 350.0}]
        out = sizing_battery(rotations)
        assert out['kelly'] == 'unbounded'
        assert out['sim']['p_ruin'] == 0.0

    def test_sizing_battery_mixed_bag(self):
        rotations = [{'raw_pnl': 300.0}, {'raw_pnl': -2500.0}, {'raw_pnl': 400.0}]
        out = sizing_battery(rotations)
        # negative-mean bag: kelly_fraction's contract returns exactly 0.0
        assert out['kelly'] == 0.0
        assert out['sim']['n_trades'] == 3

    def test_sizing_battery_notch_scale_doubles_dollars(self):
        rotations = [{'raw_pnl': 300.0}, {'raw_pnl': 400.0}]
        n1 = sizing_battery(rotations, scale=1.0)
        n2 = sizing_battery(rotations, scale=2.0)
        # same draws (common seed), doubled per-rotation returns
        assert n2['sim']['terminal']['median'] - 1.0 == pytest.approx(
            2 * (n1['sim']['terminal']['median'] - 1.0), rel=1e-6)


class TestStops:
    def test_stop_triggers_at_close(self):
        dates = D[:5]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 390.0, 370.0, 371.0], market,
                        up_signals(dates), **cell(stop=0.05))
        assert run['totals']['stop_outs'] == 1
        (rot,) = run['rotations']
        assert rot['exit_reason'] == 'stopped' and rot['exit_price'] == 370.0
        assert rot['raw_pnl'] == pytest.approx(99.35 + (370.0 - 396.0) * 100)

    def test_same_close_assignment_then_stop(self):
        dates = D[:4]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 370.0, 371.0], market,
                        up_signals(dates), **cell(stop=0.05))
        assert run['totals']['assignments'] == 1 and run['totals']['stop_outs'] == 1
        assert run['open_rotation'] is None

    def test_stop_boundary_is_inclusive(self):
        # close EXACTLY at basis x (1 - stop) triggers: 396 x 0.95 = 376.20
        dates = D[:5]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 390.0, 376.20, 377.0], market,
                        up_signals(dates), **cell(stop=0.05))
        assert run['totals']['stop_outs'] == 1

    def test_stop_then_new_put_same_close(self):
        # §2 order: settle, stop, THEN entry — a stop-out re-enters the same
        # close when a qualifying put exists (the wheel keeps wheeling)
        dates = D[:4]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [(-0.20, 360.0, 1.0, 1.0)], 'calls': []},
        )
        run = run_wheel(dates, [400.0, 400.0, 370.0, 371.0], market,
                        up_signals(dates), **cell(stop=0.05))
        assert run['totals']['assignments'] == 1
        assert run['totals']['stop_outs'] == 1
        assert run['totals']['put_sales'] == 2
        assert run['totals']['expired_puts'] == 1     # the 360 put dies worthless
        assert run['open_rotation'] is None

    def test_no_stop_holds_through(self):
        dates = D[:4]
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 370.0, 371.0], market,
                        up_signals(dates), **cell())
        assert run['totals']['stop_outs'] == 0 and run['open_rotation'] is not None


class TestInterestAndComparator:
    def test_zero_rate_earns_nothing(self):
        dates = D[:4]
        run = run_wheel(dates, [400.0] * 4, idx(dates), up_signals(dates), **cell())
        assert run['summary']['final_equity'] == CAPITAL
        assert run['summary']['comparator_final'] == CAPITAL

    def test_rf_variant_accrues_both_books(self):
        dates = D[:4]                                # Tue..Fri: two 1-day gaps
        run = run_wheel(dates, [400.0] * 4, idx(dates), up_signals(dates),
                        **cell(cash_rate=0.045))
        day = 0.045 / 365.0
        wheel_hand = CAPITAL * (1 + day) * (1 + day)
        comp_hand = 40000.0 + 60000.0 * (1 + day) * (1 + day)
        assert run['summary']['final_equity'] == pytest.approx(wheel_hand)
        assert run['summary']['comparator_final'] == pytest.approx(comp_hand)

    def test_weekend_entry_counted(self):
        dates = D[2:5]                               # Thu, Fri, Mon
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 401.0], market, up_signals(dates), **cell())
        assert run['diag']['weekend_entries'] == 1   # Fri -> Mon spans the weekend
        assert run['diag']['weekend_settles'] == 1
        assert run['diag']['weekend_wins'] == 1
        assert run['totals']['weekend_pnl'] == pytest.approx(100.0 - 0.65)

    def test_interest_lands_inside_the_rotation_it_accrued_in(self):
        # Fri-entry put assigned Monday: the weekend credit accrues BEFORE the
        # assignment (cash-period side); Tuesday's credit accrues while the
        # rotation holds (rotation side). §7's attribution, pinned by hand.
        dates = D[2:6]                               # Thu, Fri, Mon, Tue
        market = idx(dates, i1={'puts': [(-0.20, 396.0, 1.0, 1.0)], 'calls': []})
        run = run_wheel(dates, [400.0, 400.0, 390.0, 391.0], market,
                        up_signals(dates), **cell(cash_rate=0.045))
        cash_after_sale = CAPITAL + 100.0 - 0.65
        credit_weekend = cash_after_sale * 0.045 * 3 / 365.0
        cash_after_assign = cash_after_sale + credit_weekend - 39600.0
        credit_tuesday = cash_after_assign * 0.045 * 1 / 365.0
        assert run['open_rotation']['interest'] == pytest.approx(credit_tuesday)
        assert run['totals']['interest'] == pytest.approx(credit_weekend + credit_tuesday)

    def test_two_contracts_uncapped_path(self):
        # §8: double assignment at 2 contracts -> 200 shares, calls sized 2
        dates = D[:5]
        market = idx(
            dates,
            i1={'puts': [(-0.20, 300.0, 1.0, 1.0)], 'calls': []},
            i2={'puts': [], 'calls': [(0.20, 302.0, 0.8, 0.8)]},
        )
        run = run_wheel(dates, [305.0, 305.0, 295.0, 303.0, 303.0], market,
                        up_signals(dates), **cell(contracts=2))
        assert run['diag']['clamped_days'] == 0
        assert run['trades'][0]['n'] == 2
        (rot,) = run['rotations']
        assert rot['shares'] == 200
        call_entry = next(t for t in run['trades'] if t.get('side') == 'call')
        assert call_entry['n'] == 2
        # called away at 302 on 200 shares: raw = puts + calls + 2*(302-300)*100
        assert rot['raw_pnl'] == pytest.approx(
            (1.0 * 200 - 1.30) + (0.8 * 200 - 1.30) + (302.0 - 300.0) * 200)

    def test_clamp_boundary_exact(self):
        # strike*100*2 vs cash-net-of-fees: 499.99 fits exactly, 500.00 does not
        dates = D[:4]
        for strike, want_n in ((499.99, 2), (500.00, 1)):
            market = idx(dates, i1={'puts': [(-0.20, strike, 1.0, 1.0)], 'calls': []})
            run = run_wheel(dates, [400.0, 400.0, 401.0, 402.0], market,
                            up_signals(dates), **cell(contracts=2))
            assert run['trades'][0]['n'] == want_n, strike


class TestIndexBuilder:
    def test_band_expiry_and_bid_filters(self):
        dates = ['2024-01-02', '2024-01-03', '2024-01-04']
        store = {'2024-01-03': {'candidates': [
            # (dte, delta, bid, ask, mid, expiration, strike, cid)
            (1, -0.20, 1.0, 1.2, 1.1, '2024-01-04', 396.0, 'P1'),   # qualifies
            (1, -0.30, 1.0, 1.2, 1.1, '2024-01-04', 390.0, 'P2'),   # outside band
            (1, -0.20, 0.0, 0.2, 0.1, '2024-01-04', 395.0, 'P3'),   # no bid
            (2, -0.20, 1.0, 1.2, 1.1, '2024-01-05', 394.0, 'P4'),   # wrong expiry
            (1, 0.20, 0.9, 1.1, 1.0, '2024-01-04', 404.0, 'C1'),    # qualifies
            (1, 0.55, 0.9, 1.1, 1.0, '2024-01-04', 400.0, 'C2'),    # outside band
        ]}}
        index = build_wheel_index(dates, store)
        assert [r[1] for r in index['2024-01-03']['puts']] == [396.0]
        assert [r[1] for r in index['2024-01-03']['calls']] == [404.0]
        assert index['2024-01-03']['eligible'] is True
        # leading day: no store entry -> empty, ineligible (and run_wheel
        # never trades it — its first daily row is dates[1])
        assert index['2024-01-02'] == {'puts': [], 'calls': [], 'eligible': False}

    def test_eligible_without_qualifying_rows(self):
        # a listed next-session expiry whose rows all fail the band/bid
        # filters: eligible=True (the calendar denominator), no rows (the
        # no-stretch numerator)
        dates = ['2024-01-02', '2024-01-03', '2024-01-04']
        store = {'2024-01-03': {'candidates': [
            (1, -0.40, 1.0, 1.2, 1.1, '2024-01-04', 390.0, 'P1'),
        ]}}
        index = build_wheel_index(dates, store)
        assert index['2024-01-03'] == {'puts': [], 'calls': [], 'eligible': True}


# ------------------------------------------------------------ dataset-gated

DATASETS = [data_path('qqq_option_dailies.csv'), data_path('qqq_option_dailies_puts.csv'),
            data_path('qqq_intraday_1min.csv'), data_path('qqq_10yr_prices_unadjusted.csv')]


@pytest.mark.skipif(not all(os.path.exists(p) for p in DATASETS),
                    reason='QQQ chain stores + intraday archive not present')
class TestQqqWheel1dteExploration:
    """The 2026-07-20 run's decisive numbers (plan §14 step 3; exploratory,
    kill-or-justify — docs/explorations.md carries the narrative).

    Headline: every one of the 48 grid cells trails the 100-share comparator
    (daily NW t from -0.74 to -2.86, none near the +2 escalation bar), while
    the basis rule delivers a 100% raw rotation win rate — the by-construction
    effect the plan predicted. The one right-signed number is the §7
    decomposition companion: the primary cell's option legs, delta-hedged
    overnight, read NW t +2.40 (diagnostic only — a +2.54-family echo, not a
    cell, and gross of hedge frictions).

    Expensive: two full-store loads plus the 48-cell grid (~10-20 min cold).
    """

    @pytest.fixture(scope='class')
    def primary(self):
        from realchains.wheel_1dte import run_experiment
        return run_experiment('primary')

    @pytest.fixture(scope='class')
    def secondary(self):
        from realchains.wheel_1dte import run_experiment
        return run_experiment('secondary')

    PRIMARY_KEY = 'gate=on|basis=on|stop=none|n=1|rate=0.0'

    def test_intraday_archive_provenance(self):
        from realchains.wheel_1dte import INTRADAY_SHA256, wheel_intraday_sha
        assert wheel_intraday_sha() == INTRADAY_SHA256

    def test_calendar_counts(self, primary):
        # the frozen §5 table, regenerated through the engine's own index
        assert {y: e['eligible'] for y, e in primary['calendar'].items()} == {
            '2023': 250, '2024': 252, '2025': 250, '2026': 106}
        assert {y: e['put_ok'] for y, e in primary['calendar'].items()} == {
            '2023': 250, '2024': 250, '2025': 249, '2026': 106}
        assert {y: e['call_ok'] for y, e in primary['calendar'].items()} == {
            '2023': 248, '2024': 251, '2025': 249, '2026': 105}

    def test_primary_cell_headline(self, primary):
        c = primary['cells'][self.PRIMARY_KEY]
        assert c['n_days'] == 858 and c['nw_lag'] == 6
        assert c['final_equity'] == pytest.approx(134290.05, abs=0.02)
        assert c['comparator_final'] == pytest.approx(144058.00, abs=0.02)
        assert c['gap_final'] == pytest.approx(-9767.95, abs=0.02)
        assert c['daily_nw_t'] == pytest.approx(-1.04, abs=0.005)
        assert c['daily_naive_t'] == pytest.approx(-0.98, abs=0.005)

    def test_every_cell_trails_the_comparator(self, primary):
        ts = {k: v['daily_nw_t'] for k, v in primary['cells'].items()}
        assert len(ts) == 48
        assert all(t < 0 for t in ts.values())
        assert sum(1 for t in ts.values() if t > 2.0) == 0    # the §11 bar: empty
        assert max(ts.values()) == pytest.approx(-0.74, abs=0.005)
        assert min(ts.values()) == pytest.approx(-2.86, abs=0.005)
        assert min(ts, key=ts.get) == 'gate=off|basis=on|stop=0.05|n=1|rate=0.0'

    def test_rotation_ledger_two_columns(self, primary):
        r = primary['cells'][self.PRIMARY_KEY]['rotations']
        assert r['n'] == 37
        assert r['raw_win_rate'] == 100.0          # by construction (§7)
        assert r['rescue_share'] == 100.0          # coincides by construction
        assert r['raw_mean'] == pytest.approx(558.34, abs=0.02)
        assert r['gap_mean'] == pytest.approx(423.39, abs=0.02)
        assert r['gap_win_rate'] == pytest.approx(91.9, abs=0.05)
        assert r['underwater_days_median'] == 3
        assert r['underwater_days_max'] == 77

    def test_overnight_ledger(self, primary):
        ov = primary['cells'][self.PRIMARY_KEY]['overnight']
        assert ov['pooled']['n'] == 423
        assert ov['pooled']['expectancy_r'] == pytest.approx(0.2194, abs=0.0005)
        assert ov['pooled']['win_rate'] == pytest.approx(84.2, abs=0.05)
        # the junior/senior split: trade-order NW t +2.13 vs daily -1.04
        assert ov['pooled']['r_newey_west_t'] == pytest.approx(2.131, abs=0.005)
        # the Tharp trade profile, per side: the put side carries ALL the
        # per-trade expectancy; the call side is negative per trade — and
        # every call "loss" is a called-away night (the rotation's win)
        assert ov['puts'] == {'n': 276, 'win_rate': 88.0, 'expectancy_r': 0.4452,
                              'r_std': 1.9026, 'worst_r': -11.51}
        assert ov['calls'] == {'n': 147, 'win_rate': 76.9, 'expectancy_r': -0.2045,
                               'r_std': 2.8693, 'worst_r': -15.78}

    def test_all_call_losses_are_called_away_nights(self, primary):
        # the ledger convention made visible: every call-side per-trade
        # "loss" lands on a night the shares deliver at a profit — the
        # overnight ledger charges the rotation's happy ending to the call
        cell = primary['cells'][self.PRIMARY_KEY]
        assert cell['call_losses'] == 34
        assert cell['call_losses_on_away_nights'] == 34
        # ...but not every call-away is a ledger loss: 3 of the 37
        # called-away nights kept more premium than the intrinsic given up
        assert cell['totals']['call_aways'] == 37

    def test_dollar_decomposition_buckets(self, primary):
        t = primary['cells'][self.PRIMARY_KEY]['totals']
        assert t['premium_collected'] == pytest.approx(23265.00, abs=0.02)
        assert t['fees'] == pytest.approx(274.95, abs=0.02)
        assert t['assignment_loss'] == pytest.approx(7900.01, abs=0.02)
        assert t['holding_share_pnl'] == pytest.approx(19200.01, abs=0.02)
        assert t['assignments'] == 37 and t['call_aways'] == 37
        assert t['expired_puts'] == 239 and t['expired_calls'] == 110

    def test_gate_pair_and_diagnostics(self, primary):
        gate_off = primary['cells']['gate=off|basis=on|stop=none|n=1|rate=0.0']
        assert gate_off['daily_nw_t'] == pytest.approx(-1.33, abs=0.005)
        assert gate_off['gap_final'] == pytest.approx(-9375.85, abs=0.02)
        # the gate blocked 141 nights yet cost only 28 put sales (304 -> 276):
        # the state machine buffers it — blocked nights mostly delay the sale
        assert gate_off['totals']['put_sales'] == 304
        d = primary['cells'][self.PRIMARY_KEY]['diag']
        assert d['gate_blocked'] == 141
        assert d['disagree_days'] == 26 and d['signal_days'] == 859   # 3.0%
        assert d['fallback_days'] == 0
        assert d['basis_rule_binding_days'] == 297
        assert d['weekend_settles'] == 95 and d['weekend_wins'] == 84

    def test_basis_rule_pair(self, primary):
        off = primary['cells']['gate=on|basis=off|stop=none|n=1|rate=0.0']
        assert off['daily_nw_t'] == pytest.approx(-1.84, abs=0.005)
        assert off['gap_final'] == pytest.approx(-23810.50, abs=0.02)
        assert off['rotations']['raw_win_rate'] == pytest.approx(64.9, abs=0.05)

    def test_variants(self, primary):
        assert primary['variant_cc_gate']['daily_nw_t'] == pytest.approx(-1.05, abs=0.005)
        assert primary['variant_bid_fill']['daily_nw_t'] == pytest.approx(-1.09, abs=0.005)
        assert primary['variant_adjusted_basis']['daily_nw_t'] == pytest.approx(-1.09, abs=0.005)

    def test_decomposition_companion(self, primary):
        dc = primary['decomposition']
        assert dc['unhedged_total'] == pytest.approx(34290.05, abs=0.02)
        assert dc['hedged_total'] == pytest.approx(27195.19, abs=0.02)
        assert dc['direction_bill'] == pytest.approx(7094.86, abs=0.02)
        # the one right-signed number: hedged premium content, t +2.40 —
        # DIAGNOSTIC only (not a §11 cell; gross of hedge frictions)
        assert dc['hedged_daily_nw_t'] == pytest.approx(2.40, abs=0.005)

    def test_sizing_battery(self, primary):
        n1, n2 = primary['sizing_n1'], primary['sizing_n2']
        assert n1['n'] == 37 and n1['kelly'] == 'unbounded'
        assert n1['sim']['p_ruin'] == 0.0 and n1['sim']['p_ruin_25dd'] == 0.0
        assert n1['sim']['terminal']['median'] == pytest.approx(1.2269, abs=0.0005)
        assert n2['sim']['terminal']['median'] == pytest.approx(1.5028, abs=0.0005)

    def test_secondary_arm(self, secondary):
        pc, off = secondary['primary_config'], secondary['gate_off']
        assert pc['n_days'] == 2514 and pc['nw_lag'] == 8
        assert pc['daily_nw_t'] == pytest.approx(-1.27, abs=0.005)
        assert pc['gap_final'] == pytest.approx(-14112.90, abs=0.02)
        assert off['daily_nw_t'] == pytest.approx(-1.64, abs=0.005)
        # 100% raw rotation win rate THROUGH 2018/2020/2022 — the losses
        # became time: worst rotation 372 days underwater (485 gate-off)
        assert pc['rotations']['raw_win_rate'] == 100.0
        assert pc['rotations']['n'] == 48
        assert pc['rotations']['underwater_days_max'] == 372
        assert off['rotations']['underwater_days_max'] == 485
        assert secondary['calendar']['2016']['both_ok'] == 13
        assert secondary['calendar']['2022']['both_ok'] == 170
