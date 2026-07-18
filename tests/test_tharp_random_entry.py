"""Always-run mechanics tests for the Tharp random-entry replication engine.

Every assertion hand-derived on synthetic markets (docs/tharp_random_entry_plan.md
section 5): ATR arithmetic, trail ratchet + EOD stop-market fills, percent-risk
sizing, coin determinism, cost/borrow accounting, Tharp R units, the one-day
re-entry gap, and carried marks on no-bar days. The dataset-gated ensemble pins
(Phase 1/2) are appended once the runs exist; no ensemble number is produced
before these are green (the plan's section 7 ordering).
"""
from __future__ import annotations

import math
import os
import random
from typing import Any

import numpy as np
import pytest

from common.paths import DATA_DIR
from engine.tharp_random_entry import (
    ATR_PERIOD,
    CAREER_SEED_BASE,
    N_CAREERS,
    STOP_MULT,
    TICKERS,
    atr_series,
    build_market,
    drift_twin,
    no_stop_ensemble,
    placebo_exit_ensemble,
    run_career,
)


def _flat_market(
    closes: list[float], atr: float = 1.0, ticker: str = 'AAA'
) -> dict[str, Any]:
    """A one-instrument synthetic market with a constant, pre-warm ATR."""
    n = len(closes)
    return {
        'dates': [f'2020-01-{i + 1:02d}' for i in range(n)],
        'closes': {ticker: np.asarray(closes, dtype=float)},
        'atrs': {ticker: np.full(n, float(atr))},
        'tickers': (ticker,),
    }


def _direction(seed: int) -> int:
    """The engine's first coin flip for a seed (long +1 / short -1)."""
    return 1 if random.Random(seed).random() < 0.5 else -1


def _seed_with(direction: int, start: int = 1) -> int:
    s = start
    while _direction(s) != direction:
        s += 1
    return s


class TestAtr:
    def test_hand_computed_true_range_and_warmup(self) -> None:
        high = np.array([10.0, 11.0, 12.0, 15.0])
        low = np.array([9.0, 10.0, 11.0, 13.0])
        close = np.array([9.5, 10.5, 11.5, 14.0])
        out = atr_series(high, low, close, period=2)
        # TR: day1 max(1, |11-9.5|, |10-9.5|)=1.5; day2 max(1, 1.5, .5)=1.5;
        # day3 max(2, 3.5, 1.5)=3.5 -> ATR(2)[2]=(1.5+1.5)/2, [3]=(1.5+3.5)/2
        assert math.isnan(out[0]) and math.isnan(out[1])
        assert out[2] == pytest.approx(1.5)
        assert out[3] == pytest.approx(2.5)

    def test_default_period_warmup_length(self) -> None:
        n = ATR_PERIOD + 3
        out = atr_series(np.full(n, 2.0), np.full(n, 1.0), np.full(n, 1.5))
        assert np.isnan(out[:ATR_PERIOD]).all()
        assert out[ATR_PERIOD] == pytest.approx(1.0)   # constant TR = h-l


class TestTrailAndFills:
    def test_long_trail_ratchets_and_stops_at_close(self) -> None:
        # long entry at 100 (ATR 1, stop dist 3): trail 97; closes rise to
        # 110 (trail ratchets to 107), then 106 breaches -> exit AT 106.
        seed = _seed_with(+1)
        m = _flat_market([100.0, 104.0, 110.0, 106.0, 106.0, 106.0])
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        assert c['n_trades'] >= 1
        tr = c['trades'][0]
        assert tr['direction'] == 1
        assert tr['entry_px'] == 100.0 and tr['exit_px'] == 106.0
        assert tr['pnl'] == pytest.approx(6.0 * tr['shares'])
        assert tr['r_multiple'] == pytest.approx(6.0 / 3.0)   # risk_ps = 3*ATR

    def test_short_mirror_stops_on_a_rally(self) -> None:
        seed = _seed_with(-1)
        m = _flat_market([100.0, 96.0, 90.0, 94.0, 94.0, 94.0])
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        tr = c['trades'][0]
        assert tr['direction'] == -1
        # trail after 90: 93; close 94 breaches -> exit at 94
        assert tr['entry_px'] == 100.0 and tr['exit_px'] == 94.0
        assert tr['pnl'] == pytest.approx(6.0 * tr['shares'])

    def test_trail_never_loosens(self) -> None:
        seed = _seed_with(+1)
        # rise then chop: the trail set at the peak must hold
        m = _flat_market([100.0, 110.0, 108.0, 108.0, 107.1, 106.9])
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        tr = c['trades'][0]
        assert tr['exit_px'] == 106.9        # 110 peak -> trail 107, held

    def test_one_day_reentry_gap(self) -> None:
        seed = _seed_with(+1)
        m = _flat_market([100.0, 90.0, 100.0, 101.0, 102.0, 103.0, 104.0])
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        assert len(c['trades']) >= 2
        first, second = c['trades'][0], c['trades'][1]
        assert second['entry_date'] > first['exit_date']       # never same day


class TestSizingAndAccounting:
    def test_percent_risk_share_count(self) -> None:
        seed = _seed_with(+1)
        m = _flat_market([100.0] * (3), atr=2.0)
        c = run_career(seed, m, capital=100_000.0, cost_bps=0.0,
                       borrow_annual=0.0)
        # shares = floor(1% * 100k / (3*2)) = floor(166.67) = 166
        # (flat closes: no exit; position remains open, no trade rows)
        assert c['n_trades'] == 0
        assert c['final_equity'] == pytest.approx(100_000.0)   # marked flat

    def test_fill_costs_hit_both_sides(self) -> None:
        seed = _seed_with(+1)
        m = _flat_market([100.0, 104.0, 110.0, 106.0])
        gross = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        net = run_career(seed, m, cost_bps=10.0, borrow_annual=0.0)
        tr_g, tr_n = gross['trades'][0], net['trades'][0]
        assert tr_g['shares'] == tr_n['shares']
        expected_costs = (100.0 + 106.0) * tr_n['shares'] * 10.0 / 10_000.0
        assert tr_g['pnl'] - tr_n['pnl'] == pytest.approx(expected_costs)

    def test_borrow_accrues_daily_on_shorts_only(self) -> None:
        s_short = _seed_with(-1)
        m = _flat_market([100.0, 96.0, 90.0, 94.0])
        free = run_career(s_short, m, cost_bps=0.0, borrow_annual=0.0)
        paid = run_career(s_short, m, cost_bps=0.0, borrow_annual=0.252)
        tr_f, tr_p = free['trades'][0], paid['trades'][0]
        # 2 open days accrue at 0.252/252 = 0.1% of that day's short notional
        expected = (96.0 + 90.0) * tr_p['shares'] * 0.001
        assert tr_f['pnl'] - tr_p['pnl'] == pytest.approx(expected)
        s_long = _seed_with(+1)
        m2 = _flat_market([100.0, 104.0, 110.0, 106.0])
        a = run_career(s_long, m2, cost_bps=0.0, borrow_annual=0.0)
        b = run_career(s_long, m2, cost_bps=0.0, borrow_annual=0.252)
        assert a['trades'][0]['pnl'] == pytest.approx(b['trades'][0]['pnl'])

    def test_mae_r_is_nonpositive_and_r_units_consistent(self) -> None:
        seed = _seed_with(+1)
        m = _flat_market([100.0, 98.0, 104.0, 110.0, 106.0])
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        tr = c['trades'][0]
        assert tr['mae_r'] == pytest.approx(-2.0 / 3.0)   # 98 dip = -2/risk 3
        assert tr['r_multiple'] == pytest.approx(tr['pnl'] / tr['initial_risk'])

    def test_no_bar_days_carry_the_mark(self) -> None:
        seed = _seed_with(+1)
        m = _flat_market([100.0, 105.0, np.nan, 105.0, 105.0])
        m['closes']['AAA'][2] = np.nan
        c = run_career(seed, m, cost_bps=0.0, borrow_annual=0.0)
        # equity on the nan day equals the prior day's mark (carried)
        assert c['equity'][2] == pytest.approx(c['equity'][1])


class TestDeterminismAndEnsembles:
    def test_same_seed_same_career(self) -> None:
        m = _flat_market([100.0, 104.0, 110.0, 106.0, 100.0, 103.0, 99.0])
        a = run_career(7, m)
        b = run_career(7, m)
        assert a['trades'] == b['trades']
        assert np.array_equal(a['equity'], b['equity'])

    def test_different_seed_can_differ(self) -> None:
        m = _flat_market([100.0, 104.0, 110.0, 106.0, 100.0, 103.0, 99.0])
        dirs = {run_career(s, m)['trades'][0]['direction']
                for s in range(1, 30) if run_career(s, m)['trades']}
        assert dirs == {1, -1}

    def test_placebo_and_no_stop_are_seeded_deterministic(self) -> None:
        m = _flat_market([100.0 + i for i in range(30)])
        a = placebo_exit_ensemble(m, [2, 3, 4], n_careers=5, seed=1)
        b = placebo_exit_ensemble(m, [2, 3, 4], n_careers=5, seed=1)
        assert a == b and len(a) == 5
        c = no_stop_ensemble(m, hold_days=3, n_careers=4, seed=2)
        assert c == no_stop_ensemble(m, hold_days=3, n_careers=4, seed=2)

    def test_drift_twin_hand_check(self) -> None:
        # constant +$1000 notional on a +1%/day instrument, 4 days:
        # twin pnl = 1000 * (r1+r2+r3)
        m = _flat_market([100.0, 101.0, 102.01, 103.0301])
        career = {'notional': np.full((4, 1), 1000.0)}
        pnl = drift_twin(career, m, borrow_annual=0.0)
        assert pnl == pytest.approx(1000.0 * 0.03, rel=1e-3)

    def test_market_builder_aligns_and_warms(self) -> None:
        m = build_market(('SPY',))
        assert m['dates'][0] >= '2000-01-03'
        atr = m['atrs']['SPY']
        first_warm = int(np.argmax(~np.isnan(atr)))
        assert np.isnan(atr[:first_warm]).all()
        assert first_warm == 0                 # warmup absorbed pre-span
        assert CAREER_SEED_BASE == 20260719 and STOP_MULT == 3.0

# --- the replication pins (dataset-gated) ------------------------------------

_HAVE_OHLC = all(
    os.path.exists(os.path.join(DATA_DIR, f'{t.lower()}_daily_ohlc.csv'))
    for t in TICKERS)


@pytest.mark.skipif(not _HAVE_OHLC,
                    reason='needs the nine {ticker}_daily_ohlc.csv files')
class TestTharpReplicationEnsemble:
    """The Basso/Tharp random-entry replication, run 2026-07-18
    (docs/tharp_random_entry_plan.md; narrative in docs/explorations.md).

    EXPLORATORY — a replication study, kill-or-justify; nothing enters the
    idea ledger, no e-value is spent.

    Phase 1 — HIS SENTENCE FAILS TO REPLICATE on this universe: only 54% of
    100 seeded careers have positive expectancy and 40% end above starting
    capital (median final $89,410 over ~26 years — the median career LOSES).
    Yet the trend-follower SHAPE is exactly as he describes (win rate 36.4%,
    payoff ratio 1.75) — the mechanism produces his signature without his
    profits: the pooled bag is zero-expectancy (+0.0011R over 195,900
    trades).

    Phase 2 — HIS MORAL FAILS EVERY NULL HIS ERA DIDN'T RUN. The trailing
    stop manufactures a +25%-of-capital average long tilt out of coin flips
    (the endogenous-tilt prediction, confirmed), and the drift twin holding
    just that average exposure beats the live system in 98% of careers
    (twin median +$80,654 vs system −$10,590): the system's timing DESTROYS
    ~$91K of the drift it captures. Placebo exits put the real trailing stop
    at the 44th percentile of skill-free exits (P = 0.44 — no mechanism),
    and the fixed-hold no-stop control is indistinguishable. Sizing: at HIS
    1% risk on the zero-edge bag, P(ruin) = 0.32 and P(25% DD) = 0.70 with
    median terminal 0.853; Kelly is 0.0006 — seventeen times smaller than
    the fraction he taught. Exits and sizing did not carry the system; drift
    did, and the exits threw most of it away.

    Everything is seed-deterministic (careers 20260719+i; placebo/no-stop
    offsets +500/+2000), so the pins are tight."""

    @pytest.fixture(scope='class')
    def phases(self) -> dict[str, Any]:
        from common.position_sizing import kelly_fraction, sizing_sweep
        market = build_market()
        summaries, twins, tilt = [], [], []
        all_r: list[float] = []
        all_mae: list[float] = []
        all_holds: list[int] = []
        r_units: list[float] = []
        n_per: list[int] = []
        for i in range(N_CAREERS):
            c = run_career(CAREER_SEED_BASE + i, market, keep_positions=True)
            twins.append(drift_twin(c, market))
            tilt.append(float(c['notional'].sum(axis=1).mean()))
            rs = [t['r_multiple'] for t in c['trades']]
            all_r.extend(rs)
            all_mae.extend(t['mae_r'] for t in c['trades'])
            all_holds.extend(t['hold_days'] for t in c['trades'])
            n_per.append(len(rs))
            if rs:
                r_units.append(float(np.mean(rs)))
            summaries.append((c['expectancy_r'], c['final_equity'],
                              c['win_rate'], c['n_trades']))
        med_n = int(np.median(n_per))
        med_hold = int(np.median(all_holds))
        placebo = placebo_exit_ensemble(market, all_holds, n_careers=1000)
        nostop = no_stop_ensemble(market, hold_days=med_hold, n_careers=100)
        sweep = sizing_sweep(all_r, mae_r=all_mae, n_trades=med_n)
        return {
            'exp': np.array([s[0] for s in summaries]),
            'fin': np.array([s[1] for s in summaries]),
            'win': np.array([s[2] for s in summaries]),
            'twins': np.array(twins), 'tilt': np.array(tilt),
            'all_r': all_r, 'med_hold': med_hold, 'med_n': med_n,
            'r_units': r_units, 'placebo': np.array(placebo),
            'nostop': np.array(nostop), 'sweep': sweep,
            'kelly': kelly_fraction(all_r),
        }

    def test_phase1_his_sentence_fails(self, phases) -> None:
        exp, fin = phases['exp'], phases['fin']
        assert float(np.mean(exp > 0)) == pytest.approx(0.54, abs=0.001)
        assert float(np.mean(fin > 100_000)) == pytest.approx(0.40, abs=0.001)
        assert float(np.median(exp)) == pytest.approx(0.0028, abs=0.0005)
        assert float(np.median(fin)) == pytest.approx(89_410, abs=50)
        assert float(fin.min()) == pytest.approx(33_609, abs=50)

    def test_phase1_his_shape_holds(self, phases) -> None:
        assert float(np.median(phases['win'])) == pytest.approx(36.4, abs=0.1)
        wins = [r for r in phases['all_r'] if r > 0]
        losses = [r for r in phases['all_r'] if r <= 0]
        payoff = abs(np.mean(wins) / np.mean(losses))
        assert payoff == pytest.approx(1.75, abs=0.01)
        assert float(np.mean(phases['all_r'])) == pytest.approx(0.0011,
                                                               abs=0.0003)
        # the endogenous long tilt the design predicted, confirmed:
        assert float(np.mean(phases['tilt'])) / 100_000 == pytest.approx(
            0.25, abs=0.01)

    def test_phase2_drift_twin_demolishes_the_system(self, phases) -> None:
        sys_pnl = phases['fin'] - 100_000
        diff = sys_pnl - phases['twins']
        assert float(np.mean(diff > 0)) == pytest.approx(0.02, abs=0.001)
        assert float(np.median(phases['twins'])) == pytest.approx(
            80_654, abs=200)
        assert float(np.median(diff)) == pytest.approx(-91_073, abs=300)

    def test_phase2_placebo_and_no_stop_absorb_the_mechanism(
        self, phases
    ) -> None:
        assert phases['med_hold'] == 22
        real_mean = float(np.mean(phases['r_units']))
        assert real_mean == pytest.approx(0.0014, abs=0.0003)
        placebo = phases['placebo']
        pct = (1 + np.sum(placebo >= real_mean)) / (1 + len(placebo))
        assert pct == pytest.approx(0.4436, abs=0.01)   # mid-placebo: no skill
        assert float(np.median(phases['nostop'])) == pytest.approx(
            -0.0016, abs=0.0008)

    def test_phase2_sizing_at_his_fraction(self, phases) -> None:
        assert phases['med_n'] == 1954
        sweep = phases['sweep']
        assert sweep[0.0025]['p_ruin'] == 0.0
        assert sweep[0.01]['p_ruin'] == pytest.approx(0.3177, abs=0.005)
        assert sweep[0.01]['p_ruin_25dd'] == pytest.approx(0.7009, abs=0.005)
        assert sweep[0.01]['terminal']['median'] == pytest.approx(
            0.8533, abs=0.005)
        assert sweep[0.03]['p_ruin'] == pytest.approx(0.8805, abs=0.005)
        assert phases['kelly'] == pytest.approx(0.0006, abs=0.0002)
