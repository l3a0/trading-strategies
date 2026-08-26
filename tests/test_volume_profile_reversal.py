"""Tests for the volume-profile mean-reversion scout.

Two layers, mirroring the repo's exploration-pinning convention:

* an ALWAYS-RUN synthetic layer that pins the novel numerical logic on
  hand-computable inputs — the split adjustment, the volume-by-price /
  value-area math, the point-in-time (look-ahead-free) profile construction,
  the sign-inverted reversion outcome, and the multi-day forward path; and
* a DATASET-GATED layer (``TestNvdaVolumeProfileReversal``) that pins the
  decisive NVDA measurement so the exploration-log prose cannot drift from the
  run that produced it. It skips when the (gitignored, local-only) NVDA
  minute archive is absent, exactly as the intraday-continuation tests do.
"""
from __future__ import annotations

import dataclasses as dc
import json
import os
import zlib

import numpy as np
import pytest

from common.paths import data_path
from engine import volume_profile_reversal as vp
from engine.intraday_continuation import SESSION_BARS

_HAVE_NVDA = vp.archive_path(vp.TICKER) is not None
_UNIV_PATH = data_path('volume_profile_universe_results.json')


# ------------------------------------------------------------ synthetic bars

def _session(price_bars, vol_bars):
    """One (78,) session row pair from per-bar price and volume lists."""
    p = np.asarray(price_bars, dtype=float)
    v = np.asarray(vol_bars, dtype=float)
    assert p.size == SESSION_BARS and v.size == SESSION_BARS
    return p, v


def _bars_from(prices, volumes, dates):
    """A bars dict (open=high=low=close=price per bar) for detector tests."""
    P = np.asarray(prices, dtype=float)
    V = np.asarray(volumes, dtype=float)
    return {'dates': np.asarray(dates), 'open': P.copy(), 'high': P.copy(),
                'low': P.copy(), 'close': P.copy(), 'volume': V.copy(),
                'n_bars': np.full(P.shape[0], SESSION_BARS)}


# ----------------------------------------------------------- profile math

class TestVolumeProfileMath:
    def test_poc_and_value_area_center(self):
        # a triangular distribution peaked at 100
        prices = np.array([90, 95, 98, 100, 100, 100, 102, 105, 110], float)
        vols = np.array([1, 2, 5, 20, 20, 20, 5, 2, 1], float)
        poc, vah, val = vp.volume_profile(prices, vols, n_bins=20, va_frac=0.70)
        assert val < poc < vah
        assert abs(poc - 100) < 2.0
        # the value area is a band strictly inside the full range
        assert 90 <= val and vah <= 110

    def test_all_mass_one_bin(self):
        # everything at one price -> degenerate range -> None (guarded)
        prices = np.full(10, 100.0)
        vols = np.ones(10)
        assert vp.volume_profile(prices, vols, n_bins=20, va_frac=0.70) is None

    def test_zero_total_volume(self):
        prices = np.array([100.0, 101.0])
        assert vp.volume_profile(prices, np.zeros(2), n_bins=10, va_frac=0.70) is None

    def test_value_area_fraction_monotone(self):
        # a wider fraction can only widen (never narrow) the value area
        rng = np.random.default_rng(0)
        prices = 100 + rng.standard_normal(500)
        vols = rng.random(500) + 0.1
        _, vah70, val70 = vp.volume_profile(prices, vols, 60, 0.70)
        _, vah90, val90 = vp.volume_profile(prices, vols, 60, 0.90)
        assert vah90 >= vah70 - 1e-9 and val90 <= val70 + 1e-9

    def test_value_area_terminates_on_narrow_grid(self):
        # a grid smaller than the requested fraction must still return, not spin
        vol = np.array([1.0, 5.0, 1.0])
        edges = np.linspace(0, 3, 4)
        _poc, vah, val = vp._value_area(vol, edges, 0.99)
        assert val <= vah and 0 <= val and vah <= 3


# ----------------------------------------------------------- split adjust

class TestSplitAdjust:
    def test_single_split_halves_price_doubles_volume(self):
        bars = _bars_from([[100.0] * SESSION_BARS, [50.0] * SESSION_BARS],
                          [[10.0] * SESSION_BARS, [20.0] * SESSION_BARS],
                          ['2020-01-01', '2020-01-02'])
        adj = vp.split_adjust_bars(bars, [('2020-01-02', 2.0)])
        # day 1 (before ex-date) divided by 2, volume *2; day 2 unchanged
        assert np.allclose(adj['close'][0], 50.0)
        assert np.allclose(adj['volume'][0], 20.0)
        assert np.allclose(adj['close'][1], 50.0)
        assert np.allclose(adj['volume'][1], 20.0)

    def test_no_split_is_identity(self):
        bars = _bars_from([[7.0] * SESSION_BARS], [[3.0] * SESSION_BARS],
                          ['2020-01-01'])
        adj = vp.split_adjust_bars(bars, [])
        assert np.array_equal(adj['close'], bars['close'])
        assert np.array_equal(adj['volume'], bars['volume'])

    def test_multi_split_is_cumulative_product(self):
        bars = _bars_from(
            [[400.0] * SESSION_BARS, [200.0] * SESSION_BARS, [100.0] * SESSION_BARS],
            [[1.0] * SESSION_BARS] * 3,
            ['2020-01-01', '2020-06-01', '2021-01-01'])
        # a x2 then a x2: the first day sees BOTH (factor 4), the second sees one
        adj = vp.split_adjust_bars(bars, [('2020-06-01', 2.0), ('2021-01-01', 2.0)])
        assert np.allclose(adj['close'][0], 100.0)   # 400/4
        assert np.allclose(adj['close'][1], 100.0)   # 200/2
        assert np.allclose(adj['close'][2], 100.0)   # 100/1


# ------------------------------------------------------- look-ahead (PIT)

class TestLookAhead:
    def _profile_at(self, bars, s, i):
        typ, vol = vp._window_typical_volume(bars, slice(s, s + 1), 0, i)
        return vp.volume_profile(typ, vol, 48, 0.70)

    def test_developing_profile_ignores_future_bars(self):
        rng = np.random.default_rng(1)
        P = 100 + np.cumsum(rng.standard_normal((5, SESSION_BARS)) * 0.1, axis=1)
        V = rng.random((5, SESSION_BARS)) + 0.1
        bars = _bars_from(P, V, [f'2020-01-0{d+1}' for d in range(5)])
        before = self._profile_at(bars, 2, 30)
        # mutate every bar at clock >= 30 in EVERY session, drastically
        bars2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in bars.items()}
        for k in ('open', 'high', 'low', 'close', 'volume'):
            bars2[k][:, 30:] *= 7.0
        after = self._profile_at(bars2, 2, 30)
        assert before == after     # bars 0..29 only -> untouched by the future

    def test_prior_profile_ignores_current_and_future_sessions(self):
        rng = np.random.default_rng(2)
        P = 100 + rng.standard_normal((6, SESSION_BARS)) * 0.5
        V = rng.random((6, SESSION_BARS)) + 0.1
        bars = _bars_from(P, V, [f'2020-02-0{d+1}' for d in range(6)])
        typ, vol = vp._window_typical_volume(bars, slice(1, 4), 0, SESSION_BARS)
        before = vp.volume_profile(typ, vol, 48, 0.70)
        bars2 = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in bars.items()}
        for k in ('open', 'high', 'low', 'close', 'volume'):
            bars2[k][4:] *= 9.0    # sessions AT/after the window end
        typ2, vol2 = vp._window_typical_volume(bars2, slice(1, 4), 0, SESSION_BARS)
        assert before == vp.volume_profile(typ2, vol2, 48, 0.70)

    def test_synthetic_no_print_bars_excluded_from_profile(self):
        # a zero-volume bar carries a synthetic filled price; it must not
        # become a node nor stretch the price range
        prices = np.array([[100.0] * 40 + [999.0] + [100.0] * 37])
        vols = np.array([[1.0] * 40 + [0.0] + [1.0] * 37])
        bars = _bars_from(prices, vols, ['2020-01-01'])
        typ, _vol = vp._window_typical_volume(bars, slice(0, 1), 0, SESSION_BARS)
        assert 999.0 not in typ      # the no-print level is dropped


# ------------------------------------------------------- reversion outcome

class TestReversionOutcome:
    def test_short_target_first_is_a_win(self):
        # short from 100, target 98 (POC below), stop 103 (above the edge)
        fh = np.array([100.5, 99.0, 98.5])
        fl = np.array([99.5, 98.5, 97.9])   # fl<=98 on bar 2
        fc = np.array([100.0, 98.8, 98.2])
        ret, first, _ = vp._reversion_outcome(fh, fl, fc, 99.0, 100.0, 98.0,
                                              103.0, -1, 'barrier')
        assert first == 1
        assert ret == pytest.approx(0.02)   # short 100->98 = +2%

    def test_short_stop_first_is_a_loss(self):
        fh = np.array([101.0, 103.5])       # fh>=103 on bar 1
        fl = np.array([100.5, 101.0])
        fc = np.array([101.0, 103.2])
        ret, first, _ = vp._reversion_outcome(fh, fl, fc, 102.0, 100.0, 98.0,
                                              103.0, -1, 'barrier')
        assert first == -1
        assert ret == pytest.approx(-0.03)  # short 100->103 = -3%

    def test_both_barriers_same_bar_booked_adverse(self):
        # one bar spans both 98 and 103 -> stop-first by convention
        fh = np.array([103.5])
        fl = np.array([97.5])
        fc = np.array([100.0])
        _, first, _ = vp._reversion_outcome(fh, fl, fc, 100.0, 100.0, 98.0,
                                            103.0, -1, 'barrier')
        assert first == -1

    def test_neither_hits_times_out_at_eod(self):
        fh = np.array([100.4, 100.6])
        fl = np.array([99.6, 99.5])
        fc = np.array([100.1, 100.2])
        ret, first, _ = vp._reversion_outcome(fh, fl, fc, 100.2, 100.0, 98.0,
                                              103.0, -1, 'barrier')
        assert first == 0
        assert ret == pytest.approx(-(100.2 / 100.0 - 1))  # short, eod above

    def test_long_mirror(self):
        # long from 100, target 102 (POC above), stop 97
        fh = np.array([100.5, 102.4])
        fl = np.array([99.8, 101.0])
        fc = np.array([100.2, 102.1])
        ret, first, _ = vp._reversion_outcome(fh, fl, fc, 101.5, 100.0, 102.0,
                                              97.0, 1, 'barrier')
        assert first == 1
        assert ret == pytest.approx(0.02)

    def test_touch_fill_uses_crossing_close(self):
        fh = np.array([100.5, 99.0])
        fl = np.array([99.5, 97.5])         # target 98 crossed on bar 1
        fc = np.array([100.0, 97.8])        # touch fills at 97.8, not 98
        ret_b, _, _ = vp._reversion_outcome(fh, fl, fc, 98.0, 100.0, 98.0,
                                            103.0, -1, 'barrier')
        ret_t, _, _ = vp._reversion_outcome(fh, fl, fc, 98.0, 100.0, 98.0,
                                            103.0, -1, 'touch')
        assert ret_b == pytest.approx(0.02)              # exact level
        assert ret_t == pytest.approx(-(97.8 / 100 - 1)) # better fill for a short


class TestContinuationDirection:
    """The opposite-side bet: flip the side and swap target<->stop."""

    def test_key_encodes_direction(self):
        rev = vp.ProfileCfg('developing', 0, 48, 0.70, 1.0)
        assert rev.key() == 'developing_lb0_b48_va70_s1_hclose_md12'
        cont = dc.replace(rev, direction='continuation')
        assert cont.key() == 'developing_lb0_b48_va70_s1_hclose_md12_continuation'

    def test_make_event_flips_side_and_swaps_levels(self):
        rev = vp.ProfileCfg('developing', 0, 48, 0.70, 1.0)
        cont = dc.replace(rev, direction='continuation')
        # a short-at-the-top reversion (side -1, target=POC 98, stop 103)...
        e_rev = vp._make_event(rev, 0, '2020-01-01', 10, 100.0, -1, 98.0, 103.0)
        e_con = vp._make_event(cont, 0, '2020-01-01', 10, 100.0, -1, 98.0, 103.0)
        assert (e_rev.side, e_rev.target, e_rev.stop) == (-1, 98.0, 103.0)
        # ...becomes a long-the-breakout continuation (side +1, target/stop swapped)
        assert (e_con.side, e_con.target, e_con.stop) == (1, 103.0, 98.0)

    def test_continuation_return_is_negative_of_reversion_no_engulf(self):
        # a path where each bar tags at most one level -> exact sign mirror
        fh = np.array([100.5, 99.0, 98.5])
        fl = np.array([99.5, 98.5, 97.9])   # hits target 98 on bar 2, never 103
        fc = np.array([100.0, 98.8, 98.2])
        rev, _, _ = vp._reversion_outcome(fh, fl, fc, 98.2, 100.0, 98.0, 103.0,
                                          -1, 'barrier')
        con, _, _ = vp._reversion_outcome(fh, fl, fc, 98.2, 100.0, 103.0, 98.0,
                                          1, 'barrier')
        assert con == pytest.approx(-rev)

    def test_engulf_bar_breaks_the_mirror_both_hit_stop(self):
        # one bar spans BOTH levels -> each direction books its own stop, at
        # different prices, so the returns are NOT exact negatives
        fh = np.array([103.5])
        fl = np.array([97.5])
        fc = np.array([100.0])
        rev, fr, _ = vp._reversion_outcome(fh, fl, fc, 100.0, 100.0, 98.0, 103.0,
                                           -1, 'barrier')
        con, fc_, _ = vp._reversion_outcome(fh, fl, fc, 100.0, 100.0, 103.0, 98.0,
                                            1, 'barrier')
        assert fr == -1 and fc_ == -1          # both hit their stop
        assert rev < 0 and con < 0             # both lose
        assert con != pytest.approx(-rev)      # mirror broken


class TestForwardPath:
    def _bars(self):
        rng = np.random.default_rng(3)
        P = 100 + np.cumsum(rng.standard_normal((8, SESSION_BARS)) * 0.05, axis=1)
        V = rng.random((8, SESSION_BARS)) + 0.1
        return _bars_from(P, V, [f'2020-03-0{d+1}' for d in range(8)])

    def test_intraday_forward_is_rest_of_session(self):
        bars = self._bars()
        daily = vp._daily_hlc(bars)
        _fh, _fl, fc, final = vp._forward(bars, daily, 2, 10, 'close')
        assert len(fc) == SESSION_BARS - 11        # bars 11..77
        assert final == pytest.approx(bars['close'][2, -1])

    def test_multiday_forward_stitches_entry_tail_then_daily(self):
        bars = self._bars()
        daily = vp._daily_hlc(bars)
        _fh, _fl, fc, final = vp._forward(bars, daily, 1, 30, 3)  # 3 trading days
        # the entry session's remaining bars (31..77) PLUS 3 daily bars
        assert len(fc) == (SESSION_BARS - 31) + 3
        assert final == pytest.approx(daily[2][1 + 3])          # close of day s+3

    def test_multiday_forward_includes_entry_day_afternoon(self):
        # the fix: a stop-out on the entry-day afternoon must be visible in the
        # forward path, not silently dropped
        bars = self._bars()
        bars['high'][1, 40] = 9_999.0     # a spike after a bar-30 signal
        daily = vp._daily_hlc(bars)
        fh, _, _, _ = vp._forward(bars, daily, 1, 30, 3)
        assert fh.max() >= 9_999.0        # the afternoon spike is monitored

    def test_multiday_forward_last_session_uses_entry_tail(self):
        bars = self._bars()               # 8 sessions
        daily = vp._daily_hlc(bars)
        # last session, no following days -> falls back to the entry-day tail
        _fh, _fl, fc, final = vp._forward(bars, daily, 7, 30, 10)
        assert len(fc) == SESSION_BARS - 31
        assert final == pytest.approx(bars['close'][7, -1])
        # a bar with no remaining tail returns None
        assert vp._forward(bars, daily, 7, SESSION_BARS - 1, 10) is None


# --------------------------------------------------- measurement wiring

class TestMeasureWiring:
    def _mean_reverting_tape(self, nd=250):
        rng = np.random.default_rng(7)
        rows = []
        vols = []
        for _ in range(nd):
            # an Ornstein-Uhlenbeck-ish intraday path so edges get reached
            x = np.zeros(SESSION_BARS)
            x[0] = rng.standard_normal()
            for t in range(1, SESSION_BARS):
                x[t] = 0.9 * x[t - 1] + rng.standard_normal() * 0.4
            rows.append(100 + x)
            vols.append(rng.random(SESSION_BARS) + 0.1)
        dates = [f'2020-{1 + m // 28:02d}-{1 + m % 28:02d}' for m in range(nd)]
        return _bars_from(np.array(rows), np.array(vols), dates)

    def test_measure_cell_wellformed(self):
        bars = self._mean_reverting_tape()
        cfg = vp.ProfileCfg('developing', 0, 48, 0.70, 1.0, horizon='close')
        rng = np.random.default_rng(vp.VP_SEED)
        out = vp.measure_cell(bars, cfg, '2020-01-01', None, 'touch', rng)
        assert out['cell'] == cfg.key()
        if out['n'] >= 20:
            for k in ('excess_bp', 'excess_p', 'hac_t', 'poc_first_rate',
                      'control_poc_first_rate', 'predicted_sign'):
                assert k in out
            assert out['predicted_sign'] == 1

    def test_run_grid_tolerates_too_few_events(self):
        # a tiny tape starves the cells; None p-values must not break BY
        bars = self._mean_reverting_tape(nd=30)
        res = vp.run_grid(bars, '2020-01-01', None, 'touch')
        assert len(res['cells']) == len(vp.GRID)
        assert isinstance(res['survivors'], list)
        for c in res['cells']:
            assert 'by_survivor' in c

    def test_cell_key_encodes_horizon_and_min_dev(self):
        assert vp.ProfileCfg('prior', 20, 48, 0.70, 1.0, horizon=10).key() \
            == 'prior_lb20_b48_va70_s1_h10d'
        # min_dev_bars distinguishes the two developing cells (no key collision)
        assert vp.ProfileCfg('developing', 0, 48, 0.70, 1.0).key() \
            == 'developing_lb0_b48_va70_s1_hclose_md12'
        assert vp.ProfileCfg('developing', 0, 48, 0.70, 1.0, min_dev_bars=24).key() \
            == 'developing_lb0_b48_va70_s1_hclose_md24'
        assert len({c.key() for c in vp.GRID}) == len(vp.GRID)  # all distinct


# --------------------------------------------------- dataset-gated NVDA pin

@pytest.mark.skipif(not _HAVE_NVDA,
                    reason='needs the local NVDA 1-minute archive')
class TestNvdaVolumeProfileReversal:
    """Pins the decisive NVDA measurement — EXPLORATORY, not a registered
    verdict. The reversal is not an edge: 0 of 4 grid cells survive
    Benjamini-Yekutieli, and the developing intraday fade is significantly
    WRONG-SIGNED on realistic (touch) fills."""

    @pytest.fixture(scope='class')
    def bars(self):
        return vp._load_nvda()

    @pytest.fixture(scope='class')
    def full(self, bars):
        return vp.run_grid(bars, vp.ERA_START, None, 'touch')

    def test_no_survivors(self, full):
        assert full['survivors'] == []
        assert all(c['by_survivor'] is False for c in full['cells'])

    def test_cell_counts_and_keys(self, full):
        cells = {c['cell']: c for c in full['cells']}
        assert set(cells) == {
            'prior_lb20_b48_va70_s1_h10d', 'prior_lb10_b48_va70_s1_h5d',
            'developing_lb0_b48_va70_s1_hclose_md12',
            'developing_lb0_b48_va70_s1_hclose_md24'}
        assert cells['prior_lb20_b48_va70_s1_h10d']['n'] == 1671
        assert cells['prior_lb10_b48_va70_s1_h5d']['n'] == 1863
        assert cells['developing_lb0_b48_va70_s1_hclose_md12']['n'] == 3992
        assert cells['developing_lb0_b48_va70_s1_hclose_md24']['n'] == 3430

    def test_developing_fade_is_significantly_wrong_signed(self, full):
        dev = [c for c in full['cells'] if c['cell'].startswith('developing')]
        # both developing cells lose to the matched random-entry control
        assert dev[0]['excess_bp'] < 0 and dev[1]['excess_bp'] < 0
        assert dev[0]['hac_t'] < -1.5                 # ~ -2.02
        assert dev[0]['excess_p'] > 0.95              # one-sided: excess < 0

    def test_reversion_race_real_but_empty(self, full):
        # the event reaches POC before its stop slightly MORE than the control
        # (a real reversion signature) yet loses money — high hit-rate, negative
        # expectancy
        dev = next(c for c in full['cells'] if c['cell'].startswith('developing'))
        assert dev['poc_first_rate'] > dev['control_poc_first_rate']
        assert dev['excess_bp'] < 0

    def test_prior_window_is_null(self, full):
        cells = {c['cell']: c for c in full['cells']}
        a = cells['prior_lb20_b48_va70_s1_h10d']
        assert a['excess_p'] > 0.10 and abs(a['hac_t']) < 1.0

    # --- the opposite side (continuation / breakout): an in-sample mirage ---

    @pytest.fixture(scope='class')
    def cont_full(self, bars):
        cells = []
        for cfg in vp.GRID:
            c = dc.replace(cfg, direction='continuation')
            rng = np.random.default_rng(vp.VP_SEED ^ zlib.crc32(c.key().encode()))
            cells.append(vp.measure_cell(bars, c, vp.ERA_START, None, 'touch', rng))
        flags = vp.benjamini_yekutieli([c.get('excess_p') for c in cells],
                                       q=vp.FDR_Q)
        return cells, flags

    def test_opposite_side_flags_in_sample(self, cont_full):
        # flipping the losing fade LOOKS like a winner on the same data — the
        # developing continuation cells clear the false-discovery gate
        cells, flags = cont_full
        dev = [(c, f) for c, f in zip(cells, flags)
               if c['cell'].startswith('developing')]
        assert any(f for _, f in dev)
        assert dev[0][0]['hac_t'] > 1.5

    def test_opposite_side_dies_out_of_sample(self, bars):
        # ...but on the held-out span it never saw, the "edge" is flat. The
        # in-sample significance was the same-data mirror of the reversion
        # loss, not a stable breakout edge.
        c = dc.replace(vp.GRID[2], direction='continuation')  # developing md12
        rng = np.random.default_rng(vp.VP_SEED ^ zlib.crc32(c.key().encode()))
        h = vp.measure_cell(bars, c, vp.HOLDOUT_SPLIT, None, 'touch', rng)
        assert h['excess_p'] > 0.10 and abs(h['hac_t']) < 1.0


@pytest.mark.skipif(not os.path.exists(_UNIV_PATH),
                    reason='needs the committed universe summary')
class TestVolumeProfileUniverse:
    """Pins the cross-sectional run over all 915 names (S&P 500 + Nasdaq-100 +
    S&P 400) from the committed summary — no recompute. EXPLORATORY. Both
    experiments produce REAL, out-of-sample-stable, cross-sectionally
    consistent effects whose signs flip by horizon — and both are economically
    negligible; the double-digit t-stats are an artifact of n in the millions."""

    @pytest.fixture(scope='class')
    def U(self):
        with open(_UNIV_PATH) as fh:
            return json.load(fh)

    def _cell(self, U, pool, cellsub, direction, era):
        for k, v in U['pools'][pool]['cells'].items():
            cell, d, e = k.split('@@')
            if e == era and d == direction and cellsub in cell:
                return v
        raise KeyError((pool, cellsub, direction, era))

    def test_universe_size(self, U):
        assert U['n_tickers'] == 915
        assert U['counts'].get('ok', 0) + U['counts'].get('skip', 0) >= 912

    def test_intraday_fade_loses_market_wide_but_negligibly(self, U):
        # developing (intraday): reversion is NEGATIVE everywhere and OOS-stable
        # — fading the edge loses market-wide — but only ~0.4bp, far below cost
        for pool in ('ALL', 'sp500', 'nasdaq100', 'sp400'):
            full = self._cell(U, pool, 'hclose_md12', 'reversion', 'full')
            hold = self._cell(U, pool, 'hclose_md12', 'reversion', 'holdout')
            assert full['excess_bp'] < 0 and hold['excess_bp'] < 0   # sign-stable
            assert abs(full['excess_bp']) < 1.0                      # negligible

    def test_intraday_continuation_is_the_mirror(self, U):
        rev = self._cell(U, 'ALL', 'hclose_md12', 'reversion', 'full')
        con = self._cell(U, 'ALL', 'hclose_md12_continuation', 'continuation', 'full')
        assert con['excess_bp'] > 0 > rev['excess_bp']

    def test_multiday_fade_wins_and_strengthens_out_of_sample(self, U):
        # prior-window (multi-day): reversion is POSITIVE and OOS-stable — the
        # OPPOSITE sign to the intraday horizon — and holds up in-to-out
        for pool in ('ALL', 'sp500', 'sp400'):
            full = self._cell(U, pool, 'prior_lb20', 'reversion', 'full')
            hold = self._cell(U, pool, 'prior_lb20', 'reversion', 'holdout')
            assert full['excess_bp'] > 0 and hold['excess_bp'] > 0

    def test_multiday_reversion_strongest_in_midcaps(self, U):
        allh = self._cell(U, 'ALL', 'prior_lb20', 'reversion', 'holdout')
        s4h = self._cell(U, 'sp400', 'prior_lb20', 'reversion', 'holdout')
        assert s4h['excess_bp'] > allh['excess_bp']    # S&P 400 mid-caps carry it

    def test_huge_n_manufactures_significance(self, U):
        # millions of events -> |t| in double digits on a sub-basis-point mean
        c = self._cell(U, 'ALL', 'hclose_md12', 'reversion', 'full')
        assert c['n'] > 1_000_000
        assert abs(c['hac_t']) > 5.0 and abs(c['excess_bp']) < 1.0
