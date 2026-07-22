"""Tests for engine/intraday_continuation.py — the breakout-continuation scout.

Three layers:

- The DETECTOR battery (always runs). Every feature is asserted
  look-ahead free by mutation: change a bar AFTER the signal and no
  feature at the signal may move. Every screen condition is asserted to
  reject in isolation, so a passing event is passing for the stated
  reason rather than by accident.
- The ESTIMATOR battery (always runs). ``two_way_excess`` is calibrated
  both ways on synthetic panels: it must return ZERO when the events
  carry no effect, and must RECOVER a known effect when one is injected.
  That pair is what licenses reading the real curve, and it covers every
  horizon column, which the data placebo (close-horizon only) cannot.
- The RESULT pins. The committed ``intraday_continuation_results.json``
  is pinned always-run so the log entry's prose cannot drift from the
  run that produced it; regenerating the panel itself is DATASET-GATED
  because the minute archive is personal and gitignored.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import pytest

from common.paths import data_path
from engine.intraday_continuation import (
    CHART_SCREEN,
    FIRST_BAR,
    HORIZONS,
    LAST_BAR,
    MIN_SESSION_BARS,
    N_CLOCK,
    RESULTS_FILE,
    SESSION_BARS,
    TRIGGER_MOM6,
    VOL_WINDOWS,
    _forward_extremes,
    _leave_one_out,
    bar_features,
    breakout_mask,
    ema,
    five_minute_bars,
    forward_paths,
    load_scan,
    prior_mean,
    prior_median_columns,
    realized_vol,
    run_scan,
    session_bootstrap,
    two_way_excess,
    wilder_rsi,
)

ARCHIVE_TICKERS = ['NVDA', 'AAPL', 'MSFT']


# ------------------------------------------------------------- fixtures

def write_minutes(path, rows):
    """rows: (timestamp, open, high, low, close, volume)."""
    with open(path, 'w') as f:
        f.write('timestamp,open,high,low,close,volume\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    return str(path)


def bars_from(path, **kw):
    """``five_minute_bars`` for the cases that must produce bars.

    It returns None when the ruling tables clip a ticker away, which has
    its own test; everywhere else a None would just be a crash three
    lines later, so unwrap it here and keep the call sites readable.
    """
    b = five_minute_bars(path, **kw)
    assert b is not None
    return b


def session_rows(date, closes, volume: float | Sequence[float] = 1000.0,
                 start_minute=570):
    """One 1-minute row per close, walking forward from 09:30.

    ``volume`` may be a scalar or a per-minute sequence.
    """
    vols = ([float(volume)] * len(closes)
            if isinstance(volume, (int, float)) else list(volume))
    out = []
    for k, (c, v) in enumerate(zip(closes, vols)):
        m = start_minute + k
        out.append((f'{date} {m // 60:02d}:{m % 60:02d}:00', c, c, c, c, v))
    return out


# ---------------------------------------------------- the 5-minute matrix

class TestFiveMinuteBars:
    def test_aggregates_one_minute_rows_into_five_minute_bars(self, tmp_path):
        rows = session_rows('2020-01-02', [10, 11, 9, 12, 10.5])
        p = write_minutes(tmp_path / 'a.csv', rows)
        b = bars_from(p, start='2020-01-01')
        assert b['open'][0, 0] == 10          # first minute's open
        assert b['high'][0, 0] == 12          # highest across the five
        assert b['low'][0, 0] == 9
        assert b['close'][0, 0] == 10.5       # last minute's close
        assert b['volume'][0, 0] == 5000.0

    def test_duplicate_timestamps_take_the_last_row(self, tmp_path):
        rows = session_rows('2020-01-02', [10, 11, 12, 13, 14])
        rows.append(('2020-01-02 09:34:00', 99, 99, 99, 77, 1.0))
        p = write_minutes(tmp_path / 'a.csv', rows)
        b = bars_from(p, start='2020-01-01')
        assert b['close'][0, 0] == 77

    def test_missing_bar_inherits_the_prior_close_at_zero_volume(self, tmp_path):
        rows = (session_rows('2020-01-02', [10] * 5)
                + session_rows('2020-01-02', [12] * 5, start_minute=580))
        p = write_minutes(tmp_path / 'a.csv', rows)
        b = bars_from(p, start='2020-01-01')
        assert b['close'][0, 1] == 10          # the 09:35 bar had no print
        assert b['volume'][0, 1] == 0.0
        assert b['close'][0, 2] == 12
        assert b['n_bars'][0] == 2             # only two bars really traded

    def test_rows_outside_the_regular_session_are_dropped(self, tmp_path):
        rows = ([('2020-01-02 09:20:00', 1, 1, 1, 1, 5.0)]
                + session_rows('2020-01-02', [10] * 5)
                + [('2020-01-02 16:05:00', 9, 9, 9, 9, 5.0)])
        p = write_minutes(tmp_path / 'a.csv', rows)
        b = bars_from(p, start='2020-01-01')
        assert b['n_bars'][0] == 1
        assert b['volume'][0].sum() == 5000.0

    def test_start_clip_and_drop_windows_are_applied(self, tmp_path):
        rows = (session_rows('2020-01-02', [10] * 5)
                + session_rows('2020-01-03', [11] * 5)
                + session_rows('2020-01-06', [12] * 5))
        p = write_minutes(tmp_path / 'a.csv', rows)
        assert len(bars_from(p, start='2020-01-03')['dates']) == 2
        kept = bars_from(p, start='2020-01-01',
                         drops=[('2020-01-03', '2020-01-03')])
        assert kept['dates'].tolist() == ['2020-01-02', '2020-01-06']

    def test_returns_none_when_the_rulings_clip_everything(self, tmp_path):
        p = write_minutes(tmp_path / 'a.csv', session_rows('2020-01-02', [10] * 5))
        assert five_minute_bars(p, start='2021-01-01') is None


# ------------------------------------------------------------ indicators

class TestContinuationIndicators:
    def test_rsi_is_one_hundred_on_a_pure_advance(self):
        r = wilder_rsi(np.arange(1, 60, dtype=float))
        assert r[30] == pytest.approx(100.0)

    def test_rsi_warmup_is_undefined(self):
        r = wilder_rsi(np.arange(1, 60, dtype=float))
        assert np.isnan(r[:14]).all()
        assert np.isfinite(r[14])

    def test_rsi_straddles_fifty_on_a_symmetric_zigzag(self):
        """A strict zigzag never settles — it alternates about 50.

        Each bar is the last move's turn to dominate the smoothed average,
        so the series oscillates (here 48.15 / 51.85). Symmetry is a claim
        about the PAIR, and asserting it on one bar would be asserting the
        phase instead.
        """
        x = np.array([100.0 + (1 if k % 2 else 0) for k in range(600)])
        r = wilder_rsi(x)
        assert r[-2:].mean() == pytest.approx(50.0, abs=0.01)
        assert r[-1] != pytest.approx(r[-2], abs=1.0)

    def test_ema_converges_to_a_constant_series(self):
        assert ema(np.full(200, 7.0), 50)[-1] == pytest.approx(7.0)

    def test_prior_mean_excludes_the_current_value(self):
        a = np.array([1.0, 1.0, 1.0, 100.0, 1.0])
        out = prior_mean(a, 3)
        assert out[3] == pytest.approx(1.0)      # the spike is not in its own mean
        assert out[4] == pytest.approx(34.0)     # it enters only afterwards

    def test_prior_median_columns_excludes_the_current_row(self):
        m = np.ones((8, 2))
        m[3:] = 50.0
        out = prior_median_columns(m, 3)
        assert out[3, 0] == pytest.approx(1.0)   # rows 0-2, the step excluded
        assert out[6, 0] == pytest.approx(50.0)  # rows 3-5, fully stepped


# -------------------------------------------------------------- detector

SESSION_MINUTES = 390
RAMP_START, RAMP_LEN = 200, 40


def build_breakout(seed=1, impulse=0.03, surge=40.0):
    """A quiet base, then a ramp that satisfies every screen condition.

    Twenty-plus warm-up sessions come first because ``sigma`` and the
    relative-volume median are both trailing-20-session statistics; a
    shorter tape would leave the screen undefined rather than unmet.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(40):
        date = f'2020-{1 + d // 28:02d}-{1 + d % 28:02d}'
        walk = 100 + np.cumsum(rng.normal(0, 0.005, SESSION_MINUTES))
        rows += session_rows(date, walk)

    base = list(100 + np.cumsum(rng.normal(0, 0.005, RAMP_START)))
    ramp = [base[-1] * (1 + impulse * (k + 1) / RAMP_LEN)
            for k in range(RAMP_LEN)]
    tail = [ramp[-1]] * (SESSION_MINUTES - RAMP_START - RAMP_LEN)
    closes = base + ramp + tail
    volume = ([1000.0] * RAMP_START + [1000.0 * surge] * RAMP_LEN
              + [1000.0] * len(tail))
    rows += session_rows('2020-03-02', closes, volume=volume)
    return rows


class TestBreakoutDetector:
    @pytest.fixture(scope='class')
    def fired(self, tmp_path_factory):
        p = write_minutes(tmp_path_factory.mktemp('d') / 'b.csv',
                          build_breakout())
        bars = bars_from(p, start='2020-01-01')
        feat = bar_features(bars)
        mask = breakout_mask(feat, CHART_SCREEN)
        sess, bar = np.nonzero(mask)
        assert len(sess), 'the constructed breakout must fire'
        return bars, feat, int(sess[0]), int(bar[0]) + FIRST_BAR

    def test_the_constructed_breakout_fires_every_condition(self, fired):
        _, feat, d, b = fired
        for key, threshold in CHART_SCREEN.items():
            assert feat[key][d, b] >= threshold, key

    @pytest.mark.parametrize('key', sorted(CHART_SCREEN))
    def test_each_screen_condition_rejects_in_isolation(self, fired, key):
        """Lift one threshold just past this event; this event must vanish.

        Without it a screen key could be inert — satisfied by every bar
        the base condition already admits — and the reported event set
        would not be the one the prose describes. Testing the specific
        firing bar rather than "any bar" keeps the assertion honest when
        several bars fire.
        """
        _, feat, d, b = fired
        blocked = dict(CHART_SCREEN)
        blocked[key] = float(feat[key][d, b]) + abs(float(feat[key][d, b])) * 0.1 + 1e-6
        assert not breakout_mask(feat, blocked)[d, b - FIRST_BAR]

    def test_features_are_look_ahead_free(self, fired):
        """Mutate a bar AFTER the signal; nothing at the signal may move."""
        bars, feat, d, b = fired
        assert b + 3 < SESSION_BARS
        tampered = {k: v.copy() for k, v in bars.items()}
        for k in ('open', 'high', 'low', 'close'):
            tampered[k][d, b + 1:] *= 1.5
        tampered['volume'][d, b + 1:] *= 9.0
        after = bar_features(tampered)
        for k in ('mom6', 'brk2h', 'brkday', 'rvol', 'ext50', 'rsi', 'z6'):
            assert after[k][d, b] == pytest.approx(feat[k][d, b], rel=1e-12), k

    def test_a_dead_clock_window_gives_undefined_relative_volume(self, tmp_path):
        """Zero trailing volume must read as unknown, not as a huge surge.

        Dividing by a zero median yields ``inf``, which would be the most
        extreme relative-volume reading in the sample and would sail past
        any ``rvol`` threshold. It has to be NaN so the finiteness guard
        drops the bar.
        """
        rows = []
        for d in range(30):
            date = f'2020-{1 + d // 28:02d}-{1 + d % 28:02d}'
            closes = list(100 + np.zeros(SESSION_MINUTES))
            vols = [0.0] * 60 + [1000.0] * (SESSION_MINUTES - 60)
            rows += session_rows(date, closes, volume=vols)
        p = write_minutes(tmp_path / 'q.csv', rows)
        feat = bar_features(bars_from(p, start='2020-01-01'))
        dead = feat['rvol'][-1, :12]           # the never-traded window
        assert np.isnan(dead).all()
        assert not np.isinf(feat['rvol']).any()

    def test_the_base_condition_needs_a_fresh_two_hour_high(self, fired):
        _, feat, _, _ = fired
        sess, bar = np.nonzero(breakout_mask(feat))
        b = bar + FIRST_BAR
        assert (feat['brk2h'][sess, b] >= 0).all()
        assert (feat['mom6'][sess, b] >= TRIGGER_MOM6).all()


# --------------------------------------------------------------- outcomes

class TestBreakoutOutcomes:
    def test_forward_paths_are_nan_past_the_close(self):
        close = np.ones((1, SESSION_BARS))
        p = forward_paths(close, np.array([SESSION_BARS]))
        last_clock = LAST_BAR - FIRST_BAR
        far = HORIZONS.index(71)
        assert np.isnan(p[0, last_clock, far])
        assert np.isfinite(p[0, 0, far])

    def test_forward_paths_measure_from_the_signal_bar(self):
        close = np.arange(SESSION_BARS, dtype=float)[None, :] + 100.0
        p = forward_paths(close, np.array([SESSION_BARS]))
        assert p[0, 0, 0] == pytest.approx(
            close[0, FIRST_BAR + 1] / close[0, FIRST_BAR] - 1)

    def test_thin_sessions_are_excluded(self):
        close = np.ones((1, SESSION_BARS))
        p = forward_paths(close, np.array([MIN_SESSION_BARS - 1]))
        assert np.isnan(p).all()

    def test_realized_vol_matches_a_known_constant_move(self):
        step = np.exp(0.001)
        close = 100.0 * step ** np.arange(SESSION_BARS)
        v = realized_vol(close[None, :], np.array([SESSION_BARS]))
        # six bars of a constant 0.001 log move: sqrt(6 * 0.001**2)
        assert v[0, 0, 0] == pytest.approx(np.sqrt(6) * 0.001 * 1e4, rel=1e-6)

    def test_first_touch_resolves_up_down_and_neither(self):
        n = SESSION_BARS
        close = np.full((3, n), 100.0)
        high = np.full((3, n), 100.0)
        low = np.full((3, n), 100.0)
        high[0, FIRST_BAR + 2] = 101.0          # up first
        low[1, FIRST_BAR + 2] = 99.0            # down first
        sess = np.array([0, 1, 2])
        bar = np.array([FIRST_BAR] * 3)
        *_, touch = _forward_extremes(high, low, close, sess, bar)
        assert touch[0.005].tolist() == [1, -1, 0]

    def test_a_bar_spanning_both_barriers_counts_against_the_trade(self):
        n = SESSION_BARS
        close = np.full((1, n), 100.0)
        high = np.full((1, n), 100.0)
        low = np.full((1, n), 100.0)
        high[0, FIRST_BAR + 1] = 101.0
        low[0, FIRST_BAR + 1] = 99.0
        *_, touch = _forward_extremes(high, low, close, np.array([0]),
                                      np.array([FIRST_BAR]))
        assert touch[0.005][0] == -1

    def test_extreme_timing_is_bars_since_the_signal(self):
        n = SESSION_BARS
        close = np.full((1, n), 100.0)
        high = np.full((1, n), 100.0)
        low = np.full((1, n), 100.0)
        high[0, FIRST_BAR + 7] = 105.0
        low[0, FIRST_BAR + 3] = 95.0
        mfe, mae, h_mfe, h_mae, _ = _forward_extremes(
            high, low, close, np.array([0]), np.array([FIRST_BAR]))
        assert h_mfe[0] == 7 and h_mae[0] == 3
        assert mfe[0] == pytest.approx(0.05)
        assert mae[0] == pytest.approx(-0.05)


# -------------------------------------------------------------- estimator

def synthetic_panel(n_tickers=40, n_dates=180, n_clock=N_CLOCK, k=3,
                    effect=0.0, session_lift=0.0, seed=5):
    """A panel with additive name, session and clock effects plus noise.

    Events are sparse — one per session — so leave-one-out leakage through
    the name margin stays negligible and a recovered effect can be
    asserted tightly. Both ``effect`` (on the event cells only) and
    ``session_lift`` (on every name in every other session) are written
    into the cube BEFORE the margins are derived from it, which is what
    makes the test faithful: in the real panel the margins always contain
    the events they are being used to judge.
    """
    rng = np.random.default_rng(seed)
    name = rng.normal(0, 0.004, (n_tickers, n_clock, k))
    day = rng.normal(0, 0.006, (n_dates, n_clock, k))
    clock = rng.normal(0, 0.003, (n_clock, k))
    cube = (name[:, None] + day[None] + clock[None, None]
            + rng.normal(0, 0.01, (n_tickers, n_dates, n_clock, k)))
    ev_ticker = rng.integers(0, n_tickers, n_dates)
    ev_date = np.arange(n_dates)
    ev_clock = rng.integers(0, n_clock, n_dates)
    if session_lift:
        cube[:, ::2] += session_lift          # a market-wide up session
    cube[ev_ticker, ev_date, ev_clock] += effect
    name_sum, name_n = cube.sum(axis=1), np.full(
        (n_tickers, n_clock, k), n_dates, np.int64)
    day_sum, day_n = cube.sum(axis=0), np.full(
        (n_dates, n_clock, k), n_tickers, np.int64)
    y = cube[ev_ticker, ev_date, ev_clock]
    return dict(y=y, ticker=ev_ticker, date=ev_date,
                clock=ev_clock + FIRST_BAR, name_sum=name_sum, name_n=name_n,
                day_sum=day_sum, day_n=day_n)


class TestTwoWayExcess:
    def test_returns_zero_when_the_events_carry_no_effect(self):
        p = synthetic_panel(effect=0.0)
        ex = two_way_excess(p['y'], p['ticker'], p['date'], p['clock'],
                            p['name_sum'], p['name_n'], p['day_sum'], p['day_n'])
        for col in range(ex.shape[1]):
            b = session_bootstrap(ex[:, col], p['date'], reps=400)
            assert abs(b['mean']) < 0.0015
            assert b['lo'] < 0 < b['hi']

    def test_recovers_a_known_injected_effect(self):
        delta = 0.02
        p = synthetic_panel(effect=delta)
        ex = two_way_excess(p['y'], p['ticker'], p['date'], p['clock'],
                            p['name_sum'], p['name_n'], p['day_sum'], p['day_n'])
        assert ex.mean() == pytest.approx(delta, rel=0.05)

    def test_strips_a_pure_market_move(self):
        """A session-wide lift must not read as an event effect.

        This is the whole reason the estimator exists: the raw outcome on
        such a session is strongly positive and means nothing.
        """
        p = synthetic_panel(effect=0.0, session_lift=0.05)
        lifted = p['date'] % 2 == 0
        assert p['y'][lifted].mean() > 0.04    # the raw outcome looks superb
        ex = two_way_excess(p['y'], p['ticker'], p['date'], p['clock'],
                            p['name_sum'], p['name_n'], p['day_sum'], p['day_n'])
        assert abs(ex.mean()) < 0.002          # ... and the excess is nothing

    def test_leave_one_out_removes_the_observation_from_its_own_margin(self):
        total = np.array([10.0])
        count = np.array([5])
        y = np.array([2.0])
        assert _leave_one_out(total, count, y)[0] == pytest.approx(2.0)

    def test_leave_one_out_is_undefined_for_a_singleton_cell(self):
        out = _leave_one_out(np.array([2.0]), np.array([1]), np.array([2.0]))
        assert np.isnan(out[0])


class TestSessionBootstrap:
    def test_is_deterministic_under_the_seed(self):
        v = np.random.default_rng(0).normal(0, 1, 500)
        s = np.repeat(np.arange(50), 10)
        assert session_bootstrap(v, s) == session_bootstrap(v, s)

    def test_detects_a_strong_effect_and_not_a_null(self):
        rng = np.random.default_rng(2)
        s = np.repeat(np.arange(200), 5)
        assert session_bootstrap(rng.normal(1.0, 1.0, 1000), s)['p'] < 0.01
        null = rng.normal(0.0, 1.0, 1000)
        null -= null.mean()          # exactly zero-mean, so p straddles a half
        assert 0.3 < session_bootstrap(null, s)['p'] < 0.7

    def test_clustering_widens_the_interval(self):
        """1000 events over 5 sessions must not read like 1000 draws.

        The widening comes from CORRELATION inside a session, which is the
        real situation: one market-wide rip moves every name that fired on
        it, so those events are near-duplicates rather than fresh draws.
        Given the same underlying shocks, splitting them across 200
        sessions is genuinely more information than concentrating them in
        five, and the interval has to say so.
        """
        rng = np.random.default_rng(3)

        def clustered(n_sessions, per_session):
            shock = rng.normal(0.2, 1.0, n_sessions)      # the session's move
            v = np.repeat(shock, per_session) + rng.normal(
                0, 0.05, n_sessions * per_session)        # small idiosyncratic
            return v, np.repeat(np.arange(n_sessions), per_session)

        wide = session_bootstrap(*clustered(5, 200))
        tight = session_bootstrap(*clustered(200, 5))
        assert (wide['hi'] - wide['lo']) > 3 * (tight['hi'] - tight['lo'])

    def test_reports_both_counts_so_neither_can_be_mistaken(self):
        b = session_bootstrap(np.ones(50), np.repeat(np.arange(5), 10))
        assert b['n'] == 50 and b['n_sessions'] == 5


# ------------------------------------------------------- committed result

class TestContinuationResults:
    """Pins the published run so the log entry's prose cannot drift.

    Regenerating this file needs the archive (see the dataset-gated class);
    reading it does not, so these assertions run everywhere.
    """

    @pytest.fixture(scope='class')
    def results(self):
        p = data_path(RESULTS_FILE)
        if not os.path.exists(p):
            pytest.skip(f'{RESULTS_FILE} not committed yet')
        with open(p) as f:
            return json.load(f)

    def test_the_direction_is_null_at_the_close(self, results):
        s = results['summary']
        assert s['excess_bp'] < 0
        assert s['ci_bp'][0] < s['excess_bp'] < s['ci_bp'][1]
        assert s['p'] > 0.5          # nowhere near significant on the up side

    def test_the_breakout_underperforms_its_own_session_peers(self, results):
        s = results['summary']
        assert s['p_close_up'] < s['p_peer_up']

    def test_the_placebo_is_flat(self, results):
        mean, sd = results['placebo']
        assert abs(mean) < 1.0

    def test_the_decay_curve_is_negative_from_the_first_bar(self, results):
        """Negative on the very next bar, then worse for about an hour.

        There is no global trough to pin: the curve falls fast for the
        first hour and then sits on a flat plateau, so the single most
        negative horizon is not a stable feature. What IS stable is the
        shape — immediately negative, materially worse by an hour, and
        never recovering to positive.
        """
        curve = {r['minutes']: r['excess_bp'] for r in results['horizons']}
        assert curve[5] < 0
        assert curve[60] < curve[5]
        early = min(m for m in curve if m <= 120)
        assert curve[60] <= min(v for m, v in curve.items() if m <= 30) + 1e-9
        assert early == 5
        assert max(curve.values()) < 0

    def test_volatility_expands_and_then_decays(self, results):
        vols = {r['window']: r for r in results['volatility']}
        assert vols['0-30m']['ratio_own'] > 2.0
        assert (vols['0-30m']['ratio_own'] > vols['60-90m']['ratio_own']
                > vols['4-5h']['ratio_own'])
        assert vols['4-5h']['ratio_own'] > 1.0

    def test_the_panel_is_the_full_universe(self, results):
        s = results['summary']
        assert s['names'] > 400
        assert s['sessions'] > 2000


# ---------------------------------------------------------- dataset-gated

def _archives_present():
    from engine.intraday_continuation import archive_path
    return all(archive_path(t) is not None for t in ARCHIVE_TICKERS)


@pytest.mark.skipif(not _archives_present(),
                    reason='minute archive absent (personal, gitignored)')
class TestContinuationArchiveRoundTrip:
    """Proves the scan path runs on real tape.

    Deliberately NOT a headline pin: the session margin is a sum ACROSS
    the cross-section, so a three-ticker panel has a degenerate market
    control. The headline lives in the committed results file, produced
    by a full-universe run.
    """

    @pytest.fixture(scope='class')
    def panel(self, tmp_path_factory):
        out = str(tmp_path_factory.mktemp('scan'))
        run_scan(ARCHIVE_TICKERS, out)
        return load_scan(out)

    def test_every_requested_ticker_is_present(self, panel):
        assert sorted(panel['tickers'].tolist()) == sorted(ARCHIVE_TICKERS)

    def test_events_carry_every_recorded_column(self, panel):
        E = panel['events']
        for k in ('path', 'vol_path', 'mfe', 'mae', 'h_mfe', 'touch50',
                  'ret_eod', 'rsi', 'z6', 'rvol'):
            assert k in E and len(E[k]) == len(E['session'])
        assert E['path'].shape[1] == len(HORIZONS)
        assert E['vol_path'].shape[1] == len(VOL_WINDOWS)

    def test_the_base_condition_holds_for_every_event(self, panel):
        E = panel['events']
        assert (E['mom6'] >= TRIGGER_MOM6 - 1e-6).all()
        assert (E['brk2h'] >= -1e-9).all()

    def test_a_piecemeal_panel_is_refused(self, tmp_path):
        out = str(tmp_path / 'scan')
        run_scan(ARCHIVE_TICKERS[:2], out)
        # a stray ticker file with no matching margin entry must not load
        import shutil
        src = os.path.join(out, f'{ARCHIVE_TICKERS[0]}.npz')
        shutil.copy(src, os.path.join(out, 'ZZZZ.npz'))
        with pytest.raises(ValueError, match='margins disagree'):
            load_scan(out)
