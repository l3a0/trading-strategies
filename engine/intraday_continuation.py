"""The intraday breakout-continuation scout — when a large cap breaks out
of a base on a volume surge, does the rest of THAT SESSION keep going?

The question comes from a chart, not a model: a 5-minute candle chart where
price sat in a two-hour range, then ripped on heavy volume, leaving the
moving averages behind with RSI in the seventies. Every discretionary
trader has been asked to judge that bar. This module measures what
actually followed it, across the S&P 500 minute archive.

The DATA layer — archive access, split adjustment, the owner-signed
hygiene rulings — lives in ``pipeline/minute_archive.py`` and is consumed
here, never re-implemented (the same arrangement ``cup_handle_scan`` uses).

Layer map:

- DETECTOR — ``bar_features`` computes, at every 5-minute bar close and
  using only information available at that close, the five things the
  chart shows: the break of the trailing two-hour high (``brk2h``), the
  30-minute impulse (``mom6``) and its volatility-normalized twin
  (``z6``), the relative-volume surge (``rvol``), and the extension above
  the 50-bar EMA plus Wilder RSI. ``breakout_mask`` applies a screen;
  ``CHART_SCREEN`` is the vignette-matched one.
- OUTCOMES — ``scan_ticker`` records, per event, the forward return PATH
  over ``HORIZONS``, realized volatility over ``VOL_WINDOWS``, the
  symmetric first-touch verdicts, and when the session's forward extremes
  printed. Thresholds are NOT baked in: the features ride the event
  record, so tightening the screen is an analysis-time slice rather than
  a reason to re-read 10 GB.
- ESTIMATOR — ``two_way_excess``. A raw "it closed higher half the time"
  is uninterpretable, because the same tape that lifted this name lifted
  everything else. The excess subtracts the name's own average at that
  clock, the same session's move at that clock across every other name,
  and the clock's own drift, each leave-one-out:

      excess = y − mean[name, clock] − mean[session, clock] + mean[clock]

- INFERENCE — ``session_bootstrap``. A market-wide rip fires hundreds of
  names on the same bar, so the event count is not the sample size; the
  SESSION count is. Resampling is by calendar session, never by event.
- NULL CALIBRATION — ``placebo_excess`` keeps each event's name and clock
  but moves it to a random session, so the outcome is a real return with
  the signal scrambled out. The estimator must return zero on it. This
  runs on the return-to-close grid; unbiasedness at the intermediate
  horizons is pinned instead by the synthetic recovery tests, which
  inject a known effect and check it comes back.

This is an EXPLORATORY scout, not a registered experiment: it spends
sample it did not reserve, so it can kill an idea or justify taking one to
pre-registration, never deliver a confirmatory verdict. See
docs/explorations.md for the log entry and docs/prereg_trend_gate.md for
the line it must not cross.

The archive is personal and gitignored, so the result pins are
DATASET-GATED (skip in CI); the synthetic battery always runs.

Run:  python -m engine.intraday_continuation --scan [--tickers A,B,...]
      python -m engine.intraday_continuation --report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from common.paths import data_path
from pipeline.minute_archive import (
    TICKER_DROP_WINDOWS,
    TICKER_START_CLIPS,
    archive_path,
    universe,
)

# ---------------------------------------------------------------- geometry

SESSION_BARS = 78           # 09:30-16:00 inclusive, in 5-minute bars
FIRST_BAR = 6               # evaluate from 10:00 ...
LAST_BAR = 66               # ... to 15:00, so every event has room to run
N_CLOCK = LAST_BAR - FIRST_BAR + 1
BASE_LOOKBACK = 24          # the "consolidation" a breakout clears: 2 hours
VOL_LOOKBACK = 6            # relative volume measured over 30 minutes
SIGMA_SESSIONS = 20         # trailing sessions behind sigma and rvol medians
MIN_SESSION_BARS = 75       # drops half-days (13:00 close) and halted sessions

SCAN_START = '2010-01-01'      # the archive era this scout reads
HEADLINE_START = '2016-01-01'  # the era the log entry reports
CONTINUATION_SEED = 20260722

# The permissive scan trigger, deliberately looser than any screen we
# report so that tightening is a slice of the recorded events.
TRIGGER_MOM6 = 0.005

# The vignette screen — the chart that prompted the question, in numbers.
CHART_SCREEN: dict[str, float] = {
    'mom6': 0.015,      # +1.5% or better over the trailing 30 minutes
    'z6': 3.0,          # ... and at least 3 sigma for this name
    'rvol': 2.0,        # on twice the usual volume for that clock window
    'brkday': 0.0,      # printing a new session high
    'ext50': 0.01,      # extended >1% above the 50-bar EMA
    'rsi': 70.0,        # RSI(14) in the seventies
}

# forward horizons in 5-minute bars. 71 is the longest reachable from the
# earliest clock bar (6 -> 77), so no column is always-NaN.
HORIZONS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60, 71)
HORIZON_LABELS = tuple(f'{h * 5}m' for h in HORIZONS)
# realized-volatility windows, as (start, end] in bars after the signal
VOL_WINDOWS = ((0, 6), (6, 12), (12, 18), (18, 24), (24, 36), (36, 48),
               (48, 60), (0, 12), (0, 24), (0, 71))
VOL_LABELS = ('0-30m', '30-60m', '60-90m', '90-120m', '2-3h', '3-4h', '4-5h',
              '0-60m', '0-2h', 'to close')
TOUCH_LEVELS = (0.005, 0.010)

SCAN_DIR = 'intraday_continuation_scan'
MARGINS_FILE = '_margins.npz'
# The published run, committed so the log entry's prose is pinned by a
# test even though regenerating the panel needs the gitignored archive.
RESULTS_FILE = 'intraday_continuation_results.json'
EVENT_FEATURES = ('mom6', 'mom_open', 'brk2h', 'brkday', 'rvol', 'ext50',
                  'rsi', 'sigma', 'z6', 'dollar_volume')


# -------------------------------------------------------------- primitives

def _split_timestamps(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """('YYYY-MM-DD HH:MM:SS', ...) -> (date strings, minute-of-day).

    Fixed-width slicing through a ``U1`` view rather than ``.str``
    accessors: the archives run to several million rows per ticker and
    this is the scan's hot loop. The layout is fixed by the fetcher.
    """
    a = ts.astype('U19').view('U1').reshape(-1, 19)
    day = np.ascontiguousarray(a[:, :10]).view('U10').ravel()
    hh = np.ascontiguousarray(a[:, 11:13]).view('U2').ravel().astype(np.int16)
    mm = np.ascontiguousarray(a[:, 14:16]).view('U2').ravel().astype(np.int16)
    return day, hh * 60 + mm


def five_minute_bars(path: str, start: str = SCAN_START,
                     clip: str | None = None,
                     drops: Sequence[tuple[str, str]] = ()):
    """Regular-session 5-minute OHLCV as (n_sessions, 78) matrices.

    Duplicate timestamps collapse last-wins (the archive convention, see
    ``minute_archive.aggregate_daily``). A bar with no print inherits the
    previous close at zero volume so the matrix algebra is total; the real
    count rides ``n_bars`` so thin sessions can be filtered downstream.

    Returns ``None`` when the ruling tables clip everything away.
    """
    df = pd.read_csv(path, dtype={'timestamp': str})
    day, minute = _split_timestamps(df['timestamp'].to_numpy())
    keep = (minute >= 570) & (minute <= 960) & (day >= start)
    if clip:
        keep &= day >= clip
    for lo, hi in drops:
        keep &= (day < lo) | (day > hi)
    if not keep.any():
        return None
    day, minute = day[keep], minute[keep]
    o, h, lo_, c, v = (df[k].to_numpy()[keep].astype(np.float64)
                       for k in ('open', 'high', 'low', 'close', 'volume'))
    bar = np.minimum((minute - 570) // 5, SESSION_BARS - 1).astype(np.int16)

    dates, inv = np.unique(day, return_inverse=True)
    nd = len(dates)
    slot = inv.astype(np.int64) * SESSION_BARS + bar
    order = np.argsort(slot, kind='stable')
    slot, o, h, lo_, c, v = (x[order] for x in (slot, o, h, lo_, c, v))

    flat_o = np.full(nd * SESSION_BARS, np.nan)
    flat_c = np.full(nd * SESSION_BARS, np.nan)
    flat_h = np.full(nd * SESSION_BARS, -np.inf)
    flat_l = np.full(nd * SESSION_BARS, np.inf)
    flat_v = np.zeros(nd * SESSION_BARS)
    opens = np.ones(len(slot), bool)
    opens[1:] = slot[1:] != slot[:-1]
    closes = np.ones(len(slot), bool)
    closes[:-1] = slot[:-1] != slot[1:]
    flat_o[slot[opens]] = o[opens]
    flat_c[slot[closes]] = c[closes]
    np.maximum.at(flat_h, slot, h)
    np.minimum.at(flat_l, slot, lo_)
    np.add.at(flat_v, slot, v)

    Op, Hi, Lo, C, V = (x.reshape(nd, SESSION_BARS) for x in
                        (flat_o, flat_h, flat_l, flat_c, flat_v))
    real = np.isfinite(C)
    n_bars = real.sum(1)
    idx = np.maximum.accumulate(
        np.where(real, np.arange(SESSION_BARS)[None, :], -1), axis=1)
    rows = np.arange(nd)[:, None]
    filled = np.where(idx >= 0, C[rows, np.maximum(idx, 0)], np.nan)
    seed = C[rows, np.argmax(real, axis=1)[:, None]]     # a leading hole
    C = np.where(np.isnan(filled), seed, filled)
    return dict(dates=dates, open=np.where(real, Op, C),
                high=np.where(np.isfinite(Hi), Hi, C),
                low=np.where(np.isfinite(Lo), Lo, C),
                close=C, volume=V, n_bars=n_bars)


def wilder_rsi(x: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's RSI on a flat series; the first ``n`` values are NaN.

    Wilder smoothing rather than a simple moving average because that is
    what the charting packages draw, and the screen quotes a chart number.
    """
    d = np.diff(x, prepend=x[0])
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = np.empty_like(x), np.empty_like(x)
    au[:n], ad[:n] = np.nan, np.nan
    au[n], ad[n] = up[1:n + 1].mean(), dn[1:n + 1].mean()
    a = (n - 1) / n
    for i in range(n + 1, len(x)):
        au[i] = au[i - 1] * a + up[i] / n
        ad[i] = ad[i - 1] * a + dn[i] / n
    rs = np.divide(au, ad, out=np.full_like(x, np.inf), where=ad > 0)
    out = 100 - 100 / (1 + rs)
    # ``where=ad > 0`` leaves the inf initializer standing wherever the
    # seed averages are still NaN, which would report a warm-up bar as a
    # maximally overbought 100 and sail through an ``rsi >= 70`` screen.
    out[:n] = np.nan
    return out


def ema(x: np.ndarray, n: int) -> np.ndarray:
    k = 2 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = x[i] * k + out[i - 1] * (1 - k)
    return out


def prior_mean(a: np.ndarray, w: int) -> np.ndarray:
    """Mean of the ``w`` values STRICTLY BEFORE each position.

    Strictly-before is the point: every feature must be computable from
    the tape as it stood at the signal bar's close.
    """
    out = np.full(len(a), np.nan)
    cs = np.concatenate([[0.0], np.nancumsum(a)])
    if len(a) > w:
        out[w:] = (cs[w:-1] - cs[:-w - 1]) / w
    return out


def prior_median_columns(m: np.ndarray, w: int) -> np.ndarray:
    """Per-column median over the ``w`` rows strictly before each row."""
    out = np.full(m.shape, np.nan)
    for d in range(w, m.shape[0]):
        out[d] = np.median(m[d - w:d], axis=0)
    return out


# ---------------------------------------------------------------- detector

def bar_features(bars: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The chart's conditions at every bar close, all look-ahead free."""
    C, Hi, V, Op = bars['close'], bars['high'], bars['volume'], bars['open']
    nd = C.shape[0]
    flat = C.ravel()
    rsi = wilder_rsi(flat).reshape(nd, SESSION_BARS)
    e50 = ema(flat, 50).reshape(nd, SESSION_BARS)

    lr = np.diff(np.log(np.maximum(C, 1e-9)), axis=1)
    sigma = np.sqrt(prior_mean(np.nanvar(lr, axis=1), SIGMA_SESSIONS))

    cs = np.cumsum(np.concatenate([np.zeros((nd, 1)), V], axis=1), axis=1)
    vol_w = np.full((nd, SESSION_BARS), np.nan)
    vol_w[:, VOL_LOOKBACK - 1:] = (cs[:, VOL_LOOKBACK:]
                                   - cs[:, :SESSION_BARS - VOL_LOOKBACK + 1])
    # A clock window that traded nothing across the whole trailing sample
    # has no "usual" volume to compare against, so relative volume there
    # is UNDEFINED rather than infinite; ``breakout_mask``'s finiteness
    # guard then drops those bars instead of admitting them as the most
    # extreme surge on record.
    median_vol = prior_median_columns(vol_w, SIGMA_SESSIONS)
    rvol = np.divide(vol_w, median_vol, out=np.full_like(vol_w, np.nan),
                     where=median_vol > 0)

    hi2h = np.full((nd, SESSION_BARS), np.nan)
    for i in range(1, SESSION_BARS):
        hi2h[:, i] = Hi[:, max(0, i - BASE_LOOKBACK):i].max(axis=1)
    hiday = np.full((nd, SESSION_BARS), np.nan)
    hiday[:, 1:] = np.maximum.accumulate(Hi, axis=1)[:, :-1]

    mom6 = np.full((nd, SESSION_BARS), np.nan)
    mom6[:, 6:] = C[:, 6:] / C[:, :-6] - 1
    col = np.ones((1, SESSION_BARS))
    return dict(
        mom6=mom6, mom_open=C / Op[:, [0]] - 1,
        brk2h=C / hi2h - 1, brkday=C / hiday - 1,
        rvol=rvol, ext50=(C - e50) / C, rsi=rsi,
        sigma=sigma[:, None] * col,
        # a name with no measured movement across the trailing sample has
        # no scale to normalize by, so z6 is undefined there for the same
        # reason rvol is
        z6=np.divide(mom6, sigma[:, None] * np.sqrt(6),
                     out=np.full_like(mom6, np.nan),
                     where=sigma[:, None] > 0),
        dollar_volume=prior_mean(V.sum(1) * C[:, -1], SIGMA_SESSIONS)[:, None] * col,
    )


def breakout_mask(feat: dict[str, np.ndarray],
                  screen: dict[str, float] | None = None) -> np.ndarray:
    """Where the screen fires over the (session, clock-bar) grid.

    The base condition never changes: a fresh two-hour high on a positive
    30-minute impulse, with the trailing statistics warmed up. ``screen``
    tightens it; ``CHART_SCREEN`` is the vignette-matched setting.
    """
    sl = np.s_[:, FIRST_BAR:LAST_BAR + 1]
    m = ((feat['mom6'][sl] >= TRIGGER_MOM6) & (feat['brk2h'][sl] >= 0)
         & np.isfinite(feat['rvol'][sl]) & np.isfinite(feat['sigma'][sl])
         & np.isfinite(feat['dollar_volume'][sl]))
    for k, v in (screen or {}).items():
        m &= feat[k][sl] >= v
    return m


# ------------------------------------------------------------------- scan

def forward_paths(close: np.ndarray, n_bars: np.ndarray) -> np.ndarray:
    """(session, clock, horizon) forward returns; NaN past the close."""
    clock = np.arange(FIRST_BAR, LAST_BAR + 1)
    tgt = clock[:, None] + np.array(HORIZONS)[None, :]
    live = (tgt <= SESSION_BARS - 1)[None, :, :] & (
        n_bars[:, None, None] >= MIN_SESSION_BARS)
    p = (close[:, np.clip(tgt, 0, SESSION_BARS - 1)]
         / close[:, clock][:, :, None] - 1)
    return np.where(live & np.isfinite(p), p, np.nan)


def realized_vol(close: np.ndarray, n_bars: np.ndarray) -> np.ndarray:
    """(session, clock, window) realized volatility in basis points.

    Root sum of squared 5-minute log returns over each forward window —
    the quantity a delta-hedged option position is actually paid on.
    """
    nd = close.shape[0]
    lr = np.diff(np.log(np.maximum(close, 1e-9)), axis=1)
    cs = np.concatenate([np.zeros((nd, 1)), np.cumsum(lr ** 2, axis=1)], axis=1)
    clock = np.arange(FIRST_BAR, LAST_BAR + 1)
    out = np.full((nd, N_CLOCK, len(VOL_WINDOWS)), np.nan)
    for w, (a, b) in enumerate(VOL_WINDOWS):
        s = np.clip(clock + a, 0, SESSION_BARS - 1)
        e = np.clip(clock + b, 0, SESSION_BARS - 1)
        live = (clock + a) < SESSION_BARS - 1
        v = np.sqrt(np.maximum(cs[:, e] - cs[:, s], 0)) * 1e4
        out[:, :, w] = np.where(
            live[None, :] & (n_bars[:, None] >= MIN_SESSION_BARS), v, np.nan)
    return out


def _forward_extremes(high, low, close, sess, bar):
    """Best/worst excursion after the signal bar, and when each printed."""
    n = len(sess)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    h_mfe = np.zeros(n, np.int16)
    h_mae = np.zeros(n, np.int16)
    touch = {lvl: np.zeros(n, np.int8) for lvl in TOUCH_LEVELS}
    far = 1 << 30
    for e in range(n):
        d, b = sess[e], bar[e]
        hs, ls = high[d, b + 1:], low[d, b + 1:]
        if len(hs) == 0:
            continue
        px = close[d, b]
        mfe[e] = hs.max() / px - 1
        mae[e] = ls.min() / px - 1
        h_mfe[e] = int(np.argmax(hs)) + 1
        h_mae[e] = int(np.argmin(ls)) + 1
        for lvl in TOUCH_LEVELS:
            up, dn = hs >= px * (1 + lvl), ls <= px * (1 - lvl)
            iu = int(np.argmax(up)) if up.any() else far
            idn = int(np.argmax(dn)) if dn.any() else far
            # a bar spanning both barriers is unresolvable at this
            # resolution; charge it against the trade, never for it
            touch[lvl][e] = 0 if iu == idn == far else (1 if iu < idn else -1)
    return mfe, mae, h_mfe, h_mae, touch


def scan_ticker(ticker: str) -> dict[str, Any] | None:
    """One ticker: every permissive trigger, plus its baseline grids.

    Returns ``None`` when the archive is absent or the ruling tables clip
    the history below the warm-up the trailing statistics need.
    """
    path = archive_path(ticker)
    if path is None:
        return None
    bars = five_minute_bars(path, SCAN_START, TICKER_START_CLIPS.get(ticker),
                            TICKER_DROP_WINDOWS.get(ticker, []))
    if bars is None or len(bars['dates']) < SIGMA_SESSIONS + 5:
        return None
    close, n_bars = bars['close'], bars['n_bars']
    feat = bar_features(bars)
    paths = forward_paths(close, n_bars)
    vols = realized_vol(close, n_bars)
    clock = np.arange(FIRST_BAR, LAST_BAR + 1)
    eod = np.where(n_bars[:, None] >= MIN_SESSION_BARS,
                   close[:, [-1]] / close[:, clock] - 1, np.nan)

    sess, bar = np.nonzero(breakout_mask(feat))
    bar = bar + FIRST_BAR
    events: dict[str, np.ndarray] = {}
    if len(sess):
        mfe, mae, h_mfe, h_mae, touch = _forward_extremes(
            bars['high'], bars['low'], close, sess, bar)
        first = np.zeros(len(sess), np.int8)
        first[np.unique(sess, return_index=True)[1]] = 1
        events = dict(
            session=sess.astype(np.int32), clock=bar.astype(np.int8),
            first=first, n_bars=n_bars[sess].astype(np.int8),
            close=close[sess, bar].astype(np.float32),
            next_open=bars['open'][
                sess, np.minimum(bar + 1, SESSION_BARS - 1)].astype(np.float32),
            ret_eod=eod[sess, bar - FIRST_BAR].astype(np.float32),
            mfe=mfe.astype(np.float32), mae=mae.astype(np.float32),
            h_mfe=h_mfe, h_mae=h_mae,
            touch50=touch[0.005], touch100=touch[0.010],
            path=paths[sess, bar - FIRST_BAR].astype(np.float32),
            vol_path=vols[sess, bar - FIRST_BAR].astype(np.float32),
        )
        for k in EVENT_FEATURES:
            events[k] = feat[k][sess, bar].astype(np.float32)
    return dict(ticker=ticker, dates=bars['dates'], events=events,
                paths=paths, vols=vols, eod=eod.astype(np.float32),
                n_sessions=len(bars['dates']))


def run_scan(tickers: Iterable[str], out_dir: str) -> list[dict[str, Any]]:
    """One pass over the universe.

    Writes ``{ticker}.npz`` per ticker plus ``_margins.npz`` holding the
    cross-sectional (session, clock, horizon) and (session, clock, window)
    sums. Those day margins are a sum ACROSS tickers, so they cannot be
    resumed piecemeal — a run covers exactly the tickers it was given, and
    ``load_scan`` refuses a panel whose margins do not match the files
    present. Tickers are processed in sorted order so the accumulated
    float sums, and every number downstream, reproduce run to run.
    """
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(out_dir, '*.npz')):
        os.remove(stale)

    # The day margins accumulate as tickers stream past, so a ticker's
    # forward cube (tens of MB) is added and dropped rather than held:
    # keeping all of them to learn the session index first would cost
    # gigabytes for no benefit. New sessions extend the accumulator in
    # place — after the first ticker almost nothing extends, since the
    # names share a trading calendar — and the whole thing is sorted into
    # date order once at the end.
    index: dict[str, int] = {}
    acc: dict[str, np.ndarray] = {}
    scanned: list[str] = []
    summary: list[dict[str, Any]] = []

    def _grow(n_new: int) -> None:
        if not acc:
            acc['path_sum'] = np.zeros((0, N_CLOCK, len(HORIZONS)))
            acc['path_n'] = np.zeros((0, N_CLOCK, len(HORIZONS)), np.int64)
            acc['vol_sum'] = np.zeros((0, N_CLOCK, len(VOL_WINDOWS)))
            acc['vol_n'] = np.zeros((0, N_CLOCK, len(VOL_WINDOWS)), np.int64)
        for k, v in acc.items():
            pad = np.zeros((n_new, *v.shape[1:]), dtype=v.dtype)
            acc[k] = np.concatenate([v, pad], axis=0)

    for t in sorted(tickers):
        s = scan_ticker(t)
        if s is None:
            continue
        fresh = [d for d in s['dates'].tolist() if d not in index]
        _grow(len(fresh))
        for d in fresh:
            index[d] = len(index)
        rows = np.array([index[d] for d in s['dates'].tolist()])
        for cube, tot, cnt in ((s['paths'], acc['path_sum'], acc['path_n']),
                               (s['vols'], acc['vol_sum'], acc['vol_n'])):
            good = np.isfinite(cube)
            np.add.at(tot, rows, np.where(good, cube, 0.0))
            np.add.at(cnt, rows, good.astype(np.int64))
        payload: dict[str, Any] = dict(
            ticker=t, dates=s['dates'], eod=s['eod'],
            name_path_sum=np.nansum(s['paths'], axis=0),
            name_path_n=np.isfinite(s['paths']).sum(axis=0).astype(np.int64),
            name_vol_sum=np.nansum(s['vols'], axis=0),
            name_vol_n=np.isfinite(s['vols']).sum(axis=0).astype(np.int64),
        )
        payload.update({'ev_' + k: v for k, v in s['events'].items()})
        np.savez_compressed(os.path.join(out_dir, f'{t}.npz'), **payload)
        scanned.append(t)
        summary.append(dict(ticker=t, sessions=s['n_sessions'],
                            events=int(len(s['events'].get('session', ())))))
    if not scanned:
        raise FileNotFoundError('no ticker in the request had a usable archive')

    seen = np.array(list(index))
    order = np.argsort(seen)
    np.savez_compressed(
        os.path.join(out_dir, MARGINS_FILE), dates=seen[order],
        tickers=np.array(scanned),
        day_path_sum=acc['path_sum'][order], day_path_n=acc['path_n'][order],
        day_vol_sum=acc['vol_sum'][order], day_vol_n=acc['vol_n'][order])
    return summary


def load_scan(out_dir: str) -> dict[str, Any]:
    """Read a scan back as one panel, checking it is internally whole."""
    margins_path = os.path.join(out_dir, MARGINS_FILE)
    if not os.path.exists(margins_path):
        raise FileNotFoundError(f'no {MARGINS_FILE} under {out_dir}')
    m = np.load(margins_path, allow_pickle=False)
    dates = m['dates']
    index = {d: i for i, d in enumerate(dates.tolist())}
    files = sorted(f for f in glob.glob(os.path.join(out_dir, '*.npz'))
                   if os.path.basename(f) != MARGINS_FILE)
    tickers_on_disk = [os.path.basename(f)[:-4] for f in files]
    if tickers_on_disk != sorted(m['tickers'].tolist()):
        raise ValueError(
            'scan directory and its margins disagree on the ticker set; '
            'the day margins are a cross-sectional sum, so rerun run_scan '
            'over the whole list rather than adding tickers piecemeal')

    tickers, events = [], []
    name_path_sum, name_path_n, name_vol_sum, name_vol_n = [], [], [], []
    eod = np.full((len(files), len(dates), N_CLOCK), np.nan, np.float32)
    for ti, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        tickers.append(str(z['ticker']))
        name_path_sum.append(z['name_path_sum'])
        name_path_n.append(z['name_path_n'])
        name_vol_sum.append(z['name_vol_sum'])
        name_vol_n.append(z['name_vol_n'])
        rows = np.array([index[d] for d in z['dates'].tolist()])
        eod[ti, rows] = z['eod']
        if 'ev_session' not in z.files:
            continue
        rec = {k[3:]: z[k] for k in z.files if k.startswith('ev_')}
        rec['ticker'] = np.full(len(rec['session']), ti, np.int32)
        rec['date'] = rows[rec['session']]
        events.append(rec)
    if not events:
        raise ValueError('the scan produced no events')
    E = {k: np.concatenate([e[k] for e in events]) for k in events[0]}
    return dict(dates=dates, tickers=np.array(tickers), events=E, eod=eod,
                name_path_sum=np.array(name_path_sum),
                name_path_n=np.array(name_path_n),
                name_vol_sum=np.array(name_vol_sum),
                name_vol_n=np.array(name_vol_n),
                day_path_sum=m['day_path_sum'], day_path_n=m['day_path_n'],
                day_vol_sum=m['day_vol_sum'], day_vol_n=m['day_vol_n'])


# --------------------------------------------------------------- estimator

def _leave_one_out(total: np.ndarray, count: np.ndarray, y: np.ndarray):
    """Margin mean with this observation's own contribution removed."""
    present = np.isfinite(y)
    n = count - present
    s = total - np.where(present, y, 0.0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def two_way_excess(y, ticker, date, clock, name_sum, name_n,
                   day_sum, day_n) -> np.ndarray:
    """Leave-one-out two-way demeaned outcome.

    ``y`` is (n_events, k) — one column per horizon or window. The three
    margins are the name's own average at that clock, the session's
    average at that clock across every name, and the clock's own drift;
    the clock term is added back because subtracting both margins removes
    it twice. Leave-one-out on both margins keeps an event out of its own
    comparison, which matters at the thin cells where a session carries
    only a handful of names.
    """
    ci = clock - FIRST_BAR
    y = np.asarray(y, dtype=np.float64)
    mu_name = _leave_one_out(name_sum[ticker, ci], name_n[ticker, ci], y)
    mu_day = _leave_one_out(day_sum[date, ci], day_n[date, ci], y)
    gn = day_n.sum(axis=0).astype(np.float64)
    mu_clock = (day_sum.sum(axis=0) / np.maximum(gn, 1))[ci]
    return y - mu_name - mu_day + mu_clock


def session_bootstrap(values: np.ndarray, sessions: np.ndarray,
                      reps: int = 2000, seed: int = CONTINUATION_SEED
                      ) -> dict[str, float]:
    """Mean and one-sided p, resampling CALENDAR SESSIONS with replacement.

    A market-wide rip fires hundreds of names on the same bar, so events
    within a session are anything but independent. Resampling sessions —
    not events — is what keeps the interval honest; the event count is
    reported alongside precisely so it cannot be mistaken for the n that
    drives the width.
    """
    ok = np.isfinite(values)
    values, sessions = np.asarray(values)[ok], np.asarray(sessions)[ok]
    if len(values) == 0:
        return dict(mean=float('nan'), lo=float('nan'), hi=float('nan'),
                    p=float('nan'), n=0, n_sessions=0)
    uniq, inv = np.unique(sessions, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    counts = np.bincount(inv, minlength=len(uniq))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    sums = np.add.reduceat(values[order], offsets[:-1])
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for b in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[b] = sums[pick].sum() / max(counts[pick].sum(), 1)
    return dict(mean=float(values.mean()), lo=float(np.percentile(draws, 2.5)),
                hi=float(np.percentile(draws, 97.5)),
                p=float((draws <= 0).mean()), n=int(len(values)),
                n_sessions=int(len(uniq)))


def placebo_excess(panel: dict[str, Any], mask: np.ndarray, reps: int = 20,
                   seed: int = CONTINUATION_SEED) -> tuple[float, float]:
    """Keep each event's name and clock, move it to a random session.

    The outcome is then a real return-to-close with the signal scrambled
    out, so the estimator must return zero. If it does not, the machinery
    is manufacturing the effect and no reading of the real curve is safe.
    """
    E, eod = panel['events'], panel['eod']
    ti, ci = E['ticker'][mask], E['clock'][mask].astype(np.int64) - FIRST_BAR
    n_dates = eod.shape[1]
    # the close-horizon margins, built from the same grid the placebo draws
    name_tot = np.nansum(eod, axis=1)
    name_cnt = np.isfinite(eod).sum(axis=1)
    day_tot = np.nansum(eod, axis=0)
    day_cnt = np.isfinite(eod).sum(axis=0)
    gn = day_cnt.sum(axis=0).astype(np.float64)
    mu_clock = (day_tot.sum(axis=0) / np.maximum(gn, 1))[ci]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        d2 = rng.integers(0, n_dates, len(ti))
        y = eod[ti, d2, ci].astype(np.float64)
        mu_name = _leave_one_out(name_tot[ti, ci], name_cnt[ti, ci], y)
        mu_day = _leave_one_out(day_tot[d2, ci], day_cnt[d2, ci], y)
        out.append(np.nanmean(y - mu_name - mu_day + mu_clock) * 1e4)
    return float(np.mean(out)), float(np.std(out))


# ----------------------------------------------------------------- reports

def chart_like(panel: dict[str, Any], start: str = HEADLINE_START,
               screen: dict[str, float] | None = None) -> np.ndarray:
    """The vignette screen applied to a loaded panel, in the headline era."""
    E = panel['events']
    year = panel['dates'][E['date']]
    m = (E['n_bars'] >= MIN_SESSION_BARS) & (year >= start)
    for k, v in (screen or CHART_SCREEN).items():
        m &= E[k] >= v
    return m


def horizon_table(panel: dict[str, Any], mask: np.ndarray,
                  reps: int = 2000) -> list[dict[str, Any]]:
    """The decay curve: excess return at each forward horizon."""
    E = panel['events']
    ex = two_way_excess(E['path'][mask], E['ticker'][mask], E['date'][mask],
                        E['clock'][mask].astype(np.int64),
                        panel['name_path_sum'], panel['name_path_n'],
                        panel['day_path_sum'], panel['day_path_n'])
    raw = E['path'][mask].astype(np.float64)
    rows = []
    for k, h in enumerate(HORIZONS):
        b = session_bootstrap(ex[:, k], E['date'][mask], reps)
        if not b['n']:
            continue
        rows.append(dict(minutes=h * 5, n=b['n'], sessions=b['n_sessions'],
                         raw_bp=float(np.nanmean(raw[:, k]) * 1e4),
                         excess_bp=b['mean'] * 1e4,
                         ci_bp=[b['lo'] * 1e4, b['hi'] * 1e4], p=b['p']))
    return rows


def volatility_table(panel: dict[str, Any], mask: np.ndarray,
                     reps: int = 800) -> list[dict[str, Any]]:
    """Realized volatility after the signal, as a ratio to both baselines."""
    E = panel['events']
    y = E['vol_path'][mask].astype(np.float64)
    ci = E['clock'][mask].astype(np.int64) - FIRST_BAR
    ti, di = E['ticker'][mask], E['date'][mask]
    mu_name = _leave_one_out(panel['name_vol_sum'][ti, ci],
                             panel['name_vol_n'][ti, ci], y)
    mu_day = _leave_one_out(panel['day_vol_sum'][di, ci],
                            panel['day_vol_n'][di, ci], y)
    rows = []
    for w, label in enumerate(VOL_LABELS):
        ok = np.isfinite(y[:, w]) & (mu_name[:, w] > 0)
        if ok.sum() < 200:
            continue
        b = session_bootstrap(y[ok, w] / mu_name[ok, w], di[ok], reps)
        d = session_bootstrap(
            y[ok, w] / np.maximum(mu_day[ok, w], 1e-9), di[ok], reps)
        rows.append(dict(window=label, n=int(ok.sum()),
                         realized_bp=float(np.nanmean(y[ok, w])),
                         own_normal_bp=float(np.nanmean(mu_name[ok, w])),
                         ratio_own=b['mean'], ci_own=[b['lo'], b['hi']],
                         ratio_day=d['mean'], ci_day=[d['lo'], d['hi']]))
    return rows


def continuation_summary(panel: dict[str, Any], mask: np.ndarray,
                         reps: int = 2000) -> dict[str, Any]:
    """The headline: does the session close higher, and versus what?"""
    E = panel['events']
    ci = E['clock'][mask].astype(np.int64) - FIRST_BAR
    ti, di = E['ticker'][mask], E['date'][mask]
    y = E['ret_eod'][mask].astype(np.float64)
    eod = panel['eod']
    day_tot, day_cnt = np.nansum(eod, axis=0), np.isfinite(eod).sum(axis=0)
    name_tot, name_cnt = np.nansum(eod, axis=1), np.isfinite(eod).sum(axis=1)
    up = (y > 0).astype(np.float64)
    peer_up_tot = np.nansum(np.where(np.isfinite(eod), eod > 0, np.nan), axis=0)
    mu_name = _leave_one_out(name_tot[ti, ci], name_cnt[ti, ci], y)
    mu_day = _leave_one_out(day_tot[di, ci], day_cnt[di, ci], y)
    gn = day_cnt.sum(axis=0).astype(np.float64)
    mu_clock = (day_tot.sum(axis=0) / np.maximum(gn, 1))[ci]
    b = session_bootstrap(y - mu_name - mu_day + mu_clock, di, reps)
    peer_up = _leave_one_out(peer_up_tot[di, ci], day_cnt[di, ci], up)
    return dict(
        n=int(mask.sum()), sessions=b['n_sessions'],
        names=int(len(np.unique(ti))),
        p_close_up=float(np.mean(y > 0)),
        p_peer_up=float(np.nanmean(peer_up)),
        excess_bp=b['mean'] * 1e4, ci_bp=[b['lo'] * 1e4, b['hi'] * 1e4],
        p=b['p'],
        mfe_bp=float(np.nanmean(E['mfe'][mask]) * 1e4),
        mae_bp=float(np.nanmean(E['mae'][mask]) * 1e4),
        sd_bp=float(np.nanstd(y) * 1e4),
        touch_up=float(np.mean(E['touch50'][mask] == 1)),
        touch_down=float(np.mean(E['touch50'][mask] == -1)),
        median_bars_to_high=float(np.median(E['h_mfe'][mask])),
        slippage_bp=float(np.nanmean(
            E['next_open'][mask] / E['close'][mask] - 1) * 1e4),
    )


def breadth(panel: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    """Share of the eligible universe firing the same screen on that bar.

    The single stratifier that separates "this name broke out" from "the
    whole tape went up", which is the difference between a signal and a
    re-labelled market move.
    """
    E = panel['events']
    n_dates, n_clock = panel['day_path_n'].shape[:2]
    firing = np.zeros((n_dates, n_clock), np.int64)
    np.add.at(firing, (E['date'][mask], E['clock'][mask].astype(np.int64)
                       - FIRST_BAR), 1)
    eligible = np.maximum(panel['day_path_n'][:, :, 0], 1)
    return (firing / eligible)[E['date'], E['clock'].astype(np.int64) - FIRST_BAR]


# -------------------------------------------------------------------- CLI

def build_results(panel: dict[str, Any]) -> dict[str, Any]:
    """Everything the log entry quotes, in one serializable block."""
    mask = chart_like(panel)
    pm, ps = placebo_excess(panel, mask)
    br = breadth(panel, mask)
    bands = ((0.0, 0.005, 'lone mover'), (0.005, 0.02, 'thin'),
             (0.02, 0.05, 'moderate'), (0.05, 0.15, 'broad'),
             (0.15, 1.01, 'tape-wide'))
    by_breadth = []
    for lo, hi, label in bands:
        m = mask & (br >= lo) & (br < hi)
        if m.sum() < 50:
            continue
        s = continuation_summary(panel, m)
        by_breadth.append(dict(band=label, **s))
    return dict(
        era=HEADLINE_START, screen=CHART_SCREEN,
        summary=continuation_summary(panel, mask),
        horizons=horizon_table(panel, mask),
        volatility=volatility_table(panel, mask),
        breadth=by_breadth, placebo=[pm, ps])


def _fmt(rows: list[dict[str, Any]], cols: Sequence[str]) -> str:
    head = ''.join(f'{c:>14}' for c in cols)
    body = [''.join(f'{r[c]:>14.2f}' if isinstance(r[c], float)
                    else f'{str(r[c]):>14}' for c in cols) for r in rows]
    return '\n'.join([head, '-' * len(head), *body])


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan', action='store_true',
                    help='read the archive and write the scan directory')
    ap.add_argument('--report', action='store_true',
                    help='read the scan directory and print the tables')
    ap.add_argument('--tickers', default=None,
                    help='comma-separated subset (default: the S&P 500 '
                         'universe plus SPY and QQQ)')
    ap.add_argument('--out', default=None, help=f'default: data/{SCAN_DIR}')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--write-results', action='store_true',
                    help=f'refresh the committed data/{RESULTS_FILE}')
    a = ap.parse_args(argv)
    out = a.out or data_path(SCAN_DIR)

    if a.scan:
        tickers = (a.tickers.split(',') if a.tickers
                   else sorted(set(universe()) | {'SPY', 'QQQ'}))
        rows = run_scan(tickers, out)
        print(f'scanned {len(rows)} tickers, '
              f'{sum(r["events"] for r in rows)} permissive triggers -> {out}')
    if not (a.report or a.write_results):
        return

    panel = load_scan(out)
    results = build_results(panel)
    summary, horizons = results['summary'], results['horizons']
    vols, (pm, ps) = results['volatility'], results['placebo']
    if a.write_results:
        with open(data_path(RESULTS_FILE), 'w') as f:
            json.dump(results, f, indent=1, sort_keys=True)
            f.write('\n')
        print(f'wrote data/{RESULTS_FILE}')
    if a.json:
        print(json.dumps(results, indent=1))
        return
    if not a.report:
        return
    print(f'chart-like breakouts, {HEADLINE_START}+: n={summary["n"]} over '
          f'{summary["sessions"]} sessions and {summary["names"]} names')
    print(f'  closes higher {summary["p_close_up"]*100:.1f}% vs peers '
          f'{summary["p_peer_up"]*100:.1f}% on the same session and clock')
    print(f'  excess to the close {summary["excess_bp"]:+.2f} bp '
          f'[{summary["ci_bp"][0]:+.1f},{summary["ci_bp"][1]:+.1f}] '
          f'p={summary["p"]:.3f}')
    print(f'  placebo {pm:+.2f} bp (sd {ps:.2f}) -- must be ~0')
    print('\nDECAY CURVE')
    print(_fmt(horizons, ('minutes', 'n', 'raw_bp', 'excess_bp', 'p')))
    print('\nREALIZED VOLATILITY')
    print(_fmt(vols, ('window', 'n', 'realized_bp', 'own_normal_bp',
                      'ratio_own', 'ratio_day')))


if __name__ == '__main__':
    main()
