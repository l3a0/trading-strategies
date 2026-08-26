"""Volume-profile mean-reversion scout — EXPLORATORY, not a registered verdict.

The idea, in the trader's words: build a volume profile, SHORT when price
reaches the top of it and LONG when it reaches the bottom, betting on
reversion toward the point of control (POC, the highest-volume price). The
question is narrow — *is the reversal an edge?* — and the honest answer needs
three things this module is built to give: a profile that never peeks at the
future, an outcome that is stripped of NVDA's ferocious upward drift, and a
verdict judged across the whole grid of profile-definition knobs at once.

WHY THE LOOK-AHEAD RULE IS EVERYTHING. A charting package draws the profile
over the whole visible window, including bars in the *future* of any candidate
entry. "Short at the top" using a profile that already knows where the top
ended up is the single reason these backtests look brilliant and die live.
Every profile here is strictly point-in-time: the levels that classify a bar
are built only from bars *before* it. Two anchorings, both searched as grid
variants:

  * ``prior``      — the profile is frozen from the prior ``lookback`` SESSIONS
                     at each session's open, then traded as fixed support /
                     resistance shelves. Look-ahead is structurally impossible.
  * ``developing`` — the profile is rebuilt from the CURRENT session's bars
                     strictly before the signal bar, and price fading beyond
                     the developing edge is the "excess" trade. This is the
                     chart-style fade; it is noisier early in the session, so
                     it only fires once ``min_dev_bars`` bars have printed.

WHY A SINGLE TICKER CHANGES THE ESTIMATOR. The cross-sectional two-way
demeaning in :mod:`engine.intraday_continuation` needs peer names to form the
session margin; with NVDA alone there are none. The drift strip here is the
matched random-entry control instead (Tharp's null, adapted): the SAME clock
bar, a random OTHER session, the SAME bracket geometry (the event's own
target/stop fractions). The event minus the mean of its controls is the
excess — what reaching the profile edge adds over entering at that time of day
at random with an identical bracket. The drift-agnostic companion is the
first-touch race: from the edge, does price reach POC (the target) before the
stop, more often than the control does?

WHY THE WHOLE GRID IS JUDGED AT ONCE. Lookback, bin count, value-area width,
and stop distance are all knobs, and sweeping them is a multiple-comparisons
hazard. The committed grid is judged as one batch by Benjamini-Yekutieli
(:func:`search.edge_search.benjamini_yekutieli`, valid under the arbitrary
dependence these overlapping-window variants have). A single ticker has no
sealed peer vault, so the honesty rail is instead a TIME-AXIS holdout: the
grid is measured on an early NVDA span and any survivor is re-measured on a
held-out later span it never saw.

This is a sample-spending, kill-or-justify scout. Pinning the numbers stops
the idea being re-derived every session; it does NOT promote the scout to a
registered finding. A survivor escalates to manual pre-registration, never an
automatic promotion.
"""
from __future__ import annotations

import json
import os
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from multiprocessing import Pool
from typing import Any

import numpy as np

from common.paths import data_path
from common.stats import newey_west_summary
from engine.intraday_continuation import (
    FIRST_BAR,
    LAST_BAR,
    MIN_SESSION_BARS,
    SESSION_BARS,
    five_minute_bars,
    session_bootstrap,
)
from pipeline.minute_archive import (
    TICKER_DROP_WINDOWS,
    TICKER_START_CLIPS,
    UNIVERSE_SNAPSHOTS,
    archive_path,
    archived_tickers,
    load_splits,
)
from search.edge_search import benjamini_yekutieli

# --------------------------------------------------------------- constants

VP_SEED = 20260724
TICKER = 'NVDA'
# The archive era this scout reads and the sub-span it reports, matching the
# intraday-continuation convention (2016+ is the modern-microstructure era).
ERA_START = '2016-01-01'
# The time-axis holdout that stands in for a sealed peer vault on a single
# ticker: tune-free measurement on TRAIN, confirmation of any survivor on the
# later span it never saw. The split is a calendar date, committed up front.
HOLDOUT_SPLIT = '2022-01-01'

# How a 5-minute bar's volume is placed in the price histogram: at its typical
# price (H+L+C)/3, the profile convention. Bars with no print carry zero
# volume (see ``five_minute_bars``) and are dropped from BOTH the histogram
# and the price range, so a synthetic filled level is never a "touched" node.
FDR_Q = 0.10          # the Benjamini-Yekutieli false-discovery level
N_CONTROLS = 5        # matched random-entry controls per event (Tharp's null)
MIN_WINDOW_BARS = 60  # a profile needs at least this many real bars to be built


@dataclass(frozen=True)
class ProfileCfg:
    """One profile definition — a cell in the FDR grid.

    ``predicted_sign`` is +1 by construction: the reversal hypothesis says the
    drift-stripped excess is POSITIVE (fading the edge pays). Stating it up
    front is the honest-search precondition — the kill-gate tests the edge
    against this prior, and the repo's standing finding is that conditioning
    entry on recent price action tends to be sign-backwards on these names.
    """

    anchoring: str            # 'prior' | 'developing'
    lookback: int             # prior: number of sessions; developing: unused
    n_bins: int               # price bins in the volume histogram
    va_frac: float            # value-area fraction (0.70 = the 70% band)
    stop_mult: float          # stop sits stop_mult * (VAH-VAL) beyond the edge
    direction: str = 'reversion'  # 'reversion' (fade the edge) | 'continuation' (ride it)
    horizon: object = 'close'  # 'close' = intraday to session close; int = trading days
    min_dev_bars: int = 12    # developing: earliest signal bar within a session

    def key(self) -> str:
        h = self.horizon if self.horizon == 'close' else f'{self.horizon}d'
        base = (f'{self.anchoring}_lb{self.lookback}_b{self.n_bins}'
                f'_va{int(self.va_frac * 100)}_s{self.stop_mult:g}_h{h}')
        if self.anchoring == 'developing':
            base += f'_md{self.min_dev_bars}'   # distinguishes the min_dev cells
        if self.direction != 'reversion':
            base += f'_{self.direction}'        # reversion keys stay unchanged
        return base


# The committed grid: both anchorings at their NATURAL horizons. A prior-window
# value area spans many sessions, so its POC sits days away from an intraday
# edge touch — a prior cell is a multi-day swing (held ``horizon`` trading
# days). A developing value area is within-session, so its POC is minutes away
# — a developing cell is intraday, held to the close. Small on purpose:
# BY power bleeds as the batch grows, and a menu this size is enumerable.
GRID: tuple[ProfileCfg, ...] = (
    ProfileCfg('prior', 20, 48, 0.70, 1.0, horizon=10),
    ProfileCfg('prior', 10, 48, 0.70, 1.0, horizon=5),
    ProfileCfg('developing', 0, 48, 0.70, 1.0, horizon='close'),
    ProfileCfg('developing', 0, 48, 0.70, 1.0, horizon='close', min_dev_bars=24),
)


# --------------------------------------------------------- split adjustment

def split_adjust_bars(bars: dict[str, np.ndarray],
                      splits: Sequence[tuple[str, float]]) -> dict[str, np.ndarray]:
    """Put every session's OHLC and volume on one (latest) price scale.

    ``five_minute_bars`` returns RAW, AS-TRADED prices and volume — a profile
    window that crosses a split ex-date would scatter volume across the wrong
    price bins (NVDA's 2024-06-10 x10 turns a $1200 close into a $120 open).
    The daily :func:`pipeline.minute_archive.split_adjust` rule, applied here
    per session (the factor is constant within a day): divide OHLC by the
    product of ratios dated strictly AFTER the session, multiply volume by the
    same factor. A no-split ticker gets factor 1.0 and is byte-unchanged.
    """
    dates = bars['dates']
    factor = np.ones(len(dates))
    for ex_date, ratio in splits:
        factor[dates < ex_date] *= float(ratio)
    out = dict(bars)
    inv = factor[:, None]
    for k in ('open', 'high', 'low', 'close'):
        out[k] = bars[k] / inv
    out['volume'] = bars['volume'] * inv
    return out


# --------------------------------------------------------- profile builder

def _value_area(vol: np.ndarray, edges: np.ndarray,
                va_frac: float) -> tuple[float, float, float] | None:
    """POC / VAH / VAL from a per-bin volume histogram.

    POC is the center of the highest-volume bin. The value area grows outward
    from the POC bin, at each step annexing whichever adjacent bin (above or
    below) carries more volume, until the accumulated volume reaches
    ``va_frac`` of the total — the standard market-profile value-area rule.
    VAH/VAL are the outer edges of the highest/lowest annexed bins.
    """
    total = float(vol.sum())
    if total <= 0:
        return None
    poc = int(np.argmax(vol))
    lo = hi = poc
    acc = float(vol[poc])
    target = va_frac * total
    k = len(vol)
    while acc < target and (lo > 0 or hi < k - 1):
        below = float(vol[lo - 1]) if lo > 0 else -1.0
        above = float(vol[hi + 1]) if hi < k - 1 else -1.0
        if above >= below:
            hi += 1
            acc += float(vol[hi])
        else:
            lo -= 1
            acc += float(vol[lo])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return float(centers[poc]), float(edges[hi + 1]), float(edges[lo])


def volume_profile(typical: np.ndarray, volume: np.ndarray, n_bins: int,
                   va_frac: float) -> tuple[float, float, float] | None:
    """Volume-by-price POC/VAH/VAL over a window of bars.

    ``typical`` and ``volume`` are flat per-bar arrays (already restricted to
    real, non-empty bars by the caller). Returns ``None`` for a degenerate
    window (no volume, or a single price level).
    """
    v = np.asarray(volume, dtype=np.float64)
    t = np.asarray(typical, dtype=np.float64)
    if v.sum() <= 0:
        return None
    lo, hi = float(t.min()), float(t.max())
    if not (hi > lo):
        return None
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    hist = np.zeros(n_bins)
    np.add.at(hist, idx, v)
    return _value_area(hist, edges, va_frac)


def _window_typical_volume(bars: dict[str, np.ndarray], rows: slice,
                           bar_lo: int, bar_hi: int
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Typical price (H+L+C)/3 and volume of every REAL bar in a window.

    ``rows`` selects sessions; ``[bar_lo:bar_hi]`` selects clock bars within
    them. Bars with zero volume (no print — a synthetic filled level) are
    dropped, so they touch neither the histogram nor the price range.
    """
    H = bars['high'][rows, bar_lo:bar_hi]
    L = bars['low'][rows, bar_lo:bar_hi]
    C = bars['close'][rows, bar_lo:bar_hi]
    V = bars['volume'][rows, bar_lo:bar_hi]
    mask = V > 0
    typ = ((H + L + C) / 3.0)[mask]
    return typ, V[mask]


# --------------------------------------------------------------- detector

@dataclass
class _Event:
    session: int          # session index into the bars matrices
    date: str
    bar: int              # signal bar (clock index)
    side: int             # -1 short at the top, +1 long at the bottom
    entry: float          # the signal-bar close (symmetric with the control)
    target: float         # POC — the reversion target
    stop: float           # beyond the edge — the invalidation


def reversal_events(bars: dict[str, np.ndarray], cfg: ProfileCfg,
                    era_start: str, era_end: str | None) -> list[_Event]:
    """Every profile-edge reach on split-adjusted bars, point-in-time.

    For each eligible session, the profile is built strictly from the past
    (prior sessions, or the current session's earlier bars), and the FIRST
    bar whose high tags VAH (short) or whose low tags VAL (long) is taken —
    one entry per side per session, so sitting beyond the edge does not fire
    every bar. Entries stop at ``LAST_BAR`` so every trade has room to run to
    the close.
    """
    dates = bars['dates']
    H, L, C, nb = bars['high'], bars['low'], bars['close'], bars['n_bars']
    nd = len(dates)
    events: list[_Event] = []
    for s in range(nd):
        if nb[s] < MIN_SESSION_BARS or dates[s] < era_start:
            continue
        if era_end is not None and dates[s] >= era_end:
            continue
        # Embargo the train/holdout boundary: an int-horizon trade holds N days
        # forward, so admit it only if its whole outcome window stays in-span —
        # else a train trade would read holdout-span prices (the classic
        # overlapping-label leak). 'close' horizons never cross a session.
        if (era_end is not None and cfg.horizon != 'close'
                and dates[min(s + int(cfg.horizon), nd - 1)] >= era_end):
            continue

        prior_levels = None
        if cfg.anchoring == 'prior':
            s0 = s - cfg.lookback
            if s0 < 0:
                continue
            typ, vol = _window_typical_volume(bars, slice(s0, s), 0, SESSION_BARS)
            if typ.size < MIN_WINDOW_BARS:
                continue
            prior_levels = volume_profile(typ, vol, cfg.n_bins, cfg.va_frac)
            if prior_levels is None:
                continue
            start_bar = FIRST_BAR
        else:
            start_bar = max(FIRST_BAR, cfg.min_dev_bars)

        fired = {-1: False, 1: False}
        for i in range(start_bar, LAST_BAR + 1):
            if fired[-1] and fired[1]:
                break
            if cfg.anchoring == 'prior':
                levels = prior_levels
            else:
                typ, vol = _window_typical_volume(bars, slice(s, s + 1), 0, i)
                if typ.size < cfg.min_dev_bars:
                    continue
                levels = volume_profile(typ, vol, cfg.n_bins, cfg.va_frac)
                if levels is None:
                    continue
            poc, vah, val = levels
            width = vah - val
            if not (width > 0 and val < poc < vah):
                continue
            hi_i, lo_i, c_i = float(H[s, i]), float(L[s, i]), float(C[s, i])
            # short: price reaches the top (VAH), target the POC below it.
            # Entry is the signal-bar close — the realistic fill for a
            # signal-triggered order, and symmetric with the control.
            if (not fired[-1]) and hi_i >= vah:
                fired[-1] = True
                events.append(_make_event(cfg, s, dates[s], i, c_i, -1, poc,
                                          vah + cfg.stop_mult * width))
            # long: price reaches the bottom (VAL), target the POC above it
            if (not fired[1]) and lo_i <= val:
                fired[1] = True
                events.append(_make_event(cfg, s, dates[s], i, c_i, 1, poc,
                                          val - cfg.stop_mult * width))
    return events


def _make_event(cfg: ProfileCfg, s: int, date: str, bar: int, entry: float,
                side: int, target: float, stop: float) -> _Event:
    """A reversion event, or its mirror when direction=='continuation'.

    The reversion trade fades the edge (target = POC, stop = beyond the edge).
    Continuation is the exact opposite bet on the same touch: flip the side and
    swap target<->stop, so the trade rides the breakout THROUGH the edge with
    the stop back toward the POC. The two share the same three price levels;
    only which is the target and which the stop (and the side) differ — so a
    continuation trade's return is the NEGATIVE of the reversion trade's,
    except when a single bar engulfs both levels (both trades then hit their
    own stop, at different prices).
    """
    if cfg.direction == 'continuation':
        side, target, stop = -side, stop, target
    return _Event(s, date, bar, side, entry, target, stop)


def _flip_event(e: _Event) -> _Event:
    """The continuation mirror of a reversion event — flip side, swap levels.

    Lets a universe run detect edge touches ONCE (the expensive step) and get
    the continuation events for free, instead of re-running the developing
    profile a second time.
    """
    return _Event(e.session, e.date, e.bar, -e.side, e.entry, e.stop, e.target)


# --------------------------------------------------------------- outcomes

def _daily_hlc(bars: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-session daily high/low/close from the intraday matrices.

    ``close[:, -1]`` is the forward-filled session close; the high/low take
    the session extremes. Synthetic no-print bars carry a within-range filled
    price, so they never distort the extremes. Used for the multi-day horizon.
    """
    return bars['high'].max(1), bars['low'].min(1), bars['close'][:, -1]


def _forward(bars: dict[str, np.ndarray],
             daily: tuple[np.ndarray, np.ndarray, np.ndarray], s: int, bar: int,
             horizon: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """Forward high/low/close path and the time-stop close for one trade.

    ``'close'`` walks the remaining 5-minute bars of the SAME session (an
    intraday trade held to the close). An int ``N`` walks the entry session's
    remaining bars AND THEN the next ``N`` daily bars — the position is
    monitored continuously from the intraday entry, because the entry-day
    afternoon (where a just-tagged edge sits ~1 width from its stop) is the
    highest-risk window and dropping it would silently discard same-day
    stop-outs. Only bars at or after the signal bar are ever read, so the
    horizon is look-ahead free. The time-stop is the close of day ``s+N``.
    """
    ih, il, ic = (bars['high'][s, bar + 1:], bars['low'][s, bar + 1:],
                  bars['close'][s, bar + 1:])
    if horizon == 'close':
        if len(ic) < 2:
            return None
        return ih, il, ic, float(bars['close'][s, -1])
    dh, dl, dc = daily
    e = min(s + 1 + int(horizon), len(dc))
    if e > s + 1:                       # entry-day tail + N following days
        fh = np.concatenate([ih, dh[s + 1:e]])
        fl = np.concatenate([il, dl[s + 1:e]])
        fc = np.concatenate([ic, dc[s + 1:e]])
        final = float(dc[e - 1])
    else:                               # last session: only the entry-day tail
        fh, fl, fc, final = ih, il, ic, float(bars['close'][s, -1])
    if len(fc) < 1:
        return None
    return fh, fl, fc, final


def _reversion_outcome(fh: np.ndarray, fl: np.ndarray, fc: np.ndarray,
                       eod: float, entry: float, target: float, stop: float,
                       side: int, fill: str) -> tuple[float, int, float]:
    """Signed return, first-touch verdict, and MAE for one reversion trade.

    Along the forward path (intraday bars or daily bars, per the horizon): the
    target sits toward POC, the stop beyond the edge, the time-stop (``eod``)
    is the horizon-end close. ``first`` is the drift-agnostic race — +1 if POC
    is reached before the stop, -1 if the stop is reached first, 0 if neither
    prints before the time-stop. A bar that spans both is booked against the
    trade (stop-first), the repo convention. Level exits fill at the level
    under ``'barrier'`` and at the crossing bar's close under ``'touch'``.
    """
    far = 1 << 30
    if side == 1:                      # long: target above, stop below
        t_hits = np.nonzero(fh >= target)[0]
        s_hits = np.nonzero(fl <= stop)[0]
    else:                              # short: target below, stop above
        t_hits = np.nonzero(fl <= target)[0]
        s_hits = np.nonzero(fh >= stop)[0]
    it = int(t_hits[0]) if len(t_hits) else far
    is_ = int(s_hits[0]) if len(s_hits) else far
    if it == is_ == far:
        first, exit_px = 0, eod
    elif it < is_:                     # POC reached first — the reversion wins
        first = 1
        exit_px = target if fill == 'barrier' else float(fc[it])
    else:                              # stop first (ties -> adverse)
        first = -1
        exit_px = stop if fill == 'barrier' else float(fc[is_])
    ret = side * (exit_px / entry - 1.0) if entry > 0 else 0.0
    # worst adverse excursion UP TO the realized exit bar (not the whole path)
    exit_idx = it if first == 1 else (is_ if first == -1 else len(fc) - 1)
    if side == 1:
        mae = float(fl[:exit_idx + 1].min()) / entry - 1.0
    else:
        mae = -(float(fh[:exit_idx + 1].max()) / entry - 1.0)
    return ret, first, mae


def _control_outcome(bars: dict[str, np.ndarray],
                     daily: tuple[np.ndarray, np.ndarray, np.ndarray], s2: int,
                     bar: int, side: int, target_frac: float, stop_frac: float,
                     horizon: object, fill: str) -> tuple[float, int] | None:
    """The event's bracket geometry replayed at the same clock, a random day.

    Same clock bar, a random OTHER session, the SAME fractional target and
    stop distances the event used, and the SAME horizon — the matched
    random-entry (Tharp) control. Its mean is the counterfactual the event's
    excess is measured against.
    """
    px = float(bars['close'][s2, bar])
    if px <= 0:
        return None
    fwd = _forward(bars, daily, s2, bar, horizon)
    if fwd is None:
        return None
    fh, fl, fc, final = fwd
    entry = px          # a control has no profile edge — it enters at its bar
    target = px * (1.0 + target_frac)
    stop = px * (1.0 + stop_frac)
    ret, first, _ = _reversion_outcome(fh, fl, fc, final, entry, target, stop,
                                       side, fill)
    return ret, first


def collect_cell(bars: dict[str, np.ndarray], cfg: ProfileCfg, era_start: str,
                 era_end: str | None, fill: str, rng: np.random.Generator,
                 events: list | None = None) -> dict[str, np.ndarray]:
    """Per-event drift-stripped outcomes for one cell — the poolable unit.

    Returns parallel arrays (event return, control-mean return, event/control
    POC-first indicators, session date, side), one entry per event that had a
    valid forward path and >=2 matched controls. ``measure_cell`` aggregates
    them for one ticker; a universe run concatenates them across tickers before
    a single pooled bootstrap. Pass ``events`` to reuse a detection — the
    continuation events are the flipped reversion events, so the expensive
    developing-profile detection is not repeated per direction.
    """
    nb = bars['n_bars']
    dates = bars['dates']
    nd = len(dates)
    daily = _daily_hlc(bars)
    in_era = (dates < era_end) if era_end is not None else np.ones(nd, bool)
    # controls hold the same horizon, so embargo their forward window too
    embargo_ok = np.ones(nd, bool)
    if era_end is not None and cfg.horizon != 'close':
        fwd = np.minimum(np.arange(nd) + int(cfg.horizon), nd - 1)
        embargo_ok = dates[fwd] < era_end
    eligible = np.nonzero((nb >= MIN_SESSION_BARS) & (dates >= era_start)
                          & in_era & embargo_ok)[0]
    if events is None:
        events = reversal_events(bars, cfg, era_start, era_end)

    ev_ret, ct_ret, ev_first, ct_first = [], [], [], []
    ev_sess, ev_side = [], []
    for e in events:
        fwd = _forward(bars, daily, e.session, e.bar, cfg.horizon)
        if fwd is None:
            continue
        fh, fl, fc, final = fwd
        entry = e.entry
        if entry <= 0:
            continue
        r, first, _ = _reversion_outcome(fh, fl, fc, final, entry, e.target,
                                         e.stop, e.side, fill)
        tf, sf = e.target / entry - 1.0, e.stop / entry - 1.0
        pool = rng.choice(eligible, size=N_CONTROLS * 3, replace=True)
        controls = [int(x) for x in pool if int(x) != e.session][:N_CONTROLS]
        c_r, c_win = [], []
        for s2 in controls:
            got = _control_outcome(bars, daily, s2, e.bar, e.side, tf, sf,
                                   cfg.horizon, fill)
            if got is not None:
                c_r.append(got[0])
                c_win.append(1.0 if got[1] == 1 else 0.0)   # POC-first indicator
        if len(c_r) < 2:
            continue
        ev_ret.append(r)
        ct_ret.append(float(np.mean(c_r)))
        ev_first.append(1.0 if first == 1 else 0.0)         # POC-first indicator
        ct_first.append(float(np.mean(c_win)))              # control POC-first rate
        ev_sess.append(e.date)
        ev_side.append(e.side)
    return {'ev_ret': np.array(ev_ret), 'ct_ret': np.array(ct_ret),
                'ev_first': np.array(ev_first), 'ct_first': np.array(ct_first),
                'date': np.array(ev_sess), 'side': np.array(ev_side, dtype=int)}


def _summarize(label: str, fill: str, c: dict[str, np.ndarray]) -> dict[str, Any]:
    """Aggregate collected per-event arrays into the pinnable report block.

    Shared by the single-ticker ``measure_cell`` and the pooled universe run,
    so both judge significance identically — a session-clustered bootstrap of
    the excess and a Newey-West HAC t of the per-trade excess.
    """
    ev_ret, ct_ret = c['ev_ret'], c['ct_ret']
    n = len(ev_ret)
    if n < 20:
        return {'cell': label, 'fill': fill, 'n': n, 'note': 'too few events'}
    excess = ev_ret - ct_ret
    sess = c['date']
    boot = session_bootstrap(excess, sess, seed=VP_SEED)
    hac = newey_west_summary(excess)
    ev_win, ct_win = c['ev_first'], c['ct_first']
    fboot = session_bootstrap(ev_win - ct_win, sess, seed=VP_SEED + 1)
    side_arr = c['side']
    return {
        'cell': label, 'fill': fill, 'n': n, 'n_sessions': int(boot['n_sessions']),
        'n_short': int((side_arr == -1).sum()), 'n_long': int((side_arr == 1).sum()),
        'event_bp': float(ev_ret.mean() * 1e4),
        'control_bp': float(ct_ret.mean() * 1e4),
        'excess_bp': float(boot['mean'] * 1e4),
        'excess_ci_bp': [float(boot['lo'] * 1e4), float(boot['hi'] * 1e4)],
        'excess_p': float(boot['p']),
        'hac_t': round(float(hac.t_newey_west), 3),
        'naive_t': round(float(hac.t_naive), 3),
        'poc_first_rate': float(ev_win.mean()),
        'control_poc_first_rate': float(ct_win.mean()),
        'poc_first_excess': float(fboot['mean']),
        'poc_first_p': float(fboot['p']),
        'predicted_sign': 1,
    }


def measure_cell(bars: dict[str, np.ndarray], cfg: ProfileCfg, era_start: str,
                 era_end: str | None, fill: str, rng: np.random.Generator
                 ) -> dict[str, Any]:
    """One grid cell on one ticker: events, matched controls, excess, verdict."""
    return _summarize(cfg.key(), fill,
                      collect_cell(bars, cfg, era_start, era_end, fill, rng))


# ------------------------------------------------------------ the campaign

def _load_ticker(ticker: str,
                 splits: dict | None = None) -> dict[str, np.ndarray] | None:
    """Split-adjusted 5-minute bars for one ticker, or None if unusable.

    ``splits`` may be passed pre-loaded so a universe run reads the split
    snapshot once instead of once per ticker.
    """
    path = archive_path(ticker)
    if path is None:
        return None
    bars = five_minute_bars(path, '2000-01-01', TICKER_START_CLIPS.get(ticker),
                            TICKER_DROP_WINDOWS.get(ticker, []))
    if bars is None:
        return None
    if splits is None:
        splits = load_splits()
    return split_adjust_bars(bars, splits.get(ticker, []))


def _load_nvda() -> dict[str, np.ndarray] | None:
    return _load_ticker(TICKER)


def run_grid(bars: dict[str, np.ndarray], era_start: str = ERA_START,
             era_end: str | None = None, fill: str = 'touch') -> dict[str, Any]:
    """Measure every grid cell and judge the batch with Benjamini-Yekutieli.

    ``fill='touch'`` is the honest fill (charges the chase into the edge);
    ``'barrier'`` is the optimistic bound. The FDR gate runs on the touch-fill
    excess one-sided p across the whole grid — never per-cell.
    """
    cells = []
    for cfg in GRID:
        rng = np.random.default_rng(VP_SEED ^ zlib.crc32(cfg.key().encode()))
        cells.append(measure_cell(bars, cfg, era_start, era_end, fill, rng))
    pvals = [c.get('excess_p') for c in cells]
    flags = benjamini_yekutieli(pvals, q=FDR_Q)
    for c, f in zip(cells, flags):
        c['by_survivor'] = bool(f)
    return {'era_start': era_start, 'era_end': era_end, 'fill': fill,
                'q': FDR_Q, 'cells': cells,
                'survivors': [c['cell'] for c in cells if c.get('by_survivor')]}


def run_holdout(bars: dict[str, np.ndarray]) -> dict[str, Any]:
    """Train / holdout split — the single-ticker stand-in for a sealed vault.

    The grid is judged on the TRAIN span; any BY survivor is re-measured on the
    later HOLDOUT span it never saw. A survivor that does not repeat out of
    sample is drift or overfit, not an edge.
    """
    train = run_grid(bars, ERA_START, HOLDOUT_SPLIT, 'touch')
    survivors = [cfg for cfg in GRID if cfg.key() in set(train['survivors'])]
    holdout_cells = []
    for cfg in survivors:
        rng = np.random.default_rng(VP_SEED ^ zlib.crc32(cfg.key().encode()))
        holdout_cells.append(
            measure_cell(bars, cfg, HOLDOUT_SPLIT, None, 'touch', rng))
    return {'train': train, 'holdout_split': HOLDOUT_SPLIT,
                'holdout_cells': holdout_cells}


# ---------------------------------------------------- the universe campaign

# Both directions of the trade, measured across the whole intraday universe
# (S&P 500 + Nasdaq-100 + S&P 400). A single ticker cannot tell reversion from
# drift-plus-luck; the cross-section can — and it makes the sealed / held-out
# test that a genuine breakout edge would have to pass. Three eras per cell so
# the full-era headline, the in-sample fit, and the out-of-sample confirmation
# all come from one per-ticker pass.
UNIVERSE_ERAS = (('full', ERA_START, None),
                 ('train', ERA_START, HOLDOUT_SPLIT),
                 ('holdout', HOLDOUT_SPLIT, None))
_POOL_FIELDS = ('ev_ret', 'ct_ret', 'ev_first', 'ct_first', 'date', 'side')


def _uni_seed(ticker: str, tag: str) -> int:
    return VP_SEED ^ zlib.crc32((ticker + '|' + tag).encode())


def _partition_eras(full_events: list, cfg: ProfileCfg, dates: np.ndarray,
                    nd: int) -> dict[str, list]:
    """Split full-era events into full/train/holdout by date.

    The train/holdout events are exactly the date-partitions of the full-era
    detection (the profile at a session depends only on prior sessions, so it
    is era-independent), so detection runs ONCE per cell instead of per era.
    The train side re-applies the boundary embargo — an int-horizon trade whose
    outcome window would cross the split is dropped, so no train trade reads a
    holdout-span bar.
    """
    split = HOLDOUT_SPLIT
    is_int = cfg.horizon != 'close'
    h = int(cfg.horizon) if is_int else 0
    train, holdout = [], []
    for e in full_events:
        if e.date >= split:
            holdout.append(e)
        elif not is_int or dates[min(e.session + h, nd - 1)] < split:
            train.append(e)
    return {'full': full_events, 'train': train, 'holdout': holdout}


def collect_ticker(ticker: str, cells: Sequence[ProfileCfg] = GRID,
                   fill: str = 'touch') -> dict | None:
    """Per-event arrays for every cell x {reversion, continuation} x era on one
    ticker — the poolable payload for a universe run, keyed by
    ``(cell_key, direction, era)``. Detection runs once per cell (train/holdout
    are date partitions of it); continuation reuses the flipped reversion
    events. Deterministic: the control RNG is seeded from the ticker, so a
    parallel run over the universe is order-independent and reproducible.
    ``None`` if the archive is absent.
    """
    bars = _load_ticker(ticker)
    if bars is None:
        return None
    dates, nd = bars['dates'], len(bars['dates'])
    out: dict = {}
    for cfg in cells:
        cont = replace(cfg, direction='continuation')
        parts = _partition_eras(reversal_events(bars, cfg, ERA_START, None),
                                cfg, dates, nd)
        for era_name, es, ee in UNIVERSE_ERAS:
            rev_events = parts[era_name]
            cont_events = [_flip_event(e) for e in rev_events]
            out[(cfg.key(), 'reversion', era_name)] = collect_cell(
                bars, cfg, es, ee, fill,
                np.random.default_rng(_uni_seed(ticker, cfg.key() + era_name)),
                rev_events)
            out[(cont.key(), 'continuation', era_name)] = collect_cell(
                bars, cont, es, ee, fill,
                np.random.default_rng(_uni_seed(ticker, cont.key() + era_name)),
                cont_events)
    return out


def _pool(collected: Sequence[dict], key: tuple) -> dict[str, np.ndarray]:
    parts = [c[key] for c in collected if key in c]
    if not parts:
        return {f: np.array([]) for f in _POOL_FIELDS}
    return {f: np.concatenate([p[f] for p in parts]) for f in _POOL_FIELDS}


def run_universe(collected: Sequence[dict], fill: str = 'touch') -> dict:
    """Pool per-ticker arrays across the universe and judge the batch.

    Summarizes each ``(cell, direction, era)`` from the pooled per-event
    excess, then runs Benjamini-Yekutieli over the FULL-era (cell x direction)
    batch. Because significance flows through the single shared ``_summarize``
    (session-clustered bootstrap, so a market-wide day is one cluster), the
    pooled verdict is computed exactly the way the single-ticker one is. Any
    full-era survivor should then be read against its held-out (2022+)
    re-measurement, the out-of-sample arbiter.
    """
    keys: set = set()
    for c in collected:
        keys.update(c.keys())
    res = {}
    for k in sorted(keys):
        cell_key, direction, era = k
        r = _summarize(cell_key, fill, _pool(collected, k))
        r['direction'] = direction
        r['era'] = era
        res[k] = r
    full_keys = sorted(k for k in keys if k[2] == 'full')
    flags = benjamini_yekutieli([res[k].get('excess_p') for k in full_keys],
                                q=FDR_Q)
    for k, f in zip(full_keys, flags):
        res[k]['by_survivor'] = bool(f)
    return res


# The parallel driver: collect every ticker to a per-ticker .npz (resumable,
# the run_scan pattern), then pool the universe + each index into the committed
# summary. The scan dir is gitignored (regenerable); the JSON is the pinned
# artifact. Regenerating needs the ~15GB local archive.
UNIVERSE_SCAN_DIR = 'volume_profile_universe_scan'
UNIVERSE_RESULTS_FILE = 'volume_profile_universe_results.json'
_NPZ_SEP = '@@'


def _encode(collected: dict) -> dict:
    flat = {}
    for (cell, direction, era), arrs in collected.items():
        for field, a in arrs.items():
            flat[_NPZ_SEP.join((cell, direction, era, field))] = a
    return flat


def _decode(z) -> dict:
    out: dict = {}
    for name in z.files:
        cell, direction, era, field = name.split(_NPZ_SEP)
        out.setdefault((cell, direction, era), {})[field] = z[name]
    return out


def _universe_worker(ticker: str):
    """One ticker: collect and save its .npz. Top-level so it is picklable for
    the process pool; resumable, so a re-run skips finished tickers."""
    dest = os.path.join(data_path(UNIVERSE_SCAN_DIR), ticker + '.npz')
    if os.path.exists(dest):
        return (ticker, 'skip')
    collected = collect_ticker(ticker)
    if collected is None:
        return (ticker, 'no-archive')
    np.savez_compressed(dest, **_encode(collected))
    return (ticker, 'ok')


def _index_members() -> dict[str, set]:
    out = {}
    for snap in UNIVERSE_SNAPSHOTS:
        name = snap.replace('_tickers_2026-07.txt', '')
        with open(data_path(snap)) as fh:
            out[name] = {ln.strip() for ln in fh if ln.strip()}
    return out


def _load_scan(tickers: Sequence[str], out_dir: str) -> list:
    collected = []
    for t in tickers:
        p = os.path.join(out_dir, t + '.npz')
        if os.path.exists(p):
            with np.load(p, allow_pickle=False) as z:
                collected.append(_decode(z))
    return collected


def run_universe_campaign(n_proc: int | None = None) -> dict:
    """Run both experiments across the whole intraday universe and write the
    committed summary. Reproduces ``data/volume_profile_universe_results.json``
    (needs the local 15GB minute archive). Invoke via
    ``python -m engine.volume_profile_reversal universe``.
    """
    out_dir = data_path(UNIVERSE_SCAN_DIR)
    os.makedirs(out_dir, exist_ok=True)
    tickers = archived_tickers()
    n_proc = n_proc or max(1, min(10, (os.cpu_count() or 4) - 2))
    counts: dict = {}
    with Pool(n_proc, maxtasksperchild=1) as pool:
        for _, status in pool.imap_unordered(_universe_worker, tickers):
            counts[status] = counts.get(status, 0) + 1
    result: dict = {'n_tickers': len(tickers), 'counts': counts, 'pools': {}}
    pools = {'ALL': tickers}
    for name, names in _index_members().items():
        pools[name] = [t for t in tickers if t in names]
    for pname, plist in pools.items():
        res = run_universe(_load_scan(plist, out_dir))
        result['pools'][pname] = {
            'n_tickers': len(plist),
            'cells': {_NPZ_SEP.join(k): v for k, v in res.items()}}
    with open(data_path(UNIVERSE_RESULTS_FILE), 'w') as fh:
        json.dump(result, fh, indent=2, default=float)
    return result


def build_results() -> dict[str, Any]:
    """The committed block: full-era grid + the train/holdout confirmation."""
    bars = _load_nvda()
    if bars is None:
        return {'available': False}
    full = run_grid(bars, ERA_START, None, 'touch')
    full_barrier = run_grid(bars, ERA_START, None, 'barrier')
    holdout = run_holdout(bars)
    return {'available': True, 'ticker': TICKER, 'seed': VP_SEED,
                'full_touch': full, 'full_barrier': full_barrier, 'holdout': holdout}


def main() -> None:
    # ``universe`` runs the cross-sectional campaign (needs the 15GB archive)
    # and rewrites the committed summary; the default prints the NVDA
    # measurement, whose pinned surface is the live-run regression in the tests.
    if len(sys.argv) > 1 and sys.argv[1] == 'universe':
        run_universe_campaign()
        print('wrote', data_path(UNIVERSE_RESULTS_FILE))
        return
    res = build_results()
    if not res.get('available'):
        print('NVDA intraday archive absent — nothing to measure.')
        return
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
