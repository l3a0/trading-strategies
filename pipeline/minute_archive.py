"""The S&P 500 minute-archive data layer — access, adjustment, hygiene.

Everything here answers "which rows of the vendor tape can be trusted?",
a question about the DATA, not about any one study. It was born inside
``engine/cup_handle_scan.py`` because that scan was the first consumer;
it lives here now so a second study cannot end up with a second, quietly
divergent set of answers.

This is the minute-price analogue of ``pipeline/validate_dailies.py``,
which does the same job for the option-chain stores (classify a store,
propose the era-clip boundary, fail closed rather than auto-trust).

Layers:

- ACCESS — ``universe``, ``archive_path`` (a COMPLETE archive only),
  ``failed_tickers``, ``aggregate_daily`` (regular-session daily bars
  from minute rows, 09:30-16:00 inclusive, duplicate timestamps
  last-wins, size+mtime-keyed cache).
- ADJUSTMENT — ``load_splits`` / ``split_adjust`` over the committed
  ``data/sp500_splits_2026-07.csv`` snapshot (strictly-after products,
  volume scaled inversely).
- RULINGS — three owner-signed tables, each entry carrying its reason
  and sign-off date: ``RESOLVED_CLIFFS`` (a real market event, kept),
  ``TICKER_START_CLIPS`` (a predecessor company's era, cut from the
  front), ``TICKER_DROP_WINDOWS`` (a corrupt patch inside an otherwise
  clean history). Plus ``CROSSCHECK_AV_REFERENCE``, the names whose
  yfinance history carries a phantom back-adjustment its own event feed
  cannot undo.
- HYGIENE — ``cliff_flags`` (the [0.5, 2.0] guard on adjusted closes),
  ``coverage_diagnostic``, and the reference cross-check battery, which
  catches the failure the cliff guard is blind to by construction: a
  vendor era that is smooth, range-plausible and entirely FAKE.
- THE ENTRY POINT — ``load_clean_daily(ticker, splits)`` returns the
  adjusted series with every ruling applied, plus its coverage row.

The cross-check needs the network, so it NEVER gates ``load_clean_daily``
— it is a triage tool whose findings become owner-signed table entries.

Run:  python -m pipeline.minute_archive [--tickers A,B,...] [--crosscheck]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from typing import Any, Sequence

import numpy as np

from common.paths import data_path

CLIFF_BAND = (0.5, 2.0)
LATE_START_FLAG = '2010-01-01'
SPLITS_PATH = 'sp500_splits_2026-07.csv'
TICKERS_PATH = 'sp500_tickers_2026-07.txt'
WORKSPACE = 'sp500_intraday_1min'

# ------------------------------------------------------------------ §2 rulings
# The owner-signed hygiene rulings live in ``data/hygiene_rulings.csv`` — one
# row per ruling, each carrying its own ``reason`` (the event, the primary
# source, and the owner-signed date — the same depth the entries used to
# carry as code comments). Four kinds, rebuilt into the tables the rest of
# the module reads:
#   resolved_cliff — a REAL market event; the flagged date is exempt from
#     exclusion and the ticker scans normally. (Data errors are fixed at the
#     source instead: NVDA's 2001 split was re-dated in the committed
#     snapshot to 2001-09-17, the post-9/11 reopen where the halving shows in
#     as-traded prices — the recorded 09-10 ex-date fell inside the closure.)
#   start_clip — a predecessor era on a reused symbol; cut from the front.
#   drop_window — a corrupt vendor patch inside a clean history; dropped.
#   return_break — the price is real but a RETURN across it is fiction: a
#     spin-off or purge dividend where value left the share (see below).
#
# WHY A DATA FILE (2026-07-23): the tables outgrew hand-written dict literals
# once S&P 400 onboarding pushed them past ~180 rulings. A committed CSV
# keeps every reason (as a column, not a comment), diffs cleanly one row at a
# time, and makes a new ruling an appended line rather than a dict edit. The
# module rebuilds the identical dicts at import, so every consumer and test
# is unchanged (proven byte-identical when the file was introduced).
#
# Prices are for patterns; returns need distributions. This archive has NO
# distribution data, so RESOLVED_CLIFFS says "this move is real, keep the
# rows" while a return_break says "this move is real AND it is not a return."
# The kept cliffs that are NOT return breaks (AAPL's 2000 profit warning, AIG
# on Lehman Monday, PCG's bankruptcy filing) are genuine losses a holder
# took, and belong in a return series. A BLENDED event (a reverse split AND a
# spin-off on one day) is in BOTH the split table and the return-break rows —
# the two mechanisms handle the two halves.
#
# COVERAGE: the return_break rows the CLIFF GUARD surfaced are joined by the
# 118 rows in ``data/value_detachments_2026-07.csv`` (the sub-guard middle —
# 20-30% spin-offs the [0.5, 2.0] guard cannot see), which
# ``return_break_indices`` also reads. Still uncovered: a detachment in
# neither table (no vendor factor, too small to trip the guard) — an
# unbounded residue, so this is a floor on known contamination, not a
# guarantee of clean returns.
HYGIENE_RULINGS_PATH = 'hygiene_rulings.csv'


def load_hygiene_rulings() -> tuple[dict[tuple[str, str], str], dict[str, str],
                                    dict[str, list[tuple[str, str]]],
                                    dict[str, tuple[str, ...]]]:
    """Rebuild the four ruling tables from the committed CSV. For a
    resolved_cliff the ``reason`` IS the dict value (as it always was); for
    the other three kinds the value is the date/window/break-dates the engine
    acts on and the reason rides the CSV as the audit trail."""
    resolved: dict[tuple[str, str], str] = {}
    clips: dict[str, str] = {}
    drops: dict[str, list[tuple[str, str]]] = {}
    breaks: dict[str, list[str]] = {}
    with open(data_path(HYGIENE_RULINGS_PATH)) as f:
        for r in csv.DictReader(f):
            kind, t, d = r['kind'], r['ticker'], r['date']
            if kind == 'resolved_cliff':
                resolved[(t, d)] = r['reason']
            elif kind == 'start_clip':
                clips[t] = d
            elif kind == 'drop_window':
                drops.setdefault(t, []).append((d, r['end']))
            elif kind == 'return_break':
                breaks.setdefault(t, []).append(d)
            else:
                raise ValueError(
                    f'unknown ruling kind {kind!r} in {HYGIENE_RULINGS_PATH}')
    return (resolved, clips, drops,
            {t: tuple(v) for t, v in breaks.items()})


RESOLVED_CLIFFS, TICKER_START_CLIPS, TICKER_DROP_WINDOWS, RETURN_BREAKS = \
    load_hygiene_rulings()


DETACHMENTS_PATH = 'value_detachments_2026-07.csv'
_DETACHMENTS: dict[str, tuple[str, ...]] | None = None


def load_detachments() -> dict[str, tuple[str, ...]]:
    """The value detachments evicted from the split snapshot.

    These were being applied as SPLITS, which rescaled every price before
    them. Abbott is the clearest case: its real 2012-12-31 close was
    $65.47, and the AbbVie factor (2.0842) was rendering it $31.41 —
    every pre-2013 Abbott price in the archive was half the tape.

    Under the adjust-only-share-count-changes convention they stop
    adjusting anything, which puts the real step back on the tape and
    makes each date a return break instead. Cached; the file is a
    committed snapshot like the split table.
    """
    global _DETACHMENTS
    if _DETACHMENTS is None:
        out: dict[str, list[str]] = {}
        try:
            with open(data_path(DETACHMENTS_PATH)) as f:
                for row in csv.DictReader(f):
                    out.setdefault(row['ticker'], []).append(row['ex_date'])
        except FileNotFoundError:
            pass
        _DETACHMENTS = {k: tuple(sorted(v)) for k, v in out.items()}
    return _DETACHMENTS


def return_break_indices(ticker: str, dates: np.ndarray) -> np.ndarray:
    """Session indices of ``ticker``'s return breaks within ``dates``.

    A break at index ``j`` contaminates any return measured across it —
    i.e. any window ``(t, t + H]`` with ``t < j <= t + H``. Dates absent
    from the series (clipped away, or before the span) simply do not
    appear; a ticker with no breaks returns an empty array.

    Reads BOTH sources: the hand-signed ``RETURN_BREAKS`` (what the cliff
    guard surfaced) and the committed detachment snapshot (what the split
    reclassification surfaced, including the sub-guard middle).
    """
    wanted = set(RETURN_BREAKS.get(ticker, ()))
    wanted.update(load_detachments().get(ticker, ()))
    if not wanted:
        return np.empty(0, dtype=int)
    pos = {str(d): i for i, d in enumerate(dates)}
    return np.array(sorted(pos[w] for w in wanted if w in pos), dtype=int)


# ------------------------------------------------------------------ §2 data

def universe() -> list[str]:
    with open(data_path(TICKERS_PATH)) as f:
        return [ln.strip() for ln in f if ln.strip()]


# Every committed universe snapshot, in fetch order. Used ONLY to name the
# archives hygiene should sweep — NOT to widen any study's population.
UNIVERSE_SNAPSHOTS = (
    TICKERS_PATH,
    'nasdaq100_tickers_2026-07.txt',
    'sp400_tickers_2026-07.txt',
)


def archived_tickers() -> list[str]:
    """Every ticker we hold a COMPLETE archive for, across all committed
    universe snapshots.

    WHY THIS IS NOT ``universe()``: hygiene and study population are
    different questions that were accidentally sharing one answer. The
    cup-and-handle design freezes its universe at the S&P 500 snapshot
    (plan §2), so ``universe()`` must keep returning exactly that. But
    data hygiene has to cover every archive on disk regardless of which
    study wants it — and while the S&P 400 was landing, the sweep kept
    re-checking the same 501 S&P 500 names and reporting clean, with
    newly-fetched mid-caps never examined. The sweep was not failing; it
    was succeeding on the wrong set, which is worse.

    Completeness is ``archive_path``'s definition (the fetcher's
    ``.months.done`` marker, or a data-root archive), so an in-flight
    download is never swept. Order follows the snapshots, deduped, so a
    ticker in two indexes appears once.
    """
    seen: dict[str, None] = {}
    for snap in UNIVERSE_SNAPSHOTS:
        try:
            with open(data_path(snap)) as f:
                for ln in f:
                    t = ln.strip()
                    if t and t not in seen and archive_path(t) is not None:
                        seen[t] = None
        except FileNotFoundError:
            continue
    return list(seen)


def archive_path(ticker: str) -> str | None:
    """A COMPLETE archive only. A workspace csv without its fetcher
    completion marker (``.months.done``) is a partial, in-flight, or
    abandoned download and must never be scanned as history — the
    silent-gap failure §2 exists to prevent. The nine data-root archives
    predate the marker and are complete by construction.

    The batch script gzips each finished ticker IN PLACE (``gzip -9``
    keeps the csv until the gz is complete, then unlinks it), so while
    compression runs both files exist and the gz is TRUNCATED. The csv
    is therefore preferred whenever its marker vouches for it, and the
    workspace gz is trusted only with the marker too — the marker file
    is a sibling the gzip never touches."""
    stem = ticker.replace('.', '-').lower()
    ws_gz = data_path(f'{WORKSPACE}/{stem}_intraday_1min.csv.gz')
    ws_csv = data_path(f'{WORKSPACE}/{stem}_intraday_1min.csv')
    done = ws_csv + '.months.done'
    if os.path.exists(ws_csv) and os.path.exists(done):
        return ws_csv
    if os.path.exists(ws_gz) and os.path.exists(done):
        return ws_gz
    for root in (data_path(f'{stem}_intraday_1min.csv.gz'),
                 data_path(f'{stem}_intraday_1min.csv')):
        if os.path.exists(root):
            return root
    return None


def failed_tickers() -> set[str]:
    """§2: tickers the fetch gave up on — hard-excluded and listed."""
    p = data_path(f'{WORKSPACE}/failed_tickers.txt')
    if not os.path.exists(p):
        return set()
    with open(p) as f:
        return {ln.split()[0] for ln in f if ln.strip()}


def aggregate_daily(path: str, cache_dir: str | None = None) -> dict[str, np.ndarray]:
    """§2: regular-session daily bars from the minute archive. Bars with
    timestamps 09:30:00–16:00:00 inclusive; minute rows sorted, exact-
    duplicate timestamps collapsed to the LAST row; a session with zero
    regular bars contributes no daily bar.

    The §2 cache: the aggregate is written beside the archive (keyed to
    the source's size+mtime) and read back on later invocations, so the
    \~25 GB gz parse happens once, not per run."""
    if cache_dir is None and os.path.dirname(path):
        cache_dir = os.path.dirname(path)
    stat = os.stat(path)
    key = f'{stat.st_size}:{int(stat.st_mtime)}'
    cache = (os.path.join(cache_dir, os.path.basename(path).split('.')[0]
                          + '_daily_cache.csv') if cache_dir else None)
    if cache and os.path.exists(cache):
        with open(cache) as f:
            rows = list(csv.reader(f))
        if rows and rows[0] == ['#key', key]:
            cols = list(zip(*rows[2:])) if len(rows) > 2 else [[]] * 6
            return {'dates': np.array(cols[0]),
                    'open': np.array(cols[1], dtype=float),
                    'high': np.array(cols[2], dtype=float),
                    'low': np.array(cols[3], dtype=float),
                    'close': np.array(cols[4], dtype=float),
                    'volume': np.array(cols[5], dtype=float)}
    per_day: dict[str, dict[str, Any]] = {}
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        for row in csv.DictReader(f):
            ts = row['timestamp']
            t = ts[11:19]
            if not ('09:30:00' <= t <= '16:00:00'):
                continue
            d = ts[:10]
            slot = per_day.setdefault(d, {})
            slot[t] = row                    # duplicate timestamp: last wins
    dates, o, h, low, c, v = [], [], [], [], [], []
    for d in sorted(per_day):
        bars = [per_day[d][t] for t in sorted(per_day[d])]
        dates.append(d)
        o.append(float(bars[0]['open']))
        h.append(max(float(b['high']) for b in bars))
        low.append(min(float(b['low']) for b in bars))
        c.append(float(bars[-1]['close']))
        v.append(sum(float(b['volume']) for b in bars))
    out = {'dates': np.array(dates), 'open': np.array(o), 'high': np.array(h),
           'low': np.array(low), 'close': np.array(c), 'volume': np.array(v)}
    if cache:
        with open(cache, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['#key', key])
            w.writerow(['date', 'open', 'high', 'low', 'close', 'volume'])
            for i in range(len(dates)):
                w.writerow([dates[i], o[i], h[i], low[i], c[i], v[i]])
    return out


def load_splits(path: str | None = None) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    with open(path or data_path(SPLITS_PATH)) as f:
        for row in csv.DictReader(f):
            out.setdefault(row['ticker'], []).append(
                (row['ex_date'], float(row['ratio'])))
    return out


def split_adjust(d: dict[str, np.ndarray], splits: Sequence[tuple[str, float]],
                 ) -> dict[str, np.ndarray]:
    """§2: each as-traded price divided by the product of ratios of splits
    dated STRICTLY AFTER that day; volume multiplied by the same factor."""
    factor = np.ones(len(d['dates']))
    for ex_date, ratio in splits:
        before = d['dates'] < ex_date
        factor[before] *= ratio
    out = dict(d)
    for k in ('open', 'high', 'low', 'close'):
        out[k] = d[k] / factor
    out['volume'] = d['volume'] * factor
    return out


def cliff_flags(adjusted_close: np.ndarray, dates: np.ndarray,
                splits: Sequence[tuple[str, float]]) -> list[str]:
    """§2 cliff guard, run on ADJUSTED closes. A correctly-committed split
    leaves NO cliff after adjustment, so there is no split-day exemption:
    an out-of-band ratio remaining ON a split day means the committed
    ratio failed to explain the move — exactly what must be flagged.
    (``splits`` stays in the signature as the audit trail of what was
    already applied.)"""
    del splits                    # already applied upstream; no exemption
    flags = []
    for i in range(1, len(adjusted_close)):
        r = adjusted_close[i] / adjusted_close[i - 1]
        if not (CLIFF_BAND[0] <= r <= CLIFF_BAND[1]):
            flags.append(str(dates[i]))
    return flags


def coverage_diagnostic(ticker: str, d: dict[str, np.ndarray],
                        flags: list[str],
                        calendar: set[str] | None = None) -> dict[str, Any]:
    """§2: first/last session, session count vs. the trading calendar over
    the ticker's own span (the mid-span-hole detector the rename lesson
    demands — ``calendar`` is the union of all scanned tickers' sessions,
    attached by run_scan), the cliff flags, and the late-start flag."""
    first = str(d['dates'][0]) if len(d['dates']) else None
    last = str(d['dates'][-1]) if len(d['dates']) else None
    expected = missing = None
    if calendar is not None and first is not None:
        expected = sum(1 for day in calendar if first <= day <= last)
        missing = expected - int(len(d['dates']))
    return {
        'ticker': ticker, 'first': first, 'last': last,
        'sessions': int(len(d['dates'])),
        'expected_sessions': expected,
        'missing_sessions': missing,
        'late_start': bool(first and first > LATE_START_FLAG),
        'cliff_flags': flags,
    }


# ----------------------------------------------- §2 reference cross-check
#
# The ELV lesson (owner-signed 2026-07-21): a vendor era can be smooth,
# range-plausible, and entirely FAKE — ELV's 2006-2010 tape ran up to 49%
# off the real company with no single-day jump, so the cliff guard is
# blind to it by construction. The only defense is comparing against an
# independent daily reference. This battery is REPORTING ONLY: it needs
# the network, so it never gates run_scan (which stays deterministic) —
# a flagged ticker goes to triage and exclusion happens through the
# owner-signed tables above, like every other ruling.

CROSSCHECK_TOL = 0.02      # per-day |ours/ref - 1| beyond this = mismatch
CROSSCHECK_SEVERE = 0.05   # ... beyond this = severe (scale-level, not noise)
# Flag thresholds, calibrated on the first full sweep (2026-07-21, 184
# tickers): benign noise — our close is the last minute-bar trade at or
# before 16:00, the reference's is the official auction close, and on
# wild days (2008-09, 2020-03) they differ >2% for up to 5 straight
# sessions; the noisiest honest name (BX) logged 270 scattered days but
# never past ~3% and never a run over 4. Every structural fake era ran
# 28..2,672 consecutive days at scale-level offsets. So: a LONG RUN
# flags regardless of size (the CCI shape, a sustained ~3% era), and
# the total-days backstop counts only SEVERE days so honest-but-noisy
# names never flag on accumulated jitter.
CROSSCHECK_MAX_RUN = 10    # flag at this many consecutive mismatch days
CROSSCHECK_MAX_SEVERE = 20  # ... or this many severe days overall


def unsplit_reference(closes: np.ndarray, dates: np.ndarray,
                      split_events: list[tuple[str, float]]) -> np.ndarray:
    """Reconstruct the reference's AS-TRADED closes. yfinance
    split-adjusts its Close column even with auto_adjust=False (the XLE
    lesson), so multiply each close back up by the product of the
    REFERENCE'S OWN split factors dated strictly after the row. Using
    the reference's own events keeps the comparison
    convention-independent — our owner-signed snapshot never enters, so
    a snapshot ruling (the EXPE removal) cannot mask a real mismatch."""
    out = np.asarray(closes, dtype=float).copy()
    for sdate, factor in split_events:
        if factor:
            out[dates < sdate] *= factor
    return out


def crosscheck_series(our_dates: np.ndarray, our_closes: np.ndarray,
                      ref_dates: np.ndarray, ref_closes: np.ndarray,
                      ref_splits: list[tuple[str, float]],
                      tol: float = CROSSCHECK_TOL) -> dict[str, Any]:
    """Pure comparison core (network-free, tested synthetically). Days
    the reference lacks are counted ``unreferenced`` — they neither
    match nor mismatch, and they do not reset a mismatch run (a
    reference hole must not split one bad era into two short runs)."""
    ref_as_traded = unsplit_reference(ref_closes, ref_dates, ref_splits)
    ref_map = dict(zip(ref_dates.tolist(), ref_as_traded.tolist()))
    mismatches: list[str] = []
    matched = unreferenced = severe = run = max_run = 0
    for x, c in zip(our_dates.tolist(), our_closes.tolist()):
        r = ref_map.get(x)
        if r is None or r <= 0:
            unreferenced += 1
            continue
        off = abs(c / r - 1)
        if off > tol:
            mismatches.append(x)
            severe += off > CROSSCHECK_SEVERE
            run += 1
            max_run = max(max_run, run)
        else:
            matched += 1
            run = 0
    return {
        'compared': matched + len(mismatches),
        'mismatch_days': len(mismatches),
        'severe_days': severe,
        'mismatch_first': mismatches[0] if mismatches else None,
        'mismatch_last': mismatches[-1] if mismatches else None,
        'max_run': max_run,
        'unreferenced_days': unreferenced,
        'flagged': (max_run >= CROSSCHECK_MAX_RUN
                    or severe >= CROSSCHECK_MAX_SEVERE),
    }


# Tickers whose YFINANCE history carries a phantom back-adjustment that
# its own event feed cannot un-adjust (verified against a primary source
# to the penny, 2026-07-21): BLDR — a Dec-2009 rights-offering factor
# (x0.841) baked into every earlier "raw" close with no recorded event;
# CCI — a x0.970 factor before 2002-05-29, no corporate action exists
# there; HWM — a constant x0.767 before 2020-04-01 (the Arconic Corp
# spinoff, 1 new share per 4), our as-traded $18.92 Arconic close on
# 2016-11-01 externally anchored to a contemporaneous Forbes/Spin-Off
# Research note. For these, the battery uses Alpha Vantage's
# TIME_SERIES_DAILY as the reference instead (as-traded). Same-vendor
# caveat acknowledged: AV daily vs AV minutes is a self-consistency
# check, but each name's as-traded price was externally anchored to a
# primary source first.
CROSSCHECK_AV_REFERENCE = {'BLDR', 'CCI', 'HWM'}


def fetch_reference(ticker: str) -> dict[str, Any] | None:
    """Daily closes + split events from yfinance (network)."""
    import yfinance as yf
    h = yf.Ticker(ticker.replace('.', '-')).history(period='max',
                                                    auto_adjust=False)
    if h is None or not len(h):
        return None
    return {
        'dates': np.array([x.strftime('%Y-%m-%d') for x in h.index]),
        'closes': h['Close'].to_numpy(dtype=float),
        'splits': [(x.strftime('%Y-%m-%d'), float(s))
                   for x, s in h['Stock Splits'].items() if s],
    }


def fetch_reference_av(ticker: str) -> dict[str, Any] | None:
    """As-traded daily closes from Alpha Vantage (network; needs
    ALPHAVANTAGE_API_KEY in the environment). No split reconstruction —
    the series is as-traded already, so ``splits`` is empty."""
    key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not key:
        return None
    import urllib.request
    url = ('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY'
           f'&symbol={ticker.replace(".", "-")}&outputsize=full'
           f'&datatype=csv&apikey={key}')
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode()
    rows = [ln.split(',') for ln in text.strip().splitlines()[1:]
            if ln.count(',') >= 4]
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    return {
        'dates': np.array([r[0] for r in rows]),
        'closes': np.array([float(r[4]) for r in rows]),
        'splits': [],
    }


def crosscheck_ticker(ticker: str) -> dict[str, Any] | None:
    """Cross-check one archive against the reference, AFTER the
    owner-signed clips/drops (ruled-out eras must not re-flag; what we
    scan is what gets checked)."""
    path = archive_path(ticker)
    if path is None:
        return None
    d = aggregate_daily(path)
    if not len(d['dates']):
        return None
    clip = TICKER_START_CLIPS.get(ticker)
    if clip is not None:
        keep = d['dates'] >= clip
        d = {k: v[keep] for k, v in d.items()}
    for w0, w1 in TICKER_DROP_WINDOWS.get(ticker, []):
        keep = (d['dates'] < w0) | (d['dates'] > w1)
        d = {k: v[keep] for k, v in d.items()}
    if not len(d['dates']):
        return None
    fetch = (fetch_reference_av if ticker in CROSSCHECK_AV_REFERENCE
             else fetch_reference)
    ref = fetch(ticker)
    if ref is None:
        return {'ticker': ticker, 'flagged': None, 'note': 'no reference data'}
    out = crosscheck_series(d['dates'], d['close'],
                            ref['dates'], ref['closes'], ref['splits'])
    out['ticker'] = ticker
    return out


def run_crosscheck(tickers: Sequence[str]) -> list[dict[str, Any]]:
    failed = failed_tickers()
    rows: list[dict[str, Any]] = []
    for t in tickers:
        if t in failed:
            continue
        try:
            r = crosscheck_ticker(t)
        except (EOFError, FileNotFoundError, OSError):
            continue  # archive mid-move (rolling gzip); next pass gets it
        if r is not None:
            rows.append(r)
    return rows



# ------------------------------------------------------------ entry point

def load_clean_daily(ticker: str,
                     splits: dict[str, list[tuple[str, float]]],
                     ) -> tuple[dict[str, np.ndarray] | None,
                                dict[str, Any] | None]:
    """The adjusted daily series with every owner-signed ruling applied,
    plus its coverage row. Returns (None, None) when the archive is
    absent/empty or a clip removes everything.

    Order matters: adjust, then clip the predecessor era, then drop
    corrupt windows, THEN flag cliffs — so a ruled-out era cannot raise
    a flag, and a flag can only come from retained rows."""
    path = archive_path(ticker)
    if path is None:
        return None, None
    d = aggregate_daily(path)
    if not len(d['dates']):
        return None, None
    adj = split_adjust(d, splits.get(ticker, []))
    clip = TICKER_START_CLIPS.get(ticker)
    if clip is not None:
        keep = adj['dates'] >= clip
        adj = {k: v[keep] for k, v in adj.items()}
        if not len(adj['dates']):
            return None, None
    dropped = TICKER_DROP_WINDOWS.get(ticker, [])
    for w0, w1 in dropped:
        keep = (adj['dates'] < w0) | (adj['dates'] > w1)
        adj = {k: v[keep] for k, v in adj.items()}
    flags = cliff_flags(adj['close'], adj['dates'], splits.get(ticker, []))
    # excluded until resolved by hand — an owner-signed RESOLVED_CLIFFS
    # entry clears its date; only UNRESOLVED flags exclude
    unresolved = [f for f in flags if (ticker, f) not in RESOLVED_CLIFFS]
    cov = coverage_diagnostic(ticker, adj, unresolved)
    cov['resolved_cliffs'] = [f for f in flags if (ticker, f) in RESOLVED_CLIFFS]
    cov['start_clip'] = clip
    cov['drop_windows'] = dropped
    return adj, cov


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default=None,
                    help='comma-separated subset (default: the committed universe)')
    ap.add_argument('--all-archives', action='store_true',
                    help='sweep EVERY complete archive across all committed '
                         'universe snapshots, not just the S&P 500 list. '
                         'This is the hygiene sweep; it does NOT widen any '
                         "study's frozen population")
    ap.add_argument('--crosscheck', action='store_true',
                    help='reference cross-check battery (network; reporting '
                         'only — load_clean_daily never touches the network)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    if a.tickers:
        tks = a.tickers.split(',')
    elif a.all_archives:
        tks = archived_tickers()
    else:
        tks = universe()
    if a.crosscheck:
        rows = run_crosscheck(tks)
        flagged = [r for r in rows if r.get('flagged')]
        noref = [r['ticker'] for r in rows if r.get('flagged') is None]
        if a.json:
            print(json.dumps({'rows': rows}, default=str))
        else:
            print(f"crosschecked {len(rows)} tickers: {len(flagged)} flagged, "
                  f"{len(noref)} without reference data {noref}")
            for r in flagged:
                print(f"  FLAG {r['ticker']}: {r['mismatch_days']} mismatch "
                      f"days ({r['mismatch_first']}..{r['mismatch_last']}, "
                      f"max run {r['max_run']}), "
                      f"{r['unreferenced_days']} unreferenced")
            big_unref = [(r['ticker'], r['unreferenced_days']) for r in rows
                         if r.get('flagged') is False
                         and r['unreferenced_days'] > 25]
            if big_unref:
                print('  large unreferenced spans (predecessor-era smell):',
                      big_unref)
        raise SystemExit(0)
    splits = load_splits()
    rows = []
    for t in tks:
        adj, cov = load_clean_daily(t, splits)
        if cov is not None:
            rows.append(cov)
    if a.json:
        print(json.dumps({'coverage': rows}, default=str))
    else:
        print(f'loaded {len(rows)} tickers')
        for c in rows:
            if c['cliff_flags'] or c['late_start']:
                print(f"  FLAG {c['ticker']}: first={c['first']} "
                      f"cliffs={c['cliff_flags'][:3]}")
