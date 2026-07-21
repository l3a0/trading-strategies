"""The cup-and-handle scan — the frozen docs/cup_handle_scan_plan.md build.

O'Neil's flagship pattern, detected by frozen mechanical rules across the
S&P 500 minute archive and judged at the calendar-day-cluster level
against stratified matched-count random-entry nulls.

Layer map (every §ref is the plan doc):

- §2 data: ``aggregate_daily`` (regular-session daily bars from minute
  rows, 09:30–16:00 inclusive, duplicate timestamps last-wins),
  ``split_adjust`` (the committed ``data/sp500_splits_2026-07.csv``
  snapshot, strictly-after products, volume scaled inversely, the
  [0.5, 2.0] cliff guard), ``coverage_diagnostic``.
- §3 detector: ``detect_cup_handle`` — the frozen iteration (every
  session past behavioral warm-up, chronological; handle lengths tried
  ascending 5..25; the FIRST passing window is THE detection; a
  rim-band failure rejects the window, never retries a second-best
  left rim; next candidate after a detection is t+25).
- §5 evaluation: ``build_trades`` (per-ticker flat-only, H lockout,
  end-of-span skip), ``build_clusters`` (same-entry-date pooling,
  equal-weight member returns), ``cluster_null_p`` (per-stratum
  matched-count draws with the frozen stream derivation
  ``f'{CUP_SEED}:{variant}|H{H}|{stratum}'``), the survival read.

The archive is personal and gitignored, so the result pins are
DATASET-GATED (skip in CI); the synthetic battery always runs. Nothing
here computes a return before the §4 validity gate has run — the
``__main__`` report prints detection counts and coverage FIRST and the
evaluation only behind ``--evaluate`` (the §10 step-4 switch).

Run:  python -m engine.cup_handle_scan [--tickers A,B,...] [--evaluate] [--json]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from typing import Any, Sequence

import numpy as np

from common.paths import data_path

CUP_SEED = 20260720
B_RESAMPLES = 10_000
HORIZONS = (5, 10, 15, 20, 60, 120)
HEADLINE_HORIZONS = (20, 60)
SURVIVAL_P = 0.01
MIN_CLUSTERS = 100
HANDLE_MIN, HANDLE_MAX = 5, 25
CUP_MIN, CUP_MAX = 35, 325
UPTREND_LOOKBACK = 90
UPTREND_MIN = 1.3
DEPTH_MIN, DEPTH_MAX = 0.12, 0.33
RIM_BAND = (0.85, 1.05)
HANDLE_DEPTH_MAX = 0.15
ROUNDNESS_MIN = 0.15
INTERIOR_TOL = 1.02
VOL_SURGE = 1.5                 # O'Neil via the committed Tharp notes, Loc 5559
VOL_AVG_WINDOW = 50
DEDUP_SKIP = 25
CLIFF_BAND = (0.5, 2.0)
LATE_START_FLAG = '2010-01-01'
ERAS = (('1999-01-01', '2009-12-31'), ('2010-01-01', '2019-12-31'),
        ('2020-01-01', '2026-12-31'))
SPLITS_PATH = 'sp500_splits_2026-07.csv'
TICKERS_PATH = 'sp500_tickers_2026-07.txt'
WORKSPACE = 'sp500_intraday_1min'

# §2 hand-resolutions, owner-signed 2026-07-21: cliff flags confirmed as
# REAL market events — the flagged date is exempt from exclusion and the
# ticker scans normally. (Data errors are fixed at the source instead:
# NVDA's 2001 split was re-dated in the committed snapshot to 2001-09-17,
# the post-9/11 reopen where the halving actually shows in as-traded
# prices — the recorded 09-10 ex-date fell inside the four-day closure.)
RESOLVED_CLIFFS = {
    ('AAPL', '2000-09-29'):
        'real event: the Sept-2000 profit-warning crash, ~-52% in one session',
    ('AIG', '2008-09-15'):
        'real event: Lehman Monday, ~-60% in the bailout week (owner-signed '
        '2026-07-21)',
    ('APA', '2020-03-09'):
        'real event: the COVID/OPEC oil-price-war Black Monday, ~-55% — '
        "Apache's worst single day (owner-signed 2026-07-21)",
}

# §2 hand-resolutions, owner-signed 2026-07-21: START CLIPS for tickers
# whose archive stitches a PREDECESSOR company's history under the current
# symbol (the QQQQ-rename poison). Rows before the clip are dropped before
# anything else runs.
TICKER_START_CLIPS = {
    # pre-2019 rows are Bemis (~$56) wearing the AMCR symbol; the current
    # Amcor plc began NYSE trading 2019-06-11 (~$11)
    'AMCR': '2019-06-11',
    # pre-gap rows are American Apparel (bankrupt 2015-10-05, last session
    # 2015-10-02 at ~$0.11); AppLovin IPO'd on the vacated symbol
    # 2021-04-15 (owner-signed 2026-07-21)
    'APP': '2021-04-15',
    # pre-2019 rows are Axovant Sciences (biotech; its trial-failure
    # crashes are real but the wrong company's); Axovant vacated the
    # symbol at the 2019-02-14 open and the vendor tape carries Axon
    # Enterprise from that day (owner-signed 2026-07-21)
    'AXON': '2019-02-14',
    # pre-2026 rows are the BlackRock New York Municipal Income Trust
    # (~$10 closed-end fund, merged away 2026-02); BNY Mellon moved its
    # stock from BK to BNY effective 2026-05-21 (owner-signed 2026-07-21)
    'BNY': '2026-05-21',
    # pre-2022-07 rows are the ORIGINAL Coherent Inc (acquired by II-VI
    # at $266/share deal value, last day 2022-06-30); the tape carries
    # the surviving II-VI/Coherent Corp from the close date — also
    # neutralizes the snapshot's 2011-06-27 split, which is II-VI's,
    # not old Coherent's (owner-signed 2026-07-21)
    'COHR': '2022-07-01',
    # pre-gap rows are Converted Organics (fertilizer penny stock, gone
    # dark 2013); Coinbase direct-listed on the vacated symbol
    # 2021-04-14 (owner-signed 2026-07-21)
    'COIN': '2021-04-14',
    # pre-gap rows are C2C CrowdFunding (OTC shell, registration
    # revoked 2015); CrowdStrike IPO'd 2019-06-12, first close $58.00
    # matching the tape (owner-signed 2026-07-21)
    'CRWD': '2019-06-12',
    # pre-gap rows are old Dell Inc (private 2013-10-29) plus two
    # when-issued days distorted by the snapshot's DVMT-lineage 1.806
    # factor; regular-way Dell Technologies trading began 2018-12-28
    # (owner-signed 2026-07-21)
    'DELL': '2018-12-28',
    # pre-2013 rows are EnergySolutions (taken private at $4.15/share,
    # last day 2013-05-24); the tape carries Northeast Utilities /
    # Eversource from the vacancy (owner-signed 2026-07-21)
    'ES': '2013-05-28',
    # the vendor's pre-2011 ELV tape is FICTION: 2000-2005 matches no
    # real security in the Anthem/WellPoint lineage (SEC 10-K price
    # tables prove it), 2006-2010 is a smooth-but-wrong blend (1,077
    # sessions >2% off the reference, up to 49%); the tape matches the
    # real company exactly from 2010-12-17 onward — supersedes the
    # earlier 2005 drop-window ruling (owner-signed 2026-07-21)
    'ELV': '2010-12-17',
    # ---- the reference-cross-check catch (owner-signed 2026-07-21) ----
    # these seven surfaced only because the day-by-day reference check
    # sees smooth wrong-company eras the cliff guard is blind to.
    # pre-2013 rows are Michael Baker Corp (engineering, taken private
    # at $40.50, last day 2013-10-11); the tape back-stitches the Baker
    # Hughes lineage from the vacancy (and only actually has rows from
    # 2017-07, when the merged BHGE listed)
    'BKR': '2013-10-14',
    # pre-2016 rows are the OLD Chubb Corp, absorbed by ACE after the
    # 2016-01-14 close; the tape carries ACE-renamed-Chubb Limited from
    # 2016-01-15 (NYSE Form 25-NSE pins the boundary)
    'CB': '2016-01-15',
    # pre-2024 rows are Physicians Realty Trust (real, merged into
    # Healthpeak at 0.674/share); Healthpeak assumed DOC at the
    # 2024-03-04 open (the tape correctly has no rows on the 03-01
    # PEAK-final day)
    'DOC': '2024-03-04',
    # pre-2012 rows are the OLD Constellation Energy Group (merged into
    # Exelon 2012-03-12); the NEW Constellation spun off from Exelon
    # and began regular Nasdaq trading 2022-02-02 — the clip also drops
    # ten thin when-issued prints (2022-01-19..02-01) that traded under
    # the temporary CEGVV symbol
    'CEG': '2022-02-02',
    # pre-2006 rows are a nearly-untraded NASDAQ ADR line (ticker
    # CRHCY, stale quotes, at least one fabricated print); CRH's US
    # line moved to NYSE as CRH on 2006-03-31 (8-A12B); the modern era
    # then clears the cross-check with only scattered noise
    'CRH': '2006-03-31',
    # the FOX tape is FOUR issuers spliced: old News Corp B (misfiled —
    # it traded as NWS), two weeks of new-News-Corp-B when-issued
    # prints from the 2013 split, 21st Century Fox B (2013-2019), then
    # the real Fox Corporation B from 2019-03-19 (its first regular-way
    # session under FOX, and where the vendor's own daily series starts)
    'FOX': '2019-03-19',
}

# §2 hand-resolutions, owner-signed 2026-07-21: DROP WINDOWS — spans
# (inclusive) where the vendor minute tape is corrupt inside an
# otherwise clean history. Both rulings were verified to join
# seamlessly after the drop (ECL $159.13 -> $159.16; ELV $67.71 ->
# $68.90). Distinct from RESOLVED_CLIFFS (a real move, kept) and
# TICKER_START_CLIPS (a predecessor era, cut from the front).
TICKER_DROP_WINDOWS = {
    # 2019-02-05 is Ecopetrol's (EC — one character off) tape filed
    # under ECL: the corrupt $18.92 close matches EC to the cent while
    # Ecolab really traded ~$159 that day
    'ECL': [('2019-02-05', '2019-02-05')],
    # 2000-04-07 is a corrupt vendor day: the minute tape closed $20.81
    # but Alpha Vantage's daily series shows $48.31, bracketed by
    # $46.38 and $50.50 — a -55%-and-back round trip that never traded
    # (the Ecolab disease; owner-signed 2026-07-21)
    'FAST': [('2000-04-07', '2000-04-07')],
}


# ------------------------------------------------------------------ §2 data

def universe() -> list[str]:
    with open(data_path(TICKERS_PATH)) as f:
        return [ln.strip() for ln in f if ln.strip()]


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
# its own event feed cannot un-adjust (verified against SEC 10-K cover
# prices to the penny, 2026-07-21): BLDR — a Dec-2009 rights-offering
# factor (x0.841) baked into every earlier "raw" close with no recorded
# event; CCI — a x0.970 factor before 2002-05-29, no corporate action
# exists there. For these, the battery uses Alpha Vantage's TIME_SERIES_
# DAILY as the reference instead (as-traded, matched SEC prices exactly
# in the same verification). Same-vendor caveat acknowledged: AV daily
# vs AV minutes is a self-consistency check, but the AV daily series
# for BOTH names was externally anchored to SEC filings first.
CROSSCHECK_AV_REFERENCE = {'BLDR', 'CCI'}


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


# -------------------------------------------------------------- §3 detector

def _ols_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def _earliest_argmax(a: np.ndarray) -> int:
    return int(np.argmax(a))     # numpy argmax returns the FIRST maximum


def _earliest_argmin(a: np.ndarray) -> int:
    return int(np.argmin(a))


def _try_window(c: np.ndarray, v: np.ndarray, t: int, length: int,
                vol_surge: float = VOL_SURGE) -> dict[str, Any] | None:
    """Rules 1–6 for one (t, handle-length) candidate. Returns the recorded
    anatomy or None. All indices are session positions in the ticker's own
    series; any reference before index 0 rejects (behavioral warm-up)."""
    h0 = t - length
    if h0 < 0:
        return None
    w_close = c[h0:t]
    # rule 1 (geometry): handle high = highest close, earliest tie, index in
    # the first third (zero-based i with 3i < len)
    i = _earliest_argmax(w_close)
    if 3 * i >= length:
        return None
    r = h0 + i
    handle_high = c[r]
    if _ols_slope(w_close) > 0:
        return None
    if np.min(w_close) < (1.0 - HANDLE_DEPTH_MAX) * handle_high:
        return None
    # rule 2 (cup)
    lo_edge, hi_edge = r - CUP_MAX, r - CUP_MIN
    if lo_edge < 0:
        return None
    left = lo_edge + _earliest_argmax(c[lo_edge:hi_edge + 1])
    if not (RIM_BAND[0] * c[left] <= c[r] <= RIM_BAND[1] * c[left]):
        return None       # rim-band failure rejects the window; no 2nd-best left rim
    # left <= r - CUP_MIN always, so (left, r) is never degenerate
    b = left + 1 + _earliest_argmin(c[left + 1:r])
    rim_max = max(c[left], c[r])
    depth = (rim_max - c[b]) / rim_max
    if not (DEPTH_MIN <= depth <= DEPTH_MAX):
        return None
    pos = (b - left) / (r - left)
    if not (0.2 < pos < 0.8):
        return None
    if b + 1 < r and np.max(c[b + 1:r]) > INTERIOR_TOL * c[r]:
        return None
    # rule 3 (roundness, primary)
    threshold = c[b] + 0.25 * (rim_max - c[b])
    frac = float(np.mean(c[left:r + 1] <= threshold))
    if frac < ROUNDNESS_MIN:
        return None
    # rule 4 (handle in the upper half)
    if np.min(w_close) < c[b] + 0.5 * (rim_max - c[b]):
        return None
    # rule 5 (prior uptrend)
    if left - UPTREND_LOOKBACK < 0:
        return None
    if c[left] < UPTREND_MIN * np.min(c[left - UPTREND_LOOKBACK:left]):
        return None
    # rule 6 (trigger)
    if t - VOL_AVG_WINDOW < 0:
        return None
    if not (c[t] > handle_high):
        return None
    if v[t] < vol_surge * np.mean(v[t - VOL_AVG_WINDOW:t]):
        return None
    # rule 1's volume clause (needs the cup): handle volume below cup volume
    if np.mean(v[h0:t]) >= np.mean(v[left:r + 1]):
        return None
    return {'t': t, 'h0': h0, 'r': r, 'l': left, 'b': b,
            'depth': round(depth, 4), 'roundness': round(frac, 4)}


def detect_cup_handle(c: np.ndarray, v: np.ndarray,
                      *, use_volume_trigger: bool = True) -> list[dict[str, Any]]:
    """§3, the frozen iteration: t chronological; handle lengths ascending
    5..25; the FIRST passing window is THE detection; after a detection the
    next candidate is t + 25. ``use_volume_trigger=False`` is the §5
    ablation (rule 6's volume clause skipped; everything else identical)."""
    surge = VOL_SURGE if use_volume_trigger else 0.0
    out: list[dict[str, Any]] = []
    t = DETECT_FLOOR                # identical to the null's eligibility floor
    n = len(c)
    while t < n:
        hit = None
        for length in range(HANDLE_MIN, HANDLE_MAX + 1):
            hit = _try_window(c, v, t, length, vol_surge=surge)
            if hit is not None:
                break
        if hit is not None:
            out.append(hit)
            t += DEDUP_SKIP
        else:
            t += 1
    return out


# ------------------------------------------------------------ §5 evaluation

def build_trades(detections: Sequence[int], horizon: int, n_days: int) -> list[int]:
    """Per-ticker flat-only book (the harness convention): H-session
    lockout, re-entry at the exit close, end-of-span skip."""
    entries: list[int] = []
    next_ok = -1
    for t in detections:
        if t >= next_ok and t + horizon < n_days:
            entries.append(int(t))
            next_ok = t + horizon
    return entries


def stratum_of(date: str, horizon: int) -> str:
    y, m = date[:4], int(date[5:7])
    if horizon <= 20:
        return f'{y}-{m:02d}'
    if horizon == 60:
        return f'{y}Q{(m - 1) // 3 + 1}'
    return f'{y}H{1 if m <= 6 else 2}'


def build_clusters(trades: dict[str, list[int]],
                   data: dict[str, dict[str, np.ndarray]],
                   horizon: int) -> list[dict[str, Any]]:
    """§5: all traded entries sharing one calendar date form one
    cluster-trade; return = equal-weight mean of member simple returns."""
    by_date: dict[str, list[tuple[str, float]]] = {}
    for ticker, entries in trades.items():
        c = data[ticker]['close']
        dates = data[ticker]['dates']
        for t in entries:
            ret = float(c[t + horizon] / c[t] - 1.0)
            by_date.setdefault(str(dates[t]), []).append((ticker, ret))
    return [{'date': d, 'members': [m for m, _ in mem],
             'ret': float(np.mean([r for _, r in mem])),
             'stratum': stratum_of(d, horizon)}
            for d, mem in sorted(by_date.items())]


# The hard floor a detection's own guards imply: the cup window needs
# r >= CUP_MAX and the handle needs t >= r + HANDLE_MIN, so no detection
# can occur before session 330. The detector STARTS its iteration here and
# the null draws only from here — the two domains are identical by
# construction (the plan's behavioral warm-up rule governs; its "near 440"
# aside over-counted by adding the uptrend lookback, which bounds l, not t).
DETECT_FLOOR = CUP_MAX + HANDLE_MIN


def cluster_null_p(clusters: list[dict[str, Any]],
                   data: dict[str, dict[str, np.ndarray]], horizon: int,
                   variant: str = 'primary', b: int = B_RESAMPLES,
                   ) -> dict[str, Any]:
    """§5's matched-count cluster null. Per stratum with k clusters, each
    resample draws k distinct sessions from the union calendar of that
    stratum; drawn days (in draw order) pair with the stratum's real
    clusters sorted by (member count desc, date asc); each member's
    H-session return is taken from the drawn day in ITS OWN session
    indexing, dropped (and counted) if ineligible there."""
    if not clusters:
        return {'n_clusters': 0}
    date_index = {t: {str(d): i for i, d in enumerate(v['dates'])}
                  for t, v in data.items()}
    strata: dict[str, list[dict[str, Any]]] = {}
    for cl in clusters:
        strata.setdefault(cl['stratum'], []).append(cl)
    all_days: dict[str, set[str]] = {}
    for t, v in data.items():
        for d in v['dates']:
            all_days.setdefault(stratum_of(str(d), horizon), set()).add(str(d))

    obs_rate = float(np.mean([c['ret'] > 0 for c in clusters]))
    obs_mean = float(np.mean([c['ret'] for c in clusters]))
    n_cl = len(clusters)
    # per-resample accumulators: wins, return sum, and the count of
    # SURVIVING (non-empty) null clusters — a fully-diluted null cluster is
    # renormalized away, never scored as a loss (the anti-conservative bias
    # the review caught); the empty count is reported beside the dilution
    win_acc = np.zeros(b)
    ret_acc = np.zeros(b)
    live_acc = np.zeros(b)
    diluted = 0
    empty_clusters = 0
    for stratum, cls in strata.items():
        days = sorted(all_days[stratum])
        rng = np.random.default_rng(int(hashlib.sha256(
            f'{CUP_SEED}:{variant}|H{horizon}|{stratum}'.encode()
        ).hexdigest()[:12], 16))
        ordered = sorted(cls, key=lambda c: (-len(c['members']), c['date']))
        k = len(ordered)
        for i in range(b):
            picks = rng.permutation(len(days))[:k]
            for cl, pi in zip(ordered, picks):
                d = days[pi]
                rets = []
                for m in cl['members']:
                    idx = date_index[m].get(d)
                    if (idx is None or idx < DETECT_FLOOR
                            or idx + horizon >= len(data[m]['close'])):
                        diluted += 1
                        continue
                    cm = data[m]['close']
                    rets.append(float(cm[idx + horizon] / cm[idx] - 1.0))
                if rets:
                    r = float(np.mean(rets))
                    win_acc[i] += r > 0
                    ret_acc[i] += r
                    live_acc[i] += 1
                else:
                    empty_clusters += 1
    live = np.maximum(live_acc, 1.0)
    null_rate = win_acc / live
    null_mean = ret_acc / live
    p = (1 + int(np.sum(null_rate >= obs_rate))) / (1 + b)
    return {
        'n_clusters': n_cl, 'win_rate': round(obs_rate, 4),
        'mean_ret': round(obs_mean, 5),
        'null_rate_mean': round(float(np.mean(null_rate)), 4),
        'mean_ret_pctile': round(float(np.mean(null_mean < obs_mean)), 4),
        'p': round(p, 5),
        'dilution_per_resample': round(diluted / b, 2),
        'empty_null_clusters_per_resample': round(empty_clusters / b, 3),
        'underpowered': n_cl < MIN_CLUSTERS,
    }


# ------------------------------------------------------------------- driver

def scan_ticker(ticker: str, splits: dict[str, list[tuple[str, float]]],
                ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None,
                           list[dict[str, Any]]]:
    path = archive_path(ticker)
    if path is None:
        return None, None, []
    d = aggregate_daily(path)
    if not len(d['dates']):
        return None, None, []
    adj = split_adjust(d, splits.get(ticker, []))
    clip = TICKER_START_CLIPS.get(ticker)
    if clip is not None:
        keep = adj['dates'] >= clip
        adj = {k: v[keep] for k, v in adj.items()}
        if not len(adj['dates']):
            return None, None, []
    dropped = TICKER_DROP_WINDOWS.get(ticker, [])
    for w0, w1 in dropped:
        keep = (adj['dates'] < w0) | (adj['dates'] > w1)
        adj = {k: v[keep] for k, v in adj.items()}
    flags = cliff_flags(adj['close'], adj['dates'], splits.get(ticker, []))
    # §2: excluded until resolved by hand — an owner-signed RESOLVED_CLIFFS
    # entry clears its date; only UNRESOLVED flags exclude
    unresolved = [f for f in flags if (ticker, f) not in RESOLVED_CLIFFS]
    cov = coverage_diagnostic(ticker, adj, unresolved)
    cov['resolved_cliffs'] = [f for f in flags if (ticker, f) in RESOLVED_CLIFFS]
    cov['start_clip'] = clip
    cov['drop_windows'] = dropped
    detections = ([] if unresolved
                  else detect_cup_handle(adj['close'], adj['volume']))
    return adj, cov, detections


def quadratic_roundness(c: np.ndarray, left: int, r: int) -> dict[str, Any]:
    """§3 rule 3's labeled variant (reported, never gating): a quadratic fit
    to the cup's normalized closes; R² = the fit's explained-variance share;
    vertex position as a fraction of the cup span."""
    y = c[left:r + 1]
    y = (y - y.min()) / (y.max() - y.min() if y.max() > y.min() else 1.0)
    x = np.linspace(0.0, 1.0, len(y))
    coef = np.polyfit(x, y, 2)
    fit = np.polyval(coef, x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    vertex = float(-coef[1] / (2 * coef[0])) if coef[0] != 0 else float('nan')
    return {'r2': round(r2, 4),
            'vertex': round(vertex, 4),
            'passes': bool(r2 >= 0.70 and 1 / 3 <= vertex <= 2 / 3)}


def _era_of(date: str) -> int:
    for j, (lo, hi) in enumerate(ERAS):
        if lo <= date <= hi:
            return j
    return -1


def run_scan(tickers: Sequence[str], evaluate: bool = False) -> dict[str, Any]:
    splits = load_splits()
    failed = failed_tickers()
    data: dict[str, dict[str, np.ndarray]] = {}
    coverage: list[dict[str, Any]] = []
    detections: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    excluded: dict[str, str] = {}
    skipped_in_flight: list[str] = []
    for t in tickers:
        if t in failed:
            excluded[t] = 'failed_fetch'
            continue
        try:
            adj, cov, dets = scan_ticker(t, splits)
        except (EOFError, FileNotFoundError, OSError):
            # the archive moved under us (the batch's rolling gzip):
            # skip this pass, the next increment picks it up complete
            skipped_in_flight.append(t)
            continue
        if adj is None:
            missing.append(t)
            continue
        if cov['cliff_flags']:
            excluded[t] = f"cliff:{cov['cliff_flags'][:3]}"
        data[t] = adj
        coverage.append(cov)
        detections[t] = dets
    # the union trading calendar (all scanned tickers) backs the §2
    # sessions-vs-calendar comparison
    calendar: set[str] = set()
    for v in data.values():
        calendar.update(str(d) for d in v['dates'])
    for cov in coverage:
        t = cov['ticker']
        first, last = cov['first'], cov['last']
        expected = sum(1 for day in calendar if first <= day <= last)
        cov['expected_sessions'] = expected
        cov['missing_sessions'] = expected - cov['sessions']
    # the frozen §4 statistic: total detections / sum of per-ticker span in
    # decades (calendar span, not session count)
    def _span_decades(cov: dict[str, Any]) -> float:
        y0, y1 = int(cov['first'][:4]), int(cov['last'][:4])
        m0, m1 = int(cov['first'][5:7]), int(cov['last'][5:7])
        return ((y1 - y0) * 12 + (m1 - m0)) / 120.0
    decades = sum(_span_decades(c) for c in coverage
                  if c['ticker'] not in excluded)
    total = sum(len(v) for t, v in detections.items() if t not in excluded)
    out: dict[str, Any] = {
        'tickers_scanned': len(data), 'missing': missing,
        'skipped_in_flight': skipped_in_flight,
        'excluded': excluded,
        'coverage': coverage,
        'total_detections': total,
        'rate_per_ticker_decade': round(total / decades, 3) if decades else None,
        'detections': {t: [d['t'] for d in v] for t, v in detections.items()},
        'quadratic_variant': {
            t: [quadratic_roundness(data[t]['close'], d['l'], d['r'])['passes']
                for d in v]
            for t, v in detections.items() if v},
    }
    if evaluate:
        pooled_data = {t: v for t, v in data.items() if t not in excluded}
        out['evaluation'] = {}
        for hz in HORIZONS:
            trades = {t: build_trades([d['t'] for d in detections[t]], hz,
                                      len(pooled_data[t]['close']))
                      for t in pooled_data}
            clusters = build_clusters(trades, pooled_data, hz)
            e = cluster_null_p(clusters, pooled_data, hz)
            # §5 reported diagnostics: pooled per-trade win rate, member
            # counts, per-ticker and per-era splits, survivorship bracket
            rets = []
            for t, entries in trades.items():
                cc = pooled_data[t]['close']
                rets += [float(cc[i + hz] / cc[i] - 1.0) for i in entries]
            e['n_trades'] = len(rets)
            e['per_trade_win_rate'] = (round(float(np.mean(
                [r > 0 for r in rets])), 4) if rets else None)
            e['members_per_cluster'] = (round(float(np.mean(
                [len(cl['members']) for cl in clusters])), 2)
                if clusters else None)
            e['per_era_clusters'] = {
                j: sum(1 for cl in clusters if _era_of(cl['date']) == j)
                for j in range(len(ERAS))}
            e['per_ticker_trades'] = {t: len(v) for t, v in trades.items() if v}
            # the survivorship bracket: unconditional P(close[i+H] > close[i])
            wins = tot = 0
            for t, v in pooled_data.items():
                cc = v['close']
                seg = cc[DETECT_FLOOR:len(cc) - hz]
                if len(seg):
                    fwd = cc[DETECT_FLOOR + hz:]
                    wins += int(np.sum(fwd > seg))
                    tot += len(seg)
            e['base_rate_bracket'] = round(wins / tot, 4) if tot else None
            out['evaluation'][f'H{hz}'] = e
        surv = all(
            not out['evaluation'][f'H{hz}'].get('underpowered', True)
            and out['evaluation'][f'H{hz}']['p'] <= SURVIVAL_P
            for hz in HEADLINE_HORIZONS)
        out['survives'] = bool(surv)
        # the §5 ablation: the volume trigger removed, its own null streams
        abl_det = {t: [d['t'] for d in detect_cup_handle(
            pooled_data[t]['close'], pooled_data[t]['volume'],
            use_volume_trigger=False)] for t in pooled_data}
        out['ablation_no_volume'] = {}
        for hz in HEADLINE_HORIZONS:
            trades = {t: build_trades(abl_det[t], hz,
                                      len(pooled_data[t]['close']))
                      for t in pooled_data}
            clusters = build_clusters(trades, pooled_data, hz)
            out['ablation_no_volume'][f'H{hz}'] = cluster_null_p(
                clusters, pooled_data, hz, variant='no_volume')
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default=None,
                    help='comma-separated subset (default: the committed universe)')
    ap.add_argument('--evaluate', action='store_true',
                    help='the §10 step-4 switch: compute returns and verdicts')
    ap.add_argument('--crosscheck', action='store_true',
                    help='reference cross-check battery (network, reporting '
                         'only — run_scan itself never touches the network)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    tks = a.tickers.split(',') if a.tickers else universe()
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
    res = run_scan(tks, evaluate=a.evaluate)
    if a.json:
        print(json.dumps(res, default=str))
    else:
        print(f"scanned {res['tickers_scanned']} tickers "
              f"({len(res['missing'])} missing archives)")
        print(f"detections: {res['total_detections']} "
              f"(rate {res['rate_per_ticker_decade']}/ticker-decade)")
        for c in res['coverage']:
            if c['late_start'] or c['cliff_flags']:
                print(f"  FLAG {c['ticker']}: first={c['first']} "
                      f"cliffs={c['cliff_flags'][:3]}")
        if 'evaluation' in res:
            for hz, e in res['evaluation'].items():
                print(hz, e)
            print('survives:', res['survives'])
