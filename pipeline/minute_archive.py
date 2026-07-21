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
    ('GL', '2024-04-11'):
        'real event: the Fuzzy Panda short-report crash, ~-53% close-to-'
        'close (~-63% intraday) — Globe Life, no split (owner-signed '
        '2026-07-21)',
    ('HIG', '2008-10-30'):
        'real event: -51.6% the day after a big Q3-2008 loss + '
        'capital-raise/downgrade fears — The Hartford in the crisis; '
        'triple-confirmed on price (owner-signed 2026-07-21)',
    ('HIG', '2008-12-05'):
        'real event: ~+102% (stock doubled) after a Dec-4 8-K raised the '
        'profit forecast + reaffirmed capital, on a broad market rally — '
        'The Hartford; triple-confirmed on price (owner-signed 2026-07-21)',
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
    # FOXA is the class-A twin of FOX — same splice; Nasdaq's own notice
    # (#dtn2019-05) calls it a "symbol reuse" and rules pre-2019 history
    # belongs to the old 21st Century Fox / News Corp lineage; Fox
    # Corporation class A assumed FOXA on 2019-03-19 (owner-signed
    # 2026-07-21)
    'FOXA': '2019-03-19',
    # the GEN tape splices THREE unrelated issuers: Reliant/GenOn (a
    # power utility, 2001-2012), Genesis HealthCare (nursing homes,
    # 2015-2021), then the real Gen Digital (Norton/Symantec, ~$21)
    # from 2021-03-26 — the 49x overnight jump is the switch. The
    # snapshot's three 2:1 GEN splits are Symantec's, mis-applied to the
    # utility tape (they fabricated the pre-2005 split-day cliffs); the
    # clip discards the era they touch, so they are left in place, inert
    # (owner-signed 2026-07-21)
    'GEN': '2021-03-26',
    # pre-2020 rows are old Ingersoll-Rand plc (~$145, now Trane
    # Technologies / ticker TT); the ticker passed to the new Ingersoll
    # Rand Inc (the Gardner Denver lineage, ~$36) at the 2020-02-29
    # Reverse Morris Trust merger. 2020-02-24 (the distribution record
    # date) is where the tape's new-company era begins and matches both
    # references to the cent (owner-signed 2026-07-21)
    'IR': '2020-02-24',
    # the JCI tape holds OLD Johnson Controls Inc's prices while the
    # snapshot holds TYCO's splits — applying one company's actions to
    # the other's prices is what fabricated the 2007/2012 cliffs. Tyco
    # was the LEGAL acquirer (renamed Johnson Controls International plc,
    # merger effective 2016-09-02, first trading day under JCI
    # 2016-09-06), and the issuer's own FY2016 10-K states its
    # pre-merger share prices are Tyco's — the convention both
    # references follow. Old JCI is NOT the continuous per-share series:
    # each share became 0.8357 new shares PLUS $5.73 cash, which a split
    # table cannot express (owner-signed 2026-07-21)
    'JCI': '2016-09-06',
    # 2018-07-09 is a DIFFERENT INSTRUMENT, not merely an awkward bar:
    # Dr Pepper Snapple's $103.75/share special cash dividend was large
    # enough that NYSE ran DUE-BILL trading, so a share bought that day
    # carried the stock PLUS a detachable right to the cash. Holders lost
    # nothing ($123.68 cum = $19.93 stock + $103.75 cash; it traded to
    # $22.19), but the -82% print would be a guaranteed false crash for
    # any detector — and would poison trailing vol/drawdown features for
    # a whole lookback window. The archive holds exactly ONE pre-event
    # session, so the clip costs a single bar (owner-signed 2026-07-21)
    'KDP': '2018-07-10',
    # pre-2013-09-30 rows are old SAIC, Inc. (NYSE: SAI), which on
    # 2013-09-27 spun off ~30% of its value as a NEW SAIC (1 share per 7
    # held) and then did a 1-for-4 reverse split before the 2013-09-30
    # open, renaming itself Leidos. 2013-09-16..09-27 are LDOS
    # EX-DISTRIBUTION when-issued prints (1.5k-36k shares/day against
    # 2-4.6M regular-way) running alongside regular-way SAI — the clip
    # subsumes them, so no drop-window is needed.
    #
    # The pre-separation era is NOT rescuable by repairing the snapshot's
    # 0.405 factor. That factor is a vendor composite (a plain 1-for-4 is
    # 0.25) and it is WRONG: the combined multiplier read off the 10
    # sessions where both lines traded is 2.871, not the 2.469 that 0.405
    # implies, which is what inflates the join to +20% (it would be
    # +3.1% correctly). The right value ~0.348 appears in NO committed
    # reference, so deriving it would be inventing a cleaning rule that
    # pins numbers; and even a perfect factor yields a SYNTHETIC series
    # — 4 x (P_SAI - P_newSAIC/7) never traded, and the spin fraction is
    # observable only for those 10 days, so carrying it back to 2007
    # manufactures exactly the smooth-but-fake era the ELV ruling exists
    # to catch. This is the JCI case verbatim: each share became 0.25
    # LDOS shares PLUS 1/7 of another company, which a split table
    # cannot express. The 0.405 row is left in place, inert, as GEN's
    # mis-applied splits were (owner-signed 2026-07-21)
    'LDOS': '2013-09-30',
    # a THREE-part splice. 2007-05-07..2014-12-19 is LIN TV Corp / LIN
    # Media (a local-TV broadcaster, itself back-stitched: it traded as
    # TVL until 2013-07-30, then as LIN), acquired by Media General
    # 2014-12-19 at $25.97 cash — the tape's last bar is $25.45, trading
    # just under the election. Its equity curve settles the identity
    # beyond doubt: $16.01 (2007) -> $1.11 (2009-04-01, the GFC ad
    # recession) -> $28.30 (2014); nothing in the Praxair/Linde lineage
    # goes to a dollar in 2009. The clip also removes the 2007-06-01
    # vendor bad print ($50.05 on 2.82M shares against ~$19.50 on ~300K).
    #
    # From 2014-12-22 the vendor back-stitches PRAXAIR (as traded under
    # PX) into the vacated symbol, and Linde plc took LIN on 2018-10-31.
    # The VACANCY date is the right boundary here — unlike JCI and IR —
    # because Praxair -> Linde plc is a CLEAN 1:1 continuation: one Linde
    # plc share per Praxair share, no cash and no election, so the seam
    # needs no factor (+0.24% across it). The issuer agrees: Praxair was
    # the ASC 805 ACCOUNTING ACQUIRER, so Linde plc's own 10-K reports
    # pre-combination periods as Praxair's. Contrast JCI (0.8952 shares
    # PLUS cash) and IR (a different business arriving under the symbol),
    # which is why those clip at the current-issuer date instead.
    #
    # Caveats: 2014-12-22 is a SPLICE SEAM, not a corporate event; and
    # the 2014-12-22..2018-10-30 span is Praxair under PX, so work
    # needing literal-ticker fidelity (e.g. joining option chains keyed
    # on LIN) should use 2018-10-31, Linde plc's first real session
    # (owner-signed 2026-07-21)
    'LIN': '2014-12-22',
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
    # the same disease on the same date as FAST: our minute close is
    # $20.44 while BOTH independent daily references say $37.94, with
    # neighbours at $39.62 and $37.19. Found by the one-bar-V sweep
    # (a day far below both neighbours that fully recovers — real
    # crashes do not round-trip) (owner-signed 2026-07-21)
    'EXPD': [('2000-04-07', '2000-04-07')],
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
    ap.add_argument('--crosscheck', action='store_true',
                    help='reference cross-check battery (network; reporting '
                         'only — load_clean_daily never touches the network)')
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
