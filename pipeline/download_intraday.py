"""1-minute intraday OHLCV fetcher — Alpha Vantage TIME_SERIES_INTRADAY.

Month-by-month and resumable, matching the nine existing personal
archives' conventions exactly: AS-TRADED prices (``adjusted=false`` —
the QQQ archive shows 192 pre-split in Jan 2000), extended hours
included, schema ``timestamp,open,high,low,close,volume`` ascending.

Resume state: a sidecar ``<out>.months`` file records every completed
``YYYY-MM`` — including legitimately EMPTY months (pre-listing), so a
resume never refetches them. A ``<out>.months.done`` marker appears when
the whole span is complete (the batch driver gzips on that signal).

Symbol validity probe: the END month is fetched first; a symbol whose
latest month is empty is almost certainly mis-formatted (dot-class
tickers normalize ``BRK.B`` -> ``BRK-B``) or delisted — the fetcher
exits with a loud NO_DATA message instead of burning 300+ requests on
empty months.

These archives are personal cold-storage backups (gitignored, backed up
to OneDrive) — NOT analysis inputs; promotion to an analysis surface is
a separate, human-gated decision, like the full option chains.

Usage:
    ALPHAVANTAGE_API_KEY=... python -m pipeline.download_intraday \
        --symbol AAPL --start 2000-01 --end 2026-07 --out-dir data/sp500_intraday_1min
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import time
import urllib.request

BASE = 'https://www.alphavantage.co/query'
# Per-request pause. 0.85s is the single-worker premium-tier cadence every
# house fetcher uses; concurrent workers (scripts/fetch_sp500_intraday.sh
# with WORKER_COUNT > 1) raise it via AV_FETCH_SLEEP so the AGGREGATE rate
# stays near the 75-requests/minute cap — the throttle-note backoff in
# fetch_month self-regulates any brief overshoot.
SLEEP_SECONDS = float(os.environ.get('AV_FETCH_SLEEP', '0.85'))


def month_range(start: str, end: str):
    y, m = (int(x) for x in start.split('-'))
    y2, m2 = (int(x) for x in end.split('-'))
    while (y, m) <= (y2, m2):
        yield f'{y:04d}-{m:02d}'
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_month(symbol: str, month: str, key: str, retries: int = 6) -> list[list[str]]:
    """One month of 1-min bars, ascending. [] = legitimately empty month.
    Raises after exhausting retries on transport/throttle trouble."""
    url = (f'{BASE}?function=TIME_SERIES_INTRADAY&symbol={symbol}'
           f'&interval=1min&month={month}&outputsize=full&adjusted=false'
           f'&extended_hours=true&datatype=csv&apikey={key}')
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                text = r.read().decode()
        except Exception:               # noqa: BLE001 — socket hiccup: back off, retry
            time.sleep(5 * attempt)
            continue
        stripped = text.lstrip()
        if stripped.startswith('{'):
            low = stripped.lower()
            if 'error message' in low or 'invalid api call' in low:
                return []               # pre-listing / no data for this month
            # throttle note or transient information payload
            time.sleep(15 * attempt)
            continue
        rows = list(csv.reader(io.StringIO(text)))
        if not rows or rows[0][:1] != ['timestamp']:
            time.sleep(5 * attempt)
            continue
        return sorted(rows[1:])
    raise RuntimeError(f'{symbol} {month}: retries exhausted')


def run(symbol: str, start: str, end: str, out_dir: str) -> int:
    key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not key:
        print('ALPHAVANTAGE_API_KEY not set', file=sys.stderr)
        return 2
    api_symbol = symbol.replace('.', '-').upper()
    stem = api_symbol.lower()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{stem}_intraday_1min.csv')
    state_path = out + '.months'
    done_path = state_path + '.done'
    if os.path.exists(done_path):
        print(f'{symbol}: already complete')
        return 0

    done_months: set[str] = set()
    if os.path.exists(state_path):
        with open(state_path) as f:
            done_months = {ln.strip() for ln in f if ln.strip()}

    # validity probe on the END month (fresh symbols always have it)
    if end not in done_months:
        probe = fetch_month(api_symbol, end, key)
        time.sleep(SLEEP_SECONDS)
        if not probe:
            print(f'NO_DATA {symbol}: end month {end} empty — symbol '
                  f'mis-formatted or delisted; skipping', file=sys.stderr)
            return 3

    new_file = not os.path.exists(out)
    months = list(month_range(start, end))
    todo = [m for m in months if m not in done_months]
    with open(out, 'a', newline='') as f_out, open(state_path, 'a') as f_state:
        w = csv.writer(f_out)
        if new_file:
            w.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        for i, month in enumerate(todo):
            rows = fetch_month(api_symbol, month, key)
            w.writerows(rows)
            f_out.flush()
            f_state.write(month + '\n')
            f_state.flush()
            if i % 24 == 0:
                print(f'{symbol}: {month} ({len(rows)} bars) '
                      f'[{len(done_months) + i + 1}/{len(months)}]', flush=True)
            time.sleep(SLEEP_SECONDS)
    with open(done_path, 'w') as f:
        f.write('complete\n')
    print(f'{symbol}: complete -> {out}')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--start', default='2000-01')
    ap.add_argument('--end', required=True)
    ap.add_argument('--out-dir', default='data')
    a = ap.parse_args()
    sys.exit(run(a.symbol, a.start, a.end, a.out_dir))
