"""Download daily option-chain slices for real-premium engine wiring.

Where download_option_chains.py grabs one target call per monthly roll (entry
premiums), this fetches a filtered slice of the chain for EVERY trading day:
all calls with DTE 1..--max-dte and strike within ±--strike-band of spot.
That's what the engine needs to (a) select a call at any roll date the
strategy lands on, and (b) mark the held contract to market every day until
it's closed, expired, or assigned.

Spot is inferred from the chain itself — the call strike whose delta is
closest to 0.50 at the nearest expiration — so strikes are filtered in
actual-price space without needing an (adjusted!) external price series.

Usage:
    export ALPHAVANTAGE_API_KEY=...   # premium (options endpoints are gated)
    python download_option_dailies.py --ticker QQQ --start 2016-06-06 --end 2026-06-05

Output: {ticker}_option_dailies.csv, one row per surviving contract per day,
written incrementally; resumable — re-running skips days already present.
Trading days are taken from --dates-from (default qqq_10yr_prices.csv; any
NYSE/Nasdaq-calendar price file works for any US-listed underlying).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from download_option_chains import fetch_chain

OUT_FIELDS = [
    'date', 'expiration', 'dte', 'strike', 'bid', 'ask', 'mark', 'last',
    'volume', 'open_interest', 'implied_volatility', 'delta', 'contractID',
]


def trading_days(price_csv: str, start: str, end: str) -> list[str]:
    """Trading-day list from a yfinance price CSV, clipped to [start, end]."""
    days: list[str] = []
    with open(price_csv) as f:
        for row in csv.reader(f):
            if not row or not row[0][:4].isdigit():
                continue
            if start <= row[0] <= end:
                days.append(row[0])
    return days


def infer_spot(calls: list[dict], asof: str) -> float | None:
    """Spot ~= the strike whose call delta is nearest 0.50 at the nearest expiry."""
    best: tuple[float, float] | None = None  # (|delta-0.5|, strike)
    nearest_exp: str | None = None
    for row in calls:
        exp = row.get('expiration', '')
        if exp <= asof:
            continue
        if nearest_exp is None or exp < nearest_exp:
            nearest_exp = exp
    if nearest_exp is None:
        return None
    for row in calls:
        if row.get('expiration') != nearest_exp:
            continue
        try:
            delta = float(row['delta'])
            strike = float(row['strike'])
        except (KeyError, ValueError):
            continue
        score = abs(delta - 0.5)
        if best is None or score < best[0]:
            best = (score, strike)
    return best[1] if best else None


def filter_calls(
    data: list[dict], asof: str, max_dte: int, strike_band: float
) -> list[dict]:
    """Calls with 1 <= DTE <= max_dte and strike within ±strike_band of spot."""
    asof_d = datetime.strptime(asof, '%Y-%m-%d').date()
    calls = [r for r in data if r.get('type') == 'call']
    spot = infer_spot(calls, asof)
    if spot is None:
        return []
    lo, hi = spot * (1 - strike_band), spot * (1 + strike_band)
    out: list[dict] = []
    for r in calls:
        try:
            exp = datetime.strptime(r['expiration'], '%Y-%m-%d').date()
            strike = float(r['strike'])
        except (KeyError, ValueError):
            continue
        dte = (exp - asof_d).days
        if 1 <= dte <= max_dte and lo <= strike <= hi:
            out.append({**r, 'dte': dte})
    return out


def load_done(out_path: str) -> set[str]:
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline='') as f:
        return {row['date'] for row in csv.DictReader(f) if row.get('date')}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--ticker', required=True)
    p.add_argument('--start', required=True, help='First trading day, YYYY-MM-DD')
    p.add_argument('--end', required=True, help='Last trading day, YYYY-MM-DD')
    p.add_argument('--dates-from', default='qqq_10yr_prices.csv',
                   help='Price CSV supplying the trading-day calendar')
    p.add_argument('--max-dte', type=int, default=60)
    p.add_argument('--strike-band', type=float, default=0.35,
                   help='Keep strikes within ±this fraction of inferred spot')
    p.add_argument('--sleep', type=float, default=0.85,
                   help='Seconds between requests (premium: 75/min)')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or f'{args.ticker.lower()}_option_dailies.csv'

    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        sys.exit('Set ALPHAVANTAGE_API_KEY first')

    days = trading_days(args.dates_from, args.start, args.end)
    done = load_done(out)
    todo = [d for d in days if d not in done]
    print(f'{args.ticker}: {len(todo)} day(s) to fetch ({len(done)} already done) -> {out}',
          flush=True)
    if not todo:
        return

    new_file = not os.path.exists(out)
    n_rows = 0
    with open(out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction='ignore')
        if new_file:
            writer.writeheader()
        for i, day in enumerate(todo):
            try:
                data = fetch_chain(args.ticker, day, api_key)
            except RuntimeError as exc:
                print(f'\nAlpha Vantage stopped us at {day}: {exc}', flush=True)
                print(f'{n_rows} row(s) written this run; re-run to resume.', flush=True)
                return
            kept = filter_calls(data, day, args.max_dte, args.strike_band)
            for r in kept:
                writer.writerow({
                    'date': day, 'expiration': r['expiration'], 'dte': r['dte'],
                    'strike': r['strike'], 'bid': r.get('bid'), 'ask': r.get('ask'),
                    'mark': r.get('mark'), 'last': r.get('last'),
                    'volume': r.get('volume'), 'open_interest': r.get('open_interest'),
                    'implied_volatility': r.get('implied_volatility'),
                    'delta': r.get('delta'), 'contractID': r.get('contractID'),
                })
            f.flush()
            n_rows += len(kept)
            if i % 25 == 0 or i == len(todo) - 1:
                print(f'  [{i + 1}/{len(todo)}] {day}: chain {len(data)} -> kept {len(kept)} '
                      f'(total rows {n_rows})', flush=True)
            if args.sleep:
                time.sleep(args.sleep)
    print(f'Done — {args.ticker}: {n_rows} row(s) appended to {out}.', flush=True)


if __name__ == '__main__':
    main()
