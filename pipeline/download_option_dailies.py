"""Download daily option-chain slices for real-premium engine wiring.

Where download_option_chains.py grabs one target call per monthly roll (entry
premiums), this fetches a filtered slice of the chain for EVERY trading day:
calls AND puts (or one wing via --keep) with DTE 1..--max-dte and strike within
±--strike-band of spot. That's what the engine needs to (a) select a call or put
at any roll date the strategy lands on, and (b) mark the held contract to market
every day until it's closed, expired, or assigned.

Spot is inferred from the chain's CALLS — the call strike whose delta is closest
to 0.50 at the nearest expiration (always present in a full chain) — so strikes
are filtered in actual-price space without an (adjusted!) external price series.
Puts in that same strike band are kept too; their NEGATIVE vendor delta is what
the short-vol engine reads to tell the wings apart (no extra column needed).

Usage:
    export ALPHAVANTAGE_API_KEY=...   # premium (options endpoints are gated)
    # fresh ticker, both wings (default):
    python download_option_dailies.py --ticker IWM --start 2010-12-01 \
        --end 2026-06-05 --dates-from iwm_20yr_prices.csv
    # add ONLY the put wing to a ticker whose calls file already exists:
    python download_option_dailies.py --ticker SPY --keep put \
        --out spy_option_dailies_puts.csv --dates-from spy_20yr_prices.csv \
        --start 2010-12-01 --end 2026-06-05

Output: {ticker}_option_dailies.csv (or --out), one row per surviving contract
per day, written incrementally; resumable — re-running skips days already present.
--keep {both|call|put} chooses the wing(s) (default both). Trading days come from
--dates-from (any NYSE/Nasdaq-calendar price file for any US-listed underlying).
"""

from __future__ import annotations
from common.paths import data_path

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from pipeline.download_option_chains import fetch_chain

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


def filter_chain(
    data: list[dict], asof: str, max_dte: int, strike_band: float, keep: str = 'both'
) -> list[dict]:
    """Calls and/or puts with 1 <= DTE <= max_dte and strike within ±strike_band
    of spot. `keep` is 'both' (default), 'call', or 'put'. Spot is always inferred
    from the CALLS (delta nearest 0.50 — present in any full chain); the strike
    band then applies to whichever wing(s) `keep` selects, so puts (negative delta)
    are kept by the same band when requested."""
    asof_d = datetime.strptime(asof, '%Y-%m-%d').date()
    calls = [r for r in data if r.get('type') == 'call']
    spot = infer_spot(calls, asof)
    if spot is None:
        return []
    lo, hi = spot * (1 - strike_band), spot * (1 + strike_band)
    rows = data if keep == 'both' else [r for r in data if r.get('type') == keep]
    out: list[dict] = []
    for r in rows:
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
    p.add_argument('--dates-from', default=data_path('qqq_10yr_prices.csv'),
                   help='Price CSV supplying the trading-day calendar')
    p.add_argument('--max-dte', type=int, default=60)
    p.add_argument('--strike-band', type=float, default=0.35,
                   help='Keep strikes within ±this fraction of inferred spot')
    p.add_argument('--keep', choices=['both', 'call', 'put'], default='both',
                   help='Which option wing(s) to keep (default both)')
    p.add_argument('--sleep', type=float, default=0.85,
                   help='Seconds between requests (premium: 75/min)')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    out = args.out or data_path(f'{args.ticker.lower()}_option_dailies.csv')

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
            kept = filter_chain(data, day, args.max_dte, args.strike_band, args.keep)
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
