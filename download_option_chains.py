"""Download real historical option-chain data for the covered-call backtest.

For each monthly roll date, fetch the full EOD option chain from Alpha Vantage's
HISTORICAL_OPTIONS endpoint (which returns greeks and implied volatility
directly) and pick the call closest to the engine's target — ~21 DTE, ~0.25
delta. The result is the honest replacement for cc_backtest.py's estimate_iv
proxy: actual traded premiums instead of Black-Scholes priced off a
realized-vol-times-regime-multiplier guess.

Usage:
    export ALPHAVANTAGE_API_KEY=your_free_key   # alphavantage.co/support/#api-key
    python download_option_chains.py --ticker QQQ --start 2024-01-01 --end 2024-12-31

Output: a CSV with one row per roll date — the selected call's strike,
expiration, DTE, delta, IV, and bid/ask/mark — written incrementally so the run
is resumable. Alpha Vantage's free tier allows only ~25 requests/day; re-run the
same command on later days and it skips roll dates already in the output file.

This fetcher only produces the dataset. Wiring it into the engine (replacing
estimate_iv with a lookup of the real premium per roll) is the next step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

AV_URL = 'https://www.alphavantage.co/query'
OUT_FIELDS = [
    'roll_date', 'data_date', 'expiration', 'dte', 'strike', 'delta',
    'implied_volatility', 'mark', 'bid', 'ask', 'last', 'volume',
    'open_interest', 'contractID',
]


def fetch_chain(ticker: str, day: str, api_key: str) -> list[dict]:
    """Fetch the EOD option chain for `ticker` on `day` (YYYY-MM-DD).

    Returns the raw list of contract dicts. Raises RuntimeError on an API-level
    message (rate limit, invalid key, premium-gated date) — Alpha Vantage
    signals those via an 'Information'/'Note'/'Error Message' field instead of
    'data', so a missing 'data' key means "stop and tell the user why".
    """
    params = urllib.parse.urlencode({
        'function': 'HISTORICAL_OPTIONS',
        'symbol': ticker,
        'date': day,
        'apikey': api_key,
    })
    with urllib.request.urlopen(f'{AV_URL}?{params}', timeout=30) as resp:
        payload = json.load(resp)
    if 'data' not in payload:
        msg = (
            payload.get('Information')
            or payload.get('Note')
            or payload.get('Error Message')
            or str(payload)
        )
        raise RuntimeError(msg)
    return payload['data']


def select_target_call(
    data: list[dict], asof: str, target_dte: int, target_delta: float
) -> dict | None:
    """Pick the call closest to (target_dte, target_delta) from a raw chain.

    Mirrors the engine's selection: among expirations, take the one whose
    days-to-expiry is nearest target_dte (skipping expired / same-day rows);
    within that expiration, take the call whose delta is nearest target_delta.
    Returns an enriched copy (with an integer 'dte') or None if the chain has no
    usable calls. Pure function — unit-testable against a captured chain.
    """
    asof_d = datetime.strptime(asof, '%Y-%m-%d').date()
    calls: list[tuple[int, float, dict]] = []
    for row in data:
        if row.get('type') != 'call':
            continue
        try:
            exp = datetime.strptime(row['expiration'], '%Y-%m-%d').date()
            delta = float(row['delta'])
        except (KeyError, ValueError):
            continue
        dte = (exp - asof_d).days
        if dte < 1 or not 0.0 < delta < 1.0:
            continue
        calls.append((dte, delta, row))
    if not calls:
        return None
    # Nearest expiration to the target DTE, then nearest delta within it.
    best_dte = min({d for d, _, _ in calls}, key=lambda d: abs(d - target_dte))
    cohort = [c for c in calls if c[0] == best_dte]
    dte, _delta, row = min(cohort, key=lambda c: abs(c[1] - target_delta))
    return {**row, 'dte': dte}


def _snap_to_weekday(d: datetime) -> datetime:
    """Move a weekend date back to the prior Friday (options don't trade weekends)."""
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d - timedelta(days=2)
    return d


def roll_dates(start: str, end: str, step_days: int) -> list[str]:
    """Stepped roll dates in [start, end], each snapped off weekends, deduped."""
    cur = datetime.strptime(start, '%Y-%m-%d')
    last = datetime.strptime(end, '%Y-%m-%d')
    seen: set[str] = set()
    out: list[str] = []
    while cur <= last:
        iso = _snap_to_weekday(cur).date().isoformat()
        if iso not in seen:
            seen.add(iso)
            out.append(iso)
        cur += timedelta(days=step_days)
    return out


def load_done(out_path: str) -> set[str]:
    """Roll dates already present in the output CSV (for resuming)."""
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline='') as f:
        return {row['roll_date'] for row in csv.DictReader(f) if row.get('roll_date')}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--ticker', default='QQQ')
    p.add_argument('--start', required=True, help='First roll date, YYYY-MM-DD')
    p.add_argument('--end', required=True, help='Last roll date, YYYY-MM-DD')
    p.add_argument('--target-delta', type=float, default=0.25)
    p.add_argument('--dte', type=int, default=21, help='Target days-to-expiry (default 21)')
    p.add_argument('--roll-days', type=int, default=21, help='Calendar days between rolls')
    p.add_argument('--max-lookback', type=int, default=3,
                   help='Walk back up to this many days to find a trading day with data')
    p.add_argument('--sleep', type=float, default=0.0,
                   help='Seconds to wait between requests (for premium per-minute limits)')
    p.add_argument('--out', default=None, help='Output CSV (default {ticker}_option_rolls.csv)')
    args = p.parse_args()
    if args.out is None:
        args.out = f'{args.ticker.lower()}_option_rolls.csv'
    return args


def main() -> None:
    args = parse_args()
    api_key = os.environ.get('ALPHAVANTAGE_API_KEY')
    if not api_key:
        sys.exit('Set ALPHAVANTAGE_API_KEY first '
                 '(free key: https://www.alphavantage.co/support/#api-key)')

    done = load_done(args.out)
    pending = [d for d in roll_dates(args.start, args.end, args.roll_days) if d not in done]
    if not pending:
        print(f'Nothing to do — all roll dates already in {args.out}.')
        return
    print(f'{len(pending)} roll date(s) to fetch ({len(done)} already done) -> {args.out}')

    new_file = not os.path.exists(args.out)
    requests = 0
    with open(args.out, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction='ignore')
        if new_file:
            writer.writeheader()

        for roll in pending:
            chosen: dict | None = None
            data_date: str | None = None
            base = datetime.strptime(roll, '%Y-%m-%d').date()
            for back in range(args.max_lookback + 1):
                day = (base - timedelta(days=back)).isoformat()
                try:
                    data = fetch_chain(args.ticker, day, api_key)
                except RuntimeError as exc:
                    print(f'\nAlpha Vantage stopped us: {exc}')
                    print(f'Made {requests} request(s) this run; progress saved to '
                          f'{args.out}. Re-run the same command to resume.')
                    return
                requests += 1
                if data:
                    chosen = select_target_call(data, day, args.dte, args.target_delta)
                    if chosen:
                        data_date = day
                        break
                if args.sleep:
                    time.sleep(args.sleep)

            if not chosen or data_date is None:
                print(f'  {roll}: no usable chain within {args.max_lookback}d — skipped')
                continue

            writer.writerow({
                'roll_date': roll,
                'data_date': data_date,
                'expiration': chosen['expiration'],
                'dte': chosen['dte'],
                'strike': chosen['strike'],
                'delta': chosen['delta'],
                'implied_volatility': chosen['implied_volatility'],
                'mark': chosen.get('mark'),
                'bid': chosen.get('bid'),
                'ask': chosen.get('ask'),
                'last': chosen.get('last'),
                'volume': chosen.get('volume'),
                'open_interest': chosen.get('open_interest'),
                'contractID': chosen.get('contractID'),
            })
            f.flush()
            print(f"  {roll} -> {chosen.get('contractID')}  strike {chosen['strike']}  "
                  f"delta {float(chosen['delta']):.3f}  dte {chosen['dte']}  "
                  f"mark {chosen.get('mark')}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f'Done — {requests} request(s) this run; rolls written to {args.out}.')


if __name__ == '__main__':
    main()
