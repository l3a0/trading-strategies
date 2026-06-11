"""Walk-forward optimization on REAL option chains (or the proxy, aligned).

Mirrors cc_backtest.walk_forward_optimization exactly — same 3-year train /
6-month test / 6-month roll windows, same in-sample-Sharpe selection, same
per-window $100K restarts — but every backtest inside it is
real_cc_backtest.run_real_cc_overlay on actual market quotes instead of the
Black-Scholes + estimate_iv proxy.

`--prices proxy` swaps the engine back to run_cc_overlay while keeping
everything else identical — the same unadjusted close series clipped to the
chain span, the same windows, and the same CALENDAR-day grid (converted to
the proxy's trading-day clock at the engine boundary: 21/30/45 calendar ->
14/21/31 trading, and the dte=30 fixed-defaults comparator -> 21 trading
days, the published default). A real-vs-proxy gap in this report is
therefore attributable to the option-pricing source alone. The proxy has no
fill model (it carries its own 3% slippage + commission), so --fill is
rejected under --prices proxy.

Parameter-grid convention: `dte` values are CALENDAR days to expiration
(run_real_cc_overlay's convention) — 21/30/45, the same numerals as the
proxy grid. Note the proxy's 21/30/45 are TRADING days (~30/43/65
calendar), so this grid rolls on a faster cycle than the proxy ever did;
the fixed-defaults comparator keeps the published strategy's calendar twin
(dte=30 ~ 21 trading days). Deltas and profit-target levels carry over
unchanged.

Convention note (consistent, unlike the published 324/378/317 mix): every
number this script reports — walk-forward OOS, fixed-defaults OOS, and
buy-and-hold OOS — is chained from per-window restarts over the SAME test
windows, so the three are directly comparable.

Usage:
    python walk_forward_real.py MSFT [--prices real|proxy] [--fill bid_ask|mid]
                                     [--train-years N] [--min-trades N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from cc_backtest import _param_combinations, run_cc_overlay
from real_cc_backtest import (
    CHAIN_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
)

PARAM_GRID: dict[str, list[float]] = {
    'call_delta': [0.15, 0.20, 0.25],
    'dte': [21, 30, 45],  # CALENDAR days to expiration
    'close_at_pct': [0.50, 0.75, 1.00],
}
FIXED_PARAMS: dict[str, float] = {'capital': 100_000}
DEFAULTS: dict[str, float] = {'call_delta': 0.25, 'dte': 30, 'close_at_pct': 0.75}


def _sharpe(daily_eq: pd.DataFrame) -> float:
    """Annualized Sharpe of the equity curve's daily returns (proxy WF metric)."""
    returns = daily_eq['equity'].pct_change().dropna().tolist()
    if not returns:
        return -float('inf')
    avg = sum(returns) / len(returns)
    std = math.sqrt(sum((r - avg) ** 2 for r in returns) / max(1, len(returns) - 1))
    return (avg / std) * math.sqrt(252) if std > 0 else 0.0


def walk_forward_real(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    param_grid: dict[str, list[float]],
    fixed_params: dict[str, Any] | None = None,
    train_years: int = 3,
    test_months: int = 6,
    roll_months: int = 6,
    min_trades: int = 0,
    engine: str = 'real',
) -> list[dict[str, Any]]:
    """Walk-forward optimization driving run_real_cc_overlay (or the proxy).

    Window arithmetic, train/test boundary semantics (>= start, < end), the
    in-sample Sharpe selection rule, and the per-period record schema all
    mirror cc_backtest.walk_forward_optimization. Each record additionally
    carries the OOS window returns for the chosen params, the fixed
    defaults, and buy-and-hold, so callers can chain all three on one
    consistent per-window-restart convention — plus per-window grid trade
    stats for the Pardo sample-size check. `min_trades > 0` enforces that
    floor at selection time: in-sample fits with fewer trades are
    disqualified rather than trusted.

    `engine='proxy'` runs run_cc_overlay instead, on the same dates/prices
    and the same calendar-day grid: the dte of every combo (and of the
    fixed-defaults comparator) is converted to the proxy's trading-day
    clock at this boundary (round(cal * 252 / 365)), and the real-only
    'fill' param is dropped. `store` is unused in that mode.
    """
    if fixed_params is None:
        fixed_params = dict(FIXED_PARAMS)

    if engine == 'proxy':
        def run(d: list[str], p: list[float], prm: dict[str, Any]) -> tuple[Any, Any, Any]:
            prm = {k: v for k, v in prm.items() if k != 'fill'}
            prm['dte'] = max(1, round(prm['dte'] * 252 / 365))
            return run_cc_overlay(d, np.asarray(p, dtype=float), prm)
    elif engine == 'real':
        def run(d: list[str], p: list[float], prm: dict[str, Any]) -> tuple[Any, Any, Any]:
            return run_real_cc_overlay(d, p, store, prm)
    else:
        raise ValueError(f"unknown engine {engine!r} — use 'real' or 'proxy'")

    df = pd.DataFrame({'date': pd.to_datetime(dates), 'd': dates, 'price': prices})
    start_date = df['date'].min()
    end_date = df['date'].max()
    current_date = start_date + pd.DateOffset(years=train_years)

    records: list[dict[str, Any]] = []
    while current_date + pd.DateOffset(months=test_months) <= end_date:
        train_start = current_date - pd.DateOffset(years=train_years)
        train_df = df[(df['date'] >= train_start) & (df['date'] < current_date)]
        test_end = current_date + pd.DateOffset(months=test_months)
        test_df = df[(df['date'] >= current_date) & (df['date'] < test_end)]

        if len(train_df) < 30 or len(test_df) < 5:
            current_date += pd.DateOffset(months=roll_months)
            continue

        train_dates = list(train_df['d'])
        train_prices = [float(p) for p in train_df['price']]
        test_dates = list(test_df['d'])
        test_prices = [float(p) for p in test_df['price']]

        # === Step 1: OPTIMIZE on training data ===
        best_sharpe = -float('inf')
        best_params: dict[str, float] | None = None
        best_n_trades = 0
        grid_trades: list[int] = []  # IS trade count of every combo (Pardo floor stats)
        for combo in _param_combinations(param_grid):
            try:
                summary, _trades, daily_eq = run(
                    train_dates, train_prices, {**fixed_params, **combo})
            except Exception:
                continue
            n_trades = int(summary['num_calls_sold'])
            grid_trades.append(n_trades)
            if min_trades and n_trades < min_trades:
                continue  # thin fit: not enough trades to trust the Sharpe
            sharpe = _sharpe(daily_eq)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = combo
                best_n_trades = n_trades

        if best_params is None:
            current_date += pd.DateOffset(months=roll_months)
            continue

        # === Step 2: TEST out-of-sample with locked rules ===
        oos_sum, _, _ = run(test_dates, test_prices, {**fixed_params, **best_params})
        fix_sum, _, _ = run(test_dates, test_prices, {**fixed_params, **DEFAULTS})

        records.append({
            'train_start': train_start.date().isoformat(),
            'train_end': current_date.date().isoformat(),
            'test_start': current_date.date().isoformat(),
            'test_end': test_end.date().isoformat(),
            'best_params': best_params,
            'train_sharpe': round(best_sharpe, 3),
            'n_trades': best_n_trades,
            'min_grid_trades': min(grid_trades),
            'n_below_30': sum(1 for t in grid_trades if t < 30),
            'oos_return_pct': oos_sum['total_return_pct'],
            'oos_trades': oos_sum['num_calls_sold'],
            'fixed_return_pct': fix_sum['total_return_pct'],
            'bh_return_pct': oos_sum['buy_hold_return_pct'],
        })

        # === Step 3: ROLL FORWARD ===
        current_date += pd.DateOffset(months=roll_months)

    return records


def _chain(returns_pct: list[float]) -> float:
    """Cumulative return from chaining per-window restarts."""
    growth = 1.0
    for r in returns_pct:
        growth *= 1 + r / 100
    return (growth - 1) * 100


def main() -> None:
    ap = argparse.ArgumentParser(description='Walk-forward optimization on real option chains')
    ap.add_argument('ticker', nargs='?', default='MSFT')
    ap.add_argument('--prices', choices=('real', 'proxy'), default='real',
                    help='option-pricing source: real chains (default) or the '
                         'BS + estimate_iv proxy on the same series/windows/grid')
    ap.add_argument('--fill', choices=('bid_ask', 'mid'), default=None,
                    help='real-chain fill model (default bid_ask); rejected with --prices proxy')
    ap.add_argument('--train-years', type=int, default=3)
    ap.add_argument('--min-trades', type=int, default=0,
                    help='disqualify in-sample fits with fewer trades (Pardo floor)')
    ap.add_argument('--extra-dailies', nargs='*', default=[],
                    help='additional dailies CSVs merged into the chain store '
                         '(e.g. msft_option_dailies_2008_2016.csv, the backfill)')
    ap.add_argument('--stop-loss-mult', type=float, default=None,
                    help='real-chain stop-loss: buy the call back when its ask reaches '
                         'this multiple of the premium collected (e.g. 2.0); applied to '
                         'every train/test run including the fixed-defaults comparator')
    args = ap.parse_args()
    ticker = args.ticker.upper()
    if args.prices == 'proxy' and args.fill is not None:
        ap.error('--fill applies to real chains only (the proxy has its own slippage model)')
    if args.prices == 'proxy' and args.stop_loss_mult is not None:
        ap.error('--stop-loss-mult applies to real chains only (no proxy stop rule)')
    fill = args.fill or 'bid_ask'
    dailies_path = f'{ticker.lower()}_option_dailies.csv'
    for p in (dailies_path, *args.extra_dailies):
        if not (os.path.exists(p) or os.path.exists(p + '.gz')):
            sys.exit(f'{p}[.gz] not found — run fetch_option_data.sh first')

    start = CHAIN_CLEAN_START.get(ticker)
    print(f'Loading chain store ({dailies_path}'
          + (f' + {", ".join(args.extra_dailies)}' if args.extra_dailies else '')
          + (f', from {start}' if start else '') + ') ...',
          flush=True)
    store = load_chain_store(dailies_path, args.extra_dailies, start=start)
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    dates = [d for d, _ in pairs]
    prices = [p for _, p in pairs]
    print(f'{ticker}: {len(dates)} trading days {dates[0]} -> {dates[-1]}, '
          f'{len(days)} chain days', flush=True)

    if args.prices == 'proxy':
        fixed_params: dict[str, Any] = {**FIXED_PARAMS, 'risk_free_rate': 0.045}
        engine_tag = 'PROXY engine (BS + estimate_iv), unadjusted closes'
    else:
        fixed_params = {**FIXED_PARAMS, 'fill': fill}
        engine_tag = f'REAL chains, {fill} fills'
        if args.stop_loss_mult is not None:
            fixed_params['stop_loss_mult'] = args.stop_loss_mult
            engine_tag += f', {args.stop_loss_mult:g}x stop'
    records = walk_forward_real(dates, prices, store, PARAM_GRID,
                                fixed_params=fixed_params,
                                train_years=args.train_years,
                                min_trades=args.min_trades,
                                engine=args.prices)

    floor_tag = f', >={args.min_trades}-trade selection floor' if args.min_trades else ''
    print(f'\n=== {ticker} walk-forward on {engine_tag} '
          f'({len(records)} periods, {args.train_years}y train{floor_tag}) ===')
    if args.prices == 'proxy':
        print('  dte grid is CALENDAR days on both engines; proxy trading-day '
              'equivalents: 21->14, 30->21, 45->31 (defaults comparator 30->21)')
    print(f"  {'test window':<24}{'delta':>6}{'dte':>5}{'close':>6}"
          f"{'IS-shp':>8}{'IS-tr':>6}{'<30':>5}{'OOS%':>8}{'FIX%':>8}{'B&H%':>8}")
    for r in records:
        p = r['best_params']
        print(f"  {r['test_start']} -> {r['test_end']}"
              f"{p['call_delta']:>6.2f}{int(p['dte']):>5}{p['close_at_pct']:>6.2f}"
              f"{r['train_sharpe']:>8.2f}{r['n_trades']:>6}{r['n_below_30']:>5}"
              f"{r['oos_return_pct']:>8.2f}{r['fixed_return_pct']:>8.2f}"
              f"{r['bh_return_pct']:>8.2f}")

    n = len(records)
    print('\n  What the optimizer chose, per axis:')
    for axis in ('call_delta', 'dte', 'close_at_pct'):
        counts = Counter(r['best_params'][axis] for r in records)
        line = ', '.join(f'{v:g}: {c}/{n}' for v, c in sorted(counts.items()))
        print(f'    {axis:<14}{line}')

    def _avg(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    combo_counts = Counter(
        (r['best_params']['call_delta'], r['best_params']['dte'],
         r['best_params']['close_at_pct']) for r in records)
    print('\n  Wins by combination (periods where the triple won the in-sample fit):')
    print(f"    {'delta':>5}{'dte':>5}{'close':>6}{'wins':>6}{'avg IS-shp':>11}"
          f"{'avg IS-tr':>10}{'avg OOS%':>9}{'avg B&H%':>9}{'OOS-B&H':>9}")
    for (d, t, c), k in combo_counts.most_common():
        rows = [r for r in records
                if (r['best_params']['call_delta'], r['best_params']['dte'],
                    r['best_params']['close_at_pct']) == (d, t, c)]
        print(f"    {d:>5.2f}{int(t):>5}{c:>6.2f}{k:>6}{_avg(rows, 'train_sharpe'):>11.2f}"
              f"{_avg(rows, 'n_trades'):>10.1f}{_avg(rows, 'oos_return_pct'):>9.2f}"
              f"{_avg(rows, 'bh_return_pct'):>9.2f}"
              f"{_avg(rows, 'oos_return_pct') - _avg(rows, 'bh_return_pct'):>9.2f}")

    floor_fail = [r for r in records if r['n_trades'] < 30]
    print(f'\n  Pardo 30-trade floor: winning fit below 30 IS trades in '
          f'{len(floor_fail)}/{n} windows'
          + (f" ({', '.join(r['test_start'][:7] for r in floor_fail)})" if floor_fail else ''))
    print(f'  Grid combos under 30 IS trades per window: '
          f'min {min(r["n_below_30"] for r in records)}, '
          f'max {max(r["n_below_30"] for r in records)} of 27')

    wf = _chain([r['oos_return_pct'] for r in records])
    fx = _chain([r['fixed_return_pct'] for r in records])
    bh = _chain([r['bh_return_pct'] for r in records])
    span = f"{records[0]['test_start']} -> {records[-1]['test_end']}"
    print(f'\n  Chained OOS cumulative return, per-window $100K restarts ({span}):')
    print(f'    walk-forward picks   {wf:>10.2f}%')
    print(f'    fixed defaults       {fx:>10.2f}%   '
          f'(delta 0.25, dte 30cal, close 0.75)')
    print(f'    buy-and-hold         {bh:>10.2f}%')
    print(f'    WF wins fixed in {sum(1 for r in records if r["oos_return_pct"] > r["fixed_return_pct"])}/{n} windows, '
          f'beats B&H in {sum(1 for r in records if r["oos_return_pct"] > r["bh_return_pct"])}/{n}')


if __name__ == '__main__':
    main()
