"""Covered-call overlay on REAL option chains — the honest re-run.

Mirrors cc_backtest.run_cc_overlay's rules exactly, but every option number
comes from the market instead of the Black-Scholes + estimate_iv proxy:

- Entry: the actual ~target-DTE / ~target-delta call from that day's chain,
  sold at the real BID (replaces the proxy's 3% slippage model; the $0.65
  commission per contract is kept on both legs).
- Marks: the contract's real mid (mark) each day for equity; carried forward
  on missing-quote days.
- Profit target & deep-ITM close: triggered on the real ASK (what a buyback
  actually costs) and the real delta; filled at the ask.
- Expiration: the contract's real expiration date (not a trading-day clock),
  settled against the UNADJUSTED close (strikes live in actual price space).
- Buy-and-hold benchmark: same unadjusted series, so dividends cancel in the
  excess-return comparison.

Data: {ticker}_option_dailies.csv from download_option_dailies.py and an
unadjusted close series (auto-downloaded to {ticker}_10yr_prices_unadjusted.csv).

Usage:
    python real_cc_backtest.py QQQ
    python real_cc_backtest.py MSFT msft_option_dailies_2008_2016.csv   # merge a backfill
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import sys
from datetime import datetime
from typing import Any, Sequence, TextIO

import pandas as pd

from cc_backtest import compute_statistics, run_cc_overlay

COMMISSION_PER_SHARE = 0.0065  # $0.65 per 100-share contract, both legs (engine convention)


def open_dailies(path: str) -> TextIO:
    """Open a dailies CSV, transparently falling back to its .gz twin.

    Neither file lives in git history (65-281MB): the .gz ships as a release
    asset (tag data-2026-06) fetched by fetch_option_data.sh locally and by
    the CI workflow, checksum-verified either way. The raw CSV, when present
    (fresh fetcher output or a local gunzip), is preferred for speed.
    """
    if os.path.exists(path):
        return open(path, newline='')
    gz = path + '.gz'
    if os.path.exists(gz):
        return io.TextIOWrapper(gzip.open(gz, 'rb'), encoding='utf-8', newline='')
    raise FileNotFoundError(f'{path} (or {gz})')


# ---- data loading ----

def load_unadjusted_prices(ticker: str, start: str, end: str) -> tuple[list[str], list[float]]:
    """Unadjusted closes (actual traded prices, matching option strikes)."""
    path = f'{ticker.lower()}_10yr_prices_unadjusted.csv'
    if not os.path.exists(path):
        import yfinance as yf  # lazy: only on first run
        raw = pd.DataFrame(yf.download(ticker, start=start, end=end, auto_adjust=False,
                                       progress=False))
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date', 'close'])
            for d, v in raw['Close'].itertuples():
                w.writerow([d.strftime('%Y-%m-%d'), float(v)])
        print(f'Saved unadjusted closes -> {path}')
    dates: list[str] = []
    closes: list[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            dates.append(row['date'])
            closes.append(float(row['close']))
    return dates, closes


def load_chain_store(
    path: str, extra_paths: Sequence[str] = ()
) -> dict[str, dict[str, Any]]:
    """One pass over the dailies CSV(s) -> per-date entry candidates + mark index.

    Returns {date: {'candidates': [(dte, delta, bid, ask, mid, expiration,
    strike, contractID), ...], 'marks': {contractID: (bid, ask, mid, delta)}}}.

    `extra_paths` merge additional dailies CSVs into the same store (e.g. the
    2008-2016 MSFT backfill alongside the canonical 2016-2026 file); the
    per-date setdefault makes the merge order-independent.

    Mark sanity clamp: a quoted mark outside [bid, ask] is bad vendor data —
    the 2008-2010 era of the backfill carries marks like 0.01 on a
    10.15/10.35 quote — so out-of-band marks are replaced by the quote
    midpoint. The canonical 2016+ files carry a small tail of these too
    (0.05-0.14% of rows), so the clamp applies uniformly, not per-era.
    """
    store: dict[str, dict[str, Any]] = {}
    for p in (path, *extra_paths):
        with open_dailies(p) as f:
            for r in csv.DictReader(f):
                d = r['date']
                try:
                    dte = int(r['dte'])
                    delta = float(r['delta'])
                    bid = float(r['bid'])
                    ask = float(r['ask'])
                    mid = float(r['mark'])
                    strike = float(r['strike'])
                except (TypeError, ValueError):
                    continue
                if not (bid <= mid <= ask):
                    mid = (bid + ask) / 2
                day = store.setdefault(d, {'candidates': [], 'marks': {}})
                day['candidates'].append(
                    (dte, delta, bid, ask, mid, r['expiration'], strike, r['contractID'])
                )
                day['marks'][r['contractID']] = (bid, ask, mid, delta)
    return store


def select_entry(
    day: dict[str, Any], target_dte: int, target_delta: float
) -> tuple[int, float, float, float, float, str, float, str] | None:
    """Nearest-DTE expiration, then nearest-delta call — mirrors the roll fetcher."""
    cands = [c for c in day['candidates'] if c[2] > 0 and 0.05 < c[1] < 0.60]
    if not cands:
        return None
    best_dte = min({c[0] for c in cands}, key=lambda x: abs(x - target_dte))
    cohort = [c for c in cands if c[0] == best_dte]
    return min(cohort, key=lambda c: abs(c[1] - target_delta))


# ---- the overlay loop (run_cc_overlay semantics, real prices) ----

def run_real_cc_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    call_delta = params.get('call_delta', 0.25)
    close_at_pct = params.get('close_at_pct', 0.75)
    dte = int(params.get('dte', 21))
    # 'bid_ask' (default): sell at bid, buy back at ask — executable worst case.
    # 'mid': both legs at the quoted mark — the academic convention; isolates
    # how much of the result is bid/ask spread vs the premium level itself.
    fill = str(params.get('fill', 'bid_ask'))

    initial_price = prices[0]
    contract_cost = initial_price * 100
    capital = float(params.get('capital', contract_cost))
    num_contracts = int(capital // contract_cost)
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts
    initial_cash = capital - shares * initial_price

    position: dict[str, Any] | None = None
    realized_pnl = 0.0
    num_calls_sold = 0
    total_premium_collected = 0.0
    wins = losses = 0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)

        if position is None:
            # Consider entry — needs a chain for today.
            if day is not None:
                pick = select_entry(day, dte, call_delta)
                if pick is not None:
                    _dte, _delta, bid, _ask, mid, expiration, strike, cid = pick
                    sell_px = bid if fill == 'bid_ask' else mid
                    net_premium = sell_px - COMMISSION_PER_SHARE  # real quote replaces slippage model
                    if net_premium > 0:
                        position = {
                            'strike': strike,
                            'premium_collected': net_premium,
                            'expiration': expiration,
                            'contract': cid,
                            'entry_date': date,
                            'last_mid': mid,
                            'real_delta': _delta,
                        }
                        num_calls_sold += 1
                        total_premium_collected += net_premium * shares
                        trades.append({'date': date, 'price': price, 'action': 'sell',
                                       'premium': net_premium, 'strike': strike,
                                       'contract': cid, 'dte': _dte, 'delta': _delta,
                                       'pnl': 0, 'realized_pnl': realized_pnl})
        else:
            if date >= position['expiration']:
                # Real expiration: settle against the unadjusted close of the
                # last trading day ON or BEFORE the expiration date. Every
                # expiration in the 2016+ datasets is a trading day, so this
                # is today's close — identical to the original lag-0
                # convention, and the pinned regressions hold. Pre-Feb-2015
                # standard expirations are SATURDAY-dated (the old listing
                # convention), so the first loop date past expiration is the
                # following Monday: settle against Friday's close, the last
                # day the option traded (a Good-Friday week settles against
                # Thursday's, for the same reason). A gap larger than a long
                # weekend means corrupt data, so fail loudly on that instead.
                if date == position['expiration']:
                    settle_price = price
                else:
                    assert prev_date is not None and prev_price is not None, (
                        f'position expired {position["expiration"]} before the '
                        f'first trading day of the series'
                    )
                    assert prev_date <= position['expiration'], (
                        f'last close {prev_date} is after expiration '
                        f'{position["expiration"]} — settlement logic error'
                    )
                    gap = (datetime.strptime(position['expiration'], '%Y-%m-%d')
                           - datetime.strptime(prev_date, '%Y-%m-%d')).days
                    assert gap <= 4, (
                        f'{gap} calendar days between last close {prev_date} and '
                        f'expiration {position["expiration"]} — missing data?'
                    )
                    settle_price = prev_price
                if settle_price >= position['strike']:
                    pnl = (position['premium_collected']
                           - (settle_price - position['strike'])) * shares
                else:
                    pnl = position['premium_collected'] * shares
                realized_pnl += pnl
                wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
                position = None
                trades.append({'date': date, 'price': settle_price, 'action': 'expiration',
                               'pnl': pnl, 'realized_pnl': realized_pnl})
            else:
                quote = day['marks'].get(position['contract']) if day else None
                if quote is not None:
                    bid_q, ask_q, mid_q, delta_q = quote
                    position['last_mid'] = mid_q
                    position['real_delta'] = delta_q
                    buy_px = ask_q if fill == 'bid_ask' else mid_q
                    buyback_cost = buy_px + COMMISSION_PER_SHARE
                    # Profit target: what the buyback actually costs vs premium kept.
                    hit_target = buy_px <= position['premium_collected'] * (1 - close_at_pct)
                    deep_itm = delta_q > 0.70
                    if hit_target or deep_itm:
                        pnl = (position['premium_collected'] - buyback_cost) * shares
                        realized_pnl += pnl
                        wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
                        action = 'close' if hit_target else 'close_itm'
                        position = None
                        trades.append({'date': date, 'price': price, 'action': action,
                                       'call_value': ask_q, 'pnl': pnl,
                                       'realized_pnl': realized_pnl})
                # No quote today: no close can trigger; mark carries forward below.

        equity = price * shares + initial_cash + realized_pnl
        if position is not None:
            equity += (position['premium_collected'] - position['last_mid']) * shares
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    total_return = (final_equity - capital) / capital * 100
    buy_hold_final = prices[-1] * shares + initial_cash
    buy_hold_return = (buy_hold_final - capital) / capital * 100
    net_overlay_pnl = final_equity - buy_hold_final
    retention = (net_overlay_pnl / total_premium_collected * 100
                 if total_premium_collected > 0 else 0.0)

    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())

    summary = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'initial_stock_cost': round(shares * initial_price, 2),
        'cash': round(initial_cash, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_final': round(buy_hold_final, 2),
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'excess_return_pct': round(total_return - buy_hold_return, 2),
        'net_overlay_pnl': round(net_overlay_pnl, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'overlay_costs': round(round(total_premium_collected, 2) - round(net_overlay_pnl, 2), 2),
        'premium_retention_pct': round(retention, 1),
        'num_calls_sold': num_calls_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / max(wins + losses, 1) * 100, 1),
        'max_drawdown_pct': round(max_dd, 2),
    }
    return summary, trades, daily_equity


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else 'QQQ').upper()
    extra_dailies = sys.argv[2:]  # e.g. msft_option_dailies_2008_2016.csv (the backfill)
    dailies_path = f'{ticker.lower()}_option_dailies.csv'
    for p in (dailies_path, *extra_dailies):
        if not (os.path.exists(p) or os.path.exists(p + '.gz')):
            sys.exit(f'{p}[.gz] not found — run download_option_dailies.py first')

    params = {'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
              'risk_free_rate': 0.045, 'capital': 100_000}
    # Parity note: the engine's dte=21 is TRADING days (T = 21/252 — about a
    # month), so the real leg must target the calendar-day equivalent or it
    # sells shorter, cheaper calls on a faster cycle than the proxy ever did.
    real_params = {**params, 'dte': round(params['dte'] / 252 * 365)}  # ~30 calendar days

    print(f'Loading chain store ({dailies_path}'
          + (f' + {", ".join(extra_dailies)}' if extra_dailies else '') + ') ...', flush=True)
    store = load_chain_store(dailies_path, extra_dailies)
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    # Clip the price series to the chain-covered span.
    lo, hi = days[0], days[-1]
    pairs = [(d, p) for d, p in zip(dates, prices) if lo <= d <= hi]
    dates = [d for d, _ in pairs]
    prices = [p for _, p in pairs]
    print(f'{ticker}: {len(dates)} trading days {dates[0]} -> {dates[-1]}, '
          f'{len(days)} chain days', flush=True)

    real_sum, real_trades, real_eq = run_real_cc_overlay(dates, prices, store, real_params)
    import numpy as np
    proxy_sum, _, proxy_eq = run_cc_overlay(dates, np.array(prices), params)
    real_st = compute_statistics(real_eq, num_contracts=real_sum['num_contracts'],
                                 cash=real_sum['cash'])
    proxy_st = compute_statistics(proxy_eq, num_contracts=proxy_sum['num_contracts'],
                                  cash=proxy_sum['cash'])

    print(f"\n=== {ticker} covered-call overlay: REAL chains vs PROXY (same unadjusted series) ===")
    rows = [
        ('Buy & hold return', 'buy_hold_return_pct', '%'),
        ('Overlay total return', 'total_return_pct', '%'),
        ('Net overlay P&L', 'net_overlay_pnl', '$'),
        ('Gross premium collected', 'total_premium_collected', '$'),
        ('Premium retention', 'premium_retention_pct', '%'),
        ('Calls sold', 'num_calls_sold', ''),
        ('Win rate', 'win_rate', '%'),
        ('Max drawdown', 'max_drawdown_pct', '%'),
    ]
    print(f"  {'metric':<26}{'REAL':>14}{'PROXY':>14}")
    print(f"  {'-' * 26}{'-' * 14}{'-' * 14}")
    def fmt(v: Any, unit: str) -> str:
        if unit == '$':
            return f'${v:,.0f}'
        return f'{v:,.2f}%' if unit == '%' else f'{v}'

    for label, key, unit in rows:
        print(f'  {label:<26}{fmt(real_sum[key], unit):>14}{fmt(proxy_sum[key], unit):>14}')
    print(f"  {'Ann. excess return':<26}{real_st['ann_excess_return_pct']:>13.3f}%"
          f"{proxy_st['ann_excess_return_pct']:>13.3f}%")
    print(f"  {'Sharpe of excess':<26}{real_st['sharpe_excess']:>14.3f}{proxy_st['sharpe_excess']:>14.3f}")
    print(f"  {'t-stat (Newey-West)':<26}{real_st['t_stat_newey_west']:>14.2f}"
          f"{proxy_st['t_stat_newey_west']:>14.2f}")


if __name__ == '__main__':
    main()
