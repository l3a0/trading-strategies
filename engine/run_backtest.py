"""Generalized covered-call backtest runner — any ticker, any analysis.

A thin CLI over the cc_backtest.py engine: point it at a ticker and run any
of the engine's analyses. Non-invasive — it imports the engine and never
edits it, so the MSFT regression tests stay green.

Usage:
    python run_backtest.py QQQ                          # every analysis on QQQ
    python run_backtest.py MSFT --test overlay significance
    python run_backtest.py AAPL --download              # force a fresh data pull
    python run_backtest.py QQQ  --test walk-forward --train-years 2
    python run_backtest.py SPY  --call-delta 0.30 --dte 30 --test overlay

Data: reads {ticker}_10yr_prices.csv (yfinance format). Downloads it via
download_prices.py if the file is missing or --download is passed.

Tests: overlay, significance, risk-managed, regime, dof, sensitivity,
       walk-forward, monte-carlo   (or "all", the default).
       walk-forward and monte-carlo re-run the backtest hundreds of times,
       so they are the slow ones.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from engine.cc_backtest import (
    _param_combinations,
    compute_statistics,
    degrees_of_freedom,
    monte_carlo_shuffle,
    regime_analysis,
    run_cc_overlay,
    sensitivity_analysis,
    walk_forward_optimization,
)

# Run order: fast analyses first, the two slow ones (hundreds of re-runs) last.
ALL_TESTS = [
    'overlay',
    'significance',
    'risk-managed',
    'regime',
    'dof',
    'sensitivity',
    'walk-forward',
    'monte-carlo',
]

# Canonical walk-forward / DOF grid — mirrors test_cc_backtest.py and the notebook.
PARAM_GRID: dict[str, list[float]] = {
    'call_delta': [0.15, 0.20, 0.25],
    'dte': [21, 30, 45],
    'close_at_pct': [0.50, 0.75, 1.00],
}


def load_prices(path: str):
    """Load a yfinance-format CSV (3 header rows, then date,close) -> (dates, prices).

    Mirrors cc_backtest.py's __main__ parser; works for any ticker's file.
    """
    df = pd.read_csv(path, skiprows=3, header=None, names=['date', 'close'])
    return df['date'].tolist(), df['close'].to_numpy(dtype=float)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---- individual analyses (each prints its own block) ----

def report_overlay(s: dict) -> None:
    section('Overlay — covered-call vs. buy & hold')
    print(f"  Capital:             ${s['capital']:>13,.2f}")
    print(f"  Contracts:           {s['num_contracts']:>14}   "
          f"(${s['initial_stock_cost']:,.0f} stock + ${s['cash']:,.0f} cash)")
    print(f"  Buy & Hold Final:    ${s['buy_hold_final']:>13,.2f}   {s['buy_hold_return_pct']:>+8.2f}%")
    print(f"  + Net Overlay P&L:   ${s['net_overlay_pnl']:>13,.2f}   {s['excess_return_pct']:>+8.2f} pp")
    print(f"  = CC Overlay Final:  ${s['final_equity']:>13,.2f}   {s['total_return_pct']:>+8.2f}%")
    print(f"  Premium Collected:   ${s['total_premium_collected']:>13,.2f}   "
          f"({s['num_calls_sold']} calls, {s['premium_retention_pct']:.1f}% retained)")
    print(f"  Win Rate:            {s['win_rate']:>13.1f}%")
    print(f"  Max Drawdown:        {s['max_drawdown_pct']:>13.2f}%")


def report_significance(s: dict, daily_equity) -> None:
    section('Statistical Significance  (H0: overlay adds zero value vs. buy & hold)')
    st = compute_statistics(daily_equity, num_contracts=s['num_contracts'], cash=s['cash'])
    print(f"  Days / Years:        {st['n_days']} / {st['years_of_data']}")
    print(f"  Ann. Excess Return:  {st['ann_excess_return_pct']:>+8.3f}%")
    print(f"  Ann. Excess Vol:     {st['ann_excess_vol_pct']:>8.2f}%")
    print(f"  Sharpe of Excess:    {st['sharpe_excess']:>+8.3f}")
    print(f"  t-stat (naive IID):  {st['t_stat_naive']:>+8.2f}   (inflated — ignores autocorrelation)")
    print(f"  t-stat (Newey-West): {st['t_stat_newey_west']:>+8.2f}   (correct; lag L={st['nw_lag']})")
    print(f"  Clears t=2 / t=3:    {st['passes_t_2']!s} / {st['passes_t_3']!s}")


def report_risk_managed(dates, prices, params, s: dict, daily_equity) -> None:
    section('Risk-Managed (Delta-Hedged) vs. Naive  — Israelov & Nielsen (2015)')
    naive = compute_statistics(daily_equity, num_contracts=s['num_contracts'], cash=s['cash'])
    hs, _, h_daily = run_cc_overlay(dates, prices, {**params, 'delta_hedge': 1.0})
    hedged = compute_statistics(h_daily, num_contracts=hs['num_contracts'], cash=hs['cash'])
    print(f"  {'Metric':<24}{'Naive':>14}{'Risk-Managed':>16}")
    print(f"  {'-' * 24}{'-' * 14}{'-' * 16}")
    print(f"  {'Excess Return / yr':<24}{naive['ann_excess_return_pct']:>+13.3f}%{hedged['ann_excess_return_pct']:>+15.3f}%")
    print(f"  {'Excess Vol / yr':<24}{naive['ann_excess_vol_pct']:>13.2f}%{hedged['ann_excess_vol_pct']:>15.2f}%")
    print(f"  {'Sharpe of Excess':<24}{naive['sharpe_excess']:>+14.3f}{hedged['sharpe_excess']:>+16.3f}")
    print(f"  {'t-stat (Newey-West)':<24}{naive['t_stat_newey_west']:>+14.2f}{hedged['t_stat_newey_west']:>+16.2f}")


def report_regime(dates, prices, trades) -> None:
    section('Regime Analysis — realized overlay P&L by market regime')
    reg = regime_analysis(dates, prices, trades)
    print(f"  {'Regime':<10}{'Days':>8}{'Total P&L':>16}{'P&L / Day':>14}")
    print(f"  {'-' * 10}{'-' * 8}{'-' * 16}{'-' * 14}")
    for name in ('bull', 'bear', 'sideways', 'unknown'):
        r = reg[name]
        print(f"  {name:<10}{r['days']:>8}{r['total_pnl']:>16,.2f}{r['avg_pnl_per_day']:>14,.2f}")


def report_dof(dates, prices, train_years: int) -> None:
    section(f'Degrees of Freedom — {train_years}-year in-sample window (Pardo 2008)')
    train_cut = pd.to_datetime(dates[0]) + pd.DateOffset(years=train_years)
    is_dates = [d for d in dates if pd.to_datetime(d) < train_cut]
    is_prices = prices[:len(is_dates)]
    base = {'risk_free_rate': 0.045, 'capital': 100_000}
    counts = sorted(
        int(run_cc_overlay(is_dates, is_prices, {**base, **c})[0]['num_calls_sold'])
        for c in _param_combinations(PARAM_GRID)
    )
    median_trades = counts[len(counts) // 2]
    dof = degrees_of_freedom(len(is_dates), n_parameters=3, indicator_lookback=30, n_trades=median_trades)
    print(f"  Observations (days): {dof['n_observations']:>8}")
    print(f"  Consumed (3+LB30):   {dof['consumed']:>8}")
    print(f"  Remaining (free):    {dof['remaining']:>8}   ({dof['pct_remaining'] * 100:.1f}% — Pardo floor 90%)")
    print(f"  Bar-level adequate?  {dof['passes_dof']!s:>8}   (necessary, not sufficient)")
    print(f"  Median trades:       {dof['n_trades']:>8}   (grid range {counts[0]}-{counts[-1]})")
    print(f"  >= 30 trades?        {dof['passes_trades']!s:>8}")


def report_sensitivity(dates, prices, params) -> None:
    section('Sensitivity — robustness to one-param-at-a-time perturbations')
    sens = sensitivity_analysis(dates, prices, params)
    for name, d in sens.items():
        verdict = 'robust' if d['worst_drop_pct'] < 10 else 'fragile'
        print(f"  {name}:  base {d['base_return']:+.1f}%, worst {d['worst_return']:+.1f}%, "
              f"max drop {d['worst_drop_pct']:.1f}%  ->  {verdict}")
        for off, ret in d['returns']:
            print(f"      offset {off:>+5.2f}:  {ret:>+8.1f}%")


def report_walk_forward(dates, prices, train_years: int) -> None:
    section(f'Walk-Forward Optimization — {train_years}-yr train / 6-mo test (slow)')
    oos_equity, records = walk_forward_optimization(dates, prices, PARAM_GRID, train_years=train_years)
    if not records:
        print('  No complete walk-forward windows for this data span.')
        return
    print(f"  Periods: {len(records)}")
    print(f"  {'test window':<25}{'chosen params':<40}{'train Sh':>9}{'trades':>8}")
    print(f"  {'-' * 25}{'-' * 40}{'-' * 9}{'-' * 8}")
    for r in records:
        bp = r['best_params']
        bp_str = f"delta={bp.get('call_delta')}, dte={bp.get('dte')}, close={bp.get('close_at_pct')}"
        window = f"{r['test_start']}->{r['test_end']}"
        print(f"  {window:<25}{bp_str:<40}{r['train_sharpe']:>9}{r['n_trades']:>8}")

    # Compounded out-of-sample return: each window's equity restarts at capital,
    # so chain per-window growth (eq_end / eq_start) across the windows. Use a
    # HALF-OPEN interval [test_start, test_end): each window's test_end equals the
    # next window's test_start, so a closed <= would double-count that shared
    # boundary day and collapse the chained return. (Matches test_cc_backtest.py's
    # test_walk_forward_optimization convention.)
    oos = oos_equity.copy()
    oos['date'] = pd.to_datetime(oos['date'])
    cum = 1.0
    for r in records:
        mask = (oos['date'] >= pd.Timestamp(r['test_start'])) & (oos['date'] < pd.Timestamp(r['test_end']))
        eq = oos.loc[mask, 'equity'].to_numpy(dtype=float)
        if len(eq) >= 2:
            cum *= eq[-1] / eq[0]
    print(f"  Compounded OOS return (rules locked per window): {(cum - 1) * 100:+.2f}%")


def report_monte_carlo(dates, prices, params, n_shuffles: int, seed: int) -> None:
    section(f'Monte Carlo Shuffle — {n_shuffles} paths, seed {seed} (slow)')
    mc = monte_carlo_shuffle(dates, prices, params, n_shuffles=n_shuffles, seed=seed)
    print(f"  Real (ordered) path: {mc['real_return']:>+9.2f}%")
    print(f"  Shuffled mean:       {mc['mc_mean']:>+9.2f}%")
    print(f"  Shuffled best:       {mc['mc_max']:>+9.2f}%")
    print(f"  Percentile:          {mc['percentile']:>9}   "
          f"(real beat {mc['percentile']}% of {mc['n_completed']} shuffles; ~50 = no edge from ordering)")


# ---- driver ----

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ticker', help='Stock/ETF symbol, e.g. QQQ, MSFT, SPY, AAPL')
    p.add_argument('--test', nargs='+', default=['all'], choices=[*ALL_TESTS, 'all'],
                   metavar='TEST', help="Which analyses to run (default: all). " + ', '.join(ALL_TESTS))
    p.add_argument('--download', action='store_true', help='Force a fresh yfinance download')
    p.add_argument('--call-delta', type=float, default=0.25)
    p.add_argument('--dte', type=int, default=21)
    p.add_argument('--close-at-pct', type=float, default=0.75)
    p.add_argument('--rfr', type=float, default=0.045, help='Risk-free rate (default 0.045)')
    p.add_argument('--capital', type=float, default=100_000)
    p.add_argument('--train-years', type=int, default=3, help='Walk-forward / DOF train window (default 3)')
    p.add_argument('--mc-shuffles', type=int, default=500, help='Monte Carlo paths (default 500)')
    p.add_argument('--seed', type=int, default=42, help='Monte Carlo seed (default 42)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    csv_path = f'{ticker.lower()}_10yr_prices.csv'

    if args.download or not os.path.exists(csv_path):
        from pipeline.download_prices import download_prices  # lazy: only import yfinance when needed
        print(f"Downloading {ticker} (10y) -> {csv_path} ...")
        download_prices(ticker, '10y', csv_path)

    dates, prices = load_prices(csv_path)
    params = {
        'call_delta': args.call_delta,
        'close_at_pct': args.close_at_pct,
        'dte': args.dte,
        'risk_free_rate': args.rfr,
        'capital': args.capital,
    }
    tests = ALL_TESTS if 'all' in args.test else args.test

    # Run the overlay once; overlay/significance/risk-managed/regime reuse it.
    summary, trades, daily_equity = run_cc_overlay(dates, prices, params)
    print(f"\n{ticker}  |  {dates[0]} -> {dates[-1]}  |  {len(dates)} trading days")
    print(f"params: call_delta={params['call_delta']}, dte={params['dte']}, "
          f"close_at_pct={params['close_at_pct']}, rfr={params['risk_free_rate']}, "
          f"capital=${params['capital']:,.0f}")

    for t in tests:
        if t == 'overlay':
            report_overlay(summary)
        elif t == 'significance':
            report_significance(summary, daily_equity)
        elif t == 'risk-managed':
            report_risk_managed(dates, prices, params, summary, daily_equity)
        elif t == 'regime':
            report_regime(dates, prices, trades)
        elif t == 'dof':
            report_dof(dates, prices, args.train_years)
        elif t == 'sensitivity':
            report_sensitivity(dates, prices, params)
        elif t == 'walk-forward':
            report_walk_forward(dates, prices, args.train_years)
        elif t == 'monte-carlo':
            report_monte_carlo(dates, prices, params, args.mc_shuffles, args.seed)


if __name__ == '__main__':
    main()
