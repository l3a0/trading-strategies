"""Registered put-side VRP experiment — the one-shot run.

Executes docs/prereg_vol_premium.md exactly. The instrument (§2.1): a daily
delta-neutral short PUT at target delta -0.25, 30 DTE, hold-to-expiry, sold at the
bid, hedged with SHORT stock rebalanced daily on the signed vendor delta, on real
SPY (primary) and IWM (out-of-sample) chains over each ticker's CHAIN_CLEAN_START
clean span. The verdict (§4) is the rate-invariant Bakshi-Kapadia delta-hedged-gain
Newey-West t (short_vol_statistics) at hedge_cost_bps=0.5; the 0/0.2/1.0 bp t's are
the reported cost curve. Pass rule §5, outcome language §6.

This is the analysis code the results PR cites. The engine and the significance
measure are frozen at their TestSpyShortVolRegression (call-wing) form — the only
change is the wing (target_delta -0.25, option_type='put'), per §2.3.
"""
from __future__ import annotations

from typing import Any

from real_cc_backtest import CHAIN_CLEAN_START, load_chain_store, load_unadjusted_prices
from vol_premium import run_real_short_vol_overlay, short_vol_statistics

COSTS = [0.0, 0.2, 0.5, 1.0]
# Pinned SPY call-wing t's (TestSpyShortVolRegression), for the like-for-like
# comparison §4 / §1.3 require beside the put numbers.
CALL_WING_T = {0.0: 2.54, 0.2: 2.42, 0.5: 2.25, 1.0: 1.97}


def run_put(daily_path: str, ticker: str, rf: float = 0.045) -> tuple[list[str], dict[float, tuple[Any, Any]]]:
    store = load_chain_store(daily_path, start=CHAIN_CLEAN_START[ticker])
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    dts = [d for d, _ in pairs]
    pxs = [p for _, p in pairs]
    out: dict[float, tuple[Any, Any]] = {}
    for bps in COSTS:
        s, _, eq = run_real_short_vol_overlay(
            dts, pxs, store,
            {'target_delta': -0.25, 'dte': 30, 'capital': 100_000,
             'option_type': 'put', 'risk_free_rate': rf, 'hedge_cost_bps': bps})
        out[bps] = (s, short_vol_statistics(eq, s['capital'], rf=rf))
    return days, out


def report(ticker: str, days: list[str], out: dict[float, tuple[Any, Any]]) -> None:
    s0, _ = out[0.0]
    sold = s0.get('num_puts_sold', s0.get('num_calls_sold'))
    print(f"\n=== {ticker} short PUT -0.25d 30DTE  {days[0]} -> {days[-1]}  ({len(days)} chain days) ===")
    print(f"  contracts={s0['num_contracts']}  options_sold={sold}  win%={s0['win_rate']:.1f}")
    print(f"  {'bps':>5} {'NW_t':>7} {'sharpe':>8} {'annEx%':>7} {'volP&L$':>11} {'net$':>11} "
          f"{'hedge$':>9} {'maxDD%':>7} {'lag':>4} {'>2':>5}  callwing")
    for bps in COSTS:
        s, st = out[bps]
        print(f"  {bps:>5.1f} {st['t_stat_newey_west']:>7.2f} {st['sharpe']:>8.3f} "
              f"{st['ann_excess_return_pct']:>7.2f} {s['alpha_vs_cash']:>11.0f} {s['net_pnl']:>11.0f} "
              f"{s.get('total_hedge_cost', 0.0):>9.0f} {s['max_drawdown_pct']:>7.2f} {st['nw_lag']:>4} "
              f"{str(st['passes_t_2']):>5}   t={CALL_WING_T[bps]}")
    # full dicts for the 0.5 bp headline (so the results test can pin every field)
    s5, st5 = out[0.5]
    print(f"  -- {ticker} 0.5bp summary: " + ", ".join(f"{k}={v}" for k, v in sorted(s5.items())))
    print(f"  -- {ticker} 0.5bp stats:   " + ", ".join(f"{k}={v}" for k, v in sorted(st5.items())))


def main() -> None:
    spy_days, spy = run_put('spy_option_dailies_puts.csv', 'SPY')
    report('SPY', spy_days, spy)
    iwm_days, iwm = run_put('iwm_option_dailies.csv', 'IWM')
    report('IWM', iwm_days, iwm)

    spy_t0 = spy[0.0][1]['t_stat_newey_west']
    spy_t5 = spy[0.5][1]['t_stat_newey_west']
    iwm_t0 = iwm[0.0][1]['t_stat_newey_west']
    iwm_t5 = iwm[0.5][1]['t_stat_newey_west']
    spy_pass = spy_t5 > 2 and spy_t0 > 2          # §5.1
    mechanism = spy_t5 >= 2.54                     # §1.3 / §5.1
    iwm_confirm = iwm_t5 > 2 and iwm_t0 > 2        # §5.2
    confirmed = spy_pass and iwm_confirm           # §6 row 1

    print("\n================ VERDICT (prereg §5) ================")
    print(f"  SPY  gross t={spy_t0:+.2f}  net0.5bp t={spy_t5:+.2f}  -> H1 {'PASS' if spy_pass else 'FAIL'}")
    print(f"  mechanism clause (SPY net0.5bp >= +2.54 call wing): {'MET' if mechanism else 'NOT MET'}")
    print(f"  IWM  gross t={iwm_t0:+.2f}  net0.5bp t={iwm_t5:+.2f}  -> {'CONFIRMS' if iwm_confirm else 'DOES NOT CONFIRM'}")
    print(f"  ==> {'CONFIRMED' if confirmed else 'NOT CONFIRMED (see §6 row for exact language)'}")


if __name__ == '__main__':
    main()
