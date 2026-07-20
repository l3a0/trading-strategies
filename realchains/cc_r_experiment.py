"""SPY covered call re-measured in R-multiples — hedged and unhedged books,
exit grid, sizing battery (docs/spy_cc_r_experiment_plan.md).

Two books on identical cycles: Book U is the plain covered call
(`run_real_cc_overlay`, 0.25Δ / 30 DTE / 0.75 close / $100K, bid/ask fills);
Book H is the same run with `delta_hedge=True` (the engine's Israelov-Nielsen
risk-managed path). Entry and exit decisions never read the hedge, so the two
books' option cycles are identical by construction — the paired per-cycle
difference in R is the direction bill.

The accounting seam this module exists to fix (plan §3): the engine records
hedge share-trades only in the daily equity series, never in the trade-event
list, so a ledger built from events alone measures a hedged book on the raw
option-cycle basis — the trap flagged in the jitter exploration. Here each
cycle's P&L is attributed from the daily OVERLAY EXCESS series (equity minus
the reconstructed buy-and-hold), windowed (entry, close]:

- Book U: the attribution must reproduce the engine's own per-trade pnl
  (within daily 2dp-rounding cents) — a validation identity, asserted on
  every run, and the reason the helper can be trusted for Book H.
- Book H: the attribution IS the measurement — option P&L plus the hedge's
  share P&L inside the cycle window, divided by the same premium-collected
  risk basis as Book U. MAE for Book H is likewise excess-path based, so its
  R and MAE share one basis.
- Conservation: outside cycle windows (and an open tail, if the series ends
  with a position on) the excess series must be FLAT — no position, no hedge,
  no interest — so the summed (unrounded) cycle P&L plus the tail residual
  telescopes to the final overlay excess to float precision.
  `attribute_cycles` measures the worst out-of-window drift and the caller
  asserts it is at most quote-rounding width; on the real SPY books the
  cycles tile the span back-to-back, so the drift check is structural
  headroom there and the synthetic tests are what exercise it.

Epistemic class (plan header): EXPLORATORY measurement per the Gap E
precedent — sample-spending, kill-or-justify, never a registered verdict;
nothing enters the idea ledger and no e-value is spent. The engine's daily
Newey-West t (`compute_statistics`) remains the significance authority;
`sqn` / `r_newey_west_t` are reported, never gates — the trade-order score
grades per-cycle endpoints equal-weighted per premium dollar, the daily t
grades the actual dollar path, and why the daily judge outranks (path noise,
size weighting, the calendar, and the one-judge no-shopping rule) is
unpacked in docs/van_tharp_gap_a.md's "What each judge counts" passage. The
§8 escalation bar deliberately sits on the trade-order score, so a cell can
ring the bar while the authority stays unimpressed — by design, that
combination escalates to a human, never to a belief. The exit-grid knobs are
the engine's own. The deep-ITM close (quoted delta > 0.70) WAS an engine
invariant; the 2026-07-19 owner-directed widening made it the
`manage_deep_itm` param (the structure engine's existing name, default True
= every pinned run byte-identical), and the grid now crosses it: managed
cells keep their original keys and pins, `_noitm` cells disable the forced
buyback, so `close1.0_stopNone_noitm` is a literal hold-to-expiry.

Usage:
    python -m realchains.cc_r_experiment            # both arms, full report
    python -m realchains.cc_r_experiment --json X   # also dump the result dict
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from typing import Any, Sequence

import pandas as pd

from common.paths import data_path
from common.position_sizing import kelly_fraction, sizing_sweep
from common.trade_ledger import (
    TradeRecord,
    build_trade_ledger,
    ledger_statistics,
    regime_ledger_statistics,
)
from engine.cc_backtest import compute_statistics, six_regime_map
from realchains.real_cc_backtest import (
    CHAIN_CLEAN_START,
    REGISTERED_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
)

CC_R_SEED = 20260719
BASE_PARAMS: dict[str, Any] = {
    'call_delta': 0.25, 'dte': 30, 'close_at_pct': 0.75, 'capital': 100_000,
}
CLOSE_AT_GRID: tuple[float, ...] = (1.0, 0.75, 0.50)
STOP_GRID: tuple[float | None, ...] = (None, 2.0, 1.5)
SIZING_FRACTIONS: tuple[float, ...] = (0.005, 0.01, 0.02)
OVERWRITE_RATIOS: tuple[float, ...] = (1.0, 0.5, 0.25)
REGIME_WARM_START = '2009-06-01'   # ~200 trading days of SMA warmup before the live span


def exit_grid() -> list[tuple[float, float | None, bool]]:
    """The plan §4 3x3 (close_at_pct outer, stop inner) crossed with the
    deep-ITM knob — the 2026-07-19 owner-directed widening: manage_deep_itm
    True (the engine's historical invariant, every pinned run) vs False (the
    forced buyback at quoted delta > 0.70 disabled; positions ride to target,
    stop, or expiry). (0.75, None, True) is the published baseline cell;
    managed cells' keys are unchanged so the original pins stand untouched."""
    return [(c, s, m) for m in (True, False)
            for c in CLOSE_AT_GRID for s in STOP_GRID]


def overlay_excess(eq: pd.DataFrame, shares: int, initial_cash: float) -> list[float]:
    """Daily overlay excess: equity minus the reconstructed buy-and-hold
    (shares x unadjusted close + constant initial leftover cash — the
    engine's own comparator, per the summary['cash'] convention)."""
    return [float(e) - (float(p) * shares + initial_cash)
            for e, p in zip(eq['equity'], eq['price'])]


def open_tail_entry(trades: list[dict[str, Any]]) -> str | None:
    """Entry date of a position still open when the series ends (the ledger
    drops it; its excess movement must be excluded from the gap-flat check)."""
    last_entry: str | None = None
    for event in trades:
        if event.get('action') == 'sell':
            last_entry = str(event['date'])
        elif 'pnl' in event:
            last_entry = None
    return last_entry


def attribute_cycles(
    records: Sequence[TradeRecord],
    dates: Sequence[str],
    excess: Sequence[float],
    *,
    open_entry_date: str | None = None,
) -> tuple[list[TradeRecord], float, float, float]:
    """Re-derive each cycle's P&L (and MAE) from the excess path — the Book-H
    seam fix (plan §3).

    Cycle i's window is (entry-1, close]: its P&L is the excess at close
    minus the excess at the day before entry (0.0 when entry is day one),
    which telescopes across cycles and, on a hedged run, includes the hedge
    share-trades the event stream never records. MAE is the running minimum
    of the same windowed excess, floored Sweeney-style at min(mae, pnl, 0).

    Returns (adjusted records, gap_drift, tail_residual, raw_cycle_sum):

    - gap_drift — the largest single-day |excess| move OUTSIDE every cycle
      window (and outside the open tail): must be ~0, or the flat-between-
      cycles premise (no interest, hedge unwound at close) is broken. On a
      book whose cycles tile the whole span (back-to-back re-entry — the
      real SPY books) there are no outside days and the check is structural
      headroom; the synthetic tests are what exercise it.
    - tail_residual — excess accumulated by a still-open final position
      (0.0 when the series ends flat).
    - raw_cycle_sum — the UNROUNDED sum of windowed cycle P&Ls, so that
      raw_cycle_sum + tail_residual == final excess to float precision (the
      conservation identity the tests pin to the cent). Per-record pnl is
      rounded to 2dp for display/R, so summing record pnls instead would
      accumulate rounding noise ~ sqrt(n) cents.
    """
    idx = {d: i for i, d in enumerate(dates)}
    n = len(dates)
    covered = [False] * n
    out: list[TradeRecord] = []
    raw_cycle_sum = 0.0
    for rec in records:
        i0, i1 = idx[rec.entry_date], idx[rec.close_date]
        base = excess[i0 - 1] if i0 > 0 else 0.0
        pnl = excess[i1] - base
        raw_cycle_sum += pnl
        for j in range(i0, i1 + 1):
            covered[j] = True
        path_min = min(excess[j] - base for j in range(i0, i1 + 1))
        mae = min(path_min, pnl, 0.0)
        risk = rec.initial_risk
        out.append(replace(
            rec,
            pnl=round(pnl, 2),
            r_multiple=round(pnl / risk, 4),
            mae=round(mae, 2),
            mae_r=round(mae / risk, 4),
            outcome='win' if pnl >= 0 else 'loss',
        ))

    tail_residual = 0.0
    if open_entry_date is not None:
        i0 = idx[open_entry_date]
        base = excess[i0 - 1] if i0 > 0 else 0.0
        tail_residual = excess[-1] - base
        for j in range(i0, n):
            covered[j] = True

    gap_drift = 0.0
    for j in range(n):
        if covered[j]:
            continue
        step = abs(excess[j] - (excess[j - 1] if j > 0 else 0.0))
        gap_drift = max(gap_drift, step)
    return out, gap_drift, tail_residual, raw_cycle_sum


def paired_direction_bill(
    records_u: Sequence[TradeRecord], records_h: Sequence[TradeRecord],
) -> dict[str, Any]:
    """Per-cycle R_U - R_H on the shared cycles — the direction bill the
    unhedged book pays, in R units, exactly as plan §3 froze it (a NEGATIVE
    mean means Book U underperforms its hedged twin on the average cycle).
    `hedge_pnl_dollars` is the separately-labeled opposite flow: the hedge's
    own P&L, sum(H - U) in dollars. That number is not a trading profit —
    the hedge mirrors the short call's delta (long 0..1 of the base shares,
    never short), so it pays the buy-high/sell-low whipsaw in chop cycles
    and collects the drift ride in rally cycles, gaining almost exactly what
    the call's direction exposure loses; it is the direction bill seen from
    the repayment side, which is why U = H - hedge closes to the dollar.
    The two books' entry sequences are identical by construction; a mismatch
    means the runs diverged and the comparison is invalid, so raise."""
    if len(records_u) != len(records_h):
        raise ValueError(f'book cycle counts differ: {len(records_u)} vs {len(records_h)}')
    if not records_u:
        return {'n': 0, 'mean_r': 0.0, 'median_r': 0.0,
                'hedge_pnl_dollars': 0.0, 'share_positive': 0.0}
    diffs_r: list[float] = []
    hedge_dollars = 0.0
    for u, h in zip(records_u, records_h):
        if u.entry_date != h.entry_date or u.close_date != h.close_date:
            raise ValueError(f'cycle mismatch: U {u.entry_date}->{u.close_date} '
                             f'vs H {h.entry_date}->{h.close_date}')
        diffs_r.append(u.r_multiple - h.r_multiple)
        hedge_dollars += h.pnl - u.pnl
    diffs_sorted = sorted(diffs_r)
    mid = len(diffs_sorted) // 2
    median = (diffs_sorted[mid] if len(diffs_sorted) % 2
              else (diffs_sorted[mid - 1] + diffs_sorted[mid]) / 2)
    return {
        'n': len(diffs_r),
        'mean_r': round(sum(diffs_r) / len(diffs_r), 4),
        'median_r': round(median, 4),
        'hedge_pnl_dollars': round(hedge_dollars, 2),
        # Share of cycles where the bill is positive (U beat its hedged twin
        # — the hedge whipsawed); the rally cycles are the deep-negative tail.
        'share_positive': round(sum(1 for d in diffs_r if d > 0) / len(diffs_r), 4),
    }


def overwrite_ratio_curve(
    eq: pd.DataFrame, shares: int, initial_cash: float, capital: float,
    ratios: Sequence[float] = OVERWRITE_RATIOS,
) -> dict[float, dict[str, float]]:
    """The covered-call sizing dial (plan §5): scale the overlay to a fraction
    of the shares. equity_rho(t) = buy&hold(t) + rho x excess(t) — a
    deterministic blend; rho -> 0 converges to buy-and-hold. Max drawdown uses
    the engine's convention (peak clipped below at starting capital)."""
    excess = overlay_excess(eq, shares, initial_cash)
    bh = [float(p) * shares + initial_cash for p in eq['price']]
    out: dict[float, dict[str, float]] = {}
    for rho in ratios:
        series = [b + rho * x for b, x in zip(bh, excess)]
        peak = capital
        max_dd = 0.0
        for v in series:
            peak = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak)
        out[rho] = {'final_equity': round(series[-1], 2),
                    'max_drawdown_pct': round(max_dd * 100, 2)}
    return out


def regime_spread(cells: dict[str, dict[str, Any]]) -> float:
    """Max minus min expectancy_r over floor-passing named cells (plan §6) —
    the regime-structure size a hedge is predicted to shrink."""
    vals = [st['expectancy_r'] for name, st in cells.items()
            if name != 'unknown' and st.get('meets_floor')]
    return round(max(vals) - min(vals), 4) if len(vals) >= 2 else 0.0


def run_cell(
    dates: list[str], prices: list[float], store: dict[str, dict[str, Any]],
    *, hedged: bool, close_at_pct: float, stop_loss_mult: float | None,
    manage_deep_itm: bool = True,
) -> dict[str, Any]:
    """One engine pass -> summary + native/attributed ledgers + statistics.
    manage_deep_itm=True leaves the params dict without the key (the engine
    default) so every managed cell's engine call is byte-identical to the
    original run; False disables the delta>0.70 forced buyback."""
    params: dict[str, Any] = {**BASE_PARAMS, 'close_at_pct': close_at_pct}
    if stop_loss_mult is not None:
        params['stop_loss_mult'] = stop_loss_mult
    if hedged:
        params['delta_hedge'] = True
    if not manage_deep_itm:
        params['manage_deep_itm'] = False
    summary, trades, eq = run_real_cc_overlay(dates, prices, store, params)
    shares = summary['num_contracts'] * 100
    initial_cash = summary['cash']
    native = build_trade_ledger(trades, strategy='covered_call', ticker='SPY',
                                shares=shares, risk_basis='premium_collected')
    excess = overlay_excess(eq, shares, initial_cash)
    attributed, gap_drift, tail_residual, raw_cycle_sum = attribute_cycles(
        native, list(eq['date']), excess, open_entry_date=open_tail_entry(trades))
    if gap_drift > 0.02:
        raise AssertionError(f'excess drifts {gap_drift:.4f} outside cycle windows '
                             '— flat-between-cycles premise broken')
    if not hedged:
        # Book U identity: the attribution must reproduce the engine's own
        # per-trade pnl within daily 2dp-rounding cents. This validated, the
        # same helper is trusted for Book H, where it IS the measurement.
        worst = max((abs(a.pnl - n_.pnl) for a, n_ in zip(attributed, native)),
                    default=0.0)
        if worst > 0.05:
            raise AssertionError(f'Book U attribution deviates {worst:.4f} from '
                                 'engine pnl — attribution bug')
    records = attributed if hedged else native
    # Conservation from the UNROUNDED window deltas (see attribute_cycles):
    # exact to float precision, so the pin tolerance is a real cent, not
    # accumulated per-record rounding noise.
    conservation = round(raw_cycle_sum + tail_residual, 2)
    return {
        'summary': summary,
        'records': records,
        'ledger': ledger_statistics(records),
        'daily': compute_statistics(eq, num_contracts=summary['num_contracts'],
                                    cash=summary['cash']),
        'eq': eq,
        'excess_final': round(excess[-1], 2),
        'conservation_sum': conservation,
        'gap_drift': round(gap_drift, 4),
        'tail_residual': round(tail_residual, 2),
    }


def load_spy_market(start: str) -> tuple[dict[str, dict[str, Any]], list[str], list[float]]:
    store = load_chain_store(data_path('spy_option_dailies.csv'), start=start)
    days = sorted(store)
    dates, prices = load_unadjusted_prices('SPY', days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    return store, [d for d, _ in pairs], [p for _, p in pairs]


def _cell_key(close_at_pct: float, stop: float | None, manage_itm: bool = True) -> str:
    """Managed cells keep the original key format (their pins predate the
    deep-ITM widening); unmanaged cells append '_noitm'."""
    base = (f'close{close_at_pct:g}_stop{stop:g}' if stop is not None
            else f'close{close_at_pct:g}_stopNone')
    return base if manage_itm else base + '_noitm'


def _slim_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe projection of a cell (drop records/eq)."""
    led = cell['ledger']
    return {
        'n': led['n'],
        'expectancy_r': led['expectancy_r'],
        'win_rate': led['win_rate'],
        'avg_win_r': led['avg_win_r'],
        'avg_loss_r': led['avg_loss_r'],
        'worst_r': min((r.r_multiple for r in cell['records']), default=0.0),
        'mae_r': led['mae_r_distribution'],
        'sqn': led['sqn'],
        'r_newey_west_t': led['r_newey_west_t'],
        'daily_nw_t': round(cell['daily']['t_stat_newey_west'], 3),
        'sharpe_excess': round(cell['daily']['sharpe_excess'], 3),
        'net_overlay_pnl': cell['summary']['net_overlay_pnl'],
        'final_equity': cell['summary']['final_equity'],
        'max_drawdown_pct': cell['summary']['max_drawdown_pct'],
        'num_calls_sold': cell['summary']['num_calls_sold'],
        'conservation_sum': cell['conservation_sum'],
        'excess_final': cell['excess_final'],
        'gap_drift': cell['gap_drift'],
        'tail_residual': cell['tail_residual'],
    }


def run_experiment(arm: str = 'live') -> dict[str, Any]:
    """One arm end-to-end. 'live' (plan §2 primary): full grid + sizing +
    regime + pairing. 'registered' (comparability): baseline cells only."""
    start = (CHAIN_CLEAN_START if arm == 'live' else REGISTERED_CLEAN_START)['SPY']
    store, dates, prices = load_spy_market(start)
    result: dict[str, Any] = {'arm': arm, 'span': [dates[0], dates[-1]],
                              'n_days': len(dates)}

    # Registered arm: the baseline managed pair (the original comparability
    # cells, keys/pins unchanged) plus the TRUE-hold unmanaged pair — with
    # neither profit-take nor the deep-ITM buyback, close1.0/noitm is the
    # closest CC-frame replica of the short-vol +2.54 convention (hold every
    # cycle to settlement) on the same frozen span as that pin.
    grid = exit_grid() if arm == 'live' else [(0.75, None, True), (1.0, None, False)]
    cells: dict[str, dict[str, Any]] = {}
    for hedged in (False, True):
        book = 'H' if hedged else 'U'
        for close_at, stop, manage in grid:
            key = _cell_key(close_at, stop, manage)
            print(f'[{arm}] book {book} cell {key} ...', flush=True)
            cells[f'{book}:{key}'] = run_cell(
                dates, prices, store, hedged=hedged, close_at_pct=close_at,
                stop_loss_mult=stop, manage_deep_itm=manage)
    result['cells'] = {k: _slim_cell(c) for k, c in cells.items()}

    base_u = cells[f'U:{_cell_key(0.75, None)}']
    base_h = cells[f'H:{_cell_key(0.75, None)}']
    result['paired'] = paired_direction_bill(base_u['records'], base_h['records'])

    if arm == 'live':
        # Sizing battery (plan §5) on the baseline cells' R streams.
        sizing: dict[str, Any] = {}
        for book, cell in (('U', base_u), ('H', base_h)):
            rs = [r.r_multiple for r in cell['records']]
            maes = [r.mae_r for r in cell['records']]
            sweep = sizing_sweep(rs, fractions=SIZING_FRACTIONS,
                                 n_trades=len(rs), seed=CC_R_SEED, mae_r=maes)
            sizing[book] = {
                'kelly': round(kelly_fraction(rs), 4),
                'sweep': {f: {'p_ruin': s['p_ruin'],
                              'p_ruin_25dd': s['p_ruin_25dd'],
                              'terminal_median': s['terminal']['median']}
                          for f, s in sweep.items()},
            }
        result['sizing'] = sizing
        summary_u = base_u['summary']
        result['overwrite'] = overwrite_ratio_curve(
            base_u['eq'], summary_u['num_contracts'] * 100, summary_u['cash'],
            summary_u['capital'])
        summary_h = base_h['summary']
        result['overwrite_h'] = overwrite_ratio_curve(
            base_h['eq'], summary_h['num_contracts'] * 100, summary_h['cash'],
            summary_h['capital'])

        # Regime read (plan §6) — map warmed from REGIME_WARM_START. The
        # loader ignores start/end whenever the tracked unadjusted CSV
        # already exists, so the warmup is satisfied by the FILE's own start;
        # assert it, or a regenerated (shorter) file would silently push the
        # first ~200 live days into 'unknown' and move the pinned cells.
        warm_dates, warm_prices = load_unadjusted_prices(
            'SPY', REGIME_WARM_START, '2026-06-06')
        if warm_dates[0] > REGIME_WARM_START:
            raise AssertionError(
                f'unadjusted price file starts {warm_dates[0]}, after the '
                f'regime warm start {REGIME_WARM_START} — SMA warmup lost '
                '(was the tracked CSV regenerated from a clipped span?)')
        regime = six_regime_map(warm_dates, warm_prices)
        reg: dict[str, Any] = {}
        for book, cell in (('U', base_u), ('H', base_h)):
            cells_stats = regime_ledger_statistics(cell['records'], regime)
            reg[book] = {
                'cells': {name: {'n': st['n'],
                                 'expectancy_r': st['expectancy_r'],
                                 'win_rate': st['win_rate'],
                                 'meets_floor': st['meets_floor']}
                          for name, st in cells_stats.items()},
                'spread': regime_spread(cells_stats),
            }
        result['regime'] = reg
    return result


def main() -> None:
    json_path = None
    if '--json' in sys.argv:
        i = sys.argv.index('--json')
        if i + 1 >= len(sys.argv):
            sys.exit('--json needs a path argument')
        json_path = sys.argv[i + 1]
    out = {'live': run_experiment('live'),
           'registered': run_experiment('registered')}

    for arm, res in out.items():
        print(f"\n=== {arm} arm  {res['span'][0]} -> {res['span'][1]} "
              f"({res['n_days']} days) ===")
        print(f"{'cell':<24}{'n':>5}{'expR':>9}{'win%':>7}{'worstR':>9}"
              f"{'tradeNWt':>9}{'dailyNWt':>9}{'overlay$':>13}")
        for key, c in res['cells'].items():
            print(f"{key:<24}{c['n']:>5}{c['expectancy_r']:>9.4f}"
                  f"{c['win_rate']:>7.1f}{c['worst_r']:>9.3f}"
                  f"{c['r_newey_west_t']:>9.2f}{c['daily_nw_t']:>9.2f}"
                  f"{c['net_overlay_pnl']:>13,.0f}")
        p = res['paired']
        print(f"direction bill (U-H): mean {p['mean_r']:+.4f}R  "
              f"median {p['median_r']:+.4f}R  hedge P&L ${p['hedge_pnl_dollars']:+,.0f}  "
              f"bill>0 share {p['share_positive']:.2f}  (n={p['n']})")
        if 'sizing' in res:
            for book, s in res['sizing'].items():
                print(f"sizing {book}: kelly {s['kelly']:.4f}  "
                      + '  '.join(f"f={f:g}: ruin {v['p_ruin']:.4f}/25dd "
                                  f"{v['p_ruin_25dd']:.4f}"
                                  for f, v in s['sweep'].items()))
            for label, key in (('U', 'overwrite'), ('H', 'overwrite_h')):
                for rho, v in res[key].items():
                    print(f"overwrite {label} {rho:g}: final ${v['final_equity']:,.0f}  "
                          f"maxDD {v['max_drawdown_pct']:.2f}%")
            for book, r in res['regime'].items():
                print(f"regime {book}: spread {r['spread']:+.4f}R  " + '  '.join(
                    f"{name} n={st['n']} {st['expectancy_r']:+.3f}R"
                    + ('' if st['meets_floor'] else ' <30')
                    for name, st in r['cells'].items() if st['n'] > 0))

    if json_path:
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=1)
        print(f'\nJSON -> {json_path}')


if __name__ == '__main__':
    main()
