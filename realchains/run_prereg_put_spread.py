"""Registered put-credit-spread experiment — the one-shot run.

Executes docs/prereg_put_credit_spread.md exactly (registration effective at
merge commit 4ddbbbe, PR #133; Amendment 1 — the bracket75 exit variant and
the 69-cell lattice — recorded at PR #134's merge). This is the analysis code
the results PR cites, alongside the registration merge commit (never a later
amendment commit, per section 10).

Order of operations (all print-only; the results tests pin from re-runs):

1. Arm C1 — the drift alarm, FATAL: the campaign's committed cell re-run at
   the campaign's exact coordinates (search.edge_search._load_ticker_data,
   live CHAIN_CLEAN_START clip, STRUCTURE_END) must reproduce NW t
   -0.91 +/- 0.02 before any other number is read.
2. SPY (registered span 2010-12-01 .. 2026-06-05, merged calls+puts store,
   day grid = call days): arm A (the 69-cell walk-forward verdict arm),
   arm B (unhedged replay of A's winners), arm C2 (the central cell forced
   through the identical machinery), the entry-only and exit-only ablations,
   arm E (20 entry-jitter careers, seeds 20260717+i), the stationary block
   bootstrap (seed 20260718) and leave-one-year-out on the stitched stream.
3. IWM (its single both-wings store, same span convention): arms A, B, C2.
4. The section-8 verdict block, with the pre-committed qualifiers.

No section-5 number may exist before this module's merge to main; the first
real-data execution happens after, C1-gated. Stores load sequentially and
are freed between tickers (the one-store budget).
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from typing import Any, Optional

from common.paths import data_path
from realchains.real_cc_backtest import (
    REGISTERED_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    open_dailies,
)
from realchains.vol_premium import run_real_credit_spread_overlay, short_vol_statistics
from realchains.walk_forward_structure import (
    BOOTSTRAP_SEED,
    CENTRAL_CELL,
    EXIT_VARIANTS,
    JITTER_K,
    JITTER_SEED,
    ROLL_MONTHS,
    TEST_MONTHS,
    TRAIN_YEARS,
    Cell,
    _window_frames,
    enumerate_joint_cells,
    jitter_select_factory,
    loyo_nw,
    replay_records,
    stationary_bootstrap,
    stitch_records,
    verdict_stats,
    walk_forward_structure,
)

SPAN_START = '2010-12-01'        # section 3.3: THIS registration's own frozen
                                 # choice (not REGISTERED_CLEAN_START's
                                 # authority — asserted equal as a cross-check)
SPAN_END = '2026-06-05'          # the puts file's last day (section 3.3)
PRICE_END = '2026-06-06'         # the price-load end (run_registered_vrp form)
C1_EXPECTED_T = -0.91            # the campaign's committed-cell pin
C1_TOLERANCE = 0.02              # plan D6 / prereg section 10
N_CAREERS = 20
COST_CURVE_BPS = (0.0, 0.2, 1.0)  # section 3.4: reported beside the 0.5 verdict


def c1_drift_alarm() -> float:
    """Arm C1: reproduce the campaign's SPY credit-spread cell at the
    campaign's exact coordinates. Aborts the whole run on a miss."""
    from search.edge_search import STRUCTURE_CAPITAL, _load_ticker_data

    store, dates, prices = _load_ticker_data('SPY')
    params = {'dte': 30, 'short_delta': 0.25, 'wing_delta': 0.10,
              'capital': STRUCTURE_CAPITAL}
    summary, _, eq = run_real_credit_spread_overlay(dates, prices, store, params)
    stats = short_vol_statistics(eq, summary['capital'])
    t = stats['t_stat_newey_west']
    print(f'C1 drift alarm: campaign-coordinate NW t = {t:+.2f} '
          f'(expected {C1_EXPECTED_T:+.2f} +/- {C1_TOLERANCE})')
    if abs(t - C1_EXPECTED_T) > C1_TOLERANCE:
        sys.exit('C1 DRIFT ALARM FAILED — no other number may be read. '
                 'Investigate engine/data drift before re-running.')
    del store
    return t


def _call_days(path: str) -> set:
    """Distinct dates in a chain CSV (or its .gz twin, via open_dailies) —
    the cheap scan that defines the call-day grid without a second store load."""
    days = set()
    with open_dailies(path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return days
        i = header.index('date')
        for row in reader:
            if row:
                days.add(row[i])
    return days


def load_spy() -> tuple:
    """SPY registered market: merged calls+puts store, day grid = call days
    clipped to [SPAN_START, SPAN_END] (section 3.3 — the merged union
    otherwise leaks calls-only days past the puts end)."""
    assert SPAN_START == REGISTERED_CLEAN_START['SPY']  # cross-check only
    calls = data_path('spy_option_dailies.csv')
    puts = data_path('spy_option_dailies_puts.csv')
    store = load_chain_store(calls, extra_paths=[puts], start=SPAN_START)
    call_days = _call_days(calls)
    px_dates, px = load_unadjusted_prices('SPY', SPAN_START, PRICE_END)
    price_map = dict(zip(px_dates, px))
    days = sorted(
        d for d in store
        if d in call_days and d in price_map and SPAN_START <= d <= SPAN_END
    )
    return store, days, [price_map[d] for d in days]


def load_iwm() -> tuple:
    """IWM confirmation market: one both-wings file, same span convention
    (the call-day clip is trivially satisfied — both wings share the file)."""
    assert SPAN_START == REGISTERED_CLEAN_START['IWM']  # cross-check only
    store = load_chain_store(data_path('iwm_option_dailies.csv'),
                             start=SPAN_START)
    px_dates, px = load_unadjusted_prices('IWM', SPAN_START, PRICE_END)
    price_map = dict(zip(px_dates, px))
    days = sorted(
        d for d in store if d in price_map and SPAN_START <= d <= SPAN_END
    )
    return store, days, [price_map[d] for d in days]


def _winner_table(records: list) -> None:
    print('  window            winner                     IS-sharpe  n  '
          '<30  minN  deployed$')
    for r in records:
        w = r['winner'].key() if r['winner'] is not None else 'SKIPPED'
        sh = '' if r.get('train_sharpe') is None else f"{r['train_sharpe']:+.3f}"
        n = '' if r.get('n_trades') is None else str(r['n_trades'])
        b30 = '' if r.get('n_below_30') is None else str(r['n_below_30'])
        mn = '' if r.get('min_grid_trades') is None else str(r['min_grid_trades'])
        print(f"  {r['test_start']}..{r['test_end']}  {w:<26} {sh:>8}  {n:>3}"
              f"  {b30:>3}  {mn:>4}  {r['deployed_notional']:>9,.0f}")
        if r.get('failed_cells'):
            print(f'    FAILED CELLS: {r["failed_cells"]}')


def _exit_reasons(records: list) -> dict[str, int]:
    total: Counter = Counter()
    for r in records:
        total.update(r.get('exit_reasons') or {})
    return dict(total)


def _chain(returns: list[float]) -> float:
    """Geometric chaining of per-window restart returns (walk_forward_real)."""
    growth = 1.0
    for r in returns:
        growth *= 1.0 + r
    return (growth - 1.0) * 100.0


def _window_test_prices(
    dates: list[str], prices: list[float], recs: list
) -> list[list[float]]:
    """The test-window price slices for `recs`, keyed by test_start — the
    buy-and-hold comparator's inputs on the identical window spans."""
    frames = _window_frames(dates, prices, TRAIN_YEARS, TEST_MONTHS, ROLL_MONTHS)
    by_start = {tdf['d'].iloc[0]: list(tdf['price']) for _, tdf in frames}
    return [by_start[r['test_start']] for r in recs]


def _axis_stability(records: list) -> dict[str, Counter]:
    axes: dict[str, Counter] = {
        'dte': Counter(), 'short_delta': Counter(), 'exit': Counter(),
    }
    for r in records:
        w: Optional[Cell] = r['winner']
        if w is None:
            axes['dte']['SKIPPED'] += 1
            axes['short_delta']['SKIPPED'] += 1
            axes['exit']['SKIPPED'] += 1
        else:
            axes['dte'][w.dte] += 1
            axes['short_delta'][w.short_delta] += 1
            axes['exit'][w.exit_name] += 1
    return axes


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(f'  {label}: ' + '  '.join(
        f'{k}={v:+.4f}' if isinstance(v, float) else f'{k}={v}'
        for k, v in sorted(stats.items())
    ))


def run_ticker(name: str, market: tuple, *, full: bool) -> dict[str, Any]:
    """Arms A/B/C2 (+ ablations and arm E when `full`, i.e. SPY)."""
    store, dates, prices = market
    print(f'\n===== {name}: {dates[0]} .. {dates[-1]} ({len(dates)} days) =====')

    cells = enumerate_joint_cells()
    print(f'arm A — walk-forward over {len(cells)} cells...')
    rec_a = walk_forward_structure(dates, prices, store, cells=cells)
    stitched_a, dates_a = stitch_records(rec_a)
    stats_a = verdict_stats(stitched_a)
    _winner_table(rec_a)
    for axis, counts in _axis_stability(rec_a).items():
        print(f'  axis {axis}: {dict(counts)}')
    print(f'  exit reasons (OOS, section 6.3): {_exit_reasons(rec_a)}')
    winner_is = [r['train_sharpe'] for r in rec_a
                 if r.get('train_sharpe') is not None]
    if winner_is:
        mean_is = sum(winner_is) / len(winner_is)
        wfe = stats_a['sharpe'] / mean_is if mean_is else float('nan')
        print(f'  WFE (Pardo lore, OOS sharpe / mean winner IS sharpe): '
              f'{stats_a["sharpe"]:+.3f} / {mean_is:+.3f} = {wfe:+.2f}')
    print(f'  seam charges total: '
          f'{sum(r["seam_charge"] for r in rec_a):,.2f}; '
          f'day-0 omission bound: {sum(r["day0_bound"] for r in rec_a):,.2f}')
    _print_stats('A (verdict arm, 0.5 bp)', stats_a)

    print(f'cost curve (section 3.4) — winners replayed at {COST_CURVE_BPS} bp...')
    cost_curve: dict[float, float] = {0.5: stats_a['t_newey_west']}
    for bps in COST_CURVE_BPS:
        rec_c = replay_records(rec_a, dates, prices, store, hedged=True,
                               extra_params={'hedge_cost_bps': bps})
        cost_curve[bps] = verdict_stats(stitch_records(rec_c)[0])['t_newey_west']
    print('  ' + '  '.join(f'{b}bp: t={cost_curve[b]:+.2f}'
                           for b in sorted(cost_curve)))

    print('arm B — unhedged replay of the winners...')
    rec_b = replay_records(rec_a, dates, prices, store, hedged=False)
    b_summaries = [r['oos_summary'] for r in rec_b if r['oos_summary']]
    raw_pnl = sum(s['net_pnl'] for s in b_summaries)
    interest = sum(s['interest_earned'] for s in b_summaries)
    alpha = sum(s['alpha_vs_cash'] for s in b_summaries)
    wins = sum(s['wins'] for s in b_summaries)
    losses = sum(s['losses'] for s in b_summaries)
    n_spreads = sum(s['num_credit_spreads_sold'] for s in b_summaries)
    win_rate = 100.0 * wins / max(1, wins + losses)
    residual = float(stitched_a.sum()) * 100_000.0
    worst_dd = max((s['max_drawdown_pct'] for s in b_summaries), default=0.0)
    b_chained = _chain([(s['final_equity'] - s['capital']) / s['capital']
                        for s in b_summaries])
    cash_chained = _chain([s['interest_earned'] / s['capital']
                           for s in b_summaries])
    bh_windows = [r for r in rec_b if r['oos_summary'] is not None]
    bh_chained = _chain([
        price_map_window[-1] / price_map_window[0] - 1.0
        for price_map_window in _window_test_prices(dates, prices, bh_windows)
    ])
    print(f'  B raw (binding clause, all three in one breath): '
          f'net {raw_pnl:+,.2f} = interest {interest:+,.2f} '
          f'+ delta P&L {alpha - residual:+,.2f} '
          f'+ hedged residual {residual:+,.2f} (arm A, net of 0.5 bp)')
    print(f'  B win rate {win_rate:.1f}% with per-spread mean '
          f'{raw_pnl / max(1, n_spreads):+,.2f} ({n_spreads} spreads); '
          f'worst per-window drawdown {worst_dd:.2f}%')
    print(f'  B scoreboard (chained per-window restarts): '
          f'B {b_chained:+.1f}% vs cash-at-rf {cash_chained:+.1f}% '
          f'vs buy-and-hold {bh_chained:+.1f}%')

    print('arm C2 — the central cell forced through the machinery...')
    rec_c2 = walk_forward_structure(
        dates, prices, store, forced_cell=CENTRAL_CELL
    )
    stats_c2 = verdict_stats(stitch_records(rec_c2)[0])
    _print_stats('C2 (fixed defaults, hedged)', stats_c2)
    rec_c2_u = replay_records(rec_c2, dates, prices, store, hedged=False)
    c2u = [r['oos_summary'] for r in rec_c2_u if r['oos_summary']]
    print(f'  C2 unhedged: net {sum(s["net_pnl"] for s in c2u):+,.2f} = '
          f'interest {sum(s["interest_earned"] for s in c2u):+,.2f} '
          f'+ alpha-vs-cash {sum(s["alpha_vs_cash"] for s in c2u):+,.2f}')

    out: dict[str, Any] = {
        'records_a': rec_a, 'records_b': rec_b, 'stats_a': stats_a,
        'stats_c2': stats_c2, 'stitched_a': stitched_a,
        'stitched_dates': dates_a, 'cost_curve': cost_curve,
        'b_decomposition': {'net': raw_pnl, 'interest': interest,
                            'alpha': alpha, 'residual': residual,
                            'win_rate': win_rate, 'worst_dd': worst_dd},
    }
    if not full:
        del store
        return out

    print('ablations — entry-only (9 hold cells) and exit-only (8 variants)...')
    entry_cells = [c for c in cells if c.exit_name == 'hold']
    rec_entry = walk_forward_structure(dates, prices, store, cells=entry_cells)
    _print_stats('entry-only ablation',
                 verdict_stats(stitch_records(rec_entry)[0]))
    exit_cells = [
        Cell(CENTRAL_CELL.dte, CENTRAL_CELL.short_delta,
             CENTRAL_CELL.wing_delta, name)
        for name, _ in EXIT_VARIANTS
    ]
    rec_exit = walk_forward_structure(dates, prices, store, cells=exit_cells)
    _print_stats('exit-only ablation',
                 verdict_stats(stitch_records(rec_exit)[0]))

    print(f'arm E — {N_CAREERS} entry-jitter careers (k={JITTER_K}, '
          f'seeds {JITTER_SEED}+i)...')
    career_ts: list[float] = []
    for i in range(N_CAREERS):
        rng = random.Random(JITTER_SEED + i)
        rec_e = replay_records(
            rec_a, dates, prices, store, hedged=True,
            select_factory=jitter_select_factory(rng),
        )
        career_ts.append(verdict_stats(stitch_records(rec_e)[0])['t_newey_west'])
    career_ts_sorted = sorted(career_ts)
    median = 0.5 * (career_ts_sorted[N_CAREERS // 2 - 1]
                    + career_ts_sorted[N_CAREERS // 2])
    rank = sum(1 for t in career_ts if stats_a['t_newey_west'] > t)
    print(f'  career t band: min {career_ts_sorted[0]:+.2f} / '
          f'median {median:+.2f} / max {career_ts_sorted[-1]:+.2f}; '
          f'careers above 2: {sum(1 for t in career_ts if t > 2)}/{N_CAREERS}; '
          f'verdict t above {rank} of {N_CAREERS} careers')
    out['career_ts'] = career_ts

    boot = stationary_bootstrap(stitched_a)
    print(f'  block bootstrap (seed {BOOTSTRAP_SEED}): '
          f'p_boot={boot["p_boot"]:.4f}')
    out['bootstrap'] = boot
    loyo = loyo_nw(stitched_a, dates_a)
    flips = sorted(y for y, t in loyo.items()
                   if (stats_a['t_newey_west'] > 2) != (t > 2))
    print(f'  LOYO t range: {min(loyo.values()):+.2f} .. '
          f'{max(loyo.values()):+.2f}; verdict-flipping years: {flips or "none"}')
    out['loyo'] = loyo
    out['loyo_flips'] = flips

    del store
    return out


def main() -> None:
    c1_t = c1_drift_alarm()

    spy = run_ticker('SPY', load_spy(), full=True)
    iwm = run_ticker('IWM', load_iwm(), full=False)

    t_spy = spy['stats_a']['t_newey_west']
    t_iwm = iwm['stats_a']['t_newey_west']
    spy_pass = t_spy > 2.0                     # section 7.2, strict
    iwm_confirm = t_iwm > 2.0
    # section 2.3 mechanism clause
    c2_positive = spy['stats_c2']['t_newey_west'] > 0.0
    sd_counts = Counter(
        r['winner'].short_delta for r in spy['records_a']
        if r['winner'] is not None
    )
    n_windows = len(spy['records_a'])
    modal_ok = bool(sd_counts) and max(sd_counts.values()) >= (n_windows // 2 + 1)
    mechanism = spy_pass and c2_positive and modal_ok

    print('\n================ VERDICT (prereg sections 7-8) ================')
    print(f'C1 drift alarm: {c1_t:+.2f} (passed)')
    print(f'SPY stitched OOS hedged-excess NW t = {t_spy:+.2f} '
          f'-> {"PASS" if spy_pass else "FAIL"} (bar: > 2, strict)')
    print(f'IWM stitched OOS hedged-excess NW t = {t_iwm:+.2f} '
          f'-> {"CONFIRMS" if iwm_confirm else "DOES NOT CONFIRM"}')
    print(f'mechanism clause: C2 t>0 {"MET" if c2_positive else "NOT MET"}; '
          f'modal short_delta >= {n_windows // 2 + 1}/{n_windows} '
          f'{"MET" if modal_ok else "NOT MET"} ({dict(sd_counts)}) '
          f'-> {"MET" if mechanism else "NOT MET"}')
    print('scope (section 2.4, every reporting surface): no-GFC span; '
          'daily-close exits; EOD stop-markets.')
    if spy_pass:
        # pass-scoped arm-E reads; both print when both hold (no precedence
        # is registered between them)
        above_all = t_spy > max(spy['career_ts'])
        robust = sum(1 for t in spy['career_ts'] if t > 2) >= N_CAREERS // 2
        if above_all:
            print('  qualifier: PLACEMENT-FRAGILE (verdict above all careers)')
        if robust:
            print('  qualifier: placement-robust (>= 10 of 20 careers above 2)')
        if not above_all and not robust:
            print('  qualifier: none (indeterminate at n=20)')
    if spy['loyo_flips']:  # direction-neutral, attaches pass or fail (7.3)
        print(f'  qualifier: SINGLE-YEAR-DEPENDENT ({spy["loyo_flips"]})')
    if spy_pass and iwm_confirm:
        row = 'section 8 row 1: confirmed (scoped: no-GFC span, daily-close exits)'
    elif spy_pass:
        row = 'section 8 row 2: index-specific, not confirmed'
    elif t_spy > 0:
        row = 'section 8 row 3: consistent with, not evidence for'
    else:
        row = ('section 8 row 4: null — optimization does not rescue the '
               'family; the campaign kill generalizes')
    print(f'OUTCOME -> {row}')
    print('(The section-8 sentence is published verbatim in the results doc; '
          'no result supports trading decisions.)')


if __name__ == '__main__':
    main()
