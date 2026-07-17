"""Walk-forward driver for the registered put-credit-spread experiment.

Implements docs/prereg_put_credit_spread.md (registered at merge 4ddbbbe;
Amendment 1 widens the exit lattice to 8 variants / 69 joint cells) per the
implementation plan docs/put_spread_analysis_plan.md. The frozen rules live in
the prereg; this module only mechanizes them:

- the 69-cell joint lattice (9 derived-wing entry cells x 8 Gap E exit
  variants, minus the 3 invalid 21-DTE-entry x dte21-exit cells), in frozen
  lattice order (entry axes in section-3.1 order, then exit variants in
  section-3.2 order) — the order IS the registered tie-break;
- per-window in-sample selection by the UNROUNDED annualized Sharpe of the
  rf-netted daily hedged excess, a 30-entry floor, strict-greater comparison;
- the section-5.5 stitched stream: concatenated per-window excess arrays,
  a synthetic bid/ask close of any structure open at a window's end (the
  seam charge), and the reported day-0 entry-mark omission bound;
- replay machinery for arm B (unhedged), arm C2 (forced cell), and arm E
  (entry-calendar jitter, the Gap F emission-keyed wait);
- the section-7.3 companions: a stationary block bootstrap and a
  leave-one-year-out sweep over the stitched stream.

Engine access is deliberately DRIVER-SIDE (plan decision D1): the credit
spread's spec knobs are bound by hand around run_real_structure_overlay so
the hedged/unhedged switch uses the engine's existing hedge modes with a
zero-line engine diff — the random_entry_scout precedent. The hedged path is
pinned byte-identical to run_real_credit_spread_overlay by
tests/test_walk_forward_structure.py::TestHedgeOverrideEquivalence.

No function here touches data files: callers supply (dates, prices, store).
The registered runner is realchains/run_prereg_put_spread.py.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import pandas as pd

from common.stats import newey_west_summary
from realchains.real_cc_backtest import COMMISSION_PER_SHARE
from realchains.vol_premium import STRUCTURE_SPECS, run_real_structure_overlay

# --- frozen lattice (prereg sections 3.1-3.2, Amendment 1) -------------------

ENTRY_DTES = (21, 30, 45)
ENTRY_SHORT_DELTAS = (0.20, 0.25, 0.30)
WING_GAP = 0.05  # wing_delta = short_delta - WING_GAP, derived (section 3.1)

EXIT_VARIANTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ('hold', {}),
    ('target50', {'close_at_pct': 0.50}),
    ('target75', {'close_at_pct': 0.75}),
    ('stop2x', {'stop_loss_mult': 2.0}),
    ('stop3x', {'stop_loss_mult': 3.0}),
    ('dte21', {'exit_dte': 21}),
    ('bracket', {'close_at_pct': 0.50, 'stop_loss_mult': 2.0}),
    ('bracket75', {'close_at_pct': 0.75, 'stop_loss_mult': 1.5}),  # Amendment 1
)

TRAIN_YEARS = 4
TEST_MONTHS = 6
ROLL_MONTHS = 6
MIN_TRADES = 30           # entry count (num_credit_spreads_sold), section 5.2
DTE21_MIN_ACTUAL_DTE = 23  # dte21 cells skip entries with actual DTE <= 22
JITTER_K = 10             # arm E chain-day wait bound (Gap F convention)
JITTER_SEED = 20260717    # career i uses JITTER_SEED + i (plan D8)
BOOTSTRAP_SEED = 20260718  # plan D9, committed per prereg section 10
BOOTSTRAP_BLOCK = 21
BOOTSTRAP_B = 10_000


class Cell(NamedTuple):
    """One joint lattice cell: entry coordinates plus a named exit variant."""

    dte: int
    short_delta: float
    wing_delta: float
    exit_name: str

    def key(self) -> str:
        return f'dte{self.dte}/sd{self.short_delta:.2f}/{self.exit_name}'

    def params(self) -> dict[str, Any]:
        exit_params = dict(EXIT_PARAMS[self.exit_name])
        return {
            'dte': self.dte,
            'short_delta': self.short_delta,
            'wing_delta': self.wing_delta,
            'capital': 100_000,
            'risk_free_rate': 0.045,
            **exit_params,
        }


EXIT_PARAMS: dict[str, dict[str, Any]] = {name: params for name, params in EXIT_VARIANTS}

CENTRAL_CELL = Cell(30, 0.25, 0.20, 'hold')  # arm C2 / the ablations' anchor


def enumerate_joint_cells(
    entry_dtes: tuple[int, ...] = ENTRY_DTES,
    short_deltas: tuple[float, ...] = ENTRY_SHORT_DELTAS,
    exits: tuple[tuple[str, dict[str, Any]], ...] = EXIT_VARIANTS,
) -> list[Cell]:
    """The frozen-order joint lattice; dte21-exit x 21-DTE-entry excluded."""
    cells: list[Cell] = []
    for dte in entry_dtes:
        for sd in short_deltas:
            wd = round(sd - WING_GAP, 2)
            for exit_name, _ in exits:
                if exit_name == 'dte21' and dte == 21:
                    continue
                cells.append(Cell(dte, sd, wd, exit_name))
    return cells


# --- engine access (plan D1/D2: driver-side spec binding, zero engine diff) --

def _cell_select(cell: Cell) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """The cell's selector: the spec's, dte21-guard-wrapped when applicable."""
    base = STRUCTURE_SPECS['credit_spread']['select']
    if cell.exit_name != 'dte21':
        return base

    def guarded(day: dict[str, Any], params: dict[str, Any]) -> Any:
        picked = base(day, params)
        if picked is None:
            return None
        short = next((leg for leg in picked if leg['sign'] < 0), None)
        if short is None:
            return None
        cand = next(
            (c for c in day['candidates'] if c[7] == short['contract']), None
        )
        if cand is None or cand[0] < DTE21_MIN_ACTUAL_DTE:
            return None  # actual DTE <= 22 (or unverifiable): no entry
        return picked

    return guarded


def run_cell(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    cell: Cell,
    *,
    hedged: bool = True,
    select: Optional[Callable[[dict[str, Any], dict[str, Any]], Any]] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """One credit-spread overlay run — run_structure_via_spec's three lines
    with the hedge mode chosen here (the engine's existing 'combined'/'none').

    `extra_params` merges LAST (the section-3.4 cost-curve seam: replaying at
    hedge_cost_bps 0/0.2/1 beside the 0.5 verdict). None leaves the default
    path byte-identical (pinned by TestHedgeOverrideEquivalence)."""
    spec = STRUCTURE_SPECS['credit_spread']
    merged = {**spec['defaults'], **cell.params(), **(extra_params or {})}
    summary, trades, eq = run_real_structure_overlay(
        dates, prices, store, merged,
        select=select if select is not None else _cell_select(cell),
        entry_guard=spec['entry_guard'],
        hedge_mode='combined' if hedged else 'none',
        management=spec['management'],
    )
    return spec['summary'](summary, merged), trades, eq


# --- the excess stream and its statistics ------------------------------------

def excess_stream(daily_eq: pd.DataFrame, capital: float) -> np.ndarray:
    """The rf-netted per-capital daily excess (the short_vol_statistics
    recipe, which does not expose its array): diff(equity)/capital minus the
    engine's actual per-day rf credit (off by one: the credit inside
    eq[k+1]-eq[k] was applied at the start of day k+1)."""
    eq = daily_eq['equity'].to_numpy(dtype=float)
    pnl = np.diff(eq) / capital
    if 'rf_credit' in daily_eq.columns:
        rf = daily_eq['rf_credit'].to_numpy(dtype=float)
        pnl = pnl - rf[1:] / capital
    return pnl


def sharpe_unrounded(excess: np.ndarray) -> float:
    """Annualized Sharpe of the daily excess, unrounded (the selection
    metric of record, section 5.3). Empty -> -inf (never selectable)."""
    if len(excess) == 0:
        return -math.inf
    std = float(np.std(excess, ddof=1)) if len(excess) > 1 else 0.0
    if std <= 0.0:
        return 0.0
    return float(np.mean(excess)) / std * math.sqrt(252)


# --- seam accounting (section 5.5, plan D5) ----------------------------------

def _find_put_quote(
    store: dict[str, dict[str, Any]], date: str, leg: dict[str, Any]
) -> Optional[tuple]:
    """The leg's candidate tuple on `date`, matched by (expiration, strike,
    negative delta) — 'enter' legs_detail carries no contract ID."""
    day = store.get(date)
    if not day:
        return None
    for c in day['candidates']:
        if c[5] == leg['expiration'] and c[6] == leg['strike'] and c[1] < 0:
            return c
    return None


def _open_legs_at_end(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """legs_detail of a structure still open after the last event, else [].

    Single-expiration structures only: a staggered 'settle_leg' event (the
    calendar's near leg) would leave this over-charging the seam, so it
    fails loudly rather than mis-accounting."""
    open_legs: list[dict[str, Any]] = []
    for event in trades:
        if event['action'] == 'settle_leg':
            raise ValueError('seam accounting supports single-expiration '
                             'structures only (got settle_leg)')
        if event['action'] == 'enter':
            open_legs = list(event['legs_detail'])
        elif event['action'] in ('settle', 'close'):
            open_legs = []
    return open_legs


def seam_charge(
    store: dict[str, dict[str, Any]],
    window_dates: list[str],
    trades: list[dict[str, Any]],
    shares: int,
) -> float:
    """Dollars to synthetically close the window-end open structure at
    bid/ask + per-leg commission, relative to its mid marks. Quotes come
    from the LAST within-window day each leg is quoted (never a
    manufactured fill on an unquoted day)."""
    total = 0.0
    for leg in _open_legs_at_end(trades):
        quote = None
        for date in reversed(window_dates):
            quote = _find_put_quote(store, date, leg)
            if quote is not None:
                break
        if quote is None:
            continue
        _, _, bid, ask, mid = quote[0], quote[1], quote[2], quote[3], quote[4]
        if leg['sign'] < 0:
            total += (ask - mid) + COMMISSION_PER_SHARE  # buy the short back
        else:
            total += (mid - bid) + COMMISSION_PER_SHARE  # sell the long
    return total * shares


def day0_omission(
    store: dict[str, dict[str, Any]],
    window_dates: list[str],
    trades: list[dict[str, Any]],
    shares: int,
) -> float:
    """The window's diff-invisible entry mark (reported, never applied): when
    the first entry lands on the window's first day, its fill-vs-mid loss
    sits inside equity[0] and never enters the diff stream."""
    first = next((t for t in trades if t['action'] == 'enter'), None)
    if first is None or not window_dates or first['date'] != window_dates[0]:
        return 0.0
    total = 0.0
    for leg in first['legs_detail']:
        quote = _find_put_quote(store, first['date'], leg)
        if quote is None:
            continue
        bid, ask, mid = quote[2], quote[3], quote[4]
        if leg['sign'] < 0:
            total += (mid - bid) + COMMISSION_PER_SHARE  # sold at bid, marked mid
        else:
            total += (ask - mid) + COMMISSION_PER_SHARE  # bought at ask, marked mid
    return total * shares


# --- the walk-forward loop ---------------------------------------------------

RunCellFn = Callable[..., tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]]


def _window_frames(
    dates: list[str],
    prices: list[float],
    train_years: int,
    test_months: int,
    roll_months: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """The walk_forward_real window arithmetic, verbatim: pandas DateOffset,
    half-open boundaries, the len(train)<30 / len(test)<5 skip."""
    df = pd.DataFrame(
        {'date': pd.to_datetime(dates), 'd': dates, 'price': prices}
    )
    start_date = df['date'].iloc[0]
    end_date = df['date'].iloc[-1]
    current = start_date + pd.DateOffset(years=train_years)
    windows: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    while current + pd.DateOffset(months=test_months) <= end_date:
        train_start = current - pd.DateOffset(years=train_years)
        test_end = current + pd.DateOffset(months=test_months)
        train_df = df[(df['date'] >= train_start) & (df['date'] < current)]
        test_df = df[(df['date'] >= current) & (df['date'] < test_end)]
        if len(train_df) >= 30 and len(test_df) >= 5:
            windows.append((train_df, test_df))
        current += pd.DateOffset(months=roll_months)
    return windows


def _oos_record(
    store: dict[str, dict[str, Any]],
    test_df: pd.DataFrame,
    cell: Cell,
    *,
    hedged: bool,
    run_cell_fn: RunCellFn,
    select: Optional[Callable[..., Any]] = None,
    extra_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run one cell on one test window and package the section-5.5 pieces."""
    t_dates = list(test_df['d'])
    t_prices = list(test_df['price'])
    summary, trades, eq = run_cell_fn(
        t_dates, t_prices, store, cell, hedged=hedged, select=select,
        extra_params=extra_params,
    )
    capital = float(summary['capital'])
    shares = int(summary['num_contracts']) * 100
    stream = excess_stream(eq, capital)
    charge = seam_charge(store, t_dates, trades, shares)
    if len(stream) and charge:
        stream = stream.copy()
        stream[-1] -= charge / capital
    reasons = Counter(
        t['reason'] for t in trades if t['action'] == 'close' and 'reason' in t
    )
    return {
        'oos_excess': stream,
        'oos_dates': t_dates[1:],
        'seam_charge': charge,
        'day0_bound': day0_omission(store, t_dates, trades, shares),
        'oos_summary': summary,
        'oos_trades': int(summary['num_credit_spreads_sold']),
        'exit_reasons': dict(reasons),
        'deployed_notional': int(summary['num_contracts']) * 100 * t_prices[0],
    }


def _skipped_record(test_df: pd.DataFrame) -> dict[str, Any]:
    t_dates = list(test_df['d'])
    return {
        'oos_excess': np.zeros(max(0, len(t_dates) - 1)),
        'oos_dates': t_dates[1:],
        'seam_charge': 0.0,
        'day0_bound': 0.0,
        'oos_summary': None,
        'oos_trades': 0,
        'exit_reasons': {},
        'deployed_notional': 0.0,
    }


def walk_forward_structure(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    *,
    cells: Optional[list[Cell]] = None,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    roll_months: int = ROLL_MONTHS,
    min_trades: int = MIN_TRADES,
    forced_cell: Optional[Cell] = None,
    run_cell_fn: RunCellFn = run_cell,
) -> list[dict[str, Any]]:
    """The registered pipeline: per-window in-sample selection over `cells`
    (or `forced_cell` with no selection — arm C2 / the ablations' anchor),
    hedged OOS execution, section-5.5 packaging. Returns one record per
    window; stitch with `stitch_records`.

    `run_cell_fn` is a test seam: synthetic tests inject canned engines;
    the semantics of selection/stitching never depend on it.
    """
    if cells is None:
        cells = enumerate_joint_cells()
    records: list[dict[str, Any]] = []
    for train_df, test_df in _window_frames(
        dates, prices, train_years, test_months, roll_months
    ):
        rec: dict[str, Any] = {
            'train_start': train_df['d'].iloc[0],
            'train_end': train_df['d'].iloc[-1],
            'test_start': test_df['d'].iloc[0],
            'test_end': test_df['d'].iloc[-1],
        }
        if forced_cell is not None:
            winner: Optional[Cell] = forced_cell
            rec.update({'train_sharpe': None, 'n_trades': None,
                        'min_grid_trades': None, 'n_below_30': None})
        else:
            tr_dates = list(train_df['d'])
            tr_prices = list(train_df['price'])
            best_sharpe = -math.inf
            winner = None
            winner_trades = 0
            grid_trades: list[int] = []
            failed_cells: list[str] = []
            for cell in cells:
                try:
                    summary, _, eq = run_cell_fn(
                        tr_dates, tr_prices, store, cell, hedged=True,
                        select=None, extra_params=None,
                    )
                except ValueError as exc:
                    # the engine's known benign raise (capital < one contract)
                    # is a recorded skip, never a silent one; anything else
                    # propagates — section 5.2's ONLY frozen disqualifier is
                    # the 30-entry floor, so unknown errors must abort loudly.
                    failed_cells.append(f'{cell.key()}: {exc}')
                    continue
                n = int(summary['num_credit_spreads_sold'])
                grid_trades.append(n)
                if n < min_trades:
                    continue
                sh = sharpe_unrounded(
                    excess_stream(eq, float(summary['capital']))
                )
                if sh > best_sharpe:  # strict >: frozen-lattice-order ties
                    best_sharpe = sh
                    winner = cell
                    winner_trades = n
            rec.update({
                'train_sharpe': None if winner is None else best_sharpe,
                'n_trades': None if winner is None else winner_trades,
                'min_grid_trades': min(grid_trades) if grid_trades else None,
                'n_below_30': sum(1 for n in grid_trades if n < min_trades),
                'failed_cells': failed_cells,
            })
        rec['winner'] = winner
        if winner is None:
            rec.update(_skipped_record(test_df))
        else:
            rec.update(_oos_record(
                store, test_df, winner, hedged=True, run_cell_fn=run_cell_fn
            ))
        records.append(rec)
    return records


def replay_records(
    records: list[dict[str, Any]],
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    *,
    hedged: bool,
    select_factory: Optional[Callable[[Cell], Callable[..., Any]]] = None,
    extra_params: Optional[dict[str, Any]] = None,
    train_years: int = TRAIN_YEARS,
    test_months: int = TEST_MONTHS,
    roll_months: int = ROLL_MONTHS,
    run_cell_fn: RunCellFn = run_cell,
) -> list[dict[str, Any]]:
    """Replay the per-window winners with a different hedge state (arm B) or
    selector (arm E). Selection is NEVER re-run (plan D7); windows re-derive
    from the identical arithmetic and are asserted to match the records."""
    windows = _window_frames(dates, prices, train_years, test_months, roll_months)
    if len(windows) != len(records):
        raise ValueError(
            f'window mismatch: {len(windows)} vs {len(records)} records'
        )
    out: list[dict[str, Any]] = []
    for rec, (_, test_df) in zip(records, windows):
        if rec['test_start'] != test_df['d'].iloc[0]:
            raise ValueError('replay window misalignment')
        winner = rec['winner']
        base = {'test_start': rec['test_start'], 'test_end': rec['test_end'],
                'winner': winner}
        if winner is None:
            base.update(_skipped_record(test_df))
        else:
            select = select_factory(winner) if select_factory else None
            base.update(_oos_record(
                store, test_df, winner, hedged=hedged,
                run_cell_fn=run_cell_fn, select=select,
                extra_params=extra_params,
            ))
        out.append(base)
    return out


def jitter_select_factory(
    rng: random.Random, k: int = JITTER_K
) -> Callable[[Cell], Callable[..., Any]]:
    """Arm E's entry-calendar jitter: the Gap F emission-keyed wait around
    each winner cell's own selector (dte21 guard included). One career = one
    rng threaded across windows in order; k=0 reproduces the replay exactly."""

    def factory(cell: Cell) -> Callable[..., Any]:
        base = _cell_select(cell)
        state = {'j': -1, 'waited': 0}

        def select(day: dict[str, Any], params: dict[str, Any]) -> Any:
            if state['j'] < 0:
                state['j'] = rng.randint(0, k)
                state['waited'] = 0
            if state['waited'] < state['j']:
                state['waited'] += 1
                return None
            picked = base(day, params)
            if picked is not None:
                state['j'] = -1  # emission ends the stretch (Gap F semantics)
            return picked

        return select

    return factory


def stitch_records(
    records: list[dict[str, Any]]
) -> tuple[np.ndarray, list[str]]:
    """Section 5.5: concatenated per-window excess arrays (seam charges
    already applied inside each window's last element) plus aligned dates."""
    arrays = [r['oos_excess'] for r in records]
    dates: list[str] = []
    for r in records:
        dates.extend(r['oos_dates'])
    if not arrays:
        return np.zeros(0), dates
    return np.concatenate(arrays), dates


def verdict_stats(stitched: np.ndarray) -> dict[str, float]:
    """The one-sided verdict block (section 7.2) on a stitched stream."""
    s = newey_west_summary(stitched)
    p = 0.5 * math.erfc(s.t_newey_west / math.sqrt(2.0))
    return {
        'n': float(s.n),
        't_naive': s.t_naive,
        't_newey_west': s.t_newey_west,
        'nw_lag': float(s.lag),
        'one_sided_p': p,
        'sharpe': sharpe_unrounded(stitched),
        'mean_daily_excess': float(np.mean(stitched)) if len(stitched) else 0.0,
    }


# --- section-7.3 companions --------------------------------------------------

def stationary_bootstrap(
    x: np.ndarray,
    block: int = BOOTSTRAP_BLOCK,
    n_boot: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Politis-Romano stationary bootstrap of the mean: geometric block
    lengths (expected `block`), wrap-around, add-one one-sided p =
    (1 + #{mean_i <= 0}) / (1 + B). Deterministic under the committed seed."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return {'p_boot': 1.0, 'mean': 0.0, 'n_boot': float(n_boot)}
    rng = np.random.default_rng(seed)
    p_restart = 1.0 / block
    pos = np.arange(n)
    count_le_zero = 0
    chunk = 1_000
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        restarts = rng.random((b, n)) < p_restart
        restarts[:, 0] = True
        starts = rng.integers(0, n, size=(b, n))
        last_restart = np.maximum.accumulate(
            np.where(restarts, pos, -1), axis=1
        )
        rows = np.arange(b)[:, None]
        start_used = starts[rows, last_restart]
        idx = (start_used + (pos - last_restart)) % n
        means = x[idx].mean(axis=1)
        count_le_zero += int(np.sum(means <= 0.0))
        done += b
    return {
        'p_boot': (1 + count_le_zero) / (1 + n_boot),
        'mean': float(np.mean(x)),
        'n_boot': float(n_boot),
    }


def loyo_nw(
    stitched: np.ndarray, dates: list[str]
) -> dict[str, float]:
    """Leave-one-year-out NW t on the stitched stream (section 7.3): drop
    each calendar year's observations, recompute. Years keyed 'YYYY'."""
    if len(stitched) != len(dates):
        raise ValueError('stitched/dates length mismatch')
    years = sorted({d[:4] for d in dates})
    out: dict[str, float] = {}
    year_arr = np.array([d[:4] for d in dates])
    for year in years:
        mask = year_arr != year
        if mask.sum() < 2:
            continue
        out[year] = newey_west_summary(stitched[mask]).t_newey_west
    return out
