"""QQQ 1-DTE wheel — the frozen docs/qqq_wheel_1dte_plan.md build.

The owner-specified wheel: sell the ~0.20-delta put expiring the next
trading session from cash; on assignment, sell ~0.20-delta covered calls
struck at or above cost basis until called away. Entry gated on an up day
(the 3:55pm signal, strictly prior to the fill). Verdict: the daily
Newey-West t on the gap to the 100-shares-per-contract comparator.

Frozen conventions (the plan doc is the authority; this header is a map):

- Qualifying row: next-session contract, bid > 0, vendor delta within
  +/-0.05 of the 0.20 target. No qualifying row -> sell nothing tonight
  (the owner's no-stretch rule).
- Every option trade is atomic: sold at one close, settled at the next
  against that close (assigned/called exactly when strictly beyond the
  strike; exact ties expire worthless). No buybacks exist at 1-DTE on
  end-of-day data.
- Fills at the quote midpoint (owner-directed; 'bid' is the conservative
  floor variant). $0.65/contract fee on every sale; $0 for assignment,
  exercise, and stock trades.
- Idle cash earns the arm's cash rate (0% primary — the Schwab sweep
  reality — or 4.5% simple), accrued on calendar days, identically in
  both books.
- The comparator holds exactly 100 shares per contract notch from the
  arm's first close, residual capital in cash at the same rate.
- The gate signal is the last 1-minute bar at or before 15:55 ET (12:55
  on the static half-day calendar below), no older than 15 minutes,
  vs. yesterday's official close; fallback to close-over-close (counted)
  when the bar is missing or off-scale (>5% from the official close).

Epistemic class: exploratory (kill-or-justify). Nothing here enters the
idea ledger and no e-value is spent. The dataset-gated pins live in
tests/test_wheel_1dte.py::TestQqqWheel1dteExploration.

Run:  python -m realchains.wheel_1dte [--json]
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import date as _date
from typing import Any, Sequence

from common.paths import data_path
from common.position_sizing import kelly_fraction, simulate_sizing
from common.stats import newey_west_summary
from common.trade_ledger import TradeRecord, build_trade_ledger, ledger_statistics
from realchains.real_cc_backtest import load_chain_store, load_unadjusted_prices

WHEEL_SEED = 20260720
CAPITAL = 100_000.0
FEE_PER_CONTRACT = 0.65
RF_VARIANT = 0.045              # the money-market twin (house frozen rf)
DELTA_TARGET = 0.20
DELTA_TOL = 0.05                # the owner's no-stretch band: [0.15, 0.25] absolute
PRIMARY_START = '2023-01-01'
SECONDARY_START = '2016-06-06'  # first canonical QQQ call day
WHEEL_END = '2026-06-05'        # canonical store end — never appended to
SIGNAL_WINDOW_MINUTES = 15
SCALE_MISMATCH_TOL = 0.05

# sha256 of data/qqq_intraday_1min.csv as of the build (plan §3: recorded, and
# re-verifiable via wheel_intraday_sha()). The archive is a SIGNAL input only.
INTRADAY_SHA256 = '1bdef1953e37b9811d2387f681503fd0397ed39e1d07dc8b5690d131429eb1b5'

# The static 1:00pm-ET half-day calendar (plan §3), derived from the archive's
# own volume signature (regular-session volume collapses after the 13:00 close;
# detector: 14:30-15:30 volume < 2% of 09:30-13:00 volume, nearest non-half day
# at 5.6%) and matching the published NYSE pattern exactly: every
# day-after-Thanksgiving 2016-2025, Jul 3 when a weekday session (2017-2019,
# 2023-2025), Dec 24 when a weekday session (2018-2020, 2024-2025).
HALF_DAYS = frozenset({
    '2016-11-25', '2017-07-03', '2017-11-24', '2018-07-03', '2018-11-23',
    '2018-12-24', '2019-07-03', '2019-11-29', '2019-12-24', '2020-11-27',
    '2020-12-24', '2021-11-26', '2022-11-25', '2023-07-03', '2023-11-24',
    '2024-07-03', '2024-11-29', '2024-12-24', '2025-07-03', '2025-11-28',
    '2025-12-24',
})

# The frozen primary cell (plan §8): gate on, basis rule on, no stop,
# 1 contract, cash 0% — the owner's specification verbatim.
PRIMARY_CELL: dict[str, Any] = {
    'gate': '355', 'basis_rule': True, 'stop': None, 'contracts': 1,
    'cash_rate': 0.0,
}


def _calendar_days(d0: str, d1: str) -> int:
    y0, m0, dd0 = (int(x) for x in d0.split('-'))
    y1, m1, dd1 = (int(x) for x in d1.split('-'))
    return (_date(y1, m1, dd1) - _date(y0, m0, dd0)).days


def wheel_intraday_sha(path: str | None = None) -> str:
    """sha256 of the intraday archive — the §3 provenance record."""
    p = path or data_path('qqq_intraday_1min.csv')
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def load_gate_signals(
    dates: Sequence[str], closes: Sequence[float],
    intraday_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """The frozen §3 gate signal per trading day.

    ``dates``/``closes`` are the full official-close series (the unadjusted
    price file), which must extend at least one day before the first day a
    signal is wanted — the first date in the series gets no signal (no prior
    close) and the engine treats a missing signal as gate-blocked.

    Returns {date: {'up_355', 'up_cc', 'fallback', 'disagree'}} where
    ``up_355`` is the gate's frozen primary sign (falling back to the
    close-over-close sign, flagged, when the signal bar is missing from the
    15-minute window or sits >5% from the official close), ``up_cc`` is the
    naive variant, and ``disagree`` marks non-fallback days where the two
    signs differ (the size of the flattery the naive convention enjoys).
    """
    p = intraday_path or data_path('qqq_intraday_1min.csv')
    wanted = set(dates[1:])
    # last bar close within the 15-minute window up to the session's cutoff
    signal_px: dict[str, tuple[str, float]] = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            ts = row['timestamp']
            d = ts[:10]
            if d not in wanted:
                continue
            t = ts[11:16]
            cutoff = '12:55' if d in HALF_DAYS else '15:55'
            lo = '12:40' if d in HALF_DAYS else '15:40'
            if lo <= t <= cutoff:
                prev = signal_px.get(d)
                if prev is None or t >= prev[0]:
                    signal_px[d] = (t, float(row['close']))
    out: dict[str, dict[str, Any]] = {}
    for i in range(1, len(dates)):
        d, close, prior = dates[i], closes[i], closes[i - 1]
        up_cc = close > prior
        bar = signal_px.get(d)
        if bar is None or abs(bar[1] / close - 1.0) > SCALE_MISMATCH_TOL:
            out[d] = {'up_355': up_cc, 'up_cc': up_cc,
                      'fallback': True, 'disagree': False}
        else:
            up_355 = bar[1] > prior
            out[d] = {'up_355': up_355, 'up_cc': up_cc,
                      'fallback': False, 'disagree': up_355 != up_cc}
    return out


def load_wheel_market(start: str) -> tuple[list[str], list[float], dict[str, Any]]:
    """Merged calls+puts store (the house calls-only-canonical pattern) plus
    the matching official-close series, clipped to [start, WHEEL_END] with one
    extra leading price day so the first session has a prior close for the
    gate. The canonical files are never appended to; QQQ needs no era clip."""
    store = load_chain_store(
        data_path('qqq_option_dailies.csv'),
        [data_path('qqq_option_dailies_puts.csv')],
        start=start,
    )
    all_dates, all_closes = load_unadjusted_prices('QQQ', '2016-01-01', WHEEL_END)
    keep = [i for i, d in enumerate(all_dates) if start <= d <= WHEEL_END]
    lo = max(keep[0] - 1, 0)          # one leading day: the gate's first prior close
    hi = keep[-1]
    return all_dates[lo:hi + 1], all_closes[lo:hi + 1], store


def build_wheel_index(
    dates: Sequence[str], store: dict[str, Any],
) -> dict[str, dict[str, list[tuple[float, float, float, float]]]]:
    """Per-day qualifying rows for the NEXT session, both sides.

    {date: {'puts': [(delta, strike, bid, mid), ...], 'calls': [...]}} filtered
    to expiration == the next trading day in ``dates``, bid > 0, and vendor
    delta within the frozen +/-0.05 band. Empty lists mean the no-stretch rule
    sits out that side. Built once; every grid cell reads it.
    """
    index: dict[str, dict[str, Any]] = {}
    for i in range(len(dates) - 1):
        d, nxt = dates[i], dates[i + 1]
        day = store.get(d)
        entry: dict[str, Any] = {'puts': [], 'calls': [], 'eligible': False}
        if day is not None:
            for (dte, delta, bid, ask, mid, expiration, strike, cid) in day['candidates']:
                if expiration != nxt:
                    continue
                entry['eligible'] = True     # a next-session expiry is listed at all
                if bid <= 0:
                    continue
                if -DELTA_TARGET - DELTA_TOL <= delta <= -DELTA_TARGET + DELTA_TOL:
                    entry['puts'].append((delta, strike, bid, mid))
                elif DELTA_TARGET - DELTA_TOL <= delta <= DELTA_TARGET + DELTA_TOL:
                    entry['calls'].append((delta, strike, bid, mid))
        index[d] = entry
    return index


def _pick(rows: list[tuple[float, float, float, float]], target: float,
          min_strike: float | None = None) -> tuple[float, float, float, float] | None:
    """Nearest-|delta - target| qualifying row; a distance tie takes the lower
    strike. ``min_strike`` is the basis floor (call side, rule on)."""
    cand = rows if min_strike is None else [r for r in rows if r[1] >= min_strike]
    if not cand:
        return None
    return min(cand, key=lambda r: (abs(r[0] - target), r[1]))


def run_wheel(
    dates: Sequence[str], closes: Sequence[float],
    index: dict[str, dict[str, list[tuple[float, float, float, float]]]],
    signals: dict[str, dict[str, Any]],
    *,
    gate: str | None = '355',       # '355' | 'cc' | None (ablation)
    basis_rule: bool = True,
    basis_variant: str = 'strike',  # 'strike' | 'adjusted' (premium-ratcheted floor)
    stop: float | None = None,      # None | 0.05 | 0.10 (share-side stop below basis)
    contracts: int = 1,
    cash_rate: float = 0.0,
    fill: str = 'mid',              # 'mid' | 'bid'
    capital: float = CAPITAL,
) -> dict[str, Any]:
    """One wheel book vs. its per-contract comparator over ``dates[1:]``.

    ``dates[0]`` is the leading prior-close day: both books open at the close
    of ``dates[1]`` (the arm's first session), so day 1 carries no P&L and the
    gate has a prior close from day 0. Returns the daily series, the trade
    events, the per-overnight R records, the two-column rotation ledger, the
    diagnostics, and the frozen summary/decomposition — with the conservation
    identity asserted (plan §7), not trusted.
    """
    assert fill in ('mid', 'bid') and basis_variant in ('strike', 'adjusted')
    fill_i = 3 if fill == 'mid' else 2      # row = (delta, strike, bid, mid)

    start_i = 1
    close0 = closes[start_i]
    comp_shares = 100 * contracts
    comp_cost = comp_shares * close0
    assert comp_cost < capital, 'comparator shares must fit the capital'
    comp_cash = capital - comp_cost

    cash = capital
    shares = 0
    basis_strike = 0.0              # assignment strike (raw basis)
    basis_floor = 0.0               # the floor the call rule uses (variant-dependent)
    open_opt: dict[str, Any] | None = None
    rotation: dict[str, Any] | None = None
    last_put_entry: dict[str, Any] | None = None

    daily: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    records: list[TradeRecord] = []
    record_sides: list[str] = []
    rotations: list[dict[str, Any]] = []
    cash_period_pnl = 0.0           # worthless-put premiums + interest outside rotations
    diag = {
        'eligible_put_days': 0, 'eligible_call_days': 0,
        'gate_blocked': 0, 'no_signal_days': 0,
        'no_qualifying_put': 0, 'no_qualifying_call': 0,
        'basis_rule_binding_days': 0, 'clamped_days': 0, 'clamp_zero_days': 0,
        'weekend_entries': 0, 'weekend_settles': 0, 'weekend_wins': 0,
        'fallback_days': 0, 'disagree_days': 0, 'signal_days': 0,
    }
    totals = {
        'premium_collected': 0.0, 'fees': 0.0, 'interest': 0.0,
        'assignment_loss': 0.0, 'holding_share_pnl': 0.0, 'weekend_pnl': 0.0,
        'put_sales': 0, 'call_sales': 0, 'assignments': 0, 'call_aways': 0,
        'stop_outs': 0, 'expired_puts': 0, 'expired_calls': 0,
    }

    def equity_now(px: float) -> float:
        liab = open_opt['premium'] * 100 * open_opt['n'] if open_opt else 0.0
        return cash + shares * px - liab

    for i in range(start_i, len(dates)):
        d, px = dates[i], closes[i]

        # 1) interest (simple, calendar days, positive cash; both books alike)
        rf_credit = 0.0
        if i > start_i and cash_rate > 0.0:
            gap = _calendar_days(dates[i - 1], d)
            rf_credit = max(cash, 0.0) * cash_rate * gap / 365.0
            cash += rf_credit
            comp_cash += comp_cash * cash_rate * gap / 365.0
            totals['interest'] += rf_credit
            if rotation is not None:
                rotation['interest'] += rf_credit
            else:
                cash_period_pnl += rf_credit

        # 2) settlement of the expiring option against this close
        if open_opt is not None:
            assert open_opt['expiration'] == d, 'atomic 1-DTE: settles next session'
            n, k, prem = open_opt['n'], open_opt['strike'], open_opt['premium']
            side = open_opt['side']
            if side == 'put':
                assigned = px < k
                intrinsic = max(k - px, 0.0)
                pnl = (prem - intrinsic) * 100 * n - FEE_PER_CONTRACT * n
                if assigned:
                    cash -= k * 100 * n
                    shares += 100 * n
                    basis_strike = k
                    # adjusted variant: strike minus GROSS premiums collected
                    # (§8 verbatim — fees are their own §6 bucket)
                    basis_floor = k - prem if basis_variant == 'adjusted' else k
                    totals['assignments'] += 1
                    totals['assignment_loss'] += (k - px) * 100 * n
                    assert last_put_entry is not None
                    rotation = {
                        'entry_date': last_put_entry['date'], 'entry_i': last_put_entry['i'],
                        'assign_date': d, 'assign_close': px, 'strike': k,
                        'shares': 100 * n,
                        'premiums': prem * 100 * n - FEE_PER_CONTRACT * n,
                        'interest': 0.0, 'underwater_days': 0, 'holding_days': 0,
                    }
                else:
                    totals['expired_puts'] += 1
                    cash_period_pnl += pnl
            else:
                called = px > k
                intrinsic = max(px - k, 0.0)
                pnl = (prem - intrinsic) * 100 * n - FEE_PER_CONTRACT * n
                assert rotation is not None
                rotation['premiums'] += prem * 100 * n - FEE_PER_CONTRACT * n
                if called:
                    cash += k * 100 * n
                    shares -= 100 * n
                    totals['call_aways'] += 1
                    _close_rotation(rotations, rotation, d, i, k, totals, 'called_away')
                    rotation = None
                else:
                    totals['expired_calls'] += 1
            # §5 weekend diagnostic: the trades that carried >1 calendar night
            if _calendar_days(open_opt['entry_event']['date'], d) > 1:
                diag['weekend_settles'] += 1
                diag['weekend_wins'] += int(pnl >= 0)
                totals['weekend_pnl'] += pnl
            trades.append({'action': 'settle', 'date': d, 'pnl': round(pnl, 2)})
            recs = build_trade_ledger(
                [open_opt['entry_event'], trades[-1]],
                strategy='wheel', ticker='QQQ', shares=100 * n,
                risk_basis='premium_collected')
            records.extend(recs)
            record_sides.extend([side] * len(recs))
            open_opt = None

        # 3) share-side stop (after settlement, per the frozen §2 order)
        if shares > 0 and stop is not None and px <= basis_strike * (1.0 - stop):
            assert rotation is not None
            totals['stop_outs'] += 1
            cash += shares * px
            _close_rotation(rotations, rotation, d, i, px, totals, 'stopped')
            rotation = None
            shares = 0

        # rotation day counters (holding state after settlement/stop)
        if rotation is not None and shares > 0:
            rotation['holding_days'] += 1
            if px < basis_strike:
                rotation['underwater_days'] += 1

        # 4) entry for tonight (never on the final day; index has no entry there)
        day_idx = index.get(d)
        if day_idx is not None:
            if shares == 0 and open_opt is None:
                sig = signals.get(d)
                if gate is None:
                    allowed = True
                elif sig is None:
                    allowed = False
                    diag['no_signal_days'] += 1
                else:
                    allowed = sig['up_355'] if gate == '355' else sig['up_cc']
                if allowed:
                    if day_idx['puts']:
                        diag['eligible_put_days'] += 1
                        row = _pick(day_idx['puts'], -DELTA_TARGET)
                        assert row is not None
                        delta, k, bid, _mid = row
                        prem = row[fill_i]
                        n = contracts
                        while n > 0 and k * 100 * n > cash - FEE_PER_CONTRACT * n:
                            n -= 1
                        if n == 0:
                            diag['clamp_zero_days'] += 1
                        else:
                            if n < contracts:
                                diag['clamped_days'] += 1
                            entry = {'action': 'sell', 'date': d, 'side': 'put',
                                     'strike': k, 'delta': delta, 'premium': prem, 'n': n}
                            trades.append(entry)
                            cash += prem * 100 * n - FEE_PER_CONTRACT * n
                            totals['premium_collected'] += prem * 100 * n
                            totals['fees'] += FEE_PER_CONTRACT * n
                            totals['put_sales'] += 1
                            if _calendar_days(d, dates[i + 1]) > 1:
                                diag['weekend_entries'] += 1
                            open_opt = {'side': 'put', 'strike': k, 'premium': prem,
                                        'n': n, 'expiration': dates[i + 1],
                                        'entry_event': entry}
                            last_put_entry = {'date': d, 'i': i}
                    else:
                        diag['no_qualifying_put'] += 1
                elif sig is not None:
                    diag['gate_blocked'] += 1
            elif shares > 0 and open_opt is None:
                if day_idx['calls']:
                    diag['eligible_call_days'] += 1
                    floor = basis_floor if basis_rule else None
                    row = _pick(day_idx['calls'], DELTA_TARGET, min_strike=floor)
                    unconstrained = _pick(day_idx['calls'], DELTA_TARGET)
                    if basis_rule and unconstrained is not None and (
                            row is None or row[1] != unconstrained[1]):
                        diag['basis_rule_binding_days'] += 1
                    if row is not None:
                        delta, k, bid, _mid = row
                        prem = row[fill_i]
                        n = shares // 100
                        entry = {'action': 'sell', 'date': d, 'side': 'call',
                                 'strike': k, 'delta': delta, 'premium': prem, 'n': n}
                        trades.append(entry)
                        cash += prem * 100 * n - FEE_PER_CONTRACT * n
                        totals['premium_collected'] += prem * 100 * n
                        totals['fees'] += FEE_PER_CONTRACT * n
                        totals['call_sales'] += 1
                        if _calendar_days(d, dates[i + 1]) > 1:
                            diag['weekend_entries'] += 1
                        if basis_variant == 'adjusted':
                            basis_floor -= prem
                        open_opt = {'side': 'call', 'strike': k, 'premium': prem,
                                    'n': n, 'expiration': dates[i + 1],
                                    'entry_event': entry}
                else:
                    diag['no_qualifying_call'] += 1

        # gate diagnostics over all signal days in the window
        sig = signals.get(d)
        if sig is not None:
            diag['signal_days'] += 1
            diag['fallback_days'] += int(sig['fallback'])
            diag['disagree_days'] += int(sig['disagree'])

        # 5) mark both books at this close
        eq = equity_now(px)
        comp_eq = comp_shares * px + comp_cash
        daily.append({'date': d, 'price': px, 'equity': eq,
                      'comparator': comp_eq, 'rf_credit': rf_credit})

    # window end: an open rotation is reported open (marked, never force-sold)
    open_rotation = None
    if rotation is not None:
        px = closes[-1]
        exact = (rotation['premiums'] + rotation['interest']
                 + (px - rotation['strike']) * rotation['shares'])
        open_rotation = dict(rotation)
        open_rotation['mark_close'] = px
        open_rotation['unrealized_share_pnl'] = round(
            (px - rotation['strike']) * rotation['shares'], 2)
        open_rotation['raw_pnl_marked'] = round(exact, 2)
        open_rotation['raw_pnl_exact'] = exact
        totals['holding_share_pnl'] += (px - rotation['assign_close']) * rotation['shares']

    # Gap column for closed rotations (needs the finished daily series).
    # The window is (entry close, rotation close], based at the ENTRY close —
    # the position starts at that close, so the pre-entry session (when the
    # wheel is in cash and the comparator moves) is deliberately outside the
    # window; charging it would bias the column against gated cells, whose
    # entry days are conditioned up days. The entry-night sale fee (the only
    # rotation economics landing at the entry close itself) is folded back in
    # by lifting the base by the fee, so the column carries every rotation
    # cost. This also partitions same-close transitions cleanly: a rotation
    # closing at close c and a new one entered at c share the boundary, never
    # a session.
    gap = [row['equity'] - row['comparator'] for row in daily]
    for rot in rotations:
        base_i = rot['entry_i'] - start_i         # daily[] index of the entry close
        close_i = rot['close_i'] - start_i
        base = gap[base_i] + FEE_PER_CONTRACT * (rot['shares'] // 100)
        rot['gap_pnl'] = round(gap[close_i] - base, 2)

    # conservation (plan §7): rotations + cash periods == the book's total
    # P&L. Summed from the UNROUNDED per-rotation components (the
    # attribute_cycles lesson: summing 2dp-rounded rows accumulates ~sqrt(n)
    # cents of noise and can trip the assert on an honest book).
    final_eq = daily[-1]['equity']
    open_adj = 0.0
    if open_rotation is not None:
        open_adj = open_rotation['raw_pnl_exact']
    recon = sum(r['raw_pnl_exact'] for r in rotations) + cash_period_pnl + open_adj
    assert abs(recon - (final_eq - capital)) < 0.01, (
        f'conservation broke: rotations+cash {recon:.2f} vs book {final_eq - capital:.2f}')

    gap_diffs = [(gap[j] - gap[j - 1]) / capital for j in range(1, len(gap))]
    nw = newey_west_summary(gap_diffs)

    return {
        'daily': daily, 'trades': trades, 'records': records,
        'record_sides': record_sides, 'rotations': rotations,
        'open_rotation': open_rotation, 'diag': diag, 'totals': totals,
        'summary': {
            'final_equity': round(final_eq, 2),
            'comparator_final': round(daily[-1]['comparator'], 2),
            'gap_final': round(final_eq - daily[-1]['comparator'], 2),
            'daily_nw_t': round(nw.t_newey_west, 2),
            'daily_naive_t': round(nw.t_naive, 2),
            'nw_lag': nw.lag,
            'n_days': nw.n,
        },
    }


def _close_rotation(
    rotations: list[dict[str, Any]], rotation: dict[str, Any],
    d: str, i: int, exit_px: float, totals: dict[str, Any], reason: str,
) -> None:
    """Finish a rotation row: realized share P&L at the exit price (the call
    strike when called away, the close when stopped), raw column summed."""
    share_pnl = (exit_px - rotation['strike']) * rotation['shares']
    totals['holding_share_pnl'] += (exit_px - rotation['assign_close']) * rotation['shares']
    exact = rotation['premiums'] + rotation['interest'] + share_pnl
    rotation.update({
        'close_date': d, 'close_i': i, 'exit_price': exit_px, 'exit_reason': reason,
        'share_pnl': round(share_pnl, 2),
        'raw_pnl': round(exact, 2),
        'raw_pnl_exact': exact,     # unrounded: the conservation identity's term
    })
    rotations.append(rotation)


def rotation_summary(rotations: list[dict[str, Any]]) -> dict[str, Any]:
    """The §7 rotation-ledger summary: raw/gap expectancies, win rates, the
    rescue share, and the days-underwater distribution."""
    n = len(rotations)
    if n == 0:
        return {'n': 0}
    raw = [r['raw_pnl'] for r in rotations]
    gaps = [r['gap_pnl'] for r in rotations]
    under = sorted(r['underwater_days'] for r in rotations)
    rescued = sum(1 for r in rotations if r['raw_pnl'] > 0)  # all assigned start underwater
    return {
        'n': n,
        'raw_mean': round(sum(raw) / n, 2),
        'raw_win_rate': round(sum(1 for x in raw if x > 0) / n * 100, 1),
        'gap_mean': round(sum(gaps) / n, 2),
        'gap_win_rate': round(sum(1 for x in gaps if x > 0) / n * 100, 1),
        'rescue_share': round(rescued / n * 100, 1),
        'underwater_days_median': under[n // 2],
        'underwater_days_max': under[-1],
    }


def overnight_summary(records: list[TradeRecord], sides: list[str]) -> dict[str, Any]:
    """The §7 per-overnight ledger, split by side, plus the pooled stats.

    The per-side expectancy and R standard deviation are the Tharp trade
    profile: the numbers a live execution is monitored against."""
    def side_stats(side: str) -> dict[str, Any]:
        rs = [r.r_multiple for r, s in zip(records, sides) if s == side]
        if not rs:
            return {'n': 0}
        n = len(rs)
        mean = sum(rs) / n
        var = sum((x - mean) ** 2 for x in rs) / (n - 1) if n > 1 else 0.0
        return {'n': n, 'win_rate': round(sum(1 for x in rs if x >= 0) / n * 100, 1),
                'expectancy_r': round(mean, 4), 'r_std': round(math.sqrt(var), 4),
                'worst_r': round(min(rs), 2)}
    pooled = ledger_statistics(records) if records else {'n': 0}
    return {'pooled': pooled, 'puts': side_stats('put'), 'calls': side_stats('call')}


def sizing_battery(rotations: list[dict[str, Any]], capital: float = CAPITAL,
                   scale: float = 1.0) -> dict[str, Any]:
    """The §9 marble-bag on the PRIMARY cell's per-rotation dollar stream:
    r = raw P&L (times the notch ``scale``) as a fraction of capital,
    fraction=1.0 (each resampled career replays that sizing), n_trades = the
    book's rotation count. Both notches draw from the primary stream — the
    frozen §9 wording — so notch 2 is the same rotations at twice the dollars
    (the 2-contract cell's own clamp-affected stream stays visible in its
    grid row). Kelly reports 'unbounded' when the bag has no losing rotation
    — the §7 by-construction effect made measurable."""
    if not rotations:
        return {'n': 0}
    bag = [r['raw_pnl'] * scale / capital for r in rotations]
    sim = simulate_sizing(bag, fraction=1.0, n_trades=len(bag), seed=WHEEL_SEED)
    try:
        kelly = kelly_fraction(bag)
    except ValueError:
        kelly = None                # no losing rotation in the bag: no absorption boundary
    return {'n': len(bag), 'sim': sim,
            'kelly': kelly if kelly is not None else 'unbounded'}


def decomposition_companion(
    run: dict[str, Any], dates: Sequence[str], closes: Sequence[float],
    capital: float = CAPITAL,
) -> dict[str, Any]:
    """The §7 companion: the primary cell's realized leg sequence replayed
    exactly — same entries, strikes, fills — with a static overnight hedge
    bolted on. At each option entry the position's net delta (held stock plus
    the short option, at the entry row's vendor delta) is offset in shares at
    the same close and unwound at settlement; nights holding stock with no
    option sold carry no entry and hence no hedge. Unhedged-minus-hedged is
    the direction bill in dollars; the hedged book's own daily NW t answers
    "was there premium at all"."""
    px = {d: p for d, p in zip(dates, closes)}
    hedge_by_day: dict[str, float] = {}
    entry: dict[str, Any] | None = None
    for ev in run['trades']:
        if ev.get('action') == 'sell':
            entry = ev
        elif ev.get('action') == 'settle' and entry is not None:
            c0, c1 = px[entry['date']], px[ev['date']]
            n = entry['n']
            if entry['side'] == 'put':
                pos_delta = -entry['delta'] * 100 * n            # short put: long exposure
            else:
                pos_delta = (1.0 - entry['delta']) * 100 * n     # stock + short call
            hedge_by_day[ev['date']] = (hedge_by_day.get(ev['date'], 0.0)
                                        - pos_delta * (c1 - c0))
            entry = None
    daily = run['daily']
    u_pnl = [daily[0]['equity'] - capital]
    u_pnl += [daily[j]['equity'] - daily[j - 1]['equity'] for j in range(1, len(daily))]
    h_pnl = [u + hedge_by_day.get(row['date'], 0.0) for u, row in zip(u_pnl, daily)]
    nw_h = newey_west_summary([x / capital for x in h_pnl])
    direction_bill = -sum(hedge_by_day.values())
    return {
        'unhedged_total': round(sum(u_pnl), 2),
        'hedged_total': round(sum(h_pnl), 2),
        'direction_bill': round(direction_bill, 2),
        'hedged_daily_nw_t': round(nw_h.t_newey_west, 2),
        'hedged_nw_lag': nw_h.lag,
    }


GRID_GATES = ('355', None)
GRID_BASIS = (True, False)
GRID_STOPS = (None, 0.05, 0.10)
GRID_CONTRACTS = (1, 2)
GRID_RATES = (0.0, RF_VARIANT)


def cell_key(gate: str | None, basis: bool, stop: float | None,
             n: int, rate: float) -> str:
    return (f"gate={'on' if gate else 'off'}|basis={'on' if basis else 'off'}"
            f"|stop={stop if stop else 'none'}|n={n}|rate={rate}")


def summarize_cell(run: dict[str, Any]) -> dict[str, Any]:
    """The per-cell report row: the senior verdict plus the junior layers."""
    # The ledger convention made visible: how many of the call side's
    # per-trade losses land on called-away nights (the rotation's win).
    away_nights = {r['close_date'] for r in run['rotations']
                   if r['exit_reason'] == 'called_away'}
    call_losses = [r for r, s in zip(run['records'], run['record_sides'])
                   if s == 'call' and r.r_multiple < 0]
    return {
        **run['summary'],
        'call_losses': len(call_losses),
        'call_losses_on_away_nights': sum(
            1 for r in call_losses if r.close_date in away_nights),
        'rotations': rotation_summary(run['rotations']),
        'overnight': overnight_summary(run['records'], run['record_sides']),
        'totals': {k: (round(v, 2) if isinstance(v, float) else v)
                   for k, v in run['totals'].items()},
        'diag': run['diag'],
        'open_rotation': (
            {k: run['open_rotation'][k] for k in
             ('entry_date', 'assign_date', 'strike', 'shares', 'underwater_days',
              'holding_days', 'raw_pnl_marked')}
            if run['open_rotation'] else None),
    }


def run_experiment(arm: str = 'primary') -> dict[str, Any]:
    """The frozen §8 batch: the 48-cell grid plus the named variants on the
    primary arm, or the two secondary-arm runs (§5)."""
    start = PRIMARY_START if arm == 'primary' else SECONDARY_START
    dates, closes, store = load_wheel_market(start)
    index = build_wheel_index(dates, store)
    del store
    signals = load_gate_signals(dates, closes)

    calendar: dict[str, dict[str, int]] = {}
    for i, d in enumerate(dates[1:-1], start=1):
        y = d[:4]
        ent = calendar.setdefault(
            y, {'eligible': 0, 'put_ok': 0, 'call_ok': 0, 'both_ok': 0})
        ent['eligible'] += int(index[d]['eligible'])
        ent['put_ok'] += int(bool(index[d]['puts']))
        ent['call_ok'] += int(bool(index[d]['calls']))
        ent['both_ok'] += int(bool(index[d]['puts']) and bool(index[d]['calls']))

    out: dict[str, Any] = {'arm': arm, 'calendar': calendar}
    if arm == 'secondary':
        for label, g in (('primary_config', '355'), ('gate_off', None)):
            run = run_wheel(dates, closes, index, signals, **{**PRIMARY_CELL, 'gate': g})
            out[label] = summarize_cell(run)
        return out

    cells: dict[str, dict[str, Any]] = {}
    primary_run: dict[str, Any] | None = None
    for g in GRID_GATES:
        for b in GRID_BASIS:
            for s in GRID_STOPS:
                for n in GRID_CONTRACTS:
                    for rate in GRID_RATES:
                        run = run_wheel(dates, closes, index, signals,
                                        gate=g, basis_rule=b, stop=s,
                                        contracts=n, cash_rate=rate)
                        cells[cell_key(g, b, s, n, rate)] = summarize_cell(run)
                        if (g, b, s, n, rate) == ('355', True, None, 1, 0.0):
                            primary_run = run
    assert primary_run is not None
    out['cells'] = cells
    out['sizing_n1'] = sizing_battery(primary_run['rotations'], scale=1.0)
    out['sizing_n2'] = sizing_battery(primary_run['rotations'], scale=2.0)
    out['decomposition'] = decomposition_companion(primary_run, dates, closes)

    for label, overrides in (
        ('variant_cc_gate', {'gate': 'cc'}),
        ('variant_bid_fill', {'fill': 'bid'}),
        ('variant_adjusted_basis', {'basis_variant': 'adjusted'}),
    ):
        run = run_wheel(dates, closes, index, signals, **{**PRIMARY_CELL, **overrides})
        out[label] = summarize_cell(run)
    return out


def _print_report(out: dict[str, Any]) -> None:
    print(f"== arm: {out['arm']} ==")
    for y, ent in sorted(out['calendar'].items()):
        print(f"  {y}: eligible {ent['eligible']}  put_ok {ent['put_ok']}  "
              f"call_ok {ent['call_ok']}  both_ok {ent['both_ok']}")
    if out['arm'] == 'secondary':
        for label in ('primary_config', 'gate_off'):
            c = out[label]
            print(f"{label:>24}: NW t {c['daily_nw_t']:+.2f}  gap ${c['gap_final']:,.0f}  "
                  f"final ${c['final_equity']:,.0f} vs comp ${c['comparator_final']:,.0f}")
        return
    print(f"{'cell':<44}{'NWt':>7}{'gap$':>11}{'rot':>5}{'rawWR':>7}{'gapWR':>7}")
    for key, c in out['cells'].items():
        r = c['rotations']
        print(f"{key:<44}{c['daily_nw_t']:>7.2f}{c['gap_final']:>11,.0f}"
              f"{r.get('n', 0):>5}{r.get('raw_win_rate', 0):>7.1f}{r.get('gap_win_rate', 0):>7.1f}")
    for label in ('variant_cc_gate', 'variant_bid_fill', 'variant_adjusted_basis'):
        c = out[label]
        print(f"{label:>24}: NW t {c['daily_nw_t']:+.2f}  gap ${c['gap_final']:,.0f}")
    for label in ('sizing_n1', 'sizing_n2'):
        s = out[label]
        if s.get('n'):
            sim = s['sim']
            print(f"{label}: n={s['n']} kelly={s['kelly']} "
                  f"p_ruin={sim['p_ruin']} p_25dd={sim['p_ruin_25dd']} "
                  f"median_terminal={sim['terminal']['median']}")
    dc = out['decomposition']
    print(f"decomposition: unhedged ${dc['unhedged_total']:,.0f}  "
          f"hedged ${dc['hedged_total']:,.0f}  direction bill ${dc['direction_bill']:,.0f}  "
          f"hedged NW t {dc['hedged_daily_nw_t']:+.2f}")


if __name__ == '__main__':
    results = {'primary': run_experiment('primary'),
               'secondary': run_experiment('secondary')}
    if '--json' in sys.argv:
        print(json.dumps(results, default=str))
    else:
        _print_report(results['primary'])
        _print_report(results['secondary'])
