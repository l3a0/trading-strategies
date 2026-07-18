"""The Tharp/Basso random-entry replication engine (docs/tharp_random_entry_plan.md).

A shares-only trend simulator at the plan's frozen coordinates: coin-flip
direction per instrument (seeded stdlib RNG), a 3.0 x ATR(20) trailing stop
evaluated and filled on the daily close (the repo's EOD stop-market
convention), re-entry the next trading day, percent-risk sizing (1% of
current portfolio equity per position, initial risk per share = the entry
stop distance — Tharp's own R), nine instruments against one equity stream.

Conventions, all on the record in the plan:
- ATR(20) = the 20-day simple mean of Wilder true range; an instrument is
  eligible once it has 21 observations (warmup).
- Costs: `cost_bps` of traded share notional per fill; `borrow_annual` on
  short notional, accrued daily (/252). Dividends ignored (price-return
  series; splits are backed out by the data source).
- Trades carry Tharp units: r_multiple = pnl / initial_risk_dollars and
  mae_r = the worst close-marked adverse excursion in R.

Phase-2 helpers live here too: the drift twin (the career's average signed
notional held constant), the placebo-exit and no-stop ensembles (measured in
sizing-free per-trade R — the apples-to-apples unit), and the pooled R bag
the Gap C+B sizing replay consumes. EXPLORATORY machinery throughout; the
pins live in tests/test_tharp_random_entry.py, the narrative in
docs/explorations.md.
"""
from __future__ import annotations

import csv
import math
import random
from typing import Any, Optional

import numpy as np

from common.paths import data_path

TICKERS = ('SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'XLE', 'EEM', 'MSFT', 'NVDA')
ATR_PERIOD = 20
STOP_MULT = 3.0
RISK_FRACTION = 0.01
CAPITAL = 100_000.0
N_CAREERS = 100
CAREER_SEED_BASE = 20260719   # 20260717/-18 are taken (jitter, bootstrap)
COST_BPS = 0.5
BORROW_ANNUAL = 0.005
SPAN_START, SPAN_END = '2000-01-03', '2026-06-30'


def load_ohlc(ticker: str) -> dict[str, np.ndarray]:
    """One ticker's daily OHLC file as date list + float arrays."""
    dates: list[str] = []
    cols: dict[str, list[float]] = {'open': [], 'high': [], 'low': [], 'close': []}
    with open(data_path(f'{ticker.lower()}_daily_ohlc.csv'), newline='') as f:
        for row in csv.DictReader(f):
            dates.append(row['date'])
            for k in cols:
                cols[k].append(float(row[k]))
    return {'dates': dates, **{k: np.asarray(v) for k, v in cols.items()}}


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = ATR_PERIOD) -> np.ndarray:
    """ATR(period) as the simple mean of Wilder true range; NaN during warmup."""
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    out = np.full_like(close, np.nan)
    for i in range(period, len(tr)):
        out[i] = float(np.mean(tr[i - period + 1:i + 1]))
    return out


def build_market(
    tickers: tuple[str, ...] = TICKERS,
    start: str = SPAN_START,
    end: str = SPAN_END,
) -> dict[str, Any]:
    """The shared calendar (union of instrument dates, clipped) plus aligned
    per-instrument close/ATR arrays (NaN where an instrument has no bar).
    Instruments join the basket once their ATR is warm — the plan's rule."""
    raw = {t: load_ohlc(t) for t in tickers}
    all_dates = sorted({d for r in raw.values() for d in r['dates']
                        if start <= d <= end})
    ix = {d: i for i, d in enumerate(all_dates)}
    n = len(all_dates)
    closes = {t: np.full(n, np.nan) for t in tickers}
    atrs = {t: np.full(n, np.nan) for t in tickers}
    for t, r in raw.items():
        a = atr_series(r['high'], r['low'], r['close'])
        for j, d in enumerate(r['dates']):
            if d in ix:
                closes[t][ix[d]] = r['close'][j]
                atrs[t][ix[d]] = a[j]
    return {'dates': all_dates, 'closes': closes, 'atrs': atrs,
            'tickers': tickers}


def run_career(
    seed: int,
    market: dict[str, Any],
    *,
    capital: float = CAPITAL,
    risk_fraction: float = RISK_FRACTION,
    stop_mult: float = STOP_MULT,
    cost_bps: float = COST_BPS,
    borrow_annual: float = BORROW_ANNUAL,
    keep_positions: bool = False,
) -> dict[str, Any]:
    """One coin-flip career across the basket. Returns the summary, the
    Tharp-unit trade list, the daily equity array, and (optionally) the
    per-instrument daily signed-notional matrix the drift twin consumes."""
    rng = random.Random(seed)
    dates = market['dates']
    tickers = market['tickers']
    n = len(dates)
    cash = capital
    pos: dict[str, dict[str, Any]] = {}       # open positions by ticker
    trades: list[dict[str, Any]] = []
    equity = np.empty(n)
    notional = np.zeros((n, len(tickers))) if keep_positions else None
    cost_rate = cost_bps / 10_000.0
    daily_borrow = borrow_annual / 252.0

    for i in range(n):
        for k, t in enumerate(tickers):
            px = market['closes'][t][i]
            if math.isnan(px):
                continue
            atr = market['atrs'][t][i]
            p = pos.get(t)
            if p is not None:
                p['last_px'] = px              # carried mark for no-bar days
                # close-marked excursion in R (negative when underwater)
                adverse = (p['entry_px'] - px) if p['dir'] > 0 else (px - p['entry_px'])
                r_now = -adverse / p['risk_ps']
                if r_now < p['worst_r']:
                    p['worst_r'] = r_now
                hit = (px <= p['trail']) if p['dir'] > 0 else (px >= p['trail'])
                if hit:
                    fill_cost = px * p['shares'] * cost_rate
                    cash += p['dir'] * p['shares'] * px - fill_cost
                    pnl = (p['dir'] * (px - p['entry_px']) * p['shares']
                           - fill_cost - p['entry_cost'] - p['borrow_paid'])
                    risk_dollars = p['risk_ps'] * p['shares']
                    trades.append({
                        'ticker': t, 'direction': p['dir'],
                        'entry_date': p['entry_date'], 'exit_date': dates[i],
                        'entry_px': p['entry_px'], 'exit_px': px,
                        'shares': p['shares'], 'initial_risk': risk_dollars,
                        'pnl': pnl, 'r_multiple': pnl / risk_dollars,
                        'mae_r': min(0.0, p['worst_r']),
                        'hold_days': i - p['entry_ix'],
                    })
                    del pos[t]
                    # re-entry happens no earlier than the NEXT day's loop
                    # pass (the one-day-gap rule): this instrument's branch
                    # has already run today.
                else:
                    if p['dir'] > 0:
                        p['trail'] = max(p['trail'], px - stop_mult * atr) \
                            if not math.isnan(atr) else p['trail']
                    else:
                        p['trail'] = min(p['trail'], px + stop_mult * atr) \
                            if not math.isnan(atr) else p['trail']
                    if p['dir'] < 0:
                        b = px * p['shares'] * daily_borrow
                        cash -= b
                        p['borrow_paid'] += b
            elif not math.isnan(atr):          # flat + warm ATR: enter
                eq_now = cash + sum(
                    q['dir'] * q['shares'] * market['closes'][tk][i]
                    for tk, q in pos.items()
                    if not math.isnan(market['closes'][tk][i]))
                risk_ps = stop_mult * atr
                shares = int((risk_fraction * eq_now) / risk_ps)
                if shares >= 1:
                    direction = 1 if rng.random() < 0.5 else -1
                    fill_cost = px * shares * cost_rate
                    cash -= direction * shares * px + fill_cost
                    pos[t] = {
                        'dir': direction, 'shares': shares, 'entry_px': px,
                        'entry_ix': i, 'entry_date': dates[i],
                        'risk_ps': risk_ps, 'entry_cost': fill_cost,
                        'borrow_paid': 0.0, 'worst_r': 0.0, 'last_px': px,
                        'trail': px - direction * stop_mult * atr,
                    }
        mark = cash
        for tk, q in pos.items():
            mark += q['dir'] * q['shares'] * q['last_px']
            if notional is not None:
                notional[i, tickers.index(tk)] = q['dir'] * q['shares'] * q['last_px']
        equity[i] = mark

    rs = [tr['r_multiple'] for tr in trades]
    wins = sum(1 for r in rs if r > 0)
    out = {
        'seed': seed, 'final_equity': float(equity[-1]),
        'n_trades': len(trades),
        'expectancy_r': float(np.mean(rs)) if rs else 0.0,
        'win_rate': 100.0 * wins / len(rs) if rs else 0.0,
        'avg_win_r': float(np.mean([r for r in rs if r > 0])) if wins else 0.0,
        'avg_loss_r': float(np.mean([r for r in rs if r <= 0])) if wins < len(rs) else 0.0,
        'trades': trades, 'equity': equity,
    }
    if notional is not None:
        out['notional'] = notional
    return out


def run_ensemble(
    n_careers: int = N_CAREERS,
    market: Optional[dict[str, Any]] = None,
    seed_base: int = CAREER_SEED_BASE,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Phase 1: the seeded career ensemble (career i -> seed_base + i)."""
    if market is None:
        market = build_market()
    return [run_career(seed_base + i, market, **kwargs)
            for i in range(n_careers)]


# --- Phase 2 arms ------------------------------------------------------------

def drift_twin(career: dict[str, Any], market: dict[str, Any],
               *, borrow_annual: float = BORROW_ANNUAL) -> float:
    """The replication null: hold the career's average signed notional per
    instrument constant across the span (daily-rebalanced to that dollar
    figure; entry/exit costs negligible at this turnover and omitted; borrow
    charged on short legs). Returns the twin's terminal P&L in dollars."""
    notional = career['notional']
    tickers = market['tickers']
    avg = notional.mean(axis=0)               # constant dollar exposure per leg
    pnl = 0.0
    for k, t in enumerate(tickers):
        px = market['closes'][t]
        valid = ~np.isnan(px)
        rets = np.diff(px[valid]) / px[valid][:-1]
        pnl += float(np.sum(avg[k] * rets))
        if avg[k] < 0:
            pnl -= abs(avg[k]) * borrow_annual / 252.0 * (valid.sum() - 1)
    return pnl


def _r_stream_career(
    rng: random.Random,
    market: dict[str, Any],
    hold_sampler: Any,
    *,
    stop_mult: float = STOP_MULT,
    cost_bps: float = COST_BPS,
) -> float:
    """One sizing-free career in per-trade R units: coin-flip directions at
    the same eligibility rule, exits at sampled holding periods (no trail).
    The placebo/no-stop measurement unit (plan section 4)."""
    cost_rate = cost_bps / 10_000.0
    rs: list[float] = []
    for t in market['tickers']:
        px = market['closes'][t]
        atr = market['atrs'][t]
        i, n = 0, len(px)
        while i < n:
            if math.isnan(px[i]) or math.isnan(atr[i]):
                i += 1
                continue
            direction = 1 if rng.random() < 0.5 else -1
            h = hold_sampler(rng)
            j = min(i + max(1, h), n - 1)
            while j < n and math.isnan(px[j]):
                j += 1
            if j >= n:
                break
            gross = direction * (px[j] - px[i])
            costs_ps = (px[i] + px[j]) * cost_rate
            rs.append((gross - costs_ps) / (stop_mult * atr[i]))
            i = j + 1                          # the one-day-gap rule
    return float(np.mean(rs)) if rs else 0.0


def placebo_exit_ensemble(
    market: dict[str, Any],
    hold_multiset: list[int],
    n_careers: int = 1000,
    seed: int = CAREER_SEED_BASE + 500,
) -> list[float]:
    """The mechanism null: skill-free exits drawn from the real ensemble's
    pooled holding-period multiset. Returns per-career mean R."""
    holds = list(hold_multiset)

    def sampler(rng: random.Random) -> int:
        return holds[rng.randrange(len(holds))]

    return [_r_stream_career(random.Random(seed + i), market, sampler)
            for i in range(n_careers)]


def no_stop_ensemble(
    market: dict[str, Any],
    hold_days: int,
    n_careers: int = 100,
    seed: int = CAREER_SEED_BASE + 2000,
) -> list[float]:
    """Tharp's implicit control made explicit: fixed-H holds, coin flips."""
    return [_r_stream_career(random.Random(seed + i), market,
                             lambda rng: hold_days)
            for i in range(n_careers)]


def real_r_ensemble_units(ensemble: list[dict[str, Any]]) -> list[float]:
    """The real careers restated in the same sizing-free unit (mean per-trade
    R) so real-vs-placebo is apples to apples."""
    return [float(np.mean([t['r_multiple'] for t in c['trades']]))
            for c in ensemble if c['trades']]
