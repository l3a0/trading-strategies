"""Exploration log — cheap kill-gate scouts on ideas that did not survive.

These are EXPLORATORY scouts, NOT registered experiments. Each spends the
sample and can only KILL an idea or JUSTIFY a registration — none is a
confirmatory verdict. They are pinned (test_explorations.py) and logged
(docs/explorations.md) so a dead end is settled once instead of re-derived
from scratch every session.

The discipline that keeps this honest: a scout result is labeled exploratory
on every surface. Pinning the number prevents re-work; it does NOT promote
the scout to a registered finding — that distinction is the whole point of
docs/prereg_trend_gate.md. Every number reproduces from the pinned naked
runs plus a fixed RNG seed and the documented data-hygiene rules
(CHAIN_CLEAN_START era clip; per-ticker tagging — a rip on one name must
not cool down another).

Entries:
- cooldown_scout — "after a deep-ITM/assignment rip, suspend selling for N
  days." KILLED: the per-cycle effect is wrong-signed (post-rip cycles lose
  LESS, not more — D_A > 0 at every horizon) and there is no return memory
  to set N to (a rip is weakly mean-reverting: forward returns sit BELOW
  baseline, daily-return lag-1 autocorrelation is negative). The third
  confirmation that conditioning call-selling entry on recent upward price
  action has the sign backwards on these names (cf. the trend gate,
  docs/trend_gate_results.md).
- iv_richness_scout — "sell only when implied vol is rich vs realized" (the
  volatility-risk-premium gate). KILLED: at the sold ~25-delta/30-day
  contract the ex-post VRP (entry IV minus the realized vol over the option's
  life) is ~0, and the entry-richness signal does not predict cycle P&L
  (Spearman ~0; the rich-vs-not split is mid-pack against its permutation
  null) — so there is no premium to gate on. Reads the vendor
  implied_volatility column directly (the engine's loader discards it as
  unreliable), with a fail-closed IV < 0.05 floor for the lattice/placeholder
  rows.

Usage:
    python explorations.py            # run + print both scouts
"""

from __future__ import annotations

from common.paths import data_path

import csv
import json
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from realchains.real_cc_backtest import (
    CHAIN_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    open_dailies,
    run_real_cc_overlay,
)

# The published-baseline real-chain config (calendar-day DTE), naked — no
# cap_delta / stop_loss_mult / delta_hedge. The scouts run on these exact
# runs, so their inputs are the pinned regression runs.
NAKED_PARAMS: dict[str, float] = {
    'call_delta': 0.25,
    'close_at_pct': 0.75,
    'dte': 30,
    'risk_free_rate': 0.045,
    'capital': 100_000,
}

SCOUT_TICKERS = ('MSFT', 'QQQ', 'SPY')
COOLDOWN_HORIZONS = (7, 14, 21, 30, 45, 60, 90, 120, 180)  # calendar days
FORWARD_HORIZONS = (21, 30, 45, 60, 90, 120)               # trading days
PERMUTATION_SEED = 20260613
PERMUTATION_DRAWS = 1000
TERMINAL_ACTIONS = ('close', 'close_itm', 'expiration')

# IV-richness scout knobs.
IV_FLOOR = 0.05      # vendor IVs below this are lattice/placeholder garbage;
                     # fail-closed — a cycle whose entry IV is missing or
                     # < floor can't be assessed for richness and is dropped.
RV_WINDOW = 30       # trailing realized-vol window (trading days)
VRP_FORWARD = 21     # ~30 calendar days = the sold option's life (trading days)


def _ord(date: str) -> int:
    """ISO date string -> proleptic-Gregorian ordinal (for fast day math)."""
    return datetime.strptime(date, '%Y-%m-%d').toordinal()


def load_naked_run(ticker: str) -> dict[str, Any]:
    """Naked baseline run on the CLEAN canonical chains.

    CHAIN_CLEAN_START is applied (the SPY canonical file still carries the
    2008-2010 placeholder-greeks era that every pinned SPY run excludes;
    MSFT/QQQ canonical files start past it, so the clip is a no-op there).
    Returns the price series and the reconstructed cycles.
    """
    canonical = data_path(f'{ticker.lower()}_option_dailies.csv')
    store = load_chain_store(canonical, start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    dates = [d for d, _ in pairs]
    prices = [p for _, p in pairs]
    _, trades, _ = run_real_cc_overlay(dates, prices, store, NAKED_PARAMS)
    return {'ticker': ticker, 'dates': dates, 'prices': prices,
            'cycles': reconstruct_cycles(trades)}


def reconstruct_cycles(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each sell with the next terminal record; flag the rip triggers.

    A RIP = a deep-ITM buyback (close_itm) or a loss-making assignment at
    expiration (expiration with pnl < 0) — the events the proposed cooldown
    reacts to.
    """
    cycles: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = None
    for t in trades:
        if t['action'] == 'sell':
            entry = t
        elif t['action'] in TERMINAL_ACTIONS:
            assert entry is not None, 'terminal record without a sell'
            cycles.append({
                'entry_date': entry['date'],
                'entry_contract': entry.get('contract'),
                'terminal_date': t['date'],
                'action': t['action'],
                'pnl': t['pnl'],
                'rip': t['action'] == 'close_itm'
                       or (t['action'] == 'expiration' and t['pnl'] < 0),
            })
            entry = None
    return cycles


def post_rip_mask(entry_ords: Sequence[int], ticker_ids: Sequence[str],
                  rip_ords_by_ticker: dict[str, list[int]],
                  horizon: int) -> np.ndarray:
    """Per-cycle post-rip flag at `horizon` calendar days, tagged PER TICKER:
    a cycle is post-rip iff a rip on its OWN ticker terminated strictly
    before its entry and within `horizon` days. `rip_ords_by_ticker` values
    are sorted ascending."""
    out = np.empty(len(entry_ords), dtype=bool)
    for i, e in enumerate(entry_ords):
        rips = rip_ords_by_ticker.get(ticker_ids[i], ())
        lo, hi = 0, len(rips)
        while lo < hi:
            mid = (lo + hi) // 2
            if rips[mid] < e:
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo > 0 and 0 < (e - rips[lo - 1]) <= horizon
    return out


def _d_a(pnls: np.ndarray, post_rip: np.ndarray) -> float | None:
    """D_A = mean(pnl | post-rip) − mean(pnl | other). None on an empty cell."""
    a, b = pnls[post_rip], pnls[~post_rip]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(a.mean() - b.mean())


def cooldown_scout(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The post-rip-cooldown kill-gate. Pools naked cycles across tickers and
    measures, at each horizon N, whether cycles entered in the N-day shadow
    of a same-ticker rip do WORSE (the hypothesis: D_A < 0). Adds a
    trigger-placement permutation null and a signal-side forward-return /
    autocorrelation memory check. EXPLORATORY — kills or justifies only."""
    pooled: list[dict[str, Any]] = []
    rip_ords: dict[str, list[int]] = {}
    term_ords: dict[str, list[int]] = {}
    for r in runs:
        t = r['ticker']
        rip_ords[t] = sorted(_ord(c['terminal_date']) for c in r['cycles'] if c['rip'])
        term_ords[t] = [_ord(c['terminal_date']) for c in r['cycles']]
        for c in r['cycles']:
            pooled.append({**c, 'ticker': t})
    pnls = np.array([c['pnl'] for c in pooled], dtype=float)
    entry_ords = [_ord(c['entry_date']) for c in pooled]
    ticker_ids = [c['ticker'] for c in pooled]
    n_rips = sum(len(v) for v in rip_ords.values())

    rng = np.random.default_rng(PERMUTATION_SEED)
    grid = []
    for N in COOLDOWN_HORIZONS:
        mask = post_rip_mask(entry_ords, ticker_ids, rip_ords, N)
        d_a = _d_a(pnls, mask)
        # Permutation null: redraw each ticker's rip dates from its OWN
        # terminals (same count, same per-ticker structure), recompute D_A.
        perm_le = 0
        for _ in range(PERMUTATION_DRAWS):
            fake = {}
            for t, rips in rip_ords.items():
                terms = term_ords[t]
                idx = rng.choice(len(terms), size=len(rips), replace=False)
                fake[t] = sorted(terms[j] for j in idx)
            pd = _d_a(pnls, post_rip_mask(entry_ords, ticker_ids, fake, N))
            if pd is not None and d_a is not None and pd <= d_a:
                perm_le += 1
        grid.append({
            'N_days': N,
            'n_post_rip': int(mask.sum()),
            'n_other': int((~mask).sum()),
            'mean_pnl_post_rip': round(float(pnls[mask].mean()), 2) if mask.any() else None,
            'mean_pnl_other': round(float(pnls[~mask].mean()), 2) if (~mask).any() else None,
            'D_A': round(d_a, 2) if d_a is not None else None,
            'perm_percentile': round(perm_le / PERMUTATION_DRAWS, 3),
            'net_pnl_delta_if_skipped': round(-float(pnls[mask].sum()), 2),
        })

    return {
        'tickers': [r['ticker'] for r in runs],
        'n_cycles': len(pooled),
        'n_rips': n_rips,
        'grid': grid,
        'memory': _forward_return_memory(runs),
    }


def _forward_return_memory(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Signal-side (price-only): after the actual rip dates, are forward
    returns ELEVATED (momentum → a cooldown could help) or not? Plus the
    pooled daily-return lag-1 autocorrelation. If forward returns are NOT
    elevated and acf is non-positive, no nonzero cooldown N is justified."""
    fwd_after: dict[int, list[float]] = {h: [] for h in FORWARD_HORIZONS}
    fwd_all: dict[int, list[float]] = {h: [] for h in FORWARD_HORIZONS}
    daily_returns: list[float] = []
    for r in runs:
        prices = np.array(r['prices'], dtype=float)
        idx = {d: i for i, d in enumerate(r['dates'])}
        rip_dates = [c['terminal_date'] for c in r['cycles'] if c['rip']]
        daily_returns.extend(np.diff(prices) / prices[:-1])
        for h in FORWARD_HORIZONS:
            for i in range(len(prices) - h):
                fwd_all[h].append(prices[i + h] / prices[i] - 1)
            for d in rip_dates:
                i = idx.get(d)
                if i is not None and i + h < len(prices):
                    fwd_after[h].append(prices[i + h] / prices[i] - 1)
    forward = []
    for h in FORWARD_HORIZONS:
        after = float(np.mean(fwd_after[h])) * 100 if fwd_after[h] else None
        base = float(np.mean(fwd_all[h])) * 100 if fwd_all[h] else None
        forward.append({
            'horizon_days': h,
            'after_rip_pct': round(after, 3) if after is not None else None,
            'baseline_pct': round(base, 3) if base is not None else None,
            'diff_pct': round(after - base, 3) if (after is not None and base is not None) else None,
        })
    dr = np.array(daily_returns, dtype=float)
    dr = dr - dr.mean()
    acf1 = float((dr[:-1] * dr[1:]).mean() / (dr * dr).mean())
    return {'forward': forward, 'daily_return_acf_lag1': round(acf1, 3)}


def load_entry_ivs(ticker: str, wanted: set[tuple[str, str]],
                   path: str | None = None) -> dict[tuple[str, str], float]:
    """One streaming pass over the dailies → vendor implied_volatility for the
    requested (date, contractID) entry rows. The engine's load_chain_store
    drops the IV column (it is unreliable in places), so the IV-richness scout
    reads it here instead of through the store. `path` overrides the
    ticker-derived filename (for tests)."""
    out: dict[tuple[str, str], float] = {}
    with open_dailies(path or data_path(f'{ticker.lower()}_option_dailies.csv')) as f:
        for r in csv.DictReader(f):
            key = (r['date'], r['contractID'])
            if key in wanted:
                try:
                    out[key] = float(r['implied_volatility'])
                except (TypeError, ValueError):
                    pass
    return out


def _ann_vol(log_returns: np.ndarray) -> float:
    return float(np.std(log_returns, ddof=1)) * float(np.sqrt(252))


def iv_richness_scout(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The volatility-risk-premium gate kill-gate. For each naked cycle, read
    the entry contract's vendor IV and ask two things: (1) is the option
    actually rich — what is the ex-post VRP (entry IV minus the realized vol
    over the option's life)? and (2) does the entry-richness signal (entry IV
    minus trailing realized vol, the thing you could gate on) predict cycle
    P&L? Cycles whose entry IV is missing or below IV_FLOOR are dropped
    (fail-closed). EXPLORATORY — kills or justifies only."""
    rows: list[dict[str, Any]] = []
    dropped = 0
    for r in runs:
        prices = np.array(r['prices'], dtype=float)
        logret = np.diff(np.log(prices))
        idx = {d: i for i, d in enumerate(r['dates'])}
        wanted = {(c['entry_date'], c['entry_contract']) for c in r['cycles']
                  if c.get('entry_contract')}
        ivs = load_entry_ivs(r['ticker'], wanted)
        for c in r['cycles']:
            i = idx.get(c['entry_date'])
            iv = ivs.get((c['entry_date'], c.get('entry_contract')))
            if i is None or i < RV_WINDOW or iv is None or iv < IV_FLOOR:
                dropped += 1
                continue
            trailing = _ann_vol(logret[i - RV_WINDOW:i])   # known at entry
            fwd = (_ann_vol(logret[i:i + VRP_FORWARD])
                   if i + VRP_FORWARD <= len(logret) else None)
            rows.append({
                'ticker': r['ticker'],
                'pnl': float(c['pnl']),
                'entry_iv': iv,
                'trailing_rv': trailing,
                'richness': iv - trailing,           # gating signal (trailing)
                'vrp': (iv - fwd) if fwd is not None else None,  # ex-post
            })

    pnls = np.array([x['pnl'] for x in rows], dtype=float)
    rich = np.array([x['richness'] > 0 for x in rows], dtype=bool)
    d_a = _d_a(pnls, rich)
    # permutation null: random same-size "rich" split (does the IV-defined
    # split beat a random split of the same size?)
    rng = np.random.default_rng(PERMUTATION_SEED)
    n_rich = int(rich.sum())
    perm_le = 0
    for _ in range(PERMUTATION_DRAWS):
        fake = np.zeros(len(rows), dtype=bool)
        fake[rng.choice(len(rows), size=n_rich, replace=False)] = True
        pd_ = _d_a(pnls, fake)
        if pd_ is not None and d_a is not None and pd_ <= d_a:
            perm_le += 1
    # Spearman = Pearson on ranks (avg-rank ties), computed without scipy.
    rich_rank = pd.Series([x['richness'] for x in rows]).rank().to_numpy()
    pnl_rank = pd.Series(pnls).rank().to_numpy()
    spearman = float(np.corrcoef(rich_rank, pnl_rank)[0, 1])
    vrps = np.array([x['vrp'] for x in rows if x['vrp'] is not None], dtype=float)
    trail = np.array([x['trailing_rv'] for x in rows], dtype=float)
    return {
        'tickers': [r['ticker'] for r in runs],
        'n_assessed': len(rows),
        'n_dropped_iv_guard': dropped,
        'vrp_median_pct': round(float(np.median(vrps)) * 100, 3),
        'vrp_mean_pct': round(float(np.mean(vrps)) * 100, 3),
        'frac_rich_entries': round(float(rich.mean()), 3),
        'spearman_richness_pnl': round(spearman, 3),
        'D_A_rich_vs_not': round(d_a, 2) if d_a is not None else None,
        'perm_percentile': round(perm_le / PERMUTATION_DRAWS, 3),
        # The vol-level confound behind the binary split: "rich" (IV >
        # trailing RV) entries cluster where trailing vol is LOW — calm
        # markets, where covered calls do better regardless of any premium.
        'mean_trailing_rv_rich': round(float(trail[rich].mean()), 4),
        'mean_trailing_rv_not': round(float(trail[~rich].mean()), 4),
    }


def main() -> None:
    print('Loading naked runs (3 chain stores; a few minutes cold) ...', flush=True)
    runs = [load_naked_run(t) for t in SCOUT_TICKERS]
    print(json.dumps({'cooldown_scout': cooldown_scout(runs),
                      'iv_richness_scout': iv_richness_scout(runs)},
                     indent=2, default=str))


if __name__ == '__main__':
    main()
