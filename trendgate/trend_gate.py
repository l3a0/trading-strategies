"""Analysis machinery for the registered trend-gate experiment.

Implements the binding sections of docs/prereg_trend_gate.md — §2 (signal),
§3.1 (spans), §4 (arms), §5 (Stage 1 kill-gate), §6 (Stage 2 verdict) — as
one module, committed before any Stage 1 number exists (§10 Ordering). Every
function cites the clause it implements; where the registration leaves an
implementation detail open, the choice made here becomes the detail of
record at this file's commit.

Ordering discipline (§10): Stage 1 may not run before the registration merge
commit (it exists: PR #12) and this file's commit. The CLI refuses to run
stage1/stage2 from a dirty working tree unless --allow-dirty is passed, so
results carry an honest code provenance by default.

Usage:
    python trend_gate.py characterize          # §3.1/§3.3 signal-side only (no outcomes)
    python trend_gate.py stage1                # §5 Tests A+B, gate rule, §9(a) MDE table
    python trend_gate.py placebo-mde           # §9(b): first 100 Family R engine re-runs
    python trend_gate.py stage2                # §6 Family R/S, arms, verdict, secondaries

stage2 work is checkpointed to trend_gate_runs/ (JSONL keyed by sequence
index) and resumes; sequences come from the single §5.1 stream, so a resumed
run consumes identical inputs.
"""

from __future__ import annotations
from common.paths import data_path

import json
import math
import os
import subprocess
import sys
from typing import Any, Iterator, Sequence

import numpy as np

from engine.cc_backtest import calc_rolling_volatility, classify_regime
from realchains.real_cc_backtest import (
    REGISTERED_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
)

# ---- registered constants ----

TICKERS: tuple[str, ...] = ('MSFT', 'SPY', 'QQQ')

# §3.1: canonical + backfill dailies per ticker (merged via load_chain_store;
# REGISTERED_CLEAN_START applies its clip where defined).
CHAIN_FILES: dict[str, tuple[str, tuple[str, ...]]] = {
    'MSFT': (data_path('msft_option_dailies.csv'), (data_path('msft_option_dailies_2008_2016.csv'),)),
    'SPY': (data_path('spy_option_dailies.csv'), ()),
    'QQQ': (data_path('qqq_option_dailies.csv'), (data_path('qqq_option_dailies_2011_2016.csv'),)),
}

# §3.2: the published baseline configuration, fixed for every arm.
ENGINE_PARAMS: dict[str, float] = {
    'call_delta': 0.25,
    'close_at_pct': 0.75,
    'dte': 30,
    'risk_free_rate': 0.045,
    'capital': 100_000,
}

F_STAR = 0.626453          # §3.3: pooled suspension fraction, pinned at registration
ACCEPT_LO = 0.595130       # §5.1: ±5% relative band around F_STAR, as registered
ACCEPT_HI = 0.657776
SEQUENCE_SEED = 20260611   # §5.1: the single placebo-sequence stream
FAMILY_S_SEED = 42         # §6.2: circular-shift offsets
STAGE1_SEQUENCES = 10_000  # §5.1
FAMILY_R_SEQUENCES = 1_000  # §6.2
FAMILY_S_DRAWS = 500       # §6.2
RAW_DRAW_GUARD = 1_000_000  # §5.1: amendment trigger on acceptance < 1%
REPLACEMENT_AMENDMENT_FRACTION = 0.02  # §5.1 / §6.2

CHECKPOINT_DIR = data_path('trend_gate_runs')

# §6.4 stress windows (record-arm minus baseline-arm equity change).
STRESS_WINDOWS: tuple[tuple[str, str], ...] = (
    ('2020-03-23', '2020-08-31'),
    ('2025-04-01', '2025-06-30'),
)

# First date of each committed unadjusted price CSV at registration. §2.1
# computes the signal on the FULL file; the files are pinned by the
# registration commit and a data refresh is a §11 amendment. Guarded at
# load: if a CSV were deleted, load_unadjusted_prices' lazy yfinance branch
# would regenerate it starting at the CHAIN-clipped date, silently moving
# the SMA warm-up into the span — fail loudly here instead.
PRICE_FILE_STARTS: dict[str, str] = {
    'MSFT': '2008-01-02', 'SPY': '2008-01-02', 'QQQ': '2011-03-23',
}


# ---- §2.1: the signal of record ----

def shifted_states(closes: Sequence[float]) -> list[str]:
    """The no-peek regime series: classify_regime(...).shift(1) (§2.1).

    Day d's state uses closes through d-1 only; day 0 has no prior close and
    is 'unknown'. Computed on the FULL unadjusted close series (the same one
    the engine trades against), then restricted to the §3.1 span by callers.
    """
    states = classify_regime(list(closes), window=200, threshold=0.05)
    return ['unknown', *states.iloc[:-1].tolist()]


def is_suspended(state: str) -> bool:
    """§2.1: suspended iff the shifted state is 'bull'; bear/sideways trade."""
    return state == 'bull'


# ---- §3.1: market loading and analysis spans ----

def load_market(ticker: str) -> dict[str, Any]:
    """Load the full price series, chain store, and §3.1 analysis span.

    Span = price-file dates within [max(first clean chain day, first day
    with a non-'unknown' shifted state), min(last chain day, last price
    day)] — the intersection of the clean chain span and the signal-warm
    span, ending at the engine's existing data-clipped end (§3.1).
    """
    canonical, extras = CHAIN_FILES[ticker]
    store = load_chain_store(canonical, extras, start=REGISTERED_CLEAN_START.get(ticker))
    chain_days = sorted(store)
    dates, closes = load_unadjusted_prices(ticker, chain_days[0], '2026-06-06')
    assert dates[0] == PRICE_FILE_STARTS[ticker], (
        f'{ticker} price file starts {dates[0]}, registered '
        f'{PRICE_FILE_STARTS[ticker]} — a regenerated/clipped file changes '
        'the §2.1 signal of record; restore the committed CSV or amend (§11)')
    states = shifted_states(closes)
    warm = next(d for d, s in zip(dates, states) if s != 'unknown')
    lo = max(chain_days[0], warm)
    hi = min(chain_days[-1], dates[-1])
    span_idx = [i for i, d in enumerate(dates) if lo <= d <= hi]
    return {
        'ticker': ticker,
        'store': store,
        'full_dates': dates,
        'full_closes': closes,
        'full_states': states,
        'span_dates': [dates[i] for i in span_idx],
        'span_prices': [closes[i] for i in span_idx],
        'span_states': [states[i] for i in span_idx],
        'span_offset': span_idx[0],  # index of span start in the full series
    }


def characterize(markets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """§3.3 signal-side characterization (treatment assignment only),
    including the registered table's sideways/bear columns and the pairwise
    daily bull-state agreements (computed on common dates)."""
    per: dict[str, Any] = {}
    susp_total = span_total = 0
    for t, m in markets.items():
        flags = [is_suspended(s) for s in m['span_states']]
        episodes = [len(list(g)) for v, g in _runs(flags) if v]
        n = len(flags)
        per[t] = {
            'span': (m['span_dates'][0], m['span_dates'][-1]),
            'days': n,
            'suspended': sum(flags),
            'bull_fraction': sum(flags) / n,
            'sideways_fraction': sum(
                s == 'sideways' for s in m['span_states']) / n,
            'bear_fraction': sum(s == 'bear' for s in m['span_states']) / n,
            'tradeable_fraction': 1 - sum(flags) / n,
            'episodes': len(episodes),
            'episode_len_min_med_max': (
                (min(episodes), int(np.median(episodes)), max(episodes))
                if episodes else (0, 0, 0)),
        }
        susp_total += sum(flags)
        span_total += n
    bull_by_date = {
        t: dict(zip(m['span_dates'],
                    (is_suspended(s) for s in m['span_states'])))
        for t, m in markets.items()}
    tickers = list(markets)
    agreement = {}
    for a_i in range(len(tickers)):
        for b_i in range(a_i + 1, len(tickers)):
            a, b = tickers[a_i], tickers[b_i]
            common = bull_by_date[a].keys() & bull_by_date[b].keys()
            agreement[f'{a}/{b}'] = float(np.mean(
                [bull_by_date[a][d] == bull_by_date[b][d] for d in common]))
    return {'per_ticker': per, 'pooled_fraction': susp_total / span_total,
            'suspended_days': susp_total, 'span_days': span_total,
            'pairwise_bull_agreement': agreement}


def _runs(flags: Sequence[bool]) -> Iterator[tuple[bool, Iterator[bool]]]:
    from itertools import groupby
    return groupby(flags)


# ---- §5.1: the shared placebo-sequence generator ----

def master_calendar(markets: dict[str, dict[str, Any]]) -> list[str]:
    """Sorted union of the three §3.1 span date lists (§5.1)."""
    cal: set[str] = set()
    for m in markets.values():
        cal.update(m['span_dates'])
    return sorted(cal)


def run_length_multisets(
    markets: dict[str, dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Bull-run / non-bull-run length multisets, pooled across tickers (§5.1).

    Measured on each ticker's span-restricted shifted signal, binarized to
    suspended vs tradeable; boundary-censored first and last runs included
    at their observed lengths.

    The multiset is registered as unordered, but draw_raw_sequence samples
    it by INDEX, so the stream's byte-identity depends on this list's order
    — i.e. on the markets dict being built in TICKERS order, as every call
    site does. The sha256 fingerprint pin in test_trend_gate.py turns any
    silent reorder into a loud CI failure.
    """
    bull: list[int] = []
    nonbull: list[int] = []
    for m in markets.values():
        flags = [is_suspended(s) for s in m['span_states']]
        for v, g in _runs(flags):
            (bull if v else nonbull).append(len(list(g)))
    return bull, nonbull


def draw_raw_sequence(
    rng: np.random.Generator,
    n: int,
    bull_lengths: Sequence[int],
    nonbull_lengths: Sequence[int],
) -> np.ndarray:
    """One §5.1 draw: alternating runs over the n-day master calendar.

    Initial state suspended with probability F_STAR; each run length sampled
    with replacement from its multiset; the final run truncated at the
    calendar end. RNG call pattern (one uniform for the initial state, one
    integers() per run) is the implementation of record at this commit.
    """
    out = np.empty(n, dtype=bool)
    suspended = bool(rng.random() < F_STAR)
    i = 0
    while i < n:
        lengths = bull_lengths if suspended else nonbull_lengths
        run = int(lengths[int(rng.integers(len(lengths)))])
        out[i:i + run] = suspended
        i += run
        suspended = not suspended
    return out


def pooled_fraction(seq: np.ndarray, span_masks: Sequence[np.ndarray]) -> float:
    """§3.3 formula applied to a master-calendar sequence (§5.1 acceptance):
    (Σ suspended-in-span ticker-days) / (Σ span ticker-days)."""
    susp = sum(int(seq[mask].sum()) for mask in span_masks)
    total = sum(int(mask.sum()) for mask in span_masks)
    return susp / total


def span_masks(markets: dict[str, dict[str, Any]],
               calendar: Sequence[str]) -> dict[str, np.ndarray]:
    """Boolean mask over the master calendar for each ticker's span dates."""
    index = {d: i for i, d in enumerate(calendar)}
    masks: dict[str, np.ndarray] = {}
    for t, m in markets.items():
        mask = np.zeros(len(calendar), dtype=bool)
        for d in m['span_dates']:
            mask[index[d]] = True
        masks[t] = mask
    return masks


def accepted_sequences(markets: dict[str, dict[str, Any]]) -> Iterator[np.ndarray]:
    """The single accepted-sequence stream (§5.1): seed 20260611, draws taken
    sequentially, accepted iff pooled fraction ∈ [ACCEPT_LO, ACCEPT_HI],
    kept in order. Raises (→ §11 amendment) if acceptance < 1% over the
    first 10⁶ raw draws. Stage 1 consumes the first 10,000 accepted; Family
    R is the first 1,000 of the SAME stream.
    """
    calendar = master_calendar(markets)
    masks = list(span_masks(markets, calendar).values())
    bull, nonbull = run_length_multisets(markets)
    rng = np.random.default_rng(SEQUENCE_SEED)
    raw = accepted = 0
    while True:
        seq = draw_raw_sequence(rng, len(calendar), bull, nonbull)
        raw += 1
        if ACCEPT_LO <= pooled_fraction(seq, masks) <= ACCEPT_HI:
            accepted += 1
            yield seq
        if raw == RAW_DRAW_GUARD and accepted < 0.01 * raw:
            raise RuntimeError(
                f'§5.1 guard: {accepted}/{raw} raw draws accepted (<1%) — '
                'stop and amend per §11')


# ---- §5.2: Test A (entry-state cycle split) ----

TERMINAL_ACTIONS = ('expiration', 'close', 'close_itm')


def reconstruct_cycles(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each 'sell' with the next terminal record (§5.2); drop the
    at-most-one cycle still open at span end. Uses the recorded terminal
    `pnl` field. Refuses any action outside the registered vocabulary
    (notably 'close_stop': unreachable under §3.2's no-stop config, but a
    stopped run fed here by mistake would otherwise silently merge cycles).
    """
    cycles: list[dict[str, Any]] = []
    entry: dict[str, Any] | None = None
    for t in trades:
        assert t['action'] in ('sell', *TERMINAL_ACTIONS), (
            f"unregistered trade action {t['action']!r} — §3.2 runs no stop")
        if t['action'] == 'sell':
            entry = t
        elif t['action'] in TERMINAL_ACTIONS:
            assert entry is not None, 'terminal record without a sell'
            cycles.append({'entry_date': entry['date'],
                           'terminal_date': t['date'], 'pnl': t['pnl']})
            entry = None
    return cycles


def d_a(cycles: Sequence[dict[str, Any]],
        suspended: set[str] | None = None) -> float | None:
    """D_A = mean per-cycle pnl of bull-entry cycles minus mean of non-bull
    entries, pooled across tickers in per-cycle dollars (§5.2).

    Tagging is PER TICKER for the real statistic: each pooled cycle carries
    a 'bull' flag precomputed from its own ticker's §2.1 state on its entry
    date (the per-ticker signals agree only 0.68-0.81 of days, §3.3 — a
    shared date→state map would mis-tag ~19% of MSFT/SPY observations).
    Placebo re-tagging (`suspended` given) is date membership in the shared
    master-calendar sequence — the registered §5.1 design. None = empty
    cell (→ §5.1 degenerate-draw replacement)."""
    bull: list[float] = []
    nonbull: list[float] = []
    for c in cycles:
        tagged_bull = (c['entry_date'] in suspended if suspended is not None
                       else c['bull'])
        (bull if tagged_bull else nonbull).append(c['pnl'])
    if not bull or not nonbull:
        return None
    return float(np.mean(bull) - np.mean(nonbull))


# ---- §5.3: Test B (price-path exceedance) ----

def test_b_observations(market: dict[str, Any]) -> list[tuple[str, int, bool]]:
    """Per §5.3: for each span day t where select_entry(day, 30, 0.25)
    returns a candidate and t + 30 calendar days does not fall past the span
    end, record (t, x_t, real_bull_t) with x_t = 1 iff the last close on or
    before t + 30 calendar days STRICTLY exceeds the candidate's strike, and
    real_bull_t the ticker's OWN §2.1 state on t (carried per observation so
    pooling across tickers cannot cross-tag)."""
    from datetime import datetime, timedelta

    from realchains.real_cc_backtest import select_entry

    span_dates = market['span_dates']
    span_prices = market['span_prices']
    span_states = market['span_states']
    store = market['store']
    end = span_dates[-1]
    obs: list[tuple[str, int, bool]] = []
    for i, t in enumerate(span_dates):
        horizon = (datetime.strptime(t, '%Y-%m-%d')
                   + timedelta(days=30)).strftime('%Y-%m-%d')
        if horizon > end:
            continue  # §5.3: days within 30 calendar days of span end excluded
        day = store.get(t)
        if day is None:
            continue
        pick = select_entry(day, 30, 0.25)
        if pick is None:
            continue
        strike = pick[6]
        # last close on or before t + 30 calendar days
        j = i
        while j + 1 < len(span_dates) and span_dates[j + 1] <= horizon:
            j += 1
        obs.append((t, int(span_prices[j] > strike),
                    is_suspended(span_states[i])))
    return obs


def d_b(observations: Sequence[tuple[str, int, bool]],
        suspended: set[str] | None = None) -> float | None:
    """D_B = pooled day-weighted exceedance rate on bull days minus the rate
    on non-bull days, one point estimate across tickers (§5.3).

    Observations are (date, x, real_bull) triples; real_bull is precomputed
    PER TICKER from that ticker's own §2.1 state (same rationale as d_a).
    Placebo re-tagging (`suspended` given) is date membership in the shared
    sequence."""
    bull_x: list[int] = []
    nonbull_x: list[int] = []
    for date, x, real_bull in observations:
        tagged_bull = (date in suspended if suspended is not None
                       else real_bull)
        (bull_x if tagged_bull else nonbull_x).append(x)
    if not bull_x or not nonbull_x:
        return None
    return float(np.mean(bull_x) - np.mean(nonbull_x))


def add_one_p(real: float, placebo: Sequence[float], tail: str) -> float:
    """The add-one Monte Carlo p-value (§5.2/§5.3/§6.3; Davison & Hinkley).
    tail='le' counts placebo ≤ real (Test A); 'ge' counts placebo ≥ real
    (Test B, Family R)."""
    arr = np.asarray(placebo, dtype=float)
    hits = int((arr <= real).sum()) if tail == 'le' else int((arr >= real).sum())
    return (1 + hits) / (1 + len(arr))


class SequencePool:
    """The §5.1 pool discipline: the first `size` accepted sequences are a
    FIXED pool shared by both Stage 1 tests; degenerate draws are replaced
    from the tail of the same stream (sequence size+1, size+2, ...), with
    each test replaying the tail from its start so both tests' first
    replacement is the same sequence."""

    def __init__(self, markets: dict[str, dict[str, Any]], size: int) -> None:
        self._stream = accepted_sequences(markets)
        self.pool = [next(self._stream) for _ in range(size)]
        self._tail: list[np.ndarray] = []

    def tail(self, j: int) -> np.ndarray:
        while len(self._tail) <= j:
            self._tail.append(next(self._stream))
        return self._tail[j]


def placebo_statistics(
    stat_fn: Any,
    pool: SequencePool,
    calendar: Sequence[str],
) -> tuple[list[float], int]:
    """Evaluate a re-tagging statistic under every pool sequence, applying
    the §5.1 degenerate-draw rule: a sequence producing an empty cell
    (stat_fn → None) is replaced by the next accepted sequence past the
    pool (itself replaced if degenerate). Returns (statistics, replacement
    count) — the count is reported, and > 2% of the pool triggers a §11
    amendment."""
    cal = list(calendar)

    def evaluate(seq: np.ndarray) -> float | None:
        return stat_fn({cal[i] for i in np.flatnonzero(seq)})

    out: list[float] = []
    replacements = 0
    tail_j = 0
    for seq in pool.pool:
        val = evaluate(seq)
        while val is None:
            val = evaluate(pool.tail(tail_j))
            tail_j += 1
            replacements += 1
        out.append(val)
    return out, replacements


# ---- §6.1: the primary statistic ----

def short_call_days(trades: list[dict[str, Any]], span_dates: Sequence[str]) -> int:
    """Trading days with a call open (§6.1): closed cycles count entry date
    inclusive to terminal date exclusive; the at-most-one final open cycle
    counts entry inclusive through the final span date inclusive."""
    index = {d: i for i, d in enumerate(span_dates)}
    days = 0
    entry: str | None = None
    for t in trades:
        if t['action'] == 'sell':
            entry = t['date']
        elif t['action'] in TERMINAL_ACTIONS:
            assert entry is not None
            days += index[t['date']] - index[entry]
            entry = None
    if entry is not None:  # final open cycle
        days += len(span_dates) - index[entry]
    return days


def statistic_t(per_ticker: dict[str, dict[str, float]]) -> float:
    """T = (1/3) Σ_k net_overlay_pnl_k / short_call_days_k (§6.1), equal
    ticker weights. Caller guarantees short_call_days > 0 for every ticker
    (§6.2 replaces zero-exposure placebo sequences)."""
    return float(np.mean([v['net_overlay_pnl'] / v['short_call_days']
                          for v in per_ticker.values()]))


# ---- §4: arms ----

def record_suspension(market: dict[str, Any]) -> set[str]:
    """Arm 2: suspend on (real-signal) bull days (§2.1, §4)."""
    return {d for d, s in zip(market['span_dates'], market['span_states'])
            if is_suspended(s)}


def complement_suspension(market: dict[str, Any]) -> set[str]:
    """Arm 3: sell ONLY on bull days — i.e. suspend the non-bull days (§4)."""
    return {d for d, s in zip(market['span_dates'], market['span_states'])
            if not is_suspended(s)}


def vol_ablation_suspension(market: dict[str, Any]) -> set[str]:
    """Arm 4 (§4): suspend iff the prior day's 30-day rolling annualized vol
    (calc_rolling_volatility, shifted one day) is below the per-ticker
    in-span quantile (numpy.quantile, method 'linear') at the level equal to
    that ticker's exact bull fraction. Diagnostic, not a tradable strategy:
    the threshold uses the full-span vol distribution."""
    closes = np.asarray(market['full_closes'], dtype=float)
    vols = calc_rolling_volatility(closes, window=30)
    # vols[i] pairs with the return realized on full-series day i+1, so the
    # vol KNOWN at the open of day j (through close j-1) is vols[j-2].
    offset = market['span_offset']
    shifted: list[float] = []
    for k in range(len(market['span_dates'])):
        j = offset + k
        shifted.append(float(vols[j - 2]) if j >= 2 else float('nan'))
    defined = ~np.isnan(np.asarray(shifted))
    flags = [is_suspended(s) for s in market['span_states']]
    level = sum(flags) / len(flags)
    q = float(np.quantile(np.asarray(shifted)[defined], level, method='linear'))
    return {d for d, v in zip(market['span_dates'], shifted)
            if not math.isnan(v) and v < q}


def run_arm(market: dict[str, Any],
            suspension: set[str] | None,
            fill: str = 'bid_ask') -> dict[str, Any]:
    """One engine run (§3.2 baseline params, §10 seam) → the §6 quantities."""
    params = dict(ENGINE_PARAMS)
    if fill != 'bid_ask':
        params['fill'] = fill  # type: ignore[assignment]
    summary, trades, eq = run_real_cc_overlay(
        market['span_dates'], market['span_prices'], market['store'],
        params, suspended_dates=suspension)
    cycles = reconstruct_cycles(trades)
    closed_pnl = sum(c['pnl'] for c in cycles)
    scd = short_call_days(trades, market['span_dates'])
    open_entry = _open_entry_date(trades)
    return {
        'summary': summary,
        'trades': trades,
        'daily_equity': eq,
        'net_overlay_pnl': summary['net_overlay_pnl'],
        'total_premium_collected': summary['total_premium_collected'],
        'short_call_days': scd,
        'cycles': cycles,
        # §6.4 LOYO: the final open cycle counts as a cycle with its entry
        # year and its mark-to-market as pnl (= net overlay P&L minus the
        # sum of closed-cycle pnls).
        'open_cycle': (
            {'entry_date': open_entry,
             'pnl': summary['net_overlay_pnl'] - closed_pnl,
             'days': len(market['span_dates'])
                     - market['span_dates'].index(open_entry)}
            if open_entry is not None else None),
    }


def _open_entry_date(trades: list[dict[str, Any]]) -> str | None:
    entry: str | None = None
    for t in trades:
        if t['action'] == 'sell':
            entry = t['date']
        elif t['action'] in TERMINAL_ACTIONS:
            entry = None
    return entry


# ---- §6.2: placebo families ----

def sequence_record(markets: dict[str, dict[str, Any]],
                    calendar: Sequence[str],
                    seq: np.ndarray,
                    complement: bool) -> dict[str, Any] | None:
    """Run one placebo sequence (or its complement gate) through full engine
    re-runs on all three tickers (§6.2). Returns None when any ticker errors
    or produces zero short-call-days (→ replacement, reported)."""
    cal = list(calendar)
    susp_dates = {cal[i] for i in np.flatnonzero(seq if not complement
                                                 else ~seq)}
    per: dict[str, dict[str, Any]] = {}
    for t, m in markets.items():
        try:
            arm = run_arm(m, susp_dates & set(m['span_dates']))
        except Exception:
            return None
        if arm['short_call_days'] == 0:
            return None
        per[t] = {
            'net_overlay_pnl': arm['net_overlay_pnl'],
            'total_premium_collected': arm['total_premium_collected'],
            'short_call_days': arm['short_call_days'],
            # retained for the §6.4 leave-one-year-out recomputation
            'cycles': [[c['entry_date'],
                        _cycle_days(c, m['span_dates']), c['pnl']]
                       for c in arm['cycles']]
                      + ([[arm['open_cycle']['entry_date'],
                           arm['open_cycle']['days'],
                           arm['open_cycle']['pnl']]]
                         if arm['open_cycle'] else []),
        }
    return per


def _cycle_days(cycle: dict[str, Any], span_dates: Sequence[str]) -> int:
    index = {d: i for i, d in enumerate(span_dates)}
    return index[cycle['terminal_date']] - index[cycle['entry_date']]


def family_s_offsets() -> list[int]:
    """§6.2 Family S: 500 shared offsets, uniform integers on [250, 3374]
    (250 to shortest span − 250), from numpy.random.default_rng(42)."""
    rng = np.random.default_rng(FAMILY_S_SEED)
    return [int(x) for x in rng.integers(250, 3_374 + 1, size=FAMILY_S_DRAWS)]


def shifted_signal_suspension(market: dict[str, Any], offset: int) -> set[str]:
    """Circularly shift the ticker's own real in-span signal by `offset`
    trading days within its own span (§6.2 Family S)."""
    flags = [is_suspended(s) for s in market['span_states']]
    n = len(flags)
    rotated = [flags[(i - offset) % n] for i in range(n)]
    return {d for d, f in zip(market['span_dates'], rotated) if f}


# ---- §6.4: mandatory secondary analyses ----

def loyo_t(per_ticker_cycles: dict[str, list[list[Any]]],
           year: str) -> float | None:
    """LOYO-T_Y (§6.4): the §6.1 statistic recomputed from cycle records with
    entry-date year ≠ Y. Cycle record = [entry_date, days, pnl]; the final
    open cycle is included with its mark-to-market pnl. None when any ticker
    has zero remaining short-call-days."""
    parts: list[float] = []
    for cycles in per_ticker_cycles.values():
        kept = [c for c in cycles if not c[0].startswith(year)]
        days = sum(c[1] for c in kept)
        if days == 0:
            return None
        parts.append(sum(c[2] for c in kept) / days)
    return float(np.mean(parts))


def common_base_nw_t(record_eq: Any, baseline_summary: dict[str, Any]) -> dict[str, Any]:
    """§6.4: daily common-base excess Newey-West t on the record arm,
    e_t = (ΔE_gated − ΔE_bh) / E_bh,t−1 — zero by construction on uncovered
    days. Mirrors compute_statistics' NW estimator (Bartlett weights,
    L = floor(4 (n/100)^(2/9))); descriptive only."""
    shares = baseline_summary['num_contracts'] * 100
    cash = baseline_summary['cash']
    equity = record_eq['equity'].to_numpy(dtype=float)
    prices = record_eq['price'].to_numpy(dtype=float)
    bh = shares * prices + cash
    excess = (np.diff(equity) - np.diff(bh)) / bh[:-1]
    n = len(excess)
    mean_e = float(np.mean(excess))
    var_e = float(np.var(excess, ddof=1))
    lag = int(4 * (n / 100) ** (2 / 9))
    nw_sum = 0.0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1)
        nw_sum += w * float(np.mean((excess[:-k] - mean_e) * (excess[k:] - mean_e)))
    se = math.sqrt(max((var_e + 2 * nw_sum) / n, 0.0))
    return {'t_nw': mean_e / se if se > 0 else 0.0, 'lag': lag, 'n': n}


def stress_window_deltas(record_eq: Any, baseline_eq: Any,
                         window: tuple[str, str]) -> float:
    """§6.4 stress metric: record-arm equity change minus baseline-arm equity
    change over the window; window change = last value in window minus last
    value strictly before the window start."""
    lo, hi = window

    def change(eq: Any) -> float:
        dates = eq['date'].tolist()
        vals = eq['equity'].to_numpy(dtype=float)
        before = [i for i, d in enumerate(dates) if d < lo]
        inside = [i for i, d in enumerate(dates) if lo <= d <= hi]
        if not before or not inside:
            return float('nan')
        return float(vals[inside[-1]] - vals[before[-1]])

    return change(record_eq) - change(baseline_eq)


# ---- checkpointing (Family R/S engine runs are hours of compute) ----

def checkpoint_path(name: str) -> str:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, name)


def load_checkpoint(path: str) -> dict[int, Any]:
    """Load a JSONL checkpoint, tolerating exactly one failure shape: a
    partial FINAL line from a crash mid-append (dropped and truncated away,
    so the next append cannot concatenate onto it; the dropped record was
    never entered into the in-memory map, so it simply recomputes). A
    malformed line anywhere else is corruption and raises."""
    out: dict[int, Any] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        lines = f.read().splitlines()
    good: list[str] = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                with open(path, 'w') as f:
                    f.write(''.join(s + '\n' for s in good))
                break
            raise
        good.append(line)
        out[rec['i']] = rec
    return out


def append_checkpoint(path: str, rec: dict[str, Any]) -> None:
    """Append one record, healing a missing trailing newline first (a crash
    after the JSON but before the newline would otherwise merge records)."""
    line = json.dumps(rec) + '\n'
    with open(path, 'a+b') as f:
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                f.write(b'\n')
        f.write(line.encode())


# ---- CLI ----

def _require_clean_tree(allow_dirty: bool) -> None:
    """§10 ordering: results must cite the analysis-code commit, so stage
    runs refuse a dirty tree by default."""
    if allow_dirty:
        return
    proc = subprocess.run(['git', 'status', '--porcelain'],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit('git status failed — run from the repo root (the §10 '
                 'ordering guard needs a readable tree), or pass '
                 '--allow-dirty')
    if proc.stdout.strip():
        sys.exit('working tree is dirty — commit first so results carry an '
                 'honest code provenance, or pass --allow-dirty')


def cmd_characterize(markets: dict[str, dict[str, Any]]) -> None:
    rep = characterize(markets)
    print(json.dumps(rep, indent=2, default=str))
    print(f"\nregistered F_STAR = {F_STAR}; recomputed pooled fraction = "
          f"{rep['pooled_fraction']:.6f}")


def stage1_baseline(markets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The cheap half of Stage 1 (§5.2/§5.3 real statistics + §9(a) MDE):
    three signal-unconditioned baseline engine runs, the pooled cycle/
    exceedance series tagged by each ticker's OWN §2.1 state, and the
    derived D_A/D_B/σ/MDE. No placebo pool — split out so the deterministic
    core is pinnable without paying for the 10,000-sequence re-tagging.

    Each cycle and exceedance observation is tagged with its own ticker's
    state at pooling time: the per-ticker signals agree only 0.68-0.81 of
    days (§3.3), so any shared date→state map would cross-tag.
    """
    cycles: list[dict[str, Any]] = []
    obs: list[tuple[str, int, bool]] = []
    n_expected_record = 0.0
    for m in markets.values():
        bull_dates = record_suspension(m)
        arm = run_arm(m, None)
        for c in arm['cycles']:
            c['bull'] = c['entry_date'] in bull_dates
        cycles.extend(arm['cycles'])
        obs.extend(test_b_observations(m))
        # §9: expected record-arm cycle count = baseline rate × the ticker's
        # tradeable (non-bull) fraction, assuming uniform cycle rate.
        tradeable = 1 - len(bull_dates) / len(m['span_dates'])
        n_expected_record += len(arm['cycles']) * tradeable

    real_a = d_a(cycles)
    real_b = d_b(obs)
    assert real_a is not None and real_b is not None

    # §9(a): σ from the (signal-unconditioned) baseline cycles, but SE and
    # the t=2 mean at the EXPECTED RECORD-ARM n (~325 in the registration's
    # arithmetic) — the sample the verdict will actually have, not the
    # baseline's ~2.6x larger one.
    sigma_cycles = [c['pnl'] for c in cycles]
    sd = float(np.std(sigma_cycles, ddof=1))
    n_baseline = len(sigma_cycles)
    n_exp = max(int(round(n_expected_record)), 1)
    return {
        'cycles': cycles, 'obs': obs,
        'D_A': real_a, 'D_B': real_b,
        'n_cycles_baseline': n_baseline, 'n_exceedance_days': len(obs),
        'mde_9a': {'per_cycle_sigma': sd,
                   'n_baseline': n_baseline,
                   'n_expected_record': n_exp,
                   'se_of_mean': sd / math.sqrt(n_exp),
                   'mean_at_t2': 2 * sd / math.sqrt(n_exp)},
    }


def stage1_pvalues(markets: dict[str, dict[str, Any]],
                   baseline: dict[str, Any]) -> dict[str, Any]:
    """The expensive half of Stage 1: re-tag the fixed baseline cycles/obs
    under the first 10,000 accepted placebo sequences (§5.1 shared pool),
    the add-one p-values (§5.2/§5.3), and the §5.4 gate rule."""
    calendar = master_calendar(markets)
    cycles, obs = baseline['cycles'], baseline['obs']
    pool = SequencePool(markets, STAGE1_SEQUENCES)
    pa_stats, rep_a = placebo_statistics(
        lambda susp: d_a(cycles, suspended=susp), pool, calendar)
    pb_stats, rep_b = placebo_statistics(
        lambda susp: d_b(obs, suspended=susp), pool, calendar)
    p_a = add_one_p(baseline['D_A'], pa_stats, 'le')
    p_b = add_one_p(baseline['D_B'], pb_stats, 'ge')
    passes = baseline['D_A'] < 0 and baseline['D_B'] > 0 and min(p_a, p_b) <= 0.10
    return {
        'p_A': p_a, 'replacements_A': rep_a,
        'p_B': p_b, 'replacements_B': rep_b,
        'stage1_passes': passes,
        'replacement_amendment_triggered':
            max(rep_a, rep_b) > REPLACEMENT_AMENDMENT_FRACTION * STAGE1_SEQUENCES,
    }


def stage1_report(markets: dict[str, dict[str, Any]],
                  baseline: dict[str, Any] | None = None,
                  pvals: dict[str, Any] | None = None) -> dict[str, Any]:
    """§5: the full Stage 1 record — Tests A and B against the first 10,000
    accepted sequences, the §5.4 gate rule, and the §9(a) MDE table. A pure
    merge of the two halves; pass precomputed `baseline`/`pvals` to reuse
    work (the regression test does, to avoid a second 10,000-sequence pool)."""
    if baseline is None:
        baseline = stage1_baseline(markets)
    if pvals is None:
        pvals = stage1_pvalues(markets, baseline)
    return {
        'D_A': baseline['D_A'], 'p_A': pvals['p_A'],
        'replacements_A': pvals['replacements_A'],
        'D_B': baseline['D_B'], 'p_B': pvals['p_B'],
        'replacements_B': pvals['replacements_B'],
        'n_cycles_baseline': baseline['n_cycles_baseline'],
        'n_exceedance_days': baseline['n_exceedance_days'],
        'stage1_passes': pvals['stage1_passes'],
        'mde_9a': baseline['mde_9a'],
        'replacement_amendment_triggered': pvals['replacement_amendment_triggered'],
    }


def cmd_stage1(markets: dict[str, dict[str, Any]]) -> None:
    print(json.dumps(stage1_report(markets), indent=2))


def first_family_r_record(markets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """§9(b)/§6.2 drift pin: the first accepted sequence's T and T_c via six
    engine re-runs (record + complement gates), with NO checkpoint side
    effects. Pinning this one record locks the whole Family R pipeline
    cheaply; the published 100-sequence summary (mean/sd/percentiles) is
    reproducible via `placebo-mde`, which checkpoints the same first record."""
    calendar = master_calendar(markets)
    seq = next(accepted_sequences(markets))
    rec_per = sequence_record(markets, calendar, seq, complement=False)
    comp_per = sequence_record(markets, calendar, seq, complement=True)
    assert rec_per is not None and comp_per is not None
    return {
        'T': statistic_t(rec_per), 'T_c': statistic_t(comp_per),
        'premium_ratio': float(np.mean(
            [v['net_overlay_pnl'] / v['total_premium_collected']
             for v in rec_per.values()])),
    }


def _stream_fingerprint(markets: dict[str, dict[str, Any]]) -> str:
    """sha256 of the first accepted sequence — identifies the generator's
    inputs and RNG call pattern, so a resumed checkpoint can prove it was
    written by the same stream (the keys are bare stream positions)."""
    import hashlib
    probe = accepted_sequences(markets)
    return hashlib.sha256(next(probe).tobytes()).hexdigest()


def _family_r_records(markets: dict[str, dict[str, Any]], count: int,
                      path: str) -> tuple[dict[int, Any], int]:
    """Push the first `count` accepted sequences through §6.2 engine re-runs
    (record + complement gates), checkpointed and resumable. Replacement
    sequences come from the same stream, past the pool, in order. Returns
    (records keyed by stream position, replacement count) — §6.2 requires
    the count be reported, and > 2% triggers an amendment."""
    calendar = master_calendar(markets)
    done = load_checkpoint(path)
    fp = _stream_fingerprint(markets)
    header = done.pop(-1, None)
    if header is None:
        append_checkpoint(path, {'i': -1, 'fingerprint': fp})
    elif header['fingerprint'] != fp:
        sys.exit(f'{path} was written by a different sequence stream '
                 f'(fingerprint {header["fingerprint"][:12]}… != current '
                 f'{fp[:12]}…) — the code or data changed mid-experiment; '
                 'amend per §11 and start a fresh checkpoint')
    stream = accepted_sequences(markets)
    kept = 0      # sequences accepted into the family so far
    consumed = 0  # accepted-stream positions consumed (incl. replaced)
    replacements = 0
    while kept < count:
        seq = next(stream)
        i = consumed
        consumed += 1
        if i in done:
            if not done[i].get('replaced'):
                kept += 1
            replacements += int(bool(done[i].get('replaced')))
            continue
        rec_per = sequence_record(markets, calendar, seq, complement=False)
        comp_per = (sequence_record(markets, calendar, seq, complement=True)
                    if rec_per is not None else None)
        if rec_per is None or comp_per is None:
            append_checkpoint(path, {'i': i, 'replaced': True})
            done[i] = {'i': i, 'replaced': True}
            replacements += 1
            continue
        rec = {'i': i, 'replaced': False,
               'T': statistic_t(rec_per), 'T_c': statistic_t(comp_per),
               'premium_ratio': float(np.mean(
                   [v['net_overlay_pnl'] / v['total_premium_collected']
                    for v in rec_per.values()])),
               'cycles': {t: v['cycles'] for t, v in rec_per.items()}}
        append_checkpoint(path, rec)
        done[i] = rec
        kept += 1
    if replacements > REPLACEMENT_AMENDMENT_FRACTION * count:
        print(f'WARNING: {replacements} replacements (> 2% of {count}) — '
              'amendment required per §11')
    return done, replacements


def _family_members(done: dict[int, Any], count: int) -> list[dict[str, Any]]:
    """The family in stream order: non-replaced records, ascending position."""
    return [done[i] for i in sorted(done)
            if not done[i].get('replaced')][:count]


def cmd_placebo_mde(markets: dict[str, dict[str, Any]]) -> None:
    """§9(b): the spread of T_i over the first 100 Family R sequences (they
    remain the first 100 of the 1,000; runs are checkpointed and reused by
    stage2). Does not unblind the record arm."""
    done, replacements = _family_r_records(markets, 100,
                                           checkpoint_path('family_r.jsonl'))
    ts = [r['T'] for r in _family_members(done, 100)]
    print(json.dumps({'n': len(ts), 'replacements': replacements,
                      'mean': float(np.mean(ts)),
                      'sd': float(np.std(ts, ddof=1)),
                      'p5': float(np.percentile(ts, 5)),
                      'p95': float(np.percentile(ts, 95))}, indent=2))


def cmd_stage2(markets: dict[str, dict[str, Any]]) -> None:
    """§6: arms, Family R (1,000) and Family S (500), the §6.3 pass rule,
    and every §6.4 mandatory secondary."""
    # Arms (§4)
    arms: dict[str, dict[str, dict[str, Any]]] = {
        'baseline': {}, 'record': {}, 'complement': {}, 'vol_ablation': {}}
    for t, m in markets.items():
        arms['baseline'][t] = run_arm(m, None)
        arms['record'][t] = run_arm(m, record_suspension(m))
        arms['complement'][t] = run_arm(m, complement_suspension(m))
        arms['vol_ablation'][t] = run_arm(m, vol_ablation_suspension(m))

    def arm_t(name: str) -> float:
        return statistic_t({t: a for t, a in arms[name].items()})

    real_t, comp_t, vol_t = arm_t('record'), arm_t('complement'), arm_t('vol_ablation')

    # Family R (§6.2)
    done, r_replacements = _family_r_records(markets, FAMILY_R_SEQUENCES,
                                             checkpoint_path('family_r.jsonl'))
    fam = _family_members(done, FAMILY_R_SEQUENCES)
    t_i = [r['T'] for r in fam]
    tc_i = [r['T_c'] for r in fam]
    p_r = add_one_p(real_t, t_i, 'ge')
    p_c = add_one_p(comp_t, tc_i, 'le')

    # Family S (§6.2). Checkpointed like Family R (1,500 engine runs). A
    # zero-exposure draw is recorded as null and DROPPED, not replaced —
    # §6.2 defines no Family S replacement rule, and replacing would break
    # the shared-offset structure; the shrinkage is visible in family_s_n.
    # (Practically unreachable: a circular shift preserves the real
    # suspension fractions, leaving 31-43% of days tradeable per ticker.)
    s_path = checkpoint_path('family_s.jsonl')
    s_done = load_checkpoint(s_path)
    s_stats: list[float] = []
    for k, off in enumerate(family_s_offsets()):
        if k in s_done:
            if s_done[k]['T'] is not None:
                s_stats.append(s_done[k]['T'])
            continue
        per = {}
        for t, m in markets.items():
            arm = run_arm(m, shifted_signal_suspension(m, off))
            if arm['short_call_days'] == 0:
                per = {}
                break
            per[t] = arm
        val = statistic_t(per) if per else None
        append_checkpoint(s_path, {'i': k, 'offset': off, 'T': val})
        s_done[k] = {'i': k, 'T': val}
        if val is not None:
            s_stats.append(val)

    # §6.3 pass rule + binding clause
    passes = real_t > 0 and p_r <= 0.05
    vol_clause = vol_t >= real_t

    # §6.4 secondaries
    record_cycles = {
        t: [[c['entry_date'], _cycle_days(c, markets[t]['span_dates']), c['pnl']]
            for c in arms['record'][t]['cycles']]
           + ([[arms['record'][t]['open_cycle']['entry_date'],
                arms['record'][t]['open_cycle']['days'],
                arms['record'][t]['open_cycle']['pnl']]]
              if arms['record'][t]['open_cycle'] else [])
        for t in markets}
    years = sorted({c[0][:4] for cs in record_cycles.values() for c in cs})
    loyo = {}
    for y in years:
        ly = loyo_t(record_cycles, y)
        ly_placebo = [v for v in (loyo_t(r['cycles'], y) for r in fam)
                      if v is not None]
        p_y = add_one_p(ly, ly_placebo, 'ge') if ly is not None else None
        # §6.4: "if dropping any single year flips the §6.3 verdict" — the
        # flip is DIRECTION-NEUTRAL: a registered-null result that becomes a
        # pass without (say) 2022 must carry the single-year-dependent
        # qualifier just as a pass that becomes a null must. (An undefined
        # LOYO-T — a ticker emptied by the removal — analogizes to a fail.)
        analog_pass = ly is not None and ly > 0 and p_y is not None and p_y <= 0.05
        loyo[y] = {'T': ly, 'p': p_y, 'n_placebo': len(ly_placebo),
                   'flips_verdict': analog_pass != passes}

    bull_days = {t: record_suspension(m) for t, m in markets.items()}
    scd_on_bull = {}
    for t, m in markets.items():
        idx = {d: i for i, d in enumerate(m['span_dates'])}
        open_days: set[str] = set()
        entry = None
        for tr in arms['record'][t]['trades']:
            if tr['action'] == 'sell':
                entry = tr['date']
            elif tr['action'] in TERMINAL_ACTIONS:
                open_days.update(m['span_dates'][idx[entry]:idx[tr['date']]])
                entry = None
        if entry is not None:
            open_days.update(m['span_dates'][idx[entry]:])
        scd_on_bull[t] = (len(open_days & bull_days[t]) / len(open_days)
                          if open_days else 0.0)

    per_ticker_t = {t: arms['record'][t]['net_overlay_pnl']
                       / arms['record'][t]['short_call_days']
                    for t in markets}
    report = {
        'T': {'record': real_t, 'complement': comp_t, 'vol_ablation': vol_t,
              'baseline': arm_t('baseline')},
        'per_ticker_T': per_ticker_t,
        # §6.4: same-sign tally beside the components (descriptive; three
        # correlated underlyings are ~1.2 independent tests, not 3).
        'per_ticker_T_same_sign': sum(
            1 for v in per_ticker_t.values() if (v > 0) == (real_t > 0)),
        'p_R': p_r, 'p_C': p_c,
        'family_r_n': len(fam),
        'family_r_replacements': r_replacements,
        'family_r_amendment_triggered':
            r_replacements > REPLACEMENT_AMENDMENT_FRACTION * FAMILY_R_SEQUENCES,
        'family_s_percentile': (float(np.mean([s <= real_t for s in s_stats]))
                                if s_stats else None),
        'family_s_n': len(s_stats),
        'passes_6_3': passes,
        'vol_ablation_binding_clause': vol_clause,
        'premium_ratio': {'record': float(np.mean(
            [arms['record'][t]['net_overlay_pnl']
             / arms['record'][t]['total_premium_collected'] for t in markets])),
            'family_r_p': add_one_p(float(np.mean(
                [arms['record'][t]['net_overlay_pnl']
                 / arms['record'][t]['total_premium_collected']
                 for t in markets])),
                [r['premium_ratio'] for r in fam], 'ge')},
        'short_call_days_on_bull_fraction': scd_on_bull,
        'loyo': loyo,
        # §6.4: per ticker AND pooled (the pooled value is the dollar sum).
        'stress_windows': {f'{lo}->{hi}': (lambda per: {
            **per, 'pooled': float(sum(per.values()))})({
                t: stress_window_deltas(arms['record'][t]['daily_equity'],
                                        arms['baseline'][t]['daily_equity'],
                                        (lo, hi)) for t in markets})
            for lo, hi in STRESS_WINDOWS},
        'common_base_nw': {t: common_base_nw_t(
            arms['record'][t]['daily_equity'],
            arms['baseline'][t]['summary']) for t in markets},
    }
    print(json.dumps(report, indent=2, default=str))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'characterize'
    allow_dirty = '--allow-dirty' in sys.argv
    if cmd in ('stage1', 'stage2', 'placebo-mde'):
        _require_clean_tree(allow_dirty)
    print('Loading markets (3 chain stores; a few minutes cold) ...',
          flush=True)
    markets = {t: load_market(t) for t in TICKERS}
    {'characterize': cmd_characterize,
     'stage1': cmd_stage1,
     'placebo-mde': cmd_placebo_mde,
     'stage2': cmd_stage2}[cmd](markets)


if __name__ == '__main__':
    main()
