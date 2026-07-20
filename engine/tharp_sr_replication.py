"""Tharp's support/resistance catalog, replicated — the frozen
docs/tharp_sr_replication_plan.md build.

Two price-only counting phases on the committed daily OHLC files:

- Phase 1 (§3): the extremes claims (book notes Location 4374) — the
  quote-bearing "more extreme opening" reading as C1's headline, the
  close companion (C2), and the trending-day reversal (C3) — each judged
  against a matched-count, era-stratified date-resampled null (never the
  textbook binomial, never the unconditional coin).
- Phase 2 (§4): the LeBeau–Lucas entries-vs-random frame — six frozen
  signals x four time-exit horizons x two sides, per-cell independent
  flat-only books, per-cell seeded matched-count nulls, a NO-VERDICT
  floor under 15 trades, and the pre-committed 3-of-8 survival bar.
  Turtle Soup's era-decay claim counted as a by-product.

Everything here implements the plan doc verbatim; where the doc froze an
algorithm (the null's shuffle-and-greedily-accept sampler, Wilder's RSI
recurrence), the code follows it literally. Deterministic: the only seed
is SR_SEED, with one derived stream per claim/cell.

Epistemic class: exploratory replication (kill-or-justify). No e-value
is spent. The result pins live in
tests/test_tharp_sr_replication.py and RUN IN CI (the OHLC inputs are
committed CSVs).

Run:  python -m engine.tharp_sr_replication [--json]
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from typing import Any, Callable, Sequence

import numpy as np

from common.paths import data_path

SR_SEED = 20260720
B_RESAMPLES = 10_000
MIN_TRADES = 15                 # below this a Phase-2 cell reports NO-VERDICT
PRIMARY_TICKERS = ('QQQ', 'SPY')
ROBUSTNESS_TICKERS = ('MSFT', 'NVDA', 'XLE', 'IWM', 'EEM', 'GLD')  # TLT excluded (plan §2)
HORIZONS = (5, 10, 15, 20)
ERAS = (('1999-01-01', '2009-12-31'), ('2010-01-01', '2019-12-31'),
        ('2020-01-01', '2026-12-31'))
SURVIVAL_P = 0.01
PHASE2_SURVIVAL_MIN_CELLS = 3   # of the 8 primary long cells, spanning both tickers
TURTLE_FAIL_WINDOW = 5
TOP_Q = 0.75                    # top-quartile range position
BOT_Q = 0.25


# ------------------------------------------------------------------ data

def load_ohlc(ticker: str) -> dict[str, np.ndarray]:
    """The committed split-adjusted daily OHLC file as parallel arrays."""
    dates: list[str] = []
    cols: dict[str, list[float]] = {'open': [], 'high': [], 'low': [], 'close': []}
    with open(data_path(f'{ticker.lower()}_daily_ohlc.csv')) as f:
        for row in csv.DictReader(f):
            dates.append(row['date'])
            for k in cols:
                cols[k].append(float(row[k]))
    return {'dates': np.array(dates),
            **{k: np.array(v) for k, v in cols.items()}}


def era_index(date: str) -> int:
    for j, (lo, hi) in enumerate(ERAS):
        if lo <= date <= hi:
            return j
    raise ValueError(f'date {date} outside every era panel')


def derived_rng(label: str) -> np.random.Generator:
    """One deterministic stream per claim/cell: SR_SEED folded with a
    stable hash of the label (plan §3/§4)."""
    h = int(hashlib.sha256(f'{SR_SEED}:{label}'.encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def addone_p_two_sided(null: np.ndarray, observed: float) -> float:
    hi = int(np.sum(null >= observed))
    lo = int(np.sum(null <= observed))
    b = len(null)
    return min(1.0, 2.0 * min((1 + hi) / (1 + b), (1 + lo) / (1 + b)))


def addone_p_one_sided(null: np.ndarray, observed: float) -> float:
    return (1 + int(np.sum(null >= observed))) / (1 + len(null))


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def fisher_exact_one_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """P(second-sample count >= k2 | margins) — the conditional exact
    (hypergeometric) tail for 'rate2 exceeds rate1'. Hand-rolled via
    log-gamma (the repo is dependency-light; no scipy)."""
    def lchoose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float('-inf')
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    total_k = k1 + k2
    total_n = n1 + n2
    denom = lchoose(total_n, total_k)
    p = 0.0
    for j in range(k2, min(total_k, n2) + 1):
        p += math.exp(lchoose(n2, j) + lchoose(n1, total_k - j) - denom)
    return min(1.0, p)


# --------------------------------------------------------------- phase 1

def phase1_measurements(d: dict[str, np.ndarray], ticker: str = '') -> dict[str, Any]:
    """Every §3 measurement on one ticker: conditional sets, outcomes,
    base rates, and the era-stratified matched-count null p — the null
    stream derived per claim x ticker (plan §3), so no two tickers share
    Monte-Carlo noise.

    The eligible universe (conditional and base alike): non-degenerate
    range and a next session in the file. All comparisons strict; exact
    ties count against the claim (plan §0). Returns
    {'measurements': {...}, 'n_degenerate': the frozen §2 diagnostic,
    'era_starts': first eligible date per panel (labels the truncated
    first panels of late-start tickers)}.
    """
    o, h, low, c = d['open'], d['high'], d['low'], d['close']
    n = len(c)
    idx = np.arange(n - 1)                      # a next session exists
    rng_ok = h[idx] > low[idx]
    n_degenerate = int(np.sum(~rng_ok))         # the §2 diagnostic count
    idx = idx[rng_ok]
    rp = (c[idx] - low[idx]) / (h[idx] - low[idx])
    nxt_o, nxt_c = o[idx + 1], c[idx + 1]
    eras = np.array([era_index(dt) for dt in d['dates'][idx]])

    # same-day open position for C3's trending-day classification
    op = (o[idx] - low[idx]) / (h[idx] - low[idx])

    top = rp >= TOP_Q
    bot = rp <= BOT_Q
    trend_up = (op <= BOT_Q) & (rp >= TOP_Q)
    trend_dn = (op >= TOP_Q) & (rp <= BOT_Q)

    meas: dict[str, dict[str, Any]] = {}

    def add(label: str, cond: np.ndarray, outcome: np.ndarray) -> None:
        # the null stream folds the ticker (plan §3: per claim x ticker)
        meas[label] = _conditional_vs_null(f'{ticker}|{label}', cond, outcome, eras)

    # C1 — quote-bearing strict headline + loose companion, both mirrors
    add('C1_strict_top', top, nxt_o > h[idx])
    add('C1_loose_top', top, nxt_o > c[idx])
    add('C1_strict_bot', bot, nxt_o < low[idx])
    add('C1_loose_bot', bot, nxt_o < c[idx])
    # C2 — the close (wheel-tenor) companion
    add('C2_top', top, nxt_c > c[idx])
    add('C2_bot', bot, nxt_c < c[idx])
    # C3 — headline: next session closes against its own open
    add('C3_up', trend_up, nxt_c < nxt_o)
    add('C3_dn', trend_dn, nxt_c > nxt_o)
    # C3 labeled variants (reported, non-gating): gap-continuation
    # conditioning, and the close-vs-close reading
    add('C3_up_gapcont', trend_up & (nxt_o > c[idx]), nxt_c < nxt_o)
    add('C3_dn_gapcont', trend_dn & (nxt_o < c[idx]), nxt_c > nxt_o)
    add('C3_up_closeclose', trend_up, nxt_c < c[idx])
    add('C3_dn_closeclose', trend_dn, nxt_c > c[idx])
    era_starts = []
    for j in range(len(ERAS)):
        in_era = np.array([era_index(dt) == j for dt in d['dates'][idx]])
        era_starts.append(str(d['dates'][idx][in_era][0]) if np.any(in_era) else None)
    return {'measurements': meas, 'n_degenerate': n_degenerate,
            'era_starts': era_starts}


def _conditional_vs_null(label: str, cond: np.ndarray, outcome: np.ndarray,
                         eras: np.ndarray, b: int = B_RESAMPLES) -> dict[str, Any]:
    """One §3 measurement: the matched-count, era-stratified resampled
    null (uniform without replacement within each era panel)."""
    n_cond = int(np.sum(cond))
    out: dict[str, Any] = {'n': n_cond}
    if n_cond == 0:
        out.update({'rate': None, 'base_rate': None, 'gap': None, 'p': None,
                    '_p_raw': None, '_gap_raw': None, 'binom_p': None,
                    'per_era': []})
        return out
    rate = float(np.mean(outcome[cond]))
    base = float(np.mean(outcome))
    per_era = []
    era_counts: list[tuple[int, np.ndarray]] = []
    for j in range(len(ERAS)):
        in_era = eras == j
        n_e = int(np.sum(cond & in_era))
        era_counts.append((n_e, outcome[in_era].astype(float)))
        if int(np.sum(in_era)):
            per_era.append({
                'era': j, 'n': n_e,
                'rate': float(np.mean(outcome[cond & in_era])) if n_e else None,
                'base_rate': float(np.mean(outcome[in_era])),
            })
    rng = derived_rng(label)
    null = np.empty(b)
    for i in range(b):
        s = 0.0
        for n_e, y in era_counts:
            if n_e:
                pick = rng.permutation(len(y))[:n_e]
                s += float(np.sum(y[pick]))
        null[i] = s / n_cond
    p_raw = addone_p_two_sided(null, rate)
    out.update({
        'rate': round(rate, 4), 'base_rate': round(base, 4),
        'gap': round(rate - base, 4),
        'p': round(p_raw, 5),
        '_p_raw': p_raw,                # the verdict gate reads the raw value
        '_gap_raw': rate - base,
        'binom_p': round(_binom_two_sided(int(np.sum(outcome[cond])), n_cond, base), 5),
        'per_era': per_era,
    })
    return out


def _binom_two_sided(k: int, n: int, p0: float) -> float:
    """The §3 exact-binomial DIAGNOSTIC (reported, never gating): the
    doubled smaller tail of Binom(n, p0) at k, capped at 1."""
    if n == 0 or not (0.0 < p0 < 1.0):
        return 1.0
    logp, log1p = math.log(p0), math.log(1.0 - p0)

    def logpmf(j: int) -> float:
        return (math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                + j * logp + (n - j) * log1p)
    lo = sum(math.exp(logpmf(j)) for j in range(0, k + 1))
    hi = sum(math.exp(logpmf(j)) for j in range(k, n + 1))
    return min(1.0, 2.0 * min(lo, hi))


def phase1_verdicts(primary: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The frozen §3 survival bar + the C1/C3 ambiguity rules, evaluated
    on the two primary tickers. Headlines: C1_strict_top, C2_top, C3_up
    (mirrors and variants never gate except through the ambiguity rule)."""
    def survives(label: str) -> bool:
        for t in PRIMARY_TICKERS:
            m = primary[t][label]
            # gate on the UNROUNDED p and gap: 5dp display rounding can
            # flip a boundary verdict (e.g. p_raw 0.0099990 -> 0.01)
            if m['_p_raw'] is None or m['_p_raw'] >= SURVIVAL_P:
                return False
            sign = 1 if m['_gap_raw'] > 0 else -1
            agree = sum(1 for e in m['per_era']
                        if e['rate'] is not None and e['n'] > 0
                        and (1 if e['rate'] - e['base_rate'] > 0 else -1) == sign)
            if agree < 2:
                return False
        return True

    v: dict[str, Any] = {}
    s_strict, s_loose = survives('C1_strict_top'), survives('C1_loose_top')
    v['C1'] = ('AMBIGUOUS' if s_strict != s_loose else
               'SURVIVED' if s_strict else 'CLOSED')
    v['C2'] = 'SURVIVED' if survives('C2_top') else 'CLOSED'
    s_head = survives('C3_up')
    s_var = (survives('C3_up_gapcont'), survives('C3_up_closeclose'))
    v['C3'] = ('AMBIGUOUS' if any(sv != s_head for sv in s_var) else
               'SURVIVED' if s_head else 'CLOSED')
    return v


# --------------------------------------------------------------- phase 2

def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        cs = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def wilder_rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder's RSI, frozen per plan §4: plain-mean seed over the first n
    gains/losses, then avg_t = ((n-1)*avg_{t-1} + x_t)/n. Indices < n are
    warm-up (NaN). All-loss windows read 0, all-gain 100."""
    out = np.full(len(close), np.nan)
    if len(close) <= n:
        return out
    diff = np.diff(close)
    gains = np.maximum(diff, 0.0)
    losses = np.maximum(-diff, 0.0)
    avg_g = float(np.mean(gains[:n]))
    avg_l = float(np.mean(losses[:n]))
    for i in range(n, len(close)):
        if i > n:
            avg_g = ((n - 1) * avg_g + gains[i - 1]) / n
            avg_l = ((n - 1) * avg_l + losses[i - 1]) / n
        if avg_l == 0.0:
            out[i] = 100.0 if avg_g > 0 else 0.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    return out


def rolling_max_prior(x: np.ndarray, n: int) -> np.ndarray:
    """max of x over the n sessions strictly before each index (NaN in
    warm-up). O(n·len) — fine at these sizes, and obviously correct."""
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = np.max(x[i - n:i])
    return out


def rolling_min_prior(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = np.min(x[i - n:i])
    return out


def signal_fires(d: dict[str, np.ndarray], signal: str, side: str) -> np.ndarray:
    """Fire-day indices for one frozen menu signal (§4), warm-up excluded."""
    c, h, low = d['close'], d['high'], d['low']
    n = len(c)
    fires = np.zeros(n, dtype=bool)
    if signal.startswith('CB-'):
        nb = int(signal.split('-')[1])
        if side == 'long':
            level = rolling_max_prior(h, nb)
            fires = c > level
        else:
            level = rolling_min_prior(low, nb)
            fires = c < level
        fires &= ~np.isnan(level)
    elif signal == 'MA-200':
        m = sma(c, 200)
        prev_ok = ~np.isnan(np.roll(m, 1))
        if side == 'long':
            fires = (np.roll(c, 1) <= np.roll(m, 1)) & (c > m)
        else:
            fires = (np.roll(c, 1) >= np.roll(m, 1)) & (c < m)
        fires &= ~np.isnan(m) & prev_ok
        fires[0] = False
    elif signal == 'GX':
        f, s = sma(c, 50), sma(c, 200)
        prev_ok = ~np.isnan(np.roll(s, 1))
        if side == 'long':
            fires = (np.roll(f, 1) <= np.roll(s, 1)) & (f > s)
        else:
            fires = (np.roll(f, 1) >= np.roll(s, 1)) & (f < s)
        fires &= ~np.isnan(s) & prev_ok
        fires[0] = False
    elif signal == 'RSI-30':
        r = wilder_rsi(c)
        rp = np.roll(r, 1)
        if side == 'long':
            fires = (rp < 30.0) & (r >= 30.0)
        else:
            fires = (rp > 70.0) & (r <= 70.0)
        fires &= ~np.isnan(r) & ~np.isnan(rp)
        fires[0] = False
    else:
        raise ValueError(f'unknown signal {signal}')
    return np.flatnonzero(fires)


def warmup_len(signal: str) -> int:
    if signal.startswith('CB-'):
        return int(signal.split('-')[1])
    return {'MA-200': 200, 'GX': 200, 'RSI-30': 15}[signal]


def build_trades(fires: Sequence[int], horizon: int, n_days: int) -> list[int]:
    """The §4 flat-only book: entries in fire order, lockout of H
    sessions, re-entry permitted at the exit close, end-of-span skip."""
    entries: list[int] = []
    next_ok = -1
    for i in fires:
        if i >= next_ok and i + horizon < n_days:
            entries.append(int(i))
            next_ok = i + horizon
    return entries


def cell_run(d: dict[str, np.ndarray], signal: str, side: str, horizon: int,
             ticker: str, b: int = B_RESAMPLES) -> dict[str, Any]:
    """One Phase-2 cell: the flat-only book, the two frozen baselines,
    and the one-sided add-one Monte-Carlo verdict (p <= 0.01 beats)."""
    c = d['close']
    n = len(c)
    fires = signal_fires(d, signal, side)
    entries = build_trades(fires, horizon, n)
    wu = warmup_len(signal)
    universe = np.arange(wu, n - horizon)
    ret_all = c[universe + horizon] / c[universe] - 1.0
    if side == 'short':
        ret_all = -ret_all
    win_all = (ret_all > 0).astype(float)

    # the frozen §2/§4 per-era reporting: base rate and trade count per panel
    u_eras = np.array([era_index(dt) for dt in d['dates'][universe]])
    e_eras = np.array([era_index(dt) for dt in d['dates'][entries]]) if entries else np.array([])
    per_era = []
    for j in range(len(ERAS)):
        in_e = u_eras == j
        per_era.append({
            'era': j,
            'n_trades': int(np.sum(e_eras == j)) if len(e_eras) else 0,
            'base_rate': (round(float(np.mean((ret_all[in_e] > 0))), 4)
                          if np.any(in_e) else None),
        })
    cell: dict[str, Any] = {
        'signal': signal, 'side': side, 'h': horizon, 'ticker': ticker,
        'n_trades': len(entries), 'n_fires': len(fires),
        'base_rate': round(float(np.mean(win_all)), 4) if len(universe) else None,
        'per_era': per_era,
    }
    if len(entries) < MIN_TRADES:
        cell.update({'verdict': 'NO-VERDICT', 'win_rate': (
            round(float(np.mean(win_all[np.searchsorted(universe, entries)])), 4)
            if entries else None)})
        return cell

    pos = np.searchsorted(universe, entries)
    wins = win_all[pos]
    rets = ret_all[pos]
    obs_rate = float(np.mean(wins))
    obs_mean = float(np.mean(rets))

    # the frozen sampler: shuffle the drawable days, accept greedily with
    # >= H spacing, until the cell's own trade count is reached
    rng = derived_rng(f'{signal}|{side}|{horizon}|{ticker}')
    n_u = len(universe)
    need = len(entries)
    null_rate = np.empty(b)
    null_mean = np.empty(b)
    for k in range(b):
        order = rng.permutation(n_u)
        blocked = np.zeros(n_u, dtype=bool)
        got = 0
        s_w = 0.0
        s_r = 0.0
        for j in order:
            if blocked[j]:
                continue
            blocked[max(0, j - horizon + 1):j + horizon] = True
            s_w += win_all[j]
            s_r += ret_all[j]
            got += 1
            if got == need:
                break
        assert got == need, f'null sampler exhausted: {cell}'
        null_rate[k] = s_w / need
        null_mean[k] = s_r / need
    p = addone_p_one_sided(null_rate, obs_rate)
    cell.update({
        'win_rate': round(obs_rate, 4),
        'mean_ret': round(obs_mean, 5),
        'null_rate_mean': round(float(np.mean(null_rate)), 4),
        'mean_ret_pctile': round(float(np.mean(null_mean < obs_mean)), 4),
        'p': round(p, 5),
        'verdict': 'BEATS' if p <= SURVIVAL_P else 'no',
    })
    return cell


def phase2_survival(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """The frozen §4 bar: >= 3 of the 8 primary long cells beat, spanning
    both primary tickers."""
    verdicts: dict[str, Any] = {}
    signals = sorted({c['signal'] for c in cells})
    for sig in signals:
        prim = [c for c in cells
                if c['signal'] == sig and c['side'] == 'long'
                and c['ticker'] in PRIMARY_TICKERS]
        beats = [c for c in prim if c.get('verdict') == 'BEATS']
        tickers = {c['ticker'] for c in beats}
        verdicts[sig] = {
            'primary_cells': len(prim), 'beating': len(beats),
            'survives': (len(beats) >= PHASE2_SURVIVAL_MIN_CELLS
                         and len(tickers) == len(PRIMARY_TICKERS)),
        }
    return verdicts


def turtle_soup(d: dict[str, np.ndarray]) -> dict[str, Any]:
    """The §4 by-product: raw CB-20 fire days; level fixed at entry (max
    high of the 20 sessions strictly before); fail = any close in
    t+1..t+5 strictly below the level. Verdicts frozen: modern-panel
    rate > 50%; decay = earliest < latest by one-sided exact test."""
    c, h = d['close'], d['high']
    n = len(c)
    fires = signal_fires(d, 'CB-20', 'long')
    per_era: dict[int, list[int]] = {j: [] for j in range(len(ERAS))}
    skipped_tail = 0
    for i in fires:
        if i + TURTLE_FAIL_WINDOW >= n:
            skipped_tail += 1
            continue
        level = float(np.max(h[i - 20:i]))
        fail = bool(np.any(c[i + 1:i + 1 + TURTLE_FAIL_WINDOW] < level))
        per_era[era_index(str(d['dates'][i]))].append(int(fail))
    rows = []
    for j in range(len(ERAS)):
        k, m = sum(per_era[j]), len(per_era[j])
        lo, hi = wilson_interval(k, m)
        rows.append({'era': j, 'n': m, 'fails': k,
                     'rate': round(k / m, 4) if m else None,
                     'wilson': (round(lo, 4), round(hi, 4))})
    k1, n1 = rows[0]['fails'], rows[0]['n']
    k3, n3 = rows[2]['fails'], rows[2]['n']
    decay_p = fisher_exact_one_sided(k1, n1, k3, n3) if n1 and n3 else None
    return {
        'per_era': rows, 'skipped_tail': skipped_tail,
        'modern_rate_above_half': bool(rows[2]['rate'] and rows[2]['rate'] > 0.5),
        'decay_p': round(decay_p, 5) if decay_p is not None else None,
        'decay_confirmed': bool(decay_p is not None and decay_p < SURVIVAL_P),
    }


# ------------------------------------------------------------------ runs

SIGNALS = ('CB-20', 'CB-40', 'CB-100', 'MA-200', 'GX', 'RSI-30')


def run_phase1(tickers: Sequence[str]) -> dict[str, Any]:
    per_ticker = {t: phase1_measurements(load_ohlc(t), t) for t in tickers}
    out: dict[str, Any] = {'measurements': per_ticker}
    if all(t in per_ticker for t in PRIMARY_TICKERS):
        out['verdicts'] = phase1_verdicts(
            {t: per_ticker[t]['measurements'] for t in PRIMARY_TICKERS})
    return out


def run_phase2(tickers: Sequence[str], sides: Sequence[str] = ('long', 'short'),
               progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    soup: dict[str, Any] = {}
    for t in tickers:
        d = load_ohlc(t)
        for sig in SIGNALS:
            for side in sides:
                for hz in HORIZONS:
                    cells.append(cell_run(d, sig, side, hz, t))
            if progress:
                progress(f'{t} {sig}')
        soup[t] = turtle_soup(d)
    return {'cells': cells, 'survival': phase2_survival(cells), 'turtle_soup': soup}


def _print_report(p1: dict[str, Any], p2: dict[str, Any]) -> None:
    print('== Phase 1 (conditional | base | gap | p | binom diag) ==')
    for t, tick in p1['measurements'].items():
        print(f"-- {t}  (degenerate-range days excluded: {tick['n_degenerate']}; "
              f"era starts {tick['era_starts']})")
        for label, m in tick['measurements'].items():
            if m['rate'] is None:
                print(f'  {label:<18} n={m["n"]}  (empty)')
                continue
            print(f'  {label:<18} n={m["n"]:>5}  {m["rate"]:.3f} | {m["base_rate"]:.3f} '
                  f'| {m["gap"]:+.3f} | p={m["p"]} | binom {m["binom_p"]}')
    if 'verdicts' in p1:
        print('verdicts:', p1['verdicts'])
    print('\n== Phase 2 ==')
    for c in p2['cells']:
        if c.get('verdict') == 'NO-VERDICT':
            print(f"  {c['ticker']:<5}{c['signal']:<8}{c['side']:<6}H{c['h']:<3}"
                  f"n={c['n_trades']:<4} NO-VERDICT")
        else:
            print(f"  {c['ticker']:<5}{c['signal']:<8}{c['side']:<6}H{c['h']:<3}"
                  f"n={c['n_trades']:<4}win {c['win_rate']:.3f} vs null "
                  f"{c['null_rate_mean']:.3f}  p={c['p']}  {c['verdict']}")
    print('survival:', p2['survival'])
    for t, s in p2['turtle_soup'].items():
        print(f'turtle soup {t}:', s)


if __name__ == '__main__':
    tickers = list(PRIMARY_TICKERS) + list(ROBUSTNESS_TICKERS)
    p1 = run_phase1(tickers)
    p2 = run_phase2(tickers, progress=lambda m: print(f'  ...{m}', file=sys.stderr))
    if '--json' in sys.argv:
        print(json.dumps({'phase1': p1, 'phase2': p2}, default=str))
    else:
        _print_report(p1, p2)
