"""The wing-premium existence diagnostic (docs/wing_premium_diagnostic_plan.md).

A forecast-calibration STUDY, not a strategy: per non-overlapping ~30-DTE
cycle, compare what the 0.25-delta call wing CHARGED (implied vol backed out
of the quote midpoint — vendor deltas select strikes, vendor IVs are never a
measurement input) against what HAPPENED (realized upside semivolatility
over the option's life, plus a terminal breach read against the implied
N(d2) probability), conditioned on the wing spread's own point-in-time
percentile. One measurement adjudicates the three session gate proposals
(sell-rich / sell-cheap / sell-into-spike): all are claims that the wing
risk premium is STATE-DEPENDENT.

Frozen by the merged plan (#143) — the definitions here implement, never
amend: expiry = min |DTE-30| (ties toward smaller DTE); legs = vendor delta
closest to 0.50 (ATM) and 0.25 (wing) among bid>0 calls, priced at the quote
midpoint; IV via the structure engine's `implied_vol` at the leg's own
calendar tenor and the frozen rf=0.045; S_t = IV_wing - IV_atm; conditioning
percentile point-in-time over a trailing 756-day window (expanding from a
252-day minimum, data <= t only); non-overlapping cycles settling on the
last trading close on or before expiration; primary outcome
P = IV_wing - RSV+ with RSV+ = sqrt(252/n * sum(max(log r, 0)^2)); primary
statistic = Spearman rho of (percentile, P) judged against 1,000 seeded
circular shifts in cycle space; verdict LIVE iff BOTH QQQ and SPY reach
placebo-p <= 0.05 with the same sign, else H-flat. MSFT/NVDA are
exploratory contrast only.

Epistemic class: EXPLORATORY measurement (exploration-log family) — no
trades, no strategy sample, no idea-ledger rows, no e-value spent. A LIVE
read licenses a registration, never a trade.

Price conventions: spot and settlement closes are AS-TRADED
(`load_unadjusted_prices` — the strikes' own price space). Realized returns
come from the split-adjusted OHLC closes (returns are split-safe there);
the max-high path read rescales each day's split-adjusted high by the
as-traded/split-adjusted close ratio so highs land in strike space too
(factor 1.0 everywhere but NVDA's split era).

Usage:
    python -m search.wing_premium              # all four tickers + verdict
    python -m search.wing_premium --json PATH  # also dump the result dict
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from typing import Any, Sequence

from common.paths import data_path
from realchains.real_cc_backtest import (
    CHAIN_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    open_dailies,
)
from realchains.vol_premium import bs_price, implied_vol

WING_PLACEBO_SEED = 20260719
WING_RF = 0.045                     # the structure engine's frozen convention
TENOR_TARGET = 30                   # calendar DTE
ATM_DELTA = 0.50
WING_DELTA = 0.25
PCTL_WINDOW = 756                   # trailing trading days (plan §3.5)
PCTL_MIN = 252
ROUNDTRIP_TOL = 0.01                # $ — the plan §3.6 repricing guard
MIN_CYCLE_RETURNS = 5               # data-gap guard; skipped cycles are counted
N_SHIFTS = 1000
VERDICT_TICKERS = ('QQQ', 'SPY')
EXPLORATORY_TICKERS = ('MSFT', 'NVDA')

# Canonical call chains + era backfills, live hygiene boundaries (plan §2).
TICKER_STORES: dict[str, tuple[str, tuple[str, ...]]] = {
    'QQQ': ('qqq_option_dailies.csv', ('qqq_option_dailies_2011_2016.csv',)),
    'SPY': ('spy_option_dailies.csv', ()),
    'MSFT': ('msft_option_dailies.csv', ('msft_option_dailies_2008_2016.csv',)),
    'NVDA': ('nvda_option_dailies.csv', ()),
}


# ---- small numerics (no scipy in requirements) ----

def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean rank — the Spearman
    convention."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation = Pearson on average ranks."""
    if len(x) != len(y):
        raise ValueError(f'length mismatch {len(x)} vs {len(y)}')
    n = len(x)
    if n < 3:
        return 0.0
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def breach_prob_n_d2(spot: float, strike: float, years: float, rf: float,
                     sigma: float) -> float:
    """Risk-neutral P(S_T > K) = N(d2) at the backed-out IV (plan §4)."""
    d1 = (math.log(spot / strike) + (rf + 0.5 * sigma * sigma) * years) / (
        sigma * math.sqrt(years))
    return norm_cdf(d1 - sigma * math.sqrt(years))


# ---- signal extraction (plan §3) ----

def select_legs(day: dict[str, Any], target_dte: int = TENOR_TARGET,
                atm_delta: float = ATM_DELTA, wing_delta: float = WING_DELTA,
                ) -> tuple[tuple, tuple] | None:
    """The frozen leg rule: expiry minimizing |DTE - target| (ties toward the
    smaller DTE), then among that expiry's bid>0 calls the strikes whose
    vendor deltas sit closest to the ATM and wing targets. Returns the two
    candidate tuples (dte, delta, bid, ask, mid, expiration, strike, cid),
    or None when no expiry has two distinct qualifying strikes."""
    if not day['candidates']:
        return None
    # Plan §3 step order: the LISTED expiry nearest the target is chosen
    # first (ties toward the smaller DTE); the bid>0 hygiene applies WITHIN
    # it — a fully zero-bid nearest expiry fails the day, never falls
    # through to a farther one.
    best_dte = min({c[0] for c in day['candidates']},
                   key=lambda d: (abs(d - target_dte), d))
    cohort = [c for c in day['candidates'] if c[0] == best_dte and c[2] > 0]
    if not cohort:
        return None
    atm = min(cohort, key=lambda c: abs(c[1] - atm_delta))
    wing = min(cohort, key=lambda c: abs(c[1] - wing_delta))
    if atm[7] == wing[7]:            # one strike matched both targets
        return None
    return atm, wing


def _leg_iv(leg: tuple, spot: float, rf: float = WING_RF) -> float | None:
    """Quote-midpoint IV at the leg's own calendar tenor, with the plan §3.6
    round-trip guard: the backed-out IV must reprice the midpoint within
    ROUNDTRIP_TOL or the leg is treated as failed."""
    dte, _delta, bid, ask, _mid, _exp, strike, _cid = leg
    midpoint = (bid + ask) / 2
    years = dte / 365.0
    iv = implied_vol('call', midpoint, spot, strike, years, rf)
    if iv is None:
        return None
    if abs(bs_price('call', spot, strike, years, rf, iv) - midpoint) > ROUNDTRIP_TOL:
        return None
    return iv


def extract_signal(store: dict[str, dict[str, Any]], dates: list[str],
                   prices: list[float], wing_delta: float = WING_DELTA,
                   ) -> dict[str, dict[str, Any]]:
    """S_t for every trading day with two clean legs (plan §3): the daily
    series that feeds the conditioning percentile. Values carry everything
    the cycle sampler needs so no re-selection happens at entry time."""
    out: dict[str, dict[str, Any]] = {}
    for date, spot in zip(dates, prices):
        day = store.get(date)
        if day is None:
            continue
        pick = select_legs(day, wing_delta=wing_delta)
        if pick is None:
            continue
        atm, wing = pick
        atm_iv = _leg_iv(atm, spot)
        wing_iv = _leg_iv(wing, spot)
        if atm_iv is None or wing_iv is None:
            continue
        out[date] = {
            'spread': wing_iv - atm_iv,
            'atm_iv': atm_iv, 'wing_iv': wing_iv,
            'atm_delta': atm[1], 'wing_delta': wing[1],
            # Misses vs the VARIANT'S OWN targets (plan §9: the target
            # actually asked for, so the §8 wing variants self-report
            # honestly).
            'atm_miss': abs(atm[1] - ATM_DELTA),
            'wing_miss': abs(wing[1] - wing_delta),
            'wing_strike': wing[6], 'wing_dte': wing[0],
            'expiration': wing[5], 'atm_cid': atm[7], 'spot': spot,
        }
    return out


def pit_percentile(series: list[float], positions: list[int] | None = None,
                   window: int = PCTL_WINDOW,
                   min_obs: int = PCTL_MIN) -> list[float | None]:
    """Point-in-time percentile of each value within its trailing window
    (inclusive of the value itself — data <= t only, plan §3.5): the share
    of window values at or below it, mapped to [0, 1] as
    (count_le - 1) / (n - 1). None until min_obs S_t observations exist.
    Appending future values never changes an earlier assignment (pinned).

    `positions` are each observation's TRADING-DAY indices: the plan freezes
    the window in trading days, so on a ticker with signal-failure days the
    window admits only observations within the last `window` trading days —
    fewer than `window` observations. None (the synthetic default) treats
    the series as dense, positions = 0..n-1."""
    if positions is None:
        positions = list(range(len(series)))
    if len(positions) != len(series):
        raise ValueError(f'positions length {len(positions)} != series {len(series)}')
    out: list[float | None] = []
    lo = 0
    for i, v in enumerate(series):
        while positions[lo] <= positions[i] - window:
            lo += 1
        hist = series[lo:i + 1]
        if len(hist) < max(min_obs, 2):    # a percentile needs >= 2 values
            out.append(None)
            continue
        count_le = sum(1 for h in hist if h <= v)
        out.append((count_le - 1) / (len(hist) - 1))
    return out


# ---- outcomes and cycle sampling (plan §4) ----

def rsv_plus(log_returns: Sequence[float]) -> float:
    """Annualized realized UPSIDE semivolatility: sqrt(252/n * sum of squared
    positive daily log returns)."""
    n = len(log_returns)
    return math.sqrt(252.0 / n * sum(r * r for r in log_returns if r > 0))


def realized_vol(log_returns: Sequence[float]) -> float:
    """Annualized full realized vol (zero-mean convention) — the §8
    robustness variant of the outcome."""
    n = len(log_returns)
    return math.sqrt(252.0 / n * sum(r * r for r in log_returns))


def sample_cycles(signal: dict[str, dict[str, Any]],
                  pct_by_date: dict[str, float], trading_dates: list[str],
                  closes: dict[str, float], highs_scaled: dict[str, float],
                  outcome: str = 'rsv+') -> tuple[list[dict[str, Any]], int, int]:
    """Non-overlapping cycles (plan §4), walked over TRADING days: enter at
    the first day with a valid signal and percentile history; settle at the
    last trading close on or before the wing's expiration; re-enter at the
    next valid day strictly after settlement. Once the first valid entry has
    occurred (the warm-up boundary), every later trading day that is sought
    for entry and fails — no clean signal, no percentile, or a data gap —
    counts as a SKIP (§4/§9: attempts advance day-by-day and are reported).
    Returns (cycles, skips, tail_dropped):

    - tail_dropped counts entries whose expiration lies beyond the data end
      — the span is exhausted for that tenor, so the cycle is DROPPED, never
      measured on a truncated window (and excluded from the coverage
      denominator: not a data failure).
    - breach_maxhigh reads the day's split-scaled high over (entry, settle];
      the entry day's own high is deliberately excluded (the position exists
      only from the entry close), and a missing high reads as no-breach —
      both conventions disclosed here, inactive on the current OHLC files.
    """
    idx = {d: i for i, d in enumerate(trading_dates)}
    cycles: list[dict[str, Any]] = []
    skips = 0
    pending = 0        # signal-gap days awaiting an eventual attempt: they
                       # count as skips only if they DELAYED a later attempt
                       # (a cycle or a data-failure at a valid day); trailing
                       # gap days after the last possible cycle are span
                       # exhaustion, not availability failures, and are
                       # discarded so the §9 coverage rail measures what it
                       # claims to.
    tail_dropped = 0
    warm = False
    n_t = len(trading_dates)
    last_date = trading_dates[-1]
    ti = 0
    while ti < n_t:
        d = trading_dates[ti]
        sig = signal.get(d)
        pct = pct_by_date.get(d)
        if sig is None or pct is None:
            if warm:
                pending += 1
            ti += 1
            continue
        warm = True
        exp = sig['expiration']
        if exp > last_date:
            tail_dropped += 1
            ti += 1
            continue
        j = ti
        while j + 1 < n_t and trading_dates[j + 1] <= exp:
            j += 1
        if j <= ti:
            skips += pending + 1
            pending = 0
            ti += 1
            continue
        window = trading_dates[ti:j + 1]
        rets = []
        ok = True
        for a, b in zip(window, window[1:]):
            ca, cb = closes.get(a), closes.get(b)
            if ca is None or cb is None or ca <= 0 or cb <= 0:
                ok = False
                break
            rets.append(math.log(cb / ca))
        if not ok or len(rets) < MIN_CYCLE_RETURNS:
            skips += pending + 1
            pending = 0
            ti += 1
            continue
        vol_out = rsv_plus(rets) if outcome == 'rsv+' else realized_vol(rets)
        settle_date = window[-1]
        years = sig['wing_dte'] / 365.0
        cycles.append({
            'entry': d, 'settle': settle_date, 'pct': pct,
            'premium': sig['wing_iv'] - vol_out,
            'wing_iv': sig['wing_iv'], 'realized': vol_out,
            'implied_breach': breach_prob_n_d2(sig['spot'], sig['wing_strike'],
                                               years, WING_RF, sig['wing_iv']),
            'breach_terminal': closes[settle_date] > sig['wing_strike'],
            'breach_maxhigh': any(highs_scaled.get(x, 0.0) > sig['wing_strike']
                                  for x in window[1:]),
            'atm_miss': sig['atm_miss'],
            'wing_miss': sig['wing_miss'],
        })
        skips += pending
        pending = 0
        ti = idx[settle_date] + 1
    return cycles, skips, tail_dropped   # trailing `pending` gap days discarded


# ---- the statistic and its null (plan §5-§6) ----

def placebo_p(pcts: Sequence[float], premiums: Sequence[float],
              n_shifts: int = N_SHIFTS, seed: int = WING_PLACEBO_SEED,
              ) -> tuple[float, float]:
    """The primary rho and its two-sided placebo p: 1,000 circular shifts of
    the conditioning series against the outcome series in cycle-index space
    (offsets uniform on 1..N-1, seeded), p = (1 + #{|rho_s| >= |rho|}) / (1 + n)."""
    rho = spearman(pcts, premiums)
    n = len(pcts)
    if n < 3:
        return rho, 1.0
    rng = random.Random(seed)
    exceed = 0
    for _ in range(n_shifts):
        k = rng.randrange(1, n)
        shifted = [pcts[(i + k) % n] for i in range(n)]
        if abs(spearman(shifted, premiums)) >= abs(rho):
            exceed += 1
    return rho, (1 + exceed) / (1 + n_shifts)


def quintile_table(cycles: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Reporting only (the statistic is the continuous rho): Q1 cheapest
    wing ... Q5 richest, by the point-in-time percentile."""
    out: dict[str, dict[str, float]] = {}
    for q in range(5):
        lo, hi = q / 5, (q + 1) / 5
        rows = [c for c in cycles
                if (lo <= c['pct'] < hi) or (q == 4 and c['pct'] == 1.0)]
        if not rows:
            out[f'Q{q + 1}'] = {'n': 0}
            continue
        n = len(rows)
        out[f'Q{q + 1}'] = {
            'n': n,
            'mean_premium': round(sum(c['premium'] for c in rows) / n, 4),
            'implied_breach': round(sum(c['implied_breach'] for c in rows) / n, 4),
            'real_breach': round(sum(c['breach_terminal'] for c in rows) / n, 4),
            'maxhigh_breach': round(sum(c['breach_maxhigh'] for c in rows) / n, 4),
        }
    return out


# ---- the §9 sanity rails ----

def vendor_iv_sample(paths: Sequence[str], wanted: dict[str, str],
                     modern_start: str = '2016-06-06') -> dict[str, float]:
    """One streaming pass over the dailies CSV(s) collecting the vendor
    implied_volatility for {date: contractID} pairs on modern days — the §9
    ATM cross-check's independent column. Never a measurement input."""
    out: dict[str, float] = {}
    for p in paths:
        with open_dailies(p) as f:
            for row in csv.DictReader(f):
                d = row['date']
                if d < modern_start:
                    continue
                if wanted.get(d) == row['contractID']:
                    try:
                        out[d] = float(row['implied_volatility'])
                    except (TypeError, ValueError):
                        pass
    return out


def atm_cross_check(signal: dict[str, dict[str, Any]], store_paths: Sequence[str],
                    max_sample: int = 200) -> float:
    """Rank correlation of the extracted ATM IV against the vendor's, on a
    sample of clean modern days. Below 0.9 means OUR extraction is broken
    (plan §9) — the caller halts. Vendor IVs still carry no measurement
    weight; this is a bug detector."""
    modern = [d for d in sorted(signal) if d >= '2016-06-06']
    step = max(1, len(modern) // max_sample)
    sample_dates = modern[::step][:max_sample]
    wanted = {d: signal[d]['atm_cid'] for d in sample_dates}
    vendor = vendor_iv_sample(store_paths, wanted)
    pairs = [(signal[d]['atm_iv'], vendor[d]) for d in sample_dates
             if d in vendor and vendor[d] > 0.01]
    if len(pairs) < 30:
        raise RuntimeError(f'ATM cross-check starved: {len(pairs)} usable pairs')
    return spearman([a for a, _ in pairs], [b for _, b in pairs])


# ---- per-ticker orchestration ----

def load_ohlc(ticker: str) -> dict[str, tuple[float, float]]:
    """date -> (high, close) from the split-adjusted OHLC file."""
    out: dict[str, tuple[float, float]] = {}
    with open(data_path(f'{ticker.lower()}_daily_ohlc.csv')) as f:
        for row in csv.DictReader(f):
            out[row['date']] = (float(row['high']), float(row['close']))
    return out


def run_ticker(ticker: str, *, wing_delta: float = WING_DELTA,
               pctl_window: int = PCTL_WINDOW, outcome: str = 'rsv+',
               store: dict[str, dict[str, Any]] | None = None,
               cross_check: bool = True) -> dict[str, Any]:
    """One ticker end-to-end at one variant setting. Passing `store` lets the
    §8 robustness variants reuse a loaded store (the load dominates runtime)."""
    canonical, extras = TICKER_STORES[ticker]
    paths = [data_path(canonical)] + [data_path(e) for e in extras]
    if store is None:
        store = load_chain_store(paths[0], paths[1:],
                                 start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    tdates = [d for d, _ in pairs]
    unadj = dict(pairs)

    ohlc = load_ohlc(ticker)
    closes = {d: unadj[d] for d in tdates}
    # Highs into strike space: split-adjusted high x (as-traded / split-adjusted
    # close) — factor 1.0 except across a split (NVDA-era guard).
    highs_scaled = {d: ohlc[d][0] * (unadj[d] / ohlc[d][1])
                    for d in tdates if d in ohlc and ohlc[d][1] > 0}

    signal = extract_signal(store, tdates, [unadj[d] for d in tdates],
                            wing_delta=wing_delta)
    sdates = sorted(signal)
    tindex = {d: i for i, d in enumerate(tdates)}
    pcts = pit_percentile([signal[d]['spread'] for d in sdates],
                          positions=[tindex[d] for d in sdates],
                          window=pctl_window)
    pct_by_date = {d: p for d, p in zip(sdates, pcts) if p is not None}

    cycles, skips, tail_dropped = sample_cycles(
        signal, pct_by_date, tdates, closes, highs_scaled, outcome=outcome)
    attempted = len(cycles) + skips
    result: dict[str, Any] = {
        'ticker': ticker, 'wing_delta': wing_delta,
        'pctl_window': pctl_window, 'outcome': outcome,
        'signal_days': len(signal), 'trading_days': len(tdates),
        'cycles': len(cycles), 'skips': skips, 'tail_dropped': tail_dropped,
        'coverage': round(len(cycles) / attempted, 4) if attempted else 0.0,
    }
    if not cycles:
        result['rho'], result['placebo_p'] = 0.0, 1.0
        return result

    rho, p = placebo_p([c['pct'] for c in cycles],
                       [c['premium'] for c in cycles])
    prem = [c['premium'] for c in cycles]
    mean_p = sum(prem) / len(prem)
    sd_p = math.sqrt(sum((x - mean_p) ** 2 for x in prem) / max(len(prem) - 1, 1))
    med = sorted(c['atm_miss'] for c in cycles)[len(cycles) // 2]
    med_w = sorted(c['wing_miss'] for c in cycles)[len(cycles) // 2]
    quints = quintile_table(cycles)
    result.update({
        'rho': round(rho, 4), 'placebo_p': round(p, 4),
        'mean_premium': round(mean_p, 4), 'premium_sd': round(sd_p, 4),
        'quintiles': quints,
        'd_rich': round((quints['Q5'].get('mean_premium', 0.0) or 0.0)
                        - (quints['Q3'].get('mean_premium', 0.0) or 0.0), 4),
        'd_cheap': quints['Q1'].get('mean_premium'),
        'median_atm_miss': round(med, 4), 'median_wing_miss': round(med_w, 4),
        'demoted': bool(med > 0.05 or med_w > 0.05
                        or (len(cycles) / attempted) < 0.80),
        'span': [cycles[0]['entry'], cycles[-1]['settle']],
    })
    if cross_check:
        cc = atm_cross_check(signal, paths)
        if cc < 0.9:
            raise RuntimeError(f'{ticker}: ATM IV vs vendor rank corr {cc:.3f} '
                               '< 0.9 — extraction bug, halting (plan §9)')
        result['atm_cross_check'] = round(cc, 4)
    return result


def run_diagnostic(robustness: bool = True) -> dict[str, Any]:
    """All four tickers at the primary setting, the §5 verdict, and the §8
    non-verdict robustness grid on the verdict tickers."""
    out: dict[str, Any] = {'tickers': {}, 'robustness': {}}
    for t in (*VERDICT_TICKERS, *EXPLORATORY_TICKERS):
        canonical, extras = TICKER_STORES[t]
        store = load_chain_store(data_path(canonical),
                                 [data_path(e) for e in extras],
                                 start=CHAIN_CLEAN_START.get(t))
        print(f'[{t}] primary ...', flush=True)
        out['tickers'][t] = run_ticker(t, store=store)
        if robustness and t in VERDICT_TICKERS:
            rob: dict[str, Any] = {}
            for label, kw in (('wing20', {'wing_delta': 0.20}),
                              ('wing30', {'wing_delta': 0.30}),
                              ('window504', {'pctl_window': 504}),
                              ('rv', {'outcome': 'rv'})):
                print(f'[{t}] robustness {label} ...', flush=True)
                r = run_ticker(t, store=store, cross_check=False, **kw)
                rob[label] = {'rho': r['rho'], 'placebo_p': r['placebo_p'],
                              'cycles': r['cycles']}
            out['robustness'][t] = rob
        del store

    qqq, spy = out['tickers']['QQQ'], out['tickers']['SPY']
    # The frozen §5 rule, nothing else: both verdict tickers at p <= 0.05
    # with the same sign of rho.
    frozen_live = (qqq['placebo_p'] <= 0.05 and spy['placebo_p'] <= 0.05
                   and qqq['rho'] * spy['rho'] > 0)
    # The §9 rails gate RELIABILITY, not the rule: a demoted or zero-cycle
    # verdict ticker means the MEASUREMENT failed — that is never reported
    # as a substantive family-closing null.
    rails_ok = (not qqq.get('demoted') and not spy.get('demoted')
                and qqq['cycles'] > 0 and spy['cycles'] > 0)
    if not rails_ok:
        reading = ('MEASUREMENT UNRELIABLE: a verdict ticker failed the §9 '
                   'rails (demotion / zero cycles) — no verdict; this is not '
                   'a family-closing null')
    elif frozen_live:
        reading = 'LIVE'
    else:
        reading = ('H-flat: the conditioning family closes — rich-wing, '
                   'cheap-wing, and vol-spike gates are all declined without '
                   'spending strategy sample')
    out['verdict'] = {
        'live': frozen_live and rails_ok,
        'frozen_rule_live': frozen_live, 'rails_ok': rails_ok,
        'reading': reading,
        'qqq': {'rho': qqq['rho'], 'p': qqq['placebo_p']},
        'spy': {'rho': spy['rho'], 'p': spy['placebo_p']},
    }
    return out


def main() -> None:
    json_path = None
    if '--json' in sys.argv:
        i = sys.argv.index('--json')
        if i + 1 >= len(sys.argv):
            sys.exit('--json needs a path argument')
        json_path = sys.argv[i + 1]
    res = run_diagnostic()
    print(f"\n{'ticker':<7}{'cycles':>7}{'cov':>7}{'rho':>8}{'p':>8}"
          f"{'meanP':>8}{'sdP':>7}{'xchk':>7}{'demoted':>8}")
    for t, r in res['tickers'].items():
        print(f"{t:<7}{r['cycles']:>7}{r['coverage']:>7.2f}{r['rho']:>8.3f}"
              f"{r['placebo_p']:>8.3f}{r.get('mean_premium', 0):>8.3f}"
              f"{r.get('premium_sd', 0):>7.3f}{r.get('atm_cross_check', 0):>7.3f}"
              f"{str(r.get('demoted', '')):>8}")
        for q, row in r.get('quintiles', {}).items():
            print(f'   {q}: {row}')
    for t, rob in res['robustness'].items():
        print(f'robustness {t}: ' + '  '.join(
            f"{k}: rho {v['rho']:+.3f} p {v['placebo_p']:.3f} (n={v['cycles']})"
            for k, v in rob.items()))
    v = res['verdict']
    print(f"\nVERDICT: {'LIVE' if v['live'] else 'H-FLAT'} — "
          f"QQQ rho {v['qqq']['rho']:+.3f} p {v['qqq']['p']:.3f}; "
          f"SPY rho {v['spy']['rho']:+.3f} p {v['spy']['p']:.3f}")
    print(v['reading'])
    if json_path:
        with open(json_path, 'w') as f:
            json.dump(res, f, indent=1)
        print(f'JSON -> {json_path}')


if __name__ == '__main__':
    main()
