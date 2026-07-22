"""The cup-and-handle scan — the frozen docs/cup_handle_scan_plan.md build.

O'Neil's flagship pattern, detected by frozen mechanical rules across the
S&P 500 minute archive and judged at the calendar-day-cluster level
against stratified matched-count random-entry nulls.

The §2 DATA layer — archive access, split adjustment, the owner-signed
hygiene rulings and the reference cross-check — lives in
``pipeline/minute_archive.py``. It answers "which vendor rows can be
trusted?", a question about the data rather than about this hypothesis,
so it is shared rather than owned here; this module consumes it through
``load_clean_daily``. The plan doc's §2 rules are unchanged, only
relocated.

Layer map (every §ref is the plan doc):

- §2 data: ``pipeline.minute_archive`` (see that module's docstring).
- §3 detector: ``detect_cup_handle`` — the frozen iteration (every
  session past behavioral warm-up, chronological; handle lengths tried
  ascending 5..25; the FIRST passing window is THE detection; a
  rim-band failure rejects the window, never retries a second-best
  left rim; next candidate after a detection is t+25).
- §5 evaluation: ``build_trades`` (per-ticker flat-only, H lockout,
  end-of-span skip), ``build_clusters`` (same-entry-date pooling,
  equal-weight member returns), ``cluster_null_p`` (per-stratum
  matched-count draws with the frozen stream derivation
  ``f'{CUP_SEED}:{variant}|H{H}|{stratum}'``), the survival read.

The archive is personal and gitignored, so the result pins are
DATASET-GATED (skip in CI); the synthetic battery always runs. Nothing
here computes a return before the §4 validity gate has run — the
``__main__`` report prints detection counts and coverage FIRST and the
evaluation only behind ``--evaluate`` (the §10 step-4 switch).

Run:  python -m engine.cup_handle_scan [--tickers A,B,...] [--evaluate] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from pipeline.minute_archive import (  # noqa: F401 — re-exported for callers
    CLIFF_BAND,
    CROSSCHECK_AV_REFERENCE,
    LATE_START_FLAG,
    RESOLVED_CLIFFS,
    RETURN_BREAKS,
    TICKER_DROP_WINDOWS,
    TICKER_START_CLIPS,
    aggregate_daily,
    archive_path,
    cliff_flags,
    coverage_diagnostic,
    crosscheck_series,
    crosscheck_ticker,
    failed_tickers,
    fetch_reference,
    fetch_reference_av,
    load_clean_daily,
    load_splits,
    return_break_indices,
    run_crosscheck,
    split_adjust,
    universe,
    unsplit_reference,
)

CUP_SEED = 20260720
B_RESAMPLES = 10_000
HORIZONS = (5, 10, 15, 20, 60, 120)
HEADLINE_HORIZONS = (20, 60)
SURVIVAL_P = 0.01
MIN_CLUSTERS = 100
HANDLE_MIN, HANDLE_MAX = 5, 25
CUP_MIN, CUP_MAX = 35, 325
UPTREND_LOOKBACK = 90
UPTREND_MIN = 1.3
DEPTH_MIN, DEPTH_MAX = 0.12, 0.33
RIM_BAND = (0.85, 1.05)
HANDLE_DEPTH_MAX = 0.15
ROUNDNESS_MIN = 0.15
INTERIOR_TOL = 1.02
VOL_SURGE = 1.5                 # O'Neil via the committed Tharp notes, Loc 5559
VOL_AVG_WINDOW = 50
DEDUP_SKIP = 25
ERAS = (('1999-01-01', '2009-12-31'), ('2010-01-01', '2019-12-31'),
        ('2020-01-01', '2026-12-31'))


# -------------------------------------------------------------- §3 detector

def _ols_slope(y: np.ndarray) -> float:
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    return float(np.dot(x, y - y.mean()) / np.dot(x, x))


def _earliest_argmax(a: np.ndarray) -> int:
    return int(np.argmax(a))     # numpy argmax returns the FIRST maximum


def _earliest_argmin(a: np.ndarray) -> int:
    return int(np.argmin(a))


def _try_window(c: np.ndarray, v: np.ndarray, t: int, length: int,
                vol_surge: float = VOL_SURGE) -> dict[str, Any] | None:
    """Rules 1–6 for one (t, handle-length) candidate. Returns the recorded
    anatomy or None. All indices are session positions in the ticker's own
    series; any reference before index 0 rejects (behavioral warm-up)."""
    h0 = t - length
    if h0 < 0:
        return None
    w_close = c[h0:t]
    # rule 1 (geometry): handle high = highest close, earliest tie, index in
    # the first third (zero-based i with 3i < len)
    i = _earliest_argmax(w_close)
    if 3 * i >= length:
        return None
    r = h0 + i
    handle_high = c[r]
    if _ols_slope(w_close) > 0:
        return None
    if np.min(w_close) < (1.0 - HANDLE_DEPTH_MAX) * handle_high:
        return None
    # rule 2 (cup)
    lo_edge, hi_edge = r - CUP_MAX, r - CUP_MIN
    if lo_edge < 0:
        return None
    left = lo_edge + _earliest_argmax(c[lo_edge:hi_edge + 1])
    if not (RIM_BAND[0] * c[left] <= c[r] <= RIM_BAND[1] * c[left]):
        return None       # rim-band failure rejects the window; no 2nd-best left rim
    # left <= r - CUP_MIN always, so (left, r) is never degenerate
    b = left + 1 + _earliest_argmin(c[left + 1:r])
    rim_max = max(c[left], c[r])
    depth = (rim_max - c[b]) / rim_max
    if not (DEPTH_MIN <= depth <= DEPTH_MAX):
        return None
    pos = (b - left) / (r - left)
    if not (0.2 < pos < 0.8):
        return None
    if b + 1 < r and np.max(c[b + 1:r]) > INTERIOR_TOL * c[r]:
        return None
    # rule 3 (roundness, primary)
    threshold = c[b] + 0.25 * (rim_max - c[b])
    frac = float(np.mean(c[left:r + 1] <= threshold))
    if frac < ROUNDNESS_MIN:
        return None
    # rule 4 (handle in the upper half)
    if np.min(w_close) < c[b] + 0.5 * (rim_max - c[b]):
        return None
    # rule 5 (prior uptrend)
    if left - UPTREND_LOOKBACK < 0:
        return None
    if c[left] < UPTREND_MIN * np.min(c[left - UPTREND_LOOKBACK:left]):
        return None
    # rule 6 (trigger)
    if t - VOL_AVG_WINDOW < 0:
        return None
    if not (c[t] > handle_high):
        return None
    if v[t] < vol_surge * np.mean(v[t - VOL_AVG_WINDOW:t]):
        return None
    # rule 1's volume clause (needs the cup): handle volume below cup volume
    if np.mean(v[h0:t]) >= np.mean(v[left:r + 1]):
        return None
    return {'t': t, 'h0': h0, 'r': r, 'l': left, 'b': b,
            'depth': round(depth, 4), 'roundness': round(frac, 4)}


def detect_cup_handle(c: np.ndarray, v: np.ndarray,
                      *, use_volume_trigger: bool = True) -> list[dict[str, Any]]:
    """§3, the frozen iteration: t chronological; handle lengths ascending
    5..25; the FIRST passing window is THE detection; after a detection the
    next candidate is t + 25. ``use_volume_trigger=False`` is the §5
    ablation (rule 6's volume clause skipped; everything else identical)."""
    surge = VOL_SURGE if use_volume_trigger else 0.0
    out: list[dict[str, Any]] = []
    t = DETECT_FLOOR                # identical to the null's eligibility floor
    n = len(c)
    while t < n:
        hit = None
        for length in range(HANDLE_MIN, HANDLE_MAX + 1):
            hit = _try_window(c, v, t, length, vol_surge=surge)
            if hit is not None:
                break
        if hit is not None:
            out.append(hit)
            t += DEDUP_SKIP
        else:
            t += 1
    return out


# ------------------------------------------------------------ §5 evaluation

def build_trades(detections: Sequence[int], horizon: int, n_days: int) -> list[int]:
    """Per-ticker flat-only book (the harness convention): H-session
    lockout, re-entry at the exit close, end-of-span skip."""
    entries: list[int] = []
    next_ok = -1
    for t in detections:
        if t >= next_ok and t + horizon < n_days:
            entries.append(int(t))
            next_ok = t + horizon
    return entries


def stratum_of(date: str, horizon: int) -> str:
    y, m = date[:4], int(date[5:7])
    if horizon <= 20:
        return f'{y}-{m:02d}'
    if horizon == 60:
        return f'{y}Q{(m - 1) // 3 + 1}'
    return f'{y}H{1 if m <= 6 else 2}'


_NO_BREAKS = np.empty(0, dtype=int)


def _break_map(data: dict[str, dict[str, np.ndarray]],
               ) -> dict[str, np.ndarray]:
    """Per-ticker return-break session indices (see ``RETURN_BREAKS``)."""
    return {t: return_break_indices(t, v['dates']) for t, v in data.items()}


def spans_return_break(breaks: np.ndarray, t: int, horizon: int) -> bool:
    """Does the return window ``(t, t + horizon]`` cross a value
    detachment? Entry ON a break day is FINE — the detachment is already
    in ``close[t]``, so the forward return is honest. Only a break strictly
    after the entry and at or before the exit corrupts the measurement."""
    if not len(breaks):
        return False
    return bool(np.any((breaks > t) & (breaks <= t + horizon)))


def build_clusters(trades: dict[str, list[int]],
                   data: dict[str, dict[str, np.ndarray]],
                   horizon: int) -> tuple[list[dict[str, Any]], int]:
    """§5: all traded entries sharing one calendar date form one
    cluster-trade; return = equal-weight mean of member simple returns.

    Entries whose window spans a RETURN BREAK are dropped and counted, not
    scored — a spin-off or purge dividend inside the window would book a
    60-70% "loss" against a holder who was made whole. The identical test
    runs inside ``cluster_null_p``, so the real book and its null share one
    eligibility rule; applying it to only one side would bias the
    comparison in whichever direction the breaks happen to fall.

    The drop happens HERE rather than at detection time on purpose: the
    breakout itself is a real, correctly-detected price event. It is only
    the forward RETURN that is unmeasurable. (One consequence, disclosed:
    the flat-only lockout in ``build_trades`` has already been spent by a
    dropped entry, so a later entry inside that window stays suppressed.
    That is conservative and left alone rather than re-deriving the book.)

    Returns ``(clusters, n_dropped)``.
    """
    breaks = _break_map(data)
    by_date: dict[str, list[tuple[str, float]]] = {}
    dropped = 0
    for ticker, entries in trades.items():
        c = data[ticker]['close']
        dates = data[ticker]['dates']
        bk = breaks.get(ticker, np.empty(0, dtype=int))
        for t in entries:
            if spans_return_break(bk, t, horizon):
                dropped += 1
                continue
            ret = float(c[t + horizon] / c[t] - 1.0)
            by_date.setdefault(str(dates[t]), []).append((ticker, ret))
    clusters = [{'date': d, 'members': [m for m, _ in mem],
                 'ret': float(np.mean([r for _, r in mem])),
                 'stratum': stratum_of(d, horizon)}
                for d, mem in sorted(by_date.items())]
    return clusters, dropped


# The hard floor a detection's own guards imply: the cup window needs
# r >= CUP_MAX and the handle needs t >= r + HANDLE_MIN, so no detection
# can occur before session 330. The detector STARTS its iteration here and
# the null draws only from here — the two domains are identical by
# construction (the plan's behavioral warm-up rule governs; its "near 440"
# aside over-counted by adding the uptrend lookback, which bounds l, not t).
DETECT_FLOOR = CUP_MAX + HANDLE_MIN


def cluster_null_p(clusters: list[dict[str, Any]],
                   data: dict[str, dict[str, np.ndarray]], horizon: int,
                   variant: str = 'primary', b: int = B_RESAMPLES,
                   ) -> dict[str, Any]:
    """§5's matched-count cluster null. Per stratum with k clusters, each
    resample draws k distinct sessions from the union calendar of that
    stratum; drawn days (in draw order) pair with the stratum's real
    clusters sorted by (member count desc, date asc); each member's
    H-session return is taken from the drawn day in ITS OWN session
    indexing, dropped (and counted) if ineligible there."""
    if not clusters:
        return {'n_clusters': 0}
    date_index = {t: {str(d): i for i, d in enumerate(v['dates'])}
                  for t, v in data.items()}
    breaks = _break_map(data)
    strata: dict[str, list[dict[str, Any]]] = {}
    for cl in clusters:
        strata.setdefault(cl['stratum'], []).append(cl)
    all_days: dict[str, set[str]] = {}
    for t, v in data.items():
        for d in v['dates']:
            all_days.setdefault(stratum_of(str(d), horizon), set()).add(str(d))

    obs_rate = float(np.mean([c['ret'] > 0 for c in clusters]))
    obs_mean = float(np.mean([c['ret'] for c in clusters]))
    n_cl = len(clusters)
    # per-resample accumulators: wins, return sum, and the count of
    # SURVIVING (non-empty) null clusters — a fully-diluted null cluster is
    # renormalized away, never scored as a loss (the anti-conservative bias
    # the review caught); the empty count is reported beside the dilution
    win_acc = np.zeros(b)
    ret_acc = np.zeros(b)
    live_acc = np.zeros(b)
    diluted = 0
    empty_clusters = 0
    for stratum, cls in strata.items():
        days = sorted(all_days[stratum])
        rng = np.random.default_rng(int(hashlib.sha256(
            f'{CUP_SEED}:{variant}|H{horizon}|{stratum}'.encode()
        ).hexdigest()[:12], 16))
        ordered = sorted(cls, key=lambda c: (-len(c['members']), c['date']))
        k = len(ordered)
        for i in range(b):
            picks = rng.permutation(len(days))[:k]
            for cl, pi in zip(ordered, picks):
                d = days[pi]
                rets = []
                for m in cl['members']:
                    idx = date_index[m].get(d)
                    # Same three eligibility tests the real book faces,
                    # plus the return-break test — the null must be judged
                    # under identical rules or the comparison is biased.
                    if (idx is None or idx < DETECT_FLOOR
                            or idx + horizon >= len(data[m]['close'])
                            or spans_return_break(
                                breaks.get(m, _NO_BREAKS), idx, horizon)):
                        diluted += 1
                        continue
                    cm = data[m]['close']
                    rets.append(float(cm[idx + horizon] / cm[idx] - 1.0))
                if rets:
                    r = float(np.mean(rets))
                    win_acc[i] += r > 0
                    ret_acc[i] += r
                    live_acc[i] += 1
                else:
                    empty_clusters += 1
    live = np.maximum(live_acc, 1.0)
    null_rate = win_acc / live
    null_mean = ret_acc / live
    p = (1 + int(np.sum(null_rate >= obs_rate))) / (1 + b)
    return {
        'n_clusters': n_cl, 'win_rate': round(obs_rate, 4),
        'mean_ret': round(obs_mean, 5),
        'null_rate_mean': round(float(np.mean(null_rate)), 4),
        'mean_ret_pctile': round(float(np.mean(null_mean < obs_mean)), 4),
        'p': round(p, 5),
        'dilution_per_resample': round(diluted / b, 2),
        'empty_null_clusters_per_resample': round(empty_clusters / b, 3),
        'underpowered': n_cl < MIN_CLUSTERS,
    }


# ------------------------------------------------------------------- driver

def scan_ticker(ticker: str, splits: dict[str, list[tuple[str, float]]],
                ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None,
                           list[dict[str, Any]]]:
    """Clean data in, detections out. The hygiene lives in
    ``pipeline.minute_archive``; a ticker with an UNRESOLVED cliff flag
    yields no detections (§2 excludes it until a hand ruling clears it)."""
    adj, cov = load_clean_daily(ticker, splits)
    if adj is None:
        return None, None, []
    detections = ([] if cov['cliff_flags']
                  else detect_cup_handle(adj['close'], adj['volume']))
    return adj, cov, detections


def quadratic_roundness(c: np.ndarray, left: int, r: int) -> dict[str, Any]:
    """§3 rule 3's labeled variant (reported, never gating): a quadratic fit
    to the cup's normalized closes; R² = the fit's explained-variance share;
    vertex position as a fraction of the cup span."""
    y = c[left:r + 1]
    y = (y - y.min()) / (y.max() - y.min() if y.max() > y.min() else 1.0)
    x = np.linspace(0.0, 1.0, len(y))
    coef = np.polyfit(x, y, 2)
    fit = np.polyval(coef, x)
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    vertex = float(-coef[1] / (2 * coef[0])) if coef[0] != 0 else float('nan')
    return {'r2': round(r2, 4),
            'vertex': round(vertex, 4),
            'passes': bool(r2 >= 0.70 and 1 / 3 <= vertex <= 2 / 3)}


def _era_of(date: str) -> int:
    for j, (lo, hi) in enumerate(ERAS):
        if lo <= date <= hi:
            return j
    return -1


def run_scan(tickers: Sequence[str], evaluate: bool = False) -> dict[str, Any]:
    splits = load_splits()
    failed = failed_tickers()
    data: dict[str, dict[str, np.ndarray]] = {}
    coverage: list[dict[str, Any]] = []
    detections: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    excluded: dict[str, str] = {}
    skipped_in_flight: list[str] = []
    for t in tickers:
        if t in failed:
            excluded[t] = 'failed_fetch'
            continue
        try:
            adj, cov, dets = scan_ticker(t, splits)
        except (EOFError, FileNotFoundError, OSError):
            # the archive moved under us (the batch's rolling gzip):
            # skip this pass, the next increment picks it up complete
            skipped_in_flight.append(t)
            continue
        if adj is None:
            missing.append(t)
            continue
        if cov['cliff_flags']:
            excluded[t] = f"cliff:{cov['cliff_flags'][:3]}"
        data[t] = adj
        coverage.append(cov)
        detections[t] = dets
    # the union trading calendar (all scanned tickers) backs the §2
    # sessions-vs-calendar comparison
    calendar: set[str] = set()
    for v in data.values():
        calendar.update(str(d) for d in v['dates'])
    for cov in coverage:
        t = cov['ticker']
        first, last = cov['first'], cov['last']
        expected = sum(1 for day in calendar if first <= day <= last)
        cov['expected_sessions'] = expected
        cov['missing_sessions'] = expected - cov['sessions']
    # the frozen §4 statistic: total detections / sum of per-ticker span in
    # decades (calendar span, not session count)
    def _span_decades(cov: dict[str, Any]) -> float:
        y0, y1 = int(cov['first'][:4]), int(cov['last'][:4])
        m0, m1 = int(cov['first'][5:7]), int(cov['last'][5:7])
        return ((y1 - y0) * 12 + (m1 - m0)) / 120.0
    decades = sum(_span_decades(c) for c in coverage
                  if c['ticker'] not in excluded)
    total = sum(len(v) for t, v in detections.items() if t not in excluded)
    out: dict[str, Any] = {
        'tickers_scanned': len(data), 'missing': missing,
        'skipped_in_flight': skipped_in_flight,
        'excluded': excluded,
        'coverage': coverage,
        'total_detections': total,
        'rate_per_ticker_decade': round(total / decades, 3) if decades else None,
        'detections': {t: [d['t'] for d in v] for t, v in detections.items()},
        'quadratic_variant': {
            t: [quadratic_roundness(data[t]['close'], d['l'], d['r'])['passes']
                for d in v]
            for t, v in detections.items() if v},
    }
    if evaluate:
        pooled_data = {t: v for t, v in data.items() if t not in excluded}
        out['evaluation'] = {}
        for hz in HORIZONS:
            trades = {t: build_trades([d['t'] for d in detections[t]], hz,
                                      len(pooled_data[t]['close']))
                      for t in pooled_data}
            clusters, n_broken = build_clusters(trades, pooled_data, hz)
            e = cluster_null_p(clusters, pooled_data, hz)
            # §5 reported diagnostics: pooled per-trade win rate, member
            # counts, per-ticker and per-era splits, survivorship bracket
            breaks = _break_map(pooled_data)
            rets = []
            for t, entries in trades.items():
                cc = pooled_data[t]['close']
                bk = breaks.get(t, _NO_BREAKS)
                rets += [float(cc[i + hz] / cc[i] - 1.0) for i in entries
                         if not spans_return_break(bk, i, hz)]
            # every return-bearing number on this surface honours the guard
            e['return_break_trades_dropped'] = n_broken
            e['n_trades'] = len(rets)
            e['per_trade_win_rate'] = (round(float(np.mean(
                [r > 0 for r in rets])), 4) if rets else None)
            e['members_per_cluster'] = (round(float(np.mean(
                [len(cl['members']) for cl in clusters])), 2)
                if clusters else None)
            e['per_era_clusters'] = {
                j: sum(1 for cl in clusters if _era_of(cl['date']) == j)
                for j in range(len(ERAS))}
            e['per_ticker_trades'] = {t: len(v) for t, v in trades.items() if v}
            # the survivorship bracket: unconditional P(close[i+H] > close[i])
            wins = tot = 0
            for t, v in pooled_data.items():
                cc = v['close']
                seg = cc[DETECT_FLOOR:len(cc) - hz]
                if len(seg):
                    fwd = cc[DETECT_FLOOR + hz:]
                    ok = fwd > seg
                    # the bracket measures returns too, so it takes the
                    # same guard: an entry i is contaminated when a break
                    # j satisfies i < j <= i + hz, i.e. i in [j-hz, j-1]
                    keep = np.ones(len(seg), dtype=bool)
                    for j in breaks.get(t, _NO_BREAKS):
                        lo = max(0, j - hz - DETECT_FLOOR)
                        hi = min(len(seg), j - DETECT_FLOOR)
                        if hi > lo:
                            keep[lo:hi] = False
                    wins += int(np.sum(ok & keep))
                    tot += int(np.sum(keep))
            e['base_rate_bracket'] = round(wins / tot, 4) if tot else None
            out['evaluation'][f'H{hz}'] = e
        surv = all(
            not out['evaluation'][f'H{hz}'].get('underpowered', True)
            and out['evaluation'][f'H{hz}']['p'] <= SURVIVAL_P
            for hz in HEADLINE_HORIZONS)
        out['survives'] = bool(surv)
        # the §5 ablation: the volume trigger removed, its own null streams
        abl_det = {t: [d['t'] for d in detect_cup_handle(
            pooled_data[t]['close'], pooled_data[t]['volume'],
            use_volume_trigger=False)] for t in pooled_data}
        out['ablation_no_volume'] = {}
        for hz in HEADLINE_HORIZONS:
            trades = {t: build_trades(abl_det[t], hz,
                                      len(pooled_data[t]['close']))
                      for t in pooled_data}
            clusters, n_broken = build_clusters(trades, pooled_data, hz)
            abl = cluster_null_p(clusters, pooled_data, hz,
                                 variant='no_volume')
            abl['return_break_trades_dropped'] = n_broken
            out['ablation_no_volume'][f'H{hz}'] = abl
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tickers', default=None,
                    help='comma-separated subset (default: the committed universe)')
    ap.add_argument('--evaluate', action='store_true',
                    help='the §10 step-4 switch: compute returns and verdicts')
    ap.add_argument('--crosscheck', action='store_true',
                    help='reference cross-check battery (network, reporting '
                         'only — run_scan itself never touches the network)')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    tks = a.tickers.split(',') if a.tickers else universe()
    if a.crosscheck:
        rows = run_crosscheck(tks)
        flagged = [r for r in rows if r.get('flagged')]
        noref = [r['ticker'] for r in rows if r.get('flagged') is None]
        if a.json:
            print(json.dumps({'rows': rows}, default=str))
        else:
            print(f"crosschecked {len(rows)} tickers: {len(flagged)} flagged, "
                  f"{len(noref)} without reference data {noref}")
            for r in flagged:
                print(f"  FLAG {r['ticker']}: {r['mismatch_days']} mismatch "
                      f"days ({r['mismatch_first']}..{r['mismatch_last']}, "
                      f"max run {r['max_run']}), "
                      f"{r['unreferenced_days']} unreferenced")
            big_unref = [(r['ticker'], r['unreferenced_days']) for r in rows
                         if r.get('flagged') is False
                         and r['unreferenced_days'] > 25]
            if big_unref:
                print('  large unreferenced spans (predecessor-era smell):',
                      big_unref)
        raise SystemExit(0)
    res = run_scan(tks, evaluate=a.evaluate)
    if a.json:
        print(json.dumps(res, default=str))
    else:
        print(f"scanned {res['tickers_scanned']} tickers "
              f"({len(res['missing'])} missing archives)")
        print(f"detections: {res['total_detections']} "
              f"(rate {res['rate_per_ticker_decade']}/ticker-decade)")
        for c in res['coverage']:
            if c['late_start'] or c['cliff_flags']:
                print(f"  FLAG {c['ticker']}: first={c['first']} "
                      f"cliffs={c['cliff_flags'][:3]}")
        if 'evaluation' in res:
            for hz, e in res['evaluation'].items():
                print(hz, e)
            print('survives:', res['survives'])
