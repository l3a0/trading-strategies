#!/usr/bin/env python3
"""
validate_dailies.py — option-chain data-hygiene validator and CHAIN_CLEAN_START proposer.

The repo excludes the 2008-2010 placeholder-greeks eras from its real-chain runs by
clipping each store at a per-ticker boundary (``CHAIN_CLEAN_START`` in
``real_cc_backtest.py``). Picking that boundary has been a manual read of the
validation battery. This script automates it.

It streams a ``{ticker}_option_dailies*.csv[.gz]`` store and, per trading day, measures
whether the covered-call ENTRY BAND — call rows with ``bid > 0`` and
``0.05 < delta < 0.60`` — contains at least one row whose mark sits inside ``[bid, ask]``
(a "clean" entry candidate). The placeholder eras fail this loudly and for the documented
reasons: marks are a constant ``0.01`` placeholder (wildly outside the quote), IVs sit on
the ``0.01488`` lattice, and deltas are a step function (``1.0`` ITM, jumping straight to
``~0.02`` OTM) that puts *nothing* in the entry band.

From the per-day stream it proposes a boundary — the first day past the placeholder era —
or **fails closed** ("UNVERIFIED, needs review") when the defect pattern does not resolve
into a clean cliff (defects too late, persistent, or the clean era unusable). A secondary
Black-Scholes delta reconstruction — spot inferred from put-call parity, which is
*greek-independent* and so survives the placeholder deltas — flags the smoothly-wrong case
the mark check alone would miss.

This is an onboarding tool, not a pinned/registered surface. The boundary it proposes is a
recommendation that a human confirms the first time a novel ticker is onboarded; the value
then lands in ``CHAIN_CLEAN_START``. The classifier and boundary scan are pure functions so
the synthetic + calibration tests in ``test_validate_dailies.py`` can pin their behaviour
(calibration target: the MSFT 2008-2016 backfill must reproduce ``2010-05-10``).

Usage:
    python validate_dailies.py GLD TLT XLE EEM
    python validate_dailies.py MSFT --dailies msft_option_dailies_2008_2016.csv
"""
from __future__ import annotations
from common.paths import data_path

import argparse
import csv
import gzip
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime

from engine.cc_backtest import bs_delta

# --- entry band and placeholder signatures (mirror the documented battery) ---
ENTRY_DELTA_LO = 0.05          # the covered-call delta entry band, matching run_real_cc_overlay
ENTRY_DELTA_HI = 0.60
PLACEHOLDER_IV = 0.01488       # the vendor placeholder IV base on the 2008-2010 lattice
LATTICE_STEP = 0.00976         # spacing of the placeholder IV lattice (0.01488 + k*0.00976)
MARK_TOL = 0.005               # a mark within [bid - tol, ask + tol] counts as inside the quote
RISK_FREE = 0.045              # for the secondary Black-Scholes delta reconstruction
BS_DELTA_TOL = 0.20            # a clean in-band delta off BS by more than this is "smoothly wrong"

# --- boundary-scan fail-closed thresholds ---
LEAD_WINDOW = 60               # a placeholder era is recognized only if this leading window is dense-defective
LEAD_DEFECT_FRAC = 0.50        # ...at or above this defective fraction (else early defects are isolated strays)
CLUSTER_GAP_DAYS = 30          # defective days >this many trading days apart end the initial placeholder cluster
DEFECT_DENSITY_EPS = 0.005     # <=0.5% of the post-clip tail may be defective (rare strays); above => pervasive
STRAY_MAX_FRAC = 0.02          # a clean-start store tolerates up to this fraction of isolated stray defects
MIN_USABLE_FRAC = 0.50         # the usable region must be at least this fraction usable days
MAX_BOUNDARY_FRAC = 0.60       # a boundary past this fraction of the span => placeholder era implausibly long
SCAN_MIN_DAYS = 40             # need at least this many post-boundary days to trust the densities


def _to_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _f(s: str) -> float:
    try:
        return float(s)
    except (ValueError, TypeError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Per-day classification
# --------------------------------------------------------------------------- #
@dataclass
class DayMetrics:
    """Hygiene metrics for one trading day's chain."""
    day: date
    n_rows: int = 0
    n_call_inband_raw: int = 0      # bid>0, 0.05<delta<0.60 (call wing)
    n_call_inband_clean: int = 0    # ...and mark inside [bid, ask]   -> usable entry candidates
    n_call_inband_defect: int = 0   # ...but mark outside [bid, ask]  -> dangerous placeholder-in-band rows
    mark_outside_frac: float = 0.0  # over all priced rows
    zero_bid_frac: float = 0.0
    lattice_iv_frac: float = 0.0    # fraction of rows on the placeholder IV lattice
    n_exp_near: int = 0             # distinct expirations with dte<=60 (weeklies vs monthlies-only)
    bs_flagged: int = 0             # clean in-band rows whose delta is off BS by > BS_DELTA_TOL
    bs_checked: int = 0             # clean in-band rows the BS reconstruction could score

    @property
    def usable(self) -> bool:
        """A trustworthy entry exists and no defective row sits in the band."""
        return self.n_call_inband_clean >= 1 and self.n_call_inband_defect == 0

    @property
    def defective(self) -> bool:
        """A dangerous placeholder row sits in the entry band, or the day is a pure step-function placeholder day."""
        if self.n_call_inband_defect >= 1:
            return True
        # step-function placeholder day: nothing lands in the band AND the chain is mostly lattice IVs
        return self.n_call_inband_raw == 0 and self.lattice_iv_frac >= 0.5 and self.n_rows >= 8


def _on_lattice(iv: float) -> bool:
    if not math.isfinite(iv) or iv <= 0:
        return False
    if abs(iv - PLACEHOLDER_IV) < 1e-6:
        return True
    k = round((iv - PLACEHOLDER_IV) / LATTICE_STEP)
    return k >= 0 and abs(PLACEHOLDER_IV + k * LATTICE_STEP - iv) < 1e-6


def _mark_inside(bid: float, ask: float, mark: float) -> bool:
    if not (math.isfinite(bid) and math.isfinite(ask) and math.isfinite(mark)):
        return False
    lo, hi = (bid, ask) if bid <= ask else (ask, bid)
    return lo - MARK_TOL <= mark <= hi + MARK_TOL


def _parity_spot(rows: list[dict]) -> float | None:
    """Infer spot from put-call parity (greek-independent): the strike where call_mid ~= put_mid.

    Uses the expiration with dte closest to 30 in [7, 90]. Returns None if the wings or
    common strikes are missing (e.g. calls-only stores)."""
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        if 7 <= r["dte"] <= 90:
            by_exp.setdefault(r["expiration"], []).append(r)
    if not by_exp:
        return None
    exp = min(by_exp, key=lambda e: abs(by_exp[e][0]["dte"] - 30))
    calls: dict[float, float] = {}
    puts: dict[float, float] = {}
    for r in by_exp[exp]:
        mid = (r["bid"] + r["ask"]) / 2.0
        if not math.isfinite(mid) or mid <= 0:
            continue
        (calls if r["delta"] > 0 else puts)[r["strike"]] = mid
    common = set(calls) & set(puts)
    if not common:
        return None
    return min(common, key=lambda k: abs(calls[k] - puts[k]))


def classify_day(day: date, rows: list[dict], *, spot: float | None, vol: float | None) -> DayMetrics:
    """Compute hygiene metrics for one day's chain (pure; testable on synthetic chains)."""
    m = DayMetrics(day=day, n_rows=len(rows))
    n_priced = mark_outside = zero_bid = lattice = 0
    exps_near: set[str] = set()
    clean_inband: list[dict] = []
    for r in rows:
        bid, ask, mark, dlt, iv = r["bid"], r["ask"], r["mark"], r["delta"], r["iv"]
        if math.isfinite(iv) and _on_lattice(iv):
            lattice += 1
        if r["dte"] <= 60:
            exps_near.add(r["expiration"])
        if math.isfinite(bid) and bid == 0:
            zero_bid += 1
        if math.isfinite(mark) and math.isfinite(bid) and math.isfinite(ask):
            n_priced += 1
            if not _mark_inside(bid, ask, mark):
                mark_outside += 1
        # call-wing entry band
        if dlt > 0 and math.isfinite(bid) and bid > 0 and ENTRY_DELTA_LO < dlt < ENTRY_DELTA_HI:
            m.n_call_inband_raw += 1
            if _mark_inside(bid, ask, mark):
                m.n_call_inband_clean += 1
                clean_inband.append(r)
            else:
                m.n_call_inband_defect += 1
    m.mark_outside_frac = mark_outside / n_priced if n_priced else 0.0
    m.zero_bid_frac = zero_bid / len(rows) if rows else 0.0
    m.lattice_iv_frac = lattice / len(rows) if rows else 0.0
    m.n_exp_near = len(exps_near)

    # secondary: reconstruct delta on the clean in-band rows and flag smoothly-wrong ones.
    # Spot from put-call parity (greek-independent, in unadjusted-strike space) is preferred over
    # the price-CSV close, which is dividend-adjusted and diverges from the strikes for payers
    # (TLT/XLE/EEM). Sigma is the row's own sane IV when available, so a *correctly* computed
    # vendor delta reproduces almost exactly and only inconsistent (placeholder) rows are flagged.
    parity = _parity_spot(rows)
    spot = parity if parity is not None else spot
    for r in clean_inband:
        iv = r["iv"]
        sigma = iv if (0.05 < iv < 2.0 and not _on_lattice(iv)) else vol
        if spot is None or sigma is None or sigma <= 0 or r["dte"] <= 0:
            continue
        try:
            bs = bs_delta(spot, r["strike"], r["dte"] / 365.0, RISK_FREE, sigma, "call")
        except (ValueError, ZeroDivisionError):
            continue
        m.bs_checked += 1
        if abs(r["delta"] - bs) > BS_DELTA_TOL:
            m.bs_flagged += 1
    return m


# --------------------------------------------------------------------------- #
# Boundary scan (fail-closed)
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    ticker: str
    status: str                       # "CLEAN" | "CLIP" | "UNVERIFIED"
    boundary: date | None = None
    reason: str = ""
    n_days: int = 0
    span: tuple[date, date] | None = None
    last_defect: date | None = None
    post_usable_frac: float = 0.0
    post_defect_frac: float = 0.0
    stray_defects: list[date] = field(default_factory=list)
    bs_flag_rate: float = 0.0


def propose_boundary(ticker: str, days: list[DayMetrics]) -> Verdict:
    """Propose CHAIN_CLEAN_START, or fail closed. Pure over the per-day metrics.

    The boundary is the repo's definition — the first trading day *past* the placeholder
    era. The era is the initial cluster of defective in-band days: defective days recurring
    within CLUSTER_GAP_DAYS trading days of each other, starting near the series head. The
    clip is the next trading day after that cluster's last day; isolated strays *after* a
    long clean gap are reported, not clipped to. Fails closed if no clean cliff exists, if
    the era runs implausibly long, if post-clip strays are pervasive, or if post-clip clean
    deltas fail the Black-Scholes reconstruction en masse."""
    days = sorted(days, key=lambda d: d.day)
    v = Verdict(ticker=ticker, status="UNVERIFIED", n_days=len(days))
    if not days:
        v.reason = "no trading days in store"
        return v
    v.span = (days[0].day, days[-1].day)
    n = len(days)
    defect_idx = [i for i, d in enumerate(days) if d.defective]

    if not defect_idx:
        v.status = "CLEAN"
        v.boundary = days[0].day
        v.post_usable_frac = sum(d.usable for d in days) / n
        v.bs_flag_rate = _bs_rate(days)
        v.reason = "no defective in-band days found; store is clean from the first day"
        if v.post_usable_frac < MIN_USABLE_FRAC:
            v.status = "UNVERIFIED"
            v.reason = (f"no placeholder era, but only {v.post_usable_frac:.0%} of days carry a clean "
                        f"in-band entry (< {MIN_USABLE_FRAC:.0%}) — store may be too thin to trust")
        return v

    v.last_defect = days[defect_idx[-1]].day

    # A placeholder era is recognized only if the store STARTS dense-defective. Otherwise the
    # defective days are isolated strays (the modern "tail of out-of-band marks" the midpoint
    # clamp repairs) — report them, never clip 4 clean years to the first one.
    lead = min(LEAD_WINDOW, n)
    lead_defect_frac = sum(1 for i in range(lead) if days[i].defective) / lead
    if lead_defect_frac < LEAD_DEFECT_FRAC:
        v.boundary = days[0].day
        v.stray_defects = [days[i].day for i in defect_idx]
        v.post_defect_frac = len(defect_idx) / n
        v.post_usable_frac = sum(d.usable for d in days) / n
        v.bs_flag_rate = _bs_rate(days)
        if v.post_defect_frac > STRAY_MAX_FRAC:
            v.status = "UNVERIFIED"
            v.reason = (f"no dense placeholder era at the head, but {len(defect_idx)} isolated defective days "
                        f"({v.post_defect_frac:.1%} > {STRAY_MAX_FRAC:.0%}) — defects pervasive; needs review")
        elif v.post_usable_frac < MIN_USABLE_FRAC:
            v.status = "UNVERIFIED"
            v.reason = (f"only {v.post_usable_frac:.0%} of days carry a clean in-band entry "
                        f"(< {MIN_USABLE_FRAC:.0%}) — store may be too thin to trust")
        elif v.bs_flag_rate > 0.02:
            v.status = "UNVERIFIED"
            v.reason = (f"{v.bs_flag_rate:.1%} of in-band deltas disagree with Black-Scholes (> 2%) — "
                        f"possible smoothly-wrong greeks; needs review")
        else:
            v.status = "CLEAN"
            v.reason = (f"no placeholder era at the head; {len(defect_idx)} isolated defective day(s) "
                        f"({v.post_defect_frac:.2%}) the midpoint clamp repairs — no clip needed")
        return v

    # dense early era: extend the initial cluster while defective days recur within the gap
    era_end = defect_idx[0]
    for i in defect_idx[1:]:
        if i - era_end <= CLUSTER_GAP_DAYS:
            era_end = i
        else:
            break

    era_end_day = days[era_end].day
    boundary_idx = era_end + 1
    if boundary_idx >= n - SCAN_MIN_DAYS:
        v.status = "UNVERIFIED"
        v.reason = (f"defective in-band days persist to {v.last_defect}, within {n - boundary_idx} days of the "
                    f"store's end — no usable clean tail to validate against; needs review")
        return v

    tail = days[boundary_idx:]
    v.boundary = tail[0].day
    v.post_usable_frac = sum(d.usable for d in tail) / len(tail)
    v.stray_defects = [d.day for d in tail if d.defective]
    v.post_defect_frac = len(v.stray_defects) / len(tail)
    v.bs_flag_rate = _bs_rate(tail)

    if era_end / n > MAX_BOUNDARY_FRAC:
        v.status = "UNVERIFIED"
        v.reason = (f"placeholder cluster runs to {v.last_defect}, {era_end / n:.0%} into the span "
                    f"(> {MAX_BOUNDARY_FRAC:.0%}) — era implausibly long, or defects pervasive")
        return v
    if v.post_defect_frac > DEFECT_DENSITY_EPS:
        v.status = "UNVERIFIED"
        v.reason = (f"clip at {v.boundary} still leaves {v.post_defect_frac:.1%} of the tail defective "
                    f"(> {DEFECT_DENSITY_EPS:.1%}) — defects are not a clean early cluster; needs review")
        return v
    if v.post_usable_frac < MIN_USABLE_FRAC:
        v.status = "UNVERIFIED"
        v.reason = (f"clip at {v.boundary} leaves a tail only {v.post_usable_frac:.0%} usable "
                    f"(< {MIN_USABLE_FRAC:.0%}) — clean era too thin to trust; needs review")
        return v
    if v.bs_flag_rate > 0.02:
        v.status = "UNVERIFIED"
        v.reason = (f"clip at {v.boundary} passes the mark check, but {v.bs_flag_rate:.1%} of post-clip "
                    f"in-band deltas disagree with Black-Scholes (> 2%) — possible smoothly-wrong greeks; "
                    f"needs review")
        return v

    v.status = "CLIP" if boundary_idx > 0 else "CLEAN"
    v.reason = (f"placeholder in-band cluster ends {era_end_day}; clip at {v.boundary} leaves a tail that is "
                f"{v.post_usable_frac:.0%} usable with {v.post_defect_frac:.2%} defective days")
    return v


def _bs_rate(days: list[DayMetrics]) -> float:
    checked = sum(d.bs_checked for d in days)
    flagged = sum(d.bs_flagged for d in days)
    return flagged / checked if checked else 0.0


# --------------------------------------------------------------------------- #
# Streaming + price loading
# --------------------------------------------------------------------------- #
def _open(path: str):
    return gzip.open(path, "rt", newline="") if path.endswith(".gz") else open(path, "rt", newline="")


def iter_day_chains(path: str):
    """Yield (date, rows) per trading day, streaming in file order.

    Each date's rows must be *contiguous* (true for a single fetch and for piecewise-sorted
    merges of disjoint spans, e.g. a backfill concatenated with a canonical store). The global
    order need not be sorted — ``propose_boundary`` sorts the per-day metrics. A date recurring
    in a non-contiguous block (a genuinely shuffled store) raises, since its chain would split."""
    with _open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        cur: str | None = None
        buf: list[dict] = []
        seen: set[str] = set()
        for raw in reader:
            d = raw[col["date"]]
            if d != cur:
                if buf and cur is not None:
                    yield _to_date(cur), buf
                if cur is not None:
                    seen.add(cur)
                if d in seen:
                    raise ValueError(f"{path}: date {d} recurs in a non-contiguous block; "
                                     f"sort the store by date first")
                cur = d
                buf = []
            buf.append({
                "expiration": raw[col["expiration"]],
                "dte": int(_f(raw[col["dte"]])) if raw[col["dte"]] else 0,
                "strike": _f(raw[col["strike"]]),
                "bid": _f(raw[col["bid"]]),
                "ask": _f(raw[col["ask"]]),
                "mark": _f(raw[col["mark"]]),
                "iv": _f(raw[col["implied_volatility"]]),
                "delta": _f(raw[col["delta"]]),
            })
        if buf and cur is not None:
            yield _to_date(cur), buf


def load_prices(ticker: str) -> dict[date, float]:
    """Load a yfinance close series ({ticker}_*_prices*.csv) for the BS spot/vol fallback.

    Parses yfinance's 3-row-header dump (Price,Close / Ticker,X / Date,). Prefers the
    unadjusted file (strikes are unadjusted) then the widest span. Returns {} if none."""
    import glob
    cands = (sorted(glob.glob(data_path(f"{ticker.lower()}_*_unadjusted.csv")))
             + sorted(glob.glob(data_path(f"{ticker.lower()}_*_prices.csv"))))
    for path in cands:
        try:
            out: dict[date, float] = {}
            with open(path, newline="") as f:
                for raw in csv.reader(f):
                    if not raw or len(raw) < 2:
                        continue
                    try:
                        out[_to_date(raw[0])] = float(raw[1])
                    except (ValueError, TypeError):
                        continue  # skip the 3 header rows
            if out:
                return out
        except OSError:
            continue
    return {}


def realized_vol(prices: dict[date, float], day: date, window: int = 21) -> float | None:
    """Trailing annualized close-to-close vol up to `day` (independent of the chain)."""
    if not prices:
        return None
    hist = sorted(p for p in prices if p <= day)
    if len(hist) < window + 1:
        return None
    closes = [prices[d] for d in hist[-(window + 1):]]
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes)) if closes[i - 1] > 0]
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def validate(ticker: str, path: str) -> tuple[Verdict, list[DayMetrics]]:
    prices = load_prices(ticker)
    days: list[DayMetrics] = []
    for day, rows in iter_day_chains(path):
        spot = prices.get(day)
        vol = realized_vol(prices, day)
        days.append(classify_day(day, rows, spot=spot, vol=vol))
    return propose_boundary(ticker, days), days


# --------------------------------------------------------------------------- #
# Price-vs-chain scale guard — a second hygiene axis. The entry-band check above
# validates the CHAIN; this checks that the unadjusted PRICE FILE is on the same
# (as-traded) scale as the strikes. A clean chain with a rescaled price file still
# blows up the delta-hedged overlays: yfinance split-adjusts Close even with
# auto_adjust=False, so a ticker that split (XLE, 2:1 on 2025-12-05) has a halved
# price history that mismatches its ~2x strikes, and the hedge runs on the wrong scale.
# --------------------------------------------------------------------------- #
SCALE_TOL = 0.12   # |median ATM-strike / price - 1| above this flags a split/scale mismatch


def scale_ratio(atm_by_day: dict, prices: dict, sample: int = 8) -> float | None:
    """Median (ATM call strike / underlying price) over `sample` days spread across the
    overlap. ~1.0 means the price file matches the as-traded strikes; ~2.0 is the
    signature of a 2:1 split mismatch. None if uncheckable. Keys may be date objects or
    'YYYY-MM-DD' strings as long as both dicts agree."""
    common = sorted(set(atm_by_day) & set(prices))
    if not common:
        return None
    n = min(sample, len(common))
    idxs = sorted({round(i * (len(common) - 1) / max(n - 1, 1)) for i in range(n)})
    ratios = [atm_by_day[common[i]] / prices[common[i]] for i in idxs if prices[common[i]] > 0]
    if not ratios:
        return None
    return sorted(ratios)[len(ratios) // 2]


def check_price_chain_scale(ticker: str, dailies_path: str | None = None,
                            tol: float = SCALE_TOL) -> dict:
    """Cross-check the unadjusted price file against the chain's strikes. Streams the
    chain for the ATM call strike (delta nearest 0.5) per day and compares to the price
    file; a median ratio off 1.0 by more than `tol` flags a split/adjustment mismatch —
    the scale the delta-hedged overlays would silently run the hedge on the wrong side
    of. Necessary alongside the entry-band check, which never looks at the price file."""
    path = dailies_path or _find_dailies(ticker)
    prices = load_prices(ticker)
    if not path or not prices:
        return {"ticker": ticker, "ok": True, "median_ratio": None,
                "detail": "no chain or unadjusted price file to cross-check"}
    atm_by_day: dict = {}
    for day, rows in iter_day_chains(path):
        calls = [r for r in rows if 0.0 < r["delta"] < 1.0]
        if calls:
            atm_by_day[day] = min(calls, key=lambda r: abs(r["delta"] - 0.5))["strike"]
    r = scale_ratio(atm_by_day, prices)
    ok = r is None or abs(r - 1.0) <= tol
    detail = ("scale matches the chain" if ok else
              f"price file is {r:.2f}x the as-traded strikes — likely a split/adjustment "
              f"mismatch; the overlays would run the hedge on the wrong price scale")
    return {"ticker": ticker, "ok": ok, "median_ratio": r, "detail": detail}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def format_report(v: Verdict, days: list[DayMetrics]) -> str:
    lines: list[str] = []
    lines.append(f"\n{'=' * 72}")
    lines.append(f"  {v.ticker}  —  {path_span(v)}")
    lines.append(f"{'=' * 72}")
    # per-year table
    lines.append(f"  {'year':4}  {'days':>4}  {'usable':>7}  {'defect':>6}  "
                 f"{'mk_out':>6}  {'lattice':>7}  {'0bid':>5}  {'exp<=60':>7}")
    by_year: dict[int, list[DayMetrics]] = {}
    for d in days:
        by_year.setdefault(d.day.year, []).append(d)
    for yr in sorted(by_year):
        ds = by_year[yr]
        nd = len(ds)
        usable = sum(d.usable for d in ds) / nd
        defect = sum(d.defective for d in ds)
        mk = sum(d.mark_outside_frac for d in ds) / nd
        lat = sum(d.lattice_iv_frac for d in ds) / nd
        zb = sum(d.zero_bid_frac for d in ds) / nd
        exp = sorted(d.n_exp_near for d in ds)[nd // 2]
        lines.append(f"  {yr:4}  {nd:>4}  {usable:>6.0%}  {defect:>6}  "
                     f"{mk:>6.1%}  {lat:>6.1%}  {zb:>4.0%}  {exp:>7}")
    # verdict
    lines.append(f"{'-' * 72}")
    tag = {"CLEAN": "CLEAN  (no clip needed)",
           "CLIP": f"CLIP   CHAIN_CLEAN_START['{v.ticker}'] = '{v.boundary}'",
           "UNVERIFIED": "UNVERIFIED  — needs human review"}[v.status]
    lines.append(f"  VERDICT: {tag}")
    lines.append(f"  {v.reason}")
    if v.last_defect:
        lines.append(f"  last defective in-band day: {v.last_defect}")
    if v.status in ("CLIP", "CLEAN"):
        lines.append(f"  post-clip: {v.post_usable_frac:.0%} usable, {v.post_defect_frac:.2%} defective, "
                     f"BS-disagree {v.bs_flag_rate:.2%}")
        if v.stray_defects:
            shown = ", ".join(str(d) for d in v.stray_defects[:8])
            more = "" if len(v.stray_defects) <= 8 else f" (+{len(v.stray_defects) - 8} more)"
            lines.append(f"  stray post-clip defective days to eyeball: {shown}{more}")
    return "\n".join(lines)


def path_span(v: Verdict) -> str:
    if not v.span:
        return "empty"
    return f"{v.span[0]} -> {v.span[1]}  ({v.n_days} trading days)"


def _find_dailies(ticker: str) -> str | None:
    import glob
    for pat in (f"{ticker.lower()}_option_dailies.csv", f"{ticker.lower()}_option_dailies.csv.gz",
                f"{ticker.lower()}_option_dailies*.csv", f"{ticker.lower()}_option_dailies*.csv.gz"):
        hits = sorted(glob.glob(data_path(pat)))
        if hits:
            return hits[0]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate option-dailies hygiene and propose CHAIN_CLEAN_START.")
    ap.add_argument("tickers", nargs="+", help="ticker symbols (e.g. GLD TLT XLE EEM)")
    ap.add_argument("--dailies", help="explicit dailies path (overrides the glob; single ticker)")
    args = ap.parse_args(argv)

    rc = 0
    for ticker in args.tickers:
        path = args.dailies if args.dailies else _find_dailies(ticker)
        if not path:
            print(f"[{ticker}] no option-dailies store found", file=sys.stderr)
            rc = 2
            continue
        verdict, days = validate(ticker, path)
        print(format_report(verdict, days))
        scale = check_price_chain_scale(ticker, path)
        if scale["median_ratio"] is not None:
            tag = "OK" if scale["ok"] else "SCALE MISMATCH"
            print(f"  price-vs-chain scale: ratio {scale['median_ratio']:.3f}  [{tag}] — {scale['detail']}")
            if not scale["ok"]:
                rc = max(rc, 1)
        if verdict.status == "UNVERIFIED":
            rc = max(rc, 1)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
