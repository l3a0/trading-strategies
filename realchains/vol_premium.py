"""Delta-NEUTRAL short-volatility overlay on real chains — the clean VRP isolator.

WHY THIS EXISTS
---------------
The covered-call experiments in this repo (run_real_cc_overlay) measured the
volatility risk premium (VRP) with an instrument the literature says harvests
its weakest slice: a single 0.25-delta CALL, in a covered-call structure that is
mostly equity beta, "hedged" by pinning net delta to the BUY-AND-HOLD level
rather than to zero. On real MSFT/QQQ chains the captured premium came back ~0
(Newey-West t -0.23 / +0.18), consistent with — not contradicting — a literature
whose robust premium is a whole-strip, delta-NEUTRAL, put-heavy, INDEX object.

This module builds the missing clean isolator: a daily delta-NEUTRAL short
option, the Bakshi-Kapadia (2003) "delta-hedged gains" construction. With net
delta held at ~0, the residual P&L is the gamma/vega P&L

    ~ 1/2 * Gamma * S^2 * (sigma_implied^2 - sigma_realized^2)

i.e. the variance risk premium itself, with the directional equity exposure
removed. A significantly POSITIVE mean P&L means the seller was paid for bearing
variance risk; ~0 means the premium isn't there at these strikes/names/era.

PHASES
------
Phase A (THIS FILE, runs today on existing call-only data): the CALL leg.
  Default ATM (~0.50 delta), where gamma/vega — and thus VRP exposure — peak.
  Set target_delta=0.25 to isolate the hedge-target change alone (net-zero vs
  buy-and-hold) holding the strike fixed to the covered-call runs.

Phase B (engine ready; UNBLOCKED — awaiting the one-shot run): the PUT leg. The
  equity-index VRP is concentrated in OTM PUTS (the skew / crash-insurance
  premium). The engine capability exists — `option_type='put'` selects via
  `select_put_entry` and hedges with SHORT stock (signed delta) — pinned by
  synthetic tests. Both prerequisites are now met: the registration merged
  (docs/prereg_vol_premium.md) and the put-inclusive data landed
  (download_option_dailies.py grew `--keep put`; the SPY-put and both-wing IWM
  daily chains were published to the data release). What remains is the prereg's
  one-shot execution — it runs once, on a deliberate start. The ATM straddle
  (two legs) remains a further extension.

EPISTEMIC STATUS
----------------
EXPLORATORY, sample-spending (MSFT/SPY/QQQ are already used). It can kill the
"a clean delta-neutral position would have surfaced the premium" hypothesis or
justify a registration; it is not itself a confirmatory verdict. The literature
review (docs/vol_premium.md) gives it a STRONG PRIOR of ~0 on these
single-name / one-index, post-2010, call-only inputs.

PINNED & AUDITED. The accounting was adversarially audited (engine bookkeeping
clean; one benchmark-base bug in short_vol_statistics found and fixed), and the
SPY headline is pinned by TestSpyShortVolRegression: the rate-invariant
Bakshi-Kapadia delta-hedged premium is +2.54 (Sharpe 0.52) and survives SPY's
realistic transaction costs. Phase B (the put side) is the remaining open work.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from common.stats import newey_west_summary
from realchains.real_cc_backtest import (
    COMMISSION_PER_SHARE,
    load_chain_store,
    load_unadjusted_prices,
    select_entry,
)


def select_put_entry(
    day: dict[str, Any], target_dte: int, target_delta: float
) -> tuple[int, float, float, float, float, str, float, str] | None:
    """Nearest-DTE expiration, then nearest-delta PUT — the put mirror of
    `select_entry` (real_cc_backtest.py). Puts carry NEGATIVE vendor delta, so
    `target_delta` is negative (e.g. -0.25 for the 25-delta put, the mirror of
    the call wing's +0.25). Band: `bid > 0` and `-0.60 < delta < -0.05` (the
    sign-flipped image of select_entry's call band). Returns the same
    candidate tuple shape select_entry does, or None.

    Implements docs/prereg_vol_premium.md §2.1's entry rule. On a put-inclusive
    store (the SPY-put and IWM daily chains, now published) this selects the
    target put; on the calls-only canonical stores it finds nothing (their
    deltas are all positive). Its mechanism is also exercised by synthetic put
    candidates in the tests.
    """
    cands = [c for c in day['candidates'] if c[2] > 0 and -0.60 < c[1] < -0.05]
    if not cands:
        return None
    best_dte = min({c[0] for c in cands}, key=lambda x: abs(x - target_dte))
    cohort = [c for c in cands if c[0] == best_dte]
    return min(cohort, key=lambda c: abs(c[1] - target_delta))


def select_straddle(
    day: dict[str, Any], target_dte: int,
    call_delta: float = 0.50, put_delta: float = -0.50,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """The two legs of an ATM short straddle on the SAME expiration: the
    ~call_delta call (via select_entry) and, at that call's expiration, the put
    nearest put_delta (bid > 0, in the put band) — the canonical Coval-Shumway /
    AQR variance harvester. Forcing one expiration keeps it a true straddle, not a
    diagonal.

    Pre-registered §7 SECONDARY of docs/prereg_vol_premium.md: REPORTED, NEVER
    PROMOTED — it cannot change the §5 primary (short-put) verdict. Returns the
    (call, put) candidate tuples (the shape select_entry returns) or None when
    either leg is unavailable at the nearest expiration.
    """
    call = select_entry(day, target_dte, call_delta)
    if call is None:
        return None
    expiration = call[5]
    puts = [c for c in day['candidates']
            if c[5] == expiration and c[2] > 0 and -0.60 < c[1] < -0.05]
    if not puts:
        return None
    put = min(puts, key=lambda c: abs(c[1] - put_delta))
    return call, put


def run_real_short_vol_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain short-vol overlay — short a delta-targeted call (or a put via option_type),
    delta-hedged, with optional early management (close_at_pct / manage_deep_itm).

    Stage B: a thin delegate to the single generic engine — run_real_structure_overlay
    under the 'short_vol' STRUCTURE_SPEC. Retained as the named entry point (run_registered_vrp,
    the campaign, the CLI); bit-identical to the prior hand-written loop, which the
    equivalence oracle pinned field-for-field before the swap."""
    return run_structure_via_spec('short_vol', dates, prices, store, params)


def short_vol_statistics(
    daily_equity: pd.DataFrame, capital: float,
    rf: float = 0.045, periods_per_year: int = 252,
) -> dict[str, Any]:
    """Significance of the MARKET-NEUTRAL daily vol-P&L, with the risk-free
    interest netted out on the SAME base the engine accrued it (cash).

    THE ACCOUNTING CHOICE (option b of the rf-base fix). The engine credits rf on
    the *cash* balance (`cash += cash * daily_rf`), and cash sits far below the
    deployed capital — the hedge ties it up, and it even goes negative on the days
    the hedge drains it. So the rf we subtract as the benchmark MUST be the rf the
    engine actually credited, day by day (recorded in the `rf_credit` column of
    daily_equity), NOT a flat rf/252 of the capital or of the grown equity.
    Subtracting a flat rf on a base LARGER than cash removes interest the account
    never earned and can crush or flip the sign of a genuinely positive premium:
    on real SPY 0.25Δ it knocked the gross +2.5 Newey-West t down to ~+0.2 with a
    capital ($100K) base, and to a wrong-signed negative with the grown-equity
    (~$129K) base. Netting the ACTUAL credit makes

        excess = d(equity) - rf_credit  ==  the pure gamma/vega vol-P&L

    so rf CANCELS and the verdict is rate-invariant: the same answer whether the
    engine charged rf=0 (gross) or rf=4.5%. The excess sums to
    summary['alpha_vs_cash'] UP TO the day-0 entry-spread mark: np.diff(eq) starts
    from eq[0], which is ALREADY struck at the entry bid/ask mid (the short was sold
    at the bid, day 0 is marked at the mid, less commission and the day-0 hedge
    half-spread), so the summed series omits that single day-0 cost that
    alpha_vs_cash includes. The clean identity is
    excess.sum()*capital == (eq[-1] - eq[0]) - interest_earned, leaving the gap
    alpha_vs_cash - excess.sum()*capital == eq[0] - capital — ONE entry spread no
    matter how many cycles run (only the first entry predates a diff), and a COST it
    omits, so the summed excess slightly flatters the vol-P&L, never deflates it
    (test_summed_excess_omits_day0_entry_spread). This measures the
    Bakshi-Kapadia delta-hedged gain — "was the seller paid for bearing variance
    risk?" — NOT "did the $100K beat T-bills?"; the latter would charge a financing
    penalty on the hedge sleeve, which is the very base mismatch this fix removes.

    Fallback: a hand-built equity curve with no `rf_credit` column subtracts a flat
    rf/periods_per_year (the legacy synthetic path); `rf` is used only there. The
    engine path ignores `rf` and uses the recorded credit. Same Newey-West HAC
    convention as compute_statistics.
    """
    eq = daily_equity['equity'].to_numpy(dtype=float)
    ret = np.diff(eq) / capital  # FIXED deployed-capital base (not grown equity)
    if 'rf_credit' in daily_equity.columns:
        # Net the ACTUAL per-day interest the engine credited (cash base). The
        # credit inside eq[k+1]-eq[k] is the one applied at the start of day k+1.
        rf_credit = daily_equity['rf_credit'].to_numpy(dtype=float)
        excess = ret - rf_credit[1:] / capital
    else:
        excess = ret - rf / periods_per_year  # legacy flat-rf fallback (synthetic)
    n = len(excess)
    if n < 2:
        raise ValueError(f'need >=2 daily observations, got {n}')

    # The naive/NW pair, Bartlett weights, auto-lag, and guards live in
    # common.stats.newey_west_summary — the single shared definition
    # (byte-identical to the block formerly inlined here).
    s = newey_west_summary(excess)

    ann_excess = s.mean * periods_per_year
    ann_total = float(np.mean(ret)) * periods_per_year
    ann_vol = math.sqrt(s.var * periods_per_year)
    sharpe = ann_excess / ann_vol if ann_vol > 0 else 0.0

    return {
        'n_days': n,
        'years_of_data': round(n / periods_per_year, 2),
        'ann_return_pct': round(ann_total * 100, 3),         # total, incl. rf
        'ann_excess_return_pct': round(ann_excess * 100, 3),  # over rf -- the alpha
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),                          # vol-P&L Sharpe (rf netted)
        't_stat_naive': round(s.t_naive, 2),
        't_stat_newey_west': round(s.t_newey_west, 2),       # tests vol-P&L > 0
        'nw_lag': s.lag,
        'passes_t_2': abs(s.t_newey_west) > 2.0,
        'mean_daily_pnl_dollars': round(float(np.mean(np.diff(eq))), 2),  # gross, incl. rf
        'mean_daily_excess_dollars': round(s.mean * capital, 2),  # vol-P&L, rf netted
    }


def run_real_straddle_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain ATM short-straddle overlay — short an ATM call + put, combined-delta-hedged,
    held to expiry.

    Stage B: a thin delegate to the single generic engine — run_real_structure_overlay
    under the 'straddle' STRUCTURE_SPEC. Retained as the named entry point (run_registered_vrp,
    the campaign, the CLI); bit-identical to the prior hand-written loop, which the
    equivalence oracle pinned field-for-field before the swap."""
    return run_structure_via_spec('straddle', dates, prices, store, params)


def select_iron_condor(
    day: dict[str, Any], target_dte: int,
    short_delta: float = 0.25, wing_delta: float = 0.10,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]] | None:
    """Four legs of a short IRON CONDOR at ONE expiration: short ~short_delta call +
    long ~wing_delta call (higher strike), short ~short_delta put + long ~wing_delta
    put (lower strike). Sells the inner strangle, buys the outer wings — a defined-risk
    short-vol structure. Returns (short_call, long_call, short_put, long_put) candidate
    tuples or None when any leg is unavailable at the nearest expiration.

    EXPLORATORY, not a registered instrument: a practical retail structure, not the
    delta-hedged-gain VRP isolator.
    """
    short_call = select_entry(day, target_dte, short_delta)
    if short_call is None:
        return None
    exp = short_call[5]
    cands = [c for c in day['candidates'] if c[5] == exp]
    puts = [c for c in cands if c[1] < 0]
    calls = [c for c in cands if c[1] > 0]
    in_band = [p for p in puts if p[2] > 0 and -0.60 < p[1] < -0.05]
    if not in_band:
        return None
    short_put = min(in_band, key=lambda c: abs(c[1] + short_delta))
    # Long wings: strictly further OTM than the shorts, and buyable (ask > 0).
    long_put = min([p for p in puts if p[6] < short_put[6] and p[3] > 0],
                   key=lambda c: abs(c[1] + wing_delta), default=None)
    long_call = min([c for c in calls if c[6] > short_call[6] and c[3] > 0],
                    key=lambda c: abs(c[1] - wing_delta), default=None)
    if long_put is None or long_call is None:
        return None
    return short_call, long_call, short_put, long_put


def select_credit_spread(
    day: dict[str, Any], target_dte: int,
    short_delta: float = 0.30, wing_delta: float = 0.10,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """The two PUT legs of a BULL PUT CREDIT SPREAD at ONE expiration: short a
    ~short_delta put (nearer the money) + long a ~wing_delta put (further OTM, the
    defined-risk wing). This is the PUT HALF of the iron condor — it reuses that
    selector's put-side band + wing logic (short_delta carried positive, matched
    against the negative vendor delta), just without the call legs. Returns
    (short_put, long_put) candidate tuples or None when either leg is unavailable.

    EXPLORATORY, not a registered instrument: a practical retail CARRY structure
    (theta-positive, defined-risk), not the delta-hedged-gain VRP isolator.
    """
    short_put = select_put_entry(day, target_dte, -short_delta)
    if short_put is None:
        return None
    exp = short_put[5]
    puts = [c for c in day['candidates'] if c[5] == exp and c[1] < 0]
    # Long wing: strictly further OTM than the short (lower strike), and buyable (ask > 0).
    long_put = min([p for p in puts if p[6] < short_put[6] and p[3] > 0],
                   key=lambda c: abs(c[1] + wing_delta), default=None)
    if long_put is None:
        return None
    return short_put, long_put


def select_call_credit_spread(
    day: dict[str, Any], target_dte: int,
    short_delta: float = 0.30, wing_delta: float = 0.10,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """The two CALL legs of a BEAR CALL CREDIT SPREAD at ONE expiration: short a
    ~short_delta call (nearer the money) + long a ~wing_delta call (further OTM at a
    strictly HIGHER strike, the defined-risk wing). This is the CALL HALF of the iron
    condor — the exact mirror of select_credit_spread's put half, using that selector's
    call-side band + wing logic. Returns (short_call, long_call) candidate tuples or
    None when either leg is unavailable.

    EXPLORATORY, not a registered instrument (Widening 5,
    docs/call_spread_widening_plan.md): a practical retail CARRY structure
    (theta-positive, defined-risk), not the delta-hedged-gain VRP isolator.
    """
    short_call = select_entry(day, target_dte, short_delta)
    if short_call is None:
        return None
    exp = short_call[5]
    calls = [c for c in day['candidates'] if c[5] == exp and c[1] > 0]
    # Long wing: strictly further OTM than the short (HIGHER strike), and buyable (ask > 0).
    long_call = min([c for c in calls if c[6] > short_call[6] and c[3] > 0],
                    key=lambda c: abs(c[1] - wing_delta), default=None)
    if long_call is None:
        return None
    return short_call, long_call


def select_calendar(
    day: dict[str, Any], near_dte: int = 30, far_dte: int = 60,
    target_delta: float = 0.50, min_gap_dte: int = 30,
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    """The two legs of a long CALL calendar across TWO expirations: a near-month call
    near `target_delta` (~ATM) via select_entry, and a far-month call at the SAME STRIKE
    on a LATER expiration whose DTE is at least `min_gap_dte` days beyond the near's.
    Returns (near_call, far_call) candidate tuples — the SHORT near + LONG far the
    `_legs_calendar` builder signs — or None when no qualifying later expiration carries
    that strike with a buyable ask.

    The far leg is matched by STRIKE, not delta (a same-strike calendar is the canonical
    TERM structure): the far call at the near's strike has more time value and more vega,
    so the spread is net LONG vega across the two expirations. The `min_gap_dte` floor is
    the TERM precondition — a far leg only a handful of days past the near has almost the
    same vega, leaving the structure vega-NEUTRAL rather than long. Measured on SPY: a
    far−near DTE gap below ~25 days reads mostly neutral, the [25,30) band is mixed, and
    at or above 30 days it reads LONG on every sampled entry — so the floor is 30, which
    makes net_vega='long' the engine's actual signature, not a sometimes-true label. This
    is the only selector that forces a SECOND, later expiration — every other structure
    pins one. EXPLORATORY, not a registered instrument: the first TERM-family widening,
    sample-spending."""
    near = select_entry(day, near_dte, target_delta)
    if near is None:
        return None
    near_dte_actual, near_exp, strike = near[0], near[5], near[6]
    # candidate tuple = (dte, delta, bid, ask, mid, expiration, strike, contractID)
    far_cands = [c for c in day['candidates']
                 if c[6] == strike and c[5] > near_exp and c[1] > 0 and c[3] > 0  # later, same K, buyable call
                 and c[0] - near_dte_actual >= min_gap_dte]                       # genuine term separation
    if not far_cands:
        return None
    far = min(far_cands, key=lambda c: abs(c[0] - far_dte))   # nearest-DTE to the far target
    return near, far


def run_real_iron_condor_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain iron-condor overlay — short a 0.25d strangle + long 0.10d wings, static
    (unhedged), held to expiry.

    Stage B: a thin delegate to the single generic engine — run_real_structure_overlay
    under the 'iron_condor' STRUCTURE_SPEC. Retained as the named entry point (run_registered_vrp,
    the campaign, the CLI); bit-identical to the prior hand-written loop, which the
    equivalence oracle pinned field-for-field before the swap."""
    return run_structure_via_spec('iron_condor', dates, prices, store, params)


# ============================================================================
# Generic multi-leg structure engine (Ring 1 / Stage A of the "big idea desk").
#
# The three named overlays above (short_vol / straddle / iron_condor) are SPECIAL CASES
# of this one loop and, post-Stage-B, thin DELEGATES to it (run_structure_via_spec): a single
# cash account, the per-day rf credit, the gap<=4 settlement, the mark
# equity = cash + hedge*price + sum(sign*mid)*shares, and the [date, equity, price, rf_credit]
# schema short_vol_statistics consumes. They differ only in three parameterized knobs: the entry
# guard, the hedge mode, and management. The unifying leg math (verified per overlay):
#   entry credit  = sum over legs of (-sign * entry_net)   [short: sell-comm; long: buy+comm]
#   settle cash   = sum over legs of ( sign * intrinsic)
#   mark          = cash + hedge*price + sum over legs of (sign * mid) * shares
#
# This is now THE engine (Stage B done): the prior ~515 lines of hand-written bodies were retired
# after the equivalence oracle pinned every summary field + the daily_equity series bit-for-bit;
# the registered/exploratory regressions carry those numbers forward through the delegates.
# ============================================================================

def _leg_intrinsic(leg: dict[str, Any], settle_price: float) -> float:
    """Per-leg expiry intrinsic: a call pays max(0, S-K), a put max(0, K-S)."""
    return max(0.0, (settle_price - leg['strike']) if leg['right'] == 'call'
               else (leg['strike'] - settle_price))


def _settle_price_at(expiration: str, date: str, price: float,
                     prev_date: str | None, prev_price: float | None) -> float:
    """The close an expiration settles against, given today's (date, price). On the
    expiration date itself it is today's close; for a Saturday-dated expiry (the
    pre-Feb-2015 era) it is the prior trading day's close, with the same `gap <= 4`
    sanity assert the single-expiration settlement always used — applied PER expiration
    so a calendar's near and far legs each settle against the right close. Factored out
    of the loop so the staggered (multi-expiration) path reuses it byte-for-byte; the
    single-expiration path calls it with the one structure expiration and is unchanged."""
    if date == expiration:
        return price
    assert prev_date is not None and prev_price is not None
    gap = (pd.Timestamp(expiration) - pd.Timestamp(prev_date)).days
    assert gap <= 4, (f'{gap} days between {prev_date} and expiration '
                      f'{expiration} — missing data?')
    return prev_price


def _legs_short_vol(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """One short option leg (call default; put if option_type='put', via a NEGATIVE
    target_delta) — the run_real_short_vol_overlay structure as a leg list."""
    dte = int(params.get('dte', 30))
    target_delta = float(params.get('target_delta', 0.50))
    fill = str(params.get('fill', 'bid_ask'))
    is_put = str(params.get('option_type', 'call')) == 'put'
    pick = (select_put_entry if is_put else select_entry)(day, dte, target_delta)
    if pick is None:
        return None
    _dte, _delta, bid, _ask, mid, expiration, strike, cid = pick
    sell = bid if fill == 'bid_ask' else mid
    return [{'sign': -1, 'right': 'put' if is_put else 'call', 'strike': strike,
             'contract': cid, 'entry_net': sell - COMMISSION_PER_SHARE,
             'mid': mid, 'delta': _delta, 'expiration': expiration}]


def _legs_straddle(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Short call + short put at the same expiry — the run_real_straddle_overlay structure."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    call_delta = float(params.get('call_delta', 0.50))
    put_delta = float(params.get('put_delta', -0.50))
    pick = select_straddle(day, dte, call_delta, put_delta)
    if pick is None:
        return None
    call, put = pick
    c_sell = call[2] if fill == 'bid_ask' else call[4]
    p_sell = put[2] if fill == 'bid_ask' else put[4]
    return [
        {'sign': -1, 'right': 'call', 'strike': call[6], 'contract': call[7],
         'entry_net': c_sell - COMMISSION_PER_SHARE, 'mid': call[4], 'delta': call[1],
         'expiration': call[5]},
        {'sign': -1, 'right': 'put', 'strike': put[6], 'contract': put[7],
         'entry_net': p_sell - COMMISSION_PER_SHARE, 'mid': put[4], 'delta': put[1],
         'expiration': call[5]},
    ]


def _legs_iron_condor(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Short 0.25d strangle + long 0.10d wings — the run_real_iron_condor_overlay structure.
    Leg order (sc, sp, lc, lp) is the recorded identity; shorts fill at bid, longs at ask."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    short_delta = float(params.get('short_delta', 0.25))
    wing_delta = float(params.get('wing_delta', 0.10))
    pick = select_iron_condor(day, dte, short_delta, wing_delta)
    if pick is None:
        return None
    sc, lc, sp, lp = pick
    sc_in = sc[2] if fill == 'bid_ask' else sc[4]
    sp_in = sp[2] if fill == 'bid_ask' else sp[4]
    lc_in = lc[3] if fill == 'bid_ask' else lc[4]
    lp_in = lp[3] if fill == 'bid_ask' else lp[4]
    return [
        {'sign': -1, 'right': 'call', 'strike': sc[6], 'contract': sc[7],
         'entry_net': sc_in - COMMISSION_PER_SHARE, 'mid': sc[4], 'delta': sc[1], 'expiration': sc[5]},
        {'sign': -1, 'right': 'put', 'strike': sp[6], 'contract': sp[7],
         'entry_net': sp_in - COMMISSION_PER_SHARE, 'mid': sp[4], 'delta': sp[1], 'expiration': sc[5]},
        {'sign': +1, 'right': 'call', 'strike': lc[6], 'contract': lc[7],
         'entry_net': lc_in + COMMISSION_PER_SHARE, 'mid': lc[4], 'delta': lc[1], 'expiration': sc[5]},
        {'sign': +1, 'right': 'put', 'strike': lp[6], 'contract': lp[7],
         'entry_net': lp_in + COMMISSION_PER_SHARE, 'mid': lp[4], 'delta': lp[1], 'expiration': sc[5]},
    ]


# --- per-overlay summary assembly (Stage B) ---------------------------------
# The generic engine emits RICH quantities under generic keys; each overlay's frozen summary
# has a DIFFERENT field set (short-vol echoes target_delta + carries total_hedge_cost/hedge_cost_bps;
# straddle drops target_delta; the static iron-condor drops total_hedge_cost/hedge_cost_bps too) and
# its own `num_*_sold` name. These builders reproduce each frozen dict EXACTLY from the quantities —
# byte-identical (verified field-by-field by the dataset-gated equivalence test). `p` is the merged
# params the engine ran with, so a param echo (target_delta) reads the same value the frozen did.
def _summary_short_vol(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    # The frozen overlay hardcodes 'num_calls_sold' even for the put wing (option_type='put'),
    # so byte-identical reproduction does too. (run_registered_vrp reads it via a
    # `.get('num_puts_sold', ...num_calls_sold)` fallback; the num_puts_sold branch is dead today.
    # Renaming it for the put side is a behavior change, out of Stage B's byte-identical scope.)
    return {
        'capital': q['capital'], 'num_contracts': q['num_contracts'],
        'target_delta': float(p.get('target_delta', 0.50)),
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_calls_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _summary_straddle(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {
        'capital': q['capital'], 'num_contracts': q['num_contracts'],
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_straddles_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _summary_iron_condor(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                                       # static: no total_hedge_cost / hedge_cost_bps
        'capital': q['capital'], 'num_contracts': q['num_contracts'],
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'num_condors_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _legs_strangle(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Short OTM strangle — short a +short_delta call + a -short_delta put at one expiry. The
    straddle's OTM cousin (a wider short-vol structure): select_straddle with a symmetric non-ATM
    target, so the same VARIANCE leg math (short gamma/vega, combined-delta-hedged), just struck
    out of the money. `short_delta` (default 0.25) sets both wings symmetrically."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    sd = float(params.get('short_delta', 0.25))
    pick = select_straddle(day, dte, sd, -sd)
    if pick is None:
        return None
    call, put = pick
    c_sell = call[2] if fill == 'bid_ask' else call[4]
    p_sell = put[2] if fill == 'bid_ask' else put[4]
    return [
        {'sign': -1, 'right': 'call', 'strike': call[6], 'contract': call[7],
         'entry_net': c_sell - COMMISSION_PER_SHARE, 'mid': call[4], 'delta': call[1],
         'expiration': call[5]},
        {'sign': -1, 'right': 'put', 'strike': put[6], 'contract': put[7],
         'entry_net': p_sell - COMMISSION_PER_SHARE, 'mid': put[4], 'delta': put[1],
         'expiration': call[5]},
    ]


def _legs_risk_reversal(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Bullish risk reversal — SHORT a -short_delta put + LONG a +short_delta call at one expiry.
    The SKEW family's first structure: it harvests the equity put-call skew (puts priced richer than
    equidistant calls) by SELLING the rich put wing and BUYING the cheap call wing, combined-delta-
    hedged so the residual is the skew, not direction. select_straddle picks the symmetric ±short_delta
    legs; the short put fills at bid, the long call at ask. Unlike the all-short VARIANCE structures
    this is MIXED-sign — net long delta (+~0.5), net ~0 vega, net short_rich skew."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    sd = float(params.get('short_delta', 0.25))
    pick = select_straddle(day, dte, sd, -sd)
    if pick is None:
        return None
    call, put = pick
    p_sell = put[2] if fill == 'bid_ask' else put[4]      # short the rich put — collect the bid
    c_buy = call[3] if fill == 'bid_ask' else call[4]     # long the cheap call — pay the ask
    return [
        {'sign': -1, 'right': 'put', 'strike': put[6], 'contract': put[7],
         'entry_net': p_sell - COMMISSION_PER_SHARE, 'mid': put[4], 'delta': put[1],
         'expiration': call[5]},
        {'sign': +1, 'right': 'call', 'strike': call[6], 'contract': call[7],
         'entry_net': c_buy + COMMISSION_PER_SHARE, 'mid': call[4], 'delta': call[1],
         'expiration': call[5]},
    ]


def _legs_credit_spread(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Bull PUT credit spread — SHORT a -short_delta put (nearer ATM) + LONG a -wing_delta put
    (further OTM, the defined-risk wing) at one expiry. The CARRY family's first structure: it
    collects a net CREDIT and is theta-positive, the put half of the iron condor (select_credit_spread
    reuses that selector's put-side band + wing logic). Combined-delta-hedged so the residual is the
    carry, not direction. The short put fills at bid, the long-put wing at ask. Net SHORT vega (the
    short leg sits nearer the money, where vega is larger), net LONG delta (short a put = long the
    underlying), and LONG_RICH skew (engine-verified: the long OTM wing sits on the steep part of the
    put skew, so it carries HIGHER IV than the nearer-ATM short — the iron-condor's put-wing read)."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    short_delta = float(params.get('short_delta', 0.30))
    wing_delta = float(params.get('wing_delta', 0.10))
    pick = select_credit_spread(day, dte, short_delta, wing_delta)
    if pick is None:
        return None
    sp, lp = pick
    sp_in = sp[2] if fill == 'bid_ask' else sp[4]      # short the near put — collect the bid
    lp_in = lp[3] if fill == 'bid_ask' else lp[4]      # long the wing — pay the ask
    return [
        {'sign': -1, 'right': 'put', 'strike': sp[6], 'contract': sp[7],
         'entry_net': sp_in - COMMISSION_PER_SHARE, 'mid': sp[4], 'delta': sp[1],
         'expiration': sp[5]},
        {'sign': +1, 'right': 'put', 'strike': lp[6], 'contract': lp[7],
         'entry_net': lp_in + COMMISSION_PER_SHARE, 'mid': lp[4], 'delta': lp[1],
         'expiration': sp[5]},
    ]


def _legs_call_credit_spread(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Bear CALL credit spread — SHORT a short_delta call (nearer ATM) + LONG a wing_delta call
    (further OTM at a HIGHER strike, the defined-risk wing) at one expiry. The CARRY family's
    call-side structure, the exact mirror of _legs_credit_spread: it collects a net CREDIT and is
    theta-positive, the call half of the iron condor. Combined-delta-hedged so the residual is the
    carry, not direction — and the hedge goes LONG stock (short a call = short the underlying),
    making this the grammar's first short-vega-AND-short-delta overlay. The short call fills at
    bid, the long-call wing at ask. net_skew is declared in STRUCTURE_GRAMMAR only as the engine
    measures it (the Widening-3 lesson; see docs/call_spread_widening_plan.md section 3)."""
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    short_delta = float(params.get('short_delta', 0.30))
    wing_delta = float(params.get('wing_delta', 0.10))
    pick = select_call_credit_spread(day, dte, short_delta, wing_delta)
    if pick is None:
        return None
    sc, lc = pick
    sc_in = sc[2] if fill == 'bid_ask' else sc[4]      # short the near call — collect the bid
    lc_in = lc[3] if fill == 'bid_ask' else lc[4]      # long the wing — pay the ask
    return [
        {'sign': -1, 'right': 'call', 'strike': sc[6], 'contract': sc[7],
         'entry_net': sc_in - COMMISSION_PER_SHARE, 'mid': sc[4], 'delta': sc[1],
         'expiration': sc[5]},
        {'sign': +1, 'right': 'call', 'strike': lc[6], 'contract': lc[7],
         'entry_net': lc_in + COMMISSION_PER_SHARE, 'mid': lc[4], 'delta': lc[1],
         'expiration': sc[5]},
    ]


def _summary_strangle(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                                       # same field set as the straddle (its OTM cousin)
        'capital': q['capital'], 'num_contracts': q['num_contracts'],
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_strangles_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _summary_risk_reversal(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                              # hedged two-leg structure (same shape as the strangle);
        'capital': q['capital'], 'num_contracts': q['num_contracts'],   # net premium can be a DEBIT
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],     # (long call > short put)
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_risk_reversals_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _summary_credit_spread(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                              # hedged two-leg structure (same shape as the strangle);
        'capital': q['capital'], 'num_contracts': q['num_contracts'],   # CARRY: a net CREDIT
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],     # (short put > long wing)
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_credit_spreads_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _summary_call_credit_spread(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                              # hedged two-leg structure (same shape as the strangle);
        'capital': q['capital'], 'num_contracts': q['num_contracts'],   # CARRY: a net CREDIT
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],     # (short call > long wing)
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_call_credit_spreads_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


def _legs_calendar(day: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Long CALL calendar across TWO expirations — SHORT a near-month ~ATM call + LONG a far-month
    call at the SAME strike. The TERM family's first structure: it harvests the term structure of
    implied vol by SELLING the near (faster-decaying) leg and BUYING the far (richer-vega) leg, so
    the spread is net LONG vega across two expirations (opposite-sign vega — the TERM axis). The
    legs carry DIFFERENT `expiration`s, which is what drives the engine's staggered settlement: the
    near leg settles at its expiry while the far leg lives on. select_calendar matches the far leg
    by strike (a true same-strike calendar). The short near fills at bid, the long far at ask; net
    premium is typically a DEBIT (the far leg costs more than the near credit)."""
    near_dte = int(params.get('near_dte', 30))
    far_dte = int(params.get('far_dte', 60))
    fill = str(params.get('fill', 'bid_ask'))
    pick = select_calendar(day, near_dte, far_dte, 0.50)
    if pick is None:
        return None
    near, far = pick
    n_sell = near[2] if fill == 'bid_ask' else near[4]     # short the near call — collect the bid
    f_buy = far[3] if fill == 'bid_ask' else far[4]        # long the far call — pay the ask
    return [
        {'sign': -1, 'right': 'call', 'strike': near[6], 'contract': near[7],
         'entry_net': n_sell - COMMISSION_PER_SHARE, 'mid': near[4], 'delta': near[1],
         'expiration': near[5]},                            # NEAR expiry — settles first
        {'sign': +1, 'right': 'call', 'strike': far[6], 'contract': far[7],
         'entry_net': f_buy + COMMISSION_PER_SHARE, 'mid': far[4], 'delta': far[1],
         'expiration': far[5]},                             # FAR expiry — lives on past the near
    ]


def _summary_calendar(q: dict[str, Any], p: dict[str, Any]) -> dict[str, Any]:
    return {                              # hedged TWO-expiration structure; net premium is a DEBIT
        'capital': q['capital'], 'num_contracts': q['num_contracts'],   # (long far > short near)
        'final_equity': q['final_equity'], 'net_pnl': q['net_pnl'],
        'alpha_vs_cash': q['alpha_vs_cash'], 'interest_earned': q['interest_earned'],
        'total_premium_collected': q['total_premium_collected'],
        'total_hedge_cost': q['total_hedge_cost'], 'hedge_cost_bps': q['hedge_cost_bps'],
        'num_calendars_sold': q['num_sold'], 'wins': q['wins'], 'losses': q['losses'],
        'win_rate': q['win_rate'], 'max_drawdown_pct': q['max_drawdown_pct'],
        'risk_free_rate': q['risk_free_rate'], 'cash': q['cash'],
    }


# The SIX structures as their generic-engine configs (selector + the three knobs) plus
# `defaults` — the per-overlay parameter defaults that differ from the generic's own (only
# the straddle's hedge_cost_bps=0.5 vs the generic/short-vol 1.0). Merged UNDER user params,
# exactly reproducing each frozen overlay's `params.get(..., default)`. `summary` reassembles the
# engine's rich quantities into the overlay's exact frozen field set (Stage B).
STRUCTURE_SPECS: dict[str, dict[str, Any]] = {
    'short_vol':   {'select': _legs_short_vol, 'entry_guard': 'each_short_positive',
                    'hedge_mode': 'per_leg_sign', 'management': 'early_close_single',
                    'defaults': {}, 'summary': _summary_short_vol},   # bps 1.0 = generic default
    'straddle':    {'select': _legs_straddle, 'entry_guard': 'each_short_positive',
                    'hedge_mode': 'combined', 'management': 'hold',
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_straddle},
    'iron_condor': {'select': _legs_iron_condor, 'entry_guard': 'net_positive',
                    'hedge_mode': 'none', 'management': 'hold',
                    'defaults': {}, 'summary': _summary_iron_condor},
    'strangle':    {'select': _legs_strangle, 'entry_guard': 'each_short_positive',
                    'hedge_mode': 'combined', 'management': 'hold',   # straddle config, OTM
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_strangle},
    'risk_reversal': {'select': _legs_risk_reversal, 'entry_guard': 'each_short_positive',
                    'hedge_mode': 'combined', 'management': 'hold',   # SKEW: short put + long call
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_risk_reversal},
    'credit_spread': {'select': _legs_credit_spread, 'entry_guard': 'net_positive',
                    'hedge_mode': 'combined', 'management': 'hold',   # CARRY: short near put + long wing
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_credit_spread},
    'call_credit_spread': {'select': _legs_call_credit_spread, 'entry_guard': 'net_positive',
                    'hedge_mode': 'combined', 'management': 'hold',   # CARRY: short near call + long wing
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_call_credit_spread},
    'calendar':    {'select': _legs_calendar, 'entry_guard': 'each_short_positive',
                    'hedge_mode': 'combined', 'management': 'hold',   # TERM: two expirations, staggered
                    'defaults': {'hedge_cost_bps': 0.5}, 'summary': _summary_calendar},
}


def run_structure_via_spec(name: str, dates: list[str], prices: list[float],
                           store: dict[str, dict[str, Any]], params: dict[str, Any],
                           ) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Run the generic engine under a named STRUCTURE_SPEC, merging the spec's per-overlay
    defaults UNDER the caller's params (so an unspecified knob falls back to the frozen overlay's
    default), then reassemble the engine's rich quantities into the overlay's exact frozen summary
    (Stage B). The single call site the frozen-overlay wrappers, the campaign, and the equivalence
    test use."""
    spec = STRUCTURE_SPECS[name]
    merged = {**spec['defaults'], **params}
    q, trades, eq = run_real_structure_overlay(
        dates, prices, store, merged,
        select=spec['select'], entry_guard=spec['entry_guard'],
        hedge_mode=spec['hedge_mode'], management=spec['management'])
    return spec['summary'](q, merged), trades, eq


def run_real_strangle_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain OTM short-strangle overlay — short a +short_delta call + a -short_delta put,
    combined-delta-hedged, held to expiry (the straddle's OTM cousin). The first grammar WIDENING:
    a new structure that is purely a STRUCTURE_SPEC + delegate, no engine change — the payoff of
    the Stage-B consolidation. Same VARIANCE leg math as the straddle."""
    return run_structure_via_spec('strangle', dates, prices, store, params)


def run_real_risk_reversal_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain bullish risk-reversal overlay — SHORT a -short_delta put + LONG a +short_delta
    call, combined-delta-hedged, held to expiry. The first NEW-FAMILY widening (SKEW, not VARIANCE):
    its edge is the put-call skew, harvested by selling the rich put wing and buying the cheap call
    wing. Mixed-sign (the first hedged structure that isn't all-short), so it exercises the engine's
    position-delta `combined` hedge in its general form."""
    return run_structure_via_spec('risk_reversal', dates, prices, store, params)


def run_real_credit_spread_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain bull-PUT-credit-spread overlay — SHORT a -short_delta put + LONG a -wing_delta put
    (further OTM), combined-delta-hedged, held to expiry. The third grammar WIDENING and the first
    CARRY-family structure: theta-positive, defined-risk, collecting a net credit (the put half of
    the iron condor). A pure STRUCTURE_SPEC + delegate, no engine change — single-expiration, so the
    Stage-B engine already handles it. Net SHORT vega, net LONG delta, long_rich skew (engine-verified)."""
    return run_structure_via_spec('credit_spread', dates, prices, store, params)


def run_real_call_credit_spread_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain bear-CALL-credit-spread overlay — SHORT a short_delta call + LONG a wing_delta
    call (further OTM, HIGHER strike), combined-delta-hedged, held to expiry. The fifth grammar
    WIDENING (docs/call_spread_widening_plan.md): the CARRY family's call side, the exact mirror
    of the put credit spread and the call half of the iron condor. A pure STRUCTURE_SPEC +
    delegate, no engine change — single-expiration, Stage-B handles it. Net SHORT vega, net
    SHORT delta (the combined hedge goes LONG stock — the grammar's first short-vega-AND-
    short-delta overlay); net_skew as engine-measured (STRUCTURE_GRAMMAR)."""
    return run_structure_via_spec('call_credit_spread', dates, prices, store, params)


def run_real_calendar_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Real-chain long-calendar overlay — SHORT a near-month ~ATM call + LONG a far-month call at
    the SAME strike, combined-delta-hedged, each leg held to ITS OWN expiry. The first TERM-family
    widening and the hardest one: it is the first structure with TWO distinct expirations, so it
    needs the engine's STAGGERED settlement (the near leg settles while the far leg lives on). Net
    LONG vega across the two expirations — the TERM axis. Every single-expiration overlay still
    takes the byte-identical scalar-expiration path; only this one exercises the new branch."""
    return run_structure_via_spec('calendar', dates, prices, store, params)


def run_real_structure_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
    *,
    select: Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]] | None],
    entry_guard: str = 'each_short_positive',
    hedge_mode: str = 'combined',
    management: str = 'hold',
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Generic multi-leg short-vol structure runner — the single loop the three frozen
    overlays are special cases of. A structure is a list of LEGS produced at entry by
    `select(day, params)`, each {sign(+1 long/-1 short), right, strike, contract,
    entry_net, mid, delta, expiration}. Returns (summary, trades, daily_equity): `trades` is one
    record per enter/settle/close (non-empty iff the structure traded — what a caller's must_trade
    / measurement_invalid guard keys off), and daily_equity uses the schema short_vol_statistics
    consumes.

    Parameterized differences from the shared skeleton:
      entry_guard  'each_short_positive' (every short leg's net premium > 0; short_vol /
                   straddle) | 'net_positive' (the net credit > 0; iron_condor)
      hedge_mode   'per_leg_sign' (clamp the single leg's delta to its sign range [0,1]
                   call / [-1,0] put; short_vol) | 'combined' (clamp the summed delta to
                   [-1,1]; straddle) | 'none' (static; iron_condor)
      management   'hold' (mark to expiry) | 'early_close_single' (short_vol's
                   close_at_pct / manage_deep_itm on the single leg)

    THE SOLE ENGINE (Stage B done): the three named overlays — run_real_short_vol_overlay /
    run_real_straddle_overlay / run_real_iron_condor_overlay — are now thin delegates to this loop
    via run_structure_via_spec, and run_registered_vrp + the campaign run through them. The prior
    hand-written bodies were retired after the equivalence oracle pinned every summary field +
    the equity series bit-for-bit; the registered/exploratory regressions now carry those numbers
    forward through the delegates.

    Drive this via `run_structure_via_spec` / STRUCTURE_SPECS — a BARE call reverts an
    unspecified knob to the GENERIC default, which for hedge_cost_bps is 1.0 (the straddle's
    frozen default is 0.5; the spec injects it). This `summary` carries the RICH quantities under
    generic keys (`num_sold`, alpha_vs_cash, win_rate, max_drawdown_pct, wins/losses,
    total_premium_collected); `run_structure_via_spec`'s per-overlay builder reassembles them into
    each named overlay's EXACT field set — byte-identical, pinned field-for-field by the equivalence
    oracle. One caveat the oracle does NOT cover: the `net_positive` entry credit is left-folded with
    COMMISSION baked into each leg's entry_net, a different float association than the frozen
    iron-condor's `(shorts)-(longs)-4*comm` — harmless today (it rounds away in equity; verified to
    never flip the `> 0` guard across the search tickers), but a swap must confirm it cannot flip at
    a net-credit-near-zero boundary, or match the frozen association there."""
    fill = str(params.get('fill', 'bid_ask'))
    capital = float(params.get('capital', 100_000))
    rf = float(params.get('risk_free_rate', 0.045))
    hedge_cost_bps = float(params.get('hedge_cost_bps', 1.0))
    close_at_pct = params.get('close_at_pct')
    manage_deep_itm = bool(params.get('manage_deep_itm', False))
    stop_loss_mult = params.get('stop_loss_mult')   # Gap E: close cost >= mult × entry credit
    exit_dte = params.get('exit_dte')               # Gap E: close N calendar days before expiry
    # Gap E dispatch (docs/van_tharp_gap_e.md): the general exit branch arms iff a new
    # knob is set, or close_at_pct is set on a 'hold' structure (previously a silent
    # no-op). Unarmed — every pinned caller — runs the pre-Gap-E code verbatim; armed
    # runs bypass the legacy single-leg block so the two paths can never double-fire.
    exits_armed = (stop_loss_mult is not None or exit_dte is not None
                   or (close_at_pct is not None and management == 'hold'))
    daily_rf = rf / 252.0

    initial_price = prices[0]
    num_contracts = int(capital // (initial_price * 100))
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts

    cash = capital
    hedge_shares = 0
    legs: list[dict[str, Any]] | None = None
    expiration: str | None = None
    interest_earned = 0.0
    total_hedge_cost = 0.0
    num_sold = 0
    total_premium_collected = 0.0
    wins = 0
    losses = 0
    entry_credit = 0.0           # the open structure's net credit per share (for its realized P&L)
    realized_settle_flow = 0.0   # near-leg settle flow already booked this cycle (staggered/calendar)
    worst_unrealized = 0.0       # Gap A (A2): running min of the open cycle's daily MTM P&L, dollars
    daily_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []   # one record per entry/settle/close — non-empty iff it traded
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)

        # 1. rf on yesterday's cash (recorded for rf-netting in short_vol_statistics).
        day_rf_credit = 0.0
        if i > 0:
            day_rf_credit = cash * daily_rf
            cash += day_rf_credit
            interest_earned += day_rf_credit

        # 2. entry / settlement / mark+manage (all cash flows).
        if legs is None:
            if day is not None:
                picked = select(day, params)
                if picked is not None:
                    if entry_guard == 'each_short_positive':
                        ok = all(leg['entry_net'] > 0 for leg in picked if leg['sign'] < 0)
                    else:  # net_positive
                        ok = sum(-leg['sign'] * leg['entry_net'] for leg in picked) > 0
                    if ok:
                        entry_credit = sum(-leg['sign'] * leg['entry_net'] for leg in picked)
                        cash += entry_credit * shares
                        realized_settle_flow = 0.0   # fresh cycle: no near-leg settle booked yet
                        worst_unrealized = 0.0       # fresh cycle: MAE tracks from this entry
                        legs = picked
                        # `expiration` is the structure's FINAL (latest) leg expiration — the
                        # sentinel for "the whole structure is now settled". For every
                        # single-expiration overlay this is picked[0]['expiration'] (all legs
                        # share it), so the settlement branch below is byte-identical to the
                        # scalar-expiration code it replaces. A calendar/diagonal carries TWO
                        # distinct expirations; the near leg settles first (the staggered branch),
                        # the far leg lives on until this final date.
                        expiration = max(leg['expiration'] for leg in picked)
                        num_sold += 1
                        total_premium_collected += entry_credit * shares
                        # `legs_detail` is the Gap A ledger's R input (per-share, like
                        # `credit`): wing widths for defined_max_loss, short-leg gross
                        # premium for the mixed-sign premium floor (common/trade_ledger.py).
                        trades.append({'date': date, 'action': 'enter',
                                       'legs': len(picked), 'credit': round(entry_credit, 4),
                                       'legs_detail': [
                                           {'sign': leg['sign'], 'right': leg['right'],
                                            'strike': leg['strike'],
                                            'entry_net': leg['entry_net'],
                                            'expiration': leg['expiration']}
                                           for leg in picked]})
        elif legs is not None and date >= min(leg['expiration'] for leg in legs) < expiration:
            # STAGGERED settlement (multi-expiration only — a calendar/diagonal). One or more
            # NEAR legs have reached their own expiration while a LATER leg still lives, so the
            # structure is NOT yet fully settled (date < `expiration`, the final leg's expiry).
            # Settle and remove only the due legs; the survivors keep marking and hedging. The
            # `< expiration` guard makes this branch UNREACHABLE for any single-expiration
            # structure (there min == expiration, so `date >= min < expiration` is false on the
            # settlement day), which is what preserves byte-identity for every existing overlay —
            # they fall straight through to the all-at-once branch below.
            survivors: list[dict[str, Any]] = []
            for leg in legs:
                if date >= leg['expiration']:
                    sp = _settle_price_at(leg['expiration'], date, price, prev_date, prev_price)
                    leg_flow = leg['sign'] * _leg_intrinsic(leg, sp)
                    cash += leg_flow * shares
                    realized_settle_flow += leg_flow   # rolled into the structure's win/loss at final
                    trades.append({'date': date, 'action': 'settle_leg',
                                   'right': leg['right'], 'strike': leg['strike'],
                                   'expiration': leg['expiration'],
                                   'pnl': round(leg_flow * shares, 2)})
                else:
                    survivors.append(leg)
            legs = survivors            # the far leg(s) live on; the structure stays open
        elif date >= expiration:
            settle_price = _settle_price_at(expiration, date, price, prev_date, prev_price)
            settle_flow = sum(leg['sign'] * _leg_intrinsic(leg, settle_price) for leg in legs)
            cash += settle_flow * shares
            # realized P&L = entry credit + EVERY leg's settlement (near legs already booked into
            # realized_settle_flow during the staggered branch; 0.0 for a single-expiration cycle,
            # so this is byte-identical there).
            structure_flow = settle_flow + realized_settle_flow
            wins, losses = ((wins + 1, losses) if (entry_credit + structure_flow) * shares >= 0
                            else (wins, losses + 1))
            trades.append({'date': date, 'action': 'settle',
                           'pnl': round((entry_credit + structure_flow) * shares, 2),
                           'mae': round(worst_unrealized, 2)})
            legs = None
            expiration = None
            realized_settle_flow = 0.0
        elif day is not None:
            for leg in legs:
                q = day['marks'].get(leg['contract'])
                if q is not None:
                    leg['mid'], leg['delta'] = q[2], q[3]
            if exits_armed and entry_credit > 0:
                # Gap E general exit branch (docs/van_tharp_gap_e.md). Per-trade arm rule:
                # only a positive booked entry credit gives the multiple-of-credit triggers
                # a well-defined reference (excludes the net-debit calendar structurally,
                # and skips a net-debit risk-reversal entry per trade). Triggers evaluate
                # only when EVERY leg has a live quote — the conservative generalization of
                # the single-leg q-is-not-None guard; carried marks never manufacture fills.
                quotes = [day['marks'].get(leg['contract']) for leg in legs]
                if all(q is not None for q in quotes):
                    # Ex-commission net close cost per share: buy shorts back at the ask,
                    # sell longs at the bid under bid_ask; mids under mid fill. Triggers
                    # compare ex-commission (the close_ref convention); the fill adds
                    # per-leg commission.
                    if fill == 'bid_ask':
                        close_ref = sum(q[1] if leg['sign'] < 0 else -q[0]
                                        for leg, q in zip(legs, quotes))
                    else:
                        close_ref = sum(q[2] if leg['sign'] < 0 else -q[2]
                                        for leg, q in zip(legs, quotes))
                    hit_target = (close_at_pct is not None
                                  and close_ref <= entry_credit * (1 - float(close_at_pct)))
                    hit_stop = (stop_loss_mult is not None
                                and close_ref >= entry_credit * float(stop_loss_mult))
                    hit_time = (exit_dte is not None
                                and (datetime.strptime(str(expiration), '%Y-%m-%d')
                                     - datetime.strptime(str(date), '%Y-%m-%d')).days
                                <= int(exit_dte))
                    if hit_target or hit_stop or hit_time:
                        # Same-day priority: target, then stop, then time (the CC precedent).
                        reason = 'target' if hit_target else 'stop' if hit_stop else 'time'
                        close_cost = close_ref + COMMISSION_PER_SHARE * len(legs)
                        cash -= close_cost * shares
                        # NOTE for the TERM/debit widening: this pnl omits
                        # realized_settle_flow (cf. the settle branch) — unreachable in
                        # v1, where a staggered (calendar) structure is net-debit and
                        # never arms. Fold it in before arming multi-expiration exits.
                        wins, losses = ((wins + 1, losses)
                                        if (entry_credit - close_cost) * shares >= 0
                                        else (wins, losses + 1))
                        trades.append({'date': date, 'action': 'close', 'reason': reason,
                                       'pnl': round((entry_credit - close_cost) * shares, 2),
                                       'mae': round(worst_unrealized, 2)})
                        legs = None
                        expiration = None
            elif not exits_armed and management == 'early_close_single':
                # An ARMED run bypasses the legacy block entirely (the spec's
                # takeover-for-the-run rule) — including its manage_deep_itm test,
                # a spec-sanctioned deferral (docs/van_tharp_gap_e.md, open
                # question 2). Unarmed runs — every pinned caller — evaluate this
                # block exactly as before Gap E.
                leg = legs[0]
                q = day['marks'].get(leg['contract'])
                if q is not None:
                    short_buy = q[1] if fill == 'bid_ask' else q[2]
                    hit = (close_at_pct is not None
                           and short_buy <= leg['entry_net'] * (1 - close_at_pct))
                    deep = manage_deep_itm and (leg['delta'] < -0.70 if leg['right'] == 'put'
                                                else leg['delta'] > 0.70)
                    if hit or deep:
                        buyback = short_buy + COMMISSION_PER_SHARE
                        cash -= buyback * shares
                        wins, losses = ((wins + 1, losses) if (entry_credit - buyback) * shares >= 0
                                        else (wins, losses + 1))   # closed early: credit - buyback
                        trades.append({'date': date, 'action': 'close',
                                       'pnl': round((entry_credit - buyback) * shares, 2),
                                       'mae': round(worst_unrealized, 2)})
                        legs = None
                        expiration = None

        # 3. delta hedge (mode-dependent); unwinds to 0 when flat.
        if legs is not None and hedge_mode != 'none':
            if hedge_mode == 'per_leg_sign':
                leg = legs[0]
                lo, hi = (-1.0, 0.0) if leg['right'] == 'put' else (0.0, 1.0)
                target_hedge = int(round(min(max(leg['delta'], lo), hi) * shares))
            else:  # combined: neutralize the net POSITION delta (−Σ sign·delta). For an all-short
                # structure this equals Σ delta (the old form) bit-for-bit, so the straddle/strangle
                # pins are unchanged; for a MIXED-sign reversal (short put + long call) it is the only
                # correct form — Σ delta would read ~0 and leave the +0.5 position delta unhedged.
                combined = -sum(leg['sign'] * leg['delta'] for leg in legs)
                target_hedge = int(round(min(max(combined, -1.0), 1.0) * shares))
        else:
            target_hedge = 0
        hedge_trade = target_hedge - hedge_shares
        if hedge_trade != 0:
            cash -= hedge_trade * price
            cost = abs(hedge_trade) * price * (hedge_cost_bps / 10_000.0)
            cash -= cost
            total_hedge_cost += cost
            hedge_shares = target_hedge

        # 4. mark.
        equity = cash + price * hedge_shares
        if legs is not None:
            structure_mark = sum(leg['sign'] * leg['mid'] for leg in legs)
            equity += structure_mark * shares
            # Gap A (A2): running MAE — the open cycle's daily MTM P&L is the entry
            # credit plus any near-leg settle flow already booked plus current marks.
            worst_unrealized = min(
                worst_unrealized,
                (entry_credit + realized_settle_flow + structure_mark) * shares,
            )
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price,
                           'rf_credit': day_rf_credit})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price', 'rf_credit'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    net_pnl = final_equity - capital
    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())
    # RICH quantities (generic keys). run_structure_via_spec's per-overlay `summary` builder
    # renames `num_sold` and trims/echoes to reproduce each frozen overlay's exact field set.
    summary = {
        'capital': round(capital, 2), 'num_contracts': num_contracts,
        'final_equity': round(final_equity, 2),
        'net_pnl': round(net_pnl, 2),
        'alpha_vs_cash': round(net_pnl - interest_earned, 2),
        'interest_earned': round(interest_earned, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'total_hedge_cost': round(total_hedge_cost, 2),
        'hedge_cost_bps': hedge_cost_bps,
        'num_sold': num_sold,
        'wins': wins, 'losses': losses,
        'win_rate': round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        'max_drawdown_pct': round(max_dd, 2),
        'risk_free_rate': rf, 'cash': round(capital, 2),
    }
    return summary, trades, daily_equity


# ============================================================================
# Black-Scholes greeks — the signature-vs-engine consistency check.
#
# The grammar's economic typing (edge_search.STRUCTURE_GRAMMAR) declares each overlay's
# premium family + a net-greek SIGNATURE. structure_greek_signature derives that signature
# from the ENGINE's actual entry legs (BS gamma/vega on the IV backed out of each leg's mid),
# so a structure that DECLARES short gamma/vega while the engine runs something long-vega is
# caught — turning the typing from a label into an enforcement. IV is per-leg (skew-aware);
# the net-greek SIGN is robust to the exact t/r (all legs share the one expiration), which is
# all the family direction needs.
# ============================================================================

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _bs_d1(spot: float, strike: float, years: float, rf: float, sigma: float) -> float:
    return (math.log(spot / strike) + (rf + 0.5 * sigma * sigma) * years) / (sigma * math.sqrt(years))


def bs_price(right: str, spot: float, strike: float, years: float, rf: float, sigma: float) -> float:
    """Black-Scholes price of a European call/put. Degenerate (years<=0 or sigma<=0) -> intrinsic."""
    if years <= 0 or sigma <= 0:
        return max(0.0, (spot - strike) if right == 'call' else (strike - spot))
    d1 = _bs_d1(spot, strike, years, rf, sigma)
    d2 = d1 - sigma * math.sqrt(years)
    disc = math.exp(-rf * years)
    if right == 'call':
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_vega(spot: float, strike: float, years: float, rf: float, sigma: float) -> float:
    """Black-Scholes vega (per 1.0 of vol), the same for a call or a put. Always >= 0."""
    if years <= 0 or sigma <= 0:
        return 0.0
    return spot * _norm_pdf(_bs_d1(spot, strike, years, rf, sigma)) * math.sqrt(years)


def bs_gamma(spot: float, strike: float, years: float, rf: float, sigma: float) -> float:
    """Black-Scholes gamma, the same for a call or a put. Always >= 0."""
    if years <= 0 or sigma <= 0:
        return 0.0
    return _norm_pdf(_bs_d1(spot, strike, years, rf, sigma)) / (spot * sigma * math.sqrt(years))


def implied_vol(right: str, price: float, spot: float, strike: float, years: float,
                rf: float, lo: float = 1e-4, hi: float = 5.0, tol: float = 1e-7) -> float | None:
    """Back the Black-Scholes implied vol out of an option mid by BISECTION (robust — no vega
    blow-up near the boundary). Returns None for a mark that has no reliable IV: a non-finite
    input (NaN/inf), years<=0, a price whose extrinsic value is within `tol` of intrinsic (the
    whole bisection residual would sit below tolerance and converge to a junk vol), or a price
    outside the [lo, hi]-vol bracket (a stale / arbitrage-violating mark)."""
    if not (math.isfinite(price) and math.isfinite(spot) and math.isfinite(strike)) or years <= 0:
        return None
    intrinsic = max(0.0, (spot - strike) if right == 'call' else (strike - spot))
    if price <= intrinsic + tol:        # extrinsic value below the price tolerance -> no IV
        return None
    f_lo = bs_price(right, spot, strike, years, rf, lo) - price
    f_hi = bs_price(right, spot, strike, years, rf, hi) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(right, spot, strike, years, rf, mid) - price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def structure_greek_signature(legs: list[dict[str, Any]], spot: float, years: float,
                              rf: float = 0.045, neutral_tol: float = 0.15,
                              skew_tol: float = 0.05,
                              entry_date: str | None = None) -> dict[str, Any]:
    """Derive a structure's SIGNATURE from the engine's entry legs — what the grammar's declared
    signature is checked against. Three robust economic axes (NOT net_gamma — see below):

      net_vega  (the VARIANCE/TERM axis): 'short' (net < 0) | 'long' (net > 0) | 'neutral'. A
                single-expiration short-vol structure is short vega; a long CALENDAR is LONG vega
                (the far leg outweighs the near).
      net_delta (the DIRECTION axis): 'short' | 'long' | 'neutral'. Uses the vendor delta the engine
                hedges on (call +, put -).
      net_skew  (the SKEW axis): 'short_rich' if the SHORT legs sit at higher IV than the LONG legs
                (you SOLD the rich wing — what a risk reversal does, harvesting the put-call skew),
                'long_rich' if the reverse (an iron-condor longs its richer OTM wings), 'flat' if
                there are no short OR no long legs (an all-short straddle has no asymmetry to read).

    net_vega/net_delta are classified 'neutral' when the legs OFFSET: |net| / Σ|per-leg| <
    `neutral_tol`. The SCALE-INVARIANT ratio separates the families — ~1.0 for a VARIANCE structure's
    reinforcing short legs, ~0 for an offsetting reversal. net_GAMMA is deliberately NOT a signature
    field: for offset-leg structures it is irreducibly fragile — the iron-condor's net gamma (short)
    and the risk-reversal's (long) overlap in MAGNITUDE (~0.26–0.37 vs ~0.17–0.34 of Σ|leg|) with
    opposite signs, so no tolerance classifies both cleanly. net_vega carries the vol-selling claim
    instead (gamma and vega align for these single-expiration structures). net_skew types the SKEW
    family by the EDGE itself rather than that fragile greek.

    PER-LEG TENOR (the TERM/calendar schema change). The single-expiration callers pass one `years`
    (all legs share it) and are byte-unchanged. A MULTI-expiration structure (the calendar) must back
    each leg's IV out at its OWN tenor and weight its vega by that tenor — a far leg has more time
    value and more vega than a near leg at the same strike, which is exactly the LONG-vega TERM edge.
    Pass `entry_date` for that: each leg's `years` is then (leg['expiration'] − entry_date)/365, so
    the near and far calls are priced on their own clocks. Without `entry_date` the scalar `years` is
    used for every leg (the single-expiration path, byte-identical).

    Raises if any leg's IV can't be implied (a stale mark) — the caller picks a clean entry. The
    family-direction SIGN is robust to the exact t/r per leg."""
    def _leg_years(leg: dict[str, Any]) -> float:
        if entry_date is None:
            return years
        return (pd.Timestamp(leg['expiration']) - pd.Timestamp(entry_date)).days / 365.0

    nets = {'net_vega': 0.0, 'net_delta': 0.0}
    mags = {'net_vega': 0.0, 'net_delta': 0.0}
    short_iv: list[float] = []
    long_iv: list[float] = []
    short_k: list[float] = []
    long_k: list[float] = []
    for leg in legs:
        ly = _leg_years(leg)
        iv = implied_vol(leg['right'], leg['mid'], spot, leg['strike'], ly, rf)
        if iv is None:
            raise ValueError(f"could not imply vol for leg {leg.get('contract')} "
                             f"(mid={leg['mid']}, K={leg['strike']}, S={spot})")
        per = {'net_vega': bs_vega(spot, leg['strike'], ly, rf, iv),
               'net_delta': leg['delta']}     # vendor delta (call +, put -) — what the engine hedges
        for k in nets:
            nets[k] += leg['sign'] * per[k]
            mags[k] += abs(per[k])
        (short_iv if leg['sign'] < 0 else long_iv).append(iv)
        (short_k if leg['sign'] < 0 else long_k).append(leg['strike'])

    def _classify(net: float, mag: float) -> str:
        if mag == 0 or abs(net) / mag < neutral_tol:
            return 'neutral'                  # the legs offset (a risk reversal's vega)
        return 'short' if net < 0 else 'long'

    def _skew() -> str:
        if not short_iv or not long_iv:
            return 'flat'                     # all-short (or all-long) — no short-vs-long asymmetry
        # net_skew is a WING asymmetry: it only means something when the short and long legs sit at
        # DIFFERENT strikes (the risk reversal's short put vs long call; the iron-condor's inner
        # shorts vs outer long wings). A same-strike spread across two EXPIRATIONS (the calendar) has
        # no wing to be asymmetric about — its short-vs-long IV gap is the TERM-STRUCTURE slope, not
        # skew — so it reads 'flat'. Without this guard the calendar's slope would masquerade as a
        # skew edge (short_rich on an inverted term structure), mis-typing a TERM structure as SKEW.
        if set(short_k) == set(long_k):
            return 'flat'
        s, lo = sum(short_iv) / len(short_iv), sum(long_iv) / len(long_iv)
        rel = (s - lo) / ((s + lo) / 2)
        return 'short_rich' if rel > skew_tol else 'long_rich' if rel < -skew_tol else 'flat'
    sig = {'legs': len(legs), 'expirations': len({leg['expiration'] for leg in legs})}
    sig.update({k: _classify(nets[k], mags[k]) for k in nets})
    sig['net_skew'] = _skew()
    return sig


def _cli() -> None:
    """Preview run: python vol_premium.py SPY  (delta-neutral ATM short call)."""
    import sys

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else 'SPY'
    from realchains.real_cc_backtest import CHAIN_CLEAN_START

    store = load_chain_store(f'{ticker.lower()}_option_dailies.csv',
                             start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    dates, prices = [d for d, _ in pairs], [p for _, p in pairs]

    td = float(sys.argv[2]) if len(sys.argv) > 2 else 0.50
    params = {'target_delta': td, 'dte': 30, 'capital': 100_000}
    summary, _, eq = run_real_short_vol_overlay(dates, prices, store, params)
    stats = short_vol_statistics(eq, summary['capital'], rf=summary['risk_free_rate'])
    print(f'{ticker} delta-neutral short call, target_delta={td}  ({dates[0]} -> {dates[-1]})')
    print(f'  contracts {summary["num_contracts"]}  sold {summary["num_calls_sold"]}  '
          f'win% {summary["win_rate"]}  maxDD {summary["max_drawdown_pct"]}%')
    print(f'  net P&L ${summary["net_pnl"]:,.0f}  =  rf interest ${summary["interest_earned"]:,.0f}'
          f'  +  vol alpha ${summary["alpha_vs_cash"]:,.0f}  (hedge cost -${summary["total_hedge_cost"]:,.0f})')
    print(f'  total return {stats["ann_return_pct"]}%/yr   excess-over-rf {stats["ann_excess_return_pct"]}%/yr'
          f'   vol {stats["ann_vol_pct"]}%')
    print(f'  excess Sharpe {stats["sharpe"]}   Newey-West t {stats["t_stat_newey_west"]} '
          f'(L={stats["nw_lag"]})  passes t=2? {stats["passes_t_2"]}')
    print('  [audited; SPY headline pinned by TestSpyShortVolRegression. Caveats: docs/vol_premium.md]')


if __name__ == '__main__':
    _cli()
