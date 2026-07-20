"""Covered-call overlay on REAL option chains — the honest re-run.

Mirrors cc_backtest.run_cc_overlay's rules exactly, but every option number
comes from the market instead of the Black-Scholes + estimate_iv proxy:

- Entry: the actual ~target-DTE / ~target-delta call from that day's chain,
  sold at the real BID (replaces the proxy's 3% slippage model; the $0.65
  commission per contract is kept on both legs).
- Marks: the contract's real mid (mark) each day for equity; carried forward
  on missing-quote days.
- Profit target & deep-ITM close: triggered on the real ASK (what a buyback
  actually costs) and the real delta; filled at the ask.
- Expiration: the contract's real expiration date (not a trading-day clock),
  settled against the UNADJUSTED close (strikes live in actual price space).
- Buy-and-hold benchmark: same unadjusted series, so dividends cancel in the
  excess-return comparison (for the base shares — see the delta_hedge caveat).
- Optional delta_hedge param: the Israelov-Nielsen risk-managed variant on
  real chains — hold the short call's quoted delta in extra shares, rebalanced
  daily at the close (same semantics as run_cc_overlay's delta_hedge, but the
  hedge ratio is the vendor delta instead of a Black-Scholes one). Caveat:
  hedge shares are marked on the unadjusted closes, so their dividends go
  uncredited — buy-and-hold never holds them, so nothing cancels (~$12K
  across the canonical MSFT span, about the size of the measured hedged
  excess itself; unpinned estimate from the adjusted/unadjusted ratio). The
  proxy-twin comparison runs on the same series and shares the omission, so
  real-vs-proxy stays apples-to-apples; absolute hedged figures are
  conservatively understated by roughly that much.
- Optional cap_delta param: the call-spread variant — alongside each short
  sale, BUY a same-expiration further-OTM call at this target delta (e.g.
  0.10) as a cap. The cap bounds the deep-ITM buyback loss (floored at
  net_credit − strike_width at expiry) at the cost of its premium, paid every
  cycle. select_cap_leg picks the leg; the close, settlement, and equity paths
  key on the NET spread value. A structural overlay measured directly, not a
  pre-registered experiment.

Data: {ticker}_option_dailies.csv from download_option_dailies.py and an
unadjusted close series (auto-downloaded to {ticker}_10yr_prices_unadjusted.csv).

Usage:
    python real_cc_backtest.py QQQ
    python real_cc_backtest.py MSFT msft_option_dailies_2008_2016.csv   # merge a backfill
"""

from __future__ import annotations
from common.paths import data_path

import csv
import gzip
import io
import os
import sys
from datetime import datetime
from typing import Any, Container, Sequence, TextIO

import pandas as pd

from engine.cc_backtest import compute_statistics, run_cc_overlay

COMMISSION_PER_SHARE = 0.0065  # $0.65 per 100-share contract, both legs (engine convention)

# First trading day after the last vendor-placeholder row inside the entry
# band (bid > 0, 0.05 < delta < 0.60), measured per dataset. The 2008 ->
# mid/late-2010 era of the Alpha Vantage chains carries quantized lattice
# IVs (0.01488 + k*0.00976) and garbage deltas (adjacent strikes jumping
# 0.505 -> 0.087) on ~99.5% of MSFT's and ~33% of SPY's 2008-09 entry-band
# rows - data no delta-targeted entry can trade. Runs EXCLUDE that era
# (pass `start=` to load_chain_store) rather than repairing it: on MSFT,
# 2008-2009 offers ~2 trustworthy entry days a year, far too few to trade,
# so the GFC itself is untestable on these chains. QQQ needs no boundary
# (its store begins 2011-03-23, past the era - see CLAUDE.md's era gotchas).
# IWM is the same clean-from-row-one case (store begins 2010-12-01, past the
# era; validation found entry-band raw == defect-free in every month - no
# placeholder rows) but gets an explicit entry: the put-side VRP run reads its
# boundary, so its boundary is documentary, not load-bearing.
#
# SPY's boundary was corrected 2010-12-01 -> 2010-05-17 (validate_dailies.py): its
# entry band is clean from 2010-05-17, and the later 2010 stragglers are OUT-of-band
# placeholder rows that never reach a delta-targeted entry, so the old boundary
# clipped ~6 months of usable data. This is the LIVE hygiene boundary, used by
# exploratory and future work.
CHAIN_CLEAN_START: dict[str, str] = {'MSFT': '2010-05-10', 'SPY': '2010-05-17', 'IWM': '2010-12-01'}

# A registered experiment's data span is part of its pre-registration and must NOT
# move when the live hygiene boundary is later corrected. REGISTERED_CLEAN_START is
# the frozen registration snapshot — it differs from CHAIN_CLEAN_START only for SPY
# (frozen at the as-registered 2010-12-01). The registered put-side VRP
# (run_registered_vrp.py, TestSpyShortPutRegression, TestSpyStraddleSecondary) and
# the registered trend-gate (trend_gate.py) read THIS, never the live constant.
REGISTERED_CLEAN_START: dict[str, str] = {'MSFT': '2010-05-10', 'SPY': '2010-12-01', 'IWM': '2010-12-01'}


def open_dailies(path: str) -> TextIO:
    """Open a dailies CSV, transparently falling back to its .gz twin.

    Neither file lives in git history (65-281MB): the .gz ships as a release
    asset (tag data-2026-06) fetched by fetch_option_data.sh locally and by
    the CI workflow, checksum-verified either way. The raw CSV, when present
    (fresh fetcher output or a local gunzip), is preferred for speed.
    """
    if os.path.exists(path):
        return open(path, newline='')
    gz = path + '.gz'
    if os.path.exists(gz):
        return io.TextIOWrapper(gzip.open(gz, 'rb'), encoding='utf-8', newline='')
    raise FileNotFoundError(f'{path} (or {gz})')


# ---- data loading ----

def _unsplit_factor(date_str: str, splits: Sequence[tuple[str, float]]) -> float:
    """Product of split ratios dated strictly AFTER `date_str` — the factor that backs
    out yfinance's split adjustment to recover the as-traded price on that date. A date
    after every split gets 1.0; a 2:1 split dated later multiplies earlier dates by 2.0."""
    factor = 1.0
    for split_date, ratio in splits:
        if date_str < split_date:
            factor *= ratio
    return factor


def load_unadjusted_prices(ticker: str, start: str, end: str) -> tuple[list[str], list[float]]:
    """As-traded closes matching the option strikes.

    yfinance split-adjusts the `Close` column even with auto_adjust=False, but the
    option strikes are in AS-TRADED terms — so a ticker that SPLIT after (or during)
    its option span has a rescaled price history that no longer matches its strikes,
    and any delta-hedged overlay run on it blows up. We back the split adjustment out
    by multiplying each date by the product of split ratios dated strictly AFTER it.
    A ticker with no in-span split gets factor 1.0 (byte-identical to the old path).
    XLE's 2025-12-05 2:1 split is the case that surfaced this — it had HALVED every
    pre-split close, mismatching the ~2x-larger strikes."""
    path = data_path(f'{ticker.lower()}_10yr_prices_unadjusted.csv')
    if not os.path.exists(path):
        import yfinance as yf  # lazy: only on first run
        raw = pd.DataFrame(yf.download(ticker, start=start, end=end, auto_adjust=False,
                                       progress=False))
        try:
            splits = [(str(ts.date()), float(r)) for ts, r in yf.Ticker(ticker).splits.items()]
        except Exception:  # noqa: BLE001 — a network/parse hiccup → treat as no splits
            splits = []
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['date', 'close'])
            for d, v in raw['Close'].itertuples():
                ds = d.strftime('%Y-%m-%d')
                w.writerow([ds, float(v) * _unsplit_factor(ds, splits)])
        print(f'Saved unadjusted closes -> {path}')
    dates: list[str] = []
    closes: list[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            dates.append(row['date'])
            closes.append(float(row['close']))
    return dates, closes


def load_chain_store(
    path: str, extra_paths: Sequence[str] = (), start: str | None = None,
    min_dte: int | None = None,
) -> dict[str, dict[str, Any]]:
    """One pass over the dailies CSV(s) -> per-date entry candidates + mark index.

    Returns {date: {'candidates': [(dte, delta, bid, ask, mid, expiration,
    strike, contractID), ...], 'marks': {contractID: (bid, ask, mid, delta)}}}.

    `extra_paths` merge additional dailies CSVs into the same store (e.g. the
    2008-2016 MSFT backfill alongside the canonical 2016-2026 file); the
    per-date setdefault makes the merge order-independent.

    `min_dte` drops every row with DTE below it during the parse (never stored).
    It loads ONLY the far legs (DTE >= 60) of the `_180dte` calendar backfill:
    the canonical store already holds DTE <= 60, so the far file's near-term rows
    are duplicates, and loading the file whole (the dense near-term bulk) OOMs the
    structure campaign on a CI-sized (~7GB) runner. None = load every row.

    `start` drops every row dated before it. This is how the 2008 ->
    mid/late-2010 placeholder-greeks era is handled (see CHAIN_CLEAN_START):
    the era is EXCLUDED, not repaired — its in-band rows carry lattice IVs
    and deltas unrelated to moneyness, so no delta-targeted entry could have
    traded there. On the post-boundary spans the exclusion is provably
    sufficient: no pinned run ever selects a defective row (verified by
    re-running every pinned surface with a row-level guard — byte-identical).
    A row-level `IV < 0.05` filter was likewise considered and rejected:
    SPY's 2017 low-vol chains quote lattice-quantized IVs on rows whose
    deltas are sane, and delta and quotes are all this engine consumes —
    anything reading vendor IV as a *signal* needs its own sidecar guard.

    Mark sanity clamp: a quoted mark outside [bid, ask] is bad vendor data —
    the modern files carry a small tail of these (0.05-0.14% of rows) — so
    out-of-band marks are replaced by the quote midpoint.
    """
    store: dict[str, dict[str, Any]] = {}
    for p in (path, *extra_paths):
        with open_dailies(p) as f:
            for r in csv.DictReader(f):
                d = r['date']
                if start is not None and d < start:
                    continue
                try:
                    dte = int(r['dte'])
                    delta = float(r['delta'])
                    bid = float(r['bid'])
                    ask = float(r['ask'])
                    mid = float(r['mark'])
                    strike = float(r['strike'])
                except (TypeError, ValueError):
                    continue
                if min_dte is not None and dte < min_dte:
                    continue
                if not (bid <= mid <= ask):
                    mid = (bid + ask) / 2
                day = store.setdefault(d, {'candidates': [], 'marks': {}})
                day['candidates'].append(
                    (dte, delta, bid, ask, mid, r['expiration'], strike, r['contractID'])
                )
                day['marks'][r['contractID']] = (bid, ask, mid, delta)
    return store


def select_entry(
    day: dict[str, Any], target_dte: int, target_delta: float
) -> tuple[int, float, float, float, float, str, float, str] | None:
    """Nearest-DTE expiration, then nearest-delta call — mirrors the roll fetcher."""
    cands = [c for c in day['candidates'] if c[2] > 0 and 0.05 < c[1] < 0.60]
    if not cands:
        return None
    best_dte = min({c[0] for c in cands}, key=lambda x: abs(x - target_dte))
    cohort = [c for c in cands if c[0] == best_dte]
    return min(cohort, key=lambda c: abs(c[1] - target_delta))


def select_cap_leg(
    day: dict[str, Any], expiration: str, short_strike: float, cap_delta: float
) -> tuple[float, float, float, float, float, str] | None:
    """The long cap leg of a call spread: the same-expiration call nearest
    `cap_delta`, struck above the short, with a buyable ask.

    A vertical spread requires both legs on the SAME expiration (else it is a
    diagonal and the payoff no longer caps cleanly), so the search is the
    short pick's `expiration` cohort, not select_entry's nearest-DTE cohort.
    It runs against the RAW candidate list, not select_entry's 0.05<delta<0.60
    band: a deep cap legitimately sits below 0.05 delta, and excluding it
    would force the cap closer in than asked. Requires ask>0 (we BUY this leg)
    and strike strictly above the short. Returns
    (delta, bid, ask, mid, strike, contractID) or None when no higher strike
    in that expiration is quotable (the caller then degrades to a naked short
    for that cycle).
    """
    caps = [c for c in day['candidates']
            if c[5] == expiration and c[6] > short_strike and c[3] > 0]
    if not caps:
        return None
    best = min(caps, key=lambda c: abs(c[1] - cap_delta))
    # candidate tuple: (dte, delta, bid, ask, mid, expiration, strike, cid)
    return (best[1], best[2], best[3], best[4], best[6], best[7])


# ---- the overlay loop (run_cc_overlay semantics, real prices) ----

def run_real_cc_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, float],
    suspended_dates: Container[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    # suspended_dates is the trend-gate experiment's entry seam
    # (docs/prereg_trend_gate.md §10): dates in the set may not OPEN a new
    # position. A suspended day behaves exactly like an existing no-entry day
    # (no chain row / empty band / non-positive premium): no trade record,
    # shares held uncovered, the daily equity row still appended, entry
    # re-attempted the next tradeable day. It never touches an open position
    # (the gate is entry-only per §2.3). Default None = off — the pinned
    # regressions all run this byte-identical path.
    call_delta = params.get('call_delta', 0.25)
    close_at_pct = params.get('close_at_pct', 0.75)
    dte = int(params.get('dte', 21))
    # 'bid_ask' (default): sell at bid, buy back at ask — executable worst case.
    # 'mid': both legs at the quoted mark — the academic convention; isolates
    # how much of the result is bid/ask spread vs the premium level itself.
    fill = str(params.get('fill', 'bid_ask'))
    # Optional stop-loss buyback: close the short call when its buyback price
    # reaches this multiple of the (net) premium collected, e.g. 2.0 = the
    # classic "stop at 2x entry". Evaluated once per day on the close quote
    # (like the profit target), so gap-throughs fill at the day's ask, not at
    # the stop level — a stop-market approximation, the honest daily-bar
    # reading of a stop order. None/absent = off (byte-identical baseline).
    stop_loss_mult = params.get('stop_loss_mult')
    # Israelov-Nielsen risk-managed variant: when truthy, hold extra long
    # shares equal to the short call's current delta × base shares, rebalanced
    # daily at the close, so the portfolio's net delta stays pinned at the
    # buy-and-hold equivalent. Mirrors run_cc_overlay's delta_hedge semantics
    # (hedge trades draw on a working cash account that may go negative — a
    # zero-interest financing simplification; no transaction costs on the
    # share legs) with one real-chain substitution: the hedge ratio is the
    # vendor delta from the day's quote, carried forward on missing-quote
    # days exactly like the mark. False/absent = off (byte-identical baseline).
    delta_hedge = bool(params.get('delta_hedge', False))
    # Call-spread variant: when set, BUY a same-expiration further-OTM call at
    # this target delta (e.g. 0.10) as a cap alongside each short sale. The cap
    # bounds the deep-ITM buyback loss — at expiration the spread's per-share
    # P&L is floored at net_credit − (cap_strike − short_strike) (see
    # select_cap_leg and the settlement branch). The cost is the cap's premium,
    # paid every cycle. None/absent = off (byte-identical baseline — no cap leg,
    # no position['cap_*'] fields, every spread path falls through to the naive
    # short-only branch). An engineering variant measured directly, like
    # delta_hedge and stop_loss_mult — not a pre-registered experiment.
    cap_delta = params.get('cap_delta')

    initial_price = prices[0]
    contract_cost = initial_price * 100
    capital = float(params.get('capital', contract_cost))
    num_contracts = int(capital // contract_cost)
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts
    initial_cash = capital - shares * initial_price
    current_cash = initial_cash  # working balance; drained/refilled by hedge
                                 # share trades when delta_hedge is on, else
                                 # constant at initial_cash.
    hedge_shares = 0             # extra long stock offsetting the short call's
                                 # delta; stays 0 when delta_hedge is off.

    position: dict[str, Any] | None = None
    realized_pnl = 0.0
    num_calls_sold = 0
    total_premium_collected = 0.0      # NET credit (short sale minus cap cost)
    gross_premium_collected = 0.0      # short-leg sale only — the upside given
                                       # up; equals total when no cap, so
                                       # retention stays comparable to the
                                       # naked baseline.
    wins = losses = 0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)

        if position is None:
            # Consider entry — needs a chain for today and a non-suspended date.
            if day is not None and (suspended_dates is None
                                    or date not in suspended_dates):
                pick = select_entry(day, dte, call_delta)
                if pick is not None:
                    _dte, _delta, bid, _ask, mid, expiration, strike, cid = pick
                    sell_px = bid if fill == 'bid_ask' else mid
                    gross_premium = sell_px - COMMISSION_PER_SHARE  # short sale, net of its commission
                    # Cap leg: BUY a same-expiration further-OTM call. None when
                    # off, or when no quotable higher strike exists that cycle
                    # (degrade to a naked short — keeps the trade cadence equal
                    # to the baseline rather than dropping the cycle).
                    cap = (select_cap_leg(day, expiration, strike, cap_delta)
                           if cap_delta is not None else None)
                    net_premium = gross_premium
                    if cap is not None:
                        cap_delta_v, cap_bid, cap_ask, cap_mid, cap_strike, cap_cid = cap
                        buy_px = cap_ask if fill == 'bid_ask' else cap_mid
                        net_premium -= buy_px + COMMISSION_PER_SHARE  # pay the cap + its commission
                    if net_premium > 0:
                        position = {
                            'strike': strike,
                            'premium_collected': net_premium,
                            'expiration': expiration,
                            'contract': cid,
                            'entry_date': date,
                            'last_mid': mid,
                            'real_delta': _delta,
                            'worst_unrealized': 0.0,  # Gap A (A2): running min of daily MTM P&L, dollars
                        }
                        sell_record = {'date': date, 'price': price, 'action': 'sell',
                                       'premium': net_premium, 'strike': strike,
                                       'contract': cid, 'dte': _dte, 'delta': _delta,
                                       'pnl': 0, 'realized_pnl': realized_pnl}
                        if cap is not None:
                            # cap_quote = (bid, ask, mid, delta), refreshed when
                            # the cap prints and carried forward otherwise — it
                            # must be fillable on a close day even if the cap
                            # itself went unquoted.
                            position['cap_strike'] = cap_strike
                            position['cap_contract'] = cap_cid
                            position['cap_quote'] = (cap_bid, cap_ask, cap_mid, cap_delta_v)
                            sell_record['cap_strike'] = cap_strike
                            sell_record['cap_premium'] = buy_px + COMMISSION_PER_SHARE
                        num_calls_sold += 1
                        total_premium_collected += net_premium * shares
                        gross_premium_collected += gross_premium * shares
                        trades.append(sell_record)
        else:
            if date >= position['expiration']:
                # Real expiration: settle against the unadjusted close of the
                # last trading day ON or BEFORE the expiration date. Every
                # expiration in the 2016+ datasets is a trading day, so this
                # is today's close — identical to the original lag-0
                # convention, and the pinned regressions hold. Pre-Feb-2015
                # standard expirations are SATURDAY-dated (the old listing
                # convention), so the first loop date past expiration is the
                # following Monday: settle against Friday's close, the last
                # day the option traded (a Good-Friday week settles against
                # Thursday's, for the same reason). A gap larger than a long
                # weekend means corrupt data, so fail loudly on that instead.
                if date == position['expiration']:
                    settle_price = price
                else:
                    assert prev_date is not None and prev_price is not None, (
                        f'position expired {position["expiration"]} before the '
                        f'first trading day of the series'
                    )
                    assert prev_date <= position['expiration'], (
                        f'last close {prev_date} is after expiration '
                        f'{position["expiration"]} — settlement logic error'
                    )
                    gap = (datetime.strptime(position['expiration'], '%Y-%m-%d')
                           - datetime.strptime(prev_date, '%Y-%m-%d')).days
                    assert gap <= 4, (
                        f'{gap} calendar days between last close {prev_date} and '
                        f'expiration {position["expiration"]} — missing data?'
                    )
                    settle_price = prev_price
                # Spread payoff at expiry: short pays -max(0, S-Ks), the cap
                # (same expiration by construction) receives +max(0, S-Kl).
                # With no cap, long_intrinsic is 0 and this is the naked
                # settlement exactly. Above the cap strike the S terms cancel,
                # flooring the loss at premium - (Kl - Ks) per share.
                short_intrinsic = max(0.0, settle_price - position['strike'])
                long_intrinsic = (max(0.0, settle_price - position['cap_strike'])
                                  if 'cap_strike' in position else 0.0)
                pnl = (position['premium_collected']
                       - short_intrinsic + long_intrinsic) * shares
                realized_pnl += pnl
                wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
                mae_out = position['worst_unrealized']
                position = None
                trades.append({'date': date, 'price': settle_price, 'action': 'expiration',
                               'pnl': pnl, 'realized_pnl': realized_pnl,
                               'mae': round(mae_out, 2)})
            else:
                # Refresh the cap leg's quote independently of the short — the
                # two legs can print on different days, and a close may fire on
                # a day the deep-OTM cap is unquoted (then its carried quote is
                # the fill). No-op when there is no cap.
                if 'cap_contract' in position and day:
                    cap_q = day['marks'].get(position['cap_contract'])
                    if cap_q is not None:
                        position['cap_quote'] = cap_q
                quote = day['marks'].get(position['contract']) if day else None
                if quote is not None:
                    bid_q, ask_q, mid_q, delta_q = quote
                    position['last_mid'] = mid_q
                    position['real_delta'] = delta_q
                    # close_ref is the net cost to close the spread excluding
                    # commission (buy back the short, sell the cap) — the
                    # trigger reference. With no cap it is just the short ask,
                    # so the triggers and pnl are byte-identical to the naive
                    # path. The cap is unwound by SELLING at its bid (bid_ask)
                    # or mid, from the live or carried cap quote.
                    short_buy = ask_q if fill == 'bid_ask' else mid_q
                    if 'cap_contract' in position:
                        c_bid, _c_ask, c_mid, _c_delta = position['cap_quote']
                        cap_sell = c_bid if fill == 'bid_ask' else c_mid
                        close_ref = short_buy - cap_sell
                        close_commission = 2 * COMMISSION_PER_SHARE
                    else:
                        close_ref = short_buy
                        close_commission = COMMISSION_PER_SHARE
                    # Profit target / stop key on the NET spread value, so
                    # close_at_pct retains 75% of the credit actually banked.
                    # Deep-ITM keys on the SHORT delta only (the assignment leg).
                    hit_target = close_ref <= position['premium_collected'] * (1 - close_at_pct)
                    deep_itm = params.get('manage_deep_itm', True) and delta_q > 0.70
                    hit_stop = (stop_loss_mult is not None
                                and close_ref >= position['premium_collected'] * float(stop_loss_mult))
                    if hit_target or deep_itm or hit_stop:
                        pnl = (position['premium_collected']
                               - (close_ref + close_commission)) * shares
                        realized_pnl += pnl
                        wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
                        action = ('close' if hit_target
                                  else 'close_stop' if hit_stop else 'close_itm')
                        mae_out = position['worst_unrealized']
                        position = None
                        trades.append({'date': date, 'price': price, 'action': action,
                                       'call_value': ask_q, 'pnl': pnl,
                                       'realized_pnl': realized_pnl,
                                       'mae': round(mae_out, 2)})
                # No quote today: no close can trigger; mark carries forward below.

        # Risk-managed rebalance (after the entry/close logic, like the proxy
        # engine): while a call is short, hold its delta in extra shares;
        # target 0 the day it closes. position['real_delta'] is today's quoted
        # delta when one printed, else carried forward — the same convention
        # as the mark used for equity. Trades fill at the unadjusted close.
        # The delta is clamped to [0, 1] before sizing: vendor deltas are not
        # the engine's BS output, and a placeholder row could otherwise short
        # the hedge (no pinned run consumes an out-of-range delta — the clamp
        # is a guard, not a repair). Convention note: on Saturday-dated
        # expirations (pre-2015 backfill era) the option settles against
        # Friday's close but the loop reaches the expiration branch on Monday,
        # so the hedge unwinds at Monday's close — one weekend of hedge
        # exposure the option leg no longer has. The SPY cc_r_experiment pins
        # DO span that era (disclosed in its log entry); MSFT/QQQ pins do not.
        target_hedge = (int(round(min(max(position['real_delta'], 0.0), 1.0) * shares))
                        if delta_hedge and position is not None else 0)
        hedge_trade = target_hedge - hedge_shares
        if hedge_trade != 0:
            current_cash -= hedge_trade * price
            hedge_shares = target_hedge

        equity = price * (shares + hedge_shares) + current_cash + realized_pnl
        if position is not None:
            # Open-spread MTM: net credit minus the current net cost to close
            # (short mark minus cap mark). With no cap, spread_mark is the
            # short mark and this is the naive MTM exactly.
            spread_mark = position['last_mid']
            if 'cap_quote' in position:
                spread_mark -= position['cap_quote'][2]  # cap mid
            unrealized = (position['premium_collected'] - spread_mark) * shares
            equity += unrealized
            # Gap A (A2): running MAE on the open spread's daily mark (carried-forward
            # marks on missing-quote days, same convention as the equity above).
            position['worst_unrealized'] = min(position['worst_unrealized'], unrealized)
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    total_return = (final_equity - capital) / capital * 100
    buy_hold_final = prices[-1] * shares + initial_cash
    buy_hold_return = (buy_hold_final - capital) / capital * 100
    net_overlay_pnl = final_equity - buy_hold_final
    retention = (net_overlay_pnl / total_premium_collected * 100
                 if total_premium_collected > 0 else 0.0)

    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())

    summary = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'initial_stock_cost': round(shares * initial_price, 2),
        # Initial leftover cash, NOT the working balance — compute_statistics
        # rebuilds the buy-and-hold curve from shares × prices + cash, which
        # only makes sense with the constant initial cash. Under delta_hedge
        # the working cash drifts as hedge trades execute; that drift reaches
        # final_equity via the daily equity series. (Same convention as
        # run_cc_overlay's summary.)
        'cash': round(initial_cash, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_final': round(buy_hold_final, 2),
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'excess_return_pct': round(total_return - buy_hold_return, 2),
        'net_overlay_pnl': round(net_overlay_pnl, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        # gross == total and cap_cost == 0 with no cap leg, so these are
        # additive (the off-path summary's other fields are byte-identical).
        # For the spread, retention on gross stays comparable to the naked
        # baseline (gross = the short-leg upside given up); total_premium is
        # the smaller net credit actually banked.
        'gross_premium_collected': round(gross_premium_collected, 2),
        'cap_cost_paid': round(gross_premium_collected - total_premium_collected, 2),
        'overlay_costs': round(round(total_premium_collected, 2) - round(net_overlay_pnl, 2), 2),
        'premium_retention_pct': round(retention, 1),
        'num_calls_sold': num_calls_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / max(wins + losses, 1) * 100, 1),
        'max_drawdown_pct': round(max_dd, 2),
    }
    return summary, trades, daily_equity


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else 'QQQ').upper()
    extra_dailies = sys.argv[2:]  # e.g. msft_option_dailies_2008_2016.csv (the backfill)
    dailies_path = data_path(f'{ticker.lower()}_option_dailies.csv')
    for p in (dailies_path, *extra_dailies):
        if not (os.path.exists(p) or os.path.exists(p + '.gz')):
            sys.exit(f'{p}[.gz] not found — run download_option_dailies.py first')

    params = {'call_delta': 0.25, 'close_at_pct': 0.75, 'dte': 21,
              'risk_free_rate': 0.045, 'capital': 100_000}
    # Parity note: the engine's dte=21 is TRADING days (T = 21/252 — about a
    # month), so the real leg must target the calendar-day equivalent or it
    # sells shorter, cheaper calls on a faster cycle than the proxy ever did.
    real_params = {**params, 'dte': round(params['dte'] / 252 * 365)}  # ~30 calendar days

    start = CHAIN_CLEAN_START.get(ticker)
    print(f'Loading chain store ({dailies_path}'
          + (f' + {", ".join(extra_dailies)}' if extra_dailies else '')
          + (f', from {start}' if start else '') + ') ...', flush=True)
    store = load_chain_store(dailies_path, extra_dailies, start=start)
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    # Clip the price series to the chain-covered span.
    lo, hi = days[0], days[-1]
    pairs = [(d, p) for d, p in zip(dates, prices) if lo <= d <= hi]
    dates = [d for d, _ in pairs]
    prices = [p for _, p in pairs]
    print(f'{ticker}: {len(dates)} trading days {dates[0]} -> {dates[-1]}, '
          f'{len(days)} chain days', flush=True)

    real_sum, real_trades, real_eq = run_real_cc_overlay(dates, prices, store, real_params)
    import numpy as np
    proxy_sum, _, proxy_eq = run_cc_overlay(dates, np.array(prices), params)
    real_st = compute_statistics(real_eq, num_contracts=real_sum['num_contracts'],
                                 cash=real_sum['cash'])
    proxy_st = compute_statistics(proxy_eq, num_contracts=proxy_sum['num_contracts'],
                                  cash=proxy_sum['cash'])

    print(f"\n=== {ticker} covered-call overlay: REAL chains vs PROXY (same unadjusted series) ===")
    rows = [
        ('Buy & hold return', 'buy_hold_return_pct', '%'),
        ('Overlay total return', 'total_return_pct', '%'),
        ('Net overlay P&L', 'net_overlay_pnl', '$'),
        ('Gross premium collected', 'total_premium_collected', '$'),
        ('Premium retention', 'premium_retention_pct', '%'),
        ('Calls sold', 'num_calls_sold', ''),
        ('Win rate', 'win_rate', '%'),
        ('Max drawdown', 'max_drawdown_pct', '%'),
    ]
    print(f"  {'metric':<26}{'REAL':>14}{'PROXY':>14}")
    print(f"  {'-' * 26}{'-' * 14}{'-' * 14}")
    def fmt(v: Any, unit: str) -> str:
        if unit == '$':
            return f'${v:,.0f}'
        return f'{v:,.2f}%' if unit == '%' else f'{v}'

    for label, key, unit in rows:
        print(f'  {label:<26}{fmt(real_sum[key], unit):>14}{fmt(proxy_sum[key], unit):>14}')
    print(f"  {'Ann. excess return':<26}{real_st['ann_excess_return_pct']:>13.3f}%"
          f"{proxy_st['ann_excess_return_pct']:>13.3f}%")
    print(f"  {'Sharpe of excess':<26}{real_st['sharpe_excess']:>14.3f}{proxy_st['sharpe_excess']:>14.3f}")
    print(f"  {'t-stat (Newey-West)':<26}{real_st['t_stat_newey_west']:>14.2f}"
          f"{proxy_st['t_stat_newey_west']:>14.2f}")


if __name__ == '__main__':
    main()
