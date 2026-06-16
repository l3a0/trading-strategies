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
from typing import Any

import numpy as np
import pandas as pd

from real_cc_backtest import (
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
    """Daily delta-neutral short-call overlay on real option quotes.

    Differences from run_real_cc_overlay (the covered call), all deliberate:
      - NO base long-stock leg. Capital is collateral; the only stock held is the
        hedge that offsets the short call's delta. Net portfolio delta ~ 0.
      - Default strike is ATM (target_delta=0.50), not 0.25 — ATM maximizes
        gamma/vega, so it carries the most variance-premium signal.
      - Default close is HOLD-TO-EXPIRY (close_at_pct=None, manage_deep_itm=False):
        early profit-taking or deep-ITM management truncates the variance
        exposure the position exists to measure. Set them to mirror the covered
        call when you want an apples-to-apples comparison.
      - The hedge targets NET-ZERO delta (there is no buy-and-hold leg to pin to).

    Single cash account: starts at `capital`, RECEIVES the premium, PAYS buybacks /
    assignment and hedge trades, and EARNS the risk-free rate daily. Equity is then
    cash + hedge stock - the current value of the short call. This credits the idle
    collateral with rf (the omission the buy-and-hold comparison exposed) and
    charges the share-hedge bid/ask half-spread (Schwab: stock/ETF trades are
    commission-free, so spread is the only share cost; the option leg keeps
    COMMISSION_PER_SHARE).

    params: target_delta (0.50), dte (30 calendar days), fill ('bid_ask'|'mid'),
            capital (100_000), close_at_pct (None), manage_deep_itm (False),
            option_type ('call' | 'put' — for 'put' pass a NEGATIVE target_delta,
            e.g. -0.25; the short put hedges with SHORT stock),
            risk_free_rate (0.045, earned daily on cash), hedge_cost_bps (1.0,
            share-rebalance half-spread in bps of notional traded).

    Returns (summary, trades, daily_equity). daily_equity has columns
    date / equity / price / rf_credit, where rf_credit is the EXACT interest
    credited at the start of each day (0 on day 0). short_vol_statistics nets that
    series out so the VRP verdict is the pure gamma/vega vol-P&L — rf cancels on
    the same (cash) base it was earned on, making the verdict rate-invariant.
    summary['alpha_vs_cash'] is net_pnl minus the rf interest, i.e. that same vol
    premium harvested net of hedge costs. It equals the daily excess that
    short_vol_statistics sums UP TO the day-0 entry-spread mark: that series starts
    from eq[0], which is already struck at the entry bid/ask mid, so it omits the
    single day-0 entry cost that alpha_vs_cash carries (see short_vol_statistics for
    the exact gap — it OMITS a cost, so the summed excess slightly flatters).
    """
    target_delta = float(params.get('target_delta', 0.50))
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    capital = float(params.get('capital', 100_000))
    close_at_pct = params.get('close_at_pct')  # None => hold to expiry
    manage_deep_itm = bool(params.get('manage_deep_itm', False))
    # 'call' (default) or 'put'. A short put neutralizes with SHORT stock (its
    # vendor delta is negative), so the only differences are the entry selector,
    # the settlement intrinsic, and the sign of the hedge clamp; the call path is
    # byte-identical (call deltas are >= 0, so its [0,1] clamp is unchanged).
    # For puts the caller passes a NEGATIVE target_delta (e.g. -0.25).
    is_put = str(params.get('option_type', 'call')) == 'put'
    rf = float(params.get('risk_free_rate', 0.045))  # earned daily on the cash balance
    # Schwab charges $0 commission on stock/ETF trades, so the only cost of a
    # share-hedge rebalance is the bid/ask half-spread, modeled as bps of the
    # share notional traded. (The option leg keeps COMMISSION_PER_SHARE.)
    hedge_cost_bps = float(params.get('hedge_cost_bps', 1.0))
    daily_rf = rf / 252.0

    initial_price = prices[0]
    contract_cost = initial_price * 100
    num_contracts = int(capital // contract_cost)
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts  # option NOTIONAL (100/contract); NOT a long position

    # ONE cash account: starts at capital, receives premium, pays buybacks /
    # assignment and hedge trades, earns rf daily. Equity = cash + hedge stock -
    # current value of the short call. One conservation law, easy to audit.
    cash = capital
    hedge_shares = 0  # long stock offsetting the short call delta -> net ~0

    position: dict[str, Any] | None = None
    num_calls_sold = 0
    total_premium_collected = 0.0
    total_hedge_cost = 0.0
    interest_earned = 0.0
    wins = losses = 0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)

        # 1. Interest on the cash carried over from yesterday (rf when positive,
        #    financing when the hedge drove cash negative). Record the EXACT
        #    per-day credit so short_vol_statistics can net rf out on the SAME
        #    base the engine accrued it on (cash) — not a flat rf on capital or
        #    grown equity, which over-removes interest the account never earned.
        day_rf_credit = 0.0
        if i > 0:
            day_rf_credit = cash * daily_rf
            cash += day_rf_credit
            interest_earned += day_rf_credit

        # 2. Entry / close / settlement -- all cash flows.
        if position is None:
            if day is not None:
                pick = (select_put_entry if is_put else select_entry)(day, dte, target_delta)
                if pick is not None:
                    _dte, _delta, bid, _ask, mid, expiration, strike, cid = pick
                    sell_px = bid if fill == 'bid_ask' else mid
                    premium = sell_px - COMMISSION_PER_SHARE
                    if premium > 0:
                        cash += premium * shares  # RECEIVE the premium
                        position = {
                            'strike': strike, 'entry_premium': premium,
                            'expiration': expiration, 'contract': cid,
                            'entry_date': date, 'last_mid': mid, 'real_delta': _delta,
                        }
                        num_calls_sold += 1
                        total_premium_collected += premium * shares
                        trades.append({'date': date, 'price': price, 'action': 'sell',
                                       'premium': premium, 'strike': strike, 'contract': cid,
                                       'dte': _dte, 'delta': _delta})
        else:
            if date >= position['expiration']:
                # Settle on the last close on/before expiration (today's for modern
                # trading-day expiries; gap<=4 guard catches corrupt data).
                if date == position['expiration']:
                    settle_price = price
                else:
                    assert prev_date is not None and prev_price is not None
                    gap = (pd.Timestamp(position['expiration']) - pd.Timestamp(prev_date)).days
                    assert gap <= 4, (f'{gap} days between {prev_date} and '
                                      f'expiration {position["expiration"]} — missing data?')
                    settle_price = prev_price
                intrinsic = max(0.0, (position['strike'] - settle_price) if is_put
                                else (settle_price - position['strike']))
                cash -= intrinsic * shares  # PAY assignment (premium already received)
                option_pnl = (position['entry_premium'] - intrinsic) * shares
                wins, losses = (wins + 1, losses) if option_pnl >= 0 else (wins, losses + 1)
                position = None
                trades.append({'date': date, 'price': settle_price, 'action': 'expiration',
                               'pnl': option_pnl})
            else:
                quote = day['marks'].get(position['contract']) if day else None
                if quote is not None:
                    bid_q, ask_q, mid_q, delta_q = quote
                    position['last_mid'] = mid_q
                    position['real_delta'] = delta_q
                    short_buy = ask_q if fill == 'bid_ask' else mid_q
                    hit_target = (close_at_pct is not None
                                  and short_buy <= position['entry_premium'] * (1 - close_at_pct))
                    deep_itm = manage_deep_itm and (delta_q < -0.70 if is_put else delta_q > 0.70)
                    if hit_target or deep_itm:
                        buyback = short_buy + COMMISSION_PER_SHARE
                        cash -= buyback * shares  # PAY to close
                        option_pnl = (position['entry_premium'] - buyback) * shares
                        wins, losses = (wins + 1, losses) if option_pnl >= 0 else (wins, losses + 1)
                        position = None
                        trades.append({'date': date, 'price': price,
                                       'action': 'close' if hit_target else 'close_itm',
                                       'call_value': ask_q, 'pnl': option_pnl})

        # 3. Delta-NEUTRAL rebalance: hold (signed vendor delta)*shares of stock
        #    so net delta (short option -delta*shares + hedge) ~ 0 — LONG stock for
        #    a short call (delta>0), SHORT stock for a short put (delta<0). Clamp to
        #    the option's delta range ([0,1] call / [-1,0] put) to guard noisy
        #    vendor rows. Shares fill at the close, commission-free, half-spread cost.
        _lo, _hi = (-1.0, 0.0) if is_put else (0.0, 1.0)
        target_hedge = (int(round(min(max(position['real_delta'], _lo), _hi) * shares))
                        if position is not None else 0)
        hedge_trade = target_hedge - hedge_shares
        if hedge_trade != 0:
            cash -= hedge_trade * price
            cost = abs(hedge_trade) * price * (hedge_cost_bps / 10_000.0)
            cash -= cost
            total_hedge_cost += cost
            hedge_shares = target_hedge

        # 4. Mark to market: cash + hedge stock - current value of the short call.
        equity = cash + price * hedge_shares
        if position is not None:
            equity -= position['last_mid'] * shares
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price,
                           'rf_credit': day_rf_credit})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price', 'rf_credit'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    net_pnl = final_equity - capital            # includes rf interest on collateral
    alpha_vs_cash = net_pnl - interest_earned   # the part above the risk-free rate

    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())

    summary = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'target_delta': target_delta,
        'final_equity': round(final_equity, 2),
        'net_pnl': round(net_pnl, 2),               # total $ incl. rf on collateral
        'alpha_vs_cash': round(alpha_vs_cash, 2),   # net of rf -- the vol premium harvested
        'interest_earned': round(interest_earned, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'total_hedge_cost': round(total_hedge_cost, 2),
        'hedge_cost_bps': hedge_cost_bps,
        'num_calls_sold': num_calls_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        'max_drawdown_pct': round(max_dd, 2),
        'risk_free_rate': rf,
        'cash': round(capital, 2),  # initial collateral (short_vol_statistics convention)
    }
    return summary, trades, daily_equity


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

    mean_e = float(np.mean(excess))
    var_e = float(np.var(excess, ddof=1))  # variance of the vol-P&L return series
    se_naive = math.sqrt(var_e / n) if var_e > 0 else 0.0
    t_naive = mean_e / se_naive if se_naive > 0 else 0.0

    L = int(4 * (n / 100) ** (2 / 9))
    nw_sum = 0.0
    for k in range(1, L + 1):
        weight = 1.0 - k / (L + 1)
        cov_k = float(np.mean((excess[:-k] - mean_e) * (excess[k:] - mean_e)))
        nw_sum += weight * cov_k
    var_mean_nw = (var_e + 2 * nw_sum) / n
    se_nw = math.sqrt(max(var_mean_nw, 0.0))
    t_nw = mean_e / se_nw if se_nw > 0 else 0.0

    ann_excess = mean_e * periods_per_year
    ann_total = float(np.mean(ret)) * periods_per_year
    ann_vol = math.sqrt(var_e * periods_per_year)
    sharpe = ann_excess / ann_vol if ann_vol > 0 else 0.0

    return {
        'n_days': n,
        'years_of_data': round(n / periods_per_year, 2),
        'ann_return_pct': round(ann_total * 100, 3),         # total, incl. rf
        'ann_excess_return_pct': round(ann_excess * 100, 3),  # over rf -- the alpha
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),                          # vol-P&L Sharpe (rf netted)
        't_stat_naive': round(t_naive, 2),
        't_stat_newey_west': round(t_nw, 2),                 # tests vol-P&L > 0
        'nw_lag': L,
        'passes_t_2': abs(t_nw) > 2.0,
        'mean_daily_pnl_dollars': round(float(np.mean(np.diff(eq))), 2),  # gross, incl. rf
        'mean_daily_excess_dollars': round(mean_e * capital, 2),  # vol-P&L, rf netted
    }


def run_real_straddle_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Daily delta-neutral short ATM STRADDLE — short ~0.50d call + short ~-0.50d
    put at the SAME expiration, hold-to-expiry, the COMBINED net delta hedged to ~0
    with stock rebalanced daily on the signed vendor deltas.

    A pre-registered §7 SECONDARY of docs/prereg_vol_premium.md: REPORTED, NEVER
    PROMOTED — it cannot change the §5 primary (short-put) verdict; it is the one
    two-leg piece §2.2/§9 deferred. A SEPARATE loop, so the frozen single-leg
    run_real_short_vol_overlay path (and every pin on it) stays byte-identical
    (prereg §9). Same single cash account, recorded per-day rf credit, signed hedge,
    and daily_equity schema (date/equity/price/rf_credit), so short_vol_statistics
    consumes it unchanged — the same rate-invariant delta-hedged-gain verdict.

    A short straddle's combined position delta is -(call_delta + put_delta); holding
    (call_delta + put_delta)*shares of stock neutralizes it — ~0 at the money, LONG
    as spot rises and the call dominates, SHORT as it falls and the put dominates.
    The half-spread cost applies to |hedge_trade|; each leg is sold at its own bid
    and keeps COMMISSION_PER_SHARE.

    params: dte (30), capital (100_000), fill ('bid_ask'|'mid'), risk_free_rate
            (0.045), hedge_cost_bps (0.5), call_delta (0.50), put_delta (-0.50).
    """
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    capital = float(params.get('capital', 100_000))
    call_delta = float(params.get('call_delta', 0.50))
    put_delta = float(params.get('put_delta', -0.50))
    rf = float(params.get('risk_free_rate', 0.045))
    hedge_cost_bps = float(params.get('hedge_cost_bps', 0.5))
    daily_rf = rf / 252.0

    initial_price = prices[0]
    num_contracts = int(capital // (initial_price * 100))
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts  # notional of EACH leg

    cash = capital
    hedge_shares = 0
    position: dict[str, Any] | None = None
    num_straddles_sold = 0
    total_premium_collected = 0.0
    total_hedge_cost = 0.0
    interest_earned = 0.0
    wins = losses = 0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)

        # 1. Interest on cash carried from yesterday (recorded for rf-netting).
        day_rf_credit = 0.0
        if i > 0:
            day_rf_credit = cash * daily_rf
            cash += day_rf_credit
            interest_earned += day_rf_credit

        # 2. Entry (BOTH legs) / settlement (BOTH legs at expiry).
        if position is None:
            if day is not None:
                pick = select_straddle(day, dte, call_delta, put_delta)
                if pick is not None:
                    call, put = pick
                    c_sell = call[2] if fill == 'bid_ask' else call[4]
                    p_sell = put[2] if fill == 'bid_ask' else put[4]
                    c_prem = c_sell - COMMISSION_PER_SHARE
                    p_prem = p_sell - COMMISSION_PER_SHARE
                    if c_prem > 0 and p_prem > 0:
                        cash += (c_prem + p_prem) * shares  # RECEIVE both premiums
                        position = {
                            'expiration': call[5],
                            'call': {'strike': call[6], 'premium': c_prem,
                                     'contract': call[7], 'last_mid': call[4], 'delta': call[1]},
                            'put': {'strike': put[6], 'premium': p_prem,
                                    'contract': put[7], 'last_mid': put[4], 'delta': put[1]},
                        }
                        num_straddles_sold += 1
                        total_premium_collected += (c_prem + p_prem) * shares
                        trades.append({'date': date, 'price': price, 'action': 'sell',
                                       'call_strike': call[6], 'put_strike': put[6],
                                       'call_premium': c_prem, 'put_premium': p_prem,
                                       'expiration': call[5]})
        elif date >= position['expiration']:
            if date == position['expiration']:
                settle_price = price
            else:
                assert prev_date is not None and prev_price is not None
                gap = (pd.Timestamp(position['expiration']) - pd.Timestamp(prev_date)).days
                assert gap <= 4, (f'{gap} days between {prev_date} and expiration '
                                  f'{position["expiration"]} — missing data?')
                settle_price = prev_price
            c_intr = max(0.0, settle_price - position['call']['strike'])
            p_intr = max(0.0, position['put']['strike'] - settle_price)
            cash -= (c_intr + p_intr) * shares  # PAY both intrinsics (premiums received)
            pnl = (position['call']['premium'] + position['put']['premium']
                   - c_intr - p_intr) * shares
            wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
            position = None
            trades.append({'date': date, 'price': settle_price, 'action': 'expiration', 'pnl': pnl})
        elif day is not None:
            # Mark both legs to market (hold-to-expiry: no early close).
            cq = day['marks'].get(position['call']['contract'])
            pq = day['marks'].get(position['put']['contract'])
            if cq is not None:
                position['call']['last_mid'], position['call']['delta'] = cq[2], cq[3]
            if pq is not None:
                position['put']['last_mid'], position['put']['delta'] = pq[2], pq[3]

        # 3. Delta-neutral rebalance on the COMBINED delta. Hold
        #    (call_delta + put_delta)*shares of stock (clamped to [-1, 1]) to
        #    neutralize the short straddle's -(call_delta + put_delta) position delta.
        if position is not None:
            combined = position['call']['delta'] + position['put']['delta']
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

        # 4. Mark to market: cash + hedge stock - value of BOTH short legs.
        equity = cash + price * hedge_shares
        if position is not None:
            equity -= (position['call']['last_mid'] + position['put']['last_mid']) * shares
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price,
                           'rf_credit': day_rf_credit})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price', 'rf_credit'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    net_pnl = final_equity - capital
    alpha_vs_cash = net_pnl - interest_earned

    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())

    summary = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'final_equity': round(final_equity, 2),
        'net_pnl': round(net_pnl, 2),
        'alpha_vs_cash': round(alpha_vs_cash, 2),
        'interest_earned': round(interest_earned, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'total_hedge_cost': round(total_hedge_cost, 2),
        'hedge_cost_bps': hedge_cost_bps,
        'num_straddles_sold': num_straddles_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        'max_drawdown_pct': round(max_dd, 2),
        'risk_free_rate': rf,
        'cash': round(capital, 2),
    }
    return summary, trades, daily_equity


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


def run_real_iron_condor_overlay(
    dates: list[str],
    prices: list[float],
    store: dict[str, dict[str, Any]],
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Daily short IRON CONDOR — short ~0.25d strangle + long ~0.10d wings, same
    expiry, hold-to-expiry, NO stock hedge. The defined-risk, practical retail
    short-vol structure: the long wings cap the tail the naked straddle leaves open,
    at the cost of premium plus four legs of bid/ask.

    EXPLORATORY, not registered. There is NO delta hedge (a static, defined-risk
    position — delta-neutral only at entry, then it rides between the strikes), so the
    verdict is a PRACTICAL-strategy excess-over-cash Newey-West t / Sharpe via
    short_vol_statistics, NOT the rate-invariant delta-hedged-gain measure the put/call
    wings and straddle use. A separate loop, so the frozen hedged engines (and every
    pin on them) stay byte-identical. Shorts fill at the bid and longs at the ask
    (`bid_ask`) or both at the mid (`mid`, frictionless); each leg keeps
    COMMISSION_PER_SHARE. params: dte (30), capital (100_000), fill, short_delta
    (0.25), wing_delta (0.10), risk_free_rate (0.045).
    """
    dte = int(params.get('dte', 30))
    fill = str(params.get('fill', 'bid_ask'))
    capital = float(params.get('capital', 100_000))
    short_delta = float(params.get('short_delta', 0.25))
    wing_delta = float(params.get('wing_delta', 0.10))
    rf = float(params.get('risk_free_rate', 0.045))
    daily_rf = rf / 252.0

    initial_price = prices[0]
    num_contracts = int(capital // (initial_price * 100))
    if num_contracts < 1:
        raise ValueError('capital insufficient for one contract')
    shares = 100 * num_contracts

    cash = capital
    position: dict[str, Any] | None = None
    num_condors_sold = 0
    total_premium_collected = 0.0  # net credit collected
    interest_earned = 0.0
    wins = losses = 0
    trades: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    prev_date: str | None = None
    prev_price: float | None = None

    for i, (date, price) in enumerate(zip(dates, prices)):
        day = store.get(date)
        day_rf_credit = 0.0
        if i > 0:
            day_rf_credit = cash * daily_rf
            cash += day_rf_credit
            interest_earned += day_rf_credit

        if position is None:
            if day is not None:
                pick = select_iron_condor(day, dte, short_delta, wing_delta)
                if pick is not None:
                    sc, lc, sp, lp = pick
                    sc_in = sc[2] if fill == 'bid_ask' else sc[4]  # sell shorts at bid
                    sp_in = sp[2] if fill == 'bid_ask' else sp[4]
                    lc_in = lc[3] if fill == 'bid_ask' else lc[4]  # buy longs at ask
                    lp_in = lp[3] if fill == 'bid_ask' else lp[4]
                    net_credit = (sc_in + sp_in) - (lc_in + lp_in) - 4 * COMMISSION_PER_SHARE
                    if net_credit > 0:
                        cash += net_credit * shares  # RECEIVE the net credit
                        position = {
                            'expiration': sc[5], 'credit': net_credit,
                            'sc': {'k': sc[6], 'cid': sc[7], 'mid': sc[4]},
                            'lc': {'k': lc[6], 'cid': lc[7], 'mid': lc[4]},
                            'sp': {'k': sp[6], 'cid': sp[7], 'mid': sp[4]},
                            'lp': {'k': lp[6], 'cid': lp[7], 'mid': lp[4]},
                        }
                        num_condors_sold += 1
                        total_premium_collected += net_credit * shares
                        trades.append({'date': date, 'action': 'sell', 'net_credit': net_credit,
                                       'short_call_k': sc[6], 'long_call_k': lc[6],
                                       'short_put_k': sp[6], 'long_put_k': lp[6], 'expiration': sc[5]})
        elif date >= position['expiration']:
            if date == position['expiration']:
                settle = price
            else:
                assert prev_date is not None and prev_price is not None
                gap = (pd.Timestamp(position['expiration']) - pd.Timestamp(prev_date)).days
                assert gap <= 4, (f'{gap} days between {prev_date} and expiration '
                                  f'{position["expiration"]} — missing data?')
                settle = prev_price
            sc_i = max(0.0, settle - position['sc']['k'])   # short call: we PAY
            lc_i = max(0.0, settle - position['lc']['k'])   # long call: we RECEIVE
            sp_i = max(0.0, position['sp']['k'] - settle)   # short put: we PAY
            lp_i = max(0.0, position['lp']['k'] - settle)   # long put: we RECEIVE
            net_settle = (-sc_i + lc_i - sp_i + lp_i)
            cash += net_settle * shares
            pnl = (position['credit'] + net_settle) * shares
            wins, losses = (wins + 1, losses) if pnl >= 0 else (wins, losses + 1)
            position = None
            trades.append({'date': date, 'price': settle, 'action': 'expiration', 'pnl': pnl})
        elif day is not None:
            for leg in ('sc', 'lc', 'sp', 'lp'):
                q = day['marks'].get(position[leg]['cid'])
                if q is not None:
                    position[leg]['mid'] = q[2]

        # Mark to market: cash - SHORT legs (liabilities) + LONG legs (assets).
        equity = cash
        if position is not None:
            equity -= (position['sc']['mid'] + position['sp']['mid']) * shares
            equity += (position['lc']['mid'] + position['lp']['mid']) * shares
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price,
                           'rf_credit': day_rf_credit})
        prev_date, prev_price = date, price

    daily_equity = pd.DataFrame(daily_rows, columns=['date', 'equity', 'price', 'rf_credit'])
    final_equity = float(daily_equity['equity'].iloc[-1])
    net_pnl = final_equity - capital
    alpha_vs_cash = net_pnl - interest_earned

    eq = daily_equity['equity'].astype(float)
    peak = eq.cummax().clip(lower=capital)
    max_dd = float(((peak - eq) / peak * 100).max())

    summary = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'final_equity': round(final_equity, 2),
        'net_pnl': round(net_pnl, 2),
        'alpha_vs_cash': round(alpha_vs_cash, 2),
        'interest_earned': round(interest_earned, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'num_condors_sold': num_condors_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / (wins + losses) * 100, 1) if (wins + losses) else 0.0,
        'max_drawdown_pct': round(max_dd, 2),
        'risk_free_rate': rf,
        'cash': round(capital, 2),
    }
    return summary, trades, daily_equity


def _cli() -> None:
    """Preview run: python vol_premium.py SPY  (delta-neutral ATM short call)."""
    import sys

    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else 'SPY'
    from real_cc_backtest import CHAIN_CLEAN_START

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
