from __future__ import annotations

import itertools
import math
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ====================
# 1. Black-Scholes
# ====================

def normal_pdf(x: float) -> float:
    """The height of the bell curve at point x."""
    return math.exp(-x**2 / 2.0) / math.sqrt(2 * math.pi)

def normal_cdf(x: float) -> float:
    """
    Standard normal CDF Φ(x) — area under the bell curve from -∞ to x.

    Uses the identity Φ(x) = 0.5 · (1 + erf(x/√2)) and delegates to
    math.erf, which uses the C standard library's optimized rational/
    Chebyshev approximation (~15-16 decimals, near-machine-precision).

    The tutorial demonstrates the Abramowitz & Stegun 1964 polynomial
    approximation (~7 decimals) for pedagogical clarity — you can read
    the formula and see *why* it works. Here in production code we use
    math.erf because it's effectively exact: across hundreds of thousands
    of CDF calls in a backtest, A&S's 8th-decimal error compounds into
    a few cents of equity drift vs. the erf version.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> float:
    """
    Black-Scholes option price.

    Args:
        S: stock price
        K: strike price
        T: time to expiration (years)
        r: risk-free rate
        sigma: volatility (annualized)
        option_type: 'put' or 'call' (default: 'put')

    Returns:
        price: option premium
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    N_d1 = normal_cdf(d1)
    N_d2 = normal_cdf(d2)

    if option_type == 'put':
        price = K * math.exp(-r * T) * (1 - N_d2) - S * (1 - N_d1)
    else:  # call
        price = S * N_d1 - K * math.exp(-r * T) * N_d2

    return price

def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'put') -> float:
    """
    Black-Scholes delta (probability of ITM at expiration).

    Args:
        option_type: 'put' or 'call' (default: 'put')

    Returns:
        delta: -1 to 0 for puts, 0 to 1 for calls
    """
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if option_type == 'put':
        delta = normal_cdf(d1) - 1
    else:  # call
        delta = normal_cdf(d1)

    return delta

def find_strike_for_delta(
    S: float, T: float, r: float, sigma: float, target_delta: float, option_type: str = 'put'
) -> float:
    """
    Grid search to find the whole-dollar strike with delta closest to target.

    Real option chains use whole-dollar strikes (e.g., $370, $375, $380).
    Grid search naturally produces whole-dollar results because it checks
    every integer in the range.

    Returns:
        float: strike price (whole dollar amount)
    """
    best_strike = S
    best_diff = float('inf')

    if option_type == 'put':
        # Puts: search below spot (80% to 102%) — puts are OTM when strike < spot.
        start = int(S * 0.80)
        end = int(S * 1.02)
    else:
        # Calls: search above spot (98% to 125%) — calls are OTM when strike > spot.
        start = int(S * 0.98)
        end = int(S * 1.25)

    for k in range(start, end + 1):
        K = float(k)  # Each k is already a whole dollar; cast for downstream math
        delta = bs_delta(S, K, T, r, sigma, option_type=option_type)

        # Track which strike has delta closest to target. abs() handles both
        # signs (put delta is negative, call delta positive); we minimize the
        # absolute gap so the comparison works for either option type.
        diff = abs(delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_strike = K

    return best_strike

# ====================
# 2. Volatility
# ====================

def calc_rolling_volatility(prices: NDArray[np.floating[Any]], window: int = 30) -> NDArray[np.floating[Any]]:
    """
    Calculate rolling historical volatility (annualized).

    Args:
        prices: array of daily closing prices
        window: lookback (default 30 days)

    Returns:
        vols: array of annualized volatilities, NaN until window fills
    """
    # Log returns: ln(price_t / price_{t-1}) = diff(log(prices)). This
    # works because ln(a) - ln(b) = ln(a/b). Log returns are additive
    # across days and symmetric (+5% then -5% nets to zero). Order
    # matters — log(diff(prices)) is NOT the same thing and breaks on
    # negative price changes.
    log_returns = np.diff(np.log(prices))

    # Rolling sample std dev with Bessel's correction (ddof=1) because
    # these returns are a sample from the stock's theoretical distribution,
    # not the population — dividing by N-1 avoids underestimating.
    # rolling(window).std() emits NaN until the window is full, keeping
    # output index-aligned with log_returns.
    #
    # Annualize with √252: variance is additive over independent periods
    # (σ²_annual = σ²_daily × 252), so std dev scales with √time —
    # multiply by √252, NOT 252.
    # pandas-stubs types `Series.rolling()` as `Rolling[Series[Unknown]]` even
    # when the source Series has a known dtype, which then leaks into
    # `.to_numpy()`. Silence the two affected sites and `cast` the final
    # ndarray so downstream typing stays sharp.
    vol = pd.Series(log_returns).rolling(window).std(ddof=1) * math.sqrt(252)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    return cast('NDArray[np.floating[Any]]', vol.to_numpy())  # pyright: ignore[reportUnknownMemberType]

def detect_regime(rolling_vol: float) -> str:
    """Classify volatility regime based on current HV level."""
    if rolling_vol > 0.25:
        return 'high'
    elif rolling_vol < 0.15:
        return 'low'
    else:
        return 'normal'

def estimate_iv(rolling_vol: float, regime: str | None = None) -> float:
    """
    Apply a regime-based multiplier to convert HV → IV estimate.

    High vol (>25%) → 1.1× (IV already elevated; further expansion is limited)
    Normal (15-25%) → 1.3× (typical HV→IV relationship)
    Low vol (<15%)  → 1.5× (IV is suppressed; expect mean reversion to higher values)

    Args:
        rolling_vol: historical volatility (annualized) for the latest window.
        regime: optional pre-classified regime ('high', 'normal', or 'low').
            If omitted, `detect_regime(rolling_vol)` is called internally so
            callers only need to pass the vol.

    Returns:
        iv: estimated implied volatility.
    """
    if regime is None:
        regime = detect_regime(rolling_vol)
    if regime == 'high':
        multiplier = 1.1
    elif regime == 'normal':
        multiplier = 1.3
    else:  # low
        multiplier = 1.5
    return rolling_vol * multiplier

# ====================
# 3. Overlay Engine (Covered Call)
# ====================

def run_cc_overlay(
    dates: list[str] | NDArray[Any],
    prices: NDArray[np.floating[Any]],
    params: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """
    Simulate a covered call overlay strategy from start to finish.

    Args:
        dates: array of datetime objects
        prices: array of daily closing prices
        params: dict with keys:
            - call_delta: target delta for strike selection (e.g., 0.25)
            - close_at_pct: close when this % of premium captured (e.g., 0.75)
            - dte: days to expiration when opening position (e.g., 21)
            - risk_free_rate: annual risk-free rate (e.g., 0.045)
            - capital: total dollars committed to the portfolio. Sized into
              whole 100-share contracts at initial_price; any leftover sits
              as uninvested cash (0% yield). Default: cost of 1 contract.
            - delta_hedge: when True, run the Israelov-Nielsen risk-managed
              covered call. Each day, hold extra shares equal to the short
              call's current delta × base shares so the portfolio's net
              delta stays pinned at the buy-and-hold equivalent. Strips out
              the equity-timing exposure that adds variance without adding
              return. Default: False (naive covered call). Hedge purchases
              draw from `cash` (which may go negative — a zero-interest
              financing simplification; in practice you would post margin).
              No transaction costs are modeled on hedge trades.

    IV is *not* a tunable parameter. It is computed internally each day
    from rolling 30-day historical volatility, then scaled by a
    regime-based multiplier (1.1× in high-vol regimes, 1.3× in normal,
    1.5× in low-vol) via detect_regime() and estimate_iv(). Any
    `iv_multiplier` key in `params` is silently ignored.

    Returns:
        (summary, trades, daily_equity)

        daily_equity is a DataFrame with columns ['date', 'equity', 'price'],
        one row per simulated day. Downstream consumers
        (compute_statistics, make_figures, walk-forward) index by column.
    """

    # Extract parameters from dict
    call_delta = params.get('call_delta', 0.25)
    close_at_pct = params.get('close_at_pct', 0.75)
    dte = params.get('dte', 21)
    r = params.get('risk_free_rate', 0.045)
    delta_hedge = bool(params.get('delta_hedge', False))

    initial_price = float(prices[0])
    contract_cost = initial_price * 100  # cost of one 100-share contract

    # Size the portfolio. Default: single contract (the original behavior).
    capital = float(params.get('capital', contract_cost))
    num_contracts = int(capital // contract_cost)
    if num_contracts < 1:
        raise ValueError(
            f"Capital ${capital:,.2f} insufficient for 1 contract "
            f"at ${initial_price:.2f}/share (need ${contract_cost:,.2f})"
        )
    shares = 100 * num_contracts                   # base shares held (covers the short calls)
    initial_stock_cost = shares * initial_price    # actual capital deployed in stock
    initial_cash = capital - initial_stock_cost    # leftover at t=0, 0% yield. Pinned for the
                                                   # buy-and-hold benchmark; compute_statistics
                                                   # reconstructs BH equity from
                                                   # shares × prices + initial_cash.
    current_cash = initial_cash                    # working cash account; drained/refilled by
                                                   # delta-hedge share trades when delta_hedge=True.

    # Risk-managed CC state. hedge_shares is the extra long-stock position held to offset the
    # short call's negative delta. When delta_hedge=False, both stay at 0 and the loop is the
    # naive covered call. Per Israelov & Nielsen (2015), this strips out the equity-timing
    # exposure that adds variance without adding return.
    hedge_shares = 0

    num_days = len(dates)
    trades: list[dict[str, Any]] = []
    # Accumulate daily snapshots as a list-of-dicts in the hot loop (the
    # natural append shape), then materialize a DataFrame once at the
    # return boundary so downstream consumers can index by column.
    daily_rows: list[dict[str, Any]] = []

    # State tracking
    position: dict[str, Any] | None = None
    realized_pnl = 0.0  # cumulative premium overlay P&L (excludes stock appreciation)
    num_calls_sold = 0
    total_premium_collected = 0.0
    wins = 0
    losses = 0

    # Precompute 30-day rolling annualized historical volatility for the entire
    # series so the daily loop is a constant-time lookup instead of re-running
    # diff + std on a fresh slice each day. min_periods=3 mirrors the original
    # warmup threshold (day_idx >= 3 produces a real std, earlier days fall back
    # to 20% — the long-run equity baseline). rolling_vol_series[i] corresponds
    # to log_returns[i], i.e. the return realized on day i+1, so on day_idx d we
    # look up index d-1; day 0 has no return yet and uses the fallback directly.
    # (pandas-stubs types `Series.rolling()` as `Rolling[Series[Unknown]]` even
    # when the source dtype is known; same noise pattern as calc_rolling_volatility.)
    log_returns_series = pd.Series(np.diff(np.log(prices)))
    rolling_vol_series = (  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        log_returns_series.rolling(30, min_periods=3).std(ddof=1) * math.sqrt(252)  # pyright: ignore[reportUnknownMemberType]
    ).fillna(0.20)

    for day_idx in range(num_days):
        date = dates[day_idx]
        price = float(prices[day_idx])

        # Look up precomputed 30-day rolling annualized vol; fall back to 20%
        # on day 0 when no return has been realized yet.
        rolling_vol = (
            float(rolling_vol_series.iloc[day_idx - 1])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            if day_idx > 0 else 0.20
        )

        if rolling_vol <= 0:
            # Degenerate case (e.g. perfectly constant prices over the window):
            # can't price an option with non-positive vol. NaN is no longer
            # possible thanks to fillna(0.20) above.
            continue

        # IV estimate: regime-based multiplier (1.1× high, 1.3× normal, 1.5× low).
        # estimate_iv() calls detect_regime() internally when no regime is passed.
        iv_estimate = estimate_iv(rolling_vol)

        # If no position, consider opening
        if position is None:
            # Sell a call
            T = dte / 252
            strike = find_strike_for_delta(price, T, r, iv_estimate, call_delta, option_type='call')
            premium = bs_price(price, strike, T, r, iv_estimate, option_type='call')

            # Apply transaction costs
            net_premium = premium * (1 - 0.03) - 0.0065  # 3% slippage, $0.65 commission

            # Skip if premium is too small after costs (low-vol periods where
            # the OTM call is nearly worthless and slippage + commission
            # exceed the gross premium → guaranteed loss).
            if net_premium <= 0:
                continue

            # Open position
            position = {
                'strike': strike,
                'premium_collected': net_premium,
                'entry_price': price,
                'entry_idx': day_idx,
                'entry_date': date,
            }
            num_calls_sold += 1
            total_premium_collected += net_premium * shares

            trades.append({
                'date': date,
                'price': price,
                'action': 'sell',
                'premium': net_premium,
                'strike': strike,
                'pnl': 0,
                'realized_pnl': realized_pnl,
            })

        else:
            # Position is open; check conditions.
            days_left = dte - (day_idx - position['entry_idx'])

            if days_left <= 0:
                # Expiration reached. Overlay P&L only — stock appreciation
                # is tracked separately by the daily equity calculation below.
                if price >= position['strike']:
                    # Called away (assignment): the buyer exercises the call
                    # and takes our shares at the strike. To stay in the
                    # overlay business (always own 100 shares), we immediately
                    # rebuy at the current market price.
                    #
                    # Cash flow per share: collect strike, pay current price.
                    # Net to overlay: premium_collected - (price - strike).
                    #
                    # Example (per share):
                    #   strike = $310, premium = $1.50, market = $325
                    #   pnl = $1.50 - ($325 - $310) = -$13.50  → assignment loss
                    # Or if the stock barely closed ITM:
                    #   strike = $310, premium = $1.50, market = $311
                    #   pnl = $1.50 - $1.00 = +$0.50  → small win
                    #
                    # An assignment is a LOSS for the overlay when the stock
                    # rallied past `strike + premium` — you collected premium
                    # but had to pay back the upside above strike. The stock
                    # appreciation up to `strike` is still kept (it's in the
                    # daily equity tracking), so you don't lose money overall;
                    # you just lose the *uncapped* portion of the rally.
                    pnl = (position['premium_collected'] - (price - position['strike'])) * shares
                else:
                    # Expired OTM: stock closed below strike, call is worthless,
                    # we keep the full premium and the shares.
                    pnl = position['premium_collected'] * shares

                realized_pnl += pnl
                if pnl >= 0:
                    wins += 1
                else:
                    losses += 1
                position = None

                trades.append({
                    'date': date,
                    'price': price,
                    'action': 'expiration',
                    'pnl': pnl,
                    'realized_pnl': realized_pnl,
                })

            else:
                # Check profit target or early close
                T_remaining = days_left / 252
                call_value_today = bs_price(price, position['strike'], T_remaining, r, iv_estimate, option_type='call')
                profit_pct = (position['premium_collected'] - call_value_today) / position['premium_collected']

                # Close if profit target reached (close_at_pct of premium captured)
                if call_value_today <= position['premium_collected'] * (1 - close_at_pct):
                    # Buy back the call
                    pnl = (position['premium_collected'] - call_value_today) * shares - 0.65 * num_contracts
                    realized_pnl += pnl
                    if pnl >= 0:
                        wins += 1
                    else:
                        losses += 1
                    position = None

                    trades.append({
                        'date': date,
                        'price': price,
                        'action': 'close',
                        'call_value': call_value_today,
                        'profit_pct': profit_pct,
                        'pnl': pnl,
                        'realized_pnl': realized_pnl,
                    })

                else:
                    # Deep ITM check: if delta > 0.70, the call is almost
                    # certainly going to be assigned. Close now to free up
                    # capital rather than riding gamma risk into expiration.
                    delta_today = bs_delta(price, position['strike'], T_remaining, r, iv_estimate, option_type='call')
                    if delta_today > 0.70:
                        pnl = (position['premium_collected'] - call_value_today) * shares - 0.65 * num_contracts
                        realized_pnl += pnl
                        if pnl >= 0:
                            wins += 1
                        else:
                            losses += 1
                        position = None

                        trades.append({
                            'date': date,
                            'price': price,
                            'action': 'close_itm',
                            'call_value': call_value_today,
                            'pnl': pnl,
                            'realized_pnl': realized_pnl,
                        })

        # === Risk-managed (delta-hedged) rebalance ===
        # When delta_hedge=True, hold extra long-stock equal to the call's current delta times
        # base shares so net portfolio delta stays pinned at `shares` (the buy-and-hold target).
        # This is the Israelov-Nielsen fix: strip out the equity-timing exposure that adds
        # variance without contributing return. When delta_hedge=False, the target stays at 0
        # and nothing trades — the loop reduces to the naive covered call.
        if delta_hedge and position is not None:
            days_left_h = dte - (day_idx - position['entry_idx'])
            if days_left_h > 0:
                T_remaining_h = days_left_h / 252
                call_delta_today = bs_delta(
                    price, position['strike'], T_remaining_h, r, iv_estimate, option_type='call'
                )
                target_hedge_shares = int(round(call_delta_today * shares))
            else:
                # Position settled this iteration (expiration branch zeroed `position` only on
                # the close path — but if we got here, `position is not None`, so days_left_h
                # > 0 must hold). Defensive: fall back to no hedge.
                target_hedge_shares = 0
        else:
            target_hedge_shares = 0

        hedge_trade = target_hedge_shares - hedge_shares
        if hedge_trade != 0:
            # Buy (trade > 0) or sell (trade < 0) at the current market price. Cash absorbs the
            # cost; cash may go negative (zero-interest financing simplification, no slippage
            # or commission modeled on hedge trades — share legs are highly liquid).
            current_cash -= hedge_trade * price
            hedge_shares = target_hedge_shares

        # Track daily equity: total stock value + working cash + cumulative overlay P&L.
        # total_shares includes hedge_shares when delta_hedge=True; otherwise it's just `shares`.
        # Returns are measured against `capital` (the total committed dollars).
        total_shares = shares + hedge_shares
        stock_value = price * total_shares
        equity = stock_value + current_cash + realized_pnl
        if position is not None:
            days_left = dte - (day_idx - position['entry_idx'])
            T_remaining = max(days_left / 252, 0)
            call_value = bs_price(price, position['strike'], T_remaining, r, iv_estimate, option_type='call')
            # The short call covers `shares` base shares (one contract per 100 base shares).
            equity += (position['premium_collected'] - call_value) * shares
        daily_rows.append({'date': date, 'equity': round(equity, 2), 'price': price})

    # Materialize the per-day snapshots as a DataFrame once at the return
    # boundary. Schema: ['date', 'equity', 'price'], one row per simulated
    # day. Empty input (no days produced any row) becomes an empty DF with
    # the same columns so column access downstream still works.
    daily_equity: pd.DataFrame = (
        pd.DataFrame(daily_rows, columns=['date', 'equity', 'price'])
        if daily_rows
        else pd.DataFrame({'date': [], 'equity': pd.Series([], dtype=float), 'price': pd.Series([], dtype=float)})
    )

    # Compute summary stats
    final_equity: float = (
        float(daily_equity['equity'].iloc[-1])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        if not daily_equity.empty else capital
    )
    total_return = (final_equity - capital) / capital * 100

    # Buy-and-hold benchmark: hold the same `shares` for the whole period
    # without selling calls (and without any delta hedge). Initial idle cash
    # sits at 0% in both scenarios so it cancels in the excess-return
    # comparison. We use `initial_cash` (not `current_cash`) so the benchmark
    # is unaffected by hedge-trade cash flows under delta_hedge=True.
    final_price = float(prices[-1])
    buy_hold_final = final_price * shares + initial_cash
    buy_hold_return = (buy_hold_final - capital) / capital * 100
    excess_return = total_return - buy_hold_return

    # Decompose the overlay's contribution: we collected `total_premium_collected`
    # in gross premium across all sells, but had to pay it back via buybacks
    # (early closes at profit target / ITM) and assignment losses (when called
    # away above strike). The net overlay P&L equals the gap between final
    # equity and the buy-and-hold final value.
    net_overlay_pnl = final_equity - buy_hold_final
    premium_retention = (net_overlay_pnl / total_premium_collected * 100
                        if total_premium_collected > 0 else 0.0)

    # Pre-round the two independent values, then derive `overlay_costs` from the
    # already-rounded inputs so the accounting identity
    #   total_premium_collected - overlay_costs == net_overlay_pnl
    # holds exactly at 2-decimal precision. Rounding each value independently
    # would let the identity drift by up to ~1.5¢ from accumulated rounding
    # error (each round can shift its input by up to 0.5¢).
    total_premium_collected_r = round(total_premium_collected, 2)
    net_overlay_pnl_r = round(net_overlay_pnl, 2)
    overlay_costs_r = round(total_premium_collected_r - net_overlay_pnl_r, 2)

    # Max drawdown: track running peak (seeded at starting capital so a
    # day-1 dip below initial equity still registers a drawdown), then
    # take the worst peak-to-equity gap as a percentage of peak. cummax
    # gives the running max across daily equity; clipping at `capital`
    # ensures the peak never drops below the starting baseline. (Same
    # pandas-stubs Series[Unknown] noise pattern as the rolling-vol path
    # — we annotate explicitly to keep the silencing scoped.)
    if daily_equity.empty:
        max_dd = 0.0
    else:
        equity_series: pd.Series[float] = daily_equity['equity'].astype(float)  # pyright: ignore[reportUnknownMemberType, reportAssignmentType]
        peak_series = equity_series.cummax().clip(lower=capital)  # pyright: ignore[reportUnknownMemberType]
        max_dd = float(((peak_series - equity_series) / peak_series * 100).max())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

    summary: dict[str, Any] = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'initial_stock_cost': round(initial_stock_cost, 2),
        # Initial leftover cash, NOT the working balance — compute_statistics rebuilds the
        # buy-and-hold curve from `shares × prices + cash`, which only makes sense with the
        # constant initial cash. Under delta_hedge=True the working cash drifts as hedge
        # trades execute; that drift shows up in `final_equity` via the daily equity series.
        'cash': round(initial_cash, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_final': round(buy_hold_final, 2),
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'excess_return_pct': round(excess_return, 2),
        'net_overlay_pnl': net_overlay_pnl_r,
        'total_premium_collected': total_premium_collected_r,
        'overlay_costs': overlay_costs_r,
        'premium_retention_pct': round(premium_retention, 1),
        'num_calls_sold': num_calls_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / max(wins + losses, 1) * 100, 1),
        'max_drawdown_pct': round(max_dd, 2),
    }

    return summary, trades, daily_equity

# ====================
# 4. Statistical Significance
# ====================

def compute_statistics(
    daily_equity: pd.DataFrame,
    num_contracts: int,
    cash: float,
    periods_per_year: int = 252,
) -> dict[str, Any]:
    """
    Test whether the overlay's excess return over buy-and-hold is
    statistically distinguishable from zero.

    The null hypothesis is: the overlay adds zero value compared to
    simply holding the stock. We reject (i.e., conclude the overlay
    does something) when the Newey-West-adjusted t-statistic is large
    in absolute value.

    Two t-stats are reported:

    - `t_stat_naive` assumes daily excess returns are IID (independent
      and identically distributed). That assumption is violated for
      overlay strategies because the same option position drives
      multiple consecutive days of P&L — so naive standard errors are
      too small and naive t-stats are inflated.

    - `t_stat_newey_west` uses Newey-West HAC (heteroskedasticity and
      autocorrelation consistent) standard errors. Lag cutoff
      L = floor(4 * (n/100)^(2/9)) — the framework is from Andrews
      (1991); this specific operational formula is from Newey & West
      (1994). This is the correct statistic for an overlay.

    Interpretation thresholds (Harvey, Liu & Zhu 2016):
        |t_NW| > 3.0  → likely a real effect after multiple-testing
                        adjustment for the factor zoo
        |t_NW| > 2.0  → "significant" by convention, but weak evidence
                        when many parameter combinations were tested
        |t_NW| < 2.0  → not reliably different from noise

    Args:
        daily_equity: output of run_cc_overlay (DataFrame with columns
            'date', 'equity', 'price').
        num_contracts: number of option contracts in the portfolio
            (each represents 100 shares). From summary['num_contracts'].
        cash: leftover uninvested cash from initial sizing. From
            summary['cash'].
        periods_per_year: annualization factor (252 for daily data).

    Returns:
        dict with t-stats, annualized excess return, Sharpe ratio, and
        pass/fail flags for the t=2 and t=3 thresholds.
    """
    shares = num_contracts * 100

    # Reconstruct two equity curves from the same daily series.
    # The overlay curve includes mark-to-market on the short call;
    # the buy-and-hold curve is just stock value plus idle cash.
    # (pandas-stubs degrades Series.to_numpy() to ndarray[Unknown, Unknown];
    # cast back to the float ndarray we actually have.)
    equity = cast(
        'NDArray[np.float64]',
        daily_equity['equity'].to_numpy(dtype=float),  # pyright: ignore[reportUnknownMemberType]
    )
    prices = cast(
        'NDArray[np.float64]',
        daily_equity['price'].to_numpy(dtype=float),  # pyright: ignore[reportUnknownMemberType]
    )
    bh_equity = shares * prices + cash

    # Daily simple returns on each equity curve
    overlay_ret = np.diff(equity) / equity[:-1]
    bh_ret = np.diff(bh_equity) / bh_equity[:-1]

    # Excess returns: the part of return attributable to the overlay
    # alone (stock drift cancels). This is the series we test.
    excess = overlay_ret - bh_ret

    n = len(excess)
    if n < 2:
        raise ValueError(f"Need at least 2 daily observations, got {n}")

    mean_e = float(np.mean(excess))
    var_e = float(np.var(excess, ddof=1))

    # Naive t-stat: SE = sigma / sqrt(n). Assumes IID.
    se_naive = math.sqrt(var_e / n) if var_e > 0 else 0.0
    t_naive = mean_e / se_naive if se_naive > 0 else 0.0

    # Newey-West: variance of the mean under autocorrelation.
    #   Var(mean) = (1/n) * [gamma_0 + 2 * sum_{k=1}^{L} w_k * gamma_k]
    # where gamma_k is the k-th autocovariance and w_k = 1 - k/(L+1)
    # are the Bartlett weights that enforce positive-definiteness.
    L = int(4 * (n / 100) ** (2 / 9))
    nw_sum = 0.0
    for k in range(1, L + 1):
        weight = 1.0 - k / (L + 1)
        # autocovariance at lag k (demeaned)
        cov_k = float(np.mean((excess[:-k] - mean_e) * (excess[k:] - mean_e)))
        nw_sum += weight * cov_k
    var_mean_nw = (var_e + 2 * nw_sum) / n
    # Newey-West variance can be non-positive at short samples; floor at
    # zero so se_nw == 0 trips the guard below and we report t_nw = 0.
    se_nw = math.sqrt(max(var_mean_nw, 0.0))
    t_nw = mean_e / se_nw if se_nw > 0 else 0.0

    # Annualized context
    ann_excess_return = mean_e * periods_per_year
    ann_excess_vol = math.sqrt(var_e * periods_per_year)
    sharpe_excess = ann_excess_return / ann_excess_vol if ann_excess_vol > 0 else 0.0

    return {
        'n_days': n,
        'years_of_data': round(n / periods_per_year, 2),
        'ann_excess_return_pct': round(ann_excess_return * 100, 3),
        'ann_excess_vol_pct': round(ann_excess_vol * 100, 2),
        'sharpe_excess': round(sharpe_excess, 3),
        't_stat_naive': round(t_naive, 2),
        't_stat_newey_west': round(t_nw, 2),
        'nw_lag': L,
        'passes_t_2': abs(t_nw) > 2.0,
        'passes_t_3': abs(t_nw) > 3.0,
    }


# ====================
# 5. Regime Analysis
# ====================

def classify_regime(
    prices: pd.Series[float] | NDArray[np.floating[Any]] | list[float],
    window: int = 200,
    threshold: float = 0.05,
) -> pd.Series[str]:
    """
    Classify the market regime at each index of a price series using a
    trailing-`window`-day simple moving average with ±`threshold` bands:

      - 'bull'     if the close is > SMA × (1 + threshold)
      - 'bear'     if the close is < SMA × (1 − threshold)
      - 'sideways' if the close is within ±threshold of the SMA
      - 'unknown'  for the first `window` − 1 indices, where the SMA is
                   undefined (rolling-window warmup)

    Each index i is classified using prices through index i, inclusive
    (today's close vs the SMA of today and the prior `window` − 1 days).
    For "no future peeking" semantics — the regime as known at the
    *start* of day i, using only prices through day i − 1 — apply
    `.shift(1)` to the result. See `regime_analysis` for that pattern.

    Args:
        prices: chronological price series (pd.Series, ndarray, or list).
        window: SMA lookback in trading days (default 200, ~1 year).
        threshold: fractional band around the SMA that counts as
            'sideways' (default 0.05 = ±5%).

    Returns:
        pd.Series of regime labels (dtype object), one per input price.
        To get the scalar regime at the end of the series, take
        `.iloc[-1]`. An empty input returns an empty Series.
    """
    p: pd.Series[float] = pd.Series(np.asarray(prices, dtype=float), dtype=float)
    sma: pd.Series[float] = p.rolling(window).mean()  # pyright: ignore[reportUnknownMemberType, reportAssignmentType, reportUnknownVariableType]

    # Default to 'unknown', then mark every row that has a valid SMA as
    # the in-band ('sideways') case, then overwrite out-of-band rows
    # with 'bull' / 'bear'. NaN-comparison semantics return False, so
    # the warmup region (where the SMA is NaN) is never reassigned and
    # 'unknown' is preserved there automatically.
    regimes: pd.Series[str] = pd.Series('unknown', index=p.index, dtype=object)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    regimes[sma.notna()] = 'sideways'
    regimes[p > sma * (1.0 + threshold)] = 'bull'
    regimes[p < sma * (1.0 - threshold)] = 'bear'
    return regimes


def regime_analysis(
    dates: list[str] | NDArray[Any],
    prices: NDArray[np.floating[Any]] | list[float],
    trades: list[dict[str, Any]],
    window: int = 200,
    threshold: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    """
    Aggregate the overlay's realized P&L by market regime.

    For each day i, classifies the regime using `prices[:i]` only —
    strictly past prices, no peeking at today's close. For each closed
    trade, looks up the regime on the trade's close date and adds that
    trade's P&L to the matching regime bucket. The first `window` days
    are classified as "unknown" because the SMA needs `window`
    observations to compute.

    Args:
        dates: chronological list of date labels matching `prices`.
        prices: chronological array of closing prices.
        trades: list of trade dicts from `run_cc_overlay`, each with
            at least 'date' and 'pnl' keys. Only trades with non-zero
            pnl contribute (i.e., close/expiration/close_itm events).
        window: SMA lookback for regime classification (default 200).
        threshold: ±-band around the SMA for "sideways" (default 0.05).

    Returns:
        Dict keyed by 'bull', 'bear', 'sideways', 'unknown' with:
          - days: number of days classified as this regime
          - total_pnl: sum of trade pnls that closed in this regime
          - avg_pnl_per_day: total_pnl / days (0 if days == 0)
    """
    # Classify the regime at each day using only data up to that day
    # (no future peeking). classify_regime returns the regime at each
    # index using prices through that index inclusive; the .shift(1)
    # is what enforces "use only prices known at the start of day i" —
    # at index i it surfaces the regime computed from prices[:i]
    # (yesterday's close and earlier). The shift introduces one
    # leading NaN at index 0; .fillna('unknown') matches
    # classify_regime's insufficient-history convention so the warmup
    # region uniformly reads 'unknown'.
    regimes: pd.Series[str] = (
        classify_regime(prices, window, threshold)
        .shift(1)
        .fillna('unknown')  # pyright: ignore[reportUnknownMemberType]
    )

    # Per-regime day count. value_counts omits regimes with zero days;
    # reindex restores any missing buckets so all four keys are always
    # present in the result, matching the original loop's pre-init.
    # (pandas-stubs' reindex/to_dict overloads degrade to Unknown when
    # composed off Series[str], same noise pattern as the rolling-vol
    # chain — we annotate explicitly and suppress.)
    day_counts: dict[str, int] = cast(
        'dict[str, int]',
        regimes.value_counts()
        .reindex(['bull', 'bear', 'sideways', 'unknown'], fill_value=0)  # pyright: ignore[reportUnknownMemberType]
        .to_dict(),  # pyright: ignore[reportUnknownMemberType]
    )

    # Bucket each closed trade's pnl into the regime active on its
    # close date. Trades with pnl == 0 (open events with no realized
    # P&L) are filtered out up front; trades whose date isn't in
    # `dates` map to NaN and are dropped by groupby. reindex backfills
    # any regime that saw no trades with 0.0 so all four keys exist.
    # e.g., trade on a 'bull'-classified day with pnl=$120 →
    #   regime_pnl['bull'] += 120.
    trades_df = pd.DataFrame(trades, columns=['date', 'pnl'])
    nonzero = trades_df[trades_df['pnl'] != 0]
    date_to_regime: dict[str, str] = dict(zip(dates, regimes.tolist()))  # pyright: ignore[reportUnknownArgumentType]
    regime_pnl: dict[str, float] = cast(
        'dict[str, float]',
        nonzero['pnl']
        .groupby(nonzero['date'].map(date_to_regime))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        .sum()
        .reindex(['bull', 'bear', 'sideways', 'unknown'], fill_value=0.0)  # pyright: ignore[reportUnknownMemberType]
        .to_dict(),  # pyright: ignore[reportUnknownMemberType]
    )

    # Build the per-regime summary. avg_pnl_per_day guards against
    # division-by-zero for any empty regime. day_counts and regime_pnl
    # both contain all four keys by construction (reindex), so the
    # dict lookups below are safe.
    return {
        regime: {
            'days': day_counts[regime],
            'total_pnl': round(regime_pnl[regime], 2),
            'avg_pnl_per_day': round(
                regime_pnl[regime] / day_counts[regime]
                if day_counts[regime] > 0
                else 0.0,
                2,
            ),
        }
        for regime in ('bull', 'bear', 'sideways', 'unknown')
    }


# ====================
# 6. Walk-Forward Optimization
# ====================

def _param_combinations(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    """Cartesian product of a parameter grid.

    Input:  {'call_delta': [0.15, 0.25], 'dte': [21, 30]}
    Output: [{'call_delta': 0.15, 'dte': 21},
             {'call_delta': 0.15, 'dte': 30},
             {'call_delta': 0.25, 'dte': 21},
             {'call_delta': 0.25, 'dte': 30}]
    """
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def walk_forward_optimization(
    dates: list[str],
    prices: NDArray[np.floating[Any]] | list[float],
    param_grid: dict[str, list[float]],
    fixed_params: dict[str, float] | None = None,
    train_years: int = 2,
    test_months: int = 6,
    roll_months: int = 6,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Walk-forward optimization for the covered-call overlay strategy.

    Args:
        dates: chronological list of 'YYYY-MM-DD' date strings.
        prices: chronological array of closing prices matching `dates`.
        param_grid: dict of parameter combinations to test, e.g.
            `{'call_delta': [0.15, 0.20, 0.25], 'dte': [21, 30, 45],
              'close_at_pct': [0.50, 0.75, 1.0]}`.
        fixed_params: parameters held constant across every combo
            (default: `{'risk_free_rate': 0.045, 'capital': 100_000}`).
        train_years: in-sample training window length in years
            (default 2).
        test_months: out-of-sample test window length in months
            (default 6).
        roll_months: how far to advance between iterations in months
            (default 6, i.e. non-overlapping test windows).

    Returns:
        oos_equity: stitched out-of-sample equity curve as a DataFrame
            with columns ['date', 'equity', 'price'], one row per
            test-window day across all iterations (concatenated in time
            order). Empty DataFrame with those columns if no test
            window produced a result.
        period_records: list of dicts describing each iteration —
            train/test bounds (ISO date strings), the chosen
            `best_params`, and the in-sample `train_sharpe` that won.
    """
    if fixed_params is None:
        fixed_params = {'risk_free_rate': 0.045, 'capital': 100_000}

    # Convert to pandas for easier date slicing. (pandas-stubs types a couple
    # of these signatures loosely — `pd.to_datetime` and `Series.min/max` — so
    # we `cast`/`pyright: ignore` just those two spots; everything downstream
    # is plain `pd.Timestamp` arithmetic and slicing.)
    df = pd.DataFrame({'date': dates, 'price': np.asarray(prices, dtype=float)})
    df['date'] = pd.to_datetime(df['date'])  # pyright: ignore[reportUnknownMemberType]

    # Collect each OOS window's daily_equity DataFrame; concatenate at the
    # end so callers receive a single stitched curve.
    oos_frames: list[pd.DataFrame] = []
    best_params_per_period: list[dict[str, Any]] = []

    # First date in dataset (e.g., Apr 2014)
    start_date = cast('pd.Timestamp', df['date'].min())  # pyright: ignore[reportUnknownMemberType]
    # Last date in dataset (e.g., Apr 2026)
    end_date = cast('pd.Timestamp', df['date'].max())  # pyright: ignore[reportUnknownMemberType]
    # The "knife" between train and test.
    # We start train_years in so there's enough history for the first training window.
    # Example: start_date = Apr 2014, train_years = 2 → current_date = Apr 2016
    current_date = start_date + pd.DateOffset(years=train_years)

    # Keep rolling as long as there's enough data left for a complete test window.
    # If the test window would run past end_date, stop — no partial test periods.
    while current_date + pd.DateOffset(months=test_months) <= end_date:

        # current_date carves out two non-overlapping windows each iteration:
        #   train_start ←— train_years —→ train_end/test_start ←— test_months —→ test_end
        #                                       ↑ current_date
        #
        # Iter 1: [Apr 2014 – Apr 2016] train → [Apr 2016 – Oct 2016] test
        # Iter 2: [Oct 2014 – Oct 2016] train → [Oct 2016 – Apr 2017] test
        # Iter 3: [Apr 2015 – Apr 2017] train → [Apr 2017 – Oct 2017] test

        # Look BACKWARD
        train_start = current_date - pd.DateOffset(years=train_years)
        train_end = current_date
        # Look FORWARD
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=test_months)
        # train_end == test_start: windows touch but never overlap.
        # This is the key guarantee — we never test on data we trained on.

        # Slice the dataframe into train/test sets using boolean indexing:
        #   df['date'] >= train_start  → True/False for every row (is this date on or after start?)
        #   df['date'] < train_end     → True/False for every row (is this date before end?)
        #   &                          → combine: only rows where BOTH are True
        #   df[...]                    → keep only those True rows
        #
        # We use >= (inclusive) on the left and < (exclusive) on the right so that
        # the boundary date (current_date) belongs to the TEST set, not both.
        # Example: if current_date = Apr 2016, then Apr 2016 data goes to test_df,
        #          not train_df. No row appears in both sets.
        train_df = df[(df['date'] >= train_start) & (df['date'] < train_end)]
        test_df = df[(df['date'] >= test_start) & (df['date'] < test_end)]

        # Skip windows that don't have enough data to backtest meaningfully
        # (e.g., a calendar window that lands during a market closure).
        if len(train_df) < 30 or len(test_df) < 5:
            current_date += pd.DateOffset(months=roll_months)
            continue

        # === Step 1: OPTIMIZE on training data ("study for the test") ===
        best_sharpe = -float('inf')  # Initialize to negative infinity so any real Sharpe beats it
        best_params: dict[str, float] | None = None

        for combo in _param_combinations(param_grid):
            # Merge in the fixed params that don't change across combos
            # (risk_free_rate, capital). IV multiplier is regime-based
            # (detect_regime + estimate_iv), so we don't pass iv_multiplier.
            params = {**fixed_params, **combo}

            try:
                _summary, _trades, daily_eq = run_cc_overlay(  # Run backtest with these params
                    list(train_df['date'].dt.strftime('%Y-%m-%d')),
                    np.asarray(train_df['price'].values, dtype=float),
                    params,
                )
            except Exception:
                continue

            # Daily simple returns on the in-sample equity curve. pct_change
            # drops one row at the head (no prior equity), matching the
            # original (i, i-1) loop. dropna handles the leading NaN; an
            # empty result (1-day window or all-NaN) trips the else branch.
            returns = cast(
                'list[float]',
                daily_eq['equity']
                .pct_change()  # pyright: ignore[reportUnknownMemberType]
                .dropna()  # pyright: ignore[reportUnknownMemberType]
                .tolist(),  # pyright: ignore[reportUnknownMemberType]
            )

            if returns:
                # 1. Average daily return: sum all daily returns, divide by count
                avg_return = sum(returns) / len(returns)

                # 2. Standard deviation (how bumpy the ride is), built inside-out:
                #    (r - avg_return)          → each day's deviation from the mean
                #    (r - avg_return) ** 2     → square it (so negatives don't cancel positives)
                #    sum(...)                  → total squared deviation
                #    / max(1, len(returns)-1)  → divide by N-1 (Bessel's correction: less biased
                #                                estimate from a sample vs. full population;
                #                                max(1,...) is a safety net against dividing by 0)
                #    math.sqrt(...)            → undo the squaring, back to return-sized units
                std_dev = math.sqrt(
                    sum((r - avg_return) ** 2 for r in returns) / max(1, len(returns) - 1)
                )

                # 3. Sharpe ratio: reward per unit of risk, annualized
                #    avg_return / std_dev      → daily Sharpe (return per unit of bumpiness)
                #    * math.sqrt(252)          → annualize it. Returns scale with time, but
                #                                volatility scales with sqrt(time), so
                #                                daily Sharpe × √252 = annual Sharpe.
                #    Sharpe guide: <0 losing money, 0.5–1.0 decent, 1.0–2.0 strong, >2.0 suspicious
                sharpe = (avg_return / std_dev) * math.sqrt(252) if std_dev > 0 else 0
            else:
                sharpe = -float('inf')  # No returns data → treat as worst possible

            if sharpe > best_sharpe:  # Keep the best-performing parameter set
                best_sharpe = sharpe
                best_params = combo

        if best_params is None:
            current_date += pd.DateOffset(months=roll_months)
            continue

        best_params_per_period.append({  # Record what the optimizer chose for this period
            'train_start': train_start.date().isoformat(),
            'train_end': train_end.date().isoformat(),
            'test_start': test_start.date().isoformat(),
            'test_end': test_end.date().isoformat(),
            'best_params': best_params,
            'train_sharpe': round(best_sharpe, 3),
        })

        # === Step 2: TEST on out-of-sample data (rules are LOCKED — no re-tuning) ===
        # pyright drops the `dict[str, float]` narrow on `fixed_params` by
        # this point in the function (the in-loop `**fixed_params` at the
        # combo-search site is fine), so suppress the false unpack error
        # locally rather than rebinding to a fresh local.
        test_params = {**fixed_params, **best_params}  # pyright: ignore[reportGeneralTypeIssues]  # Same params from training — this is the honest score
        _summary, _trades, daily_eq = run_cc_overlay(
            list(test_df['date'].dt.strftime('%Y-%m-%d')),
            np.asarray(test_df['price'].values, dtype=float),
            test_params,
        )

        oos_frames.append(daily_eq)  # Collect OOS equity curves to stitch together later

        # === Step 3: ROLL FORWARD ===
        current_date += pd.DateOffset(months=roll_months)  # Slide both windows forward
        # Next iteration trains on newer data and tests on the next unseen chunk

    # Concat per-window frames into one continuous OOS curve. If no
    # window produced output, return an empty frame with the same schema
    # so callers can index columns without a None-check.
    oos_equity: pd.DataFrame = (
        pd.concat(oos_frames, ignore_index=True)  # pyright: ignore[reportUnknownMemberType]
        if oos_frames
        else pd.DataFrame({'date': [], 'equity': pd.Series([], dtype=float), 'price': pd.Series([], dtype=float)})
    )
    return oos_equity, best_params_per_period


# ====================
# 7. Monte Carlo Shuffle
# ====================

def monte_carlo_shuffle(
    dates: list[str] | NDArray[Any],
    prices: NDArray[np.floating[Any]],
    params: dict[str, float],
    n_shuffles: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Monte Carlo randomization test: shuffle the daily returns, rebuild
    a synthetic price path from each shuffled sequence, re-run the overlay
    on it, and compare the real ordered path's return to that distribution.

    Why it works: real prices have a specific *order* — trends, mean
    reversion, volatility clusters. Shuffling destroys the order while
    keeping the exact same set of daily returns (same mean, vol, and
    distribution). If the overlay profits on both real and shuffled paths,
    it's capturing statistical *properties* of the returns (those survive
    shuffling). If it only works on the real path, it was exploiting the
    specific *sequence* — overfitting or luck. A high percentile (80+)
    means genuine skill; ~50 means random ordering does just as well.

    Returns a dict:
      ``real_return``  total_return_pct on the real (ordered) path
      ``mc_returns``   list of total_return_pct, one per completed shuffle
      ``mc_mean``      mean of ``mc_returns``
      ``mc_max``       best shuffled path's return
      ``percentile``   % of shuffled paths the real path beat (0-100)
      ``n_completed``  shuffles that didn't blow up (== n_shuffles here)
    """
    import random  # local: keeps the module's top-level import surface unchanged

    # Baseline: the real, ordered price path.
    real_summary, _, _ = run_cc_overlay(dates, prices, params)
    real_return = float(real_summary['total_return_pct'])

    # Daily simple returns from the real series.
    daily_returns = [
        float((prices[i] - prices[i - 1]) / prices[i - 1])
        for i in range(1, len(prices))
    ]

    rng = random.Random(seed)
    mc_returns: list[float] = []
    for _ in range(n_shuffles):
        # Shuffle preserves the distribution, changes only the sequence.
        shuffled = daily_returns.copy()
        rng.shuffle(shuffled)

        # Rebuild a price path: start at the real first price, then
        # chain-multiply each return. (1 + ret) is the daily multiplier
        # (+0.02 -> 1.02, -0.01 -> 0.99). Same moves, different order.
        synthetic = [float(prices[0])]
        for ret in shuffled:
            synthetic.append(synthetic[-1] * (1 + ret))

        # A few shuffled paths can blow up the backtest (a compounded
        # crash to <= 0 price, a flat stretch -> zero vol -> div-by-zero
        # in Black-Scholes). Skipping a handful out of hundreds doesn't
        # move the distribution; at seed=42 on the bundled data none do.
        try:
            mc_summary, _, _ = run_cc_overlay(
                dates, np.array(synthetic, dtype=np.float64), params
            )
            mc_returns.append(float(mc_summary['total_return_pct']))
        except Exception:
            continue

    # Percentile: how many shuffles did the real path beat? Count the
    # shuffles that did *worse* than real, as a fraction of completed
    # shuffles. 100 -> real beat every shuffle.
    worse = sum(1 for r in mc_returns if r < real_return)
    n = len(mc_returns)
    return {
        'real_return': real_return,
        'mc_returns': mc_returns,
        'mc_mean': sum(mc_returns) / n if n else 0.0,
        'mc_max': max(mc_returns) if mc_returns else 0.0,
        'percentile': int(100 * worse / n) if n else 0,
        'n_completed': n,
    }


def sensitivity_analysis(
    dates: list[str] | NDArray[Any],
    prices: NDArray[np.floating[Any]],
    params: dict[str, float],
    sweeps: tuple[tuple[str, tuple[float, ...]], ...] = (
        ('call_delta', (-0.10, -0.05, 0.0, 0.05, 0.10)),
        ('close_at_pct', (-0.20, -0.10, 0.0, 0.10, 0.20)),
    ),
) -> dict[str, dict[str, Any]]:
    """Perturb one parameter at a time from its base value and measure
    the impact on total return — the stability counterpart to a grid
    search.

    Where a grid search asks "which params are best?", sensitivity
    analysis asks "how fragile is the optimum we already chose?". Hold
    every param fixed except one, nudge that one by a small offset in
    both directions, and watch the total return. If a small tweak moves
    returns drastically, the chosen value is a knife-edge optimum
    (overfitting); if returns stay in a similar range, the strategy sits
    on a plateau and is robust to that parameter.

    Each ``sweeps`` entry is ``(param_name, offsets)``; the offsets are
    added to ``params[param_name]`` (so an offset of ``0.0`` reproduces
    the base config). The "robust" verdict is the worst drop from base
    as a percentage of the base return — single-digit-percent means the
    strategy isn't fragile to that parameter.

    A production-grade helper would also skip invalid perturbed values
    (negative ``call_delta``, non-positive ``dte``, ``close_at_pct`` ≤ 0
    or > 1) before running the backtest; the default sweeps stay inside
    those bounds, so this implementation doesn't bother.

    Returns ``{param_name: {...}}`` where each inner dict has:
      ``returns``         list of ``(offset, total_return_pct)``
      ``base_return``     total_return_pct at offset 0.0
      ``worst_return``    the lowest return across the sweep
      ``worst_drop_pct``  (base − worst) / base × 100; < 10 → "robust"
    """
    out: dict[str, dict[str, Any]] = {}
    for name, offsets in sweeps:
        base_val = params[name]
        returns: list[tuple[float, float]] = []
        for off in offsets:
            # Hold all params fixed except `name`, which shifts by `off`.
            summary, _, _ = run_cc_overlay(
                dates, prices, {**params, name: base_val + off}
            )
            returns.append((off, float(summary['total_return_pct'])))
        base_return = next(r for off, r in returns if off == 0.0)
        worst_return = min(r for _, r in returns)
        out[name] = {
            'returns': returns,
            'base_return': base_return,
            'worst_return': worst_return,
            'worst_drop_pct': (base_return - worst_return) / base_return * 100,
        }
    return out


# ====================
# 8. Main
# ====================

if __name__ == '__main__':
    # Load price data from CSV. yfinance writes a 3-row multi-index
    # header (Price/Close, Ticker/MSFT, Date/(empty)) before the
    # actual rows, so we skip those and name the two columns
    # explicitly. If yfinance ever changes that prefix, the pinned
    # MSFT regression tests will fail loudly.
    # pandas-stubs types `read_csv` as a complex overload set whose return
    # falls back to Unknown for `Series.tolist()` and `Series.to_numpy()`.
    # `cast` the two consumed columns back to their actual runtime types so
    # downstream typing stays sharp; the suppressions are scoped to just
    # those two calls.
    prices_df = pd.read_csv(  # pyright: ignore[reportUnknownMemberType]
        'msft_10yr_prices.csv',
        skiprows=3,
        header=None,
        names=['date', 'close'],
    )
    date_list = cast('list[str]', prices_df['date'].tolist())  # pyright: ignore[reportUnknownMemberType]
    prices_arr = cast(
        'NDArray[np.float64]',
        prices_df['close'].to_numpy(dtype=float),  # pyright: ignore[reportUnknownMemberType]
    )

    params: dict[str, float] = {
        'call_delta': 0.25,
        'close_at_pct': 0.75,
        'dte': 21,
        'risk_free_rate': 0.045,
        'capital': 100_000,  # $100K portfolio (sized into whole contracts)
    }

    summary, trades, daily_equity = run_cc_overlay(date_list, prices_arr, params)

    print(f"Capital:                         ${summary['capital']:>12,.2f}")
    print(f"Contracts (100 shares each):     {summary['num_contracts']:>12}    "
          f"(${summary['initial_stock_cost']:,.2f} stock + ${summary['cash']:,.2f} cash)")
    print()
    print("Returns")
    print(f"    Buy & Hold Final:            ${summary['buy_hold_final']:>12,.2f}    {summary['buy_hold_return_pct']:>+8.2f}%")
    print(f"  + Net Overlay P&L:             ${summary['net_overlay_pnl']:>12,.2f}    {summary['excess_return_pct']:>+8.2f} pp")
    print(f"  = CC Overlay Final:            ${summary['final_equity']:>12,.2f}    {summary['total_return_pct']:>+8.2f}%")
    print()
    print("Overlay P&L Breakdown")
    print(f"    Gross Premium Collected:     ${summary['total_premium_collected']:>12,.2f}    (income from {summary['num_calls_sold']} calls sold)")
    print(f"  - Buybacks + Assignment Costs: ${summary['overlay_costs']:>12,.2f}    (paid to close ITM calls + capped upside on assignment)")
    print(f"  = Net Overlay P&L:             ${summary['net_overlay_pnl']:>12,.2f}    ({summary['premium_retention_pct']:.1f}% retained)")
    print()
    print("Activity")
    print(f"    Calls Sold:                   {summary['num_calls_sold']:>12}")
    print(f"    Win Rate:                     {summary['win_rate']:>12.1f}%")
    print(f"    Max Drawdown:                 {summary['max_drawdown_pct']:>12.2f}%")
    print()

    # Statistical significance of the overlay's excess return over buy-and-hold.
    # Null hypothesis: the overlay adds zero value vs. simply holding the stock.
    stats = compute_statistics(
        daily_equity,
        num_contracts=summary['num_contracts'],
        cash=summary['cash'],
    )
    print("Statistical Significance (H0: overlay adds zero value vs. buy-and-hold)")
    print(f"    Days in Sample:              {stats['n_days']:>12}    ({stats['years_of_data']} years)")
    print(f"    Annualized Excess Return:    {stats['ann_excess_return_pct']:>+12.3f}%")
    print(f"    Annualized Excess Vol:       {stats['ann_excess_vol_pct']:>12.2f}%")
    print(f"    Sharpe of Excess Return:     {stats['sharpe_excess']:>+12.3f}")
    print(f"    t-stat (naive, IID):         {stats['t_stat_naive']:>+12.2f}    (assumes independence — inflated for overlays)")
    print(f"    t-stat (Newey-West, L={stats['nw_lag']:<2}):   {stats['t_stat_newey_west']:>+12.2f}    (correct: accounts for position autocorrelation)")
    print(f"    Clears t=2 bar?              {str(stats['passes_t_2']):>12}    (conventional significance)")
    print(f"    Clears t=3 bar (HLZ 2016)?   {str(stats['passes_t_3']):>12}    (multiple-testing adjusted)")
    print()

    # Risk-managed (delta-hedged) variant: Israelov & Nielsen (2015). Same params, but each
    # day we hold extra long-stock equal to the call's current delta × base shares, pinning
    # net portfolio delta at the buy-and-hold equivalent. This strips out the equity-timing
    # exposure that adds variance without contributing return — yielding a cleaner test of
    # whether the volatility risk premium itself is showing up on this underlying.
    hedge_params: dict[str, float] = {**params, 'delta_hedge': 1.0}
    hedge_summary, _, hedge_daily_equity = run_cc_overlay(date_list, prices_arr, hedge_params)
    hedge_stats = compute_statistics(
        hedge_daily_equity,
        num_contracts=hedge_summary['num_contracts'],
        cash=hedge_summary['cash'],
    )

    print("Risk-Managed (Delta-Hedged) vs. Naive Covered Call  —  Israelov & Nielsen (2015)")
    print(f"    {'Metric':<32}{'Naive':>15}{'Risk-Managed':>15}")
    print(f"    {'-' * 32}{'-' * 15}{'-' * 15}")
    print(f"    {'Excess Return / yr':<32}{stats['ann_excess_return_pct']:>+14.3f}%{hedge_stats['ann_excess_return_pct']:>+14.3f}%")
    print(f"    {'Excess Vol / yr':<32}{stats['ann_excess_vol_pct']:>14.2f}%{hedge_stats['ann_excess_vol_pct']:>14.2f}%")
    print(f"    {'Sharpe of Excess':<32}{stats['sharpe_excess']:>+15.3f}{hedge_stats['sharpe_excess']:>+15.3f}")
    print(f"    {'t-stat (Newey-West)':<32}{stats['t_stat_newey_west']:>+15.2f}{hedge_stats['t_stat_newey_west']:>+15.2f}")
    print(f"    {'Clears t=2 bar?':<32}{str(stats['passes_t_2']):>15}{str(hedge_stats['passes_t_2']):>15}")
    print(f"    {'Clears t=3 bar?':<32}{str(stats['passes_t_3']):>15}{str(hedge_stats['passes_t_3']):>15}")
