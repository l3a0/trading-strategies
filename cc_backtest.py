from __future__ import annotations

import csv
import math
from typing import Any

import numpy as np
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
        start = int(S * 0.80)
        end = int(S * 1.02)
    else:  # call
        start = int(S * 0.98)
        end = int(S * 1.25)

    for k in range(start, end + 1):
        K = float(k)
        delta = bs_delta(S, K, T, r, sigma, option_type=option_type)
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
    Calculate rolling historical volatility.

    Args:
        prices: array of daily closing prices
        window: lookback (default 30 days)

    Returns:
        vols: array of annualized volatilities
    """
    log_returns = np.diff(np.log(prices))

    vols: list[float] = []
    for i in range(len(log_returns)):
        if i < window - 1:
            vols.append(float('nan'))
        else:
            window_returns = log_returns[i-window+1:i+1]
            std_dev = float(np.std(window_returns, ddof=1))
            annualized = std_dev * math.sqrt(252)
            vols.append(annualized)

    return np.array(vols)

def detect_regime(rolling_vol: float) -> str:
    """Classify volatility regime based on current HV level."""
    if rolling_vol > 0.25:
        return 'high'
    elif rolling_vol < 0.15:
        return 'low'
    else:
        return 'normal'

def estimate_iv(rolling_vol: float, regime: str = 'normal') -> float:
    """
    Adjust HV to IV estimate based on regime.

    High vol (>25%) → 1.1× (IV already elevated, won't expand much)
    Normal (15-25%) → 1.3× (typical HV→IV relationship)
    Low vol (<15%)  → 1.5× (IV is suppressed, expect mean reversion)
    """
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
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
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

        IV estimation uses the regime-based detect_regime() + estimate_iv()
        functions (multiplier varies: 1.1× in high vol, 1.3× normal, 1.5× low).

    Returns:
        (summary, trades, daily_equity)
    """

    # Extract parameters from dict
    call_delta = params.get('call_delta', 0.25)
    close_at_pct = params.get('close_at_pct', 0.75)
    dte = params.get('dte', 21)
    r = params.get('risk_free_rate', 0.045)

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
    shares = 100 * num_contracts                   # total shares held
    initial_stock_cost = shares * initial_price    # actual capital deployed in stock
    cash = capital - initial_stock_cost            # leftover, 0% yield

    num_days = len(dates)
    trades: list[dict[str, Any]] = []
    daily_equity: list[dict[str, Any]] = []

    # State tracking
    position: dict[str, Any] | None = None
    realized_pnl = 0.0  # cumulative premium overlay P&L (excludes stock appreciation)
    num_calls_sold = 0
    total_premium_collected = 0.0
    wins = 0
    losses = 0

    for day_idx in range(num_days):
        date = dates[day_idx]
        price = float(prices[day_idx])

        # Calculate rolling historical volatility over a 30-day window.
        if day_idx < 3:
            # Warmup: too few returns for a meaningful std (NaN or 0).
            # Fall back to 20% annualized vol (a long-run equity baseline).
            rolling_vol = 0.20
        elif day_idx < 30:
            # Early days: use all available history with Bessel's correction.
            rolling_vol = float(np.std(np.diff(np.log(prices[:day_idx+1])), ddof=1)) * math.sqrt(252)
        else:
            # Steady state: trailing 30-price window ([day_idx-29, day_idx]).
            rolling_vol = float(np.std(np.diff(np.log(prices[day_idx-29:day_idx+1])), ddof=1)) * math.sqrt(252)

        if math.isnan(rolling_vol) or rolling_vol <= 0:
            continue

        # IV estimate: regime-based multiplier (1.1× high, 1.3× normal, 1.5× low)
        regime = detect_regime(rolling_vol)
        iv_estimate = estimate_iv(rolling_vol, regime)

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

        # Track daily equity: stock value + idle cash + cumulative overlay P&L.
        # This is the total portfolio value today (mark-to-market on shares,
        # plus the leftover cash, plus all net premium income realized so far).
        # Returns are measured against `capital` (the total committed dollars).
        stock_value = price * shares
        equity = stock_value + cash + realized_pnl
        if position is not None:
            days_left = dte - (day_idx - position['entry_idx'])
            T_remaining = max(days_left / 252, 0)
            call_value = bs_price(price, position['strike'], T_remaining, r, iv_estimate, option_type='call')
            equity += (position['premium_collected'] - call_value) * shares
        daily_equity.append({'date': date, 'equity': round(equity, 2), 'price': price})

    # Compute summary stats
    final_equity = daily_equity[-1]['equity'] if daily_equity else capital
    total_return = (final_equity - capital) / capital * 100

    # Buy-and-hold benchmark: hold the same `shares` for the whole period
    # without selling calls. Idle cash sits at 0% in both scenarios so it
    # cancels in the excess-return comparison.
    final_price = float(prices[-1])
    buy_hold_final = final_price * shares + cash
    buy_hold_return = (buy_hold_final - capital) / capital * 100
    excess_return = total_return - buy_hold_return

    # Decompose the overlay's contribution: we collected `total_premium_collected`
    # in gross premium across all sells, but had to pay it back via buybacks
    # (early closes at profit target / ITM) and assignment losses (when called
    # away above strike). The net overlay P&L equals the gap between final
    # equity and the buy-and-hold final value.
    net_overlay_pnl = final_equity - buy_hold_final
    overlay_costs = total_premium_collected - net_overlay_pnl
    premium_retention = (net_overlay_pnl / total_premium_collected * 100
                        if total_premium_collected > 0 else 0.0)

    # Max drawdown
    peak = capital
    max_dd = 0.0
    for d in daily_equity:
        if d['equity'] > peak:
            peak = d['equity']
        dd = (peak - d['equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    summary: dict[str, Any] = {
        'capital': round(capital, 2),
        'num_contracts': num_contracts,
        'initial_stock_cost': round(initial_stock_cost, 2),
        'cash': round(cash, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'buy_hold_final': round(buy_hold_final, 2),
        'buy_hold_return_pct': round(buy_hold_return, 2),
        'excess_return_pct': round(excess_return, 2),
        'net_overlay_pnl': round(net_overlay_pnl, 2),
        'total_premium_collected': round(total_premium_collected, 2),
        'overlay_costs': round(overlay_costs, 2),
        'premium_retention_pct': round(premium_retention, 1),
        'num_calls_sold': num_calls_sold,
        'wins': wins,
        'losses': losses,
        'win_rate': round(wins / max(wins + losses, 1) * 100, 1),
        'max_drawdown_pct': round(max_dd, 2),
    }

    return summary, trades, daily_equity

# ====================
# 4. Main
# ====================

if __name__ == '__main__':
    # Load price data from CSV (date,close format)
    # Skips header rows that don't start with a date (e.g. yfinance multi-index headers)
    date_list: list[str] = []
    price_list: list[float] = []
    with open('msft_10yr_prices.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not row[0][:4].isdigit():
                continue  # skip header/metadata lines
            date_list.append(row[0])
            price_list.append(float(row[1]))

    prices_arr = np.array(price_list)

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
