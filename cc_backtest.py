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
    """Standard normal CDF (Abramowitz & Stegun, 1964, Formula 26.2.17)."""
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - normal_pdf(x_abs) * (b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
    return y if sign == 1 else 1.0 - y

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
            - iv_multiplier: HV × this = IV estimate (e.g., 1.3)

    Returns:
        (summary, trades, daily_equity)
    """

    # Extract parameters from dict
    call_delta = params.get('call_delta', 0.25)
    close_at_pct = params.get('close_at_pct', 0.75)
    dte = params.get('dte', 21)
    r = params.get('risk_free_rate', 0.045)
    iv_mult = params.get('iv_multiplier', 1.3)

    num_days = len(dates)
    trades: list[dict[str, Any]] = []
    daily_equity: list[dict[str, Any]] = []

    # State tracking
    position: dict[str, Any] | None = None
    initial_price = float(prices[0])
    realized_pnl = 0.0  # cumulative premium overlay P&L (excludes stock appreciation)
    num_calls_sold = 0
    total_premium_collected = 0.0
    wins = 0
    losses = 0

    for day_idx in range(num_days):
        date = dates[day_idx]
        price = float(prices[day_idx])

        # Calculate rolling historical volatility over a 30-day window.
        # Need at least 2 prices to compute 1 return; skip day 0.
        if day_idx < 2:
            continue
        elif day_idx < 30:
            rolling_vol = float(np.std(np.diff(np.log(prices[:day_idx+1])), ddof=1)) * math.sqrt(252)
        else:
            rolling_vol = float(np.std(np.diff(np.log(prices[day_idx-29:day_idx+1])), ddof=1)) * math.sqrt(252)

        if math.isnan(rolling_vol) or rolling_vol <= 0:
            continue

        # IV estimate: HV × multiplier
        iv_estimate = rolling_vol * iv_mult

        # If no position, consider opening
        if position is None:
            # Sell a call
            T = dte / 252
            strike = find_strike_for_delta(price, T, r, iv_estimate, call_delta, option_type='call')
            premium = bs_price(price, strike, T, r, iv_estimate, option_type='call')

            # Apply transaction costs
            net_premium = premium * (1 - 0.03) - 0.0065  # 3% slippage, $0.65 commission

            # Open position
            position = {
                'strike': strike,
                'premium_collected': net_premium,
                'entry_price': price,
                'entry_idx': day_idx,
                'entry_date': date,
            }
            num_calls_sold += 1
            total_premium_collected += net_premium * 100

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
                # Expiration reached.
                # Overlay P&L only — stock appreciation is tracked separately.
                if price >= position['strike']:
                    # Called away: keep premium, but pay to rebuy shares above strike
                    pnl = (position['premium_collected'] - (price - position['strike'])) * 100
                else:
                    # Expired OTM: keep full premium
                    pnl = position['premium_collected'] * 100

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
                    pnl = (position['premium_collected'] - call_value_today) * 100 - 0.65
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

        # Track daily equity: stock value (100 shares) + cumulative overlay P&L.
        # This measures the total value of the covered call position: what the
        # shares are worth today plus all net premium income earned so far.
        # Return is measured against initial stock cost, not the capital param.
        stock_value = price * 100
        equity = stock_value + realized_pnl
        if position is not None:
            days_left = dte - (day_idx - position['entry_idx'])
            T_remaining = max(days_left / 252, 0)
            call_value = bs_price(price, position['strike'], T_remaining, r, iv_estimate, option_type='call')
            equity += (position['premium_collected'] - call_value) * 100
        daily_equity.append({'date': date, 'equity': round(equity, 2), 'price': price})

    # Compute summary stats
    initial_cost = initial_price * 100  # cost basis of 100 shares
    final_equity = daily_equity[-1]['equity'] if daily_equity else initial_cost
    total_return = (final_equity - initial_cost) / initial_cost * 100

    # Max drawdown
    peak = initial_cost
    max_dd = 0.0
    for d in daily_equity:
        if d['equity'] > peak:
            peak = d['equity']
        dd = (peak - d['equity']) / peak * 100
        if dd > max_dd:
            max_dd = dd

    summary: dict[str, Any] = {
        'initial_cost': round(initial_cost, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round(total_return, 2),
        'total_premium_collected': round(total_premium_collected, 2),
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
        'iv_multiplier': 1.3,
    }

    summary, trades, daily_equity = run_cc_overlay(date_list, prices_arr, params)

    print(f"Initial Cost (100 shares): ${summary['initial_cost']:,.2f}")
    print(f"Final Equity: ${summary['final_equity']:,.2f}")
    print(f"Total Return: {summary['total_return_pct']:.2f}%")
    print(f"Total Premium Collected: ${summary['total_premium_collected']:,.2f}")
    print(f"Calls Sold: {summary['num_calls_sold']}")
    print(f"Win Rate: {summary['win_rate']:.1f}%")
    print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
