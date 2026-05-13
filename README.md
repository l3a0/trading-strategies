# Covered Call Backtester

[![CI](https://github.com/l3a0/covered-call-backtesting/actions/workflows/ci.yml/badge.svg)](https://github.com/l3a0/covered-call-backtesting/actions/workflows/ci.yml)

A from-scratch Python backtester for the covered call overlay strategy. Prices options with Black-Scholes (using `math.erf` for high-precision CDF), estimates IV from rolling historical volatility with regime-based multipliers, and simulates day-by-day trade decisions over multi-year price histories.

## Quick start

```bash
# 1. Set up the environment (one time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Download fresh price data — there's already an MSFT CSV in the repo
python download_prices.py                   # default: MSFT, 10y
python download_prices.py --ticker AAPL     # any ticker
python download_prices.py --ticker SPY --period 5y

# 3. Run the backtest
python cc_backtest.py
```

Sample output (MSFT 2016-04 → 2026-04, $100K portfolio):

```text
Capital:                         $  100,000.00
Contracts (100 shares each):               20    ($95,573.55 stock + $4,426.45 cash)

Returns
    Buy & Hold Final:            $  746,166.44     +646.17%
  + Net Overlay P&L:             $  298,947.87     +298.95 pp
  = CC Overlay Final:            $1,045,114.31     +945.11%

Overlay P&L Breakdown
    Gross Premium Collected:     $1,025,092.00    (income from 185 calls sold)
  - Buybacks + Assignment Costs: $  726,144.12    (paid to close ITM calls + capped upside on assignment)
  = Net Overlay P&L:             $  298,947.87    (29.2% retained)

Activity
    Calls Sold:                            185
    Win Rate:                             81.0%
    Max Drawdown:                        23.02%

Statistical Significance (H0: overlay adds zero value vs. buy-and-hold)
    Days in Sample:                      2514    (9.98 years)
    Annualized Excess Return:          +1.591%
    Annualized Excess Vol:               9.79%
    Sharpe of Excess Return:           +0.163
    t-stat (naive, IID):                +0.51    (assumes independence — inflated for overlays)
    t-stat (Newey-West, L=8 ):          +0.58    (correct: accounts for position autocorrelation)
    Clears t=2 bar?                     False    (conventional significance)
    Clears t=3 bar (HLZ 2016)?          False    (multiple-testing adjusted)
```

The portfolio is sized into whole 100-share contracts at the initial price; any leftover (here, $4,426 of $100K with MSFT at ~$48) sits as 0%-yield cash. Returns are measured against `capital`, so the cash drag is included. To run a single-contract simulation, omit `capital` from `params`.

The bottom block tests whether the overlay's excess return over buy-and-hold is statistically distinguishable from zero, using Newey-West HAC standard errors that correct for the autocorrelation introduced by holding the same option position across multiple days. On this MSFT sample the t-stat is 0.58 — well below the conventional significance bar of 2 — meaning the $299K of headline overlay P&L isn't reliably distinguishable from noise. See the [tutorial's Part 5](tutorial_covered_call_backtest.md#part-5-robustness-checks--proving-its-not-luck) for the full reasoning.

For an explanation of each output line — including what "assignment loss" means and why buybacks can dominate the overlay's gross premium income — see the [tutorial](tutorial_covered_call_backtest.md) (its Glossary defines the terms; Part 3 walks through the trade-by-trade math).

## Tests

```bash
pytest test_cc_backtest.py          # run the full test suite
pytest test_cc_backtest.py -v       # verbose
pytest --cov=. --cov-branch         # with coverage
```

CI runs `ruff`, `pyright`, the test suite, and a backtest smoke test on every PR — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Project layout

| File | What it is |
| --- | --- |
| [cc_backtest.py](cc_backtest.py) | Backtest engine: Black-Scholes pricing, rolling vol, regime-based IV, day-by-day overlay state machine, Newey-West t-stat reporting on excess returns |
| [test_cc_backtest.py](test_cc_backtest.py) | Unit and scenario tests covering pricing, the overlay state machine, and the statistics helper |
| [download_prices.py](download_prices.py) | yfinance data downloader |
| [make_figures.py](make_figures.py) | Regenerates the four educational figures embedded in the tutorial into `docs/figures/` |
| [msft_10yr_prices.csv](msft_10yr_prices.csv) | Sample MSFT price data, 2016-04 to 2026-04 |
| [tutorial_covered_call_backtest.md](tutorial_covered_call_backtest.md) | Long-form tutorial — theory, math, code walkthrough, and statistical-significance testing |
| [docs/figures/](docs/figures/) | Generated PNGs embedded in the tutorial; regenerable from `make_figures.py` |
| [requirements.txt](requirements.txt) | Runtime + dev dependencies |

## Where to look for more details

- **How any single piece works (Black-Scholes math, rolling vol, the overlay state machine, walk-forward optimization, robustness checks):** the [tutorial](tutorial_covered_call_backtest.md) is the source of truth. It explains the *why* behind every part of the engine.
- **Exact behavior of a function:** read [cc_backtest.py](cc_backtest.py) — it's heavily commented and small enough to read end-to-end.
- **What the engine guarantees:** [test_cc_backtest.py](test_cc_backtest.py) has scenario tests for the major trade flows (sell + expire OTM, called away, profit-target close, multi-cycle accumulation).

## Strategy parameters

Edit the `params` dict at the bottom of [cc_backtest.py](cc_backtest.py):

| Param | Default | Meaning |
| --- | --- | --- |
| `call_delta` | 0.25 | Target delta for strike selection (≈25% chance ITM at expiry) |
| `close_at_pct` | 0.75 | Close when 75% of premium has been captured |
| `dte` | 21 | Days to expiration when opening a new call |
| `risk_free_rate` | 0.045 | Annual risk-free rate used in Black-Scholes |
| `capital` | cost of 1 contract | Total dollars committed; sized into whole 100-share contracts (leftover sits as 0%-yield cash) |

IV is no longer a tunable param — it's derived from rolling 30-day historical vol times a regime-based multiplier (1.1× / 1.3× / 1.5× for high / normal / low vol).

## Caveats

This is an educational backtester, not a production trading system. Notable limitations:

- IV is estimated, not real (no historical option chain data)
- No earnings-week avoidance, no dividend handling, no rolling logic
- Single-stock, single-period results — see the tutorial's robustness section for how to evaluate generalizability
