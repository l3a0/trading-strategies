"""Download split-adjusted daily OHLC for the Tharp random-entry replication.

The trailing-stop system (docs/tharp_random_entry_plan.md) needs true range,
which needs highs and lows the close-only price files lack. yfinance's
auto_adjust=False bars are split-adjusted price-return series (splits backed
out, dividends left in the price path) — exactly the continuous series a
price-only trend system wants. One CSV per ticker:
``{ticker}_daily_ohlc.csv`` with columns date,open,high,low,close.

Usage:
    python -m pipeline.download_ohlc            # the plan's nine tickers
    python -m pipeline.download_ohlc SPY GLD    # a subset
"""
from __future__ import annotations

import sys

import yfinance as yf

from common.paths import data_path

TICKERS = ('SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'XLE', 'EEM', 'MSFT', 'NVDA')
START, END = '1999-11-01', '2026-07-01'   # ~2 months of ATR warmup before 2000-01


def fetch_one(ticker: str) -> str:
    df = yf.download(ticker, start=START, end=END, auto_adjust=False,
                     progress=False)
    if df.empty:
        raise RuntimeError(f'{ticker}: yfinance returned no data')
    if hasattr(df.columns, 'levels'):          # flatten MultiIndex columns
        df.columns = [c[0] for c in df.columns]
    out = data_path(f'{ticker.lower()}_daily_ohlc.csv')
    with open(out, 'w') as f:
        f.write('date,open,high,low,close\n')
        for idx, row in df.iterrows():
            f.write(f"{idx.date().isoformat()},{row['Open']:.6f},"
                    f"{row['High']:.6f},{row['Low']:.6f},{row['Close']:.6f}\n")
    print(f'{ticker}: {len(df)} rows -> {out}', flush=True)
    return out


def main() -> None:
    tickers = sys.argv[1:] or list(TICKERS)
    for t in tickers:
        fetch_one(t)


if __name__ == '__main__':
    main()
