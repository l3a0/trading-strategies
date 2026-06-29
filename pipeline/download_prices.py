"""Download historical stock prices from Yahoo Finance."""

from __future__ import annotations
from common.paths import data_path

import argparse

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]


def download_prices(  # pyright: ignore[reportUnknownParameterType]
    ticker: str = 'MSFT', period: str = '10y', output: str | None = None
) -> None:
    """
    Download daily closing prices and save to CSV.

    Args:
        ticker: stock symbol (default: MSFT)
        period: lookback period (default: 10y)
        output: output filename (default: {ticker}_10yr_prices.csv)
    """
    if output is None:
        output = data_path(f'{ticker.lower()}_10yr_prices.csv')

    data = pd.DataFrame(yf.download(ticker, period=period))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    closes = data[['Close']]  # keep as DataFrame so to_csv preserves the column

    closes.to_csv(output, header=True)  # pyright: ignore[reportUnknownMemberType]
    print(f"Saved {len(closes)} days of {ticker} closing prices to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download historical stock prices')
    parser.add_argument('--ticker', default='MSFT', help='Stock symbol (default: MSFT)')
    parser.add_argument('--period', default='10y', help='Lookback period (default: 10y)')
    parser.add_argument('--output', default=None, help='Output CSV filename')
    args = parser.parse_args()

    download_prices(args.ticker, args.period, args.output)
