"""factor_panel.py — a real equity-universe panel for the factor backend (the FIRST real factor data).

The whole factor stack (F1–F4, H1, H2) was built and pinned on a SYNTHETIC panel. This assembles a REAL
one: a committed, pre-specified universe of ~40 large-cap US equities, their split/dividend-adjusted daily
closes downloaded from Yahoo Finance into a dates×tickers panel, so `GrammarFactorBackend` / the search /
the proposer run on actual cross-sectional returns for the first time.

EXPLORATORY, and the standing limits hold: promotion stays CLOSED and any survivor stays EXPLORATORY until
the Phase-C time-axis holdout exists. A REAL survivor here is a candidate to pre-register, never a verdict.

TWO HONEST CAVEATS, stated loudly because they bound what a real result here can claim:
  1. SURVIVORSHIP. The universe is CURRENT large-caps — today's survivors, no delisted/failed names. That
     biases momentum/low-vol signals UPWARD (the panel is the winners). Fine for "does the machinery find
     real cross-sectional signal", not for a tradeable claim.
  2. FROZEN-AT-DOWNLOAD. `period='10y'` is relative to the download date, so the committed CSV is a frozen
     snapshot; regenerating gives a NEWER panel (a different end date) and would re-pin. The committed CSV
     is the reproducible artifact — the test pins results against IT, not against a live re-download.

The prices are FREE + regenerable (yfinance), so the panel CSV lives in git like the other price CSVs (it
is NOT premium option-chain data). The recorded factor ledger stays regenerable (`.gitignore`d); the
exploration's pin is the dataset-gated test + the docs write-up, not the ledger.
"""
from __future__ import annotations
from common.paths import data_path

import argparse
import hashlib
import os

import pandas as pd

# The committed universe — HAND-SELECTED ~40 large-caps with 10y+ continuous history (no post-2014
# IPOs/spinoffs), diversified across sectors. Committed/frozen for reproducibility, but NOT independently
# pre-registered: these are CURRENT large-caps the author picked, so the universe itself is survivor-biased
# (the docs/factor_real_panel.md survivorship caveat covers it). "Committed" here means changed only
# deliberately, NOT that the list predates seeing any data.
FACTOR_UNIVERSE: tuple[str, ...] = (
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'INTC', 'CSCO', 'ORCL', 'IBM', 'QCOM',   # tech
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'C', 'USB',                               # financials
    'PG', 'KO', 'PEP', 'WMT', 'MCD', 'NKE', 'HD', 'LOW', 'COST', 'SBUX',              # consumer
    'JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'AMGN',                                        # health
    'XOM', 'CVX', 'CAT', 'BA', 'HON', 'MMM', 'UNP', 'DE',                             # industrial/energy
)
FACTOR_PANEL_NAME = 'US_LARGE_CAP'                 # the universe id — fills the honest-core `ticker` slot
FACTOR_PANEL_PATH = data_path('factor_universe_prices.csv')   # committed (free, regenerable — like the other price CSVs)


def build_factor_panel(tickers: tuple[str, ...] = FACTOR_UNIVERSE, *, period: str = '10y',
                       path: str = FACTOR_PANEL_PATH) -> pd.DataFrame:
    """Download the universe's split/dividend-adjusted daily closes (one `yf.download` call) into a
    dates×tickers panel and save it to `path`. Drops all-NaN rows; per-ticker gaps are left as NaN (the
    backend's IC ranks cross-sectionally per date, dropping NaNs). REGENERATES a frozen snapshot — the
    committed CSV is the reproducible artifact (see the module caveat)."""
    import yfinance as yf
    data = yf.download(list(tickers), period=period, progress=False, auto_adjust=True)
    closes = data['Close'].dropna(how='all').sort_index()
    closes.to_csv(path)
    return closes


def load_factor_panel(path: str = FACTOR_PANEL_PATH) -> pd.DataFrame:
    """Load the committed universe panel CSV → a dates×tickers closes DataFrame."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


def panel_available(path: str = FACTOR_PANEL_PATH) -> bool:
    """True iff the committed panel CSV is present (the dataset-gate predicate, like the option stores)."""
    return os.path.exists(path)


def panel_checksum(panel: pd.DataFrame) -> str:
    """A content hash of the panel (the backend's lineage input), so a refreshed panel re-lineages."""
    return hashlib.sha256(pd.util.hash_pandas_object(panel, index=True).values.tobytes()).hexdigest()[:16]


def make_factor_backend(panel: pd.DataFrame | None = None, path: str = FACTOR_PANEL_PATH):
    """A `GrammarFactorBackend` bound to the real universe panel (loaded from `path` if not supplied),
    with the panel's content hash as the lineage checksum."""
    from factor.factor_engine import GrammarFactorBackend
    panel = load_factor_panel(path) if panel is None else panel
    end = str(panel.index[-1].date())   # the panel's ACTUAL last date — the honest as-of, not the
    #                                     synthetic-default FACTOR_END (which the real panel post-dates)
    return GrammarFactorBackend(FACTOR_PANEL_NAME, panel, checksum=panel_checksum(panel), end=end)


def _summarize(panel: pd.DataFrame) -> str:
    span = f'{panel.index[0].date()} → {panel.index[-1].date()}'
    return (f'{FACTOR_PANEL_NAME}: {panel.shape[1]} tickers × {panel.shape[0]} days ({span}); '
            f'checksum {panel_checksum(panel)}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Build/run the real factor universe panel.')
    parser.add_argument('--build', action='store_true', help='(re)download the universe panel from yfinance')
    parser.add_argument('--record', action='store_true', help='record the menu-walker run to the factor ledger')
    parser.add_argument('--period', default='10y')
    args = parser.parse_args()

    if args.build or not panel_available():
        print('Downloading the universe panel…')
        panel = build_factor_panel(period=args.period)
        print(f'Saved {FACTOR_PANEL_PATH}')
    else:
        panel = load_factor_panel()
    print(_summarize(panel))

    from factor.factor_search import run_factor_search
    backend = make_factor_backend(panel)
    result = run_factor_search(backend, record=args.record)
    print('Factor search (the bounded grammar slice on real equities):')
    for k in ('scored', 'coherent', 'incoherent', 'data_invalid', 'survivors'):
        print(f'  {k:13s} {result[k]}')
    print('  (EXPLORATORY — promotion CLOSED, survivors stay candidates for pre-registration until Phase-C)')


if __name__ == '__main__':
    main()
