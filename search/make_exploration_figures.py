"""Figures for docs/explorations.md — regenerated from the PINNED machinery.

The exploration log's charts follow the same discipline as the tutorial's
(`engine/make_figures.py`): every figure is re-derived from the exact code
paths the regression tests pin, so a re-pin regenerates the figure and the
two can never silently disagree. Nothing here is a screenshot of a
one-off run.

Three figures (house conventions: 16x9 @ 100 dpi, tab10 palette):

- ``ex1_spy_truehold_vs_spy.png`` — the flagship true-hold delta-hedged
  call-selling book indexed against SPY buy-and-hold, over its per-cycle
  R scatter (`TestSpyCcRExperiment` / the deep-ITM addendum pins).
- ``ex3_gld_coda.png`` — GLD's book overlaid on buy-and-hold, over its
  per-cycle R scatter (`TestNvdaGldTrueHoldExploration` pins).
- ``ex4_nvda_coda.png`` — NVDA's book equity crossing zero (same pins;
  the continuation past $0 is the engine's zero-interest,
  never-margin-called financing — the real account dies at the crossing,
  annotated).

NOT run in CI (it needs the full chain stores and produces committed
PNGs, like the blog-only figures). Regenerate manually after any re-pin
of the source tests:

    python -m search.make_exploration_figures
"""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from common.paths import data_path
from common.trade_ledger import TradeRecord, build_trade_ledger
from realchains.cc_r_experiment import (
    attribute_cycles,
    open_tail_entry,
    overlay_excess,
    run_cell,
)
from realchains.real_cc_backtest import (
    CHAIN_CLEAN_START,
    load_chain_store,
    load_unadjusted_prices,
    run_real_cc_overlay,
)

OUT = 'docs/figures'
FIGSIZE = (16, 9)
SAVE_DPI = 100

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
RED = '#d62728'
GRAY = '#888888'


def _yf(date: str) -> float:
    y, m, d = date.split('-')
    return int(y) + (int(m) - 1) / 12 + (int(d) - 1) / 365


def _load_market(ticker: str, canonical: str, extras: tuple[str, ...] = ()
                 ) -> tuple[dict[str, Any], list[str], list[float]]:
    store = load_chain_store(data_path(canonical),
                             [data_path(e) for e in extras],
                             start=CHAIN_CLEAN_START.get(ticker))
    days = sorted(store)
    dates, prices = load_unadjusted_prices(ticker, days[0], '2026-06-06')
    pairs = [(d, p) for d, p in zip(dates, prices) if days[0] <= d <= days[-1]]
    return store, [d for d, _ in pairs], [p for _, p in pairs]


def _true_hold_book(ticker: str, dates: list[str], prices: list[float],
                    store: dict[str, Any]) -> dict[str, Any]:
    """The coda configuration on any ticker: hedged, hold-to-expiry, no
    deep-ITM management — with hedge-inclusive cycle attribution."""
    summary, trades, eq = run_real_cc_overlay(
        dates, prices, store,
        {'call_delta': 0.25, 'dte': 30, 'close_at_pct': 1.0,
         'capital': 100_000, 'delta_hedge': True, 'manage_deep_itm': False})
    shares = summary['num_contracts'] * 100
    native = build_trade_ledger(trades, strategy='covered_call', ticker=ticker,
                                shares=shares, risk_basis='premium_collected')
    excess = overlay_excess(eq, shares, summary['cash'])
    records, _, _, _ = attribute_cycles(native, list(eq['date']), excess,
                                        open_entry_date=open_tail_entry(trades))
    return {'summary': summary, 'eq': eq, 'records': records,
            'shares': shares, 'cash': summary['cash']}


def _overlay_panel(ax: Any, eq: Any, shares: int, cash: float, capital: float,
                   ticker: str, book_label: str) -> None:
    """Indexed-to-100 book vs buy-and-hold on one axis (the honest overlay:
    one unit, no dual axes)."""
    x = [_yf(d) for d in eq['date']]
    book = [e / capital * 100 for e in eq['equity']]
    bh = [(p * shares + cash) / capital * 100 for p in eq['price']]
    ax.plot(x, bh, color=GRAY, ls='--', lw=1.4,
            label=f'{ticker} buy-and-hold (final {bh[-1]:.0f})')
    ax.plot(x, book, color=GREEN, lw=2.0,
            label=f'{book_label} (final {book[-1]:.0f})')
    ax.set_ylabel('indexed (start = 100)')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25)


def _scatter_panel(ax: Any, records: list[TradeRecord]) -> None:
    wins = [(r.entry_date, r.r_multiple) for r in records if r.r_multiple > 0]
    loss = [(r.entry_date, r.r_multiple) for r in records if r.r_multiple <= 0]
    ax.scatter([_yf(d) for d, _ in wins], [r for _, r in wins], s=18,
               color=BLUE, label=f'win ({len(wins)})')
    ax.scatter([_yf(d) for d, _ in loss], [r for _, r in loss], s=26,
               color=RED, marker='D', label=f'loss ({len(loss)})')
    ax.axhline(0, color=GRAY, lw=1)
    ax.axhline(1, color=BLUE, lw=1, ls=':',
               label='+1R = full premium (hedge can exceed)')
    ax.set_ylabel('per-cycle R')
    ax.legend(loc='upper right', frameon=False, ncol=3)
    ax.grid(alpha=0.25)


def fig_ex1(spy_cells: dict[str, dict[str, Any]]) -> Figure:
    flag = spy_cells['truehold']
    fig = Figure(figsize=FIGSIZE)
    ax1, ax2 = fig.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [3, 2]})
    s = flag['summary']
    _overlay_panel(ax1, flag['eq'], s['num_contracts'] * 100, s['cash'],
                   s['capital'], 'SPY', 'true-hold hedged call-selling book')
    ax1.set_title('The flagship cell overlaid on SPY — sixteen years, one sliver apart')
    _scatter_panel(ax2, flag['records'])
    ax2.set_xlabel('year')
    return fig


def fig_ex3_gld(gld: dict[str, Any]) -> Figure:
    fig = Figure(figsize=FIGSIZE)
    ax1, ax2 = fig.subplots(2, 1, sharex=True,
                            gridspec_kw={'height_ratios': [3, 2]})
    s = gld['summary']
    _overlay_panel(ax1, gld['eq'], s['num_contracts'] * 100, s['cash'],
                   s['capital'], 'GLD', 'true-hold hedged book')
    ax1.set_title(f"GLD: the program's best junior score — "
                  f"net ${s['net_overlay_pnl'] / 1000:+.1f}K, "
                  'earned mostly in the flat 2013-2019 decade')
    _scatter_panel(ax2, gld['records'])
    ax2.set_xlabel('year')
    return fig


def fig_ex4_nvda(nvda: dict[str, Any]) -> Figure:
    fig = Figure(figsize=FIGSIZE)
    ax = fig.subplots()
    eq = nvda['eq']
    x = [_yf(d) for d in eq['date']]
    equity_m = [e / 1e6 for e in eq['equity']]
    ax.plot(x, equity_m, color=RED, lw=2.0, label='NVDA book equity')
    ax.axhline(0, color=GRAY, lw=1.2)
    dead_i = next((i for i, e in enumerate(eq['equity']) if e <= 0), None)
    if dead_i is not None:
        dead_x = x[dead_i]
        ax.axvline(dead_x, color=RED, lw=1, ls=':')
        ax.annotate(f"account reaches $0 ({eq['date'][dead_i]})\n"
                    'the real book dies here; the continuation is the\n'
                    "engine's zero-interest, never-margin-called financing",
                    xy=(dead_x, 0), xytext=(dead_x - 6.5, -2.2),
                    fontsize=11, arrowprops={'arrowstyle': '->', 'color': GRAY})
    ax.set_ylabel('book equity ($M)')
    ax.set_xlabel('year')
    ax.set_title(f"NVDA: the fattest wing premium — sixteen years of apparent "
                 f"riches, dead in one cliff (net ${nvda['summary']['net_overlay_pnl'] / 1e6:+.2f}M)")
    ax.legend(loc='lower left', frameon=False)
    ax.grid(alpha=0.25)
    return fig


def main() -> None:
    print('SPY flagship ...', flush=True)
    store, dates, prices = _load_market('SPY', 'spy_option_dailies.csv')
    spy_cells = {'truehold': run_cell(dates, prices, store, hedged=True,
                                      close_at_pct=1.0, stop_loss_mult=None,
                                      manage_deep_itm=False)}
    del store
    print('GLD ...', flush=True)
    g_store, g_dates, g_prices = _load_market('GLD', 'gld_option_dailies.csv')
    gld = _true_hold_book('GLD', g_dates, g_prices, g_store)
    del g_store
    print('NVDA ...', flush=True)
    n_store, n_dates, n_prices = _load_market('NVDA', 'nvda_option_dailies.csv')
    nvda = _true_hold_book('NVDA', n_dates, n_prices, n_store)
    del n_store

    fig_ex1(spy_cells).savefig(f'{OUT}/ex1_spy_truehold_vs_spy.png',
                               dpi=SAVE_DPI, bbox_inches='tight')
    print(f'wrote {OUT}/ex1_spy_truehold_vs_spy.png')
    fig_ex3_gld(gld).savefig(f'{OUT}/ex3_gld_coda.png',
                             dpi=SAVE_DPI, bbox_inches='tight')
    print(f'wrote {OUT}/ex3_gld_coda.png')
    fig_ex4_nvda(nvda).savefig(f'{OUT}/ex4_nvda_coda.png',
                               dpi=SAVE_DPI, bbox_inches='tight')
    print(f'wrote {OUT}/ex4_nvda_coda.png')


if __name__ == '__main__':
    main()
