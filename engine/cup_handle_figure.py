"""The §4 eyeball-pass figure for the cup-and-handle scan.

The plan (`docs/cup_handle_scan_plan.md` §4) requires, before any return is
computed, a human to LOOK at the detector's output: a grid of the twenty
highest-volume-surge detections, each drawn as price with the cup, handle,
and breakout annotated, and — the load-bearing hygiene guarantee —
**clipped at the breakout day**. The right edge of every panel is the
breakout session ``t``; no post-breakout bar is rendered, so no return is
visible before §10 step 4. A detector nobody has looked at is not a
detector, and after Amendment 1 withdrew the detection-rate prior this
eyeball pass is the primary detector-validation check.

Location note (2026-07-23, not a methodology change): the plan named
`search/make_exploration_figures.py` as the home. That module lives in the
chain-store world (realchains / wheel imports) and its ``main`` loads
option stores; the cup-handle eyeball lives in the minute-archive world
(`pipeline.minute_archive` + this study's detector). Keeping it here, beside
the detector it validates, avoids coupling two disjoint data domains in one
module. It follows the same discipline as the other figure generators:
re-derived from the PINNED detector, regenerable, a committed PNG, NOT run
in CI (it needs the gitignored minute archive). Regenerate with:

    python -m engine.cup_handle_figure
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from common.paths import data_path  # noqa: F401 — parity w/ sibling generators
from engine.cup_handle_scan import (
    VOL_AVG_WINDOW,
    detect_cup_handle,
)
from pipeline.minute_archive import (
    archived_tickers,  # noqa: F401 — available for a wider eyeball if wanted
    load_clean_daily,
    load_splits,
    universe,
)

OUT = 'docs/figures'
SAVE_DPI = 100
TOP_N = 20
PRE_PAD = 30            # sessions of pre-cup context (shows the prior uptrend)

BLUE = '#1f77b4'        # the cup
ORANGE = '#ff7f0e'      # the handle
GREEN = '#2ca02c'       # the breakout
GRAY = '#888888'


def _surge(v: np.ndarray, t: int) -> float:
    """Rule-6 breakout volume surge: the trigger-day volume over the trailing
    ``VOL_AVG_WINDOW``-session average (the axis the plan ranks the eyeball
    panels by)."""
    base = float(np.mean(v[t - VOL_AVG_WINDOW:t]))
    return float(v[t]) / base if base > 0 else 0.0


def collect_detections(tickers: list[str]) -> list[dict]:
    """Every detection across ``tickers``, tagged with its volume surge.

    Metadata only (ticker + anatomy + surge); the plotting slices are pulled
    in a cheap second pass over just the top names, so a full run holds one
    ticker's arrays at a time.
    """
    splits = load_splits()
    found: list[dict] = []
    for tk in tickers:
        adj, cov = load_clean_daily(tk, splits)
        if adj is None or cov['cliff_flags']:
            continue                      # unresolved cliff -> excluded, per §2
        c, v, d = adj['close'], adj['volume'], adj['dates']
        for hit in detect_cup_handle(c, v):
            t = hit['t']
            found.append({'ticker': tk, 'surge': _surge(v, t), **hit,
                          'date': str(d[t])})
    # deterministic order: surge desc, then (ticker, t) so ties never wobble
    found.sort(key=lambda h: (-h['surge'], h['ticker'], h['t']))
    return found


def _panel(ax, tk: str, hit: dict, c: np.ndarray, d: np.ndarray) -> None:
    """One detection, clipped at the breakout day t (the right edge)."""
    lft, r, b, h0, t = hit['l'], hit['r'], hit['b'], hit['h0'], hit['t']
    s = max(0, lft - PRE_PAD)
    x = np.arange(s, t + 1)               # NOTHING past t: no return is shown
    ax.plot(x, c[s:t + 1], color=GRAY, lw=1.0, zorder=2)
    # the cup [lft, r] and the handle [h0, t] as shaded spans
    ax.axvspan(lft, r, color=BLUE, alpha=0.10, zorder=0)
    ax.axvspan(h0, t, color=ORANGE, alpha=0.15, zorder=0)
    # the breakout level (handle high) and the rims/bottom/breakout markers
    handle_high = float(np.max(c[h0:t]))
    ax.axhline(handle_high, color=GREEN, lw=0.7, ls='--', alpha=0.7, zorder=1)
    ax.plot([lft, r], [c[lft], c[r]], 'o', color=BLUE, ms=4, zorder=3)
    ax.plot(b, c[b], 'v', color=BLUE, ms=5, zorder=3)
    ax.plot(t, c[t], '^', color=GREEN, ms=8, zorder=4)   # breakout, right edge
    ax.set_title(f"{tk}  {hit['date']}  {hit['surge']:.1f}x vol  "
                 f"d{hit['depth']:.0%} r{hit['roundness']:.2f}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.margins(x=0.02)


def make_eyeball_figure(top: list[dict]) -> Figure:
    """The top-N detections in a grid, each clipped at its breakout."""
    splits = load_splits()
    cols = 4
    rows = (len(top) + cols - 1) // cols
    fig = Figure(figsize=(4 * cols, 2.6 * rows))
    axes = fig.subplots(rows, cols).ravel()
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ax, hit in zip(axes, top):
        tk = hit['ticker']
        if tk not in cache:
            adj, _ = load_clean_daily(tk, splits)
            cache[tk] = (adj['close'], adj['dates'])
        c, d = cache[tk]
        _panel(ax, tk, hit, c, d)
    for ax in axes[len(top):]:             # blank any trailing cells
        ax.axis('off')
    fig.suptitle(
        f'Cup-and-handle §4 eyeball pass — top {len(top)} detections by '
        'breakout volume surge (each clipped AT the breakout day; no '
        'post-breakout path shown)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def main() -> None:
    tickers = universe()
    print(f'scanning {len(tickers)} S&P 500 names for detections ...',
          flush=True)
    found = collect_detections(tickers)
    print(f'  {len(found)} detections total; rendering the top {TOP_N} by '
          'volume surge', flush=True)
    top = found[:TOP_N]
    for h in top:
        print(f"    {h['ticker']:5s} {h['date']}  {h['surge']:.1f}x  "
              f"depth {h['depth']:.0%} round {h['roundness']:.2f}")
    fig = make_eyeball_figure(top)
    path = f'{OUT}/cup_handle_eyeball.png'
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
