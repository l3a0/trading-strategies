"""Gap G — the multi-overlay portfolio harness (docs/van_tharp_gap_g.md).

Pure consumption of the engines' existing ``daily_equity`` streams: align
them on an inner date join, correlate them, combine them at pre-committed
weights. No engine imports, no I/O — a measurement primitive in the
trade_ledger / position_sizing family, and the first pandas import in the
leaf package (the leaf rule is about import direction, not third-party
dependencies; every consumer package already imports pandas).

Units, decided once (the design's basis section): every leg is **per-capital
daily P&L over zero-yield cash on its own deployed capital** — dollar diffs
over the FIXED capital base (the ``short_vol_statistics`` convention), never
prior-day-equity returns (compounding returns do not add; dollars do). A leg
carrying an ``rf_credit`` column is rf-netted with the same off-by-one
``short_vol_statistics`` uses (the credit lands at the start of the following
day), which keeps structure legs on their published statistical basis; a leg
without the column passes through raw — the column's presence is the switch,
no per-engine flags. CC legs are rf-free by engine construction.

Epistemic status: descriptive measurement substrate — EXPLORATORY, never a
registered verdict, never advice. Weights are pre-committed by callers;
weight optimization is an in-sample search and lives in no function here.
The daily Newey-West t (``common.stats.newey_west_summary``) stays the one
descriptive significance shape; drawdown comparisons carry no significance
claim at all.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _per_capital_pnl(daily_equity: pd.DataFrame, capital: float) -> pd.Series:
    """One leg's per-capital daily P&L series, indexed by the LATER day of
    each diff (diffs start at day 1 — the day-0 entry half-spread stays out,
    exactly as short_vol_statistics treats it)."""
    eq = daily_equity['equity'].to_numpy(dtype=float)
    pnl = np.diff(eq) / capital
    if 'rf_credit' in daily_equity.columns:
        # The structure engine's credit lands at the start of the FOLLOWING
        # day: net rf_credit[1:], the short_vol_statistics off-by-one.
        pnl = pnl - daily_equity['rf_credit'].to_numpy(dtype=float)[1:] / capital
    dates = daily_equity['date'].astype(str).iloc[1:]
    return pd.Series(pnl, index=pd.Index(dates, name='date'))


def align_streams(
    streams: Mapping[str, pd.DataFrame],
    *,
    capital: float | Mapping[str, float] = 100_000.0,
) -> pd.DataFrame:
    """A per-leg per-capital daily P&L panel on the INNER JOIN of dates.

    The join policy, stated once (docs/van_tharp_gap_g.md): pinned runs have
    different spans, so an inner join measures the combination where all
    legs actually ran; the surviving span is the panel's own index and must
    be reported with every result the panel feeds. Alignment is exact —
    leg-specific missing dates drop from the join, never interpolated.
    """
    if not streams:
        raise ValueError('no streams to align')
    cols = {}
    for name, df in streams.items():
        cap = capital[name] if isinstance(capital, Mapping) else capital
        if cap <= 0:
            raise ValueError(f'capital for {name!r} must be positive')
        cols[name] = _per_capital_pnl(df, float(cap))
    return pd.concat(cols, axis=1, join='inner')


def stream_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson over the aligned per-capital series. Full-sample and
    descriptive — the regime-conditional variant (the crisis-convergence
    question) is a named widening, not this function."""
    return panel.corr(method='pearson')


def combine_streams(panel: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """The combined per-capital daily P&L at PRE-COMMITTED weights summing
    to 1 — a $100K book allocating weight × capital per leg, positions
    scaled linearly. The caveat travels with every result: this linearly
    scales measured streams and ignores integer-contract granularity (a
    re-run at split capital would floor-divide contracts differently)."""
    if set(weights) != set(panel.columns):
        raise ValueError(f'weights {sorted(weights)} != panel legs {sorted(panel.columns)}')
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f'weights sum to {total}, not 1')
    out = pd.Series(0.0, index=panel.index)
    for name, w in weights.items():
        out = out + float(w) * panel[name]
    out.name = 'combined'
    return out


def max_drawdown_pct(pnl: pd.Series, *, capital: float = 100_000.0) -> float:
    """Max drawdown of a per-capital P&L stream under the ONE fixed-base
    definition every Gap G comparison shares: the cumulative curve is
    ``capital × (1 + cumsum(pnl))``, drawdown is percent of the running
    peak. Leg and combo DDs computed here are comparable to each other and
    deliberately NOT to the engines' published full-span equity-curve pins
    (different curves, different spans — new numbers get new pins)."""
    curve = capital * (1.0 + pnl.cumsum().to_numpy(dtype=float))
    peak = np.maximum.accumulate(np.maximum(curve, capital))
    dd = (peak - curve) / peak * 100.0
    return round(float(dd.max()), 2)
