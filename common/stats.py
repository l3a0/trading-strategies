"""Shared statistical primitives for the leaf ``common`` package.

``newey_west_summary`` is the single home of the repo's naive-vs-Newey-West
significance arithmetic. Every surface that reports a HAC t consumes it:
``engine.cc_backtest.compute_statistics`` (the proxy engine's excess-return
block), ``realchains.vol_premium.short_vol_statistics`` (the structure
engine's, feeding every campaign cell), ``factor/factor_backend`` (the daily
IC series), and ``common/trade_ledger`` (the per-trade R-multiple series).
It lives here — not in any of those packages — because ``common/`` is the
leaf everything else imports, so one definition serves all four without a
dependency inversion.

THE ARITHMETIC HERE IS PINNED. The proxy regressions, the real-chain and
structure-campaign pins, the committed ``idea_ledger.jsonl`` rows, and the
factor rows all trace to these exact float ops. Reordering a sum, swapping
``np.mean`` for a hand loop, or changing a guard is a repo-wide re-pin event
(and, for the structure path, a ``STRUCTURE_ENGINE_VERSION`` bump) — don't
touch the body without intending exactly that.

The lag index is whatever the caller's series index is — calendar days for
the daily engines and the factor IC, trade order for the ledger (there,
lag 1 is one trade cycle, \\~a month). The auto-lag rule and Bartlett weights
don't care; interpretation does, so each caller documents its own units.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import numpy as np


def newey_west_lag(n: int) -> int:
    """The auto-lag rule ``L = int(4·(n/100)^(2/9))`` — Andrews (1991)
    framework, Newey & West (1994) operational formula. Exposed separately so
    figure annotations quote the same L the estimator actually uses."""
    return int(4 * (n / 100) ** (2 / 9))


class NeweyWestSummary(NamedTuple):
    """The significance block every statistics function reports from."""

    n: int
    mean: float            # series mean — the quantity under test vs 0
    var: float             # ddof=1 sample variance
    t_naive: float         # IID t: mean / sqrt(var/n) — anti-conservative under autocorrelation
    t_newey_west: float    # Bartlett-weighted HAC t — the honest sibling
    lag: int               # auto-lag L actually used


def newey_west_summary(x: np.ndarray | Sequence[float]) -> NeweyWestSummary:
    """Naive and Newey-West HAC t-stats of a series' mean against H0: mean = 0.

    The HAC variance of the mean is
    ``Var(mean) = (1/n) * [gamma_0 + 2 * sum_{k=1}^{L} w_k * gamma_k]`` where
    ``gamma_k`` is the lag-k autocovariance and ``w_k = 1 - k/(L+1)`` are the
    Bartlett weights that enforce positive-definiteness. Auto-lag
    ``L = int(4·(n/100)^(2/9))`` — the framework is Andrews (1991), the
    operational formula Newey & West (1994). The NW variance is floored at 0
    (a strongly negative autocovariance sum can push it below zero at short
    samples); a zero SE reports t = 0.0 rather than raising, and n < 2
    returns an all-zero summary — the callers' own data-sufficiency guards
    fire before either case matters.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n < 2:
        return NeweyWestSummary(n, float(arr.mean()) if n else 0.0, 0.0, 0.0, 0.0, 0)
    mean = float(np.mean(arr))
    var0 = float(np.var(arr, ddof=1))
    se_naive = math.sqrt(var0 / n) if var0 > 0 else 0.0
    t_naive = mean / se_naive if se_naive > 0 else 0.0
    lag = newey_west_lag(n)
    nw_sum = 0.0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1)
        nw_sum += weight * float(np.mean((arr[:-k] - mean) * (arr[k:] - mean)))
    var_mean = (var0 + 2.0 * nw_sum) / n
    se = math.sqrt(max(var_mean, 0.0))
    t_nw = mean / se if se > 0 else 0.0
    return NeweyWestSummary(n, mean, var0, t_naive, t_nw, lag)


def newey_west_t(x: np.ndarray | Sequence[float]) -> float:
    """The HAC t alone — the ``factor``/``trade_ledger`` entry point."""
    return newey_west_summary(x).t_newey_west
