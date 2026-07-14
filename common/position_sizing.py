"""Gaps C+B — fixed-fractional position sizing + the marble-bag resampler
(docs/van_tharp_gap_cb.md).

The sizer is a REPLAY layer over the Gap A ledger, not an engine change:
risking fraction ``f`` of current equity on a trade that returns ``r``
R-multiples multiplies equity by ``(1 + f*r)`` — the fixed-fractional
identity that makes sizing separable from the system, which is why Tharp
expresses systems as R-multiple distributions. The marble bag (his Loc
3728/4098 exercise) draws R-multiples with replacement and folds each drawn
career through that identity, producing the distribution of equity paths a
sizing rule implies. Together they are Experiment 1: the risk-of-ruin /
terminal-wealth Monte Carlo.

Epistemic status: DESCRIPTIVE risk measurement of already-measured systems —
EXPLORATORY, never a registered verdict, never an edge claim, and not
investment advice (``kelly_fraction`` in particular is a reference point on
the growth curve, never a recommendation). No hypothesis is flagged, nothing
enters the idea ledger, and the daily Newey-West t remains the repo's sole
significance authority.

Known limitation, stated loudly: the bag is IID — drawing with replacement
destroys serial dependence, including the regime clustering Gap D measured —
so it can misstate drawdown/ruin risk in either direction (the direction is
not modeled; the pinned MSFT trade-level HAC t hints the short-lag
autocovariance there is mildly negative, so no direction is assumed). Block
bootstraps and per-regime bags (Gap D's cells) are named widenings. A bag
normalized by Tharp's ex-post ``avg_loss_1r`` convention may be passed like
any other R list; results then carry that bag's ex-post label — the caller's
obligation, per the design's open question 2.

Determinism: one ``random.Random(seed)`` per call (the ``monte_carlo_shuffle``
convention), each path's draw indices generated in full BEFORE the equity
fold — an absorbed path stops compounding, not drawing — so the same seed
yields identical draw sequences at every fraction and a sweep compares
fractions on common random numbers.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

# Same-package reuse — one definition, per the NW-hoist lesson (common/stats.py).
from common.trade_ledger import _percentile

RUIN_THRESHOLD_25DD = 0.75   # the practitioner 25%-drawdown tolerance (design open question 1)


def simulate_sizing(
    r_multiples: Sequence[float],
    *,
    fraction: float,
    n_paths: int = 10_000,
    n_trades: int | None = None,
    seed: int = 42,
    ruin_threshold: float = 0.5,
    mae_r: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Draw ``n_paths`` careers from the bag and fold each through the
    fixed-fractional identity ``equity *= (1 + fraction * r)``.

    Ruin accounting is three-tiered (docs/van_tharp_gap_cb.md):

    - close-only (default): a path is ruin-flagged when post-trade equity
      falls to ``ruin_threshold`` of start or below (``ruin_basis:
      'close_only'``).
    - intratrade (``mae_r`` supplied, one MAE-R per bag entry): each draw
      carries its trade's MAE-R, and the trough
      ``pre_trade_equity * (1 + fraction * mae_r)`` is tested BEFORE the
      close multiplication. ``mae <= min(pnl, 0)`` by Gap A construction, so
      the trough never sits above the close and trough-then-close cannot
      miss a breach. The max-drawdown distribution stays close-equity-based
      in both modes; the trough enters only the ruin test.
    - absorption: a draw with ``fraction * r <= -1`` (or a trough with
      ``fraction * mae_r <= -1``) zeroes the account — the path is clamped
      to 0, marked ruined, and stops compounding, even when the close R
      would have recovered.

    ``p_ruin`` is reported at ``ruin_threshold`` (headline, default 0.5) and
    ``p_ruin_25dd`` at 0.75 (the practitioner 25%-drawdown tolerance), both
    on the same basis; ``p_ruin_25dd >= p_ruin`` holds whenever the caller's
    ``ruin_threshold`` stays at or below 0.75. Threshold breach alone is a flag; only absorption
    stops the path. ``n_trades=None`` replays one career the same length as
    the bag — a bootstrap with replacement, not a permutation.
    """
    rs = [float(r) for r in r_multiples]
    if not rs:
        raise ValueError('empty bag — nothing to draw')
    maes: list[float] | None = None
    if mae_r is not None:
        maes = [float(m) for m in mae_r]
        if len(maes) != len(rs):
            raise ValueError(f'mae_r length {len(maes)} != bag length {len(rs)}')
    if n_trades is None:
        n_trades = len(rs)
    if n_trades < 1:
        raise ValueError('n_trades must be >= 1')
    if fraction < 0:
        raise ValueError('fraction must be >= 0')

    rng = random.Random(seed)
    n_bag = len(rs)
    terminals: list[float] = []
    max_dds: list[float] = []
    min_tests: list[float] = []          # per-path worst test-equity (trough or close basis)
    for _ in range(n_paths):
        # All draws up front, independent of the fold (common random numbers).
        draws = [rng.randrange(n_bag) for _ in range(n_trades)]
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        min_test = 1.0
        for idx in draws:
            if maes is not None:
                trough = equity * (1.0 + fraction * maes[idx])
                min_test = min(min_test, trough)
                if trough <= 0.0:
                    equity = 0.0            # intratrade absorption: the close never happens
                    break
            equity *= 1.0 + fraction * rs[idx]
            min_test = min(min_test, equity)
            if equity <= 0.0:
                equity = 0.0                # absorbed: nothing compounds from zero
                break
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)
        if equity <= 0.0:
            min_test = 0.0
            max_dd = 1.0
        terminals.append(equity)
        max_dds.append(max_dd)
        min_tests.append(min_test)

    terminals.sort()
    max_dds.sort()
    return {
        'fraction': fraction,
        'n_paths': n_paths,
        'n_trades': n_trades,
        'ruin_threshold': ruin_threshold,
        'ruin_basis': 'intratrade' if maes is not None else 'close_only',
        'terminal': {
            'median': round(_percentile(terminals, 0.50), 4),
            'p10': round(_percentile(terminals, 0.10), 4),
            'p90': round(_percentile(terminals, 0.90), 4),
        },
        'max_drawdown': {
            'median': round(_percentile(max_dds, 0.50), 4),
            'p90': round(_percentile(max_dds, 0.90), 4),
            'worst': round(max_dds[-1], 4),
        },
        'p_ruin': round(sum(1 for m in min_tests if m <= ruin_threshold) / n_paths, 4),
        'p_ruin_25dd': round(sum(1 for m in min_tests if m <= RUIN_THRESHOLD_25DD) / n_paths, 4),
        'p_negative_terminal': round(sum(1 for t in terminals if t < 1.0) / n_paths, 4),
    }


def sizing_sweep(
    r_multiples: Sequence[float],
    *,
    fractions: Sequence[float] = (0.0025, 0.005, 0.01, 0.02, 0.03),
    **kwargs: Any,
) -> dict[float, dict[str, Any]]:
    """One ``simulate_sizing`` per fraction, identical seed and draws each —
    common random numbers, so fractions differ only by sizing. The default
    grid is sourced line-by-line from the book (docs/van_tharp_gap_cb.md:
    the 0.25% wide-stop, Tharp's 0.5–2.5% band, the 0.8–1% practitioner,
    the 2–3% envelope, Basso's 3% "gunslinger")."""
    return {f: simulate_sizing(r_multiples, fraction=f, **kwargs) for f in fractions}


def kelly_fraction(r_multiples: Sequence[float]) -> float:
    """The log-optimal fraction: argmax of ``mean(log(1 + f*r))`` over 999
    fixed interior grid points on ``(0, 1/|min r|)`` — the absorption
    boundary, past which one draw is fatal — with ``f = 0`` (growth 0) as
    the implicit baseline candidate. Deterministic, seed-free, no RNG.

    A REFERENCE POINT ONLY, never a recommendation. Edge rules per the
    design: an empty bag raises; ``mean(r) <= 0`` returns exactly 0.0
    without searching (concave growth curve, nonpositive slope at the
    origin — an all-loser bag is the extreme case); a bag with no negative
    R has no absorption boundary and an unbounded optimum, so it raises
    rather than inventing a cap.
    """
    rs = [float(r) for r in r_multiples]
    if not rs:
        raise ValueError('empty bag — no growth curve to optimize')
    if sum(rs) / len(rs) <= 0.0:
        return 0.0
    worst = min(rs)
    if worst >= 0.0:
        raise ValueError('no losing R in the bag — the log-optimal fraction is unbounded')
    bound = 1.0 / abs(worst)
    best_f, best_g = 0.0, 0.0
    for k in range(1, 1000):
        f = bound * k / 1000.0
        g = sum(math.log(1.0 + f * r) for r in rs) / len(rs)
        if g > best_g:
            best_f, best_g = f, g
    return best_f
