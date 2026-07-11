"""Gap A — the trade-level R-multiple ledger (docs/van_tharp_gap_a.md).

Reduces an overlay's trade-EVENT stream (the ``trades`` list every engine
already returns) into uniform per-trade ``TradeRecord`` rows carrying dollar
P&L, a declared initial-risk basis R, the R-multiple, and MAE — then computes
the Van Tharp trade-level statistics (expectancy, SQN) plus the HAC-honest
sibling ``r_newey_west_t``.

Epistemic status: measurement substrate only. Any number this ledger produces
is EXPLORATORY (sample-spending, kill-or-justify — docs/explorations.md); the
daily Newey-West HAC t in ``short_vol_statistics`` / ``compute_statistics``
remains the repo's sole significance authority. ``sqn`` and ``r_newey_west_t``
are reported, never gates.

Dependency direction: ``common/`` is the leaf package both ``engine/`` and
``realchains/`` import (via ``common.paths``), so this module imports nothing
above ``common/`` — stdlib plus the sibling ``common.stats``, whose
``newey_west_t`` (numpy-vectorized Bartlett weights, auto-lag
``L = int(4·(n/100)^(2/9))``, ddof-1 zero-variance guard) is the repo's single
Newey-West definition, shared with ``factor/factor_backend`` and (via
``newey_west_summary``) both engines' statistics functions. Hoisting it into
the leaf is what lets every consumer reference one definition without
``common`` ever importing upward.

Event contract (what the engines emit today):

- entry actions: ``sell`` (both CC engines; per-share ``premium``) and
  ``enter`` (structure engine; per-share net ``credit`` + ``legs_detail`` =
  per-leg ``{sign, right, strike, entry_net, expiration}``).
- terminal actions: ``expiration`` / ``close`` / ``close_itm`` /
  ``close_stop`` / ``settle`` — each carries dollar ``pnl`` and (once A2 is
  threaded) ``mae``, the running min of the open position's daily
  mark-to-market P&L in dollars.
- ``settle_leg`` (calendar staggered settlement) is informational: its flow is
  rolled into the final ``settle`` pnl by the engine, so the reducer skips it.
- a trailing entry with no terminal event (backtest ends with the position
  open) is DROPPED — the engines' own win/loss counters ignore it the same
  way.

MAE conventions (documented, not repaired — docs/van_tharp_gap_a.md): the mark
is daily-bar (closing marks, so the true intraday worst is understated), the
real-chain engines carry marks forward on missing-quote days, and the final
record's ``mae = min(event mae, pnl, 0.0)`` so a loser's excursion includes
where it ended (Sweeney's convention). An event without ``mae`` (a caller
predating A2) degrades to ``min(pnl, 0.0)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from common.stats import newey_west_t

ENTRY_ACTIONS = frozenset({'sell', 'enter'})
TERMINAL_ACTIONS = frozenset({'expiration', 'close', 'close_itm', 'close_stop', 'settle'})
RISK_BASES = frozenset({'defined_max_loss', 'stop_distance', 'premium_collected'})


@dataclass(frozen=True)
class TradeRecord:
    """One completed trade, reduced from an entry/terminal event pair.

    Frozen: a row is a reduced historical fact whose derived fields
    (``r_multiple``, ``mae_r``, ``outcome``) are consistent with
    ``pnl``/``initial_risk`` only as computed at construction — mutating any
    one field downstream would silently desynchronize the others, and
    ``ledger_statistics`` must be able to trust rows it didn't build.
    """

    strategy: str          # "covered_call" (both CC engines) or the overlay's STRUCTURE_SPECS
                           # key: "short_vol", "straddle", "strangle", "iron_condor",
                           # "risk_reversal", "credit_spread", "calendar" (the short put is
                           # "short_vol" run with a put leg, not its own label). Free-form
                           # passthrough, not validated — the set grows with grammar widenings.
    ticker: str
    entry_date: str        # kept alongside close_date for regime_analysis (date, pnl) use
    close_date: str
    pnl: float             # realized dollars, read from the engine's terminal event
    risk_basis: str        # which R convention produced initial_risk (audit trail);
                           # "premium_collected_abs" marks the mixed-sign normalization
    initial_risk: float    # R, dollars, > 0
    r_multiple: float      # pnl / initial_risk
    mae: float             # worst intratrade unrealized P&L, dollars, <= 0
    mae_r: float           # mae / initial_risk
    outcome: str           # "win" | "loss" — pnl >= 0 is a win (the engines' convention)


def _defined_max_loss_per_share(legs: list[dict[str, Any]]) -> float:
    """Max loss per share of a defined-risk credit structure: the widest
    same-right short/long wing width minus the net credit. Exactly one short
    and one long leg per right is the credit-spread / iron-condor shape this
    basis is declared for; anything else fails loudly rather than guessing."""
    net_credit = sum(-leg['sign'] * leg['entry_net'] for leg in legs)
    widths: list[float] = []
    for right in {leg['right'] for leg in legs}:
        shorts = [leg['strike'] for leg in legs if leg['right'] == right and leg['sign'] < 0]
        longs = [leg['strike'] for leg in legs if leg['right'] == right and leg['sign'] > 0]
        if len(shorts) != 1 or len(longs) != 1:
            raise ValueError(
                f'defined_max_loss needs one short + one long {right} leg, '
                f'got {len(shorts)} short / {len(longs)} long'
            )
        widths.append(abs(shorts[0] - longs[0]))
    max_loss = max(widths) - net_credit
    if max_loss <= 0:
        raise ValueError(f'defined-risk max loss {max_loss:.4f} <= 0 — width/credit data error')
    return max_loss


def _premium_collected_per_share(entry: dict[str, Any]) -> tuple[float, str]:
    """Per-share premium R and the basis string actually recorded.

    All-short structures and covered calls: R = the net credit, basis
    "premium_collected". Mixed-sign structures (risk reversal, calendar) can
    carry a near-zero or net-debit credit, so R is floored at the GROSS short-
    leg premium — ``max(|net|, Σ short entry_net)`` — and the record says so
    via "premium_collected_abs" (the deferred floor choice in
    docs/van_tharp_gap_a.md, resolved here: the premium at risk is never less
    than what the short legs collected). For an all-short structure the two
    quantities coincide, so the floor is a no-op there by construction.
    """
    if 'premium' in entry:                      # both CC engines' `sell` event
        net = float(entry['premium'])
        if net <= 0:
            raise ValueError('covered-call premium <= 0 — the engines guard entry on this')
        return net, 'premium_collected'
    net = float(entry['credit'])                # structure engine's `enter` event
    legs = entry.get('legs_detail')
    if legs:
        gross_short = sum(leg['entry_net'] for leg in legs if leg['sign'] < 0)
        # The event's `credit` is round(x, 4) while legs_detail carries the
        # UNROUNDED entry_net, so the two can differ by float noise on a pure
        # all-short structure. The audit string is the honesty rail, so the
        # floor-binds decision is made within the 4dp rounding quantum — an
        # exact-equality test spuriously labelled ~25% of an all-short SPY
        # ledger 'premium_collected_abs' on 1-ulp differences.
        if net > 0 and gross_short <= net + 5e-5:
            return net, 'premium_collected'     # all-short: the net credit is R
        floored = max(abs(net), gross_short)
        if floored <= 0:
            raise ValueError('premium R <= 0 — no short-leg premium in legs_detail')
        return floored, 'premium_collected_abs'
    if net <= 0:
        raise ValueError('net credit <= 0 and no legs_detail to floor against')
    return net, 'premium_collected'


def build_trade_ledger(
    trades: list[dict[str, Any]],
    *,
    strategy: str,
    ticker: str,
    shares: int,
    risk_basis: str,
    stop_loss_mult: float | None = None,
) -> list[TradeRecord]:
    """Reduce one overlay's event stream to per-trade records.

    Everything after ``trades`` is keyword-only (the bare ``*``): the
    parameters are mostly same-typed (``strategy``/``ticker``/``risk_basis``
    are all strings), so a positional call could silently swap them and stamp
    wrong labels on every row — the call site must name each declaration.

    ``shares`` is the loop-level scalar every engine computes once at t=0
    (``100 * num_contracts``) — the event payloads are per-share, the ledger
    is dollars. ``risk_basis`` declares the R convention (a pinned choice, not
    a default — docs/van_tharp_gap_a.md); ``stop_distance`` additionally needs
    the overlay's ``stop_loss_mult``.
    """
    if risk_basis not in RISK_BASES:
        raise ValueError(f'unknown risk_basis {risk_basis!r} — one of {sorted(RISK_BASES)}')
    if risk_basis == 'stop_distance':
        if stop_loss_mult is None or stop_loss_mult <= 1.0:
            raise ValueError('stop_distance needs stop_loss_mult > 1.0 (the engine stop trigger)')

    records: list[TradeRecord] = []
    entry: dict[str, Any] | None = None
    for event in trades:
        action = event.get('action')
        if action in ENTRY_ACTIONS:
            entry = event
            continue
        if action not in TERMINAL_ACTIONS or entry is None:
            continue                            # settle_leg / unknown, or engine-guarded orphan
        pnl = float(event['pnl'])

        if risk_basis == 'defined_max_loss':
            legs = entry.get('legs_detail')
            if not legs:
                raise ValueError('defined_max_loss needs legs_detail on the enter event')
            risk_ps, basis = _defined_max_loss_per_share(legs), 'defined_max_loss'
        elif risk_basis == 'stop_distance':
            assert stop_loss_mult is not None
            premium = float(entry['premium'])
            if premium <= 0:
                raise ValueError('stop_distance premium <= 0 — the engines guard entry on this')
            risk_ps, basis = (stop_loss_mult - 1.0) * premium, 'stop_distance'
        else:
            risk_ps, basis = _premium_collected_per_share(entry)

        # r_multiple/mae_r divide by the STORED (2dp-rounded) risk so a consumer
        # recomputing pnl / record.initial_risk reproduces the record exactly.
        initial_risk = round(risk_ps * shares, 2)
        mae = min(float(event.get('mae', 0.0)), pnl, 0.0)
        records.append(TradeRecord(
            strategy=strategy, ticker=ticker,
            entry_date=str(entry['date']), close_date=str(event['date']),
            pnl=pnl, risk_basis=basis,
            initial_risk=initial_risk,
            r_multiple=round(pnl / initial_risk, 4),
            mae=round(mae, 2), mae_r=round(mae / initial_risk, 4),
            outcome='win' if pnl >= 0 else 'loss',
        ))
        entry = None
    return records


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default) on pre-sorted data."""
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def ledger_statistics(
    records: list[TradeRecord],
    *,
    r_normalizer: str = 'declared',
) -> dict[str, Any]:
    """Van Tharp trade-level statistics over a ledger.

    Three statistics, one judge (docs/van_tharp_gap_a.md): ``sqn`` is Tharp's
    naive one-sample t (kept solely for his interpretation bands; labelled
    anti-conservative under positive autocorrelation), ``r_newey_west_t`` is
    the HAC-corrected trade-level sibling, and NEITHER is a significance
    authority — the daily Newey-West t in the engines' statistics functions
    keeps that job.

    ``r_normalizer='avg_loss_1r'`` applies Tharp's ex-post fallback (Loc 739):
    1R := the mean absolute LOSING pnl, recomputing every R-multiple from
    dollars. It is ex-post — not knowable at entry — and offered only as the
    Tharp-comparison cross-check; 'declared' (the entry-time basis on each
    record) is the primary. With no losing trades the fallback is undefined
    and falls back to 'declared' (reported via the ``r_normalizer`` key).
    """
    if r_normalizer not in ('declared', 'avg_loss_1r'):
        raise ValueError(f'unknown r_normalizer {r_normalizer!r}')
    n = len(records)
    if n == 0:
        return {'n': 0, 'r_normalizer': 'declared', 'expectancy_r': 0.0, 'sqn': 0.0,
                'r_newey_west_t': 0.0, 'win_rate': 0.0, 'avg_win_r': 0.0, 'avg_loss_r': 0.0,
                'mae_r_distribution': {'mean': 0.0, 'median': 0.0, 'p10': 0.0, 'worst': 0.0}}
    applied = r_normalizer
    if r_normalizer == 'avg_loss_1r':
        losing = [abs(r.pnl) for r in records if r.pnl < 0]
        if losing:
            one_r = sum(losing) / len(losing)
            rs = [r.pnl / one_r for r in records]
            mae_rs = [min(r.mae, 0.0) / one_r for r in records]
        else:
            applied = 'declared'
            rs = [r.r_multiple for r in records]
            mae_rs = [r.mae_r for r in records]
    else:
        rs = [r.r_multiple for r in records]
        mae_rs = [r.mae_r for r in records]

    mean_r = sum(rs) / n
    std_r = math.sqrt(sum((v - mean_r) ** 2 for v in rs) / (n - 1)) if n > 1 else 0.0
    sqn = math.sqrt(n) * mean_r / std_r if std_r > 0 else 0.0
    wins = sum(1 for r in records if r.outcome == 'win')
    win_rs = [v for v, r in zip(rs, records) if r.outcome == 'win']
    loss_rs = [v for v, r in zip(rs, records) if r.outcome == 'loss']
    mae_sorted = sorted(mae_rs)
    return {
        'n': n,
        'r_normalizer': applied,
        'expectancy_r': round(mean_r, 4),
        'sqn': round(sqn, 3),
        # the series index is TRADE order, so lag 1 is one trade cycle (~a month), not one day
        'r_newey_west_t': round(newey_west_t(rs), 3),
        'win_rate': round(wins / n * 100, 1),
        'avg_win_r': round(sum(win_rs) / len(win_rs), 4) if win_rs else 0.0,
        'avg_loss_r': round(sum(loss_rs) / len(loss_rs), 4) if loss_rs else 0.0,
        'mae_r_distribution': {
            'mean': round(sum(mae_rs) / n, 4),
            'median': round(_percentile(mae_sorted, 0.50), 4),
            'p10': round(_percentile(mae_sorted, 0.10), 4),   # near-worst decile (mae_r <= 0)
            'worst': round(mae_sorted[0], 4),
        },
    }
