"""generative_engine.py — the engine-touching layer of the generative grammar (Phase 2).

Bridges the pure grammar core (generative_grammar.py) to the options engine (vol_premium.py). This
first increment is the rule-based FAMILY classifier `derive_family`: it replaces the closed grammar's
hand-authored per-overlay family table (`structure_family` / `STRUCTURE_GRAMMAR[o].family`) with a
RULE over the engine's robust signature axes, so an ARBITRARY composition's mechanism can be typed
without a human declaring it — the auto-typing an open grammar needs (docs/generative_grammar_plan.md,
"the mechanism check goes inline").

The COMPOSER (`compose_legs` / `run_composition`) resolves a `Composition` to the engine's entry legs
and runs it through the single generic overlay engine, replicating each named selector's inter-leg
constraints (the anchor leg sets the expiration; in-expiration shorts are band-filtered; long wings are
struck strictly further OTM than their same-right short; a `('same',)` leg matches the anchor's strike
at a later expiry). A composed NAMED overlay is byte-identical to its hand-written form — the
dataset-gated equivalence proof (docs/generative_grammar_plan.md, Phase 2).
"""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from edge_search import PremiumFamily
from generative_grammar import Composition, Leg
from real_cc_backtest import COMMISSION_PER_SHARE, select_entry
from vol_premium import run_real_structure_overlay, select_put_entry

# candidate tuple = (dte, delta, bid, ask, mid, expiration, strike, contractID)
_BAND = {'call': (0.05, 0.60), 'put': (-0.60, -0.05)}      # the entry bands select_entry/put use


def derive_family(signature: dict[str, Any]) -> PremiumFamily | None:
    """Classify a structure's robust greek signature into a `PremiumFamily` by RULE — the generative
    replacement for the per-overlay family table. `signature` is `structure_greek_signature`'s output
    `{legs, expirations, net_vega, net_delta, net_skew}` (vega/delta in {short, long, neutral}; skew in
    {short_rich, long_rich, flat}).

    Returns `None` for a MECHANISM-INCOHERENT signature that maps to no registered family — e.g. a
    long-vega SINGLE-expiration structure that harvests no committed premium. That `None` is the
    fail-closed verdict the inline gate turns into a rejected / `measurement_invalid` cell (the
    foil-paper defense, per composition rather than per overlay name).

    The rule reproduces the 7 committed overlays' DECLARED families exactly (pinned by
    test_generative_engine.py; and via `TestGrammarSignatureMatchesEngine` pinning declared ==
    engine-derived, it is engine-consistent too), in priority order:

      * TWO expirations           -> TERM     (opposite-sign vega across tenors — the calendar);
      * net_skew + NEUTRAL vega   -> SKEW     (the risk reversal's wing asymmetry on a vega-flat book);
      * LONG delta + SHORT vega   -> CARRY    (the credit spread — theta-positive defined-risk);
      * (else) SHORT vega         -> VARIANCE (the four short-vol overlays);
      * (else)                    -> None     (no term, no skew-on-neutral-vega, no short premium).
    """
    if signature['expirations'] >= 2:
        return PremiumFamily.TERM
    if signature['net_skew'] != 'flat' and signature['net_vega'] == 'neutral':
        return PremiumFamily.SKEW
    if signature['net_delta'] == 'long' and signature['net_vega'] == 'short':
        return PremiumFamily.CARRY
    if signature['net_vega'] == 'short':
        return PremiumFamily.VARIANCE
    return None


# --- the composer: resolve a Composition to the engine's entry legs (byte-identical to the selectors) -
def _signed_target(leg: Leg) -> float:
    """The signed vendor-delta target for a delta-targeted leg: +d for a call, -d for a put."""
    return leg.strike[1] if leg.right == 'call' else -leg.strike[1]


def _in_band(c: tuple, right: str) -> bool:
    lo, hi = _BAND[right]
    return lo < c[1] < hi


def _leg_dict(c: tuple, leg: Leg, fill: str) -> dict[str, Any]:
    """Build the engine leg dict from a resolved candidate — byte-identical to the named `_legs_*`
    builders: shorts fill at bid (c[2]), longs at ask (c[3]), or the mid (c[4]) under fill='mid', with
    COMMISSION_PER_SHARE baked into entry_net (subtracted for a short sale, added for a buy)."""
    sign = -1 if leg.side == 'short' else 1
    px = c[4] if fill != 'bid_ask' else (c[2] if sign == -1 else c[3])
    entry_net = px - COMMISSION_PER_SHARE if sign == -1 else px + COMMISSION_PER_SHARE
    return {'sign': sign, 'right': leg.right, 'strike': c[6], 'contract': c[7],
            'entry_net': entry_net, 'mid': c[4], 'delta': c[1], 'expiration': c[5]}


def compose_legs(composition: Composition,
                 min_gap_dte: int = 30) -> Callable[[dict, dict], list[dict] | None]:
    """Turn a `Composition` into a `select(day, params)` callable for `run_real_structure_overlay`,
    resolving legs exactly as the named selectors do (so a composed named overlay is byte-identical):

      * group legs by dte target — each group is one expiration;
      * the group's ANCHOR is its first CALL leg (else its first put) — resolved via `select_entry` /
        `select_put_entry`, fixing the expiration E (and the same-strike reference K);
      * other SHORT legs -> the band-filtered candidate of their right in E nearest the signed target;
      * LONG wings -> the candidate of their right in E struck strictly further OTM than the same-right
        short (buyable, NOT band-filtered) nearest the signed target;
      * a `('same',)` leg -> the anchor strike K at a LATER expiration (>= min_gap_dte beyond the anchor)
        nearest its far-dte target — the calendar far leg.

    Returns `None` (no trade that day) if any leg is unavailable, exactly like the named selectors."""
    legs = composition.legs
    delta_legs = [leg for leg in legs if leg.strike[0] == 'delta']
    same_legs = [leg for leg in legs if leg.strike[0] == 'same']
    strike_anchor = delta_legs[0] if same_legs else None      # the validator guarantees one delta tenor

    def _select(day: dict, params: dict) -> list[dict] | None:
        fill = str(params.get('fill', 'bid_ask'))
        cands = day['candidates']
        picked: dict[Leg, tuple] = {}
        dtes: list[int] = []
        for leg in delta_legs:
            if leg.dte not in dtes:
                dtes.append(leg.dte)
        for dte in dtes:
            group = [leg for leg in delta_legs if leg.dte == dte]
            anchor = next((leg for leg in group if leg.right == 'call'), group[0])
            picker = select_entry if anchor.right == 'call' else select_put_entry
            a = picker(day, dte, _signed_target(anchor))
            if a is None:
                return None
            picked[anchor], expiry = a, a[5]
            shorts: dict[str, tuple] = {anchor.right: a} if anchor.side == 'short' else {}
            # shorts before longs — a long wing references its same-right short's strike
            for leg in sorted((x for x in group if x is not anchor), key=lambda x: x.side == 'long'):
                tgt = _signed_target(leg)
                if leg.side == 'short':
                    pool = [c for c in cands if c[5] == expiry and c[2] > 0 and _in_band(c, leg.right)]
                    if not pool:
                        return None
                    c = min(pool, key=lambda c: abs(c[1] - tgt))
                    shorts[leg.right] = c
                else:
                    s = shorts.get(leg.right)
                    if s is None:                          # a non-anchor long with no same-right short
                        return None                         # is outside the named grammar (Phase 2b+)
                    pool = [c for c in cands if c[5] == expiry and c[3] > 0
                            and (c[1] > 0 and c[6] > s[6] if leg.right == 'call'
                                 else c[1] < 0 and c[6] < s[6])]
                    if not pool:
                        return None
                    c = min(pool, key=lambda c: abs(c[1] - tgt))
                picked[leg] = c
        if same_legs:
            a = picked[strike_anchor]
            k, near_exp, near_dte = a[6], a[5], a[0]
            for leg in same_legs:
                far = [c for c in cands if c[6] == k and c[5] > near_exp and c[3] > 0
                       and (c[1] > 0 if leg.right == 'call' else c[1] < 0)
                       and c[0] - near_dte >= min_gap_dte]
                if not far:
                    return None
                picked[leg] = min(far, key=lambda c: abs(c[0] - leg.dte))
        return [_leg_dict(picked[leg], leg, fill) for leg in legs]

    return _select


def run_composition(composition: Composition, dates: list[str], prices: list[float],
                    store: dict, params: dict | None = None, *,
                    hedge_mode: str = 'combined', entry_guard: str = 'each_short_positive',
                    management: str = 'hold',
                    ) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    """Run a `Composition` through the single generic overlay engine — the ADDITIVE generic runner (it
    does not touch the named-overlay dispatch). The defaults are the generative grammar's own config
    (delta-hedged 'combined', held to expiry); the engine config is exposed so the equivalence test can
    hold it equal to a named overlay's STRUCTURE_SPEC and isolate the COMPOSER's leg selection (which is
    what the sub-grammar proof is about — the run config is the overlay's spec, not the composer's)."""
    return run_real_structure_overlay(
        dates, prices, store, params or {}, select=compose_legs(composition),
        entry_guard=entry_guard, hedge_mode=hedge_mode, management=management)
