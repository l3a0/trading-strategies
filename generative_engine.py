"""generative_engine.py — the engine-touching layer of the generative grammar (Phase 2).

Bridges the pure grammar core (generative_grammar.py) to the options engine (vol_premium.py). This
first increment is the rule-based FAMILY classifier `derive_family`: it replaces the closed grammar's
hand-authored per-overlay family table (`structure_family` / `STRUCTURE_GRAMMAR[o].family`) with a
RULE over the engine's robust signature axes, so an ARBITRARY composition's mechanism can be typed
without a human declaring it — the auto-typing an open grammar needs (docs/generative_grammar_plan.md,
"the mechanism check goes inline").

The remaining Phase-2 pieces — the COMPOSER (resolve a `Composition` to the engine's entry legs,
replicating each named selector's inter-leg constraints) and the byte-identical EQUIVALENCE proof — are
the next, engine-faithful increment; they live here too when built.
"""
from __future__ import annotations

from typing import Any

from edge_search import PremiumFamily


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
