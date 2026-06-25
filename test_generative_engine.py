"""Always-run tests for the generative engine layer (Phase 2, generative_engine.py).

Pins the rule-based family classifier `derive_family`: it must reproduce the 7 committed overlays'
DECLARED families exactly (so the closed grammar's hand-authored family table is recovered by RULE),
hit every registered family, and fail closed (None) on a mechanism-incoherent signature.
"""
from __future__ import annotations

from edge_search import STRUCTURE_GRAMMAR, PremiumFamily
from generative_engine import derive_family


class TestDeriveFamily:
    """derive_family — the rule-based family classifier (replaces the per-overlay table lookup)."""

    def test_reproduces_all_seven_declared_families(self) -> None:
        # the rule applied to each overlay's DECLARED signature returns its DECLARED family. With
        # TestGrammarSignatureMatchesEngine pinning declared == engine-derived, the rule is therefore
        # engine-consistent: derive_family(engine signature) == the declared family, per structure.
        for overlay, og in STRUCTURE_GRAMMAR.items():
            assert derive_family(og.signature) is og.family, overlay

    def test_every_registered_family_is_reached(self) -> None:
        fams = {derive_family(og.signature) for og in STRUCTURE_GRAMMAR.values()}
        assert fams == {PremiumFamily.VARIANCE, PremiumFamily.SKEW,
                        PremiumFamily.CARRY, PremiumFamily.TERM}

    def test_incoherent_signature_is_unclassifiable(self) -> None:
        # a long-vega SINGLE-expiration structure harvests no registered premium -> None (fail-closed)
        incoherent = {'legs': 2, 'expirations': 1, 'net_vega': 'long',
                      'net_delta': 'neutral', 'net_skew': 'flat'}
        assert derive_family(incoherent) is None
        # a vega-neutral, skew-flat, delta-neutral book is likewise unclassifiable
        assert derive_family({'legs': 2, 'expirations': 1, 'net_vega': 'neutral',
                              'net_delta': 'neutral', 'net_skew': 'flat'}) is None

    def test_term_takes_priority_over_the_single_expiration_axes(self) -> None:
        # two expirations -> TERM regardless of the vega/delta/skew axes
        sig = {'legs': 2, 'expirations': 2, 'net_vega': 'short',
               'net_delta': 'long', 'net_skew': 'short_rich'}
        assert derive_family(sig) is PremiumFamily.TERM

    def test_skew_requires_neutral_vega_else_variance_or_carry(self) -> None:
        # net_skew present but SHORT vega is NOT SKEW — it stays VARIANCE (iron condor: long_rich + short
        # vega + neutral delta) or CARRY (credit spread: long_rich + short vega + LONG delta).
        ic = {'legs': 4, 'expirations': 1, 'net_vega': 'short',
              'net_delta': 'neutral', 'net_skew': 'long_rich'}
        assert derive_family(ic) is PremiumFamily.VARIANCE
        cs = {'legs': 2, 'expirations': 1, 'net_vega': 'short',
              'net_delta': 'long', 'net_skew': 'long_rich'}
        assert derive_family(cs) is PremiumFamily.CARRY
