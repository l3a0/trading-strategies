"""Always-run tests for the generative engine layer (Phase 2, generative_engine.py).

Pins (1, always-run) the rule-based family classifier `derive_family` — it must reproduce the 7
committed overlays' DECLARED families exactly, hit every registered family, and fail closed (None) on
a mechanism-incoherent signature; and (2, dataset-gated) the COMPOSER equivalence — a composed named
overlay is BYTE-IDENTICAL to its hand-written form on real SPY chains.
"""
from __future__ import annotations

import pandas as pd
import pytest

from edge_search import STRUCTURE_GRAMMAR, PremiumFamily
from generative_engine import derive_family, run_composition
from generative_grammar import composition_of
from vol_premium import STRUCTURE_SPECS

# Reuse the named-overlay equivalence fixtures (the SPY calls+puts merge, the named runners, the
# committed per-overlay params) so the composer is checked against the SAME data path + config.
from test_vol_premium import (
    _HAVE_SPY,
    _HAVE_SPY_PUTS,
    _NAMED_OVERLAY,
    _SPY_DAILIES,
    _SPY_PUTS,
    _STRUCT_PARAMS,
    _equiv_market,
)


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


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz twins)')
class TestComposerEquivalence:
    """THE Phase-2b acceptance test: a composed named overlay is BYTE-IDENTICAL (trades + daily equity)
    to its hand-written form, run under the SAME STRUCTURE_SPEC config — proving `composition_of`'s legs
    ARE the named selectors' legs. The run config (hedge_mode / entry_guard / management / defaults) is
    the overlay's spec, NOT the composer's, so holding it equal isolates the composer's leg selection."""

    @pytest.fixture(scope='class')
    def market(self):
        return _equiv_market('SPY', _SPY_DAILIES, extra_paths=[_SPY_PUTS])

    @pytest.mark.parametrize('name', sorted(_STRUCT_PARAMS))
    def test_composed_named_overlay_is_byte_identical(self, market, name) -> None:
        dates, prices, store = market
        spec = STRUCTURE_SPECS[name]
        base = {**_STRUCT_PARAMS[name], 'capital': 100_000}
        _, t_named, eq_named = _NAMED_OVERLAY[name](dates, prices, store, base)
        merged = {**spec['defaults'], **base}                       # the named runner merges these too
        _, t_comp, eq_comp = run_composition(
            composition_of(name, _STRUCT_PARAMS[name]), dates, prices, store, merged,
            hedge_mode=spec['hedge_mode'], entry_guard=spec['entry_guard'], management=spec['management'])
        assert len(t_named) > 0, f'{name} never traded on this store'   # the must_trade guard
        assert t_comp == t_named                                        # identical legs / entries / exits
        pd.testing.assert_frame_equal(eq_comp, eq_named)               # byte-identical daily equity
