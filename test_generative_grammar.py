"""Always-run tests for the generative grammar core (Phase 1, generative_grammar.py).

Pins the production-rule validator (the numberless-value boundary), the canonical normal form
(totality / order-invariance / sign-exclusion), the reachable-count bound, and — the Phase-1
acceptance test — that the 7 named overlays are a VERIFIED SUB-GRAMMAR: every named grid point
maps to a valid Composition and the 70 of them get 70 DISTINCT canonical keys, so the published
ledger dedups against the generative identity unchanged.
"""
from __future__ import annotations

import math

import pytest

import generative_grammar as g
from edge_search import enumerate_grammar_templates, grid_universe_size
from generative_grammar import (
    DELTAS,
    DTES,
    MAX_EXPIRATIONS,
    MAX_LEGS,
    Composition,
    GrammarError,
    Leg,
    canonical_key,
    composition_of,
    leg_type_count,
    reachable_upper_bound,
    validate_composition,
)


def _comp(*legs, sign=1):
    return Composition(legs=tuple(legs), predicted_sign=sign)


class TestValidator:
    """validate_composition — the type-strict production-rule gate + the numberless-value boundary."""

    def test_accepts_a_well_formed_composition(self) -> None:
        c = _comp(Leg('short', 'call', ('delta', 0.25), 30))
        assert validate_composition(c) is c

    def test_off_bucket_delta_is_a_hard_error(self) -> None:
        # the numberless-VALUE boundary: a result-derived strike (not a committed bucket) cannot enter
        with pytest.raises(GrammarError, match='delta'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.241), 30)))

    def test_off_bucket_dte_is_a_hard_error(self) -> None:
        with pytest.raises(GrammarError, match='dte'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.25), 40)))

    def test_dte_is_type_strict(self) -> None:
        # 30.0 is not the committed int 30 — strict membership, like _validate_grammar
        with pytest.raises(GrammarError, match='dte'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.25), 30.0)))

    def test_predicted_sign_must_be_int_pm1(self) -> None:
        for bad in (0, 2, -2, True, 1.0):
            with pytest.raises(GrammarError, match='predicted_sign'):
                validate_composition(_comp(Leg('short', 'call', ('delta', 0.25), 30), sign=bad))

    def test_leg_count_bound(self) -> None:
        legs = [Leg('short', 'call', ('delta', 0.25), 30)] * (MAX_LEGS + 1)
        with pytest.raises(GrammarError, match='legs must be'):
            validate_composition(_comp(*legs))
        with pytest.raises(GrammarError, match='legs must be'):
            validate_composition(_comp())

    def test_expiration_cap(self) -> None:
        # three distinct expiries > MAX_EXPIRATIONS=2
        legs = [Leg('short', 'call', ('delta', 0.25), d) for d in (21, 30, 45)][:MAX_EXPIRATIONS + 1]
        with pytest.raises(GrammarError, match='expir'):
            validate_composition(_comp(*legs))

    def test_bad_side_or_right(self) -> None:
        with pytest.raises(GrammarError, match='side'):
            validate_composition(_comp(Leg('buy', 'call', ('delta', 0.25), 30)))
        with pytest.raises(GrammarError, match='right'):
            validate_composition(_comp(Leg('short', 'straddle', ('delta', 0.25), 30)))

    def test_same_strike_needs_exactly_one_delta_anchor(self) -> None:
        # the calendar shape: one delta anchor + one same-strike leg -> OK
        ok = _comp(Leg('short', 'call', ('delta', 0.50), 30), Leg('long', 'call', ('same',), 90))
        assert validate_composition(ok) is ok
        # a same-strike leg with NO delta anchor -> ambiguous -> reject
        with pytest.raises(GrammarError, match='same'):
            validate_composition(_comp(Leg('long', 'call', ('same',), 90),
                                       Leg('long', 'call', ('same',), 60)))
        # a same-strike leg with TWO different delta anchors -> ambiguous -> reject
        with pytest.raises(GrammarError, match='same'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.25), 30),
                                       Leg('short', 'put', ('delta', 0.50), 30),
                                       Leg('long', 'call', ('same',), 90)))

    def test_bad_strike_kind(self) -> None:
        with pytest.raises(GrammarError, match='strike'):
            validate_composition(_comp(Leg('short', 'call', ('offset', 1), 30)))

    def test_delta_is_type_strict(self) -> None:
        # a non-float that == a committed bucket spells to a DIFFERENT canonical token -> reject
        from decimal import Decimal
        from fractions import Fraction
        for bad in (Fraction(1, 4), Decimal('0.25'), Decimal('0.250')):
            with pytest.raises(GrammarError, match='delta'):
                validate_composition(_comp(Leg('short', 'call', ('delta', bad), 30)))

    def test_strike_tuple_shape_is_exact(self) -> None:
        # a 3rd element would smuggle a free (result-derived) number past the boundary
        with pytest.raises(GrammarError, match='delta strike'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.25, 1234.5), 30)))
        with pytest.raises(GrammarError, match='same strike'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.50), 30),
                                       Leg('long', 'call', ('same', 'x'), 90)))

    def test_malformed_strike_raises_grammar_error_not_indexerror(self) -> None:
        # the GrammarError-only contract: a bare ('delta',) / () must not leak IndexError
        with pytest.raises(GrammarError, match='delta strike'):
            validate_composition(_comp(Leg('short', 'call', ('delta',), 30)))
        with pytest.raises(GrammarError, match='non-empty tuple'):
            validate_composition(_comp(Leg('short', 'call', (), 30)))

    def test_same_tenor_same_strike_rejected(self) -> None:
        # the straddle's d/same spelling: a same-strike leg at the SAME tenor as its anchor is a
        # redundant spelling of a delta leg -> rejected, so one structure keeps one canonical key.
        with pytest.raises(GrammarError, match='DIFFERENT tenor'):
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.50), 30),
                                       Leg('short', 'put', ('same',), 30)))


class TestCanonicalKey:
    """canonical_key — total, order-invariant, sign-excluded content-addressed identity."""

    def test_deterministic_and_16_hex(self) -> None:
        c = _comp(Leg('short', 'call', ('delta', 0.25), 30))
        k = canonical_key(c)
        assert k == canonical_key(c) and len(k) == 16 and all(ch in '0123456789abcdef' for ch in k)

    def test_leg_order_invariant(self) -> None:
        a = _comp(Leg('short', 'call', ('delta', 0.25), 30), Leg('short', 'put', ('delta', 0.25), 30))
        b = _comp(Leg('short', 'put', ('delta', 0.25), 30), Leg('short', 'call', ('delta', 0.25), 30))
        assert canonical_key(a) == canonical_key(b)

    def test_predicted_sign_excluded(self) -> None:
        # a structure and its sign-flipped twin share ONE identity (the sign-shopping guard)
        legs = (Leg('short', 'call', ('delta', 0.25), 30),)
        assert canonical_key(_comp(*legs, sign=1)) == canonical_key(_comp(*legs, sign=-1))

    def test_distinct_structures_distinct_keys(self) -> None:
        a = canonical_key(_comp(Leg('short', 'call', ('delta', 0.25), 30)))
        b = canonical_key(_comp(Leg('short', 'call', ('delta', 0.30), 30)))   # diff delta
        c = canonical_key(_comp(Leg('long', 'call', ('delta', 0.25), 30)))    # diff side
        d = canonical_key(_comp(Leg('short', 'put', ('delta', 0.25), 30)))    # diff right
        assert len({a, b, c, d}) == 4

    def test_no_same_tenor_double_spelling(self) -> None:
        # the ATM straddle has exactly ONE valid spelling (d/d); the equivalent d/same spelling is
        # rejected at validation, so it cannot produce a second canonical key (no e-LOND re-spend).
        dd = _comp(Leg('short', 'call', ('delta', 0.50), 30), Leg('short', 'put', ('delta', 0.50), 30))
        assert validate_composition(dd) is dd                     # the canonical spelling is valid
        with pytest.raises(GrammarError):                         # the redundant spelling is not
            validate_composition(_comp(Leg('short', 'call', ('delta', 0.50), 30),
                                       Leg('short', 'put', ('same',), 30)))


class TestReachableBound:
    """leg_type_count / reachable_upper_bound — the governance/review count (not the power bound)."""

    def test_leg_type_count(self) -> None:
        assert leg_type_count() == len(g.SIDES) * len(g.RIGHTS) * len(DELTAS) * len(DTES) == 140

    def test_reachable_upper_bound_recomputed(self) -> None:
        L = leg_type_count()
        expected = sum(math.comb(L + k - 1, k) for k in range(1, MAX_LEGS + 1))
        assert reachable_upper_bound() == expected
        # the doc's point: the raw reachable space is astronomical, so the LIFETIME budget — not this
        # count — is the power bound.
        assert reachable_upper_bound() > 10_000_000

    def test_buckets_frozen_to_the_eight_overlays(self) -> None:
        # the Phase-1 decision: DELTAS / DTES are exactly the union of the 7 overlays' lattice values
        from edge_search import ALLOWED_GRID
        deltas, dtes = set(), set()
        for grid in ALLOWED_GRID.values():
            for knob, vals in grid.items():
                (dtes if 'dte' in knob else deltas).update(vals)
        assert set(DELTAS) == deltas
        assert set(DTES) == dtes


class TestNamedSubGrammar:
    """THE Phase-1 acceptance test: the 7 named overlays are a verified sub-grammar — every named
    grid point maps to a valid Composition and the 70 of them get 70 DISTINCT canonical keys."""

    def test_every_named_template_is_a_valid_composition(self) -> None:
        for t in enumerate_grammar_templates():
            comp = composition_of(t.overlay, dict(t.params))
            assert validate_composition(comp) is comp                    # already valid (no raise)

    def test_seventy_templates_seventy_distinct_keys(self) -> None:
        templates = enumerate_grammar_templates()
        keys = [canonical_key(composition_of(t.overlay, dict(t.params))) for t in templates]
        assert len(templates) == grid_universe_size() == 70
        assert len(set(keys)) == 70                                      # INJECTIVE: no collision

    def test_composition_of_is_stable(self) -> None:
        # same template -> same key on every call (recordable, dedup-stable identity)
        t = enumerate_grammar_templates()[0]
        k1 = canonical_key(composition_of(t.overlay, dict(t.params)))
        k2 = canonical_key(composition_of(t.overlay, dict(t.params)))
        assert k1 == k2

    def test_calendar_is_the_only_two_expiration_same_strike(self) -> None:
        # spot-check the trickiest mapping: the calendar near/far same-strike across two expiries
        cal = composition_of('calendar', {'near_dte': 30, 'far_dte': 90})
        assert len({leg.dte for leg in cal.legs}) == 2
        assert any(leg.strike == ('same',) for leg in cal.legs)
        assert all(leg.right == 'call' for leg in cal.legs)
