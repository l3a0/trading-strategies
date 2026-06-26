"""Always-run tests for the generative engine layer (Phase 2, generative_engine.py).

Pins (1, always-run) the rule-based family classifier `derive_family` — it must reproduce the 7
committed overlays' DECLARED families exactly, hit every registered family, and fail closed (None) on
a mechanism-incoherent signature; and (2, dataset-gated) the COMPOSER equivalence — a composed named
overlay is BYTE-IDENTICAL to its hand-written form on real SPY chains.
"""
from __future__ import annotations

import pandas as pd
import pytest

from edge_search import STRUCTURE_GRAMMAR, PremiumFamily, load_idea_ledger
from generative_engine import (
    _published_cell_keys,
    _row_overlay,
    derive_family,
    judge_compositions_against_published,
    record_compositions,
    run_composition,
    run_composition_round,
    score_composition,
)
from generative_grammar import canonical_key, composition_of
from vol_premium import STRUCTURE_SPECS, short_vol_statistics

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


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz twins)')
class TestScoreComposition:
    """Phase 3b: `score_composition` (the generative kill-gate) yields the SAME HAC t-stat as the named
    kill-gate for a composed named overlay — the scoring half of the byte-identical equivalence, and the
    per-cell scorer the Phase-4 LLM author feeds into."""

    @pytest.fixture(scope='class')
    def market(self):
        return _equiv_market('SPY', _SPY_DAILIES, extra_paths=[_SPY_PUTS])

    @pytest.mark.parametrize('name', sorted(_STRUCT_PARAMS))
    def test_t_stat_matches_the_named_overlay(self, market, name) -> None:
        dates, prices, store = market
        spec = STRUCTURE_SPECS[name]
        s, _, eq_named = _NAMED_OVERLAY[name](dates, prices, store,
                                              {**_STRUCT_PARAMS[name], 'capital': 100_000})
        t_named = short_vol_statistics(eq_named, s['capital'],
                                       rf=s['risk_free_rate'])['t_stat_newey_west']
        comp = composition_of(name, _STRUCT_PARAMS[name])
        row = score_composition(
            comp, 'SPY', dates, prices, store, capital=100_000, hedge_mode=spec['hedge_mode'],
            entry_guard=spec['entry_guard'], management=spec['management'],
            params={**spec['defaults'], **_STRUCT_PARAMS[name]})
        assert not row.get('measurement_invalid')
        assert row['key'] == canonical_key(comp)                        # keyed by canonical identity
        assert row['t_stat_newey_west'] == pytest.approx(t_named, abs=1e-9)
        assert row['p_value'] is not None and -1 <= row['predicted_sign'] <= 1
        # the INLINE MECHANISM GATE: the engine's actual entry signature -> derive_family reproduces the
        # DECLARED family, per composition (the per-structure analog of TestGrammarSignatureMatchesEngine).
        assert row['mechanism_ok'] is True
        assert row['family'] == STRUCTURE_GRAMMAR[name].family.value


class TestPublishedCellKeys:
    """Phase 3c (design A): the published idea_ledger.jsonl maps cleanly to (canonical_key, ticker) CELLS,
    so a generative composition dedups against the published stream head. Always-run: the committed ledger
    is in git."""

    def test_overlay_recovery(self) -> None:
        assert _row_overlay('short_call_25') == 'short_vol'        # committed short-vol alias
        assert _row_overlay('short_call_atm') == 'short_vol'
        assert _row_overlay('credit_spread') == 'credit_spread'   # bare overlay name
        assert _row_overlay('short_vol__dte45_target_delta0.5') == 'short_vol'  # menu-walk name
        assert _row_overlay('calendar__far_dte60_near_dte21') == 'calendar'

    def test_every_published_row_maps_to_a_cell(self) -> None:
        rows = load_idea_ledger()                                  # the committed lifetime ledger
        cells = _published_cell_keys(rows)                         # RAISES on any unmappable row (fail-loud)
        assert 0 < len(cells) <= len(rows)                         # total mapping; cells <= rows


class TestJudgeCompositionsAgainstPublished:
    """Phase 3c (design A): the lifetime judge keys on the (canonical_key, ticker) CELL, drops a fresh cell
    coincident with a published one, and never returns or mutates the published prior. Always-run synthetic
    (no engine) — the stream semantics, not the t-stats."""

    def _prior(self):
        # two published cells with real overlay names so _published_cell_keys can map them
        return [{'template': 'straddle', 'params': {'dte': 30}, 'ticker': 'X', 'p_value': 0.4},
                {'template': 'iron_condor', 'params': {'dte': 30, 'short_delta': 0.25, 'wing_delta': 0.1},
                 'ticker': 'Y', 'p_value': 0.4}]

    def test_coincident_cell_is_deduped_new_ticker_is_fresh(self) -> None:
        prior = self._prior()
        sk = canonical_key(composition_of('straddle', {'dte': 30}))
        fresh_in = [{'key': sk, 'ticker': 'X', 'p_value': 0.001},  # SAME structure+ticker as prior[0] -> drop
                    {'key': sk, 'ticker': 'Z', 'p_value': 0.001}]  # same structure, NEW ticker -> keep
        out = judge_compositions_against_published(fresh_in, prior_rows=prior)
        assert [(r['key'], r['ticker']) for r in out] == [(sk, 'Z')]
        assert 'elond_survivor' in out[0]                          # the lifetime judge ran

    def test_within_batch_dedup_and_prior_not_returned(self) -> None:
        prior = self._prior()
        sk = canonical_key(composition_of('straddle', {'dte': 30}))
        ck = canonical_key(composition_of('credit_spread', {'dte': 30, 'short_delta': 0.25, 'wing_delta': 0.1}))
        fresh_in = [{'key': ck, 'ticker': 'Z', 'p_value': 0.01},
                    {'key': ck, 'ticker': 'Z', 'p_value': 0.99},   # duplicate cell within the batch -> one kept
                    {'key': sk, 'ticker': 'X', 'p_value': 0.01}]   # coincident with prior -> dropped
        out = judge_compositions_against_published(fresh_in, prior_rows=prior)
        assert [(r['key'], r['ticker']) for r in out] == [(ck, 'Z')]
        # the published prior is never echoed back
        assert all((r['key'], r['ticker']) not in {(canonical_key(composition_of('straddle', {'dte': 30})), 'X')}
                   for r in out)


class TestRecordCompositions:
    """Phase 3c (design A): record_compositions appends to a SEPARATE generative ledger, deduped by cell,
    and never touches the published idea_ledger.jsonl."""

    def test_appends_and_dedups_by_cell(self, tmp_path) -> None:
        gen = str(tmp_path / 'gen_ledger.jsonl')
        rows = [{'key': 'abc', 'ticker': 'X', 'p_value': 0.1, 'elond_survivor': False},
                {'key': 'abc', 'ticker': 'Y', 'p_value': 0.1, 'elond_survivor': False}]
        assert record_compositions(rows, gen) == 2
        assert record_compositions(rows, gen) == 0                 # idempotent: same cells re-add nothing
        assert record_compositions([{'key': 'abc', 'ticker': 'Z', 'p_value': 0.1}], gen) == 1
        assert sum(1 for line in open(gen) if line.strip()) == 3


@pytest.mark.skipif(not (_HAVE_SPY and _HAVE_SPY_PUTS),
                    reason='needs spy_option_dailies.csv + spy_option_dailies_puts.csv (or .gz twins)')
class TestRunCompositionRound:
    """Phase 3c (design A) end-to-end on real SPY chains: a NAMED composition on a searched ticker is
    already in the published ledger, so the round scores it but the lifetime judge DEDUPS it against the
    real committed stream head (`fresh==0`), and `record=False` mutates nothing — the loop the menu-walker
    and the Phase-4 author drive, validated against the real ledger rather than a synthetic prior."""

    @pytest.fixture(scope='class')
    def market(self):
        return _equiv_market('SPY', _SPY_DAILIES, extra_paths=[_SPY_PUTS])

    def test_named_cell_is_deduped_against_the_published_ledger(self, market) -> None:
        dates, prices, store = market
        spec = STRUCTURE_SPECS['straddle']
        comp = composition_of('straddle', _STRUCT_PARAMS['straddle'])
        out = run_composition_round(
            [comp], 'SPY', dates, prices, store, capital=100_000, record=False,
            hedge_mode=spec['hedge_mode'], entry_guard=spec['entry_guard'],
            management=spec['management'], params={**spec['defaults'], **_STRUCT_PARAMS['straddle']})
        assert out['scored'] == 1
        assert out['fresh'] == 0          # the straddle/SPY cell is published -> deduped (design A head)
        assert out['recorded'] == 0       # record=False -> nothing written
