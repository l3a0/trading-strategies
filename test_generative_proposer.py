"""Always-run tests for the generative LLM-author front-end (Phase 4, generative_proposer.py).

No model, no data: a STUB GenLLMProposer feeds fixed coordinate proposals through the gate, and the
numberless prompt is checked structurally. Pins the seal-critical behavior — the grammar wall, the
sealed/universe/onboarding gates, the (canonical_key, ticker) dedup (incl. the sign-shopping collapse),
the batch cap, and the numberless-corpus seal on the prompt.
"""
from __future__ import annotations

import pytest

from edge_search import STRUCTURE_CAMPAIGN, ProposalBatch
from generative_proposer import (
    _composition_from_proposal,
    build_composition_prompt,
    gate_compositions,
)
from read_gate_wire import GEN_PROPOSAL_FIELDS

# a search ticker (in STRUCTURE_CAMPAIGN.search) and the sealed ticker, named explicitly so the tests do
# not depend on which datasets happen to be present locally.
_SEARCH = STRUCTURE_CAMPAIGN.search[0]                       # e.g. 'MSFT'
_SEARCH2 = STRUCTURE_CAMPAIGN.search[1]                      # a second onboarded name
_OFF_SEARCH = STRUCTURE_CAMPAIGN.search[2]                   # a search ticker we leave UN-onboarded
_SEALED = STRUCTURE_CAMPAIGN.sealed[0]                       # e.g. 'TLT'


def _author(proposals):
    """A stub GenLLMProposer: returns the given proposals verbatim with a sentinel model identity."""
    def author(corpus, onboarded) -> ProposalBatch:
        return ProposalBatch(proposals=tuple(proposals), model_requested='stub',
                             model_served='stub', temperature=0.0, prompt_sha='stub')
    return author


def _prop(legs, ticker, sign=1, **extra):
    return {'legs': legs, 'ticker': ticker, 'predicted_sign': sign, **extra}


_SHORT_CALL = [{'side': 'short', 'right': 'call', 'delta': 0.25, 'dte': 30}]


@pytest.fixture
def onboarded_two(monkeypatch):
    """Make _SEARCH and _SEARCH2 onboarded, everything else not — deterministic regardless of local data."""
    monkeypatch.setattr('generative_proposer._is_onboarded', lambda t: t in {_SEARCH, _SEARCH2})


class TestGateCompositions:
    """gate_compositions — the coordinate-only grammar gate (the generative llm_propose_candidates)."""

    def test_accepts_a_valid_composition_on_an_onboarded_ticker(self, onboarded_two) -> None:
        cells, needs, rejected, batch = gate_compositions(
            _author([_prop(_SHORT_CALL, _SEARCH)]), corpus=[], tried_keys=set())
        assert len(cells) == 1 and not needs and not rejected
        comp, ticker = cells[0]
        assert ticker == _SEARCH and comp.legs[0].right == 'call'   # a validated Composition
        assert batch.proposals                                       # the raw batch is returned for audit

    def test_off_grammar_proposal_is_rejected_not_raised(self, onboarded_two) -> None:
        bad = [{'side': 'short', 'right': 'call', 'delta': 0.241, 'dte': 30}]  # 0.241 not a DELTAS bucket
        cells, _, rejected, _ = gate_compositions(
            _author([_prop(bad, _SEARCH)]), corpus=[], tried_keys=set())
        assert not cells and len(rejected) == 1 and 'off-grammar' in rejected[0]['reason']

    def test_sealed_ticker_is_rejected(self, onboarded_two) -> None:
        cells, _, rejected, _ = gate_compositions(
            _author([_prop(_SHORT_CALL, _SEALED)]), corpus=[], tried_keys=set())
        assert not cells and 'sealed' in rejected[0]['reason']

    def test_off_campaign_ticker_is_rejected(self, onboarded_two) -> None:
        cells, _, rejected, _ = gate_compositions(
            _author([_prop(_SHORT_CALL, 'ZZZZ')]), corpus=[], tried_keys=set())
        assert not cells and 'off-campaign' in rejected[0]['reason']

    def test_un_onboarded_search_ticker_is_flagged_not_run(self, onboarded_two) -> None:
        cells, needs, rejected, _ = gate_compositions(
            _author([_prop(_SHORT_CALL, _OFF_SEARCH)]), corpus=[], tried_keys=set())
        assert not cells and not rejected and needs == [_OFF_SEARCH]

    def test_malformed_proposal_is_rejected_not_raised(self, onboarded_two) -> None:
        # not a dict, no ticker, and no legs — each becomes a reject, none aborts the round
        props = ['not a dict', {'predicted_sign': 1}, {'ticker': _SEARCH, 'predicted_sign': 1}]
        cells, _, rejected, _ = gate_compositions(_author(props), corpus=[], tried_keys=set())
        assert not cells and len(rejected) == 3

    def test_dedup_collapses_duplicates_and_sign_flips(self, onboarded_two) -> None:
        # the SAME structure twice, then its SIGN-FLIP — canonical_key excludes predicted_sign, so all
        # three are ONE cell (the sign-shopping guard); only the first survives.
        props = [_prop(_SHORT_CALL, _SEARCH, sign=1),
                 _prop(_SHORT_CALL, _SEARCH, sign=1),
                 _prop(_SHORT_CALL, _SEARCH, sign=-1)]
        cells, _, _, _ = gate_compositions(_author(props), corpus=[], tried_keys=set())
        assert len(cells) == 1

    def test_tried_keys_excludes_a_published_cell(self, onboarded_two) -> None:
        from generative_grammar import canonical_key
        comp = _composition_from_proposal(_prop(_SHORT_CALL, _SEARCH))
        tried = {(canonical_key(comp), _SEARCH)}
        cells, _, _, _ = gate_compositions(
            _author([_prop(_SHORT_CALL, _SEARCH)]), corpus=[], tried_keys=tried)
        assert not cells                                            # already tried -> silently skipped

    def test_batch_cap(self, onboarded_two) -> None:
        # three DISTINCT structures, cap at 2 -> only 2 accepted
        a = [{'side': 'short', 'right': 'call', 'delta': 0.25, 'dte': 30}]
        b = [{'side': 'short', 'right': 'call', 'delta': 0.30, 'dte': 30}]
        c = [{'side': 'short', 'right': 'put', 'delta': 0.25, 'dte': 30}]
        props = [_prop(a, _SEARCH), _prop(b, _SEARCH), _prop(c, _SEARCH)]
        cells, _, _, _ = gate_compositions(_author(props), corpus=[], tried_keys=set(), max_batch=2)
        assert len(cells) == 2

    def test_same_strike_calendar_proposal_validates(self, onboarded_two) -> None:
        # the cross-tenor ('same',) leg: an anchor + a same-strike far leg -> a valid TERM composition
        legs = [{'side': 'short', 'right': 'call', 'delta': 0.5, 'dte': 30},
                {'side': 'long', 'right': 'call', 'strike': 'same', 'dte': 90}]
        cells, _, rejected, _ = gate_compositions(
            _author([_prop(legs, _SEARCH)]), corpus=[], tried_keys=set())
        assert len(cells) == 1 and not rejected

    def test_duplicate_leg_proposal_is_rejected(self, onboarded_two) -> None:
        # the seal finding: two identical legs is a scale-multiple (off-grammar via validate_composition),
        # so the gate rejects it rather than accepting it as a fresh e-LOND cell.
        leg = {'side': 'short', 'right': 'call', 'delta': 0.25, 'dte': 30}
        cells, _, rejected, _ = gate_compositions(
            _author([_prop([leg, dict(leg)], _SEARCH)]), corpus=[], tried_keys=set())
        assert not cells and 'off-grammar' in rejected[0]['reason']


class TestCompositionPrompt:
    """build_composition_prompt — the NUMBERLESS prompt; the seal is assert_numberless on the corpus input."""

    _SCRUBBED = [{'phase': 'structure', 'template': 'straddle', 'ticker': _SEARCH,
                  'params': {'dte': 30}, 'predicted_sign': 1, 'verdict': 'KILLED'}]

    def test_prompt_contains_grammar_primitives_and_universe(self) -> None:
        prompt = build_composition_prompt(self._SCRUBBED, (_SEARCH, _SEARCH2))
        assert '0.25' in prompt and 'short' in prompt and 'call' in prompt   # the leg primitives
        assert _SEARCH in prompt and _SEARCH2 in prompt                       # the onboarded universe
        assert 'JSON array' in prompt and 'predicted_sign' in prompt         # the output contract

    def test_seal_fires_on_a_raw_ledger_row(self) -> None:
        # the #1 builder bug: passing raw load_idea_ledger() rows (banned KEYS) instead of the scrub
        raw = [{**self._SCRUBBED[0], 't_stat_newey_west': 2.1, 'p_value': 0.02}]
        with pytest.raises(ValueError, match='numberless'):
            build_composition_prompt(raw, (_SEARCH,))

    def test_scrubbed_corpus_passes_the_seal(self) -> None:
        # a clean scrubbed corpus assembles without raising (no banned keys present)
        assert isinstance(build_composition_prompt(self._SCRUBBED, (_SEARCH,)), str)


class TestGenProposalFields:
    """The wire contract's generative coordinate allow-list."""

    def test_gen_proposal_fields(self) -> None:
        assert GEN_PROPOSAL_FIELDS == ('legs', 'ticker', 'predicted_sign')
