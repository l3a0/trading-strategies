"""Pins for the factor LLM-author front-end (factor_proposer.py, H2a of docs/integration_plan.md).

All always-run on synthetic panels, STUB-author (no real LLM — the real Claude clients are H2b). The
deliverables: the Expr<->coordinate serialization round-trips; the grammar gate rejects off-grammar /
sealed / off-search / bad-sign coordinates and dedups; the THREE-LAYER SEAL holds (numberless prompt,
every-look-recorded oracle, coordinate-only numberless reply); the corpus scrub drops result stats AND the
derived family; the env-boundary fails closed; the menu-walker is the default.
"""
from __future__ import annotations

import hashlib
import json

import pytest

import factor.factor_proposer as fp
from factor.factor_engine import GrammarFactorBackend
from factor.factor_grammar import enumerate_exprs
from proposer.read_gate_wire import FACTOR_PROPOSAL_FIELDS, ProposalBatch
from test_factor_backend import _panel
from test_factor_search import _noise_panel

EXPRS = enumerate_exprs()


def _backend(universe: str = 'SYNTH', panel=None) -> GrammarFactorBackend:
    return GrammarFactorBackend(universe, _panel(T=200) if panel is None else panel, checksum='cafe')


def _stub(proposals: list[dict], *, model='stub-served') -> fp.FactorProposer:
    def author(menu, corpus, onboarded) -> ProposalBatch:
        return ProposalBatch(tuple(proposals), model_requested='stub', model_served=model,
                             temperature=0.0, prompt_sha='abc123')
    return author


class TestExprSerialization:
    def test_round_trips_every_enumerated_expr(self) -> None:
        assert all(fp.dict_to_expr(fp.expr_to_dict(e)) == e for e in EXPRS)

    def test_serialization_is_numberless(self) -> None:
        # the coordinate tree carries only op/operand/window — no result key can appear
        from proposer.read_gate_wire import assert_numberless
        for e in EXPRS:
            assert_numberless(fp.expr_to_dict(e), 'expr')

    def test_dict_to_expr_raises_on_malformed(self) -> None:
        from factor.factor_grammar import ExprGrammarError
        bad = [42, {'no_op': 1}, {'op': 5}, {'op': 'field', 'operand': 9},
               {'op': 'ts_mean', 'args': 'notalist'},
               {'op': 'ts_mean', 'args': [{'op': 'field', 'operand': 'ret'}], 'window': '20'}]
        for d in bad:
            with pytest.raises(ExprGrammarError):
                fp.dict_to_expr(d)


class TestMenuWalker:
    def test_proposes_the_full_untried_slice(self, tmp_path) -> None:
        r = fp.run_factor_proposer_round(_backend(), path=str(tmp_path / 'l.jsonl'))
        assert r['proposed'] == len(EXPRS) and r['batch'] is None     # author=None -> menu-walker

    def test_dry_records_nothing_then_record_dedups_on_noise(self, tmp_path) -> None:
        # a no-signal panel -> 0 survivors -> the corpus holds every cell -> the next round proposes 0
        nb, ledger = _backend('NOISE', _noise_panel()), str(tmp_path / 'n.jsonl')
        assert fp.run_factor_proposer_round(nb, path=ledger, record=False)['recorded'] == 0
        first = fp.run_factor_proposer_round(nb, path=ledger, record=True)
        again = fp.run_factor_proposer_round(nb, path=ledger, record=True)
        assert first['recorded'] == len(EXPRS) and again['proposed'] == 0

    def test_lifetime_judge_dedups_re_proposed_rows(self, tmp_path) -> None:
        # the load-bearing dedup: a row already in the prior ledger (a re-proposed survivor) is NOT
        # re-appended to the e-LOND stream — only genuinely fresh rows are judged + returned
        import json
        ledger = str(tmp_path / 'l.jsonl')
        prior = {'key': 'a', 'ticker': 'X', 'predicted_sign': 1, 'p_value': 0.5, 'measurement_invalid': False}
        with open(ledger, 'w') as f:
            f.write(json.dumps(prior) + '\n')
        fresh = {'key': 'b', 'ticker': 'X', 'predicted_sign': 1, 'p_value': 0.4, 'measurement_invalid': False}
        out = fp._judge_factor_lifetime([dict(prior), fresh, dict(fresh)], ledger)   # re-propose a, fresh b x2
        assert len(out) == 1 and out[0]['key'] == 'b'            # a (in prior) + the within-batch b-dup dropped

    def test_survivors_are_re_proposed_but_dedup_at_record(self, tmp_path) -> None:
        # the planted panel survives many cells; the corpus EXCLUDES survivors so they're re-proposed,
        # but _record_factor_cells dedups them -> recorded 0 (harmless, the option-domain behavior)
        fb, ledger = _backend(), str(tmp_path / 'p.jsonl')
        first = fp.run_factor_proposer_round(fb, path=ledger, record=True)
        again = fp.run_factor_proposer_round(fb, path=ledger, record=True)
        assert first['recorded'] == len(EXPRS) and again['proposed'] > 0 and again['recorded'] == 0


class TestGate:
    def test_accepts_valid_rejects_each_failure_mode(self) -> None:
        fb = _backend()
        good, other = fp.expr_to_dict(EXPRS[0]), fp.expr_to_dict(EXPRS[1])
        proposals = [
            {'expr': good, 'universe': 'SYNTH', 'predicted_sign': 1, 'reasoning': 'ok'},
            {'expr': good, 'universe': 'HOLDOUT', 'predicted_sign': 1, 'reasoning': 'sealed'},
            {'expr': good, 'universe': 'OTHER', 'predicted_sign': 1, 'reasoning': 'off-search'},
            {'expr': {'op': 'ts_mean', 'args': [{'op': 'field', 'operand': 'close'}], 'window': 7},
             'universe': 'SYNTH', 'predicted_sign': 1, 'reasoning': 'off-grammar window'},
            {'expr': other, 'universe': 'SYNTH', 'predicted_sign': -1, 'reasoning': 'bad sign'},
        ]
        cands, needs, rejected, batch = fp.llm_propose_factor_candidates(
            _stub(proposals), fb,
            search=frozenset({'SYNTH'}), sealed=frozenset({'HOLDOUT'}), corpus=[], tried_keys=set())
        assert len(cands) == 1                                     # only the first is grammar+sign+universe valid
        reasons = ' '.join(r['reason'] for r in rejected)
        assert 'sealed' in reasons and 'off-search' in reasons and 'off-grammar' in reasons and 'predicted_sign' in reasons

    def test_routes_un_onboarded_search_universe_to_needs_onboard(self) -> None:
        # a universe in `search` but not the loaded panel -> needs_onboard (never auto-runs), not rejected
        fb = _backend()
        proposals = [{'expr': fp.expr_to_dict(EXPRS[0]), 'universe': 'OTHER', 'predicted_sign': 1}]
        cands, needs, rejected, _ = fp.llm_propose_factor_candidates(
            _stub(proposals), fb, search=frozenset({'SYNTH', 'OTHER'}), sealed=frozenset(), corpus=[])
        assert not cands and needs == ['OTHER'] and not rejected

    def test_dedups_against_tried_and_within_batch(self) -> None:
        fb = _backend()
        from factor.factor_grammar import canonical_expr_key
        tried = {fp._factor_proposer_key(canonical_expr_key(EXPRS[0]), 'SYNTH')}
        dup = fp.expr_to_dict(EXPRS[1])
        proposals = [{'expr': fp.expr_to_dict(EXPRS[0]), 'universe': 'SYNTH', 'predicted_sign': 1},  # tried
                     {'expr': dup, 'universe': 'SYNTH', 'predicted_sign': 1},                        # fresh
                     {'expr': dup, 'universe': 'SYNTH', 'predicted_sign': 1}]                        # within-batch dup
        cands, _, _, _ = fp.llm_propose_factor_candidates(
            _stub(proposals), fb, search=frozenset({'SYNTH'}), sealed=frozenset(), corpus=[], tried_keys=tried)
        assert len(cands) == 1                                     # tried dropped, within-batch dup dropped

    def test_caps_at_max_batch(self) -> None:
        fb = _backend()
        proposals = [{'expr': fp.expr_to_dict(e), 'universe': 'SYNTH', 'predicted_sign': 1} for e in EXPRS]
        cands, _, _, _ = fp.llm_propose_factor_candidates(
            _stub(proposals), fb, search=frozenset({'SYNTH'}), sealed=frozenset(), corpus=[], max_batch=5)
        assert len(cands) == 5

    def test_handles_a_malformed_batch_without_crashing(self) -> None:
        # a BROKEN author (proposals not a list, or non-dict items) yields zero candidates, never a crash
        fb = _backend()
        def bad_batch(menu, corpus, onboarded) -> ProposalBatch:
            return ProposalBatch(None, model_requested='m', model_served='s', temperature=0.0, prompt_sha='h')
        cands, _, rejected, _ = fp.llm_propose_factor_candidates(
            bad_batch, fb, search=frozenset({'SYNTH'}), sealed=frozenset(), corpus=[])
        assert cands == []                                        # non-iterable proposals -> no crash, none
        cands2, _, rej2, _ = fp.llm_propose_factor_candidates(
            _stub(['not-a-dict', 42]), fb, search=frozenset({'SYNTH'}), sealed=frozenset(), corpus=[])
        assert cands2 == [] and len(rej2) == 2                    # each non-dict item rejected with a reason


class TestSeal:
    def test_corpus_scrub_drops_result_stats_and_family(self) -> None:
        # a full ledger row -> the scrub keeps only FACTOR_SAFE_FIELDS + verdict; the DERIVED family and
        # every result statistic are gone (family is a measurement for factors, not a coordinate)
        row = {'phase': 'factor', 'key': 'k1', 'expr': fp.expr_to_dict(EXPRS[0]), 'ticker': 'SYNTH',
               'predicted_sign': 1, 'end': '2025', 'family': 'trend', 'mechanism_ok': True,
               't_stat_newey_west': 3.1, 'p_value': 0.01, 'measurement_invalid': False,
               'e_value': 5.0, 'elond_survivor': False, 'data_lineage_hash': 'deadbeef'}
        s = fp.scrub_factor_ledger_row(row)
        assert set(s) == set(fp.FACTOR_SAFE_FIELDS) | {'verdict'}
        assert 'family' not in s and 't_stat_newey_west' not in s and 'p_value' not in s
        assert s['verdict'] == 'KILLED'

    def test_corpus_excludes_survivors_and_is_numberless(self) -> None:
        rows = [{'key': 'a', 'ticker': 'SYNTH', 'predicted_sign': 1, 'expr': {}, 'measurement_invalid': False,
                 'elond_survivor': True, 't_stat_newey_west': 9.9},                       # SURVIVED -> excluded
                {'key': 'b', 'ticker': 'SYNTH', 'predicted_sign': 1, 'expr': {}, 'measurement_invalid': False,
                 'elond_survivor': False, 'p_value': 0.5}]                                # KILLED -> kept, scrubbed
        corpus = fp.build_factor_proposer_corpus(rows)
        assert len(corpus) == 1 and corpus[0]['key'] == 'b'       # survivor excluded; numberless (no raise)

    def test_prompt_runs_assert_numberless_on_the_corpus(self) -> None:
        # the #1 builder-bug defense: a RAW ledger row (banned result keys) where the scrubbed corpus
        # belongs fails loudly BEFORE the model sees anything
        raw = [{'key': 'a', 'ticker': 'SYNTH', 'predicted_sign': 1, 'expr': {}, 't_stat_newey_west': 2.0}]
        with pytest.raises(ValueError, match='numberless'):
            fp.build_factor_proposer_prompt(fp.factor_grammar_menu(), raw, ('SYNTH',))

    def test_clean_prompt_is_numberless_and_asks_for_a_json_array(self) -> None:
        p = fp.build_factor_proposer_prompt(fp.factor_grammar_menu(), [], ('SYNTH',))
        assert 'JSON array' in p and 'predicted_sign' in p


class TestOracle:
    def test_reply_is_numberless_and_exactly_the_wire_keys(self, tmp_path) -> None:
        fb = _backend()
        model = {'model_requested': 'm', 'model_served': 'm-s', 'temperature': 0.0, 'prompt_sha': 'h'}
        reply = fp.score_and_record_factor(
            [{'expr': fp.expr_to_dict(EXPRS[0]), 'universe': 'SYNTH', 'predicted_sign': 1}],
            round_id='r1', model=model, backend=fb, search=frozenset({'SYNTH'}),
            path=str(tmp_path / 'o.jsonl'), provenance_path=str(tmp_path / 'prov.jsonl'))
        assert set(reply) == {'wire_version', 'recorded', 'needs_onboard', 'rejected', 'corpus'}
        assert reply['recorded'] == 1                            # every look recorded (record=True always)

    def test_untrusted_input_cannot_ride_the_reply(self, tmp_path) -> None:
        # a rejected proposal carrying a BANNED result key + reasoning: the oracle re-scrubs rejected to
        # FACTOR_PROPOSAL_FIELDS, so neither rides back nor trips assert_numberless
        fb = _backend()
        model = {'model_requested': 'm', 'model_served': 'm-s', 'temperature': 0.0, 'prompt_sha': 'h'}
        smuggle = {'expr': fp.expr_to_dict(EXPRS[0]), 'universe': 'HOLDOUT', 'predicted_sign': 1,
                   'reasoning': 'sneaky', 't_stat_newey_west': 42.0}
        reply = fp.score_and_record_factor(
            [smuggle], round_id='r1', model=model, backend=fb, search=frozenset({'SYNTH'}),
            sealed=frozenset({'HOLDOUT'}), path=str(tmp_path / 'o.jsonl'),
            provenance_path=str(tmp_path / 'prov.jsonl'))
        echoed = reply['rejected'][0]['proposal']
        assert set(echoed) <= set(FACTOR_PROPOSAL_FIELDS)        # only coordinates echoed
        assert 't_stat_newey_west' not in echoed and 'reasoning' not in echoed

    def test_missing_model_identity_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match='model identity'):
            fp.score_and_record_factor([], round_id='r', model={'model_served': 'x'}, backend=_backend(),
                                       path=str(tmp_path / 'o.jsonl'))


class TestProvenanceAndBoundary:
    def test_provenance_records_reasoning_excluded_from_the_corpus(self, tmp_path) -> None:
        import json
        prov = str(tmp_path / 'prov.jsonl')
        batch = ProposalBatch(({'expr': {}, 'universe': 'SYNTH', 'predicted_sign': 1, 'reasoning': 'why'},),
                              model_requested='m', model_served='m-s', temperature=0.0, prompt_sha='h')
        fp.record_factor_provenance(batch, [('k', 'SYNTH')], round_id='r1', path=prov)
        with open(prov) as f:
            row = json.loads(f.readline())
        assert row['proposals'][0]['reasoning'] == 'why' and row['model_served'] == 'm-s'
        # the SAME reasoning must NOT survive the corpus scrub (FACTOR_SAFE_FIELDS allow-list)
        assert 'reasoning' not in fp.scrub_factor_ledger_row(
            {'key': 'k', 'ticker': 'SYNTH', 'predicted_sign': 1, 'expr': {}, 'reasoning': 'why'})

    def test_boundary_fails_closed_without_an_author(self) -> None:
        assert fp._resolve_factor_llm_author() is None           # no real client wired (H2b)
        with pytest.raises(SystemExit):
            fp._assert_factor_llm_boundary()                     # fail closed -> menu-walker stays default

    def test_boundary_returns_an_activated_author(self) -> None:
        author = _stub([])
        assert fp._assert_factor_llm_boundary(author) is author


class TestFactorClients:
    """H2b: the real Claude transports for factors (stub client/runner — no SDK / CLI / network). The
    transport + seal are the shared proposer_clients base's; only the factor prompt differs."""

    @staticmethod
    def _api_client(reply: str, *, served: str = 'api-served', stop_reason: str = 'end_turn',
                    captured: dict | None = None):
        class Block:
            def __init__(self, t: str) -> None:
                self.type, self.text = 'text', t

        class Resp:
            content, model = [Block(reply)], served

            def __init__(self) -> None:
                self.stop_reason = stop_reason

        class Msgs:
            def create(self, **kw):
                if captured is not None:
                    captured.update(kw)
                return Resp()

        class Client:
            messages = Msgs()

        return Client()

    def test_api_client_emits_a_batch_from_the_numberless_factor_prompt(self) -> None:
        captured: dict = {}
        reply = '[{"expr": {"op":"field","operand":"close"}, "universe":"SYNTH", "predicted_sign":1}]'
        batch = fp.FactorClaudeProposer(client=self._api_client(reply, captured=captured))(
            fp.factor_grammar_menu(), [], ('SYNTH',))
        assert batch.proposals[0]['universe'] == 'SYNTH' and batch.model_served == 'api-served'
        assert batch.temperature == 0.0 and batch.transport == 'api'      # 4.8 sentinel + API transport
        prompt = captured['messages'][0]['content']
        assert 'JSON array' in prompt and 't_stat' not in prompt and 'p_value' not in prompt   # numberless
        assert batch.prompt_sha == hashlib.sha256(prompt.encode('utf-8')).hexdigest()   # reconstructable id

    def test_api_client_raises_on_refusal(self) -> None:
        client = self._api_client('[]', stop_reason='refusal')
        with pytest.raises(RuntimeError, match='refus'):
            fp.FactorClaudeProposer(client=client)(fp.factor_grammar_menu(), [], ('SYNTH',))

    def test_seal_fires_via_the_client_before_the_model_is_called(self) -> None:
        # defense-in-depth: a RAW ledger row (banned key) in the corpus raises inside the prompt build,
        # BEFORE any API call/subprocess — the seal applies through the client transport, not just the prompt
        raw = [{'key': 'a', 'ticker': 'SYNTH', 'predicted_sign': 1, 'expr': {}, 't_stat_newey_west': 2.0}]
        with pytest.raises(ValueError, match='numberless'):
            fp.FactorClaudeProposer(client=self._api_client('[]'))(fp.factor_grammar_menu(), raw, ('SYNTH',))

    def test_claude_code_client_emits_a_batch_with_subscription_transport(self) -> None:
        reply = '[{"expr": {"op":"field","operand":"ret"}, "universe":"SYNTH", "predicted_sign":1}]'
        batch = fp.FactorClaudeCodeProposer(runner=lambda p: {'result': reply, 'model': 'cc-served'})(
            fp.factor_grammar_menu(), [], ('SYNTH',))
        assert batch.transport == 'claude_code' and batch.model_served == 'cc-served'
        assert batch.temperature == 0.0 and batch.proposals[0]['expr']['operand'] == 'ret'   # sentinel too

    def test_claude_code_client_raises_on_an_unparseable_reply(self) -> None:
        # a malformed subscription reply fails the round loudly (no silent zero-proposal pass) — the same
        # _parse_proposal_array guard the API path uses, on the claude_code transport
        with pytest.raises(ValueError, match='no JSON array'):
            fp.FactorClaudeCodeProposer(runner=lambda p: {'result': 'I have no array', 'model': 'cc'})(
                fp.factor_grammar_menu(), [], ('SYNTH',))

    def test_claude_code_hardening_is_inherited(self) -> None:
        # the seal-critical invocation is the shared base's — the factor client must carry it unchanged
        cmd, env = fp.FactorClaudeCodeProposer('claude-opus-4-8')._build_invocation('THE_PROMPT')
        assert cmd[:3] == ['claude', '-p', 'THE_PROMPT'] and '--bare' not in cmd
        assert '--disallowedTools' in cmd and '*' in cmd                  # every tool denied
        assert '--strict-mcp-config' in cmd and '--max-turns' in cmd
        assert 'ANTHROPIC_API_KEY' not in env and 'ANTHROPIC_AUTH_TOKEN' not in env   # forces subscription OAuth

    def test_resolver_is_env_gated_off_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv('EDGE_SEARCH_LLM_MODEL', raising=False)
        assert fp._resolve_factor_llm_author() is None                    # OFF unless opted in
        monkeypatch.setenv('EDGE_SEARCH_LLM_MODEL', 'claude-opus-4-8')
        monkeypatch.delenv('EDGE_SEARCH_LLM_TRANSPORT', raising=False)
        assert isinstance(fp._resolve_factor_llm_author(), fp.FactorClaudeCodeProposer)   # default: subscription
        monkeypatch.setenv('EDGE_SEARCH_LLM_TRANSPORT', 'api')
        assert isinstance(fp._resolve_factor_llm_author(), fp.FactorClaudeProposer)       # api: metered

    def test_client_plugs_into_the_gate(self) -> None:
        # the client is a FactorProposer — it drives llm_propose_factor_candidates end to end
        fb = GrammarFactorBackend('SYNTH', _panel(T=200), checksum='cafe')
        reply = '[{"expr": %s, "universe":"SYNTH", "predicted_sign":1}]' % json.dumps(fp.expr_to_dict(EXPRS[0]))
        author = fp.FactorClaudeProposer(client=self._api_client(reply))
        cands, _, _, batch = fp.llm_propose_factor_candidates(
            author, fb, search=frozenset({'SYNTH'}), sealed=frozenset(), corpus=[])
        assert len(cands) == 1 and batch.model_served == 'api-served'
