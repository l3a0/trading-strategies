"""test_read_gate_wire.py — the EXHAUSTIVE seal battery for read_gate_wire.assert_numberless.

Always-run (stdlib + pytest, no datasets, no engine import for the pure-wire layer). This is
the hardest-tested code in the read-gate: docs/llm_proposer_plan.md decided the LLM prompt
builder runs ORACLE-SIDE (option A), which removes the kernel backstop and makes
`assert_numberless` the SOLE seal on the model's prompt. So the guard must catch a banned
result key in EVERY container shape it could hide in, BANNED_RESULT_FIELDS must be COMPLETE
against every result field the engine produces, and the "read the raw ledger instead of the
scrubbed corpus" mistake must be caught loudly.

Two layers:
  * the pure-wire layer (TestRecursionShapes / TestEveryBannedFieldEveryDepth /
    TestNotANumberDetector / TestErrorMessage) imports ONLY read_gate_wire — the dependency-
    free contract — so it pins the seal exactly as the engine-free proposer sees it;
  * the cross-check layer (TestRawLedgerVsScrubbedCorpus / TestBannedSetCompleteness /
    TestProposerSurfaceIsNumberless) imports edge_search to prove the seal against the LIVE
    engine's result keys and the real scrub.
"""
import itertools

import pytest

from read_gate_wire import (
    BANNED_RESULT_FIELDS,
    PROPOSAL_FIELDS,
    assert_numberless,
)

# One representative banned key reused across the shape tests.
_BANNED = 't_stat_newey_west'


# ---- the recursion shapes: a banned key must be caught wherever a dict can sit ----
class TestRecursionShapes:
    """assert_numberless must descend into EVERY JSON container shape — dicts, lists-of-
    dicts, lists nested in dicts, dicts in tuples in lists, at arbitrary depth — so a
    banned key cannot hide behind nesting. Each fixture buries the SAME banned key one
    layer deeper or in a different container combination."""

    def test_clean_object_passes(self):
        # the canonical scrubbed corpus shape — coordinates + a one-bit verdict, no raise
        assert_numberless({'wire_version': 1, 'recorded': 2, 'needs_onboard': [],
                           'rejected': [{'proposal': {'overlay': 'short_vol', 'ticker': 'AAA',
                                                      'params': {'dte': 30}, 'predicted_sign': 1},
                                         'reason': 'off-grammar'}],
                           'corpus': [{'template': 'short_call_25', 'ticker': 'AAA',
                                       'params': {'dte': 30}, 'predicted_sign': 1,
                                       'verdict': 'KILLED'}]}) is None

    def test_top_level_dict_key(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({_BANNED: 2.1})

    def test_nested_in_dict(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'outer': {'inner': {_BANNED: 2.1}}})

    def test_in_a_list_of_dicts(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'corpus': [{'template': 'x'}, {_BANNED: 2.1}]})

    def test_list_nested_in_dict_nested_in_list(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'a': [{'b': [{_BANNED: 2.1}]}]})

    def test_dict_in_tuple_in_list(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'a': [({_BANNED: 2.1},)]})

    def test_top_level_list(self):
        # the proposer corpus is itself a LIST of rows — the guard must handle a bare list
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless([{'template': 'x'}, {_BANNED: 2.1}])

    def test_top_level_tuple(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless(({'template': 'x'}, {_BANNED: 2.1}))

    def test_banned_key_with_dict_value(self):
        # the banned key fires regardless of what its VALUE is (here a nested dict)
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({_BANNED: {'whatever': 1}})

    def test_banned_key_with_list_value(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({_BANNED: [1, 2, 3]})

    def test_banned_key_with_none_value(self):
        # measurement_invalid / scale_ratio rows can carry None — a None value is still a leak
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'p_value': None})

    def test_deeply_nested_six_layers(self):
        deep = {_BANNED: 9.9}
        for i in range(6):
            deep = {'k{}'.format(i): [deep]} if i % 2 else {'k{}'.format(i): deep}
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless(deep)

    def test_clean_at_every_one_of_those_shapes(self):
        # the mirror of the above: each container shape with NO banned key must pass
        ok = {'safe': 1, 'list': [{'a': 1}, {'b': 2}],
              'nested': {'x': [{'y': [{'z': 3}]}]},
              'tup': [({'w': 4},)]}
        assert assert_numberless(ok) is None


# ---- every banned field, at every depth ----
class TestEveryBannedFieldEveryDepth:
    """Not just one representative key: EVERY member of BANNED_RESULT_FIELDS must be caught
    at the top level, nested in a dict, nested in a list element, and deeply nested. A
    parametrize over the live set so adding a banned field automatically extends coverage."""

    @pytest.mark.parametrize('field', sorted(BANNED_RESULT_FIELDS))
    def test_caught_top_level(self, field):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({field: 1.0})

    @pytest.mark.parametrize('field', sorted(BANNED_RESULT_FIELDS))
    def test_caught_nested_in_dict(self, field):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'reply': {'row': {field: 1.0}}})

    @pytest.mark.parametrize('field', sorted(BANNED_RESULT_FIELDS))
    def test_caught_in_list_element(self, field):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'corpus': [{'template': 'x'}, {field: 1.0}]})

    @pytest.mark.parametrize('field', sorted(BANNED_RESULT_FIELDS))
    def test_caught_deeply_nested(self, field):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'a': [{'b': ({'c': [{field: 1.0}]},)}]})


# ---- the by-design NON-catches: a key-NAME guard, not a number detector ----
class TestNotANumberDetector:
    """The guard is a key-NAME check, deliberately — a statistic hidden as a string VALUE
    under a safe key is the allow-list scrub's job, not this layer's. These document the
    boundary so a future 'tighten it to scan values' change is a conscious decision, not an
    accident. (The scrub upstream is what actually prevents a value-channel leak; this layer
    never sees an un-scrubbed value because build_proposer_corpus runs first.)"""

    def test_banned_name_as_a_string_value_passes(self):
        assert assert_numberless({'reason': 'rejected: t_stat_newey_west off-menu'}) is None

    def test_banned_name_as_a_list_string_element_passes(self):
        assert assert_numberless({'needs_onboard': ['p_value', 'e_value']}) is None

    def test_a_number_under_a_safe_key_passes(self):
        # wire_version / recorded are legitimate integers under safe keys
        assert assert_numberless({'wire_version': 1, 'recorded': 42}) is None


class TestLeafTypeGuard:
    """The SOLE seal must be self-sufficient (oracle-side builder, no kernel backstop): every leaf
    must be a JSON primitive, so a banned field cannot hide in a non-primitive leaf's ATTRIBUTES
    (which the key-name guard never inspects) and the seal does not lean on a downstream json.dumps
    failure. A namedtuple is a tuple — descended as values, its field names lost to JSON — so it is
    not a banned-NAME vector and is not the concern here; a dataclass / custom object IS."""

    def test_dataclass_leaf_with_a_banned_attribute_is_rejected(self):
        import dataclasses

        @dataclasses.dataclass
        class _Sneaky:
            t_stat_newey_west: float = 2.1      # a banned name, but as an ATTR, not a dict key

        with pytest.raises(ValueError, match='non-primitive leaf'):
            assert_numberless({'corpus': [{'extra': _Sneaky()}]})

    def test_arbitrary_object_leaf_is_rejected(self):
        class _Obj:
            pass
        with pytest.raises(ValueError, match='non-primitive leaf'):
            assert_numberless({'x': _Obj()})

    def test_a_set_leaf_is_rejected(self):
        # a set is not JSON-serializable; reject it as a non-primitive leaf rather than ignore it
        with pytest.raises(ValueError, match='non-primitive leaf'):
            assert_numberless({'x': {1, 2, 3}})

    def test_primitive_leaves_and_none_still_pass(self):
        assert assert_numberless(
            {'a': 'str', 'b': 1, 'c': 2.5, 'd': True, 'e': None, 'f': [1, 'x', None]}) is None

    def test_a_primitive_subclass_leaf_is_rejected_exact_type(self):
        # numpy.float64 subclasses float, so an isinstance check would wave it through (the
        # red-team's GAP 1) — the EXACT-type check rejects it, restoring the guard's JSON-only
        # self-sufficiency. A plain float subclass stands in for np.float64 (no numpy import in the
        # dependency-free contract test); native floats of the same value still pass.
        class _NpFloat(float):
            pass
        with pytest.raises(ValueError, match='non-primitive leaf'):
            assert_numberless({'params': {'target_delta': _NpFloat(2.13)}})
        assert assert_numberless({'params': {'target_delta': 2.13}}) is None

    def test_a_non_str_dict_key_is_rejected(self):
        # a banned name on a bytes (or int) key slips `BANNED_RESULT_FIELDS & keys()` — str never
        # equals bytes (the red-team's GAP 2). JSON object keys are strings, so a non-str key is
        # rejected outright, so a banned name cannot hide on one.
        with pytest.raises(ValueError, match='non-string key'):
            assert_numberless({b't_stat_newey_west': 2.1})
        with pytest.raises(ValueError, match='non-string key'):
            assert_numberless({'corpus': [{42: 'x'}]})


class TestErrorMessage:
    """The raise must name the offending field and a path, so a leak in CI is debuggable."""

    def test_message_names_field_and_path(self):
        with pytest.raises(ValueError) as exc:
            assert_numberless({'corpus': [{'p_value': 0.01}]})
        msg = str(exc.value)
        assert 'p_value' in msg and 'corpus' in msg and '[0]' in msg

    def test_reports_all_banned_keys_in_one_dict(self):
        with pytest.raises(ValueError) as exc:
            assert_numberless({'t_stat_newey_west': 2.1, 'p_value': 0.01})
        msg = str(exc.value)
        assert 'p_value' in msg and 't_stat_newey_west' in msg


# ---- THE KEY PIN: raw ledger row REJECTED, scrubbed corpus row PASSES ----
class TestRawLedgerVsScrubbedCorpus:
    """The mistake docs/llm_proposer_plan.md singles out: an oracle-side builder calling
    `load_idea_ledger()` (the RAW answer key, with t-stats) instead of
    `build_proposer_corpus(load_idea_ledger())` (the scrubbed projection). assert_numberless
    is the load-bearing catch for exactly that bug — so the regression is: a realistic RAW
    ledger row is REJECTED, and its scrubbed counterpart PASSES."""

    @staticmethod
    def _raw_ledger_row():
        # shaped like structure_ledger_rows output — carries the answer key.
        import edge_search as es
        campaign_row = {
            'phase': 'structure', 'template': 'short_call_25', 'overlay': 'short_vol',
            'ticker': 'MSFT', 'params': {'target_delta': 0.25, 'dte': 30},
            'predicted_sign': 1, 'n_days': 1234, 't_stat_newey_west': 7.65,
            'nw_lag': 8, 'sharpe': 1.4, 'ann_excess_return_pct': 12.3,
            'sign_ok': True, 'p_value': 0.0123,
            'e_value': 3.2, 'elond_level': 0.05, 'elond_survivor': False,
            'fdr_q': 0.10, 'by_survivor': False, 'clean_survivor': False,
        }
        return es.structure_ledger_rows([campaign_row])[0]

    def test_raw_ledger_row_is_rejected(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless(self._raw_ledger_row())

    def test_a_list_of_raw_rows_is_rejected(self):
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless([self._raw_ledger_row(), self._raw_ledger_row()])

    def test_raw_row_smuggled_under_a_corpus_key_is_rejected(self):
        # the realistic builder bug: dropping the raw rows into a reply-shaped envelope
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless({'wire_version': 1, 'corpus': [self._raw_ledger_row()]})

    def test_scrubbed_corpus_row_passes(self):
        import edge_search as es
        corpus = es.build_proposer_corpus([self._raw_ledger_row()])
        assert len(corpus) == 1
        assert assert_numberless(corpus) is None        # the SCRUBBED projection is clean
        # and it really did drop the answer key
        assert 't_stat_newey_west' not in corpus[0] and 'p_value' not in corpus[0]
        assert corpus[0]['verdict'] == 'KILLED'


# ---- completeness: the banned set covers EVERY engine result field ----
class TestBannedSetCompleteness:
    """COMPLETENESS IS LOAD-BEARING — the guard only fires on names it knows, so a result
    field the engine produces that is NOT banned is a silent leak. This pins
    BANNED_RESULT_FIELDS against the live union of result keys structure_kill_gate /
    run_structure_campaign / structure_ledger_rows can emit, computed from the engine, so a
    new result field added without banning it fails CI here."""

    @staticmethod
    def _result_keys_the_engine_can_emit():
        # The union of every result-bearing key the three producers can put on a row,
        # MINUS the pure hypothesis-coordinate keys (which legitimately survive the scrub).
        coords = {'phase', 'template', 'overlay', 'ticker', 'params', 'predicted_sign', 'end'}
        kill_gate_trade = {'n_days', 't_stat_newey_west', 'nw_lag', 'sharpe',
                           'ann_excess_return_pct', 'sign_ok', 'p_value'}
        kill_gate_no_trade = {'measurement_invalid', 'no_trades', 't_stat_newey_west',
                              'sign_ok', 'p_value'}
        campaign_scale_invalid = {'measurement_invalid', 'scale_ratio',
                                  't_stat_newey_west', 'sign_ok', 'p_value'}
        online_fdr = {'e_value', 'elond_level', 'elond_survivor'}
        by_block = {'fdr_q', 'by_survivor', 'clean_survivor'}
        ledger = {'statistic_kind', 'statistic', 'p_value', 'elond_survivor',
                  'by_survivor', 'measurement_invalid', 'fdr_q', 'data_lineage_hash'}
        produced = (kill_gate_trade | kill_gate_no_trade | campaign_scale_invalid
                    | online_fdr | by_block | ledger)
        return produced - coords

    def test_every_engine_result_key_is_banned(self):
        produced = self._result_keys_the_engine_can_emit()
        missing = produced - BANNED_RESULT_FIELDS
        assert not missing, (
            'result fields produced by the engine but NOT in BANNED_RESULT_FIELDS '
            '(a silent leak channel): {}'.format(sorted(missing)))

    def test_completeness_against_a_live_kill_gate_row(self):
        # build a real structure_kill_gate row via the synthetic scorer path and confirm
        # every non-coordinate key on it is banned — the live cross-check, not a hand list.
        import edge_search as es
        coords = {'phase', 'template', 'overlay', 'ticker', 'params', 'predicted_sign', 'end'}
        rows = es.run_structure_campaign(
            es.Campaign(search=('AAA',)),
            scorer=lambda c: {'phase': 'structure', 'template': c.template,
                              'overlay': c.overlay, 'ticker': c.ticker,
                              'params': c.params_dict(), 'predicted_sign': c.predicted_sign,
                              'n_days': 100, 't_stat_newey_west': 0.5, 'nw_lag': 4,
                              'sharpe': 0.3, 'ann_excess_return_pct': 2.0,
                              'sign_ok': True, 'p_value': 0.3})
        ledger_rows = es.structure_ledger_rows(rows)
        produced = set()
        for r in rows + ledger_rows:
            produced |= (set(r) - coords)
        missing = produced - BANNED_RESULT_FIELDS
        assert not missing, 'live row carries unbanned result keys: {}'.format(sorted(missing))
        # and the raw rows really do trip the guard (sanity that they ARE result-bearing)
        with pytest.raises(ValueError, match='numberless'):
            assert_numberless(rows)


# ---- the proposer-visible surface is numberless (defense-in-depth) ----
class TestProposerSurfaceIsNumberless:
    """Item 2 of the hardening: the review flagged that the propose reply / corpus surface
    a proposer consumes was not numberless-gated. These prove it now is — the corpus built
    from a realistic ledger passes, and the score_and_record oracle reply passes."""

    @staticmethod
    def _raw_ledger_rows():
        import edge_search as es
        rows = []
        for tmpl, elond in (('short_call_25', False), ('winner', True),
                            ('by_only', False), ('broken', False)):
            rows.append({
                'phase': 'structure', 'template': tmpl, 'overlay': 'short_vol',
                'ticker': 'MSFT', 'params': {'target_delta': 0.25, 'dte': 30},
                'predicted_sign': 1, 'n_days': 100, 't_stat_newey_west': 2.0,
                'nw_lag': 4, 'sharpe': 0.5, 'ann_excess_return_pct': 3.0,
                'sign_ok': True, 'p_value': 0.02, 'e_value': 1.0, 'elond_level': 0.05,
                'elond_survivor': elond, 'fdr_q': 0.10, 'by_survivor': tmpl == 'by_only',
                'clean_survivor': False,
            })
        return es.structure_ledger_rows(rows)

    def test_build_proposer_corpus_output_is_numberless(self):
        import edge_search as es
        corpus = es.build_proposer_corpus(self._raw_ledger_rows())
        assert assert_numberless(corpus) is None    # passes (build_proposer_corpus asserts too)
        # the SURVIVED row is excluded; the rest carry no result key
        assert all('t_stat_newey_west' not in r and 'p_value' not in r for r in corpus)

    def test_build_proposer_corpus_asserts_internally(self, monkeypatch):
        # if a future SAFE_FIELDS edit ever admitted a banned-named key, build_proposer_corpus
        # itself raises (the internal assert_numberless) — not just a caller that remembers to.
        import edge_search as es
        original_scrub = es.scrub_ledger_row

        def leaky_scrub(row):
            out = original_scrub(row)
            out['p_value'] = row.get('p_value')      # simulate a regressed scrub leaking a number
            return out

        monkeypatch.setattr(es, 'scrub_ledger_row', leaky_scrub)
        with pytest.raises(ValueError, match='numberless'):
            es.build_proposer_corpus(self._raw_ledger_rows())

    def test_score_and_record_reply_is_numberless(self, monkeypatch, tmp_path):
        import edge_search as es
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        reply = es.score_and_record(
            [{'overlay': 'short_vol', 'ticker': 'AAA',
              'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}],
            round_id='r1',
            model={'model_requested': 'm', 'model_served': 'm-snap',
                   'temperature': 0.0, 'prompt_sha': 'abc'},
            campaign=es.Campaign(search=('AAA',)),
            path=str(tmp_path / 'l.jsonl'),
            provenance_path=str(tmp_path / 'p.jsonl'),
            scorer=lambda c: {'phase': 'structure', 'template': c.template,
                              'overlay': c.overlay, 'ticker': c.ticker,
                              'params': c.params_dict(), 'predicted_sign': c.predicted_sign,
                              't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3})
        assert assert_numberless(reply) is None
        assert set(reply['rejected'][0]['proposal']) <= set(PROPOSAL_FIELDS) if reply['rejected'] else True


# ---- the hard e-LOND budget cap (item 3) ----
class TestProposalsCap:
    """The e-LOND budget per round is bounded — but by the CLOSED GRAMMAR (interlock #1), not by a
    separate cap. The grammar's reach is `grid_universe_size() * len(onboarded)` distinct cells, and
    the grammar gate (llm_propose_candidates / propose_structure_candidates) canonicalizes + dedups
    EVERY author's output to that, so no author path can score past it. `max_proposals_per_round`
    restates that reach as an explicit ceiling that `run_proposer_round` slices to — a NO-OP backstop
    in normal operation (the grammar already bounds the count, as the flooding-author test shows),
    there only to bound the budget loudly if the gate ever regressed. These pin the real bound (the
    grammar reach) and that a legitimate full walk is never truncated."""

    @staticmethod
    def _scorer(cand):
        return {'phase': 'structure', 'template': cand.template, 'overlay': cand.overlay,
                'ticker': cand.ticker, 'params': cand.params_dict(),
                'predicted_sign': cand.predicted_sign,
                't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}

    def test_ceiling_equals_the_grammar_reach(self):
        import edge_search as es
        camp = es.Campaign(search=('AAA', 'BBB', 'CCC'))
        assert es.max_proposals_per_round(camp) == es.grid_universe_size() * 3

    def test_a_flooding_author_is_bounded_by_the_grammar_not_the_cap(self, monkeypatch, tmp_path):
        # The HONEST "the count can't run away": an author proposing the WHOLE menu many times over
        # for a ticker is canonicalized + deduped by the GRAMMAR GATE to exactly the grammar reach
        # (grid_universe_size() for one ticker) — below the ceiling, so the cap slice never bites.
        # The bound is the closed grammar, not the cap (a no-op backstop). (The prior
        # "cap truncates a runaway" test was vacuous: no author path can exceed the reach the cap
        # equals, so the slice could never be observed truncating — it passed with the cap deleted.)
        import edge_search as es
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        camp = es.Campaign(search=('AAA',))
        menu = es.enumerate_grammar_templates()
        flood = [{'overlay': t.overlay, 'ticker': 'AAA', 'params': dict(t.params),
                  'predicted_sign': t.predicted_sign} for t in menu] * 5   # 5x duplicates

        def author(_menu, _corpus, _onboarded):
            return es.ProposalBatch(tuple(flood), model_requested='m', model_served='m-s',
                                    temperature=0.0, prompt_sha='sha')

        res = es.run_proposer_round(camp, path=str(tmp_path / 'l.jsonl'), scorer=self._scorer,
                                    run=False, author=author, max_batch=10_000)
        assert res['proposed'] == es.grid_universe_size()            # deduped to the grammar reach
        assert res['proposed'] <= es.max_proposals_per_round(camp)   # under the ceiling (no-op cap)

    def test_cap_does_not_truncate_a_legitimate_full_menu_walk(self, monkeypatch, tmp_path):
        # the ceiling must never silently drop a valid full-universe menu-walk: a one-ticker
        # round proposes exactly grid_universe_size() cells and the cap leaves it intact.
        import edge_search as es
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        camp = es.Campaign(search=('AAA',))
        res = es.run_proposer_round(camp, path=str(tmp_path / 'l.jsonl'),
                                    scorer=self._scorer, run=False)
        assert res['proposed'] == es.grid_universe_size()     # full menu, untruncated


def test_no_banned_field_collides_with_a_proposal_field():
    # a sanity invariant: the coordinate fields a proposer SENDS must never be named like a
    # result field, or a legitimate proposal would trip the guard.
    assert not (set(PROPOSAL_FIELDS) & BANNED_RESULT_FIELDS)


def test_banned_set_and_proposal_fields_are_disjoint_from_safe_fields():
    # SAFE_FIELDS (what survives the scrub) must share no name with BANNED_RESULT_FIELDS,
    # else an allow-listed key could carry a result. (Cross-module check.)
    import edge_search as es
    assert not (set(es.SAFE_FIELDS) & BANNED_RESULT_FIELDS)


if __name__ == '__main__':
    # a quick exhaustive self-check without pytest: every banned field, four depths.
    shapes = [
        lambda f: {f: 1.0},
        lambda f: {'a': {f: 1.0}},
        lambda f: {'a': [{f: 1.0}]},
        lambda f: {'a': [({'b': [{f: 1.0}]},)]},
    ]
    for field, shape in itertools.product(sorted(BANNED_RESULT_FIELDS), shapes):
        try:
            assert_numberless(shape(field))
        except ValueError:
            continue
        raise SystemExit('SLIPPED THROUGH: {} in {}'.format(field, shape(field)))
    print('all {} banned fields caught at all {} shapes'.format(
        len(BANNED_RESULT_FIELDS), len(shapes)))
