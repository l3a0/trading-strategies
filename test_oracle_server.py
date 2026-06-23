"""Tests for oracle_server.py — the TRUSTED side of the read-gate process boundary.

ALWAYS-RUN, fully synthetic: no real engine, no datasets. The oracle loop is driven through
an injected `score_fn` shaped like `TestReadGateOracleSeam._scorer` (the same synthetic
scorer the in-process seam pins against), so these tests exercise the transport + sandbox
without running a single overlay. The injected scorer is `edge_search.score_and_record` with
an injected per-candidate `scorer` (mirroring the seam test) so the score -> lifetime-judge
-> record -> scrub chain is real while the engine is not.

What is pinned:

  * `serve` happy path: a couple of request lines in, one scrubbed numberless reply each out,
    and the comparison rows land on the tmp ledger.
  * `serve` robustness: a bad wire_version, a missing model field, and a non-list `proposals`
    each yield an `error` reply and the loop CONTINUES to the next request.
  * `launch` fail-closes on a dirty sandbox (a copied `edge_search.py` makes it raise before
    spawning anything).
  * `prepare_sandbox` writes EXACTLY menu.json + corpus.json — both valid JSON, the corpus
    numberless.
"""
from __future__ import annotations

import json
import os

import pytest

from edge_search import (
    Campaign,
    load_idea_ledger,
    score_and_record,
)
from read_gate_wire import WIRE_VERSION, assert_numberless
from oracle_server import launch, prepare_sandbox, serve

# A synthetic per-candidate scorer shaped like TestReadGateOracleSeam._scorer: a KILLED
# verdict (t=0.5 is nowhere near the e-LOND bar) carrying the banned result keys, so the
# scrub/numberless guards are genuinely exercised.
def _scorer(cand):
    return {'phase': 'structure', 'template': cand.template, 'ticker': cand.ticker,
            'params': cand.params_dict(), 'predicted_sign': cand.predicted_sign,
            't_stat_newey_west': 0.5, 'sign_ok': True, 'p_value': 0.3}


_MODEL = {'model_requested': 'claude-x', 'model_served': 'claude-x-snap',
          'temperature': 0.0, 'prompt_sha': 'abc'}
_CELL = {'overlay': 'short_vol', 'ticker': 'AAA',
         'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}


def _request(proposals, *, round_id='r1', model=None, wire_version=WIRE_VERSION):
    """A wire request line (no trailing newline — serve strips the line itself)."""
    return json.dumps({'wire_version': wire_version, 'round_id': round_id,
                       'model': _MODEL if model is None else model,
                       'proposals': proposals})


class _Pipe:
    """A queue-backed read_line/write_line pair: `feed` enqueues request lines, `replies`
    collects what serve wrote. `read_line` returns None at the end (EOF -> serve returns)."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.replies: list[str] = []

    def read_line(self):
        return self._lines.pop(0) if self._lines else None

    def write_line(self, s):
        self.replies.append(s)

    def parsed(self):
        return [json.loads(r) for r in self.replies]


class TestServe:
    """The oracle NDJSON loop driven by injected read_line/write_line (no pipes, no
    subprocess) and the real seam (`score_and_record`) over an injected synthetic scorer."""

    def _serve(self, lines, led, monkeypatch):
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        pipe = _Pipe(lines)
        serve(pipe.read_line, pipe.write_line, score_fn=score_and_record,
              campaign=Campaign(search=('AAA',)), path=led, scorer=_scorer)
        return pipe

    def test_happy_path_two_requests(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'idea_ledger.jsonl')
        # two DISTINCT cells (different ticker) so both record a fresh comparison
        c2 = {**_CELL, 'overlay': 'straddle', 'params': {'dte': 30}}
        pipe = self._serve([_request([_CELL]), _request([c2], round_id='r2')], led, monkeypatch)

        replies = pipe.parsed()
        assert len(replies) == 2
        for reply in replies:
            assert reply['wire_version'] == WIRE_VERSION
            assert_numberless(reply)                       # no result statistic crossed the wire
            assert 'rows' not in reply and 'ledger_rows' not in reply
            # the corpus is the one-bit scoreboard — verdicts only, never a t-stat
            assert all(r['verdict'] in ('KILLED', 'INVALID') for r in reply['corpus'])
            assert all('p_value' not in r and 't_stat_newey_west' not in r
                       for r in reply['corpus'])
        assert [r['recorded'] for r in replies] == [1, 1]
        # the comparisons are on the tmp ledger (record-before-reply is structural)
        assert len(load_idea_ledger(led)) == 2

    def test_bad_wire_version_then_loop_continues(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'l.jsonl')
        pipe = self._serve(
            [_request([_CELL], wire_version=999), _request([_CELL])], led, monkeypatch)
        first, second = pipe.parsed()
        assert first['type'] == 'error' and 'wire_version' in first['reason']
        assert second['recorded'] == 1                     # the loop survived and served the next
        assert len(load_idea_ledger(led)) == 1             # only the good request recorded

    def test_missing_model_field_is_error_not_crash(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'l.jsonl')
        bad_model = {k: v for k, v in _MODEL.items() if k != 'prompt_sha'}
        pipe = self._serve(
            [_request([_CELL], model=bad_model), _request([_CELL])], led, monkeypatch)
        first, second = pipe.parsed()
        assert first['type'] == 'error' and 'prompt_sha' in first['reason']
        assert second['recorded'] == 1
        assert len(load_idea_ledger(led)) == 1

    def test_non_list_proposals_is_error_not_crash(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'l.jsonl')
        bad = json.dumps({'wire_version': WIRE_VERSION, 'round_id': 'r',
                          'model': _MODEL, 'proposals': {'not': 'a list'}})
        pipe = self._serve([bad, _request([_CELL])], led, monkeypatch)
        first, second = pipe.parsed()
        assert first['type'] == 'error' and 'proposals' in first['reason']
        assert second['recorded'] == 1
        assert len(load_idea_ledger(led)) == 1

    def test_malformed_json_is_error_not_crash(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'l.jsonl')
        pipe = self._serve(['{not valid json', _request([_CELL])], led, monkeypatch)
        first, second = pipe.parsed()
        assert first['type'] == 'error' and 'JSON' in first['reason']
        assert second['recorded'] == 1

    def test_eof_ends_loop_with_no_reply(self, monkeypatch, tmp_path) -> None:
        led = str(tmp_path / 'l.jsonl')
        pipe = self._serve([], led, monkeypatch)           # empty -> read_line returns None at once
        assert pipe.replies == []
        assert load_idea_ledger(led) == []


class TestPrepareSandbox:
    """The sandbox seed: EXACTLY menu.json + corpus.json, both valid JSON, corpus numberless."""

    def test_writes_exactly_menu_and_corpus(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        led = str(tmp_path / 'idea_ledger.jsonl')          # missing file -> empty corpus
        prepare_sandbox(str(sandbox), path=led)
        assert sorted(os.listdir(sandbox)) == ['corpus.json', 'menu.json']

        menu = json.loads((sandbox / 'menu.json').read_text())
        corpus = json.loads((sandbox / 'corpus.json').read_text())
        assert isinstance(menu, list) and menu                          # the full grammar menu
        assert set(menu[0]) == {'name', 'overlay', 'params', 'predicted_sign'}
        assert isinstance(corpus, list)                                 # empty ledger -> []
        assert_numberless(corpus)                                       # the seed carries no number

    def test_corpus_reflects_the_ledger_and_stays_numberless(self, tmp_path) -> None:
        # seed a real (numberless-by-construction) ledger via the seam, then assert the
        # sandbox corpus is exactly its scrubbed projection.
        led = str(tmp_path / 'idea_ledger.jsonl')
        sandbox = tmp_path / 'sbx'

        # monkeypatch-free: score through the seam with a stubbed onboarded check
        import edge_search
        orig = edge_search._is_onboarded
        edge_search._is_onboarded = lambda tk: True
        try:
            score_and_record([_CELL], round_id='r', model=_MODEL,
                             campaign=Campaign(search=('AAA',)), path=led,
                             provenance_path=str(tmp_path / 'prov.jsonl'), scorer=_scorer)
        finally:
            edge_search._is_onboarded = orig

        prepare_sandbox(str(sandbox), path=led)
        corpus = json.loads((sandbox / 'corpus.json').read_text())
        assert [r['verdict'] for r in corpus] == ['KILLED']            # the recorded dud shows up
        assert_numberless(corpus)


class TestLaunchFailClosed:
    """`launch` must fail CLOSED — refuse to spawn into a sandbox from which the proposer could
    breach the wall (import the engine, read a chain, or git-show the answer key)."""

    def _stub_proposer(self):
        # an inert proposer that closes stdout immediately (EOF) so a clean launch returns 0
        return ['python3', '-c', 'pass']

    def test_copied_engine_makes_launch_raise(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        sandbox.mkdir()
        (sandbox / 'edge_search.py').write_text('# a copy of the engine — the wall is breached\n')
        with pytest.raises(ValueError, match='edge_search.py'):
            launch(self._stub_proposer(), sandbox_dir=str(sandbox),
                   path=str(tmp_path / 'l.jsonl'))

    def test_chain_csv_makes_launch_raise(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        sandbox.mkdir()
        (sandbox / 'spy_option_dailies.csv').write_text('date,strike\n')
        with pytest.raises(ValueError, match='breaches the wall'):
            launch(self._stub_proposer(), sandbox_dir=str(sandbox),
                   path=str(tmp_path / 'l.jsonl'))

    def test_git_dir_makes_launch_raise(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        sandbox.mkdir()
        (sandbox / '.git').mkdir()
        with pytest.raises(ValueError, match='read-gate wall'):
            launch(self._stub_proposer(), sandbox_dir=str(sandbox),
                   path=str(tmp_path / 'l.jsonl'))

    def test_test_file_makes_launch_raise(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        sandbox.mkdir()
        (sandbox / 'test_edge_search.py').write_text('# pinned t-stats are a leak surface\n')
        with pytest.raises(ValueError, match='leak surface'):
            launch(self._stub_proposer(), sandbox_dir=str(sandbox),
                   path=str(tmp_path / 'l.jsonl'))

    def test_clean_sandbox_does_not_raise_at_the_gate(self, tmp_path, monkeypatch) -> None:
        # a clean (empty) sandbox passes the fail-closed gate; with an inert proposer the
        # whole launch returns its exit code (0). This also confirms prepare_sandbox seeded
        # the two files into the (formerly empty) sandbox.
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        sandbox = tmp_path / 'sbx'
        code = launch(['python3', '-c', 'pass'], sandbox_dir=str(sandbox),
                      path=str(tmp_path / 'l.jsonl'))
        assert code == 0
        assert sorted(os.listdir(sandbox)) == ['corpus.json', 'menu.json']
