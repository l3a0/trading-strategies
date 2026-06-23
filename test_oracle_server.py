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
  * `prepare_sandbox` writes EXACTLY the four sandbox files (menu.json + corpus.json + the
    proposer's engine-free code) — the JSON seeds valid, the corpus numberless.
  * IMPORT VECTOR (`TestImportVectorClosed`): a child spawned with `cwd=sandbox` + the scrubbed
    env CANNOT `import edge_search` (ModuleNotFoundError), but CAN `import proposer_client` (its
    own code is seeded in). This is the track C-1 regression pin.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import pytest

from edge_search import (
    Campaign,
    load_idea_ledger,
    score_and_record,
)
from read_gate_wire import WIRE_VERSION, assert_numberless
from oracle_server import (
    PROPOSER_CODE_FILES,
    SANDBOX_SEED_FILES,
    assert_sandbox_clean,
    launch,
    prepare_sandbox,
    serve,
)
from oracle_server import _scrubbed_env  # the scrubbed env launch spawns the proposer with

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
    """The sandbox seed: EXACTLY the four sandbox files (menu + corpus + the proposer's
    engine-free code), the JSON seeds valid, the corpus numberless."""

    def test_writes_exactly_the_four_sandbox_files(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        led = str(tmp_path / 'idea_ledger.jsonl')          # missing file -> empty corpus
        prepare_sandbox(str(sandbox), path=led)
        # the frozen layout: menu + corpus (data) + the proposer's own engine-free code
        assert sorted(os.listdir(sandbox)) == sorted(SANDBOX_SEED_FILES)
        assert sorted(os.listdir(sandbox)) == [
            'corpus.json', 'menu.json', 'proposer_client.py', 'read_gate_wire.py']
        # the copied code is the proposer's, not the engine — and it's actually present
        for name in PROPOSER_CODE_FILES:
            assert (sandbox / name).is_file() and (sandbox / name).read_text()

        menu = json.loads((sandbox / 'menu.json').read_text())
        corpus = json.loads((sandbox / 'corpus.json').read_text())
        assert isinstance(menu, list) and menu                          # the full grammar menu
        assert set(menu[0]) == {'name', 'overlay', 'params', 'predicted_sign'}
        assert isinstance(corpus, list)                                 # empty ledger -> []
        assert_numberless(corpus)                                       # the seed carries no number

    def test_reseeding_an_already_seeded_dir_is_fine(self, tmp_path) -> None:
        # the four seed files are allowed to pre-exist (re-seed); a stray file is not.
        sandbox = tmp_path / 'sbx'
        led = str(tmp_path / 'idea_ledger.jsonl')
        prepare_sandbox(str(sandbox), path=led)
        prepare_sandbox(str(sandbox), path=led)            # re-seed: must not raise
        assert sorted(os.listdir(sandbox)) == sorted(SANDBOX_SEED_FILES)
        (sandbox / 'stray.txt').write_text('proposer-readable\n')
        with pytest.raises(ValueError, match='stray.txt'):
            prepare_sandbox(str(sandbox), path=led)

    def test_clean_gate_passes_a_seeded_sandbox(self, tmp_path) -> None:
        # The fail-closed gate forbids the ENGINE, not the proposer's own code: a SEEDED sandbox
        # (the four files, incl. the copied-in proposer_client.py + read_gate_wire.py) still
        # passes assert_sandbox_clean. Pins the launch docstring's claim that the clean gate
        # tolerates the proposer code — matters if track C-2 (the container) re-checks a populated
        # sandbox rather than the pre-seed dir launch checks today.
        sandbox = tmp_path / 'sbx'
        prepare_sandbox(str(sandbox), path=str(tmp_path / 'idea_ledger.jsonl'))
        assert_sandbox_clean(str(sandbox))                 # must not raise on the proposer code

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
        # the four sandbox files into the (formerly empty) sandbox.
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        sandbox = tmp_path / 'sbx'
        code = launch(['python3', '-c', 'pass'], sandbox_dir=str(sandbox),
                      path=str(tmp_path / 'l.jsonl'))
        assert code == 0
        assert sorted(os.listdir(sandbox)) == sorted(SANDBOX_SEED_FILES)


class TestLaunchEndToEnd:
    """The REAL composition the in-process fakes can't reach: `launch` spawns an actual
    `proposer_client`-based child over real pipes, round-trips one request, the oracle
    scores+records, and the one-bit reply reaches the proposer. This is the regression pin for
    the missing-newline DEADLOCK (a request without a trailing newline hangs `readline` forever)
    — a SIGALRM guard fails fast instead of hanging CI if it ever regresses.

    The child runs PURELY from the sandbox: it `import proposer_client` WITHOUT inserting the
    repo path, so the import resolves to the COPY `prepare_sandbox` seeded into the sandbox
    (sys.path[0] = cwd = sandbox for `python -c`). That proves the proposer's own code runs
    engine-free from the sandbox; `TestImportVectorClosed` pins the matching negative (the
    engine is NOT importable there). It proves COMPOSITION (the happy path), NOT a kernel seal:
    in this MVP the engine is still abspath-reachable."""

    def test_real_proposer_client_round_trips(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr('edge_search._is_onboarded', lambda tk: True)
        led = str(tmp_path / 'idea_ledger.jsonl')
        sandbox = str(tmp_path / 'sandbox')
        # A real proposer running PURELY from the sandbox: cwd is the sandbox and the proposer's
        # own code is copied there, so `import proposer_client` resolves with NO repo-path insert
        # (sys.path[0] = cwd). Propose one committed cell via a stub author; write the reply to cwd.
        src = (
            "import sys, json\n"
            "import proposer_client as pc\n"
            "author = lambda menu, corpus, onboarded: (\n"
            "    [{'overlay': 'short_vol', 'ticker': 'AAA',\n"
            "      'params': {'target_delta': 0.25, 'dte': 30}, 'predicted_sign': 1}],\n"
            "    {'model_requested': 'stub', 'model_served': 'stub',\n"
            "     'temperature': 0.0, 'prompt_sha': 'x'})\n"
            "def w(s):\n"
            "    sys.stdout.write(s); sys.stdout.flush()\n"
            "replies = pc.run_proposer_loop(sys.stdin.readline, w, author, rounds=1)\n"
            "open('verdicts.json', 'w').write(json.dumps(replies))\n"
        )

        def _bark(signum, frame):
            raise TimeoutError('read-gate e2e exceeded 20s — likely a transport deadlock')
        old = signal.signal(signal.SIGALRM, _bark)
        signal.alarm(20)
        try:
            code = launch([sys.executable, '-c', src], sandbox_dir=sandbox, path=led,
                          campaign=Campaign(search=('AAA',)), scorer=_scorer)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        assert code == 0
        assert len(load_idea_ledger(led)) == 1                       # the oracle recorded the comparison
        replies = json.loads((tmp_path / 'sandbox' / 'verdicts.json').read_text())
        assert replies[0]['wire_version'] == WIRE_VERSION
        assert [r['verdict'] for r in replies[0]['corpus']] == ['KILLED']  # the one-bit reply reached the child


class TestImportVectorClosed:
    """Track C-1 regression pin: a process spawned EXACTLY as `launch` spawns the proposer —
    `cwd=sandbox` + the scrubbed env — CANNOT `import edge_search` (it raises ModuleNotFoundError,
    so the child exits non-zero), because the sandbox holds no engine and `sys.path[0]` for a
    `python -c` child is the cwd (the sandbox). The proposer's OWN code IS reachable there
    (`import proposer_client` succeeds), because `prepare_sandbox` copied it in. Together these
    prove the import recompute vector the #73 review flagged is closed for the intended invocation."""

    def _spawn_in_sandbox(self, sandbox, code):
        """Run `python -c code` exactly as launch spawns the proposer: cwd=sandbox + scrubbed env.
        Returns the completed process (caller asserts returncode / stderr)."""
        return subprocess.run(
            [sys.executable, '-c', code],
            cwd=str(sandbox), env=_scrubbed_env(),
            capture_output=True, text=True, timeout=30)

    def test_engine_not_importable_from_sandbox(self, tmp_path) -> None:
        sandbox = tmp_path / 'sbx'
        prepare_sandbox(str(sandbox), path=str(tmp_path / 'idea_ledger.jsonl'))
        proc = self._spawn_in_sandbox(sandbox, 'import edge_search')
        assert proc.returncode != 0, (
            f'edge_search WAS importable from the sandbox — the import vector is OPEN.\n'
            f'stdout={proc.stdout!r} stderr={proc.stderr!r}')
        # Assert the SPECIFIC missing-module message, not a loose 'ModuleNotFoundError' +
        # 'edge_search' substring pair: if the fix were reverted (engine present in the sandbox)
        # on a bare interpreter, `import edge_search` raises "No module named 'numpy'" FROM INSIDE
        # edge_search.py — whose traceback path still contains "edge_search" — so the loose pair
        # would pass vacuously. "No module named 'edge_search'" appears only when the engine
        # MODULE itself is absent, which is exactly what the closed vector guarantees.
        assert "No module named 'edge_search'" in proc.stderr, proc.stderr

    def test_other_engine_modules_not_importable_from_sandbox(self, tmp_path) -> None:
        # the same vector for the rest of the engine surface a proposer might recompute through
        sandbox = tmp_path / 'sbx'
        prepare_sandbox(str(sandbox), path=str(tmp_path / 'idea_ledger.jsonl'))
        for mod in ('edge_search', 'vol_premium', 'cc_backtest', 'real_cc_backtest'):
            proc = self._spawn_in_sandbox(sandbox, f'import {mod}')
            # the SPECIFIC missing-module message (see test_engine_not_importable_from_sandbox):
            # "No module named '<mod>'" only appears when <mod> ITSELF is absent, not when a
            # present <mod>'s dependency is missing — so this stays non-vacuous on any interpreter.
            assert proc.returncode != 0 and f"No module named '{mod}'" in proc.stderr, (
                f'{mod} WAS importable from the sandbox.\nstderr={proc.stderr!r}')

    def test_proposer_client_is_importable_from_sandbox(self, tmp_path) -> None:
        # the proposer's OWN engine-free code is reachable (copied in) — and it does NOT drag
        # the engine in transitively (it imports only read_gate_wire + the stdlib). The leak
        # denylist below is fixed; if a new engine module is added, extend BOTH it and
        # test_other_engine_modules_not_importable_from_sandbox's tuple.
        sandbox = tmp_path / 'sbx'
        prepare_sandbox(str(sandbox), path=str(tmp_path / 'idea_ledger.jsonl'))
        code = (
            'import sys, proposer_client, read_gate_wire\n'
            "leaked = [m for m in ('edge_search', 'vol_premium', 'numpy', 'pandas') "
            'if m in sys.modules]\n'
            "print('LEAKED:' + ','.join(leaked))\n"
        )
        proc = self._spawn_in_sandbox(sandbox, code)
        assert proc.returncode == 0, (
            f'proposer_client was NOT importable from the sandbox.\nstderr={proc.stderr!r}')
        assert 'LEAKED:' in proc.stdout and proc.stdout.strip() == 'LEAKED:', (
            f'importing proposer_client dragged the engine in: {proc.stdout!r}')
