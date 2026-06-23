"""Tests for proposer_client.py — the sandboxed proposer side of the read-gate.

All ALWAYS-RUN (no datasets, no network): the proposer is engine-free by design, so its
whole test surface is pure stdlib + the wire contract.

The load-bearing test is `test_import_is_engine_free`: `import proposer_client` must NOT pull in
`edge_search`, `vol_premium`, `numpy`, or `pandas`. That emptiness is the read-gate's physical
guarantee — a proposer with no engine cannot privately score a cell, so "every look is a recorded
look" holds (docs/read_gate.md). We check it in a FRESH subprocess: this very test process already
imports the engine (pytest collects engine suites alongside), so an in-process `sys.modules` check
would always see them and prove nothing.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from proposer_client import (
    build_request,
    load_corpus,
    load_menu,
    run_proposer_loop,
)
from read_gate_wire import PROPOSAL_FIELDS, REQUIRED_MODEL_FIELDS, WIRE_VERSION

# A minimal, contract-valid model identity block (REQUIRED_MODEL_FIELDS) for the stub author.
_MODEL = {
    'model_requested': 'claude-test',
    'model_served': 'claude-test-snapshot',
    'temperature': 0.0,
    'prompt_sha': 'deadbeef',
}

# A coordinate-only proposal (PROPOSAL_FIELDS) the stub author hands back.
_COORD = {
    'overlay': 'short_vol',
    'ticker': 'MSFT',
    'params': {'entry_delta': 0.25, 'dte_target': 30},
    'predicted_sign': 1,
}


# --- the load-bearing engine-free guarantee ----------------------------------

def test_import_is_engine_free():
    """import proposer_client must not import the engine, numpy, or pandas.

    Run in a clean subprocess so the parent's already-imported engine modules don't mask a
    real import. The subprocess imports ONLY proposer_client, then reports any banned module
    that landed in sys.modules."""
    code = (
        'import sys; import proposer_client; '
        "banned = [m for m in ('edge_search', 'vol_premium', 'numpy', 'pandas') "
        'if m in sys.modules]; '
        'print(",".join(banned))'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True, cwd=__import__('os').path.dirname(__file__) or '.')
    assert proc.returncode == 0, f'subprocess failed: {proc.stderr}'
    leaked = proc.stdout.strip()
    assert leaked == '', f'proposer_client imported engine modules: {leaked}'


# --- seed loaders ------------------------------------------------------------

def _seed(tmp_path, menu, corpus):
    (tmp_path / 'menu.json').write_text(json.dumps(menu))
    (tmp_path / 'corpus.json').write_text(json.dumps(corpus))


def test_load_menu_and_corpus(tmp_path):
    """load_menu / load_corpus read exactly the seeded JSON from cwd."""
    menu = [{'overlay': 'short_vol', 'params': {'entry_delta': 0.25}, 'predicted_sign': 1}]
    corpus = [{'template': 'short_vol_25', 'ticker': 'SPY', 'params': {},
               'predicted_sign': 1, 'verdict': 'KILLED'}]
    _seed(tmp_path, menu, corpus)
    assert load_menu(str(tmp_path)) == menu
    assert load_corpus(str(tmp_path)) == corpus


# --- build_request: scrub + validate -----------------------------------------

def test_build_request_is_a_valid_wire_request():
    """A well-formed request stamps wire_version, carries the full model block, and ships
    coordinate-only proposals."""
    req = build_request('round-0', _MODEL, [_COORD])
    assert req['wire_version'] == WIRE_VERSION
    assert req['round_id'] == 'round-0'
    assert set(req['model']) == set(REQUIRED_MODEL_FIELDS)
    assert len(req['proposals']) == 1
    assert set(req['proposals'][0]) == set(PROPOSAL_FIELDS)


def test_build_request_scrubs_extra_proposal_keys():
    """A proposal carrying extra (possibly result-shaped) keys is projected to PROPOSAL_FIELDS
    only — the smuggle channel is closed on the proposer side."""
    dirty = dict(_COORD, t_stat_newey_west=2.17, secret='leak', note='fish here')
    req = build_request('r', _MODEL, [dirty])
    assert set(req['proposals'][0]) == set(PROPOSAL_FIELDS)
    assert 't_stat_newey_west' not in req['proposals'][0]
    assert 'secret' not in req['proposals'][0]


def test_build_request_scrubs_extra_model_keys():
    """The model block is exactly REQUIRED_MODEL_FIELDS — a stray key (e.g. an api_key) is dropped,
    not shipped across the wire."""
    dirty_model = dict(_MODEL, api_key='sk-should-not-cross', note='hi')
    req = build_request('r', dirty_model, [_COORD])
    assert set(req['model']) == set(REQUIRED_MODEL_FIELDS)
    assert 'api_key' not in req['model']


def test_build_request_raises_on_missing_model_field():
    """A model identity missing a REQUIRED_MODEL_FIELD raises a clear error (unauditable comparison
    must not ship)."""
    incomplete = {k: v for k, v in _MODEL.items() if k != 'prompt_sha'}
    with pytest.raises(ValueError, match='model identity missing'):
        build_request('r', incomplete, [_COORD])


def test_build_request_raises_on_missing_proposal_coordinate():
    """A proposal missing a coordinate (PROPOSAL_FIELDS) raises rather than shipping a malformed
    cell."""
    incomplete = {k: v for k, v in _COORD.items() if k != 'params'}
    with pytest.raises(ValueError, match='proposal missing'):
        build_request('r', _MODEL, [incomplete])


# --- run_proposer_loop: round-trip against a fake oracle ----------------------

def _stub_author(menu, corpus, onboarded):
    """A deterministic stand-in for the LLM author: propose one fixed coordinate cell + a valid
    model block. (The real model API call is a later PR.)"""
    return [dict(_COORD)], dict(_MODEL)


def test_run_proposer_loop_round_trip(tmp_path):
    """A stub author + a fake oracle: the loop builds a VALID wire request, the fake oracle
    captures it and returns a canned one-bit reply, and the loop parses that reply."""
    _seed(tmp_path, menu=[dict(_COORD)], corpus=[])
    captured: list[dict] = []

    canned_reply = {
        'wire_version': WIRE_VERSION,
        'recorded': 1,
        'needs_onboard': [],
        'rejected': [],
        'corpus': [{'template': 'short_vol_25', 'ticker': 'MSFT',
                    'params': _COORD['params'], 'predicted_sign': 1, 'verdict': 'KILLED'}],
    }

    def write_line(line: str) -> None:
        captured.append(json.loads(line))

    def read_line() -> str:
        return json.dumps(canned_reply)

    replies = run_proposer_loop(
        read_line, write_line, _stub_author, rounds=1, onboarded=('MSFT',), cwd=str(tmp_path))

    # The request the loop wrote is a valid wire request.
    assert len(captured) == 1
    req = captured[0]
    assert req['wire_version'] == WIRE_VERSION
    assert set(req['model']) == set(REQUIRED_MODEL_FIELDS)
    assert req['proposals'] and set(req['proposals'][0]) == set(PROPOSAL_FIELDS)
    # No result-shaped key crossed (coords only).
    assert 't_stat_newey_west' not in req['proposals'][0]

    # The loop parsed the reply.
    assert len(replies) == 1
    assert replies[0]['recorded'] == 1
    assert replies[0]['corpus'][0]['verdict'] == 'KILLED'


def test_run_proposer_loop_feeds_reply_corpus_to_next_round(tmp_path):
    """Across rounds, the reply's scrubbed corpus is folded into the corpus the next author sees —
    so a proposer can avoid re-proposing cells it just spent."""
    _seed(tmp_path, menu=[dict(_COORD)], corpus=[])
    seen_corpus_sizes: list[int] = []

    def author(menu, corpus, onboarded):
        seen_corpus_sizes.append(len(corpus))
        return [dict(_COORD)], dict(_MODEL)

    delta_row = {'template': 'short_vol_25', 'ticker': 'MSFT', 'params': {},
                 'predicted_sign': 1, 'verdict': 'KILLED'}
    reply = {'wire_version': WIRE_VERSION, 'recorded': 1, 'needs_onboard': [],
             'rejected': [], 'corpus': [delta_row]}

    run_proposer_loop(
        lambda: json.dumps(reply), lambda _line: None, author,
        rounds=2, cwd=str(tmp_path))

    # Round 0 saw an empty corpus; round 1 saw the one delta row the oracle returned.
    assert seen_corpus_sizes == [0, 1]
