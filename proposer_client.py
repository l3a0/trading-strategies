"""proposer_client.py — the SANDBOXED proposer side of the read-gate (docs/read_gate.md).

This is the untrusted half of the search desk: it picks which `(template, ticker, params)`
cells to try, but it must be **physically unable to score one**. That is the whole point of
the read-gate — "every look is a recorded look" holds only because the proposer has no engine
to score with privately (docs/read_gate.md, "The architecture that works"). So this module
imports ONLY the dependency-free wire contract (`read_gate_wire`) + the stdlib. It does NOT
import `edge_search`, `vol_premium`, `numpy`, `pandas`, or the chains — and `test_proposer_client.py`
asserts that emptiness directly (none of them land in `sys.modules` after `import proposer_client`).
If you ever reach for the engine here, you have re-opened the leak the read-gate exists to close.

WHAT THE PROPOSER SEES (seeded into its sandbox by the trusted oracle, never recomputed):

  * `menu.json`   — the closed grammar's templates (coordinates only: overlay / params /
    predicted_sign). The proposer proposes FROM this menu; an off-menu cell is rejected by the
    oracle, not silently runnable here.
  * `corpus.json` — the scrubbed lifetime corpus (`build_proposer_corpus` output): what has been
    tried + a one-bit KILLED / INVALID verdict. SURVIVED rows are absent by construction (a
    survivor escalates to manual pre-registration; it never feeds back into automated proposal).

WHAT CROSSES THE WIRE (the frozen contract in `read_gate_wire`):

  request (proposer -> oracle): `{wire_version, round_id, model{REQUIRED_MODEL_FIELDS}, proposals[]}`
    — proposals are COORDINATES ONLY (`PROPOSAL_FIELDS`); `build_request` scrubs every proposal to
    those keys so a model can't smuggle a result-shaped field across, and it requires the model
    identity block so the comparison is auditable.
  reply   (oracle -> proposer): `{wire_version, recorded, needs_onboard, rejected, corpus}` — the
    reply's `corpus` is THIS round's scrubbed verdicts (deltas), which `run_proposer_loop` folds
    into the corpus the next round's author reads.

THE MODEL AUTHOR IS STUBBED FOR NOW. `ModelAuthor` is the seam where a real Claude API call will
live in a LATER PR; here it is injected (`run_proposer_loop(..., author=...)`), so this module has
NO network and NO API key. Do not add an API client here — that is the next PR's job.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from read_gate_wire import PROPOSAL_FIELDS, REQUIRED_MODEL_FIELDS, WIRE_VERSION

# (menu, corpus, onboarded) -> (proposals, model). The seam a real Claude API call slots into
# LATER (this PR injects a stub). `proposals` are coordinate dicts the author wants tried; `model`
# is the author's self-reported identity (REQUIRED_MODEL_FIELDS) for the oracle's provenance log.
# Mirrors edge_search.LLMProposer's INPUTS, but typed against plain dicts (the sandbox has no
# StructureTemplate / ProposalBatch — those are engine types) so this stays import-clean.
ModelAuthor = Callable[
    [list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]],
    "tuple[list[dict[str, Any]], dict[str, Any]]"]


def load_menu(cwd: str = '.') -> list[dict[str, Any]]:
    """Read the closed-grammar menu the oracle seeded into the sandbox (`menu.json` in `cwd`).

    The menu is coordinate-only by construction (overlay / params / predicted_sign) — it is the
    grammar the author proposes from. We DON'T validate its shape here: the oracle authored it and
    re-gates every proposal against the real grammar on the trusted side, so the proposer trusting
    its own seed costs nothing. Returns the raw list as seeded."""
    with open(os.path.join(cwd, 'menu.json')) as f:
        return json.load(f)


def load_corpus(cwd: str = '.') -> list[dict[str, Any]]:
    """Read the scrubbed lifetime corpus the oracle seeded into the sandbox (`corpus.json`).

    This is `build_proposer_corpus` output: coordinates + a one-bit verdict, NO result statistic
    (the read-gate's whole guarantee). The proposer reads ONLY this — never the answer-key
    `idea_ledger.jsonl`, which does not exist in the sandbox. Returns the raw list as seeded."""
    with open(os.path.join(cwd, 'corpus.json')) as f:
        return json.load(f)


def build_request(
    round_id: str,
    model: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble + VALIDATE a wire request (proposer -> oracle), stamping `WIRE_VERSION`.

    Two scrubs enforce the contract before anything crosses the wire:

      * each proposal is projected to `PROPOSAL_FIELDS` ONLY — coordinates in, extra keys out. A
        model that attached a result-shaped field (a t-stat it "predicts", a private note) gets it
        dropped HERE, on the proposer side, so the oracle never has to defend against it and the
        numberless property holds end-to-end. A proposal missing a coordinate raises (a malformed
        cell is the author's bug, not something to ship half-formed).
      * `model` must carry every `REQUIRED_MODEL_FIELD` — the audit identity (which snapshot ran,
        at what temperature, hashing which prompt). A missing field raises a clear error rather than
        shipping an unauditable comparison the oracle would reject anyway.

    Returns the request dict; the caller hands it to the transport (`run_proposer_loop` does)."""
    missing_model = [f for f in REQUIRED_MODEL_FIELDS if f not in model]
    if missing_model:
        raise ValueError(
            f'build_request: model identity missing {missing_model} '
            f'(required: {list(REQUIRED_MODEL_FIELDS)})')
    # Copy only the required model fields — an extra model key (a stray "api_key", a note) must not
    # ride the wire either; the audit block is exactly REQUIRED_MODEL_FIELDS.
    model_block = {f: model[f] for f in REQUIRED_MODEL_FIELDS}

    scrubbed: list[dict[str, Any]] = []
    for p in proposals:
        missing_p = [f for f in PROPOSAL_FIELDS if f not in p]
        if missing_p:
            raise ValueError(
                f'build_request: proposal missing {missing_p} '
                f'(required coordinates: {list(PROPOSAL_FIELDS)}); got {sorted(p)}')
        scrubbed.append({f: p[f] for f in PROPOSAL_FIELDS})

    return {
        'wire_version': WIRE_VERSION,
        'round_id': round_id,
        'model': model_block,
        'proposals': scrubbed,
    }


def run_proposer_loop(
    read_line: Callable[[], str],
    write_line: Callable[[str], None],
    author: ModelAuthor,
    *,
    rounds: int = 1,
    onboarded: tuple[str, ...] = (),
    cwd: str = '.',
) -> list[dict[str, Any]]:
    """Drive the proposer over the wire for `rounds` rounds; return the parsed reply per round.

    Each round: hand the author the seeded menu, the CURRENT scrubbed corpus, and the onboarded
    tickers; `build_request` (scrub + validate) its `(proposals, model)`; serialize and `write_line`
    it; then `read_line` the oracle's reply and parse it. The reply's `corpus` is THIS round's
    scrubbed verdict deltas (KILLED / INVALID coordinates) — we EXTEND the running corpus with them
    so the next round's author sees the cells it just spent, and won't re-propose them. SURVIVED
    cells never appear in a reply corpus (the oracle drops them), so the loop never re-feeds a
    survivor into automated proposal — the manual-graduation discipline holds across rounds too.

    `read_line` / `write_line` are injected so tests can drive a fake oracle in-process (no socket,
    no subprocess); production wires them to the real transport (a later PR). The menu is loaded once
    (it's the frozen grammar); the corpus is reloaded from seed at the start, then grown from
    replies — so a multi-round loop needs no second seed write from the oracle."""
    menu = load_menu(cwd)
    corpus = load_corpus(cwd)
    replies: list[dict[str, Any]] = []
    for i in range(rounds):
        proposals, model = author(menu, corpus, onboarded)
        request = build_request(round_id=f'round-{i}', model=model, proposals=proposals)
        write_line(json.dumps(request, sort_keys=True))
        reply = json.loads(read_line())
        replies.append(reply)
        # Fold this round's scrubbed verdicts into the corpus the next author reads. The reply
        # corpus is coordinate + one-bit-verdict rows (never a statistic), so growing the corpus
        # this way keeps the proposer numberless across rounds.
        corpus = corpus + list(reply.get('corpus', []))
    return replies
