"""oracle_server.py — the TRUSTED side of the read-gate process boundary (track A).

The read-gate's real interlock is a PROCESS boundary, not a file fence (docs/read_gate.md,
red-teamed there): a sandboxed PROPOSER holding only the scrubbed corpus + grammar (no
engine, no chains) reaches the engine ONLY through a trusted ORACLE that charges every
score to the lifetime e-LOND stream BEFORE returning a one-bit verdict — "count every look;
don't hide a number you can't hide." `edge_search.score_and_record` is the in-process seam
(PR #72) that already does the score -> lifetime-judge -> record -> scrub chain; this module
is the transport + the SUPERVISED-OPERATOR sandbox around that seam. NOTE: the wall here is a
same-machine `cwd` + a scrubbed env, NOT a kernel filesystem jail — a malicious proposer could
still read the engine or the answer-key ledger by absolute path. The load-bearing control at
this stage is that the SANCTIONED proposer code (`proposer_client`) imports no engine (pinned
by `test_import_is_engine_free`); kernel-enforced absence is the container PR (docs/read_gate.md).

THE TWO SIDES:

  * The ORACLE (this module, `serve`) is TRUSTED — importing the engine here is fine. It
    reads NDJSON requests, validates each against the frozen wire contract (`read_gate_wire`),
    calls the seam, and writes back the seam's already-scrubbed, numberless reply.
  * The PROPOSER (a separate process, spawned by `launch`) is UNTRUSTED. It runs in a
    sandbox dir containing ONLY `menu.json` + `corpus.json` (the grammar + the scrubbed
    scoreboard — never the engine, the chains, or the answer-key ledger), with a scrubbed
    environment, talking to the oracle over stdin/stdout.

`serve` takes injectable `read_line`/`write_line` so the tests can drive the loop with no
pipes and no subprocess; `launch` wires them to a real subprocess. Stdlib only on this
side too (it never needs more than json/subprocess/os/sys/shutil) so the transport stays
auditable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Callable

from edge_search import (
    IDEA_LEDGER_PATH,
    STRUCTURE_CAMPAIGN,
    STRUCTURE_CAPITAL,
    Campaign,
    StructureCandidate,
    build_proposer_corpus,
    enumerate_grammar_templates,
    load_idea_ledger,
    score_and_record,
)
from read_gate_wire import (
    REQUIRED_MODEL_FIELDS,
    WIRE_VERSION,
    assert_numberless,
)

# Environment variables stripped before spawning the untrusted proposer. PYTHONPATH could
# re-expose the engine on the proposer's import path (the whole point of the sandbox is that
# it CAN'T import it); the credential vars are belt-and-suspenders so a compromised proposer
# can't exfiltrate via the network even though it shouldn't have one. NOTE this is a DENY-list
# and so intentionally INCOMPLETE — an unanticipated `*_API_KEY` / `*_TOKEN` slips through. It
# is low-stakes only because the MVP proposer is network-free (a stub author, no client); the
# container PR replaces it with a fresh minimal allow-list env (docs/read_gate.md).
_SCRUBBED_ENV_KEYS: tuple[str, ...] = (
    'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP',
    'GH_TOKEN', 'GITHUB_TOKEN', 'ANTHROPIC_API_KEY', 'ALPHAVANTAGE_API_KEY',
    'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'OPENAI_API_KEY',
    'VIRTUAL_ENV', 'CONDA_PREFIX',
)

# Filenames/patterns whose presence in the sandbox dir means the wall is NOT real — the
# proposer could import the engine, read a chain CSV, or `git show` the answer-key ledger.
# `launch` fails CLOSED on any of these BEFORE spawning anything (defense before mechanism).
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    'edge_search.py', 'vol_premium.py', 'cc_backtest.py', 'real_cc_backtest.py',
    'idea_ledger.jsonl', '.git',
})
_FORBIDDEN_SUFFIXES: tuple[str, ...] = ('.csv', '.csv.gz')


def _error_reply(reason: str) -> dict[str, Any]:
    """A malformed-request reply: stamped with the wire version, typed `error`, carrying a
    human reason and NO engine output. Distinct `type` so the proposer never confuses it
    with a real (typeless) scoreboard reply."""
    return {'wire_version': WIRE_VERSION, 'type': 'error', 'reason': reason}


def _validate_request(req: Any) -> str | None:
    """Return None if `req` is a well-formed request, else a human reason string. Validates
    the wire version + the shape the seam needs (`round_id` a str, `model` a dict carrying
    REQUIRED_MODEL_FIELDS, `proposals` a list of dicts). The seam itself re-checks the model
    fields, but the transport must fail with an `error` REPLY rather than letting an exception
    escape and kill the loop — so it checks here too."""
    if not isinstance(req, dict):
        return f'request must be a JSON object, got {type(req).__name__}'
    if req.get('wire_version') != WIRE_VERSION:
        return f'wire_version must be {WIRE_VERSION}, got {req.get("wire_version")!r}'
    if not isinstance(req.get('round_id'), str):
        return 'round_id must be a string'
    model = req.get('model')
    if not isinstance(model, dict):
        return 'model must be an object'
    missing = [f for f in REQUIRED_MODEL_FIELDS if f not in model]
    if missing:
        return f'model missing required fields {missing}'
    proposals = req.get('proposals')
    if not isinstance(proposals, list):
        return 'proposals must be a list'
    if not all(isinstance(p, dict) for p in proposals):
        return 'every proposal must be an object'
    return None


def serve(
    read_line: Callable[[], str | None],
    write_line: Callable[[str], Any],
    *,
    score_fn: Callable[..., dict[str, Any]] = score_and_record,
    campaign: Campaign = STRUCTURE_CAMPAIGN,
    path: str = IDEA_LEDGER_PATH,
    capital: float = STRUCTURE_CAPITAL,
    scorer: Callable[[StructureCandidate], dict[str, Any]] | None = None,
) -> None:
    """The oracle's NDJSON request->reply loop — one JSON object per line, in and out.

    `read_line` returns the next request line (or a falsy value / None at EOF); `write_line`
    is handed one JSON line per reply (a trailing newline is appended here). Both are
    injectable so the tests drive the loop with in-memory queues and `launch` wires them to a
    subprocess's pipes.

    Every iteration: read a line, parse + validate it against the frozen wire contract, and
    on ANY malformed request write an `error` reply and CONTINUE — the loop must never crash
    on untrusted input (a poison line costs that one request, not the whole session). On a
    good request, call `score_fn` (the seam — `score_and_record` by default), which scores,
    charges the lifetime e-LOND stream, records, and returns the ALREADY-scrubbed numberless
    scoreboard; write that as one JSON line. `score_fn` is injectable so the synthetic test
    layer drives the loop with no engine and no datasets (mirroring TestReadGateOracleSeam).

    The reply is `assert_numberless`-checked here too — defense-in-depth behind the seam's
    own scrub, so a transport bug that ever composed a leaky reply fails loudly on the
    trusted side before a number reaches the wire. The loop ends when `read_line` returns
    falsy (EOF)."""
    while True:
        line = read_line()
        if not line:                       # EOF (None or '') ends the session
            return
        line = line.strip()
        if not line:                       # a blank keep-alive line is not a request
            continue
        try:
            req = json.loads(line)
        except (ValueError, TypeError) as exc:
            write_line(json.dumps(_error_reply(f'malformed JSON: {exc}')) + '\n')
            continue
        reason = _validate_request(req)
        if reason is not None:
            write_line(json.dumps(_error_reply(reason)) + '\n')
            continue
        try:
            reply = score_fn(
                req['proposals'], round_id=req['round_id'], model=req['model'],
                campaign=campaign, path=path, capital=capital, scorer=scorer)
            assert_numberless(reply)       # belt to the seam's suspenders, on the trusted side
        except Exception as exc:           # noqa: BLE001 — a bad request must not crash the loop
            write_line(json.dumps(_error_reply(f'scoring failed: {exc}')) + '\n')
            continue
        write_line(json.dumps(reply) + '\n')


def prepare_sandbox(sandbox_dir: str, *, path: str = IDEA_LEDGER_PATH) -> None:
    """Seed the proposer's sandbox with EXACTLY the two files it is allowed to read — the
    grammar menu and the scrubbed scoreboard — and NOTHING else.

      * `menu.json`   — `enumerate_grammar_templates()` as plain coordinate dicts
                        (name / overlay / params / predicted_sign). The full hypothesis menu
                        the proposer may walk; no engine, no result.
      * `corpus.json` — `build_proposer_corpus(load_idea_ledger(path))`: the scrubbed lifetime
                        ledger (hypothesis coordinates + a one-bit KILLED/INVALID verdict,
                        SURVIVED rows excluded), already numberless by construction.

    The sandbox is created if absent and must hold ONLY these two files: if it already contains
    anything else, this RAISES (a stray file would be readable by the proposer), so the seeded
    dir is guaranteed to be exactly the two safe artifacts (re-seeding is fine — only the seeds
    are allowed to pre-exist). BOTH are `assert_numberless`-checked before they land, so a
    scoreboard/menu regression can never seed a number into the sandbox. (The read-deny that
    keeps the ENGINE out is `launch`'s fail-closed check; this guarantees the proposer-visible
    CONTENT is only menu + corpus.)"""
    os.makedirs(sandbox_dir, exist_ok=True)
    extra = set(os.listdir(sandbox_dir)) - {'menu.json', 'corpus.json'}
    if extra:
        raise ValueError(
            f'prepare_sandbox: {sandbox_dir!r} must hold only the seed files; found '
            f'{sorted(extra)} (seed a fresh/empty dir — a stray file is proposer-readable)')

    menu = [{'name': t.name, 'overlay': t.overlay,
             'params': dict(t.params), 'predicted_sign': t.predicted_sign}
            for t in enumerate_grammar_templates()]
    corpus = build_proposer_corpus(load_idea_ledger(path))
    assert_numberless(menu)                # menu is coordinate-only; pin it, same bar as corpus
    assert_numberless(corpus)              # the seed must be numberless, same bar as a reply

    with open(os.path.join(sandbox_dir, 'menu.json'), 'w', encoding='utf-8') as f:
        json.dump(menu, f, sort_keys=True)
    with open(os.path.join(sandbox_dir, 'corpus.json'), 'w', encoding='utf-8') as f:
        json.dump(corpus, f, sort_keys=True)


def _assert_sandbox_clean(sandbox_dir: str) -> None:
    """FAIL CLOSED: raise unless the sandbox TREE is free of anything that would let the
    proposer walk past the wall — the engine modules, any chain CSV, a `.git` (history holds
    the answer-key ledger under any path), or a `test_*.py` (pinned t-stats are a leak surface
    CLAUDE.md guarantees). Walks RECURSIVELY (a nested `sub/edge_search.py` breaches it too),
    matches names CASE-INSENSITIVELY (a case-insensitive FS reads `SPY...CSV` the same file),
    and rejects ANY symlink (an innocuously-named link can point at the whole engine checkout).

    This closes the IN-SANDBOX surface only — it does NOT close the same-machine residual that
    the repo-root engine is reachable by absolute path (see `launch`'s note). It runs BEFORE
    `prepare_sandbox` writes anything and BEFORE any subprocess is spawned."""
    if not os.path.isdir(sandbox_dir):
        return                             # a not-yet-created dir is trivially clean
    for root, dirs, files in os.walk(sandbox_dir):
        for name in dirs + files:
            low = name.lower()             # case-insensitive FS: EDGE_SEARCH.PY == edge_search.py
            if os.path.islink(os.path.join(root, name)):
                raise ValueError(f'sandbox not clean: symlink {name!r} could point past the wall')
            if low in _FORBIDDEN_NAMES:
                raise ValueError(f'sandbox not clean: {name!r} would breach the read-gate wall')
            if low.startswith('test_') and low.endswith('.py'):
                raise ValueError(f'sandbox not clean: {name!r} (pinned t-stats are a leak surface)')
            if any(low.endswith(suf) for suf in _FORBIDDEN_SUFFIXES):
                raise ValueError(f'sandbox not clean: {name!r} (a chain file breaches the wall)')


def _scrubbed_env() -> dict[str, str]:
    """A copy of the current environment with the engine-/credential-exposing keys dropped,
    so the spawned proposer can neither re-import the engine via PYTHONPATH nor exfiltrate
    via a stray token. Everything else (PATH so python resolves, etc.) is preserved."""
    return {k: v for k, v in os.environ.items() if k not in _SCRUBBED_ENV_KEYS}


def launch(
    proposer_argv: list[str],
    *,
    sandbox_dir: str,
    path: str = IDEA_LEDGER_PATH,
    campaign: Campaign = STRUCTURE_CAMPAIGN,
    capital: float = STRUCTURE_CAPITAL,
    scorer: Callable[[StructureCandidate], dict[str, Any]] | None = None,
) -> int:
    """Spawn the untrusted proposer in its sandbox and run the oracle loop against it.

    Order matters — the wall is built BEFORE the mechanism:

      1. FAIL CLOSED on a dirty sandbox (`_assert_sandbox_clean`): if `sandbox_dir` carries
         the engine, a chain CSV, `.git`, or a `test_*.py`, raise — never spawn into a dir
         from which the proposer could recompute or read the answer key.
      2. Seed the sandbox with exactly `menu.json` + `corpus.json` (`prepare_sandbox`).
      3. Spawn `proposer_argv` with `cwd=sandbox_dir`, a SCRUBBED env (no PYTHONPATH / no
         tokens), and stdin/stdout as pipes — the proposer reads requests from / writes
         replies to the oracle.
      4. Drive `serve` with `read_line`/`write_line` bound to the child's stdout/stdin: the
         proposer writes a request line, the oracle scores+records+scrubs and replies. EOF
         (the proposer closing its stdout) ends the loop.

    Returns the proposer's exit code.

    HONESTY — this is the SUPERVISED-OPERATOR MVP, NOT a kernel seal. `cwd=sandbox_dir` is not
    a filesystem jail: a child can still `open()` the repo-root engine or answer-key ledger by
    absolute path, and Python seeds `sys.path[0]` from the proposer SCRIPT's directory — so a
    `proposer_argv` that runs a script FROM the engine checkout can `import edge_search` inside
    the "sandbox." What `launch` ENFORCES: fail-closed on a dirty sandbox, a scrubbed env (no
    PYTHONPATH / no tokens), and a content-only `menu.json`+`corpus.json` seed. What it RELIES
    ON: the sanctioned `proposer_client` importing no engine (pinned by `test_import_is_engine_free`),
    and the operator pointing `proposer_argv` at an engine-free location. A real seal — kernel-
    enforced absence + a guaranteed engine-free proposer path — is the container PR
    (docs/read_gate.md, "The architecture that works")."""
    _assert_sandbox_clean(sandbox_dir)
    prepare_sandbox(sandbox_dir, path=path)

    proc = subprocess.Popen(
        proposer_argv,
        cwd=sandbox_dir,
        env=_scrubbed_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,                         # line-buffered: a reply flushes per line
    )
    assert proc.stdin is not None and proc.stdout is not None  # PIPE => present (narrows type)
    child_in, child_out = proc.stdin, proc.stdout

    def read_line() -> str | None:
        return child_out.readline() or None   # '' at EOF -> None ends serve

    def write_line(s: str) -> None:
        child_in.write(s)
        child_in.flush()

    try:
        serve(read_line, write_line, campaign=campaign, path=path,
              capital=capital, scorer=scorer)
    finally:
        # The proposer signals "done" by closing its stdout (EOF -> serve returns); close our
        # write end so it sees EOF too if it's still reading, then reap it.
        try:
            child_in.close()
        except (OSError, ValueError):
            pass
        proc.wait()
    return proc.returncode


def main() -> None:
    """CLI: `python oracle_server.py <proposer> [args...]` launches a proposer against the
    real lifetime ledger. With no args, prints usage (the loop needs a proposer to talk to)."""
    if len(sys.argv) < 2:
        print('usage: python oracle_server.py <proposer-cmd> [args...]\n'
              '  spawns the proposer in a sandbox (menu.json + corpus.json only) and runs\n'
              '  the trusted oracle loop against it (docs/read_gate.md).', file=sys.stderr)
        raise SystemExit(2)
    sandbox = os.environ.get('ORACLE_SANDBOX_DIR', '.oracle_sandbox')
    code = launch(sys.argv[1:], sandbox_dir=sandbox)
    raise SystemExit(code)


if __name__ == '__main__':
    main()


# Re-export the cleanliness check under a public name so a caller (or a test) can pre-flight
# a sandbox dir without spawning. The leading-underscore impl stays the single source of
# truth.
assert_sandbox_clean = _assert_sandbox_clean
