"""oracle_server.py — the TRUSTED side of the read-gate process boundary (track A).

The read-gate's real interlock is a PROCESS boundary, not a file fence (docs/read_gate.md,
red-teamed there): a sandboxed PROPOSER holding only the scrubbed corpus + grammar (no
engine, no chains) reaches the engine ONLY through a trusted ORACLE that charges every
score to the lifetime e-LOND stream BEFORE returning a one-bit verdict — "count every look;
don't hide a number you can't hide." `edge_search.score_and_record` is the in-process seam
(PR #72) that already does the score -> lifetime-judge -> record -> scrub chain; this module
is the transport + the SUPERVISED-OPERATOR sandbox around that seam.

THE IMPORT VECTOR IS NOW CLOSED (track C-1). `prepare_sandbox` COPIES the proposer's own code
(`proposer_client.py` + `read_gate_wire.py`) INTO the sandbox, and `launch` spawns it with
`cwd=sandbox`, so Python seeds `sys.path[0]` from the sandbox — which holds only those two
engine-free files. `import edge_search` from inside the sandbox raises ModuleNotFoundError
(pinned by `TestImportVectorClosed`). Before this, the proposer's CODE lived in the repo root,
so `sys.path[0]` was the repo root and `import edge_search` SUCCEEDED inside the "sandbox" — the
recompute vector the #73 adversarial review flagged.

WHAT IS STILL OPEN (the container PR, docs/read_gate.md): the abspath read + the subprocess. A
malicious proposer can still `open()` the repo-root engine or the answer-key ledger by absolute
path, or shell out — `cwd` is not a kernel filesystem jail. Closing the import path narrows the
EASY recompute (a bare `import edge_search`); kernel-enforced absence is the remaining seal.

THE TWO SIDES:

  * The ORACLE (this module, `serve`) is TRUSTED — importing the engine here is fine. It
    reads NDJSON requests, validates each against the frozen wire contract (`read_gate_wire`),
    calls the seam, and writes back the seam's already-scrubbed, numberless reply.
  * The PROPOSER (a separate process, spawned by `launch`) is UNTRUSTED. It runs in a
    sandbox dir containing EXACTLY four files: `menu.json` + `corpus.json` (the data seeds —
    the grammar + the scrubbed scoreboard) plus `proposer_client.py` + `read_gate_wire.py`
    (the proposer's OWN engine-free code, copied in so it can run with no engine on the import
    path). The engine, the chains, and the answer-key ledger are NEVER in the sandbox. With a
    scrubbed environment it talks to the oracle over stdin/stdout.

`serve` takes injectable `read_line`/`write_line` so the tests can drive the loop with no
pipes and no subprocess; `launch` wires them to a real subprocess. Stdlib only on this
side too (it never needs more than json/subprocess/os/sys/shutil) so the transport stays
auditable.
"""
from __future__ import annotations

import json
import os
import shutil
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

# Environment variables PASSED THROUGH to the untrusted proposer. This is an ALLOW-list, not a
# deny-list — SAFE BY DEFAULT: everything not named here is dropped by omission, so an
# unanticipated var (a new `*_API_KEY`, a cloud cred, a proxy/config var, or a `PYTHON*` that
# re-exposes the engine on the import path) CANNOT leak. A deny-list is unsafe by construction —
# it only stops what someone remembered to list — and the transport review flagged the old
# deny-list as known-incomplete; flipping to an allow-list closes that whole class.
#
# The set is deliberately MINIMAL. The MVP proposer is network-free, stdlib-only, and spawned
# via an ABSOLUTE `sys.executable`, so a venv Python resolves its own prefix from that path and
# needs almost nothing (empirically it starts and imports the stdlib under an EMPTY env). What's
# kept, and why each earns its place:
#   * PATH                     — a legitimate stdlib subprocess can resolve system tools.
#   * HOME                     — some stdlib paths (`~` expansion, default config dirs) read it.
#   * TMPDIR / TMP / TEMP      — tempfile honors these; a stripped temp dir breaks file writes.
#   * LANG / LC_ALL / LC_CTYPE — locale/encoding, so text I/O doesn't fall to a surprising default.
#   * TZ                       — stable local-time for any datetime the proposer formats.
# Notably ABSENT (and they STAY absent): PYTHONPATH / PYTHONHOME / PYTHONSTARTUP (any could
# re-expose the engine on the import path — the whole point of the sandbox is that it CAN'T
# import it), every token / API key / cloud credential, and proxy vars (HTTP(S)_PROXY, etc.).
#
# Track C-2 (the container, docs/read_gate.md) replaces process-env control entirely with a
# fresh minimal image env — at which point the container's env subsumes this allow-list.
_ENV_ALLOWLIST: frozenset[str] = frozenset({
    'PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP',
    'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ',
})

# Filenames/patterns whose presence in the sandbox dir means the wall is NOT real — the
# proposer could import the engine, read a chain CSV, or `git show` the answer-key ledger.
# `launch` fails CLOSED on any of these BEFORE spawning anything (defense before mechanism).
_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    'edge_search.py', 'vol_premium.py', 'cc_backtest.py', 'real_cc_backtest.py',
    'idea_ledger.jsonl', '.git',
})
_FORBIDDEN_SUFFIXES: tuple[str, ...] = ('.csv', '.csv.gz')

# The proposer's OWN engine-free code, copied into the sandbox by `prepare_sandbox` so the
# proposer runs PURELY from the sandbox (cwd=sandbox => sys.path[0]=sandbox holds no engine,
# closing the `import edge_search` recompute vector). These are NOT forbidden by
# `_assert_sandbox_clean` — they are the proposer's code, not the engine. Both import only the
# stdlib + `read_gate_wire` (pinned engine-free by proposer_client's test_import_is_engine_free),
# so seeding them adds no engine reach. Copied from this module's OWN directory (the repo root),
# where proposer_client.py + read_gate_wire.py live next to oracle_server.py.
_PROPOSER_CODE_FILES: tuple[str, ...] = ('proposer_client.py', 'read_gate_wire.py')

# The COMPLETE, frozen sandbox layout: exactly these four names and nothing else. The data
# seeds (menu + corpus) plus the proposer's engine-free code. Track C-2 (the container) builds
# against this contract. `prepare_sandbox` enforces it; `_assert_sandbox_clean` independently
# forbids the engine/chains/.git/test_*.py (a disjoint guard, defense-in-depth).
_SANDBOX_SEED_FILES: frozenset[str] = frozenset(
    {'menu.json', 'corpus.json'}) | frozenset(_PROPOSER_CODE_FILES)


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
    """Seed the proposer's sandbox with EXACTLY four files and NOTHING else — the two data
    seeds plus the proposer's own engine-free code.

      * `menu.json`          — `enumerate_grammar_templates()` as plain coordinate dicts
                               (name / overlay / params / predicted_sign). The full hypothesis
                               menu the proposer may walk; no engine, no result.
      * `corpus.json`        — `build_proposer_corpus(load_idea_ledger(path))`: the scrubbed
                               lifetime ledger (hypothesis coordinates + a one-bit KILLED/INVALID
                               verdict, SURVIVED rows excluded), numberless by construction.
      * `proposer_client.py` — the proposer's OWN code, COPIED from this module's directory.
      * `read_gate_wire.py`  — the dependency-free wire contract `proposer_client` imports.

    Copying the proposer's code IN is what CLOSES the import vector: the proposer runs with
    `cwd=sandbox` (see `launch`), so `sys.path[0]` is the sandbox — and the sandbox holds only
    these two engine-free modules. `import edge_search` from inside the sandbox raises
    ModuleNotFoundError. Both code files import only the stdlib + each other (pinned engine-free
    by `proposer_client`'s `test_import_is_engine_free`), so seeding them adds no engine reach.

    The sandbox is created if absent and must hold ONLY these four names: if it already contains
    anything else, this RAISES (a stray file would be readable by the proposer), so the seeded
    dir is guaranteed to be exactly the four safe artifacts (re-seeding is fine — only the seeds
    are allowed to pre-exist). The two JSON seeds are `assert_numberless`-checked before they
    land, so a scoreboard/menu regression can never seed a number into the sandbox. (The read-deny
    that keeps the ENGINE out is `launch`'s fail-closed `_assert_sandbox_clean`; this guarantees
    the proposer-visible CONTENT is exactly menu + corpus + the engine-free proposer code.)"""
    os.makedirs(sandbox_dir, exist_ok=True)
    extra = set(os.listdir(sandbox_dir)) - _SANDBOX_SEED_FILES
    if extra:
        raise ValueError(
            f'prepare_sandbox: {sandbox_dir!r} must hold only the seed files '
            f'{sorted(_SANDBOX_SEED_FILES)}; found {sorted(extra)} '
            f'(seed a fresh/empty dir — a stray file is proposer-readable)')

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

    # Copy the proposer's engine-free code in from THIS module's directory (the repo root,
    # where proposer_client.py + read_gate_wire.py live next to oracle_server.py), so the
    # proposer's import path (sys.path[0]=cwd=sandbox) reaches its own code but NOT the engine.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    for name in _PROPOSER_CODE_FILES:
        shutil.copy(os.path.join(src_dir, name), os.path.join(sandbox_dir, name))


def _assert_sandbox_clean(sandbox_dir: str) -> None:
    """FAIL CLOSED: raise unless the sandbox TREE is free of anything that would let the
    proposer walk past the wall — the engine modules, any chain CSV, a `.git` (history holds
    the answer-key ledger under any path), or a `test_*.py` (pinned t-stats are a leak surface
    CLAUDE.md guarantees). Walks RECURSIVELY (a nested `sub/edge_search.py` breaches it too),
    matches names CASE-INSENSITIVELY (a case-insensitive FS reads `SPY...CSV` the same file),
    and rejects ANY symlink (an innocuously-named link can point at the whole engine checkout).

    `proposer_client.py` and `read_gate_wire.py` are NOT forbidden — they are the proposer's
    OWN engine-free code, which `prepare_sandbox` deliberately COPIES in (closing the import
    vector). This guard runs on the PRE-SEED dir (BEFORE `prepare_sandbox` writes anything and
    BEFORE any subprocess is spawned), so the proposer code isn't present yet anyway, and even
    re-run on a seeded dir it would still pass them.

    This closes the IN-SANDBOX surface only — it does NOT close the same-machine residual that
    the repo-root engine is reachable by absolute path (see `launch`'s note)."""
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
    """The child environment for the spawned proposer, built from an ALLOW-list (safe by
    default): ONLY the curated `_ENV_ALLOWLIST` keys are carried over from the current
    environment; everything else is dropped by omission. So the proposer can neither re-import
    the engine via PYTHONPATH nor exfiltrate via a stray token — and, unlike a deny-list, an
    unanticipated var (a new API key, a cloud cred, a proxy setting) can't slip through because
    it was never on the list. A var that isn't set in the parent is simply absent from the
    child (we copy what's present, never invent values)."""
    return {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}


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
      2. Seed the sandbox with exactly four files — `menu.json` + `corpus.json` (data) plus
         `proposer_client.py` + `read_gate_wire.py` (the proposer's engine-free code)
         (`prepare_sandbox`). Copying the code IN is what closes the import vector (step 3).
      3. Spawn `proposer_argv` with `cwd=sandbox_dir`, an ALLOW-listed env (only the curated
         `_ENV_ALLOWLIST` keys — no PYTHONPATH, no tokens, no unanticipated var), and
         stdin/stdout as pipes — the proposer reads requests from / writes
         replies to the oracle. Because cwd is the sandbox AND the proposer's code lives there,
         `sys.path[0]` resolves to the sandbox (which holds no engine), so `import edge_search`
         raises ModuleNotFoundError — the import vector is CLOSED.
      4. Drive `serve` with `read_line`/`write_line` bound to the child's stdout/stdin: the
         proposer writes a request line, the oracle scores+records+scrubs and replies. EOF
         (the proposer closing its stdout) ends the loop.

    Returns the proposer's exit code.

    HONESTY — this is the SUPERVISED-OPERATOR MVP. The IMPORT vector is now CLOSED: the
    proposer's code is seeded INTO the sandbox and run with `cwd=sandbox`, so `sys.path[0]`
    (the sandbox) holds no engine and `import edge_search` raises ModuleNotFoundError (pinned
    by `TestImportVectorClosed`). This holds for the intended invocation — `python -c <stub>`
    or `python <a-script-in-the-sandbox>`, where `sys.path[0]` is the cwd. It does NOT hold if
    the operator points `proposer_argv` at a script that lives in the engine checkout (then
    `sys.path[0]` is that script's dir, not the sandbox); the read-gate contract is that the
    proposer is `proposer_client` running from the seeded sandbox, not an arbitrary repo script.

    What is STILL OPEN — `cwd=sandbox_dir` is not a kernel filesystem jail: a child can still
    `open()` the repo-root engine or answer-key ledger by absolute path, or shell out. What
    `launch` ENFORCES: fail-closed on a dirty sandbox, an allow-listed env (only `_ENV_ALLOWLIST`,
    so PYTHONPATH / tokens / unanticipated vars are dropped by omission), a four-file seed, and
    the closed import path. What it RELIES ON: the sanctioned
    `proposer_client` importing no engine (pinned by `test_import_is_engine_free`). A real seal
    — kernel-enforced absence of the abspath read + the subprocess — is the container PR
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
              '  spawns the proposer in a sandbox (menu.json + corpus.json + the proposer\'s\n'
              '  engine-free code) and runs the trusted oracle loop against it '
              '(docs/read_gate.md).', file=sys.stderr)
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

# The frozen sandbox-layout contract, public so track C-2 (the container) and tests can assert
# against it: the COMPLETE set of names the sandbox holds after `prepare_sandbox`, and the
# subset that is the proposer's own engine-free code (copied in, not a data seed).
SANDBOX_SEED_FILES = _SANDBOX_SEED_FILES
PROPOSER_CODE_FILES = _PROPOSER_CODE_FILES
