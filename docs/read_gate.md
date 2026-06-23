# The read-gate — why hiding the answer key fails, and what counting looks does instead

## Status (built so far)

**The in-process seam (#72) and the transport are BUILT; the container hardening and the real
LLM author remain.** `edge_search.score_and_record` is the oracle seam (score → lifetime-judge →
record → return only the one-bit scoreboard). `oracle_server.py` (`serve` / `launch` /
`prepare_sandbox`, fail-closed via `assert_sandbox_clean`) + `proposer_client.py`
(`run_proposer_loop`, imports only `read_gate_wire`) are the NDJSON transport + the
**supervised-operator** sandbox — pinned by `test_oracle_server.py` / `test_proposer_client.py`,
including a real `launch`↔`proposer_client` end-to-end. This is HONEST about being supervised, not
airtight: a same-machine `cwd` is not a kernel jail (the repo-root engine + answer-key ledger stay
reachable by absolute path), so the load-bearing control at this stage is that the sanctioned
proposer code imports no engine (`test_import_is_engine_free`). Two builds remain: the **container**
(kernel-enforced absence + a guaranteed engine-free proposer path + a single-host egress allow-list)
and the **real model author** (a temperature-0 Claude call behind the boundary). An LLM author must
NOT be activated until the container exists. The residual analysis below is unchanged.

## Why this doc exists

The structure-search desk (see [edge_search.md](edge_search.md)) automates options-structure
hypothesis testing under a lifetime false-discovery-rate control (e-LOND). Today the proposer that
picks which `(template, ticker)` cells to try is a deterministic **menu-walker**
(`run_proposer_round`). The end goal is to replace that enumerator with an **LLM author** — Phase 2.

For the FDR control to stay honest, a proposer must be **blind to the result statistics** of prior
cells: it may see *what was tried* and a one-bit *KILLED / INVALID* verdict (the scrubbed corpus,
`build_proposer_corpus`), but never the t-stats and p-values themselves. The danger is ordinary
**p-hacking**: a proposer that knows which killed cells had high t-stats can propose variants of the
noise that *looked* good, inflating the realised false-discovery rate past what e-LOND controls.

CLAUDE.md long carried a placeholder for closing this — *"the read-gate (a vault dir + scoped
read-deny, or committing only the scrubbed projection) is the unbuilt interlock; an honor-system
convention until then."* This doc records the result of actually designing it: **that read-gate is
not merely unbuilt — it is the wrong interlock.** Hiding the answer-key *file* is defeated by five
independent, verified bypasses, because the answer key is not a file. The honest control is a
different shape entirely, and it is infrastructure rather than configuration. Both halves are below:
the dead-end (so it is never re-attempted), and the architecture that works (the blueprint for the
Phase-2 proposer).

## The property at stake

e-LOND controls the false-discovery rate of a *stream* of hypotheses under arbitrary dependence,
peek-whenever. Its validity does **not** assume the hypotheses are independent or pre-counted — the
`Σγ_t ≤ 1` discount sequence replaces the offline denominator. What it *does* assume is that each
hypothesis entering the stream is a genuine, committed test — not a survivor cherry-picked from a
larger set of private looks the control never saw. The honest-search precondition is therefore:

> **Every look at a cell's result statistic must be a recorded look — one that spends FDR budget.**

A proposer that can privately score a hundred cells, see their t-stats, and submit only the three
that looked best has run a hundred tests while the control only counted three. That is the leak. It
has nothing to do with whether the *number* is written in a file the proposer can open — it has to
do with whether the *act of scoring* is counted.

## Dead-end: the file-hiding read-gate

The intuitive design — and the one CLAUDE.md gestured at — is to fence the files that hold the
statistics: move `idea_ledger.jsonl` into a `vault/` directory, add a `.claude/settings.json`
`permissions.deny` rule blocking Read and result-leaking Bash against it (and against the
result-pinning tests, the docs tables, the campaign's stdout printer), and repoint the proposer at a
committed scrubbed projection it *is* allowed to read.

**This is theater.** It was red-teamed from three independent angles before any of it was built;
every angle returned the same verdict. The answer key is a deterministic, cheap *recomputation* from
code and data the proposer must be able to run — so fencing the files where it happens to be
*stored* changes nothing. Five EXPLOITABLE bypasses, several verified live on this repo:

1. **Re-derive by re-running the engine** — the central failure. `structure_kill_gate` /
   `run_structure_campaign` recompute `t_stat_newey_west` and `p_value` from `short_vol_statistics`
   on the committed chains. None of `edge_search.py`, `vol_premium.py`, the chain CSVs, or `python`
   itself can be in the deny list — they *are* the proposer's own scoring step. Verified:
   `python -c "import edge_search; print(edge_search.run_structure_campaign())"` reproduces the
   entire 42-cell answer key. And `_data_lineage_hash` *guarantees* determinism (two rows share a
   hash iff they would produce the same statistic), so the recomputed number is byte-identical to
   the ledger's. **You cannot deny-list the engine without denying the proposer.**

2. **Git history holds the full plaintext.** Verified: `git show HEAD:idea_ledger.jsonl` returns all
   42 answer-key rows, `p_value` fields and all. The file is committed (deliberately — e-LOND
   auditability needs it) across commits #63/#66/#67. A `git mv` into `vault/` only changes the path
   *going forward*; every prior commit keeps the file at its old top-level path, and
   `git show <oldsha>:idea_ledger.jsonl`, `git log -p`, and `git cat-file` are blind to a
   `vault/**` path-deny. Git is content-addressed: even the post-move blob is reachable by its
   object hash regardless of path.

3. **`python -c` walks through any path-based Read deny** — and it is already allow-listed in this
   environment. The Read-tool deny and `cat`/`grep` Bash patterns do nothing against
   `python -c "print(open('vault/idea_ledger.jsonl').read())"`, nor against `cp`-to-a-fresh-name,
   `base64`, `od`, `tail`, or a symlink. Bash deny is pattern-based with no obfuscation resistance.

4. **The test pins ARE the answer key, in plaintext.** `TestStructureCampaign` /
   `TestNvdaStructureCampaign` pin the exact t-stats (`pytest.approx(2.17, …)`, `−0.96`, `−2.09`,
   …). Worse, **CLAUDE.md mandates pinning every campaign re-pin into a regression test** — so new
   un-denied leak surfaces are *guaranteed by the repo's own discipline*. A hardcoded two-file deny
   list cannot keep up, and a Read-deny on a `.py` file is defeated by *running* it
   (`pytest -v` prints the asserted numbers in the failure diff).

5. **Self-record-and-read.** `run_proposer_round` returns `rows` / `ledger_rows` carrying full
   t-stats — and the proposer is *supposed* to call it. It never needs the ledger file: it runs its
   own round, inspects the returned magnitudes in-process, and only then "decides what to propose."

(Two more, for completeness: the `.claude/settings.json` deny is a file the bound agent can itself
edit, or override with an un-committed `settings.local.json`; and the sealed vault — `TLT` — is a
code constant `_load_ticker_data('TLT')` will happily load, not a filesystem boundary.)

**Root cause, stated plainly:** the answer key is not stored, it is *computed*. It is a pure function
of committed engine code + committed chains, and the proposer's whole job is to run that function.
No file-fence survives contact with `python -c`, `git show`, or the engine itself.

## The reframe: count every look, don't hide the number

The five bypasses share one shape, so the fix is a single principle:

> **Make every score a recorded score. Then statistic visibility is harmless.**

If a cell *cannot* be scored without that score being permanently appended to the lifetime e-LOND
stream, then a proposer that peeks has, by the act of peeking, spent FDR budget on a counted test.
Cherry-picking collapses: there is no "private look" to cherry-pick *from*, because the look is the
record. Whether the proposer can *read* a number it already paid for is then irrelevant.

This is why the honest control is about the **boundary around scoring**, not the **secrecy of the
ledger**.

## The architecture that works: a sandboxed proposer + a recording oracle

Two processes with a real boundary between them — a separate checkout, container, or service, **not**
a `settings.json` deny inside one repo the agent fully controls.

- **The proposer process** holds *only* the scrubbed corpus (`build_proposer_corpus` output:
  coordinates + the one-bit verdict) and the closed grammar (`enumerate_grammar_templates`). It has
  **no engine, no chains, no ledger, no test files, no git history of any of them**. It cannot
  compute a t-stat because it does not possess the function or its inputs. Its output is a list of
  proposed `(template, ticker, params)` cells — coordinates only.

- **The oracle process** (trusted, the proposer cannot reach its filesystem) holds the engine, the
  chains, and the ledger. It receives a proposed cell, runs `structure_kill_gate`, **records the
  result to the lifetime stream via `judge_against_lifetime_stream` + `record_trials` before
  returning anything**, and hands back to the proposer *only the one-bit verdict* — never the
  statistic. Every call is a counted look by construction; there is no score-without-record path
  because the proposer has no engine to score with privately.

- **The boundary** is the load-bearing part. It must be a genuine process / filesystem / network
  isolation: the proposer runs where the engine and data are absent. A path-deny inside a shared
  checkout is not this — see the dead-end above. The oracle is the *only* code with read access to
  the answer key, and it is invoked by a trusted harness, not authored by the proposer.

Under this shape the three things the file-gate failed at all become non-issues: the proposer cannot
re-derive (no engine), cannot read history (no repo), and cannot self-record-and-read (the oracle
returns one bit). And critically, it is **honest about what it enforces** — it does not claim to hide
a number; it makes the number not worth hiding.

## What is safe today (the honest status)

Nothing is exposed *right now*, because the gap is entirely prospective:

- **The proposer is a deterministic menu-walker.** There is no model to peek, p-hack, or shell out;
  `run_proposer_round` enumerates the grammar and skips tried cells. The read-gate is a precondition
  for *activating an LLM author*, not a patch for a live leak.
- **Promotion stays CLOSED.** A survivor (`elond_survivor`) escalates to manual pre-registration and
  is *excluded* from the scrubbed corpus (`build_proposer_corpus` drops `SURVIVED` rows); it never
  feeds back into automated proposal. The kill-gate kills; it never crowns.
- **The corpus scrub is field-level and airtight** for what it covers (`SAFE_FIELDS` allow-list, not
  redaction), pinned by `TestProposerCorpus`. The leak is not in the scrub — it is in the fact that
  the scrub is bypassable by recomputation, which only matters once an *untrusted* author with engine
  access exists.

So the correct status line is: **the deterministic proposer is safe; an LLM proposer is not safe to
activate until the sandbox-proposer + recording-oracle boundary exists; a file-hiding read-gate would
be theater and must not be built or claimed as a control.**

## Residual honesty edges (named, not hidden)

- **Private recomputation is unpreventable inside a shared checkout.** Even a "no score without
  record" change to the *sanctioned* API (`run_proposer_round`) does not stop a proposer calling the
  raw `structure_kill_gate` / `short_vol_statistics`. Only the process boundary closes this; short of
  it, the honesty is honor-system and the proposer must be trusted (i.e. deterministic).
- **The sealed vault is the same shape of problem.** `STRUCTURE_SEALED` (`TLT`) is held out only
  because `run_structure_campaign` does not iterate it — the data file is in the repo and loadable.
  A real seal needs the same boundary (the sealed chains live where the search process cannot reach
  them), not a code constant.
- **Surface drift.** Result statistics live in the ledger, the pinned tests, `docs/edge_search.md`,
  the campaign CLI's stdout, and CI logs. Any "hide these" list is a hand-maintained enumeration the
  repo's own pin-everything discipline guarantees will fall behind. This is a *reason* the
  hide-the-number approach is wrong, not a TODO to chase.

## See also

- [edge_search.md](edge_search.md) — the desk, the e-LOND control, the scrubbed corpus, the
  menu-walker proposer.
- `build_proposer_corpus` / `scrub_ledger_row` / `SAFE_FIELDS` / `run_proposer_round` in
  `edge_search.py` — the (correct, airtight-for-what-it-covers) scrub, and the proposer loop the
  oracle architecture slots into unchanged.
- `edge_search.score_and_record` — the in-process realization of the oracle SEAM (PR1, the
  first build of this architecture): the single entry point that scores → lifetime-judges →
  records BEFORE replying and hands back only the scrubbed one-bit scoreboard. The contract it
  speaks (`WIRE_VERSION`, `BANNED_RESULT_FIELDS`, `assert_numberless`, `REQUIRED_MODEL_FIELDS`,
  `PROPOSAL_FIELDS`) lives in the dependency-free `read_gate_wire.py` so the sandboxed proposer
  shares it without importing the engine. The transport + the sandbox (the wall around the seam)
  are the remaining build; this seam is what they bolt onto.
