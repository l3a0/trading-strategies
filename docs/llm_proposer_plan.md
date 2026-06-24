# The LLM proposer author — design (item 4, WIRED in Phase B)

## What this is

The read-gate apparatus (see [read_gate.md](read_gate.md)) is feature-complete. The last piece —
**item 4: replacing the deterministic menu-walker proposer with a real LLM author** — is now WIRED
(Phase B, `ClaudeProposer`): a Claude call (Opus 4.8: adaptive thinking + `effort`, no temperature)
that reads the scrubbed scoreboard + the grammar and proposes coordinate-only `(overlay, ticker,
params, predicted_sign)` cells.

This doc records the design and its wiring. The client (`ClaudeProposer`) is built and tested, but
**OFF by default** — it activates only when `EDGE_SEARCH_LLM_MODEL` is set (the owner's deliberate
per-run opt-in). The standing rule holds even when activated: a survivor never auto-promotes
(promotion is CLOSED, escalating to manual pre-registration) and survivors stay exploratory until
Phase C is real. The doc keeps the two consequential decisions — *where the model call lives* and
*how the prompt is constructed* — on the record.

## The simplification: the container is off the LLM path

The container/transport stack — a trusted NDJSON server, an engine-free proposer client, a sealed
container image, and their launch/seal helpers (all now removed; recoverable from git history) —
was built to cage an **untrusted *code* proposer**: a process running in the sandbox that could
`import edge_search` and recompute a statistic privately. The coordinate-emitting LLM is **not** that. It plugs into
`run_proposer_round` as an in-process `LLMProposer` (`author=`), the same slot the deterministic
menu-walker fills — it has no engine, no filesystem, no code execution; it is an API call that
returns text. **So the model does not run in the container, and the transport is not on its path.**

What seals this LLM is *information*, not isolation (next section). That has three consequences:

- The pieces that carry over to the LLM are the **in-process, oracle-side** ones: the recording
  seam (`score_and_record`), the numberless guard (`assert_numberless` / `BANNED_RESULT_FIELDS`),
  the scrubbed corpus (`build_proposer_corpus`), the closed grammar, and the lifetime e-LOND ledger.
- The container/transport (a trusted NDJSON server + an engine-free proposer client + the sealed
  image) was **built, then removed** — it cages a *code*-proposer the decided LLM never is, so it
  never sat on the live model's path. It stays recoverable from git history as the blueprint should
  a code-running proposer ever be wanted.
- So the "container must-dos" (digest pin, seccomp, a round timeout, `--user`, and making the `read-gate-container` CI job a required check) are **not** Phase-B
  gates. They harden a path the decided LLM never takes. They were mis-listed as gates earlier
  (a holdover from a rejected design where the proposer made the API call from inside a container
  with a single-host egress allow-list); that design was routed around, and the container with it.

## Where the Claude call lives: oracle-side, in-process

**The API call is issued by the trusted oracle process** — the one that already holds the engine,
chains, ledger, and network — as an in-process `LLMProposer` plugged into `run_proposer_round`.
There is no separate proposer process: the model is an API call that returns coordinates, so
`ANTHROPIC_API_KEY` and all egress simply live where the rest of the engine does.

This was not always the plan. Three independently-designed architectures still converge on
oracle-side, but an earlier one ran the proposer as a *sandboxed* process and had to answer "where
does the network call happen?" — the sandbox was `--network none` with no key, so the call could
not originate inside it. That tension is what the removal dissolves: with no sandbox, the call is
just an ordinary oracle-side API call, and the seal is the information boundary (next section), not
a sandbox's network cut.

## The seal changes shape — say it loudly

The oracle-side, in-process design **trades the kernel isolation a container would give for a
correctness argument**, and this must be stated plainly because it is easy to assume a kernel jail
protects the LLM. None does — for the LLM, the seal is three things:

- the model's prompt contains **no result statistics** — only the grammar menu and the scrubbed
  corpus (coordinates + one-bit verdicts), `assert_numberless`-checked, SURVIVED rows excluded;
- the model's output **that gates the engine is coordinate-only** (`PROPOSAL_FIELDS`), gated by
  `StructureCandidate` — it may also emit an owner-facing `reasoning` string, but that is display-only
  (printed by `_format_llm_round`; excluded from the gate, ledger, corpus, provenance, and oracle
  reply), so it reaches no sink and is insight-not-evidence (pinned by `TestProposerReasoning`);
- every score is **recorded** (`score_and_record`), so every look spends FDR budget.

A model that only ever sees a numberless prompt and only emits a menu choice **cannot recompute a
score or chase a seen winner** — so the correctness argument is sound *for coordinate-only
output*. But it is a different kind of guarantee than the container provides:

| Guarantee | Holds even if… |
| --- | --- |
| **Kernel isolation** (a sandbox/container) | …there is a bug — the OS backstops it. |
| **Correctness argument** (the LLM oracle-side) | …only if the argument is airtight — no OS backstop. |

A container was built to cage an untrusted-*code* proposer that could read the ledger or run the
engine. An LLM emitting coordinates from a numberless prompt is sealed by *information*, not
isolation — so for the LLM path the container was **removed**. That is an acceptable trade for
coordinate-only output, but it relocates the load-bearing guarantee onto the `assert_numberless`
checks and the coordinate-only gate, which must therefore be the hardest-tested code in the system.

## The prompt builder is the leak surface — three options

The correctness argument lives or dies in **whatever code constructs the model's prompt**. That
code is now built — `build_proposer_prompt` (Phase A; no model wired) — a pure function of the
three inputs the `LLMProposer` receives: `enumerate_grammar_templates()` (the menu),
`build_proposer_corpus(load_idea_ledger())` (the scrubbed corpus; the `SAFE_FIELDS` allow-list
drops every statistic), and the onboarded search tickers. It was decided (A) oracle-side; the risk
is entirely in **where the builder runs and what it reads**:

- **(A) Oracle-side builder.** Simplest, and what the three designs assumed. But the builder runs
  in the trusted process that *also* holds the raw `idea_ledger.jsonl` (with the t-stats) and the
  engine. A bug — calling `load_idea_ledger()` (raw) instead of `build_proposer_corpus(...)`
  (scrubbed), or appending a "helpful" statistic — leaks a number into the prompt, and the *only*
  catch is an `assert_numberless` on the assembled prompt. No kernel backstop.
- **(B) Builder inside the container that also calls the API.** Ruled out (and moot now the
  container is removed): a sandboxed proposer would be `--network none` with no key, so it could
  not make the call.
- **(C) Build in the sealed container, relay to the oracle to call.** The proposer builds the
  prompt inside the container, which holds *only* the scrubbed seeds (`menu.json` + `corpus.json`)
  — the raw ledger is absent, so a builder bug **physically cannot leak a statistic**. It sends
  the finished prompt to the oracle, which `assert_numberless`-checks it on the wire and makes the
  API call. This recovers kernel isolation for the most leak-prone surface. The cost is a protocol
  change: the proposer sends a *prompt* for the oracle to relay, rather than the current "proposer
  emits coordinates, the author call happens inside it."

**Decision (owner, 2026-06-23): option (A), the oracle-side builder.** Prompt construction stays on
the trusted side, where it has the full context to assemble the model's prompt, rather than being
constrained to the sealed box. (Option (C) was the tighter design on the seal axis; (A) was chosen
for the construction flexibility, with the tradeoff accepted explicitly.)

Because (A) has **no kernel backstop**, the numberless guard is the load-bearing seal — and
building it (`build_proposer_prompt`) sharpened *where* the guard must run:

- it runs on the structured **corpus input** (`assert_numberless(list(scrubbed_corpus))`), the
  answer-key-sourced input, BEFORE rendering — **not** on the assembled prompt string. A key-name
  guard is a no-op on a string (a leaf), and a banned-NAME substring scan over the bytes would
  false-positive (`statistic` is a substring of `t-statistic` in the instructions themselves). The
  effective catch for the #1 bug — the raw `load_idea_ledger()` rows passed in place of the scrubbed
  `build_proposer_corpus()` — is that those rows carry banned KEYS and fail the structural assert.
- The builder reads **only** the scrubbed corpus + `enumerate_grammar_templates()` + the grammar's
  declared typing (`STRUCTURE_GRAMMAR`) — **never** the raw `load_idea_ledger()` or any engine
  output, pinned structurally by a `co_names` test (the answer key is not even in scope to format).
- It is among the **most-tested code in the system** (`TestProposerPrompt`): raw-rows-fire, a single
  banned key slipping the scrub, the no-result-magnitude-reaches-the-prompt end-to-end, and the
  answer-key-unreachable-by-construction proof.
- A leak is fail-closed: a non-numberless corpus **raises before any prompt is assembled**, never
  silently ships.

Option (C) remains recorded as the tighter fallback should that single guard ever prove
insufficient — but note it presupposes putting a sandboxed prompt-builder *back* in the path (see
*The simplification*): it trades the oracle-side in-process model for the container/transport. The
decided design (A) does not use the container at all.

## The training-leak defense is not ready

A separate, unsolved problem governs whether the LLM's *survivors* can be trusted. The
sealed-ticker vault (`STRUCTURE_SEALED` = TLT) is **not** out-of-sample for a model that may have
trained on this public repo, including the committed `idea_ledger.jsonl` t-stats and the
pinned-test numbers — a "novel" survivor confirming on a held-out ticker can be training-data
recall, not mechanism. The right defense is a **post-training-cutoff time-axis holdout**: score a
survivor only on data timestamped after the model's training cutoff.

This is inoperative today, for reasons the adversarial pass agreed on:

- the chains end \~2026-06, so the post-cutoff span is too thin for the degrees-of-freedom floor;
- the model's training cutoff is an unverifiable vendor claim;
- it does not stop **test-time retrieval** of the public repo at inference.

So the LLM can act as an **exploratory search proposer** (kill-or-justify, sample-spending) under
the correctness seal — but its survivors stay exploratory and are **never promoted to a
confirmatory finding** until the holdout is real. This is the same line the `prereg_*` docs
protect: a passing scout earns a registration, not a headline.

## A cautionary foil: how a published system does it

A recent autonomous-factor system — Huang & Fan, *Beyond Prompting: An Autonomous Framework for
Systematic Factor Investing via Agentic AI* (arXiv:2603.14288) — is the honor-system version of
this design, and a useful illustration of why the interlocks above matter. It reports a 3.11 Sharpe
/ 59.5% returns from an LLM that proposes factor "recipes" run by a deterministic layer. Its
discipline is real but partial:

- a fixed **`|t| > 3.0`** hurdle (Harvey–Liu–Zhu 2016) plus a Deflated-Sharpe *"heuristic"* — but
  **the LLM is shown the in-sample IC / Sharpe / t-stats each round** and conditions its next
  proposals on them, so the search is adaptive and a *fixed* bar cannot control the inflated
  false-discovery rate;
- an **economic rationale generated jointly with the formula** — a plausible story, **not** a
  mechanism checked against the data (the post-hoc-label failure mode);
- a **temporal OOS freeze** (discovery pre-Dec-2020, blind 2021–2024) — which is *not* out-of-sample
  for an LLM whose training corpus spans that period (no model-cutoff defense).

Crucially, it uses **no sandbox and no container** — functional separation only. That is the point:
a real, published system in this space runs without isolation; what it lacks is the **information
boundary**, the **online FDR control**, the **mechanism check**, and the **leakage defense** —
precisely the four *this* design enforces (numberless prompt, e-LOND, engine-verified grammar
signature, time-axis holdout). Read that way, its 3.11 Sharpe is the *output to distrust*, and the
foil that justifies the interlocks — none of which is a container.

## The phased plan

- **Phase A — oracle-side gates (buildable now, activates nothing).** What actually gates the live
  model, all on the trusted side: the **numberless seal** hardened into the sole prompt guard
  (`assert_numberless` + the completed ban-set + the leaf-type guard — done in #82); the
  **prompt builder** (`build_proposer_prompt` — the numberless prompt assembled from the menu +
  scrubbed corpus + onboarded tickers, sealed by `assert_numberless` on the corpus input, pinned by
  `TestProposerPrompt`); the **activation gate simplified** to its no-model backstop (the
  sandbox-specific engine-absent precondition was removed with the container — see *What must not
  happen*); and the oracle stamping the **authoritative `model_served`** rather than trusting the
  model's self-report. The *container*
  must-dos (digest pin, seccomp, a round timeout, `--user`, and making the `read-gate-container` CI job a required check) are moot — the container was removed; they
  would only matter if a code-proposer were ever rebuilt (see *The simplification* above).
- **Phase B — the model wiring (DONE; owner's go 2026-06-24).** The `anthropic` client
  (`ClaudeProposer`) takes `build_proposer_prompt`'s output, hashes it (`prompt_sha`), calls Claude
  (Opus 4.8 default: adaptive thinking + `effort`, **no `temperature`** — the 4.8 family rejects it,
  so the recorded `temperature` is a documented `0.0` sentinel; the reconstructable identity is
  `model_served` + `prompt_sha`), parses the coordinate-only JSON (`_parse_proposal_array` —
  tolerant of a code fence, RAISES on an unrecoverable reply rather than silently proposing
  nothing), and captures the SERVED `model_served` (the API's `response.model`, not the alias) into
  `record_provenance`. It is wired into `_resolve_llm_author` but **OFF by default**: it activates
  only when `EDGE_SEARCH_LLM_MODEL` is set (the owner's per-run opt-in; `ANTHROPIC_API_KEY` in the
  environment + `anthropic` installed — an OPTIONAL dependency, lazily imported, not in
  requirements.txt). The prompt builder and its seal were already built and tested (Phase A); Phase
  B wraps them in the API client. Pinned by `TestClaudeProposer` (a stub client — no SDK, no key, no
  network). **The DEFAULT transport — the Claude.ai SUBSCRIPTION via Claude Code**
  (`ClaudeCodeProposer`; `EDGE_SEARCH_LLM_TRANSPORT=api` selects the metered `ClaudeProposer`) — drives
  `claude -p` under a Max/Pro plan (no key, no `anthropic` dependency). The seal is a LARGER surface
  for an agent than for a stateless call, so the invocation is hardened: `--disallowedTools "*"`
  removes every tool from the model's context (deny-first), `--strict-mcp-config` loads no MCP, the
  subprocess runs from a NEUTRAL temp cwd (so the repo's CLAUDE.md — which carries pinned result
  numbers — is never in scope), and `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` are scrubbed from the
  child env (subscription OAuth, no metered call; NOT `--bare`, which would force a key). The API
  `ClaudeProposer` remains the seal GOLD-STANDARD (stateless, zero context/tools); the subscription
  path still loads `~/.claude` global config (a global hook could run) and is for exploratory runs
  with that tradeoff understood. It does not honor `effort='max'` (Claude Code applies its own
  defaults). `transport` is recorded to the provenance log (lineage-ADJACENT, never re-keys/re-spends
  the ledger); pinned by `TestClaudeCodeProposer`.
- **Phase C — confirmatory trust (research-blocked).** The time-axis holdout. Until it is
  operative, LLM survivors stay exploratory.

## What must not happen

- No LLM author runs unless the owner activates it. `_resolve_llm_author()` returns `None` UNLESS
  `EDGE_SEARCH_LLM_MODEL` is set, so by default `propose --llm` still fails closed on **the (b)-only
  no-model check**: `_assert_llm_boundary` refuses unless a model author is activated. The
  sandbox-specific precondition (a) — *the engine is not importable from cwd* — was **removed with
  the container** (it was true only inside the sandbox, so against an in-process oracle-side LLM it
  would have **refused the very thing the decision wants to run**). When activated, the fail-closed
  seal is the oracle-side correctness argument: the prompt is asserted numberless, the output is
  coordinate-only and grammar-gated, and every score is recorded — not engine absence.
- The model never sees a result statistic, and never emits anything but a gated coordinate.
- A survivor escalates to manual pre-registration, never back into automated proposal — and never
  to a confirmatory claim until Phase C.

---

Last updated: 2026-06-24.
