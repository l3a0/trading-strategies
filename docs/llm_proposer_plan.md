# The LLM proposer author — design (item 4, not built)

## What this is

The read-gate apparatus (see [read_gate.md](read_gate.md)) is feature-complete through the
kernel-seal container integration. The last piece is **item 4: replacing the deterministic
menu-walker proposer with a real LLM author** — a temperature-0 Claude call that reads the
scrubbed scoreboard + the grammar and proposes coordinate-only `(overlay, ticker, params,
predicted_sign)` cells.

This doc records the design **before any of it is built**. Nothing here activates a model. The
standing rule holds: no LLM author runs until the gates below are met and the owner explicitly
approves. It exists so the two consequential decisions — *where the model call lives* and *how
the prompt is constructed* — are settled on the record, not improvised at wiring time.

## Where the Claude call lives: oracle-side

Three independently-designed architectures converged on the same answer, and an adversarial pass
confirmed it: **the API call is issued by the trusted oracle process** (the one that already
holds the engine, chains, ledger, and network), not from inside the sandboxed proposer.

The reason is a hard constraint. `proposer_client.run_proposer_loop` invokes its `author` seam
*inside* the proposer process — but that process is spawned by `launch_in_container` under
`--network none` with no host env and no key. A Claude call needs network and a key. So the call
cannot happen inside the sealed container; it moves to the trusted side. `ANTHROPIC_API_KEY` and
all egress stay on the oracle; the proposer container stays `--network none`, unchanged (it is
the CI-certified seal `CONTAINER_SEAL_FLAGS` pins).

## The seal changes shape — say it loudly

Moving the LLM oracle-side **trades the container's kernel isolation for a correctness argument**,
and this must be stated plainly because it is easy to assume the kernel jail still protects the
LLM. It does not — for the LLM, the seal is now three things:

- the model's prompt contains **no result statistics** — only the grammar menu and the scrubbed
  corpus (coordinates + one-bit verdicts), `assert_numberless`-checked, SURVIVED rows excluded;
- the model's output is **coordinate-only** (`PROPOSAL_FIELDS`), gated by `StructureCandidate`;
- every score is **recorded** (`score_and_record`), so every look spends FDR budget.

A model that only ever sees a numberless prompt and only emits a menu choice **cannot recompute a
score or chase a seen winner** — so the correctness argument is sound *for coordinate-only
output*. But it is a different kind of guarantee than the container provides:

| Guarantee | Holds even if… |
| --- | --- |
| **Kernel isolation** (the C-2 container) | …there is a bug — the OS backstops it. |
| **Correctness argument** (the LLM oracle-side) | …only if the argument is airtight — no OS backstop. |

The C-2 container was built to cage an untrusted-*code* proposer that could read the ledger or run
the engine. An LLM emitting coordinates from a numberless prompt is sealed by *information*, not
isolation — so the container becomes **vestigial for the LLM path**. That is an acceptable trade
for coordinate-only output, but it relocates the load-bearing guarantee onto the
`assert_numberless` checks and the coordinate-only gate, which must therefore be the
hardest-tested code in the system.

## The prompt builder is the leak surface — three options

The correctness argument lives or dies in **whatever code constructs the model's prompt**. There
is no such code today; writing it is part of item 4. Its inputs are already numberless —
`enumerate_grammar_templates()` (the menu) and `build_proposer_corpus(load_idea_ledger())` (the
scrubbed corpus; the `SAFE_FIELDS` allow-list drops every statistic). The risk is entirely in
**where the builder runs and what it reads**:

- **(A) Oracle-side builder.** Simplest, and what the three designs assumed. But the builder runs
  in the trusted process that *also* holds the raw `idea_ledger.jsonl` (with the t-stats) and the
  engine. A bug — calling `load_idea_ledger()` (raw) instead of `build_proposer_corpus(...)`
  (scrubbed), or appending a "helpful" statistic — leaks a number into the prompt, and the *only*
  catch is an `assert_numberless` on the assembled prompt. No kernel backstop.
- **(B) Builder inside the container that also calls the API.** Ruled out: the container is
  `--network none` with no key, so it cannot make the call.
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

Because (A) has **no kernel backstop**, `assert_numberless` is now the *sole* guard on the prompt —
so it is the load-bearing seal and must be treated as such:

- it runs on the **assembled prompt string** (the exact bytes sent to the API), not merely the
  structured inputs;
- the builder reads **only** `build_proposer_corpus(load_idea_ledger())` (the scrubbed projection)
  and `enumerate_grammar_templates()` — **never** the raw `load_idea_ledger()` or any engine output;
- it is the **most-tested code in the system** — every banned-field shape, nested/stringified
  values, and explicitly the "read the raw ledger instead of the scrubbed corpus" mistake;
- a leak is fail-closed: a non-numberless prompt **raises before the API call**, never silently
  ships.

Option (C) remains recorded as the tighter fallback should that single guard ever prove
insufficient.

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

## The phased plan

- **Phase A — gates (buildable now, activates nothing).** The container must-dos (in
  `oracle_server.launch_in_container`'s docstring): a wall-clock round timeout around `serve`'s
  `readline`; a base-image digest pin in `Dockerfile.proposer`; making the `read-gate-container`
  CI job a *required* check; seccomp + an explicit `--user`. Plus the read-gate gaps the review
  found: numberless-gate the `propose` reply; make the soft `launch` *refuse* an LLM author (only
  `launch_in_container` may carry one); a hard rounds cap so a model cannot drain the e-LOND
  budget; and have the oracle override the model's *self-reported* `model_served`.
- **Phase B — the model wiring (owner's explicit go).** The oracle-side Claude author (the
  `anthropic` client, the numberless prompt + `prompt_sha`, real `model_served` capture into
  `record_provenance`), activating `_resolve_llm_author`. The prompt builder is **(A), oracle-side
  (decided)**, so `assert_numberless` on the assembled prompt is the load-bearing seal and is built
  + tested first. Gated on Phase A and on the owner accepting the seal reframe above.
- **Phase C — confirmatory trust (research-blocked).** The time-axis holdout. Until it is
  operative, LLM survivors stay exploratory.

## What must not happen

- No LLM author runs until Phase A's gates are met and the owner approves. Interlock #5
  (`_assert_llm_boundary`) enforces this today: `_resolve_llm_author()` returns `None`, so
  `propose --llm` fails closed.
- The model never sees a result statistic, and never emits anything but a gated coordinate.
- A survivor escalates to manual pre-registration, never back into automated proposal — and never
  to a confirmatory claim until Phase C.

---

Last updated: 2026-06-23.
