# Generative grammar — the design (Beyond Prompting, interlocks kept)

## Status

**Design only. Nothing built, nothing activated.** This records path (b) of the "implement the
Beyond Prompting grammar" decision (2026-06-25): make the option-structure grammar **generative** —
replace the fixed closed lattice (`grid_universe_size()` = 70 named templates) with a *bounded,
typed production grammar* an LLM composes from leg primitives — while keeping the four interlocks the
foil paper (Huang & Fan, *Beyond Prompting*, arXiv:2603.14288 — see
[llm_proposer_plan.md](llm_proposer_plan.md)) drops. It is written before any code, the way
[read_gate.md](read_gate.md) and [llm_proposer_plan.md](llm_proposer_plan.md) were. The build is
phased and gated; **Phase 1 first**, and an LLM author is not activated until its gates are met and
the owner approves.

## Why

The closed grammar is finite and human-curated: each widening (the strangle, the risk reversal, the
credit spread, the calendar) is a discrete, human-signed edit that bumps `grid_universe_size()` and
the pinned batch. *Beyond Prompting* instead lets an LLM compose factor "recipes" from an open
formula grammar. Path (b) keeps our options domain but borrows the generativity: the LLM composes a
**leg combination** from primitives, and the *same* engine-verified greek signature + e-LOND FDR
control judge it. The manual widenings are the special case; this automates the composition — without
inheriting the paper's honor-system failure (a fixed `|t|>3` bar an adaptive search defeats, and an
economic story generated *with* the formula rather than checked against the data).

## The key finding: the engine is already \~90% generative

The architecture map (four parallel readers over the closed grammar, the engine, the signature
system, and the proposer gate) found that the heavy lift is **not** the engine:

- `run_real_structure_overlay` touches a structure only through an injected
  `select(day, params) -> list[leg]` callable. The leg dict
  `{sign, right, strike, contract, entry_net, mid, delta, expiration}` is the universal interface.
- The entry-credit / settlement / mark algebra is already **N-leg-general** (plain sums over legs):
  `entry_credit = Σ(-sign·entry_net)`, `settle_flow = Σ(sign·intrinsic)`,
  `mark = cash + hedge·price + Σ(sign·mid)·shares`.
- The **combined hedge** `-Σ(sign·delta)` already neutralizes arbitrary mixed-sign legs.
- **Staggered settlement** (`expiration = max(leg expirations)`; near legs settle while survivors
  live on) already supports arbitrary distinct per-leg expirations — a diagonal or a 4-leg
  composition runs today with no loop change.
- `structure_greek_signature` already derives `{legs, expirations, net_vega, net_delta, net_skew}`
  from *any* leg list, per-leg tenor aware.

So path (b) is: **replace the closed lattice with a bounded production grammar, and move the three
guarantees the closed menu gave us for free.**

## Architecture

### Primitives (the terminals)

A **leg** is `{side ∈ {long, short}, right ∈ {call, put}, strike_target, expiry_target}`, where

- `strike_target ∈ { delta-bucket d ∈ DELTAS | same-as-leg-K | offset-bucket o ∈ OFFSETS }`
- `expiry_target ∈ { dte-bucket ∈ DTES | gap-bucket g ∈ GAPS }` (gap = "this many DTE past another
  leg", for term structures)

plus a composition-level `hedge_rule ∈ {combined}` and a `predicted_sign ∈ {-1, +1}`. **Every
coordinate is a lattice member** (`DELTAS`, `OFFSETS`, `DTES`, `GAPS` are committed bucket sets),
never a free number. This is load-bearing for the seal (see "The numberless-value hole" below).

### Composition (the production rule)

A **`Composition`** is `(legs: 1..MAX_LEGS, hedge_rule, predicted_sign)` subject to

- `≤ MAX_EXPIRATIONS` distinct expiries (a governance cap), and
- `|net_delta| ≤ 1.0` — **not a tunable cap but a correctness invariant** fixed by the engine's
  `[-1, +1]·shares` hedge clamp: a composition whose net position-delta per contract exceeds 1.0 is
  *under-hedged*, so its "delta-hedged" P&L is not vol-isolated and `short_vol_statistics`' rf-netting
  identity breaks. Because delta is entry-day dependent, this is enforced at **runtime, fail-closed**
  (→ `measurement_invalid`) on the resolved legs, alongside the inline mechanism gate — not at
  construction. It becomes a knob only if the hedge is ever generalized beyond one lot per contract.

The reachable space is every composition within `(primitive lattices, MAX_LEGS, MAX_EXPIRATIONS)` —
**finite and enumerable in principle.** That finiteness is what keeps governance and e-LOND power
(below); it is the design's central constraint, not an afterthought.

### Canonical normal form (identity)

`canonical_key(composition)` = a content-addressed hash of the **sorted leg-set + hedge rule**, with
each leg keyed by its *bucket labels* (`delta=d` / `offset=o` / `same_k`, `dte` / `gap=g`), not raw
numbers. Two syntactically different LLM spellings of the same economics collapse to one key. This
replaces `_overlay_params_key` + the systematic-naming freeze in `enumerate_grammar_templates`. The
**8 committed overlays map to their canonical keys**, so the published 75-cell ledger dedups against
the new identity unchanged — the named grammar is a *sub-grammar* of the generative one (the Phase-1
acceptance test).

### Inline mechanism gate

The map surfaced that `structure_kill_gate` dispatches through a hardcoded name→runner dict and
**never calls `structure_greek_signature`** — the mechanism check is an *offline* CI test over the 7
named overlays (`TestGrammarSignatureMatchesEngine`). For an open grammar it must move **inline**: at
candidate construction / kill-gate, resolve the legs on a *canonical* entry day, compute the
signature, classify the family by **rule** (`derive_family(signature)` — e.g. `net_vega<0` single
expiry ⇒ VARIANCE; `net_skew` dominant + delta-offset ⇒ SKEW; `expirations>1` + opposite-sign vega ⇒
TERM; net-credit theta-positive defined-risk ⇒ CARRY), enforce `predicted_sign`/family **coherence**
against the derived signature, and **fail closed** on a mechanism-incoherent or unclassifiable
composition. This is the paper's headline improvement, kept per-composition rather than per-name.

### Engine dispatch

Generic, no per-name map:
`run_real_structure_overlay(select=composer(composition), entry_guard, hedge_mode='combined',
management='hold')`. The **composer** resolves each leg-spec via the existing primitive resolvers
(`select_entry` for delta-targeting; the calendar's same-strike logic for `same-as-leg-K`) plus the
**one missing terminal: a strike-offset resolver** ("a leg `o` buckets OTM of another leg"). The
`_uses_far_chain` / put-merge data path keys off the *composition's actual leg requirements* (any put
leg? any DTE > 60?) rather than a named family, so an arbitrary composition automatically gets the
right chains merged.

## What is preserved, and what breaks

**Preserved for free.** The entire honesty stack is grammar-agnostic — it guards the wire, the
scoring boundary, and the data identity, none of which the structure-composition axis touches:

- The **recording oracle** (`score_and_record` — every look is a recorded look, returns one bit).
- The **numberless information boundary** (the `SAFE_FIELDS` scrub + `assert_numberless` +
  coordinate-only `PROPOSAL_FIELDS`).
- The **lineage-excludes-grammar** rule (`_data_lineage_hash` folds only data + era-clip + end +
  capital + engine version, *not* the grammar), so a grammar change never resets the lifetime e-LOND
  counter.
- **Sealed-ticker omission** + human-gated universe/onboard edits.
- **CLOSED promotion** — a survivor escalates to manual pre-registration, the kill-gate never crowns.

**Broken by openness.** Three things the closed menu provided that must be rebuilt:

1. **Countability.** `grid_universe_size()` (=70) and `max_proposals_per_round()` are sums of
   Cartesian products over a fixed lattice; they vanish. e-LOND *validity* survives (the `Σγ_t ≤ 1`
   discount needs no denominator), but its **power** bleeds as `γ_t → 0`, and the pre-specification /
   governance role is gone. → replaced by the bounded production grammar (below).
2. **Dedup identity.** The canonicalize-to-a-fixed-menu key and the systematic-naming freeze cannot
   pre-enumerate names for an unbounded space. → replaced by the canonical normal form above.
3. **Declared signature + import-time typing.** `STRUCTURE_GRAMMAR`'s hand-authored per-overlay
   signature table and `_assert_grammar_well_typed` (presence check at import) have no fixed dict to
   iterate. → replaced by the inline computed-and-verified signature + `derive_family` above.

## The three hard requirements (restated as the design's spine)

1. **Canonical composition identity** is the single hardest one — get it wrong and the same economic
   structure re-counts under two spellings, re-spending the lifetime e-LOND budget. The normal form
   must be *total* (every legal composition has exactly one key) and *stable* (independent of leg
   emission order and of which entry day the signature was derived on).
2. **The numberless-value hole reopens.** `assert_numberless` is a key-*name* guard; it cannot catch
   a result-derived number smuggled as a *value* (`{'strike': <n>}`). Today that is safe only because
   params are lattice members. A generative grammar that let the model emit a free strike or DTE
   would reopen the hole. **The production-rule validator owns this boundary**: every leg coordinate
   must resolve to a committed bucket (`DELTAS`/`OFFSETS`/`DTES`/`GAPS`) *before* construction;
   `assert_numberless` stays the belt behind it, not the gate.
3. **The mechanism check goes inline and fails closed** (above), preserving the foil-paper defense
   per-composition.

## Countability → a bounded production grammar (governance)

The governance artifact shifts from *the named menu* to *the grammar space itself*: you review the
*space* once, the LLM walks it freely, and a composition outside an already-reviewed sub-grammar (a
new primitive, a raised cap) is a widening — **still human-signed and pinned**, exactly as a named
widening is today, but at the level of the rule-set rather than the menu. "A survivor escalates; a
widening is human-signed" survives intact, one level up.

But the governance has **four distinct parts doing different jobs**, and only two are tunable "knobs"
in the sense the closed lattice was:

1. **Two space caps** — `MAX_LEGS` (complexity) and `MAX_EXPIRATIONS` (term reach). Initial values
   `4` and `2` are *exactly* the committed sub-grammar's bounds (the iron condor is 4 legs, the
   calendar 2 expiries), so the named grammar stays a tight sub-grammar and the obvious next
   structures (ratios, butterflies, double calendars) are reachable.
2. **One correctness invariant** — `|net_delta| ≤ 1.0`, fixed by the hedge clamp (above),
   runtime-enforced, *not* a governance dial.
3. **A lifetime e-LOND budget** (plus the per-round `max_batch` that already exists) — **the actual
   power governor.** The saturation work showed the failure mode is *spend*, not space size:
   `γ_t → 0` as you record, regardless of how large the menu is. So the honest bound on power is a
   committed ceiling on *total recorded comparisons*, with `search_saturation` as the live
   "you're out of power" monitor. This is the knob the closed grammar never needed (70 cells could
   not bleed the budget) and the open grammar cannot do without.
4. **The bucket sets + the coherence gate** — which do **more** governance than any integer cap. The
   raw reachable count at `MAX_LEGS=4` is astronomical (leg-types in the low hundreds, unordered
   multisets up to four → millions), so the caps alone bound the *review space*, not the live search.
   What actually bounds the search is (a) freezing the bucket sets (`DELTAS`/`OFFSETS`/`DTES`/`GAPS`)
   initially to *exactly* the values the 8 overlays use (frozen in Phase 1) — so the
   *coherent* space starts near the current 70, and (b) the `derive_family` / sign-coherence gate,
   which rejects the vast mechanism-incoherent majority before it enters the FDR pool. So the
   **effective denominator is empirical** — compositions that pass the inline gate *and* trade —
   far below the raw count.

So the pinned governance object is: the bucket sets + `MAX_LEGS` + `MAX_EXPIRATIONS` + the lifetime
budget + the `derive_family` rule table (the `[-1, +1]` clamp pins `|net_delta| ≤ 1.0` for free). The
always-run test recomputes the *reachable* count the way `TestClosedGrammar` recomputes
`grid_universe_size()` today — but that count is a governance/review number, **not** the power bound;
the lifetime budget is the power bound. One caveat it must cover: an infeasible composition that never
trades still consumes an e-LOND slot (`p=None → e=0` counts toward the stream), so the open grammar
should pre-filter obvious infeasibility before scoring, or treat it as a power cost the budget pays.

## The honesty argument, and the caveat I have to keep

The seal is grammar-agnostic, so an open grammar **inherits the entire honesty stack** provided
proposals stay coordinate-only, numberless (lattice-bucketed), and engine-mechanism-checked. e-LOND
validity is untouched. What changes is **power and governance**, handled by the bound above.

The caveat is not optional, and it is the through-line from the saturation work: **a bigger
hypothesis space saturates *faster*.** `search_saturation` already showed the e-LOND bar at \~t≥6.3
against a data ceiling of \~t≥2.2 on the closed grammar; an open grammar pushes `γ_t → 0` quicker, so
the bar rises *sooner*. Generativity therefore is **not** "open the floodgates" — it is a bigger but
still-bounded, still-typed, still-mechanism-checked grammar, monitored live by the saturation
readout. And it makes the **Phase-C time-axis holdout more necessary, not less**: a generative LLM
still recalls "structures that work" (iron condor, calendar) from training data, which the numberless
prompt and the sealed-ticker vault do *not* defend against. Survivors stay EXPLORATORY until that
holdout exists; no "the model composed it from scratch" relaxation is justified.

## Phasing

Each phase is a separate PR with its own pins. Phases 1–3 activate no LLM.

- **Phase 1 — the grammar core (no engine, no LLM).** The primitive bucket sets (**frozen to exactly
  the values the 8 committed overlays use**, so the reachable count starts near 70; widening buckets
  is a later, separately-pinned step) + the `MAX_LEGS=4` / `MAX_EXPIRATIONS=2` caps; the `Composition`
  type; the production-rule validator (replaces `_validate_grammar`); the canonical normal form
  (replaces `_overlay_params_key`); the reachable-count bound (replaces `grid_universe_size`).
  **Acceptance test:** the 8 committed overlays are expressible as compositions and their canonical
  keys equal the published ledger's identities (the named grammar is a verified sub-grammar; the
  75-cell ledger dedups unchanged). All deterministic, all pinnable.
- **Phase 2 — the composer + inline mechanism gate.** The `select` composer + the strike-offset
  resolver; generic kill-gate dispatch; the inline signature-derive → `derive_family` → sign-coherence
  gate (fail-closed); the 0-trades → `measurement_invalid` feasibility filter; the runtime
  `|net_delta| ≤ 1.0` invariant check. **Acceptance test (real chains):** a composed named overlay is **byte-identical** to its
  hand-written form (the equivalence guarantee, the way `TestGenericStructureEngineEquivalence` pins
  the spec engine today).
- **Phase 3 — a deterministic menu-walker over the production grammar.** Enumerate a **sampled bounded
  slice** of the composition space (replaces `enumerate_grammar_templates`) — *not* run to completion,
  which would saturate the e-LOND budget; the saturation readout is the stop signal — and run the full
  score → lifetime-judge → record loop with `author=None`. Proves the loop end-to-end with no LLM, and
  lets the saturation readout report the *new* (larger) bar before any model touches it.
- **Phase 4 — the generative LLM author.** An `LLMProposer` that emits a `Composition` (leg-spec +
  hedge + sign) instead of menu coordinates; the gate/judge/record path is byte-identical. The
  numberless prompt now describes the **primitives + caps**, not a fixed menu. OFF by default,
  owner-gated (`EDGE_SEARCH_LLM_MODEL`), promotion CLOSED, survivors exploratory until Phase C.

## Parallelization

### Building it

The phase chain is sequential, but the *components within it are mostly independent pure functions*,
which is exactly the shape a Workflow fans out. The independent units — buildable and testable in
parallel by separate agents, each against a written spec:

- the **canonical normal form** `canonical_key(Composition)` (Phase 1) — a pure total function;
- the **production-rule validator** (Phase 1) — pure, given the bucket sets;
- the **`derive_family(signature)` classifier** (Phase 2) — a pure rule table over the three axes;
- the **strike-offset resolver** (Phase 2) — a new selector primitive, independent of the rest;
- the **reachable-count bound** (Phase 1) — a pure combinatorial count over the caps.

These have no ordering dependency on one another; only the *integration* points — the `select`
composer (wires the resolvers), the generic kill-gate dispatch (wires the validator + the inline
gate), and the equivalence verification — are sequential, because they compose the units. So the
build parallelizes as: **fan out the five pure units → integrate sequentially → adversarially verify
the seal** (a skeptic pass for the numberless-value hole and the canonical-key totality, the way the
saturation readout's belt-completeness gap was caught). The acceptance tests (named-overlay
expressibility in Phase 1; byte-identical equivalence in Phase 2) are the integration gates between
fan-out and merge.

### Running it

The *search itself* has a sharp split:

- **Scoring is embarrassingly parallel.** Each `(composition × ticker)` cell is an independent
  `run_real_structure_overlay` pass; a process pool over cells parallelizes it. **But the bound is
  RAM, not CPU** — chain stores are \~2.3 GB loaded, so only \~2–3 fit on the 7 GB CI runner (a
  per-ticker pool ≥ 3 OOMs; on the current runner it was a wash, see the CI-perf note in private
  memory). So runtime parallelism is `min(cores, RAM / store_size)`, and a per-ticker pool (one
  process holding one ticker's store, scoring all that ticker's compositions) only pays off on a
  larger box (16 GB+).
- **Judging is strictly sequential — and must stay deterministic.** e-LOND is *online*,
  committed-order: each cell's verdict depends on its stream position and the discoveries before it,
  so the judge cannot be parallelized (it is cheap arithmetic regardless). The subtle requirement:
  parallel scoring must **not** affect the judged order. The **canonical key gives a deterministic
  sort**, so cells are judged in a fixed canonical arrival order independent of which engine run
  finished first — verdicts are reproducible at any concurrency. This is the same determinism the
  signature derivation needs (a canonical entry day, fixed rf/tol), and it is why the canonical normal
  form is load-bearing twice: for dedup *and* for a stable judging order.

The shape, then: **parallel score (RAM-bounded) → collect → sort by canonical key → sequential
e-LOND judge → record.** The proposer round, the engine, and the FDR control already separate these
steps cleanly, so the parallelism is contained to the scoring fan-out and does not touch the seal.

## What must not happen

- **No free numbers.** A leg coordinate that is not a committed bucket member is a hard construction
  error, never a scored cell — the numberless-value boundary lives here, not in `assert_numberless`.
- **No grammar in the lineage hash.** A grammar edit (new primitive, raised cap) must never enter
  `_data_lineage_hash`, or it resets the lifetime e-LOND counter — the menu never touches the engine.
- **No universe widening on the structure axis.** The grammar opens the *structure* space only;
  tickers, the sealed vault, and onboarding stay human-gated exactly as today.
- **No promotion.** A survivor under the open grammar is still EXPLORATORY and escalates to manual
  pre-registration; the kill-gate kills, never crowns, until the Phase-C holdout exists.
- **No silent cap raise.** Raising `MAX_LEGS` / `MAX_EXPIRATIONS` / the lifetime budget, or adding a
  bucket, is a widening — human-signed and pinned, the same governance act as a named widening today.
  (`|net_delta| ≤ 1.0` is *not* in this list: it is a correctness invariant fixed by the hedge clamp,
  not a governance dial — it changes only if the hedge itself is generalized.)

## Decisions

All three review questions are resolved (owner-signed, 2026-06-25); this doc is the Phase-1 spec.

- **Governance knobs** — two space caps `MAX_LEGS=4` / `MAX_EXPIRATIONS=2`; `|net_delta| ≤ 1.0` a
  runtime correctness invariant, *not* a dial; a lifetime e-LOND budget as the real power governor;
  the bucket sets + the `derive_family` coherence gate doing the heavy lifting (see the governance
  section).
- **Bucket sets** — frozen in Phase 1 to *exactly* the values the 8 committed overlays use, so the
  reachable count starts near 70; widening the buckets is a later, separately-pinned governance step.
- **Phase-3 enumeration** — a *sampled bounded slice*, never run-to-completion (which would saturate
  the e-LOND budget), with the saturation readout as the stop signal.
