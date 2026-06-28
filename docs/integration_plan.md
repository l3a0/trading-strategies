# Integration plan — extending the honest-search apparatus

> **Status: F1 (the backend protocol, `backend.py`) and F2 (the factor scorer, `factor_backend.py`) are
> BUILT and pinned; F3–H2 are design only.** Nothing here changes promotion: a survivor stays EXPLORATORY
> until the Phase-C time-axis holdout exists
> ([docs/read_gate.md](read_gate.md), [docs/generative_grammar_plan.md](generative_grammar_plan.md)).

This plan covers two related extensions to the apparatus:

- **Part 1 — a factor backend** behind a formalized backend protocol (a *new domain*: alpha-factor
  formulas, scored by a real backtest engine).
- **Part 2 — a hypothesis-first proposer upgrade** lifted from AlphaAgent
  ([arXiv:2502.16789](https://arxiv.org/abs/2502.16789), KDD 2025) — a *cross-backend* generation
  improvement.

They share one foundation and one throughline. The foundation: the honest core is hypothesis-blind, so a
new domain is a new *backend*, not a core rewrite. The throughline: the backend protocol's `mechanism()`
method is exactly what AlphaAgent's alignment gate checks — the option backend implements it
(`derive_family`); the factor backend cannot. The mechanism gate is what the option domain has and the
factor domain lacks, and it recurs in both parts.

## The shared foundation: the honest core is hypothesis-blind

A repo map (three Explore agents over `edge_search.py` / `evalue_fdr.py` / `generative_grammar.py` /
`generative_engine.py` / `read_gate_wire.py` / `vol_premium.py`) confirmed the load-bearing fact:

> The honest core keys on `(data_lineage_hash, phase, template, ticker, params)` and reads result
> statistics (`p_value`, `t_stat_newey_west`, `sign_ok`, `measurement_invalid`, `elond_survivor`) — never
> engine output. e-LOND is the online FDR control of record over *any* backend's p-values. The seal
> (`assert_numberless` + `BANNED_RESULT_FIELDS`) is a static key-name guard, independent of backend shape.

So both extensions sit on an unchanged core. Extract — from what `run_composition_round` /
`score_composition` / `judge_compositions_against_published` / `gate_compositions` actually call — the
minimal contract any backend must satisfy:

```python
class Backend(Protocol):
    def enumerate(self) -> list[Candidate]: ...           # the bounded, pre-specified space
    def validate(self, c: Candidate) -> Candidate: ...    # production-rule gate; raises off-grammar
    def canonical_key(self, c: Candidate) -> str: ...     # content-addressed, order-invariant, sign-excluded
    def mechanism(self, c: Candidate, data) -> str | None # the mechanism gate; None == fail-closed
    def lineage(self, c: Candidate, data) -> str: ...     # SHA over (data, engine version)
    def score(self, c: Candidate, data) -> dict: ...      # -> {statistic, p_value, sign_ok,
                                                          #     measurement_invalid, family, n_days, ...}
```

The honest core consumes only the `score` row schema. Formalizing this protocol (today implicit in the
option backend) is the precondition for *both* extensions.

## Part 1 — a factor backend behind the protocol

Integrating an alpha-factor grammar + backtest engine is: (1) formalize the protocol above, then (2) add a
second backend that implements it. The option backend is unchanged; the honest core is unchanged.

| Protocol method | Option backend (exists) | Factor backend (new) |
| --- | --- | --- |
| `enumerate` | `enumerate_compositions` / `enumerate_grammar_templates` | bounded formula grammar (Qlib ops + depth/window caps), or the LLM author |
| `validate` | `validate_composition` | bounded production rules (cap depth + operator/operand/window alphabets) |
| `canonical_key` | `canonical_key` (sha256 of sorted leg tokens) | SymPy-normalize → sha256 (partial — see caveats) |
| `mechanism` | `derive_family` (greek signature → family) | **no greek analog** — built via a loading-regression check (see caveats) |
| `lineage` | `_data_lineage_hash` | sha over the equity panel checksum + engine version |
| `score` | `run_real_structure_overlay` + `short_vol_statistics` | Qlib `D.features` (eval formula → signal) + alphalens `calc_ic` → IC-t |

**What the factor backend reuses:**

- **Grammar + engine: Microsoft Qlib** (MIT). Its expression engine (`qlib.data.ops` — `ElemOperator` /
  `PairOperator`: `Ref/Mean/Std/Corr/Cov/Rank/Add/...`) *is* the alpha grammar; string formulas over
  `$close/$volume/...`; `D.features(instruments, fields, start, end)` evaluates a formula to a
  per-name-per-day signal. The engine + eval are usable **standalone** (decoupled from Qlib's ML pipeline).
  The data layer (`.bin`/CSV/Parquet, `dump_bin.py`, US + CN universes) is the panel.
- **Scoring: alphalens-reloaded** (the actively-maintained Quantopian fork). `compute_ic` (Spearman / rank
  IC), quantile / long-short returns, the IC t-stat — the factor analog of `short_vol_statistics`. The
  `score` row emits `{statistic: IC-t, p_value, sign_ok, measurement_invalid, family}` so the honest core
  is untouched.
- **Canonicalization: SymPy** (normalize commutative/associative/idempotent forms) → sha256 — the
  `canonical_key` analog. Partial (see caveats); `egg` (e-graph equality saturation) is the
  better-but-heavier option (Rust + hand-written rewrite rules) if dedup quality bites.

```mermaid
flowchart TB
    HC["honest core — unchanged, hypothesis-blind<br/>e-LOND · ledger · numberless oracle · seal · holdout"]:::keep
    PR["backend protocol — formalized<br/>enumerate · canonical_key · validate · score to t,p,family · lineage"]:::new
    OB["option backend — existing<br/>generative_grammar + vol_premium + short_vol_statistics"]:::keep
    FB["factor backend — new<br/>Qlib grammar + engine · alphalens IC/t · SymPy dedup"]:::new
    GAP["mechanism gate — no greek to read<br/>build a loading-regression check; holdout still binds"]:::gap
    HC --> PR
    PR --> OB
    PR --> FB
    FB -. must build .-> GAP
    classDef keep fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
    classDef new fill:#e6f1fb,stroke:#185fa5,stroke-width:2px,color:#042c53
    classDef gap fill:#faeeda,stroke:#854f0b,stroke-dasharray:5,color:#412402
```

## Part 2 — the hypothesis-first proposer upgrade (from AlphaAgent)

AlphaAgent is the closest public cousin to this repo — it targets "overfitting, selection bias through
p-hacking, and multiple testing." But its **Eval Agent is a feedback loop** (propose → backtest → *refine
on the result* → repeat), exactly the in-loop p-hacking the numberless oracle forbids. So we don't
integrate its loop; we lift its two good ideas and run them behind the seal — across *either* backend.

- **Hypothesis-first generation.** Before the candidate, the author states a falsifiable economic claim
  (`{premium_family|hypothesis, direction, conditioning, claim}`), generated numberless and recorded
  audit-only (like today's `reasoning`). Add `HYPOTHESIS_FIELDS` to `read_gate_wire.py`.
- **Hypothesis-factor alignment, made rigorous.** An oracle-side gate where the backend's `mechanism()`
  must match the author's *claimed* family, or the cell is killed — the direct cure for the foil-paper
  failure mode (a persuasive rationale on a candidate that does something else).

```mermaid
flowchart LR
    subgraph integrated["Integrated — generation behind the existing seal"]
      direction LR
      H["hypothesis<br/>premium · direction · claim"]:::new
      P["LLM author<br/>numberless"]:::keep
      G["grammar gate<br/>validate + canonical_key"]:::keep
      S["engine score<br/>mechanism()"]:::keep
      A["alignment gate<br/>claimed == mechanism()?"]:::new
      E["e-LOND -> record"]:::keep
      AUD["provenance log<br/>audit only"]:::keep
      H --> P
      H -. audit only .-> AUD
      P --> G --> S --> A --> E
      E -. "scrubbed verdicts (numberless)" .-> P
    end
    subgraph omitted["Omitted — the AlphaAgent leak"]
      direction LR
      F["propose"]:::drop --> B["backtest"]:::drop --> Rf["refine on result"]:::drop --> F
    end
    classDef keep fill:#f1efe8,stroke:#5f5e5a,color:#2c2c2a
    classDef new fill:#e6f1fb,stroke:#185fa5,stroke-width:2px,color:#042c53
    classDef drop fill:#faeeda,stroke:#854f0b,stroke-dasharray:5,color:#412402
```

**The connection between the two parts:** the alignment gate (Part 2) calls the backend's `mechanism()`
(Part 1's protocol). For the **option** backend, `mechanism()` is `derive_family` — so the alignment gate
*works*, and a persuasive story that doesn't match the greeks is killed. For the **factor** backend there's
no greek to read, so `mechanism()` has to be *built* — the loading-regression check in the caveats below;
until it exists the alignment gate is a no-op for factors, so the upgrade lands first on the option backend.

**Why this specific pair.** They are one mechanism in two halves, not two independent imports. A hypothesis
with no alignment check is the foil paper — an unchecked story, the thing we reject; alignment with no
hypothesis has no left-hand side, nothing to check the measured family against. So you cannot take one
without the other. What the pair *adds* over the statistical apparatus (t-stat + e-LOND + holdout) is a
second, orthogonal control axis: those three decide whether an edge is *real and generalizes*; none of them
checks that it comes from the economic source you *claim*. That axis starts to matter only once an LLM
drives the search — a bare menu-walker has no story to be wrong about, but an LLM can attach a persuasive
rationale to a spurious cell, which is how the foil paper reported a 3.11 Sharpe on stories it never checked
against the data.

Hypothesis-first generation is the price of admission for letting economic insight into the loop: the claim
is stated *before* the evidence (a falsifiable prediction, not a post-hoc rationalization — the
generalization of the existing `predicted_sign` pre-commitment, now over the mechanism, not just the
direction), and alignment is what makes the claim *cost* something. The import keeps AlphaAgent's skeleton
(`hypothesis → align`) and swaps its checker: AlphaAgent aligns *semantically* (an LLM judges the story fits
the formula — circular), while we align against the *engine* (greeks) or a *regression* (loadings). One
honest note: alignment kills a *false claim*, not a weak edge — it never vetoes a cell for poor statistics
(that is the t-stat's job), only for a measured mechanism that contradicts the declared one. Whether a
corrected re-label may re-enter, or is blocked the way sign-shopping already is, is a dedup choice to settle
when the gate is built.

## Reuse vs replace — the verdicts

| Component | Verdict | Why |
| --- | --- | --- |
| honest core (`evalue_fdr` e-LOND, `edge_search` ledger/oracle, `read_gate_wire` seal, the holdout) | **keep / reuse as-is** | audited, the repo's contribution, hypothesis-blind |
| option backend (`generative_grammar`, `generative_engine`, `vol_premium`, `short_vol_statistics`) | **keep** | audited + pinned (`TestSpyShortVolRegression` t=2.54); the rf-netted vol-P&L is load-bearing |
| `_data_lineage_hash`, `canonical_key`, `judge_against_lifetime_stream`, `assert_numberless` | **keep** | primitives / design choices, audited; no faster-or-better swap |
| Benjamini-Yekutieli, the Newey-West lag rule | **keep** for options | pinned to the published ledger; the factor backend uses libraries from the start |
| data loading (`load_chain_store` / `select_entry`, CSV→tuple parsing) | **replace** for factors | Qlib data layer / Parquet + polars — faster, standard, the bottleneck once you score thousands of factors daily (the option path stays; a Parquet swap there is perf-only) |
| Qlib | **reuse (external, MIT)** | the factor grammar + engine + IC eval |
| alphalens-reloaded | **reuse (external)** | the factor scorer (IC / rank-IC / t / quantile) |
| SymPy | **reuse (external)** | factor canonicalization (partial) |
| `anthropic` (optional) | **reuse** | the LLM author (Part 2); or none, via `ClaudeCodeProposer` |
| `egg` (e-graphs) | **optional** | better canonicalization; defer until SymPy's incompleteness bites |
| gplearn, AlphaCFG, AlphaGen | **skip / reference** | gplearn has no dedup; AlphaCFG/AlphaGen are *search* strategies — our menu-walker + LLM author already cover proposal |

The honest answer to "replace existing components": the repo's audited core and option engine are the
*quality* pieces — keep them. The real replace is the hand-rolled **data layer** for the factor path;
everything else new is *added* (Qlib/alphalens), not swapped.

**Why the option store stays on its CSV loader.** Migrating it to Qlib/Parquet looks like reuse but isn't:
Qlib is equity-only — no chains, expiries, greeks, or mark-vs-quote checks — so a Qlib option loader would be
a from-scratch rewrite of the loader's expensively-learned, individually-tested hygiene (the
`CHAIN_CLEAN_START` era clips, Saturday-expiry settlement, the mark-outside-quote clamp, the split back-out,
the puts- and far-DTE merges, the scale guard) on a base that models none of it. It also isn't the
bottleneck — the option backend loads one ticker per campaign and the engine dominates, and the chains are
already gzipped — and a format change is the repo's worst re-pin hazard: every pinned regression clips to the
canonical store and folds its checksum into the lineage hash, so a CSV→Parquet swap moves the bytes and
risks float round-trip drift, silently re-pinning every published number. The `Backend` protocol already
lets the two backends keep different storage, so a unified store is not needed for cleanliness. The migration
earns its keep only if the option load becomes the bottleneck or during a clean re-fetch (when you re-pin
anyway) — and only behind a byte-identical equivalence test (Parquet load == CSV load, trade-for-trade,
t-stat-for-t-stat).

## The honest caveats (where quality is at risk)

1. **The mechanism gate has no factor analog — the throughline of both parts.** Qlib confirms no economic
   typing. `derive_family` (and the alignment gate that checks it) is the option domain's foil-paper
   defense; it does not transfer. A data-mined formula with a high IC has nothing to disqualify it but the
   statistics, so the factor backend leans *entirely* on e-LOND + the holdout. Either build an economic
   typing (map a formula to value/momentum/carry/quality and verify the loading — hard, aspirational) or
   accept the weaker defense and let the holdout carry it. The next subsection makes that economic typing
   concrete — and explains why an LLM may *propose* it but never *judge* it.
2. **Exact canonicalization is unsolved at scale.** SymPy is heuristic and incomplete; `egg` needs a
   rewrite-rule set; AlphaGen/AlphaAgent only *approximate* (RPN / AST similarity). Leaked duplicates are
   conservative for e-LOND validity (over-counting raises the bar) but weaken the pre-specification
   precondition and the BY-diagnostic denominator. Plan for partial dedup.
3. **A new data dependency** — a broad equity panel (Qlib's US/CN data or a fetch); the repo has option
   chains, not equities.
4. **e-LOND power, again.** An open factor space saturates the FDR budget fast (we saw t≥6.6 by cell 145);
   the harness will mostly *kill*. A tiny, mechanism-prioritized menu retains power, which a formula space
   resists by nature.
5. **Audit-trail constraint.** `short_vol_statistics`, the Newey-West lag, and BY are pinned to the
   published ledger — do not swap them for the option path; the factor path uses libraries from the start.
6. **Qlib is a real dependency to vet** — MIT, ~45k stars, but v0.9.7 (Aug 2024), uneven cadence, and the
   standalone-expression-eval path is under-documented.

### Why not an LLM mechanism judge?

A tempting shortcut for the factor backend is to let an LLM judge whether a formula's economic mechanism
"makes sense." It fails *as a gate*, for one structural reason: the option-domain gate is honest because it
is a **measurement, not an opinion**. `derive_family` never reads the story; it reads the engine's measured
greeks and checks `claimed == measured`. An LLM asked "is this coherent?" returns a plausibility judgment
from the same kind of system that proposed the candidate — the foil paper's exact failure mode (a
persuasive story, not a mechanism checked against the data; arXiv:2603.14288). The decisive technical
problem is **non-determinism**: the apparatus rests on "every gate is a reproducible function of committed
code + data, every look a recorded look," yet LLM-as-judge verdicts are not reproducible run to run — their
inter-rater reliability sits well below human experts' (*Rating Roulette*, 2025) — and a verdict that
changes between runs cannot be pinned in a test, content-addressed into `_data_lineage_hash`, or anchor an
e-LOND stream that needs each verdict permanent on arrival. Three more pile on: **circularity** (the same
model family generates and ratifies — and tellingly, no published LLM alpha-miner checks the proposed
*mechanism* with a regression; AlphaAgent / FactorEngine / RD-Agent verify the hypothesis-factor fit
*semantically*, not against the data), **gameability** (a narrative judge on the promotion path is a Goodhart
target rewarding fluent memos), and the **insight-vs-evidence violation** (it launders model insight into a
gate — exactly why the repo keeps `reasoning` strictly audit-only).

The fix changes the LLM's job from **adjudicate** to **propose**. The LLM emits a falsifiable
`claimed_family` coordinate (numberless; a closed enum: value / momentum / carry / quality / low-vol), and a
**deterministic spanning regression checks it** — regress the factor's long-short returns on the registered
premium panel (FF5 + momentum + variance-premium + carry) and require a significant, correctly-signed
*loading* — the regression coefficient, the factor's measured exposure to a premium — on the *claimed*
premium after orthogonalizing out the rest (clearing a Harvey-Liu-Zhu-style bar — roughly t > 3 for a
newly-mined factor, adjusted for multiple testing);
insignificant or wrong-signed returns `None`, the `measurement_invalid` branch. **That regression is the
factor's `derive_family`** — `claimed_family == engine_derived_family` becomes
`claimed_family == regression_derived_family`. The LLM supplies the falsifiable claim; the regression
supplies the falsification — a measurement, pinnable and reproducible, with the LLM never seeing the
loadings. Its other roles stay off the verdict path: **ordering** the menu so the most motivated candidates
spend the fat early e-LOND budget (valid under any data-independent order — it casts no verdict), and the
existing audit-only `reasoning`. The rail is one-directional — the mechanism judgment may **kill or reorder,
never promote** — so an LLM wrong in the kill direction costs power, never a false discovery.

This types the factor; it does not prove the edge. A formula can load cleanly on the claimed premium and
still be an in-sample fit, or merely repackage a known premium. So the loading regression is the *mechanism*
layer only: e-LOND remains the *multiplicity* layer and the **Phase-C holdout the binding out-of-sample
defense**. Caveat 1's gap is fillable — but by a regression the LLM feeds, never by an LLM verdict.

## Phasing

- **F1 — formalize the backend protocol.** Extract the `Backend` Protocol from the option backend; refactor
  the option path to implement it explicitly. No behavior change; the precondition for everything else.
- **F2 — the factor scorer.** Wrap Qlib `D.features` + alphalens `calc_ic` into a `score`-shaped row on a
  small fixed equity panel; verify it feeds `judge_against_lifetime_stream` unchanged.
- **F3 — the factor grammar + canonicalization.** A bounded formula grammar (`validate` + SymPy
  `canonical_key`), enumerable and pre-specified.
- **H1 — the hypothesis schema + the alignment gate** (deterministic, stub-testable behind the seal). On
  the option backend it is a real strengthening even with no LLM; on the factor backend it degrades to a
  no-op until a `mechanism()` exists.
- **F4 / H2 — the proposer.** The menu-walker / LLM author over each grammar, emitting hypotheses
  (env-gated OFF). Resolve the factor mechanism decision (economic typing vs holdout-only).
- **Promotion stays gated on Phase C** for both domains.

## Bottom line

The repo was already factored for this: the honest core is the asset, and it does not know or care whether
a hypothesis is an option structure or an alpha formula. Both extensions reduce to **one new abstraction**
(the backend protocol) plus **public libraries** (Qlib + alphalens + SymPy for the factor backend;
`anthropic` for the proposer) — with the data layer replaced for scale. The single thing that does *not*
port is the mechanism gate, and that — not the grammar, the engine, or the proposer — is what decides
whether the factor domain is worth doing before the holdout exists.

_Last updated: 2026-06-27._
