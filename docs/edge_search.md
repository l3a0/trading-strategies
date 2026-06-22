# Edge-search log — the automated kill factory

This is the repo's record of **automated, FDR-controlled sweeps** over a
committed batch of cheap hypotheses. Where [the exploration log](explorations.md)
records one hand-built scout at a time, this log records a *campaign*: a whole
template class swept at once, with the multiple-testing arithmetic done
honestly.

**Read this first — what this is and isn't.** Each campaign is run by
[edge_search.py](../edge_search.py), an **exploratory** harness, not a
registered experiment. Every candidate re-tags the cycles the pinned naked
runs already produced — so a campaign *spends the sample* and can only **kill**
a class of ideas or **justify** taking a survivor to a pre-registration. It is
never itself a confirmatory verdict. The numbers are pinned
([test_edge_search.py](../test_edge_search.py)) so a swept dead end stays dead;
pinning a campaign does **not** promote it to a finding. A survivor earns a
registration — the discipline [the trend-gate prereg](prereg_trend_gate.md)
protects — not a headline.

## Why the harness looks like this

Three guardrails carry the honesty, and skipping any one turns an automated
search into a machine that manufactures false positives at speed:

- **One shared kill-gate.** The cooldown, up-move, and IV-richness ideas are
  the same statistic wearing different masks: tag each cycle with a binary
  entry rule, compute `D_A = mean(treated P&L) − mean(other P&L)`, and
  calibrate against a same-count permutation null. The harness runs every
  candidate through that one gate.
- **A multiple-testing ledger.** Test nine hypotheses at p < 0.05 and noise
  alone hands you false positives; significance is judged across the **whole
  batch** with Benjamini-Yekutieli — the FDR procedure that stays valid under
  dependence (the candidates share tickers, overlap in time, and nest by
  window), at the cost of a harmonic penalty over plain Benjamini-Hochberg. (The
  structure phase upgrades this control to **e-LOND**, #3b — an e-value online-FDR
  procedure valid under *arbitrary* dependence; see Campaign 2.)
- **A sealed vault.** A loop that generates and tests can never "commit before
  seeing the number," so the harness commits the *data it never sees* instead:
  the search loads only MSFT + SPY; **QQQ is held out**. A survivor is
  confirmed on the sealed set in a separate, manual step. QQQ is a weak vault
  on purpose (\~0.8 correlated with the search set); the strong vault is a
  structurally different underlying the search never saw — a premium-data
  fetch, out of MVP scope. The split is a `Campaign(search, sealed)` config
  (run via `run_batch`), not a hardcoded constant, so the same templates sweep
  the next batch of tickers — roll a fresh underlying into `sealed` each round
  and the held-out vault stays genuinely unseen as the search expands.

**Scope (MVP).** Templates that *re-tag existing naked cycles* only — cheap,
no engine re-runs. Structure-side ideas (roll rules, stop-loss, spread width)
change the trades themselves and need a full `run_real_cc_overlay` per
candidate; those are the expensive verdict phase, deliberately out of scope.
The permutation null is per-template: the uniform same-count shuffle by default,
but a template with temporally-structured treatment supplies its own — cooldown
uses the structure-preserving trigger-placement permutation (redraw each
ticker's rips from its own terminals) that `cooldown_scout` uses. BY rather than
BH is used because the candidates are dependent.

### What the false-discovery rate controls

FDR — the **false-discovery rate** — is the lever the ledger pulls. It reframes
significance from "is this candidate real?" to "of everything I flag as a
discovery, what fraction is noise?" Controlling it at q (here q = 0.10) means
that across the batch's flagged survivors, no more than \~10% are expected to be
false. A raw p-value can't ask that question — it only sees the one test in
front of it, so a batch of nine candidates at p < 0.05 expects roughly one
false hit even if every idea is dead, and a real campaign tests far more.

FDR is the right knob because of the cost it bounds, and the two obvious
alternatives miss it. A bare p < 0.05 per candidate is too loose: it ignores
how many tests were run, so a big enough batch always coughs up "significant"
noise. Controlling *any* false positive (the family-wise error rate,
Bonferroni and kin) is too strict: on a batch this correlated it rejects almost
everything, killing real candidates to avoid a single fluke. FDR sits between
them, bounding the wild-goose rate among the survivors you'd actually spend
effort confirming — the cost that matters before a sealed-vault run.
Benjamini-Yekutieli (the guardrail above) is the dependence-robust procedure
that enforces it.

## Architecture

The harness is a funnel — spend the sample cheaply on the left, gate expensive
confirmation on the right:

![Automated edge-search architecture: a hypothesis generator feeds a cheap seeded scout and an FDR ledger that auto-pins most candidates as nulls and regenerates the next batch; a survivor crosses a one-way gate to human registration and confirmation on a sealed, held-out vault, ending in a pinned verdict.](figures/edge_search_architecture.svg)

*The teal stages are the automated, sample-spending loop; the amber stages are
the human-gated, sealed confirmation a survivor must cross; gray boxes are
infrastructure. Most candidates die at the ledger and the loop regenerates —
only a survivor reaches the one-way gate.*

- **Enumerate a mechanism-template batch** (`enumerate_candidates`). Each
  template is a falsifiable, sign-predicting family — cooldown(N), up-move(k),
  IV-richness — expanded across its settings into one committed batch. A
  candidate with no predicted sign is refused, so the batch is structured bets,
  not a blind grid.
- **Run each through the one shared kill-gate** (`kill_gate`): the `D_A`
  treated-minus-other split against a same-count permutation null, plus a
  generic vol-confound probe.
- **Record everything; judge the batch, not the candidate** (`run_campaign`,
  logged to `edge_ledger.jsonl`). Significance is decided across the whole
  campaign by `benjamini_yekutieli` — automating the search multiplies the
  multiple-comparisons danger, and the ledger is the only thing that keeps it
  honest.
- **Cross the gate by hand.** The loop never promotes a survivor; a survivor
  earns a pre-registration (a human step) and is confirmed on the **sealed
  vault** — the held-out tickers (`SEALED_TICKERS`) the search never loads. A
  loop can't "commit before seeing the number," so it commits the *data it
  never sees* instead; the sealed vault is the automation-compatible substitute
  for pre-registration.

**The principle:** automate the bookkeeping that keeps the search honest — the
enumeration, the shared gate, the FDR ledger — never the judgment that promotes
a result.

## Engine-re-run phase — the structure class

The re-tag class above is cheap because it never changes the trades. The
structure-side strategies — the delta-neutral short-vol / straddle / iron-condor
overlays — *do* change the trades, so each candidate is a `(template, ticker)`
cell that runs a full `run_real_*_overlay` engine pass rather than re-tagging
fixed cycles. It is **built** (`run_structure_campaign`), a parallel phase that
does not bend the cheap re-tag gate:

- **A second template class, on a closed grammar.** `STRUCTURE_TEMPLATES` — the
  short call at 0.25Δ and ATM, the two-leg ATM straddle, the defined-risk iron
  condor, the OTM short strangle (widening 1), the bullish risk reversal
  (widening 2 — the first new family), and the bull put credit spread (widening 3 —
  the first CARRY structure) — crossed with the search tickers
  (`enumerate_structure_candidates`).
  Each cell runs its overlay on the live chains. Every template draws its params
  from a fixed menu (`ALLOWED_GRID`) enforced at construction: an off-menu value
  raises rather than running, so the reachable hypothesis space stays finite,
  enumerable, and pre-specified (`grid_universe_size`) — a pre-specification and
  power precondition, not an FDR-validity one (e-LOND, the control, needs no fixed
  comparison count; the `Σγ_t ≤ 1` budget replaces the offline denominator — only
  the BY diagnostic still needs a count). The grammar is also **economically typed**
  (`STRUCTURE_GRAMMAR` — the typed source of truth, with `ALLOWED_GRID` its flat lattice
  view): each overlay declares a `PremiumFamily` (the committed six are four `VARIANCE` + the
  risk-reversal `SKEW` + the credit-spread `CARRY`) and a `signature` of three ROBUST axes —
  `net_vega` (variance), `net_delta` (direction), and `net_skew` (the skew edge: are the SHORT legs
  richer in IV than the LONG legs?). `net_gamma` is deliberately absent — for offset-leg structures
  the iron-condor's short gamma and the risk-reversal's long gamma overlap in magnitude, so no
  tolerance pins both; `net_vega` carries the vol-selling claim. Enforcement is **two-layer**:
  `_assert_grammar_well_typed` gates PRESENCE at import (a registered family + a complete
  signature — it can't run the engine without data), and the dataset-gated
  `TestGrammarSignatureMatchesEngine` cross-checks the declared signature against the engine's
  ACTUAL greeks — it runs each overlay on real chains, backs the IV out of each entry leg's mid,
  computes the three axes (`vol_premium.structure_greek_signature`), and asserts the
  engine-derived `{legs, expirations, net_vega, net_delta, net_skew}` matches what the grammar
  declares. So a composition whose greeks (or skew) contradict its claimed family fails —
  mechanism checked against the engine, not a post-hoc label. The guarantee is per-verified-overlay
  (a dataset-gated test that must run with data, not a constructor invariant): a widening adds its
  structure to that test (on a ticker that trades it — all six are verified on SPY, the put-leg
  straddle/iron-condor/risk-reversal/credit-spread by merging the separate SPY puts file at load
  since the canonical SPY store is calls-only).
- **A HAC-t kill-gate with a closed-form null.** The score is
  `short_vol_statistics`'s Newey-West (HAC) t-stat on the daily rate-netted P&L,
  whose asymptotic null is standard normal — so the p-value is closed-form
  (`erfc(t/√2)/2`, one-sided for the predicted positive premium) and there is
  *no* per-candidate permutation. The cost is N engine runs plus one
  Benjamini-Yekutieli pass over the t-stat p-values: the same ledger, a cheaper
  gate. The campaign is deterministic — no seed.
- **The seal is a non-equity name.** TLT (long bonds) is held SEALED by
  omission — `STRUCTURE_SEARCH` is MSFT/SPY/QQQ/GLD/XLE/EEM and never loads TLT.
  QQQ — the re-tag seal — appears in the structure cross-section, so it cannot
  seal this phase.
- **A price-vs-chain scale guard.** Before scoring, each ticker's price file is
  checked against the chain's as-traded strikes (`validate_dailies.scale_ratio`):
  a ticker off-scale (a split mismatch — see Campaign 2) is flagged
  `measurement_invalid` and scored `p = None`. It still COUNTS toward BY's n — a
  comparison the loop ran — but can never be rejected, so it cannot masquerade as
  a survivor and cannot shrink the denominator to loosen the bar for the other
  cells. (Dropping it before BY, as an earlier cut did, was a data-dependent
  N-shrink: fewer comparisons mechanically lowers the rejection threshold.)
- **Every comparison is recorded to a committed lifetime ledger.** `record_trials`
  appends each distinct structure comparison (template / ticker / params, its
  result, and a per-ticker `_data_lineage_hash`) to a committed, append-only
  `idea_ledger.jsonl` — distinct from the regenerable, `.gitignore`d
  `edge_ledger.jsonl`. The lineage hash folds exactly the inputs that move the
  result — store checksum, era-clip, end date, capital, engine version — and
  deliberately **not** the menu (`ALLOWED_GRID`): the same comparison gives the
  same t-stat regardless of what else the grid can express, so folding the grammar
  in would re-lineage every prior look on a grid edit and reset the counter. Deduped
  and timestamp-free, so re-running a campaign on the same data lineage adds nothing:
  it is the *same* comparison, and the git history is the timeline. This is the
  guess-counter that never silently resets — the foundation the e-value FDR control
  (#3b, `evalue_fdr.py`) reads so the comparison count is the program's lifetime
  total, not one session's. It carries the result statistics (the answer key), so an automated
  proposer must never read it.
- **A number-free scoreboard for proposers.** `build_proposer_corpus` projects the
  lifetime ledger to an allow-list view — the hypothesis coordinates (template /
  ticker / params / predicted sign) plus a one-bit verdict — and
  `render_proposer_corpus` formats it as a table. Every result statistic is dropped
  *by construction* (`scrub_ledger_row` copies only `SAFE_FIELDS`, so a result column
  added to the ledger later cannot leak); the magnitude is the dangerous channel — a
  near-miss t-stat tells a proposer where to fish. **SURVIVED rows are excluded:** a
  survivor is an `elond_survivor` — a cell flagged by **e-LOND, the FDR control of
  record** (#3b), not the retained BY diagnostic — the one genuine "fish here"
  coordinate, so it escalates to manual pre-registration out-of-band and never feeds
  back into automated proposal. (Keying the exclusion off the control rather than the
  diagnostic matters because the two need not coincide: e-LOND's `(R+1)` reward can
  flag a cell BY does not, and the prereg is explicit that *only e-LOND flags* — so a
  BY-only cell is KILLED and stays in the corpus, while an e-LOND survivor the
  diagnostic missed is correctly dropped.) The corpus is the duds to avoid (KILLED)
  plus unmeasurable tickers (INVALID, a per-ticker data-quality state). **The scrub is airtight; a
  file-hiding read-gate is NOT** (red-teamed and verified, [read_gate.md](read_gate.md)). The
  obvious next move — vault the ledger + a scoped read-deny, or commit only the scrubbed projection —
  is *theater*: the answer key is a deterministic RECOMPUTATION from committed engine code + chains,
  so `python -c`, `git show HEAD:idea_ledger.jsonl`, the pinned-test t-stats, and the proposer's own
  `run_proposer_round` return value each walk straight past any file-fence. You cannot deny-list the
  engine without denying the proposer's own scoring step. The real interlock is a **process
  boundary** (a sandboxed proposer with only the scrubbed corpus + grammar, no engine/data, plus a
  trusted oracle that charges every score to the lifetime e-LOND stream before returning a one-bit
  verdict — count every look rather than hide a number you cannot hide), which is infrastructure, not
  config. It is moot today: the proposer is the DETERMINISTIC menu-walker (no model to peek) and
  promotion stays CLOSED; the boundary is the precondition for activating an LLM author. Tried-set
  neutrality additionally rests on the grammar staying closed and fully enumerated.
- **The FDR control of record is e-LOND (#3b) — registered AND now ACTIVATED.** The
  per-batch Benjamini-Yekutieli gate has been replaced as the control by the e-value
  procedure (`evalue_fdr.py`, pre-registered in
  [docs/prereg_fdr_budget.md](prereg_fdr_budget.md)): each cell's HAC-t p-value is
  calibrated to an e-value (Vovk-Wang), and the campaign's cells are judged as a stream
  by e-LOND (Xu & Ramdas 2024) — proven online FDR under *arbitrary* dependence,
  peek-whenever, with no across-batch independence assumption. `run_structure_campaign`
  now sets `elond_survivor` (the control flag); BY is retained as a reported *diagnostic*
  (`by_survivor`). The honest price: e-LOND is *less* powerful than BY (calibration is
  lossy), buying dependence-robustness + online validity rather than power — and it is
  *stricter* here, so the verdict is unchanged: **0 / 56 cells flagged.** The strongest
  cell (SPY short-call-25, t_NW ≈ +2.17) calibrates to e ≈ 4.1, far below the
  head-of-stream bar 1/(α·γ₁) ≈ 16.3. The machinery is oracle-tested against the
  `online-fdr` package; `TestStructureCampaign` now pins the e-LOND verdict on real
  chains and `TestStructurePhase` the synthetic flagging path.
- **The control is judged over the *lifetime* stream at record time, not per batch.**
  `run_structure_campaign` runs e-LOND over a single batch — correct for the published
  one-shot, where that batch IS the head of the stream, but a second batch judged the same
  way would restart the discount sequence at `t = 1` and re-face the loosest
  `1/(α·γ₁)` bar: a silent per-session budget reset, the multiple-looks leak the
  registration exists to prevent. The `--record` path closes this with
  `judge_against_lifetime_stream`: it places the committed prior ledger ahead of the new
  rows and runs ONE e-LOND pass over the whole concatenation, recording each new row's
  lifetime-stream verdict. Because e-LOND is online (a cell's bar depends only on the cells
  before it), that verdict is fixed on arrival and never moves under later appends, so the
  published head-of-stream ledger is unchanged (empty prior ⇒ lifetime == per-batch). This is the
  cumulative-n control the future LLM proposer's judging path will reuse. Pinned by the
  always-run `TestLifetimeStreamJudge`.
- **Graduation stays manual.** A survivor earns a pre-registration and a manual
  sealed-vault confirmation, never an automated verdict. The harness surfaces
  survivors; it never crowns them.
- **The deterministic menu-walker proposer (Phase 1, no LLM).** `run_proposer_round`
  is the first consumer of the scrubbed scoreboard — and the smallest end-to-end slice
  of the loop a future LLM proposer plugs into, with a *dumb enumerator* standing in for
  the author. It **reads** only the scrubbed corpus, **proposes** every grammar template
  (`enumerate_grammar_templates` — all `grid_universe_size()` of them, the committed seven
  keeping their names so a coincident cell dedups against the published ledger instead of
  re-counting under a new name) crossed with the **onboarded** search tickers minus what's
  already tried, **grammar-gates** each at `StructureCandidate` construction, **runs** the
  engine, **judges** over the lifetime e-LOND stream (`judge_against_lifetime_stream`), and
  **records** — so the next round re-reads the corpus and skips them. An un-onboarded
  proposed ticker is flagged for the human-gated onboard pipeline (no auto-fetch). The point
  of building it now, with no model, is to prove the read → propose → gate → run → judge →
  record → re-read plumbing against the real primitives — the LLM later swaps its JSON output
  for the enumerator while the gate, the lifetime judge, and the record stay identical. Run
  via `python edge_search.py propose` (preview), `propose --run` (score, no record), or
  `propose --record`; pinned by the always-run `TestMenuWalkerProposer`.

The roll / stop-loss / spread-width structure *variants* — which need their own
engine parameters per candidate — are the natural next expansion of this class.

---

## Campaign 1 — cheap entry-conditioning class — EMPTY (2026-06-13)

**The batch.** Nine candidates from three mechanism templates, swept on the
real MSFT + SPY chains (QQQ sealed; SPY clipped at the corrected 2010-05-17
boundary — figures re-pinned 2026-06-17, verdict unchanged), campaign seed
20260613, 1,000-draw permutation null (cooldown's is the trigger-placement
variant), BY at q = 0.10:

- **cooldown(N)** — a cycle entered within N days of a same-ticker rip does
  *worse*. Predicts `D_A < 0`. N ∈ {7, 30, 60, 90}.
- **up_trend(window)** — a cycle entered after a positive trailing-window
  return does *worse* (momentum forfeits the right tail). Predicts `D_A < 0`.
  window ∈ {21, 63, 126, 252} trading days.
- **iv_rich** — a cycle whose entry IV exceeds trailing realized vol does
  *better* (richer premium). Predicts `D_A > 0`.

**The result.** No candidate survives campaign-wide BY:

| Template | Param | `D_A` | Sign as predicted? | One-sided p | vol-confound |
| --- | --- | --- | --- | --- | --- |
| cooldown | N=7 | +351 | no | 0.741 | −0.018 |
| cooldown | N=30 | +581 | no | 0.908 | −0.036 |
| cooldown | N=60 | +687 | no | 0.895 | −0.039 |
| cooldown | N=90 | +1,217 | no | 0.969 | −0.038 |
| up_trend | window=21 | +341 | no | 0.742 | −0.054 |
| up_trend | window=63 | −18 | yes | 0.492 | −0.097 |
| up_trend | window=126 | +262 | no | 0.651 | −0.115 |
| up_trend | window=252 | +965 | no | 0.903 | −0.102 |
| iv_rich | — | +733 | yes | 0.095 | −0.082 |

**The reading.** Two findings, both consistent with what the repo already
knew:

1. **Every up-move-conditioning template is wrong-signed.** All four cooldown
   horizons and three of four up_trend windows predict `D_A < 0` and deliver
   `D_A > 0` — entries after a rip or a rally *lose less*, not more. The lone
   sign-correct window (63 days) has `D_A = −18` (\~zero) at p = 0.49, i.e.
   noise. This is the third independent confirmation, now swept as a batch,
   that **conditioning call-selling entry on recent upward price action has the
   sign backwards** on these names (cf. the cooldown and trend-gate kills).
2. **The one suggestive candidate is the known confound, and it still fails
   FDR.** `iv_rich` is sign-correct (`D_A = +733`) and individually
   suggestive (p = 0.095) — but its `vol_confound` is negative (rich-IV
   entries sit in lower trailing vol, i.e. calm markets), the same low-vol
   confound the IV-richness scout pinned. And p = 0.095 is nowhere near the BY
   rank-1 threshold of \~0.0039 (= 0.10 / (9 × Σ 1/i)). The multiple-testing
   math, not a single p-value, is what empties the class.

**What this campaign settles.** The cheap entry-conditioning class, swept at
once under honest FDR control, contains no survivor. The productive search
moves to the structure side (roll/stop/spread templates that need engine
re-runs) and to structurally different underlyings (the sealed vault upgrade) —
both out of this MVP's scope, both the natural next phase.

---

## Campaign 2 — structure class — EMPTY (2026-06-17; NVDA folded in 2026-06-20; SPY/MSFT/QQQ put chains merged 2026-06-22; strangle widening 2026-06-22; risk-reversal widening 2026-06-22; credit-spread widening 2026-06-22; calendar widening 2026-06-22)

**The batch.** Forty-nine `(template, ticker)` cells — **seven** structure templates
(short call 0.25Δ, short call ATM, ATM straddle, 25Δ/10Δ iron condor, the OTM
short **strangle** (widening 1), the bullish **risk reversal** (widening 2 — the
first new family), the bull put **credit spread** (widening 3 — the first CARRY
structure), and the long **calendar** (widening 4 — the first TERM family, two
expirations; all below)) crossed with the seven search
tickers (MSFT, SPY, QQQ, GLD, XLE, EEM, NVDA; **TLT sealed**) — each a full
`run_real_*_overlay` engine pass scored by the Newey-West HAC t-stat against its
asymptotic normal null (closed-form p, no permutation), then judged as a stream by
**e-LOND** (the FDR control of record, #3b) — with Benjamini-Yekutieli at q = 0.10
retained as a diagnostic. The chains are era-clipped at the live `CHAIN_CLEAN_START`
(exploratory sees the corrected SPY boundary).

**The result.** No cell is flagged by e-LOND, the control (`0 / 56`); none survives
the BY diagnostic either. The strongest is SPY short-call 0.25Δ at t = +2.17 (the
exploratory cousin of the frozen +2.54 short-vol headline, now on the wider corrected
SPY span): individually suggestive at p \~0.015, but it calibrates to an e-value of
\~4.1 — far short of the e-LOND head-of-stream bar 1/(α·γ₁) \~16.3, and missing the BY
diagnostic's rank-1 bar (\~0.0004 for the 56-cell batch, 55 scored) by an order of magnitude.
The next cells trail off fast — SPY short-call ATM at +1.60, GLD's call wings near
+1.15 — and every put-leg straddle/iron-condor cell is at most +0.72 (SPY straddle).
The risk-reversal cells are all wrong-signed (negative alpha over cash on all seven
tickers; see widening 2 below), and so are the credit-spread cells (widening 3).

**A data-hygiene catch that mattered.** The first run flagged two XLE "survivors"
(short call t = +4.16 / +6.86) — entirely an artifact. XLE did a **2:1 split on
2025-12-05**, and yfinance split-adjusts its `Close` even with
`auto_adjust=False`, so the unadjusted price file had every pre-split close
*halved* while the option strikes stayed as-traded. The delta-hedge ran on a
price at half the real scale and the straddle equity ran to −$23M; the
short-call wing's residual mis-hedge in a falling sector fabricated the positive
t. The chain data was clean all along (the entry-band validator was right) — the
**price file** was wrong. The fix backs the split out at the source
(`load_unadjusted_prices`, with `_unsplit_factor`), and a price-vs-chain **scale
guard** (`validate_dailies.scale_ratio`, wired into the campaign) now precludes
the class. Repaired, XLE shows no edge (short-call t \~−1.7) and the batch is
empty.

**A second data-completeness catch.** The early runs loaded each ticker's
CANONICAL store, which for SPY/MSFT/QQQ is **calls-only** (their puts live in a
separate `{ticker}_option_dailies_puts.csv`). So the two put-leg templates — the ATM
straddle and the 25Δ/10Δ iron condor — never entered on those three tickers: six of
the 28 cells were idle flat rf-credit curves recording a vacuous t \~ 0, the tell
being that straddle and iron-condor came out *identical* per ticker (two different
structures cannot). The fix merges the separate puts file at load
(`_load_ticker_data` via `_put_chain_paths`, the way `run_registered_vrp` loads the
SPY straddle). The merge is **window-additive**: a puts file can PREDATE the calls
(QQQ's puts start 2011, its calls 2016, and QQQ has no era clip), so the load clips the
merged window to CALL days — every structure needs a call leg, so this is exactly the
calls-file span, and merging puts gives the put-leg structures a put to trade against
WITHOUT stretching the window into a calls-free span (which would dilute the t-stat with
idle rf days and re-measure even the call cells). So the call cells are byte-unchanged
and only the put-leg cells become real. The fix also folds the puts checksum into the
per-ticker `_data_lineage_hash` so the re-measured cells re-record honestly, and adds a
0-trades → `measurement_invalid` guard so a future non-trading cell is flagged (e=0)
rather than scored as a real \~0.
Re-measured, the six cells are real and distinct, and all still far from significance
— SPY straddle +0.72 / iron-condor −1.08; MSFT −0.67 / +0.52; QQQ −0.10 / +0.14. The
`0 / 28` verdict was unchanged (the strangle, risk-reversal, credit-spread, and calendar widenings later took the batch to 35, then 42, then 49, then 56); the cells are now honest measurements rather than
flat-curve artifacts (the same class of bug as XLE — defective inputs, not a real
edge — caught on the other side: missing data rather than mis-scaled).

**What this campaign settles.** The delta-neutral short-vol structure class on
seven underlyings contains no cell flagged by e-LOND, its honest FDR control — the
variance premium that survives a single HAC-t (SPY's call wing) does not clear the
cross-section's online-FDR bar. The next moves are a stronger sealed
vault and the roll/stop/spread structure variants that carry their own engine
parameters.

### Widening 1 — the OTM short strangle

The first structure added beyond the original three — the **Stage-B payoff**: now
that the overlays are one generic engine, a new structure is a STRUCTURE_SPEC +
selector + a typed grammar entry, with no engine change. The OTM short strangle is
the straddle's wider cousin (short a 0.25Δ call + 0.25Δ put, combined-delta-hedged,
held to expiry), same `VARIANCE` family and `{2 legs, 1 expiry, short ν}`
signature — so the dataset-gated `TestGrammarSignatureMatchesEngine` cross-checks it
for free. It took the closed grammar `30 → 39` and the committed batch `4 × 7 = 28 →
5 × 7 = 35`. Mechanically it is an **append**, not a correction: the seven new
strangle cells are judged at the tail of the lifetime e-LOND stream (after the prior
28) and recorded; the existing rows are byte-unchanged. The strangle is another null
(MSFT +0.50 / SPY +1.06 / QQQ +0.34 / GLD +0.33 / XLE −1.34 / EEM −0.78 / NVDA
−1.33 — all well short of significance), so the verdict stays `0 / 35` (then `0 / 42`
after widening 2, `0 / 49` after widening 3, `0 / 56` after widening 4, below).

### Widening 2 — the risk reversal (the first NEW family, SKEW)

The first widening into a genuinely **new family**. A bullish risk reversal — SHORT
a 0.25Δ put + LONG a 0.25Δ call at one expiry, combined-delta-hedged, held to expiry —
harvests the equity put-call **skew** (puts priced richer than equidistant calls) by
selling the rich put wing and buying the cheap call wing. Its edge is the skew, not a
position greek, which forced the typed signature to **evolve**.

**The signature schema change (a human-signed grammar decision).** Typing SKEW by net
greeks turned out to be fragile: a delta-hedged risk reversal has net gamma ≈ 0 and
net vega ≈ 0 (the put and call legs offset), and the residual net gamma's sign is
*skew-dependent and borderline* — across SPY/MSFT entries the risk-reversal's
`|net γ| / Σ|leg γ|` ratio sits at \~0.17–0.34, right on top of the iron-condor's
short-gamma ratio (\~0.26–0.37, opposite sign). No tolerance classifies both cleanly.
So `net_gamma` was **dropped** as a signature axis; the schema became three ROBUST
axes — `net_vega` (variance: VARIANCE short, SKEW neutral), `net_delta` (direction:
the RR is net long), and a new `net_skew` that types the family by the **edge itself**:
`short_rich` if the SHORT legs sit at higher IV than the LONG legs (the RR, ratio
\~+0.54 — decisive), `long_rich` if the reverse (the iron-condor longs its richer OTM
wings, \~−0.10), `flat` for an all-short structure with no asymmetry. The dataset-gated
`TestGrammarSignatureMatchesEngine` cross-checks all three against SPY's engine greeks.

**The one engine touch.** The risk reversal is the first MIXED-sign hedged structure
(short put, long call), so it exercises the engine's `combined` delta hedge in its
general form. The old form hedged `Σ delta`, which equals the net position delta only
when every leg is short; the correct general form is `−Σ sign·delta`. The fix is
bit-identical for the all-short straddle/strangle (verified by the equivalence and
registered pins) and the only correct form for the reversal.

**The result.** It took the closed grammar `39 → 48` and the committed batch
`5 × 7 = 35 → 6 × 7 = 42`, appending 7 risk-reversal cells to the lifetime e-LOND
stream. The risk reversal is another null — in fact **wrong-signed on all seven
tickers** (MSFT −2.09 / SPY −1.78 / QQQ −1.27 / GLD −3.30 / XLE −2.00 / EEM −2.64 /
NVDA −0.19): there is no harvestable put-call skew premium at these names/era. (The
overlay's large *raw* P&L is risk-free interest on the cash balance; the **alpha over
cash** the campaign scores is negative.) The verdict stays `0 / 42` (then `0 / 49`
after widening 3, `0 / 56` after widening 4, below).

### Widening 3 — the bull put credit spread (the first CARRY structure)

The first widening into the **CARRY** family — a defined-risk, theta-positive
structure. A bull put **credit spread** — SHORT a 0.25Δ put + LONG a 0.10Δ put (further
OTM, the wing) at one expiry, combined-delta-hedged, held to expiry — collects a net
credit and caps the downside with the long wing. It is the **put half of the iron
condor**, so the selector (`select_credit_spread`) reuses that structure's put-side band
and wing logic. Single-expiration, so the Stage-B engine handles it with **no engine
change** — a pure STRUCTURE_SPEC + selector + delegate + typed grammar entry, the same
clean shape as the strangle.

**The signature — engine-verified, not assumed.** The declared signature is
`{2 legs, 1 expiry, net_vega: short, net_delta: long, net_skew: long_rich}`, and the
`net_skew` axis is where the engine **corrected the initial read**. The intuition "you
sold the richer near-ATM put, bought the cheaper OTM wing → `short_rich`" is **wrong**:
on the equity put skew the further-OTM put sits on the *steep* part of the smile, so the
LONG wing carries the HIGHER implied vol. The dataset-gated
`TestGrammarSignatureMatchesEngine` backs the IV out of each SPY entry leg and reads
`long_rich` — the same read as the iron condor (which also longs its richer OTM wings).
The declared signature was fixed to match the engine. The other two axes confirm the
intuition: net `short` vega (the short leg sits nearer the money, where vega is larger)
and net `long` delta (short a put is long the underlying — the one axis that makes the
credit spread distinct from every other overlay; no other structure is short-vega AND
long-delta).

**The result.** It took the closed grammar `48 → 66` (its `dte × short_delta ×
wing_delta` lattice mirrors the iron-condor's put side: `3 × 3 × 2 = 18` new templates)
and the committed batch `6 × 7 = 42 → 7 × 7 = 49`, appending 7 credit-spread cells to
the lifetime e-LOND stream. The credit spread is another null — **wrong-signed on all
seven tickers** (MSFT −2.08 / SPY −0.91 / QQQ −0.72 / GLD −3.24 / XLE −2.74 / EEM −2.21 /
NVDA −0.06): the defined-risk carry collects a credit, but the delta-hedged vol-P&L is
negative at these names/era. (As with the risk reversal, the large *raw* P&L is rf
interest on the cash balance; the **alpha over cash** the campaign scores is negative.)
The put legs trade on the calls-only SPY/MSFT/QQQ stores via the merged puts file (175 /
105 / 107 entries), and a 0-trades → `measurement_invalid` guard would flag a future
non-trading cell rather than score a vacuous \~0. The verdict stays `0 / 49`. The first `TERM`-family
widening — the long calendar (widening 4) — follows directly below.

### Widening 4 — the long calendar (the first TERM family, two expirations)

The first widening that touches the **engine itself**, and the deepest lift so far. A
long calendar — SHORT a near-month ~ATM call + LONG a far-month call at the SAME strike,
combined-delta-hedged — harvests the **term structure** of implied vol: it sells the
near (faster-decaying) leg and buys the far (richer-vega) leg, so the spread is net LONG
vega across two expirations. That is the `TERM` family's defining axis (opposite-sign
vega across expirations), distinct from the single-expiration VARIANCE and SKEW families.

**The engine change (a human-signed surgery).** Every prior structure settled all its
legs at once: the loop tracked a single scalar expiration and an `elif date >= expiration`
branch closed the whole position. A calendar has two distinct expirations, so the engine
gained **staggered settlement** — the near leg settles at its own expiry (a `settle_leg`
trade) while the far leg keeps marking and hedging, and the structure closes only when the
far leg expires. The new branch is guarded so a single-expiration structure (every other
overlay) takes the byte-identical scalar-expiration path, which the equivalence and
registered pins confirm did not move. The Saturday-expiry handling and the `gap ≤ 4`
assert are applied **per leg**.

**The signature schema change (a human-signed grammar decision).** A two-expiration
structure can't price both legs on one clock. `structure_greek_signature` gained an
`entry_date` argument: when passed, each leg's IV is backed out at its OWN tenor (the
far leg has more time value and more vega than the near at the same strike — exactly the
long-vega edge). Single-expiration callers pass no `entry_date` and are byte-unchanged.
The `net_skew` axis was also refined to read `flat` for a SAME-strike spread: a calendar's
short-vs-long IV gap is the term-structure slope, not a wing asymmetry, so without the
guard a TERM structure would mis-type as SKEW. The calendar declares
`{2 expirations, 2 legs, net_vega long, net_delta neutral, net_skew flat}`, cross-checked
against SPY's engine greeks by the dataset-gated `TestGrammarSignatureMatchesEngine`. The
selector enforces a `min_gap_dte = 30` floor so the far leg is genuinely further out — a
near-adjacent far leg reads vega-neutral, not the long-vega calendar the family claims.

**The result.** It took the closed grammar `66 → 70` (the calendar lattice is
`near_dte × far_dte = 2 × 2 = 4`) and the committed batch `7 × 7 = 49 → 8 × 7 = 56`,
appending 7 calendar cells to the lifetime e-LOND stream. Six of the seven traded and are
**wrong-signed** (SPY −2.44 / QQQ −0.11 / GLD −0.66 / XLE −1.25 / EEM −0.49 / NVDA −1.88):
a long-vega calendar pays for term-structure exposure these names/era don't reward. The
seventh, **MSFT, is measurement-invalid** — MSFT's listed chains carry no far call at the
near leg's exact strike (a same-strike calendar needs the strike quoted ≥30 DTE past the
near, which MSFT's grid doesn't list), so the structure never enters; the no-trades guard
flags it (e = 0, counts toward n = 56, never flagged), exactly as designed. The verdict
stays `0 / 56`. The roll/stop/spread *variants* and the **diagonal** (a calendar with
different strikes) remain next.

### NVDA — the seventh ticker (live-onboarded, folded in)

NVDA was onboarded after the original six tickers were frozen, and is now folded
into `STRUCTURE_SEARCH` as the seventh — its seven `(template, NVDA)` cells are part
of the cross-section above. It keeps its own callout here because NVDA's onboarding
was the worked example of the clean-gate; the verdict is unchanged.

**The result.** No NVDA cell is flagged — every t-stat is negative, so each one is
on the wrong side of the predicted positive premium before any FDR bar (e-LOND or the
BY diagnostic) even applies:

| Template | t (NW) | One-sided p | Flagged? | Measurement valid? |
| --- | --- | --- | --- | --- |
| short_call_25 | −0.96 | 0.8315 | no | yes |
| short_call_atm | −0.96 | 0.8315 | no | yes |
| straddle | −1.22 | 0.8888 | no | yes |
| iron_condor | −1.47 | 0.9292 | no | yes |
| strangle | −1.33 | 0.9082 | no | yes |
| risk_reversal | −0.19 | 0.5753 | no | yes |
| credit_spread | −0.06 | 0.5239 | no | yes |
| calendar | −1.88 | 0.9699 | no | yes |

**A clean data-hygiene read — for once.** Where XLE needed a split repair, NVDA
passed every clean-gate check on the first try. `validate_dailies.py` streamed
NVDA over 2010-12-01 → 2026-06-05 (3,895 trading days) and returned
**VERDICT: CLEAN** — no clip needed, no defective in-band days, the store clean
from its first day (100% usable, 0.00% defective, BS-disagree 0.15%). The
price-vs-chain **scale ratio is 1.006 [OK]** — NVDA's unadjusted price file sits
on the chain's as-traded scale, so the split guard finds no mismatch and the
ticker enters the FDR cross-section with `measurement_invalid = false`. No repairs
applied, nothing flagged for human review, and **no `CHAIN_CLEAN_START` entry is
warranted**. The one wrinkle — zero-bid rates run elevated in the early years
(10–28% across 2010–2013) — does not change the call: the validator still
classifies every day usable with no defective in-band days, so the elevated
zero-bid tail is a liquidity feature of NVDA's early options, not a defect that
clips the span.

**What this addendum settles.** Adding a seventh underlying to the structure
cross-section produces no survivor and no hygiene exception — NVDA's chains are
CLEAN from the first day, its price file is on-scale, and all four templates land
wrong-signed. The short-vol structure class stays empty.
