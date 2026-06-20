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
  short call at 0.25Δ and ATM, the two-leg ATM straddle, and the defined-risk
  iron condor — crossed with the search tickers (`enumerate_structure_candidates`).
  Each cell runs its overlay on the live chains. Every template draws its params
  from a fixed menu (`ALLOWED_GRID`) enforced at construction: an off-menu value
  raises rather than running, so the reachable hypothesis space stays finite and
  countable (`grid_universe_size`) — the countability the FDR ledger rests on.
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
  survivor is itself a BY-thresholded result — the one genuine "fish here" coordinate
  — so it escalates to manual pre-registration out-of-band and never feeds back into
  automated proposal; the corpus is the duds to avoid (KILLED) plus unmeasurable
  tickers (INVALID, a per-ticker data-quality state). **Contingency, not yet an
  interlock:** this is leak-proof only for a proposer that reads *through* it —
  `idea_ledger.jsonl` is committed and carries the answer key, and nothing yet denies
  a repo-aware agent from reading it directly, so "the proposer must never read the
  ledger" is an honor-system convention today. The access boundary (a vault dir + a
  scoped read-deny, or committing only the scrubbed projection to the proposer-visible
  path) is the unbuilt interlock that makes the scoreboard meaningful; the tried-set
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
  *stricter* here, so the verdict is unchanged: **0 / 24 cells flagged.** The strongest
  cell (SPY short-call-25, t_NW ≈ +2.17) calibrates to e ≈ 4.1, far below the
  head-of-stream bar 1/(α·γ₁) ≈ 16.3. The machinery is oracle-tested against the
  `online-fdr` package; `TestStructureCampaign` now pins the e-LOND verdict on real
  chains and `TestStructurePhase` the synthetic flagging path.
- **Graduation stays manual.** A survivor earns a pre-registration and a manual
  sealed-vault confirmation, never an automated verdict. The harness surfaces
  survivors; it never crowns them.

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

## Campaign 2 — structure class — EMPTY (2026-06-17)

**The batch.** Twenty-four `(template, ticker)` cells — four structure templates
(short call 0.25Δ, short call ATM, ATM straddle, 25Δ/10Δ iron condor) crossed
with the six search tickers (MSFT, SPY, QQQ, GLD, XLE, EEM; **TLT sealed**) —
each a full `run_real_*_overlay` engine pass scored by the Newey-West HAC t-stat
against its asymptotic normal null (closed-form p, no permutation), then judged as a
stream by **e-LOND** (the FDR control of record, #3b) — with Benjamini-Yekutieli at
q = 0.10 retained as a diagnostic. The chains are era-clipped at the live
`CHAIN_CLEAN_START` (exploratory sees the corrected SPY boundary).

**The result.** No cell is flagged by e-LOND, the control (`0 / 24`); none survives
the BY diagnostic either. The strongest is SPY short-call 0.25Δ at t = +2.17 (the
exploratory cousin of the frozen +2.54 short-vol headline, now on the wider corrected
SPY span): individually suggestive at p \~0.015, but it calibrates to an e-value of
\~4.1 — far short of the e-LOND head-of-stream bar 1/(α·γ₁) \~16.3, and missing the BY
diagnostic's rank-1 bar (\~0.0011 for 24 dependent tests) by an order of magnitude.
Every other cell is t < +1.2.

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

**What this campaign settles.** The delta-neutral short-vol structure class on
six underlyings contains no cell flagged by e-LOND, its honest FDR control — the
variance premium that survives a single HAC-t (SPY's call wing) does not clear the
cross-section's online-FDR bar. The next moves are a stronger sealed
vault and the roll/stop/spread structure variants that carry their own engine
parameters.

### Campaign 2 addendum — NVDA (live-onboarded) — EMPTY

NVDA was onboarded after the campaign ran and swept through the same structure
class as a single live-staged ticker. The four `(template, NVDA)` cells extend
the cross-section by one underlying; the verdict is unchanged.

**The result.** No NVDA cell is flagged — every t-stat is negative, so each one is
on the wrong side of the predicted positive premium before any FDR bar (e-LOND or the
BY diagnostic) even applies:

| Template | t (NW) | One-sided p | Flagged? | Measurement valid? |
| --- | --- | --- | --- | --- |
| short_call_25 | −0.96 | 0.8315 | no | yes |
| short_call_atm | −0.96 | 0.8315 | no | yes |
| straddle | −1.22 | 0.8888 | no | yes |
| iron_condor | −1.47 | 0.9292 | no | yes |

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
