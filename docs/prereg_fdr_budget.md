# Pre-registration: e-value false-discovery control for the edge-search loop

**Status:** REGISTERED — effective at this file's merge to `main` (#50) — and now
ACTIVATED. The machinery (§2) and constants (§3) are implemented in `evalue_fdr.py`
and pinned by `test_evalue_fdr.py` (oracle-validated against the `online-fdr` package
/ the papers). e-LOND is now the live FDR control in `run_structure_campaign` (BY
retained as a reported diagnostic): its first governed verdict — **0 / 24 cells
flagged** in the structure campaign — is pinned by `TestStructureCampaign`. The rule
still predates every number it governs: it was registered at #50 before the e-LOND
control judged anything. (The live campaign judges its 24-cell batch as the *head* of
the e-LOND stream — R = 0, t = 1, which is why the `1/(α·γ₁)` bar applies — because the
committed lifetime ledger remains empty until `--record`; §6 is the pre-registration
snapshot.)

**Date drafted:** 2026-06-19.

**Policy of record:** The loop controls false discovery with **e-values**, not
p-values. Each cell's existing Newey-West (HAC) t-statistic p-value is calibrated
to an e-value (a registered Vovk-Wang calibrator). The single FDR control is
**e-LOND** over the lifetime *stream* of cell-level e-values — one proven
procedure that handles the within-campaign cell correlation **and** the
across-campaign stream at once, at target FDR `α = 0.10`, under **arbitrary
dependence**, with online control at every step (run and judge campaigns
whenever). The guarantee is **exact in the dependence structure but inherits the
per-cell asymptotics**: e-LOND controls FDR only insofar as each calibrated
e-value is valid, which holds to the same asymptotic-normal approximation the
HAC-t p relies on today. Promotion stays **CLOSED**.

This is interlock #3b in its e-value (anytime-valid) form. It consumes the lifetime
ledger built in #3a ([docs/edge_search.md](edge_search.md)) and **reuses the
existing per-cell statistic**; it replaces the per-batch Benjamini-Yekutieli gate
as the FDR control. It supersedes the earlier option-(b) / BatchPRDS drafts of this
file.

## 0. Reader's guide — why e-values, why "staged," and what it costs

The structure campaign calls `benjamini_yekutieli` with `n` equal to one batch, and
that `n` resets every session — so it controls each batch's FDR but nothing across
the stream of campaigns. Acting after each campaign over a growing body of evidence
is the multiple-looks / interim-analysis leak.

E-values buy two things that are *proven theorems*, not assumptions: **online FDR
over an unbounded stream under arbitrary dependence** (e-LOND; Xu & Ramdas 2024),
so the loop can run and judge campaigns whenever with FDR controlled at every step
and **no across-batch independence assumption** (the one exposure the BatchPRDS
alternative could not discharge); and FDR control that needs **no harmonic `c(n)`
penalty** for the ≈ 0.8-correlated cells (e-BH; Wang & Ramdas 2022 — kept here only
as a within-campaign diagnostic, §2).

**Why "staged."** A cell's evidence rests on its daily P&L, which is serially
autocorrelated — the reason `short_vol_statistics` uses a Newey-West HAC t. The
purest e-value design rebuilds that per-cell test as a *betting test martingale*
(absorbing the autocorrelation exactly, nonasymptotically). This registration does
**not** do that. It keeps the HAC-t p-value the repo already ships and tests, and
*calibrates* it into an e-value. The multiple-testing layer then delivers the
arbitrary-dependence and online guarantees; per-cell validity stays exactly as
asymptotic as it is today. The full betting e-process is the deferred upgrade (§7)
— its cost is construction, not theory.

**What it costs — stated plainly, because it changes the story.** The e-value route
is **not more powerful** than BY; it is *less* powerful, and the gain is robustness,
not detection. Calibrating an asymptotic p into an e-value is lossy (a calibrated
e-value is conservative under a genuinely uniform null), so on the same data
calibrated-e-LOND rejects **no more than BY did, and typically fewer**. With the
registered `κ = 0.5`, a single top-ranked cell is rejectable only at e ≳ `n/α` —
i.e. `p ≲ 4×10⁻⁶`, far stricter than BY's ≈ `0.0011` rank-1 bar at `n=24`. **The
price for e-LOND's arbitrary-dependence, peek-whenever guarantee is a stricter
per-cell bar.** For a loop that has been 0/24 every campaign — where the binding
goal is *never to falsely promote*, not to squeeze out marginal discoveries — that
is the right trade, but it is a trade, not a free win.

## 1. What this registers (and what it does not)

**Registered:** the e-value FDR procedure (§2), the constants (§3), and the rule
that the only sanctioned way to change the procedure or its parameters is a new,
human-signed registration.

**NOT registered, and explicitly out of scope:** this does not promote anything; it
does not make the per-cell test nonasymptotic (the deferred betting e-process); and
its FDR guarantee, while exact in the dependence structure, is only as valid as the
per-cell calibrated e-value, i.e. asymptotic. A result still graduates only when a
survivor exists **and** clears this gate **and** passes a confirmatory step on a
post-training-cutoff time-axis holdout — which #3b does not deliver. Promotion stays
**CLOSED**.

## 2. The mechanism (procedure of record)

Two steps; the second is the entire FDR control. Each names a published procedure;
the exact recurrence is committed in the implementation and pinned by a test
against the cited reference (the `onlineFDR` R package / the papers' worked
examples).

1. **Per-cell — reuse + calibrate.** Keep `short_vol_statistics`' HAC-t p-value
   `p_i` unchanged. Calibrate to an e-value with an **admissible Vovk-Wang (2021)
   calibrator** from the family `f(p) = κ · p^(κ−1)`, `κ ∈ (0,1)` (decreasing in
   `p`, `∫₀¹ f = 1`); registered default `κ = 0.5`, i.e. `e_i = 1/(2·√p_i)`. The
   implementation asserts `∫₀¹ f ≤ 1`. A `measurement_invalid` cell gets `e_i = 0`
   — it enters the stream but `0` can never clear any threshold, so it counts yet
   can never be rejected, the e-value analogue of the `p = None` defense pinned in
   #46. **The calibrated e-value is only *asymptotically* a valid e-value**: the
   HAC-t p is asymptotically (not exactly) uniform under the null, so `E[e_i] ≤ 1`
   holds only to that approximation — the per-cell asymptotics the whole chain
   inherits. The betting e-process (§7) is what would make it finite-sample valid.

2. **The FDR control — e-LOND over the lifetime stream.** Treat **each
   `(template, ticker)` cell as one element of a single stream**, in committed
   arrival order (campaign commit-order in `idea_ledger.jsonl`, then a fixed
   within-campaign cell order). Run **e-LOND** (Xu & Ramdas 2024): cell `t` is
   assigned level `α_t = α · γ_t · (R_{t−1} + 1)`, where `R_{t−1}` is the number of
   discoveries so far and `{γ_t}` is a committed non-negative sequence with
   `Σ γ_t ≤ 1`; **cell `t` is flagged for a human iff its e-value `e_t ≥ 1/α_t`.**
   By Xu & Ramdas (2024, Thm 1 + Cor 1, which explicitly endorses calibrating
   p-values before e-LOND), this controls **online FDR ≤ α over the unbounded
   stream under arbitrary dependence** — covering the within-campaign cell
   correlation *and* the across-campaign stream in one object, with no committed
   look-timing and no independence assumption. This is the *only* FDR guarantee;
   there is no separate across/within composition to specify.

   **e-BH is a within-campaign diagnostic only, not the control.** The campaign
   summary may report the e-BH (Wang & Ramdas 2022) rejection set over the current
   batch's `n` cells (`reject top k* = largest k with e_(k) ≥ n/(k·α)`) as an
   FDR-controlled *view of that batch in isolation* — exactly as the retired BY is
   "kept as a diagnostic." It does **not** flag cells and is **not** part of the
   lifetime guarantee; only e-LOND flags.

## 3. The budget (committed numbers)

Implemented as pinned constants (named here so the registration is concrete):

- **`ONLINE_FDR_ALPHA = 0.10`** — the target FDR, carried from the per-batch
  `FDR_Q`. *Owner risk-appetite choice.*
- **`CALIBRATOR_KAPPA = 0.5`** — the Vovk-Wang calibrator exponent (`e = 1/(2√p)`).
  *Principled default*; a lever, not free power — smaller `κ` rewards very small
  p-values more but penalizes moderate ones (the calibration is lossy either way,
  §0). Committed, not tuned post-hoc.
- **`ELOND_GAMMA`** — the e-LOND discount sequence, committed as
  `γ_t ∝ 1/(t · log²(t+1))` normalized so `Σ γ_t ≤ 1`. *Principled default.* (Per
  Xu & Ramdas 2024, `{γ_t}` need only be non-negative with `Σ γ_t ≤ 1`;
  non-increasing is a sensible design choice, not a theorem requirement. The exact
  normalization constant is pinned by the implementation test.)
- **No campaign cap is needed for the FDR guarantee** — e-LOND controls the
  unbounded stream by construction. `K = 10` is retained **only as a governance
  re-registration checkpoint** (a research-plan trigger to stop and take stock),
  explicitly *not* a statistical device.

Mark `ONLINE_FDR_ALPHA` as the risk-appetite choice; the rest are principled
defaults.

## 4. What stays unchanged (load-bearing)

- **The sealed vault** (`STRUCTURE_SEALED = ('TLT',)`, sealed by omission) is
  untouched.
- **The manual graduation gate** is untouched. The harness surfaces survivors; it
  never crowns them. No auto-promotion.
- **The exploratory/registered line** holds. A campaign remains EXPLORATORY,
  sample-spending, kill-or-justify. This registers an *accounting policy*, not a
  finding.

## 5. Monitoring over time

- **Same data, re-run:** deduped by `_ledger_key` (`(lineage, phase, template,
  ticker, params)`) — zero new comparisons, zero budget spent. An off-schedule
  re-run on identical data cannot manufacture a fresh look.
- **New data, re-run:** a refreshed store or extended span is a new lineage, a new
  element in the e-LOND stream, and consumes the next discount increment `γ_t`.
  Because e-LOND controls online FDR with no committed look-timing, this is *safe
  at any cadence* — the peek-whenever property — rather than requiring a
  pre-declared schedule.

## 6. Current state (for the record, computed pre-registration)

At registration the operative per-batch `n` is **24** (structure) / 9 (re-tag).
The lifetime ledger is **empty** (`idea_ledger.jsonl` not yet populated). The
structure comparisons already spent and to be recorded on first `--record` are the
24-cell batch plus the 4-cell NVDA addendum = **≈ 28**, the first elements the
e-LOND stream will consume. These condition on no new outcome; they are the
existing, published 0/24 results. (The re-tag phase is namespaced by `phase` in the
ledger and runs a separate stream — §7.)

## 7. Open questions, deferred (NOT resolved here)

- **The full betting e-process (the per-cell upgrade).** Replacing the calibrated
  HAC-t p-value with a nonasymptotic betting test martingale (Waudby-Smith &
  Ramdas 2024) on each cell's daily P&L would make the per-cell e-value — and hence
  the whole chain — finite-sample valid, add within-cell anytime-validity
  (continuous monitoring), and recover some of the power calibration gives up (§0),
  at the cost of a new per-cell test and a registered P&L-bound parameter. It is the
  principled endpoint; staged is the deliberate first step.
- **DSR / PBO** belong at *graduation*, not in #3b.
- **The re-tag phase** (a different statistic / null) is namespaced by `phase` and
  runs a **separate** e-LOND stream by default; pooling would be a deliberate change.
- **`measurement_invalid` across lineages** — counted once per lineage (distinct
  rows by the lineage hash); flagged for sign-off.
- **Engine-version re-lineage.** Bumping `STRUCTURE_ENGINE_VERSION` re-lineages
  every comparison (a different engine is a different answer key); the
  implementation must surface the resulting change in the stream loudly in the log.

## 8. Amendments

`α`, the calibrator `κ`, the `{γ_t}` sequence, and the choice of e-LOND are fixed
for this registration's horizon. Changing any of them requires a new, human-signed
registration (a fresh dated entry amending this file), recorded before the
campaigns it governs. Hitting the `K = 10` governance checkpoint is a
re-registration trigger, not a death.

## 9. Lineage and references

- **e-LOND (online FDR under arbitrary dependence; calibrate-then-run endorsed):**
  Xu & Ramdas (2024), *Online multiple testing with e-values*, AISTATS / PMLR 238
  (Thm 1 + Cor 1).
- **e-BH (the within-campaign diagnostic; FDR under arbitrary dependence):** Wang &
  Ramdas (2022), *False discovery rate control with e-values*, JRSS-B 84(3).
- **p-to-e calibration:** Vovk & Wang (2021), *E-values: calibration, combination,
  and applications*, Ann. Statist. 49(3).
- **The deferred betting e-process (§7):** Waudby-Smith & Ramdas (2024), *Estimating
  means of bounded random variables by betting*, JRSS-B 86(1); anytime-validity via
  Ville (1939).
- **Reference implementation (validation oracle, not a dependency):** the
  `onlineFDR` R/Bioconductor package (Robertson, Liou, Ramdas & Wason 2019).
- **The within-batch dependence-robust offline procedure retired from the gate:**
  Benjamini & Yekutieli (2001), kept as a diagnostic. (Its per-cell input shares the
  finite-sample-overlap fragility this repo distrusts elsewhere — cf.
  `docs/prereg_trend_gate.md` §9, which replaces a t-formula with a placebo null;
  the betting e-process in §7 is the analogous upgrade here.)
- **The lifetime ledger this reads:** interlock #3a, `idea_ledger.jsonl` /
  `record_trials` / `_data_lineage_hash` (#48), narrated in
  [docs/edge_search.md](edge_search.md).
- **The registration discipline:** modeled on
  [docs/prereg_trend_gate.md](prereg_trend_gate.md).
