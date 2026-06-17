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
  window), at the cost of a harmonic penalty over plain Benjamini-Hochberg.
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

## Engine-re-run phase (designed; builds after the orthogonal chains publish)

The re-tag class above is cheap because it never changes the trades. The
structure-side ideas — roll rules, stop-loss, spread width, and the
delta-neutral short-vol / straddle / iron-condor strategies — *do* change the
trades, so each candidate needs a full `run_real_*_overlay` engine run rather
than a re-tag. That is the deferred-expensive phase. Its design is fixed even
though it is not yet built:

- **A second template class.** Each candidate runs an overlay on the target
  ticker and parameter setting, instead of re-tagging fixed cycles.
- **A HAC-t kill-gate with a closed-form null.** The score is
  `short_vol_statistics`'s Newey-West (HAC) t-stat on the daily rate-netted P&L,
  which has an asymptotic null — so unlike the re-tag phase there is *no*
  per-candidate permutation. The cost is N engine runs plus one
  Benjamini-Yekutieli pass over the t-stat p-values: the same ledger, a cheaper
  gate.
- **The seal rolls to a non-equity name.** With GLD / TLT / XLE / EEM
  onboarding, the strong vault is a structurally-different underlying the
  structure work never used; the plan seals **TLT** (bonds) and searches the
  equity/gold names. QQQ — the current re-tag seal — already appears in the
  structure-side cross-section, so it cannot seal this phase.
- **Graduation stays manual.** A survivor — e.g. the SPY short-vol call wing
  (+2.54) — earns a pre-registration and a manual sealed-vault confirmation,
  never an automated verdict. The harness surfaces survivors; it never crowns
  them.

It builds once the orthogonal chains are published (the seal and the runs both
need the data live), and it does not bend the cheap re-tag gate — it is a
parallel phase with its own kill-gate.

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
