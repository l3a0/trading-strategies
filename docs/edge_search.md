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
  fetch, out of MVP scope.

**Scope (MVP).** Templates that *re-tag existing naked cycles* only — cheap,
no engine re-runs. Structure-side ideas (roll rules, stop-loss, spread width)
change the trades themselves and need a full `run_real_cc_overlay` per
candidate; those are the expensive verdict phase, deliberately out of scope.
Two simplifications are named, not silent: the permutation null is the uniform
same-count shuffle (structure-preserving per-template nulls are a follow-up),
and BY rather than BH is used because the candidates are dependent.

## Architecture

The harness is a funnel — spend the sample cheaply on the left, gate expensive
confirmation on the right:

```text
enumerate → kill-gate → FDR ledger  ║  register → confirm on sealed vault → pinned verdict
```

Everything left of the `║` is automated and spends the sample; everything to
the right is a committed, manual act, on purpose.

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

---

## Campaign 1 — cheap entry-conditioning class — EMPTY (2026-06-13)

**The batch.** Nine candidates from three mechanism templates, swept on the
real MSFT + SPY chains (QQQ sealed), campaign seed 20260613, 1,000-draw
permutation null, BY at q = 0.10:

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
| cooldown | N=7 | +362 | no | 0.712 | −0.018 |
| cooldown | N=30 | +605 | no | 0.867 | −0.037 |
| cooldown | N=60 | +743 | no | 0.872 | −0.038 |
| cooldown | N=90 | +1,406 | no | 0.896 | −0.037 |
| up_trend | window=21 | +353 | no | 0.739 | −0.053 |
| up_trend | window=63 | −33 | yes | 0.486 | −0.096 |
| up_trend | window=126 | +245 | no | 0.667 | −0.114 |
| up_trend | window=252 | +958 | no | 0.880 | −0.102 |
| iv_rich | — | +748 | yes | 0.080 | −0.083 |

**The reading.** Two findings, both consistent with what the repo already
knew:

1. **Every up-move-conditioning template is wrong-signed.** All four cooldown
   horizons and three of four up_trend windows predict `D_A < 0` and deliver
   `D_A > 0` — entries after a rip or a rally *lose less*, not more. The lone
   sign-correct window (63 days) has `D_A = −33` (\~zero) at p = 0.49, i.e.
   noise. This is the third independent confirmation, now swept as a batch,
   that **conditioning call-selling entry on recent upward price action has the
   sign backwards** on these names (cf. the cooldown and trend-gate kills).
2. **The one suggestive candidate is the known confound, and it still fails
   FDR.** `iv_rich` is sign-correct (`D_A = +748`) and individually
   suggestive (p = 0.080) — but its `vol_confound` is negative (rich-IV
   entries sit in lower trailing vol, i.e. calm markets), the same low-vol
   confound the IV-richness scout pinned. And p = 0.080 is nowhere near the BY
   rank-1 threshold of \~0.0039 (= 0.10 / (9 × Σ 1/i)). The multiple-testing
   math, not a single p-value, is what empties the class.

**What this campaign settles.** The cheap entry-conditioning class, swept at
once under honest FDR control, contains no survivor. The productive search
moves to the structure side (roll/stop/spread templates that need engine
re-runs) and to structurally different underlyings (the sealed vault upgrade) —
both out of this MVP's scope, both the natural next phase.
