# Wing-Premium Existence Diagnostic — Design

**Status: DESIGN. No measurement has been run. This document freezes every
definition before any outcome number is computed.**

**Epistemic class: exploratory measurement** — the exploration-log family
(kill-or-justify), *not* a registered strategy experiment. It runs no overlay,
makes no structure comparison, spends no e-LOND budget, and adds no
idea-ledger row. Its results pin via the standard three surfaces
(module, dataset-gated test, `docs/explorations.md` entry). A positive read
licenses a *registration*, never a strategy claim.

Date: 2026-07-19.

---

## 0. Reader's guide — why this measurement exists

Four conditioning ideas for the covered call died in sequence: the registered
trend gate (bull iff close > 1.05 × SMA200; killed wrong-signed, see
`docs/trend_gate_results.md`), the post-rip cooldown scout, the IV-richness
scout, and an illustrative session split showing "downturn" entries hold 69%
of the QQQ book's losses. Three further gate proposals came out of the same
discussion, and all three are the *same claim with different signs*:

1. **Sell calls only when the call wing is rich** (overreaction story: spikes
   in upside-insurance demand overprice melt-ups).
2. **Sell calls only when the call wing is cheap** (informed-flow story:
   elevated wings front-run continuation, so cheap wings mark safe states).
3. **Sell calls into vol spikes** (mean-reversion story: elevated premium +
   vol crush).

Each is a claim that the **call-wing risk premium is state-dependent** — that
the gap between what the wing *prices* and what subsequently *happens*
varies with the wing's own level. An option price is a probability-weighted
forecast plus a risk premium; "elevated wing IV" means the market has raised
its melt-up odds, and no gate can profit from that unless the risk-neutral
odds systematically exceed the physical odds in some observable state.

This diagnostic measures that gap directly — forecast vs. realization,
bucketed by state — **before** any strategy sample is spent. One measurement
adjudicates all three gates. The likeliest outcome, stated up front in the
house style: the wing is calibrated or the contrast is noise, the family
closes, and the value delivered is the closure.

**The price for this design is one honest disclosure:** the diagnostic is
itself a look at the data that conditions future designs. That look is paid
for once, by pinning the result whatever it says.

---

## 1. Object of measurement

For a fixed ~30-day tenor, per ticker and per non-overlapping option cycle:

- What the market charged for upside: the implied volatility of the ~0.25Δ
  call, and its spread over the at-the-money call.
- What subsequently happened: realized *upside* semivolatility over that
  option's life, and whether the wing strike was breached.
- Whether the forecast error (the wing premium) depends on the wing's own
  state at entry.

No positions, no P&L, no exits, no costs. This is a forecast-calibration
study of prices the strategy would have faced.

---

## 2. Data and span

- **Verdict tickers: QQQ and SPY.** Exploratory contrast: MSFT and NVDA
  (single-name wings face speculative call flow that index wings largely do
  not; the contrast is reported but carries no verdict weight).
- **Stores: the canonical call chains** (`{ticker}_option_dailies.csv` plus
  era backfills), loaded exactly as the engine loads them
  (`load_chain_store`, merged backfills, live `CHAIN_CLEAN_START` clip).
  Calls only — both legs of the measurement are calls, so the calls-only
  canonical SPY/MSFT stores need no puts merge. Span end: each store's last
  day (2026-06-05 for the majors).
- **Live hygiene boundary, deliberately.** This diagnostic touches nothing
  registered, so it uses the corrected live clips (`SPY 2010-05-17`,
  `MSFT 2010-05-10`; QQQ needs none — backfill starts 2011-03-23), not
  `REGISTERED_CLEAN_START`. Rationale: the frozen registered spans exist to
  protect published pins; a fresh measurement should use the best-known
  clean data.
- **Underlying closes and highs:** the split-adjusted as-traded OHLC files
  (`data/{ticker}_daily_ohlc.csv`) — the same price convention as the
  strikes.
- **Scope limit (frozen):** the 0.25Δ wing only. The deeper 0.10Δ melt-up
  wing is likely outside the canonical stores' fetch-time strike band; it
  would require promoting the full-chain personal archive to an analysis
  input through the standard onboarding pipeline (validation battery,
  checksum manifest, release publish — human-gated). That is a named
  follow-up, not part of this design.

---

## 3. Signal definition (frozen)

Computed **every trading day** t in the clean span (the daily series feeds
the conditioning percentile; outcomes are sampled per-cycle, §4):

1. **Expiry selection:** the listed expiration minimizing |DTE − 30|, ties
   broken toward the smaller DTE.
2. **Leg selection:** among that expiry's calls passing hygiene (bid > 0,
   quote midpoint used as the price), the ATM leg is the strike whose vendor
   delta is closest to 0.50 and the wing leg the strike closest to 0.25.
   Vendor deltas *select* strikes only — the placeholder-greeks era is
   already excluded by the clip, and vendor IVs are never used.
3. **IV back-out:** implied volatility from each leg's quote midpoint via
   the repo's `implied_vol` (the same machinery `structure_greek_signature`
   verification uses), at that leg's exact DTE and the structure engine's
   frozen risk-free convention. Dividend carry is not modeled; both legs
   share a tenor, so the *spread* below differences out carry conventions to
   first order (the small absolute-level bias is accepted and disclosed).
4. **Wing state:** `S_t = IV_wing − IV_atm` in annualized vol points. On
   equity indices this is normally negative (the smirk); "rich" means S_t
   high in its own history.
5. **Conditioning variable:** the **point-in-time percentile** of S_t within
   its trailing history — window of 756 trading days, expanding from a
   252-day minimum. Assignment at t uses only data ≤ t, so the bucket is
   information a real gate could have possessed. Quintiles Q1 (cheapest
   wing) … Q5 (richest wing).
6. **Failure handling:** a day where either leg fails hygiene, or `implied_vol`
   fails to converge, contributes no S_t observation. Round-trip guard: the
   backed-out IV must reprice the midpoint within $0.01 or the leg is
   treated as failed.

---

## 4. Outcome definition (frozen)

**Sampling: non-overlapping cycles**, mirroring how the strategy samples the
world and removing overlapping-window autocorrelation by construction.

- The first cycle enters at the first trading day with (a) a valid signal
  and (b) ≥ 252 days of S_t history for the percentile. Each subsequent
  cycle enters at the first valid trading day after the prior cycle's
  settlement. Entry attempts advance day-by-day past signal-failure days
  (skips are counted and reported). Expected yield: \~180 cycles per ticker
  over \~15 years.
- **Settlement date:** the last trading close on or before the expiration
  date (the engine's Saturday-expiry convention).
- **Primary outcome — the wing premium:**
  `P = IV_wing − RSV⁺`, where `RSV⁺ = sqrt(252/n × Σ max(log rᵢ, 0)²)` over
  the cycle's n daily close-to-close log returns, annualized vol points.
  P > 0 means the wing charged more than realized upside delivered.
- **Secondary outcome — probability calibration:** implied breach
  probability `N(d₂)` at the backed-out wing IV vs. the realized terminal
  indicator (settlement close > wing strike). The terminal read is
  primary-within-secondary (it is what settles a covered call); an
  intraperiod max-high breach indicator (any daily high > wing strike) is
  recorded as a path-risk read.

---

## 5. Hypotheses and committed statistics

**Primary statistic (two-sided):** Spearman rank correlation ρ between the
entry-day conditioning percentile of S_t and the cycle's wing premium P,
computed over all completed cycles per ticker. Continuous — the quintiles
are for reporting tables, not for the test.

| Label | Story | Committed directional read (secondary) |
| --- | --- | --- |
| H-rich | Overreaction: rich wings over-forecast | mean P(Q5) − mean P(Q3) > 0 |
| H-cheap | Informed flow: cheap wings under-forecast | mean P(Q1) < 0 |
| H-flat | Calibrated / state-independent | neither contrast clears the null |

**Verdict rule (frozen):** the diagnostic reads LIVE if and only if **both**
QQQ and SPY have placebo-p ≤ 0.05 on ρ (per §6) **with the same sign**.
Anything else — one ticker only, opposite signs, neither — reads H-flat for
gating purposes. MSFT/NVDA never enter the verdict.

**Decision table (pre-committed):**

| Outcome | Licensed follow-up |
| --- | --- |
| LIVE with ρ > 0 (rich wings over-forecast) | A rich-wing gate earns a full registration — its own doc, its own null. Nothing trades on this diagnostic alone. |
| LIVE with ρ < 0 (rich wings under-forecast) | The rich-wing gate is dead; the cheap-wing direction earns a registration *only if* mean P(Q1) > 0 (cheap wings still overpriced). |
| H-flat | The conditioning family closes for covered calls: rich-wing, cheap-wing, and vol-spike gates are all declined without spending strategy sample. Pinned as a null. |
| Secondary probability read contradicts the primary | Report both; the vol-calibration primary governs. A contradiction is itself pinned as a caveat on any follow-up. |

---

## 6. Null machinery (frozen)

The trend-gate placebo pattern, adapted to cycle sampling:

- Working series: the per-cycle conditioning percentiles `c₁…c_N` and
  per-cycle premiums `P₁…P_N`, in cycle order.
- **1,000 circular shifts** of the conditioning series against the outcome
  series in cycle-index space (offset drawn uniformly from 1…N−1), fixed
  seed `WING_PLACEBO_SEED = 20260719`. Each shift preserves both series'
  own autocorrelation while destroying their alignment; recompute ρ per
  shift.
- Placebo-p: the two-sided rank of the real |ρ| in the shifted distribution,
  `(1 + #{|ρ_shift| ≥ |ρ|}) / (1 + 1000)`.

---

## 7. Minimum detectable effect, published in advance

With \~180 cycles per ticker, the placebo-calibrated detection floor for ρ is
expected near |ρ| ≈ 0.15; in quintile-contrast terms, with per-cycle premium
noise plausibly 8–10 vol points, the detectable Q5−Q1 contrast is roughly
4–5 vol points. The exact per-ticker noise scale will be reported with the
results so the null reads as "no state-dependence ≥ X was detectable," not
"no state-dependence exists."

This floor is a feature, not an apology: a state-dependent mispricing
smaller than a few vol points could not survive spreads and hedging costs as
a gate. The detection floor and the economic floor sit at the same height —
whatever this diagnostic cannot see, a strategy could not monetize.

---

## 8. Robustness grid (pre-specified, non-verdict-bearing)

Reported alongside the primary, never able to change the §5 verdict:

- Wing delta targets 0.20 and 0.30 (bracketing the 0.25 primary).
- Percentile window 504 trading days (vs. the 756 primary).
- RSV⁺ replaced by full realized volatility RV (does the result depend on
  the semivariance refinement?).

---

## 9. Sanity checks and demotion rules

- **Round-trip pricing guard** (§3.6) on every backed-out IV.
- **Delta-targeting report:** the distribution of |vendor delta − target|
  actually achieved per leg; a median miss > 0.05 on either leg demotes
  that ticker to exploratory.
- **Coverage report:** cycles attempted vs. completed; completion below 80%
  demotes the ticker to exploratory.
- **ATM cross-check:** the extracted ATM IV series is correlated against
  vendor IVs on a sample of clean modern days — a rank correlation
  materially below 0.9 indicates an extraction bug, halting the run (this
  checks *our* code against an independent computation; vendor IVs still
  carry no measurement weight).

---

## 10. What this diagnostic is NOT

- Not a registered experiment: no strategy verdict of any kind can be
  claimed from it, in either direction.
- Not a campaign cell: it never touches `run_real_*` overlays, the
  idea ledger, or the e-LOND stream.
- Not a promotion path: every gate it could motivate requires its own
  registration with its own null before touching strategy sample.
- Not a deep-tail study: the 0.25Δ wing is the tradable wing of the CC
  book, not the 0.10Δ melt-up tail (§2 scope limit).

---

## 11. Build plan

- **Module:** `search/wing_premium.py` — daily signal extraction, cycle
  sampler, RSV⁺/breach outcomes, Spearman + placebo, quintile report.
  Pure measurement; reuses `load_chain_store`, `implied_vol`, and the OHLC
  files. Runtime: minutes (chain reads dominate; no engine passes).
- **Tests:** `tests/test_wing_premium.py` — an always-run synthetic layer
  (IV round-trip guard, non-overlap invariant, point-in-time property of the
  percentile — assignment at t must not change when future days are
  appended, placebo machinery on constructed data) plus a dataset-gated
  regression class pinning the decisive numbers per verdict ticker (ρ,
  placebo-p, quintile means, cycle counts).
- **Results surface:** a `docs/explorations.md` entry in the standard
  negative-results format (or its positive-read equivalent with the licensed
  next step).
- **Repo plumbing in the build PR:** `ci.yml` pytest bucket addition,
  CLAUDE.md symbol-regex additions, README file-table row.

---

## 12. Order of operations

1. This design doc merges (the definitions above are thereby frozen).
2. The build PR lands the module + synthetic tests, producing no
   measurement numbers.
3. The run executes once; decisive numbers are pinned in the dataset-gated
   test and the exploration-log entry in the same PR.
4. The §5 decision table governs what, if anything, happens next.
