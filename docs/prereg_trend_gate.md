# Pre-registration: trend-gated covered-call experiment

**Status:** DRAFT — not yet registered. Registration becomes effective at the
merge commit of this file to `main`. No Stage 1 or Stage 2 computation may run
before that commit exists, and the analysis scripts implementing §5–§6 must be
committed before any Stage 1 number is produced. (The signal-side
characterization in §3.3 was computed before drafting; it conditions on no
outcome data.)

**Date drafted:** 2026-06-11.

**Question of record:** Does suspending covered-call selling while the
underlying is in an uptrend — holding the shares uncovered — produce a
positive call-selling residual, and does the trend signal select suspension
days better than structure-matched random gates?

---

## 0. Reader's guide — why this document exists

This registration doubles as a teaching example, so it is deliberately more
annotated than a minimal spec. The operative content — the rules an
implementer must follow — lives in §2–§6 and §10; the surrounding sentences
explain why those rules are what they are.

**Why pre-register at all.** A backtest experiment is a sequence of
choices: which signal, which span, which metric, which test. Each choice
made after seeing results is a fork that silently inflates the
false-positive rate (§2.3 does the arithmetic for one family of forks).
Writing every choice down first — and letting the git history prove the
ordering — is what makes the eventual p-value mean what it claims to mean.
The discipline is the same one this repo applies to code through pinned
regression tests: commitments first, then evidence.

**Why the inference is placebo-based.** Three simpler designs fail here.
Comparing the gated overlay to the ungated one fails because the ungated
call leg loses money, so any abstinence — a coin flip — "improves" it
(§1.3). Comparing to buy-and-hold alone fails because, with a few hundred
noisy cycles, a positive sum is easily luck. And a textbook t-test fails
because no formula captures the statistic's null distribution — entries are
path-dependent, cycles overlap, the tickers are correlated (§9). So the
design manufactures the null instead of assuming it: 1,000 fake gates,
statistically identical to the real one in every respect except alignment
with actual market trends — same suspension fraction, same clumping, same
full engine treatment (§5.1, §6.2). Like the sugar pill in a drug trial,
the fakes reproduce the entire ritual of gating except the active
ingredient; whatever the real gate achieves beyond their distribution is
attributable to the signal.

**Why two stages.** Stage 1 (§5) is a kill-gate: it checks the mechanism on
records that already exist, cheaply, and can only stop the experiment —
passing it proves nothing, because this sample also generated the
hypothesis. Stage 2 (§6) is the verdict, and it is deliberately expensive:
1,000 placebo gates × 3 tickers of full backtests to buy one honest
p-value.

The price of this design is stated in §9: with this much noise and this few
cycles, the modal outcome is "positive but inconclusive." The registration
exists so that sentence gets written even when a more exciting one is
available.

---

## 1. Hypothesis

### 1.1 The identity that makes it precise

`run_real_cc_overlay` never sells shares (ITM outcomes settle in cash), and
its buy-and-hold comparator uses the same share count and the same initial
cash. Therefore, for any entry gate:

> gated overlay beats buy-and-hold ⟺ the calls actually sold sum to
> positive P&L (including open-position mark-to-market and commissions),
> i.e. `net_overlay_pnl > 0`.

The hypothesis is a claim about that residual, not about equity curves.

### 1.2 Registered hypothesis (H1)

Calls sold only on non-uptrend days (per the primary signal, §2) have:

1. Pooled per-exposure P&L (the statistic `T`, §6.1) greater than zero,
   **and**
2. `T` significant against the record placebo family at the §6.3 rule —
   i.e. the trend signal beats structure-matched, skill-free suspension.

Both conditions are required. The proposed mechanism: an uptrend in force at
entry predicts the \~30-day right tail beyond the \~25-delta strike, which is
exactly the payoff region a covered call forfeits.

The algebra behind that sentence, at expiration with stock `S`, strike `K`,
and premium `c`: the shares are worth `S` in both portfolios and cancel, so
per cycle the overlay minus buy-and-hold is `c − max(0, S − K)` — positive
by the premium everywhere except finishes above the strike, where it
declines dollar-for-dollar without limit. A 25-delta strike is by
construction the level the market prices a \~25% chance of exceeding over
the option's \~30 days, and the premium is the market's charge for that
tail; H1 says the bet is conditionally mispriced when an uptrend is in
force at entry. One caveat connects the algebra to the engine: \~97% of
cycles end in early buybacks rather than expiration, so the forfeiture is
usually realized mid-cycle as a deep-ITM buyback loss — the same tail event
by another route (the pinned MSFT 10y decomposition: 54 deep-ITM buybacks
−\$611,302 against 122 profit-target wins +\$429,037) — which is why Test
B's terminal exceedance is only a kill-gate proxy (§5.3) and the verdict
re-runs the engine itself (§6.2).

### 1.3 What is explicitly NOT claimed or tested

- **"Beats the unconditional overlay" is not a claim.** The unconditional
  call leg is negative on these chains (MSFT −\$382,209 over 16y, QQQ
  −\$156,628 over 10y, both NW-insignificant; pinned in
  `tests/test_real_cc_backtest.py`). Any abstinence — a coin-flip gate — improves
  on it in expectation. It is reported as descriptive context only.
- **No claim at GFC scale.** The GFC era is excluded at load time
  (`CHAIN_CLEAN_START`). The spans do contain crash-and-rip episodes — the
  2020 COVID crash (SPY \~34% down in five weeks, recovered in \~5 months)
  and the 2025-04 correction, both named §6.4 stress windows — but nothing
  like 2008–09: a \~57% decline over 17 months, a recovery measured in
  years, and a long sequence of violent bear-market rallies for a call
  seller to be run over by. Claims are scoped to the regimes the sample
  contains.

### 1.4 House prior

The tutorial's "What We'd Add Next" (item 10, *Entry trend filter*) predicts
this filter **won't** help the call side: selling into downtrends is
desirable for call sellers (premium collected, cost basis reduced). The June
2026 IV-richness scan on the same chains (internal scan, unpublished) found
no exploitable conditioning channel and ended placebo-calibrated negative.
The registered prior is that H1 fails; this experiment is run to turn the
design choice into an empirical result either way.

---

## 2. Primary signal definition ("during an uptrend")

### 2.1 Definition of record

A trading day `d` is **suspended** (no new call may be sold) iff the regime
state knowable at the open of `d` is `'bull'`, where the state series is:

```text
classify_regime(unadjusted_closes, window=200, threshold=0.05).shift(1)
```

using `classify_regime` from `engine/cc_backtest.py` exactly as it exists at
registration: bull iff close > 1.05 × SMA200. The series is computed on the
**full** `*_10yr_prices_unadjusted.csv` file (the same series the engine
trades against), then shifted one day so day `d` uses only closes through
`d − 1`, then restricted to the §3.1 span. Days in `'bear'`, `'sideways'`,
or (out of analysis span by construction, §3.1) `'unknown'` states are
tradeable.

### 2.2 Why this definition, fixed before any outcome is viewed

- It is the classifier that generated the hypothesis: `regime_analysis` /
  figure 10's per-day P&L attribution (\~\$23 bull / \~\$303 bear / \~\$402
  sideways per day) uses this exact function.
- It already exists in the codebase with documented no-peek semantics; no new
  signal code is interpretable as tuned-to-result.
- The ±5% band requires a confirmed uptrend — close a full 5% above the
  SMA — before suspending, a stricter trigger than a bare SMA crossing.
- Unadjusted closes avoid the dividend-lookahead leak: dividend-adjusted
  series back-propagate future dividends into the SMA, a small but
  directional bias toward "uptrend" that was not knowable in real time.

### 2.3 Hard constraints

- **Entry-only.** The gate is evaluated only on days the engine is flat and
  would otherwise attempt an entry. It never triggers an exit, and the signal
  is never re-evaluated mid-cycle: a call open when the trend flips bullish
  rides to its natural exit (profit target, deep-ITM buyback, or expiration).
  This is pinned because a gate-triggered exit rule is the most tempting
  mid-experiment addition — an unregistered second degree of freedom. Any
  exit-rule variant is a new experiment, not robustness.
- **No promotion.** If the primary definition fails Stage 1, the experiment
  reports a null. Robustness variants (§8) are descriptive and can never be
  promoted to primary within this registration. Seven definitions each
  tested at a 5% false-positive rate carry roughly a 30% chance that at
  least one passes by luck; the registered error rate is honest only if a
  single pre-named definition decides the verdict.
- **Cycle attribution is by entry state.** A cycle's trend state is the state
  on its entry day, full stop. This differs from figure 10's close-date
  attribution (a cycle entered in a downtrend that rips bullish books its
  loss to the non-bull bucket here). If entry-state results are null, the
  experiment does not retreat to close-date or holding-day framing.

---

## 3. Data, spans, and configuration

### 3.1 Analysis spans

Per-ticker span = intersection of the clean chain span (`CHAIN_CLEAN_START`
clip, canonical + backfill files via `load_chain_store`) and the signal-warm
span (first day with a non-`unknown` shifted state). Per owner decision
(2026-06-11), **no new price data is fetched for QQQ**; its signal warms on
the existing file and the analysis window shrinks accordingly. The
alternative — fetching pre-2011 unadjusted closes to pre-warm the SMA,
legitimate because the QQQQ-era exclusion applies to option chains, not
prices — was considered and declined; recording it here forecloses a
post-registration data backfill.

| Ticker | Span start | Span end | Trading days | Warm-up note |
|---|---|---|---|---|
| MSFT | 2010-05-10 | 2026-04-10 | 4,005 | signal warm 2008-10-16 (pre-span) |
| SPY | 2010-12-01 | 2026-06-05 | 3,901 | signal warm 2008-10-16 (pre-span) |
| QQQ | 2012-01-06 | 2026-06-05 | 3,624 | signal warm 2012-01-06; chains clean from 2011-03-23, first \~9.5 months consumed by SMA warm-up |

Span ends are the engine's existing data-clipped ends — the chain end for
MSFT (its price file runs to 2026-06-05), the price end for SPY (its chains
run three trading days longer); QQQ's coincide. All arms (§4) of a given
ticker run on the identical span. "Trading days" throughout this document
means the ticker's §3.1 date series. The price files and chain stores are
git-tracked and therefore pinned by the registration commit; any data
refresh after registration is an amendment (§11).

### 3.2 Engine configuration (fixed)

- Engine: `run_real_cc_overlay` only. No proxy-engine run enters the verdict.
- Parameters: `call_delta` 0.25, `dte` 30 calendar days, `close_at_pct` 0.75,
  capital \$100,000 per ticker, no stop loss — the published baseline.
- Fills: bid/ask (the published convention) for the verdict; mid-fill is a
  robustness rerun (§8).
- Entry selection: `select_entry` unchanged (band `bid > 0`,
  `0.05 < delta < 0.60`).

Why fixed parameters rather than the walk-forward optimizer's picks: the
optimizer does not produce a single operating point — its choices vary by
window and by ticker (close-at-100% in 11/11 MSFT windows with modal delta
0.15; different splits on SPY and QQQ), and its in-sample ranking predicts
out-of-sample results weakly (the expanded-grid experiment measured a mean
rank correlation of \~+0.13). Embedding the optimizer in every arm would
also confound the question — a gate changes the trades in each training
window, hence the parameters picked, hence the test result, so a difference
vs placebo could come from signal information or from perturbing a noisy
search — while multiplying every placebo's cost by the size of the grid.
The published baseline predates this hypothesis, is pinned by the
regression suite, and was not chosen to flatter the gate; adopting the
optimizer's favorite configuration (0.15-delta, hold-to-expiry) post-hoc
would select the operating point most exposed to exactly the right-tail
events the gate claims to predict — a fork no placebo test could detect.
The gate-as-grid-axis question is reserved as a separate registration
(§10).

### 3.3 Signal-side characterization (computed pre-registration)

Treatment-assignment structure only — no outcome data was viewed. Computed
2026-06-11 from the unadjusted closes with the §2.1 definition:

| Ticker | Suspended (bull) | Sideways | Bear | Tradeable fraction | Suspension episodes | Episode length min/med/max |
|---|---|---|---|---|---|---|
| MSFT | 2,506 d (62.6%) | 23.9% | 13.6% | 0.374 | 69 | 1 / 3 / 531 |
| SPY | 2,207 d (56.6%) | 35.6% | 7.8% | 0.434 | 86 | 1 / 5 / 229 |
| QQQ | 2,510 d (69.3%) | 22.2% | 8.5% | 0.307 | 85 | 1 / 4 / 277 |

Pairwise daily bull-state agreement: MSFT/SPY 0.681, MSFT/QQQ 0.812,
SPY/QQQ 0.802. Pooled suspension fraction
`f* = (Σ suspended-in-span days) / (Σ span days) = 7,223 / 11,530
= 0.626453`.

Two design consequences, fixed now: (a) the band whipsaws — most episodes
are days long, a few are years long — so leave-one-out robustness is by
**calendar year**, not by episode (§6.4); (b) the per-ticker suspension
fractions differ enough (56.6% vs 69.3%) that a single shared placebo
sequence cannot match every ticker's exposure, so the primary statistic
weights tickers equally (§6.1) and placebo acceptance is on the **pooled**
fraction (§5.1).

---

## 4. Arms

1. **Baseline** — the unconditional overlay runs per ticker on the §3.1
   spans. (MSFT's 16y configuration is already pinned; SPY and QQQ baselines
   on these exact spans are new but signal-unconditioned.)
2. **Record arm** — the overlay runs with the §2 gate.
3. **Complement arm** — the overlay sells **only** on suspended (bull) days.
   If the trend signal carries directional information, the complement must
   land in the placebo distribution's left tail.
4. **Vol-ablation arm** — the overlay suspends when the prior day's 30-day
   rolling annualized volatility (`calc_rolling_volatility`, window 30,
   shifted one day) is below the per-ticker in-span quantile
   (`numpy.quantile`, method `'linear'`, over in-span days with a defined
   shifted vol) at the level equal to that ticker's exact bull fraction, so
   suspension fractions match the record arm by construction; suspend iff
   vol < the quantile. This threshold uses the full-span vol distribution
   and is therefore **not a tradable strategy — it is a diagnostic**:
   uptrends are correlated with low volatility, and this arm measures how
   much of the record arm's result is just the vol channel (which the June
   2026 IV scan found empty).
5. **Placebo families** — defined in §5.1 (sequence generator) and §6.2
   (engine re-runs).

---

## 5. Stage 1 — mechanism kill-gate

Stage 1 runs on the baseline arm's records and the price/chain series only.
It exists to kill a mechanism-free hypothesis cheaply; **passing it carries
no confirmatory weight** (same mined sample that generated the hypothesis).

### 5.1 Shared placebo-sequence generator (used by both stages)

All placebo inference in this document draws suspended/tradeable calendar
sequences from one generator, one RNG stream, defined here:

- **Master calendar:** the sorted union of the three §3.1 span date lists. A
  sequence is restricted to each ticker's span when applied; a date
  contributes only the ticker-day observations that exist on it.
- **Run-length multisets:** measured on each ticker's span-restricted
  shifted signal, binarized to suspended (bull) vs tradeable (non-bull).
  Boundary-censored first and last runs are included at their observed
  lengths. Bull-run and non-bull-run lengths are pooled across the three
  tickers into two multisets; lengths are trading-day counts, replayed as
  master-calendar trading-day counts.
- **One draw:** alternate suspended/tradeable runs, each length sampled with
  replacement from the corresponding multiset; the initial state is
  suspended with probability `f* = 0.626453`; the final run is truncated at
  the master-calendar end.
- **Acceptance:** a draw is accepted iff its pooled suspension fraction
  (computed by the §3.3 formula, sequence restricted to each ticker's span)
  is within ±5% **relative** of `f* = 0.626453`, i.e. in
  [0.595130, 0.657776]. Rejected draws are discarded. The band
  exposure-matches placebos to the real gate so the test isolates *which*
  days the signal picks from *how many* it picks — the §1.3 abstinence
  confound otherwise re-enters through the placebos.
- **Stream:** draws are taken sequentially from a single
  `numpy.random.default_rng(20260611)`. Accepted draws are kept in order.
  If fewer than 1% of the first 10⁶ raw draws are accepted, stop and amend
  per §11. **Stage 1 consumes the first 10,000 accepted sequences**
  (label re-tagging only, no engine runs); **Family R (§6.2) is the first
  1,000 of the same accepted stream** (engine re-runs). The reuse is
  deliberate: one machinery, one stream, shared code.
- **Degenerate draws:** in Stage 1, a sequence that produces an empty cell
  in a test's statistic (no bull-tagged or no non-bull-tagged observations)
  is replaced by the next accepted sequence past the 10,000, and the
  replacement count is reported. The Stage 2 analogue is in §6.2. On either
  stage, replacements exceeding 2% trigger an amendment (§11).

This label-randomization design puts the signal's persistence and the
cross-ticker correlation structure into the null by construction, which an
ad-hoc permutation or block bootstrap would have to approximate.

### 5.2 Test A — entry-state cycle split

Reconstruct cycles from the baseline runs by pairing each `sell` record with
the next terminal record (`expiration` / `close` / `close_itm`; unambiguous,
one position at a time). Drop the at-most-one cycle per run still open at
span end. Tag each cycle with the §2.1 state on its entry date. Use the
recorded terminal `pnl` field (not an expiration-payoff formula — \~97% of
cycles end in buybacks). ITM-at-expiry, where referenced, is
`expiration.price >= sell.strike` (the expiration record carries no strike).

Statistic: `D_A` = mean per-cycle `pnl` of bull-entry cycles minus mean of
non-bull-entry cycles, pooled across tickers in per-cycle dollars (every arm
runs at \$100,000 capital, so pooled dollars are implicitly contract-count
weighted; stated, accepted). Prediction: `D_A < 0` (bull entries do worse).

Inference: recompute `D_A,i` under each of the 10,000 placebo sequences
(re-tag the same fixed cycles by entry date against sequence `i`).
One-sided add-one p-value:

```text
p_A = (1 + #{i : D_A,i ≤ D_A}) / (1 + 10,000)
```

(The add-one form counts the real arrangement among the candidates — the
standard convention that keeps the test exact and prevents a reported p of
zero; §12 cites it. Every p-value in this document uses it.)

### 5.3 Test B — price-path exceedance

For each day `t` in each ticker's span on which `select_entry(day, 30, 0.25)`
returns a candidate **and** `t` + 30 calendar days does not fall past the
ticker's span end: let `K_t` be that candidate's strike and `x_t = 1` iff
the last close on or before `t + 30` calendar days **strictly exceeds**
`K_t` (deliberately strict `>`; the engine's ITM settle uses `>=` — ties at
round-number strikes are resolved against the hypothesis). Days within 30
calendar days of span end are excluded.

Statistic: `D_B` = pooled day-weighted exceedance rate on bull days minus
the rate on non-bull days, one point estimate across all three tickers.
Prediction: `D_B > 0`.

Inference: recompute `D_B,i` under each of the 10,000 placebo sequences,
holding the `(K_t, x_t)` series fixed (under the null the labels carry no
information, so the 30-day overlap structure of `x` is irrelevant and label
persistence is matched by construction). One-sided add-one p-value:

```text
p_B = (1 + #{i : D_B,i ≥ D_B}) / (1 + 10,000)
```

This is the higher-powered test in raw count (\~3,600–4,000 days per ticker
vs \~180–290 cycles), though the effective sample is far smaller — outcome
windows overlap, regimes cluster, and the three tickers are highly
correlated — which the placebo null absorbs by construction.

### 5.4 Gate rule

Stage 2 runs iff `D_A < 0` strictly, **and** `D_B > 0` strictly, **and**
`min(p_A, p_B) ≤ 0.10`. A point estimate of exactly zero fails its sign
condition. Otherwise the experiment ends with a null verdict and the
pre-committed language of §7. The thresholds are deliberately lenient —
Stage 1 is a kill-gate, not the verdict.

---

## 6. Stage 2 — verdict

### 6.1 Primary statistic

```text
T = (1/3) × Σ over tickers k of ( net_overlay_pnl_k / short_call_days_k )
```

computed on the record arm. `net_overlay_pnl` is the engine's
`final_equity − buy_hold_final` (includes open-position mark-to-market and
all commissions). Short-call-days = trading days (on the ticker's §3.1 date
series) with a call open, summed from the trade records: closed cycles count
entry date inclusive to terminal date exclusive; the at-most-one final open
cycle counts entry date inclusive through the final span date inclusive.

Tickers are weighted equally (the 1/3 average, not a pooled-dollar ratio) so
that the verdict is insensitive to ticker mix: a shared placebo sequence
gives every placebo roughly equal per-ticker suspension fractions while the
real arm's are 0.566/0.626/0.693, and a pooled-dollar statistic would let
that mix difference shift the placebo distribution's center. Per-exposure
normalization prevents a gate from "winning" merely by trading less.

### 6.2 Placebo families

**Family R (record) — the first 1,000 accepted sequences from the §5.1
stream, each pushed through full engine re-runs** on all three tickers
(path-dependent entry schedules are part of the null — re-tagging the
ungated cycle schedule is forbidden in Stage 2), yielding one placebo `T_i`
per sequence, plus one complement statistic `T_i^c` from the sequence's
complement gate. A sequence whose engine run errors, or whose runs produce
zero short-call-days on any ticker (making its equal-weighted `T_i`
undefined), is replaced by the next accepted sequence in the stream; the
replacement count is reported, and replacements exceeding 2% trigger an
amendment (§11). Per-sequence trade records are retained for the §6.4
leave-one-year-out recomputation.

Note on correlation direction: applying one shared sequence to all tickers
gives placebo cross-ticker agreement 1.0 vs the real signal's 0.68–0.81,
which inflates the variance of the placebo `T_i` — a conservative bias on
the spread, accepted ex ante. The §6.1 equal-weighting handles the
companion centering concern.

**Family S (secondary) — common circular shifts, 500 draws.** Each ticker's
own real signal is circularly shifted within its own span by a shared
offset, drawn as uniform integers on [250, 3,374] (250 to shortest span −
250) from `numpy.random.default_rng(42)`. Exposure and cross-ticker
structure match the real arm exactly; the known weakness is low effective
diversity (trend run lengths of 100–500 days mean perhaps 25–40 genuinely
distinct alignments), so Family S informs but cannot overturn Family R.

### 6.3 Pass rule (pre-committed, one-sided)

H1 **passes** iff, on the record arm with bid/ask fills:

1. `T > 0`, and
2. `p_R = (1 + #{i : T_i ≥ T}) / (1 + 1,000) ≤ 0.05` (the add-one Monte
   Carlo p-value over Family R).

Neither condition suffices alone: a positive `T` that sits mid-pack among
the placebos is what luck looks like, and a top-5% placebo rank with
`T ≤ 0` would only crown the least-bad of the skill-free gates — by §1.1's
identity, still a loss to buy-and-hold.

Any *mechanism* language ("trend predicts the forfeited right tail")
additionally requires all three of: Stage 1 Test B passed at `p_B ≤ 0.10`;
the complement arm in Family R's left tail,
`p_C = (1 + #{i : T_i^c ≤ T^c}) / (1 + 1,000) ≤ 0.05`; and the record arm's
`T` strictly exceeding the vol-ablation arm's `T`. Absent those, a pass is
reported as "the trend-defined suspension set outperforms random
equally-structured suspension sets" and nothing more. **Binding clause:** a
§6.3 pass in which the vol-ablation arm's `T` is greater than or equal to
the record arm's `T` must carry the qualifier "indistinguishable from a
low-volatility gate" on every surface that reports it.

### 6.4 Mandatory secondary analyses (reported with the verdict, whatever it is)

- Per-ticker `T` components and a same-sign tally are reported as
  descriptive context (three correlated secular-bull underlyings are \~1.2
  independent tests, not 3).
- A premium-normalized companion statistic is reported beside `T`:
  `(1/3) × Σ_k (net_overlay_pnl_k / total_premium_collected_k)`, real vs
  Family R, guarding the channel where non-bull days carry richer premium
  per day at risk.
- The record arm's realized fraction of short-call-days falling on bull
  days is reported (entry-only gating lets cycles ride into uptrends; the
  verdict language says "entry-gated," §7).
- Leave-one-year-out: for each calendar year `Y` with at least one pooled
  cycle (partial boundary years 2010, 2012, and 2026 included),
  `LOYO-T_Y = (1/3) × Σ_k [Σ cycle pnl over k's cycles with entry-date year
  ≠ Y] / [short-call-days of those cycles]`, where the final open cycle
  counts as a cycle with its entry year and its mark-to-market as `pnl`.
  The same recomputation runs on each retained Family R sequence's records,
  giving an add-one `p_{R,Y}` per removed year. (Removing cycles by entry
  year from a path-dependent run is an approximation — the remaining
  schedule still reflects the dropped year; stated, accepted.) Flagged ex
  ante: **2022** (the only clearly profitable full calendar year for the
  unconditional overlay on both pinned baselines; the gate trades through
  it — 2022 non-bull fractions are 0.948/0.956/0.972) and the **2026 stub**
  (strongly positive on the MSFT baseline). If dropping any single year
  flips the §6.3 verdict, the headline carries a "single-year-dependent"
  qualifier on every surface that reports it.
- Stress-window P&L is reported for 2020-03-23 → 2020-08-31 and
  2025-04-01 → 2025-06-30 — the known failure windows where a 200-day trend
  reads "down" while the path rips upward, so the gate sells into the
  recovery. Metric: record-arm equity change minus baseline-arm equity
  change over the window (window change = last close in window minus last
  close strictly before the window start), per ticker and pooled.
- The daily common-base excess Newey-West t on the record arm is reported:
  `(ΔE_gated − ΔE_bh) / E_bh,t−1`, which is zero by construction on
  uncovered days — avoiding the flat-day base-offset artifact in
  `compute_statistics` (its overlay and buy-and-hold returns are computed
  on different equity bases, so uncovered days contribute nonzero excess
  once the bases diverge). Lag: the Andrews rule as implemented in
  `compute_statistics` (`L = floor(4 × (n/100)^(2/9))`). Descriptive only;
  the placebo p-value is the verdict, chosen ex ante on power and validity
  grounds (§9).
- The Family S percentile is reported with its effective-N caveat stated.
- The vol-ablation arm's `T` is reported beside the record arm's, and the
  §6.3 binding clause applies.

---

## 7. Outcome language (pre-committed)

The wording below is fixed while the outcome is unknown, because after
results exist every author reads a p of 0.07 as "nearly significant." Each
row is the sentence that will be published, verbatim, in that case.

| Outcome | Registered language |
|---|---|
| Passes §6.3 | "Entry-gated call suspension survives placebo calibration on a mined 2010–2026 sample of three correlated secular-bull underlyings; untested at GFC scale (the 2020 and 2025-04 crash-and-rips are in-sample); confirmation requires a structurally different underlying, not more months of the same three." Plus the §6.3 binding clause if the vol-ablation arm matched it. |
| `T > 0` but `p_R > 0.05` | "Consistent with, not evidence for. The house prediction stands unrefuted." |
| `T ≤ 0`, or Stage 1 kill | "Null. The engine's no-trend-filter design choice is validated empirically (tutorial, What We'd Add Next, item 10)." |
| Complement arm in Family R's right tail, `(1 + #{i : T_i^c ≥ T^c}) / (1 + 1,000) ≤ 0.05` | A distinct finding (momentum-assisted call selling), requiring its own registration before any follow-up. Not a pass of H1. |

No result of this experiment supports trading decisions; the repo's standard
disclaimer applies.

---

## 8. Robustness grid (reported, never promoted)

Run only if Stage 2 runs. Each variant is reported beside the primary with
its own Family R rebuilt from that variant's run-length multisets (same
seed, same §5.1 procedure); none can change the §6.3 verdict. All signal
variants are computed on the full price file, shifted one day, and
restricted to the §3.1 span; a variant needing more QQQ history than the
existing file provides starts at its own warm date instead (no new data is
fetched). Suspension conditions, all evaluated at `d − 1`:

- The bandless variant suspends iff close > SMA200.
- The fast-SMA variant suspends iff close > 1.05 × SMA50.
- The crossover variant suspends iff SMA50 > SMA200.
- The 12-month variant suspends iff the trailing 252-trading-day total
  return on the **adjusted** close series is positive (the one deliberate
  exception to §2.2's unadjusted rule, since total return needs dividends).
- The 3-month variant suspends iff the trailing 63-trading-day return on
  the unadjusted close series is positive.
- The high-proximity variant suspends iff close ≥ 0.95 × the maximum of the
  252 closes ending at `d − 1`.
- The fill variant reruns the record arm at mid fills.
- The dose-response variant suspends on `bull` + `sideways` (trades only in
  bear), reported for shape against the primary's bull-only suspension.

Scope limit, stated openly: the grid varies the signal and the fill, never
the option parameters — there is no 0.15-delta or hold-to-expiry row — so a
§6.3 pass speaks only for the 25-delta / 30-day / 0.75-close operating
point. Each option-parameter row would cost its own Family R of full engine
re-runs; neighboring operating points are deferred to the gate-as-grid-axis
registration (§10).

---

## 9. Power / minimum detectable effect

An MDE calculation asks: given the sample size and noise the data will
offer, how large would a true effect have to be for this experiment to
reliably detect it? It is published before running so the eventual result
reads correctly in both directions — a null means "no effect this size or
larger," not "no effect," and an underpowered positive is recognized as
such rather than spun.

From pinned unconditional records (MSFT real-chain 10y: 122 profit-target
closes summing +\$429,037, 54 deep-ITM buybacks summing −\$611,302, 6
expirations −\$736): per-cycle mean \~−\$1,005 over 182 cycles. The
between-bucket spread of those means is \~\$6,700, which is a **floor** on
the per-cycle σ (within-bucket spread adds to it); realistic σ is plausibly
\$7,000–13,000. Expected record-arm cycle counts, assuming cycle rate is
uniform across states (\~18–20 calls/year unconditional × tradeable
fraction): MSFT \~109, SPY \~128, QQQ \~88 — pooled \~325.

At n ≈ 325, the pooled per-cycle mean needed to clear a conventional t = 2
(two-sided convention, conservative for this one-sided design):

| Per-cycle σ | SE of mean | Mean at t = 2 | Swing from −\$1,005 |
|---|---|---|---|
| \$7,000 | \~\$390 | \~+\$780 | \~\$1,800 (\~0.25σ) |
| \$10,000 | \~\$555 | \~+\$1,110 | \~\$2,100 (\~0.21σ) |
| \$13,000 | \~\$720 | \~+\$1,440 | \~\$2,450 (\~0.19σ) |

Per single ticker (n ≈ 90–130) the required swing is \~0.32–0.35σ. Caveats
in both directions: non-bull periods are higher-vol, so cycles may resolve
faster (more cycles, better power) while per-cycle dispersion is likely
above the floor (worse power). Two MDE artifacts are published with the
Stage 1 report, before any record-arm run: (a) the exact per-cycle σ and
MDE table recomputed from the baseline records (signal-unconditioned), and
(b) an implied **placebo-space** MDE — the spread of `T_i` over the first
100 Family R sequences pushed through engine re-runs (these remain the
first 100 of the 1,000; placebo runs do not unblind the record arm).

Consequence, fixed ex ante: the placebo p-value — not the t-statistic — is
the verdict, on power and validity grounds. Validity: no closed-form null
distribution exists for `T` — entries are path-dependent, option cycles
overlap, the three tickers are 0.68–0.81 correlated, and a single year can
carry the result — so a t-formula's independence assumptions fail where the
placebo simulation, which re-runs the engine under every null draw, makes
no such assumptions. The modal expected outcome is "positive but
inconclusive," for which the §7 language is already committed.

---

## 10. Implementation constraints

- **Gate seam:** the entry branch of `run_real_cc_overlay`
  (`realchains/real_cc_backtest.py`, the `if position is None:` block). A suspended day
  behaves exactly like an existing no-entry day (no chain row, empty band,
  or non-positive net premium): no trade record, shares held uncovered,
  daily equity row still appended, entry re-attempted the next tradeable
  day.
- **Pin protection:** the gate arrives as a new optional parameter
  (per-date suspension set) defaulting to off. Every pinned regression must
  pass byte-identical before any experiment run; no published number is
  re-pinned by this experiment.
- **No walk-forward.** The gate is not added to any optimizer grid in this
  registration; a gate-as-grid-axis study would be a separate registration.
- **RNG and seeds:** `numpy.random.default_rng`, one fresh generator per
  procedure — sequence stream seed 20260611 (§5.1), Family S seed 42
  (§6.2). No other randomness exists in the design.
- **Environment:** the repo `.venv` at registration — Python 3.9.6,
  numpy 2.0.2, pandas 2.3.3.
- **Ordering:** Stage 1 may not begin until this file's merge commit to
  `main` exists. The analysis scripts implementing §5–§6 must be committed
  before any Stage 1 number is produced. Results land in a separate, later
  PR that cites both the original registration merge commit (not any later
  amendment commit) and the analysis-code commit.

---

## 11. Amendments

Any change to this document after its registration merge — whether or not
any result has been computed yet — must be recorded in an "Amendments"
section appended here, with date, what changed, and why; every claim
affected by an amendment is demoted to exploratory. Silent edits void the
registration.

---

## 12. Lineage and references

- Internal: `classify_regime` / `regime_analysis` (the hypothesis source —
  figure 10's bull/bear/sideways per-day P&L); pinned unconditional results
  in `tests/test_real_cc_backtest.py` (`TestMsftExtendedSpanRegression`,
  `TestQqqRealChainRegression`, `TestMsftRealChainRegression`); the era
  exclusion in `load_chain_store` / `CHAIN_CLEAN_START`; the tutorial's
  "What We'd Add Next" item 10 (the house prediction this experiment tests).
- Method lineage: placebo-calibrated inference follows the data-snooping
  reality-check family (White, 2000, "A Reality Check for Data Snooping,"
  *Econometrica* 68(5)); the add-one Monte Carlo p-value convention is
  Davison & Hinkley (1997, *Bootstrap Methods and their Application*,
  Cambridge); walk-forward and sample-size discipline elsewhere in this
  repo follows Pardo (2008), not used in this experiment's verdict.
- Precedent: the June 2026 IV-richness scan on the same chains (internal
  scan, unpublished — mechanism absent; placebo-calibrated negative), which
  fixed this design's abstinence-confound and placebo-of-record
  requirements.
