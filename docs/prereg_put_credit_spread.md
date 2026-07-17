# Experimental design (pre-registration draft): selling put credit spreads on SPY

**Status:** DESIGN DRAFT — not yet registered. Registration becomes effective at
the merge commit of this file to `main`. No walk-forward run may execute before
that commit exists, and the analysis code implementing §5–§6 (the structure-engine
walk-forward driver and the unhedged-arm seam, §10) must be committed before any
number is produced.

**Date drafted:** 2026-07-17.

**Question of record:** Does a systematically sold SPY bull put credit spread —
short an OTM put, long a further-OTM put wing, defined risk, rolled continuously —
earn anything beyond the interest on its collateral and the equity exposure it
embeds, once its entry parameters and exit rules are chosen honestly out of
sample? And do the three optimization axes the strategy offers (entry parameters,
exit rules, both together) change that answer?

---

## 0. Reader's guide — why this document exists

This design answers three questions in one registered pipeline: **how the
strategy's parameters may be optimized** (§5), **how its exit criteria may be
optimized** (§6), and **what the null hypothesis must be** (§7). The operative
rules an implementer must follow live in §3–§7 and §10; the surrounding prose
explains why those rules are what they are, in the pattern of
`docs/prereg_trend_gate.md`.

**Why pre-register at all.** The put credit spread is the most seductive
backtest in retail options: on a mostly-rising index it produces a high win
rate, a smooth equity curve, and a large terminal P&L — and this repo has
already measured where that P&L comes from. The registered put-wing experiment
(`docs/prereg_vol_premium.md`) sold the naked −0.25Δ SPY put over the same span
this design will use: **88.5% win rate, +$155,853 net — of which $150,666 was
interest on cash and only $5,187 was alpha over cash** (NW t +0.09 net of
costs; `TestSpyShortPutRegression`). Every choice left open after seeing
numbers like that — which delta, which stop, which exit — is a fork that lets
the seduction through. Writing every choice down first is what makes the
eventual verdict mean something.

**Why this experiment is worth running anyway.** The repo's two prior
measurements of this structure were both *single cells at fixed parameters with
hold-to-expiry exits*: the campaign's committed cell (30 DTE / 0.25Δ short /
0.10Δ wing, hedged NW t **−0.91**) and a menu-walker cell (45 DTE / 0.30Δ /
0.10Δ, **+0.05**). Neither searched the parameter lattice, and neither touched
the Gap E exit seams — which on the naked short vol *did* materially reshape
risk (stop 2×: worst MAE −11.41R → −3.12R) without flipping any sign. The open
question this design closes: **can honest out-of-sample parameter-and-exit
selection find a corner of the committed lattice with a real edge, or is the
campaign's kill robust to optimization?** Either answer settles a question the
one-cell measurements could not.

---

## 1. What the repo already knows (the priors, all pinned)

- **The naked put wing is a registered null.** SPY −0.25Δ / 30 DTE
  delta-hedged short put: gross NW t +0.20, net-of-0.5bp +0.09; IWM +1.00 /
  +0.91; neither clears t = 2 (`TestSpyShortPutRegression`,
  `TestIwmShortPutRegression`; verdict language published in
  `docs/vol_premium.md`).
- **The hedged credit spread measured wrong-signed.** The campaign's committed
  SPY cell scored NW t −0.91 (one-sided p 0.8186) against a predicted +1; all
  seven tickers wrong-signed (MSFT −2.08 / SPY −0.91 / QQQ −0.72 / GLD −3.24 /
  XLE −2.74 / EEM −2.21 / NVDA −0.06; `test_credit_spread_all_wrong_signed`,
  `docs/edge_search.md` Widening 3).
- **The structure buys the expensive leg.** The spread's engine-verified skew
  signature is `long_rich`: the far-OTM long wing carries *higher* IV than the
  nearer-ATM short leg (`TestGrammarSignatureMatchesEngine`). A put credit
  spread sells the cheaper-IV leg and buys the richer one — by construction it
  has no skew-harvesting story.
- **Exit rules move risk shape, not sign.** Experiment 4 on the SPY short vol:
  baseline expectancy −0.5407R (n = 174); stop 2× improves it to −0.1848R and
  truncates worst MAE from −11.41R to −3.12R with P(ruin) at f = 2% falling
  0.992 → 0.835 — and every one of six variants stays negative
  (`TestSpyExitVariantExploration`, `docs/explorations.md`).
- **Entry-timing texture is fragile; the hedged measure is robust.**
  Experiment 2's 20-career entry jitter moved raw per-cycle expectancy across
  −0.58R…−0.03R (baseline at the 5th percentile) while the hedged NW t stayed
  inside its band (85th percentile) — per-cycle raw numbers are
  placement-fragile (`TestRandomEntryScout`).
- **Entry conditioning on IV richness is dead on these chains.** Ex-post VRP at
  the sold strike ≈ 0/negative; the one positive-looking split was a
  low-vol-regime confound (`TestIvRichnessScout`, edge-search Campaign 1).

**House prior: H1 fails.** Both ingredients of the spread are pinned nulls or
negatives, the skew signature is adverse, and the exit exploration's own
verdict sentence is "exit choice moves risk shape, not sign." This experiment
is run to turn "the campaign killed one cell" into "optimization cannot rescue
the family" — or to be surprised, on the record.

---

## 2. Hypothesis

### 2.1 The decomposition that makes it precise

An unhedged put-credit-spread program's raw P&L decomposes as:

```text
raw P&L = interest on cash (rf)
        + equity exposure (the spread's net positive delta: short_delta - wing_delta,
          +0.10 to +0.25 across the lattice, +0.15 at the committed cell)
        + the carry/vol residual (what selling the spread adds beyond the first two)
```

The first term is replicable with T-bills. The second is replicable with a
small SPY position. Only the third is an *options* edge, and the engine
already isolates it: `run_real_structure_overlay` with the `combined` daily
delta hedge plus `short_vol_statistics`' rf-netting produces a daily excess
series that is exactly the strategy's return over its cash-plus-delta-matched
replication (the Bakshi-Kapadia delta-hedged gain, rate-invariant by
construction). **The verdict of this experiment lives on that residual.** The
unhedged retail arm is reported beside it (§4), descriptively.

### 2.2 Registered hypothesis (H1)

The walk-forward pipeline of §5–§6 — entry parameters and exit rules selected
jointly, in sample only, on the committed lattice of §3 — produces a stitched
out-of-sample daily hedged-excess stream whose Newey-West t-statistic (Andrews
lag, one-sided) is **greater than 2**, net of the §3.4 frictions.

### 2.3 Mechanism clause (declared now)

A §8 pass earns *carry-premium mechanism* language ("the put spread harvests a
premium") only if both of the following also hold: (a) the fixed-defaults
anchor (§4, arm C2 — the committed cell forced through the identical
window-restart-and-stitch machinery as arm A, so the two differ only in
selection) has a positive stitched-OOS hedged-excess t, and (b) the
per-window winning `short_delta` is modal — the same value in at least 12 of
the 23 windows, with no-winner windows (§5.2) counting against modality. A
pass without them is reported as "walk-forward selection found a profitable
corner of the lattice" — a selection result, not a mechanism — and is scoped
accordingly.

### 2.4 What is explicitly NOT claimed or tested

- **No "beats buy-and-hold" claim.** A short-put-spread program and long SPY
  hold different risks; their raw comparison is reported context, never a
  verdict.
- **No "beats T-bills" claim.** The verdict statistic nets the engine's actual
  rf credit; the financing lens is `docs/vol_premium.md`'s, not this test's.
- **No GFC-scale claim.** The span (§3.3) starts 2010-12-01; 2008–09 is
  untestable on these chains. The span contains the 2020 COVID crash, the 2018
  and 2025 vol events, and the 2022 bear — nothing like a \~57%-drawdown
  crash-insurance payout. **A crash-insurance seller's worst regime is absent
  from the sample**, and every surface reporting a result must say so.
- **No intraday claim.** Stops are stop-markets evaluated on daily closes
  (§6.4); assignment/pin risk and American early exercise are not modeled
  (§6.5).
- **No sizing claim.** Gap C+B sizing output is descriptive risk accounting,
  never advice.

### 2.5 Why SPY, and what "new sample" means here

SPY's put chains have been spent repeatedly — the registered put wing, both
ledger spread cells, the §7 straddle secondary, and the campaign's
iron-condor cell — so SPY cannot pretend to be naive. The design accepts this openly: the
walk-forward's out-of-sample discipline controls *parameter-selection*
snooping within the span, not *hypothesis-selection* snooping across the
repo's history. The genuinely naive confirmation is the IWM arm (§4, arm D) —
both wings exist in `iwm_option_dailies.csv`, and IWM has never run a spread.

---

## 3. Strategy space and frozen conventions

### 3.1 Entry lattice (18 cells, frozen — the committed grammar values)

Exactly the `ALLOWED_GRID` credit-spread lattice, unchanged:

| Axis | Values |
|---|---|
| `dte` (calendar days) | 21, 30, 45 |
| `short_delta` | 0.20, 0.25, 0.30 |
| `wing_delta` | 0.05, 0.10 |

Selection per `select_credit_spread`: short = `select_put_entry`'s band
(`bid > 0`, −0.60 < δ < −0.05), nearest-DTE expiration, nearest-|δ| put; wing =
same expiration, strike strictly below the short, buyable ask, nearest-|δ|.
Entry guard `net_positive` (credit after commissions > 0). Re-entry on the next
chain day after a cycle terminates (the engine's one-day-minimum gap).

### 3.2 Exit lattice (7 variants, frozen — the Gap E seams)

| Variant | Knobs |
|---|---|
| `hold` | none (hold to expiry — the campaign convention, the baseline) |
| `target50` | `close_at_pct = 0.50` |
| `target75` | `close_at_pct = 0.75` |
| `stop2x` | `stop_loss_mult = 2.0` |
| `stop3x` | `stop_loss_mult = 3.0` |
| `dte21` | `exit_dte = 21` (invalid for 21-DTE entries, §3.5) |
| `bracket` | `close_at_pct = 0.50` + `stop_loss_mult = 2.0` (the practitioner default) |

Semantics are the engine's, byte-frozen: target fires when the ex-commission
close cost ≤ credit × (1 − pct); stop when close cost ≥ credit × mult; time
when calendar days to expiration ≤ `exit_dte`; same-day priority
target > stop > time; triggers evaluate only on days every leg has a live
quote; closes fill shorts at the ask and longs at the bid plus per-leg
commission. `manage_deep_itm` stays OUT of the lattice (its structure-side
semantics are a pinned deferral; adding it is a widening, not a variant).
One frozen guard on `dte21`: because `select_put_entry` picks the *nearest*
available expiration, a 30-day target can land at an actual DTE ≤ 21 on
thin-expiration days, and `exit_dte = 21` would then fire on the first
quoted day — a 1–2-day churn cycle paying four commission legs plus spread.
A `dte21` cell therefore skips any entry whose actually-selected short DTE
is ≤ 22; this is part of the variant's definition, not a post-hoc repair.

### 3.3 Data and span (frozen)

SPY calls + merged puts (`spy_option_dailies.csv` + `spy_option_dailies_puts.csv`),
clipped to **2010-12-01 → 2026-06-05**; the driver's day grid is the merged
store's call days restricted to that window: \~3,900 trading days, \~175
hold-to-expiry monthly cycles. The start date is **this registration's own
frozen choice**, not `REGISTERED_CLEAN_START`'s authority (that constant
freezes *already-registered* work; new work defaults to the live
`CHAIN_CLEAN_START['SPY'] = 2010-05-17`). The price of the choice is named:
\~6.5 months of validated-clean call chains are forgone — and they are
vacuous here, because the puts file only begins 2010-12-01, so no spread can
enter before that date anyway; clipping there removes put-less rf-only lead-in
days and makes the span identical to the registered put wing's, which §7.1's
scale comparison relies on.
IWM (arm D) uses its own store (both wings, clean from row one, same span
convention). The chain stores are release-pinned; any data refresh after
registration is an amendment (§11). No new SPY fetch enters this experiment
(the full-chain archive fetched 2026-07-17 is a backup artifact, not an
analysis input — using it would move `entry-band` selection off the pinned
store and is foreclosed here).

### 3.4 Engine configuration and frictions (frozen)

- Engine: `run_real_structure_overlay` via the `credit_spread` spec —
  `entry_guard='net_positive'`, `management='hold'` plus the §3.2 exit knobs,
  capital $100,000, `num_contracts = capital // (price × 100)`, rf 0.045
  (verdict is rate-invariant), fills `bid_ask` with $0.0065/share commission
  per leg.
- Verdict arm hedge: `combined` daily delta hedge at **0.5 bp** half-spread
  (the spec's committed default — the same level the call wing survived and
  the put wing died at). The 0 / 0.2 / 1 bp curve is reported beside it.
- Per-window capital restarts at $100,000 (the `walk_forward_real`
  convention); OOS windows are stitched into one daily stream by the frozen
  §5.5 rule.

### 3.5 The joint lattice, counted

18 entry × 7 exit = 126, minus the 6 invalid (21-DTE entry × `dte21` exit)
= **120 valid cells** per training window. The count is fixed here so the
hypothesis space is finite and pre-specified — the same closed-grammar
discipline the campaign enforces, applied to a registered experiment. The
walk-forward selects among the 120 in sample; **no cell is individually
tested** (§7.4).

---

## 4. Arms

1. **A — verdict arm (hedged).** The §5 walk-forward on the 120-cell lattice,
   `combined` hedge, 0.5 bp. The §2.2 statistic lives here and only here.
2. **B — retail arm (unhedged).** The identical walk-forward selections
   replayed with the hedge off (`hedge_mode='none'`, §10): raw equity curve,
   win rate, max drawdown, and the scoreboard vs (i) cash at rf and (ii)
   buy-and-hold SPY. Descriptive only — this is the arm a retail reader
   recognizes, and the arm the §8 binding clause disciplines.
3. **C1 — drift alarm.** The campaign's committed cell (30 / 0.25Δ / 0.10Δ,
   hold, hedged) re-run once at the campaign's *exact* coordinates — the live
   `CHAIN_CLEAN_START['SPY'] = 2010-05-17` clip, `STRUCTURE_END = 2026-06-06`,
   the campaign loader's puts merge and call-day window. It must reproduce
   NW t **−0.91 ± 0.02** (the `portfolio_scout` tolerance convention) before
   any other number is read. Note the span: the −0.91 was *not* measured on
   the §3.3 span, so the alarm reproduces it where it lives; the \~137 extra
   lead-in days are put-less rf-only days.
4. **C2 — fixed-defaults anchor.** The committed cell forced through the
   §5 machinery as the winner of *every* test window — same restarts, same
   stitching, same statistic as arm A, differing only in that no selection
   occurs. Its stitched-OOS t feeds the §2.3 mechanism clause and is the
   selection-vs-defaults comparator. It carries no reproduction requirement
   (it is a new number on a new construction). Run hedged and unhedged.
5. **D — IWM confirmation.** The identical §5–§6 pipeline on IWM, run
   whatever SPY does. Confirms iff its stitched OOS hedged-excess t > 2.
6. **E — random-entry ensemble.** Gap F's selector seam ported to the spread:
   **20 careers, `k = 10`, seed 20260717**. Each career replays arm A's
   *realized per-window winning cells* — selection is NOT re-run per career —
   jittering only the entry calendar via the Gap F emission-keyed wait inside
   each test window, hedged at 0.5 bp, stitched by the §5.5 rule, scored by
   the same NW t. Pre-committed reads: a §8 pass whose verdict t exceeds all
   20 careers is reported "placement-fragile — the pass does not survive
   entry-calendar jitter"; a pass with ≥ 10 of 20 careers also above t = 2 is
   "placement-robust"; anything between is reported without qualifier
   (indeterminate at n = 20). Career exposure dilution (\~k/2 chain days lost
   per cycle) biases career t low and therefore biases toward the *fragile*
   read — accepted ex ante as the conservative direction.

---

## 5. Parameter optimization (the answer to "how may parameters be optimized?")

### 5.1 Walk-forward, not full-span search

A full-span grid search over 120 cells with a reported best is 120 hypotheses
wearing one t-statistic — the exact fork the campaign's FDR machinery exists
to control, re-created by hand. The registered alternative is walk-forward:
parameters are chosen only from data that precedes their use, the choice is
re-made every window, and the *only* judged object is the stitched
out-of-sample stream a real operator could have earned. Precedents in this repo
temper expectations honestly: on the SPY real-chain walk-forward, the
expanded-grid experiment measured a small positive in-sample-rank → OOS-rank
correlation (\~+0.13 — measured pre-correction and explicitly NOT pinned; the
test instructs re-derivation before publishing), and the pinned SPY
walk-forward scoreboard lost to buy-and-hold (+149.55% vs +190.67% over 24
windows) — walk-forward is the honest harness, not a magic one.

### 5.2 Window arithmetic (frozen)

Train **4 years**, test **6 months**, roll **6 months** — the
`TestMsftRealWalkForwardRegression` precedent: 4 years is the smallest integer
train span in which the slowest cell (45-DTE, hold) clears the Pardo \~30-trade
floor (\~32–34 cycles; the MSFT leanest-fit pin measured 33). First train
window 2010-12-01 → 2014-12-01; **23 OOS windows, 2014-12-01 → 2026-06-01**
(\~11.5 OOS years; the four trailing days past the last full window are
unused). A cell with fewer than 30 trades in a training window is
disqualified from selection in that window; `min_trades` is evaluated on the
engine's entry count (`num_sold`, the `walk_forward_real` convention).
Early-exit variants raise cycle counts (stop 2× produced n = 243 vs 174 on the
short vol), so the floor binds hardest exactly where it should — on the slow
hold-to-expiry cells. **A window in which no cell qualifies trades nothing**:
its OOS days enter the stitched stream as zeros, it counts in every
per-window denominator (including §2.3(b)) as a no-winner window, and it is
reported in the winner table as SKIPPED — the branch is fixed here so it
cannot be chosen after seeing which windows are affected.

### 5.3 Selection rule (frozen, one metric)

Within each training window, each qualified cell is scored by the
**annualized Sharpe of its daily hedged-excess stream**, computed unrounded
from the same rf-netted excess array the verdict uses — not from
`short_vol_statistics`' rounded summary fields (identical window lengths make
this rank-equivalent to the naive t; it is the `walk_forward_real` convention
adapted to the excess stream). Training and test runs both execute at the
§3.4 frictions (bid/ask fills, 0.5 bp hedge) — no friction asymmetry between
selection and verdict. The argmax wins the window; ties break by lattice
order (entry axes in §3.1 order, then exit variants in §3.2 order —
deterministic, boring, stated). The winner trades the next 6 months out of
sample. No mid-window re-selection.

Why the *hedged* excess drives selection, and not the simpler unhedged
Sharpe: (1) selection must optimize the object the verdict grades — a
pipeline that picks by one metric and is judged on another leaves a null
ambiguous between "no edge" and "objective mismatch." (2) Unhedged Sharpe is
a beta contest: the unhedged daily P&L is dominated by
(short_delta − wing_delta) × SPY return plus rf (the put wing's
$150,666-of-$155,853 interest share is the pinned demonstration), so
in-sample ranking by it mostly ranks cells by delta × window direction —
near-unforecastable, the worst case for IS→OOS transfer — while the hedged
residual is the one component that could persist. (3) The span crosses \~0%
and \~4.5% rate regimes; rf-netting keeps windows commensurable. A cell that
shines unhedged but is flat hedged is replicable with shares and T-bills —
no reason to pay four legs of spread for it. The scope cost is conceded
openly: exit behavior differs by hedge state (§6.2's CC-whipsaw vs
hedged-truncation contrast), so exit conclusions here are scoped to the
hedged measurement object; arm B's unhedged replay makes the gap visible,
and optimizing the unhedged retail book on its own terms would be a
different registration.

### 5.4 What is reported (diagnostics, never verdicts)

Per-window winner table (including each window's deployed notional, §9);
per-axis modal stability (the fig7-style read: does `short_delta` stabilize
even when the exact triple churns?); walk-forward efficiency (OOS/IS
retention, against Pardo's \~two-thirds lore, hedged as lore); `n_below_30`
per window; the count of windows where `hold` itself won (the
exits-add-nothing diagnostic).

### 5.5 The stitched stream (frozen construction)

The verdict object is built exactly one way: **the concatenation, in window
order, of each test window's per-day rf-netted excess array**
(`short_vol_statistics`' excess series — `np.diff(equity)/capital` minus the
engine's actual per-day rf credit), with SKIPPED windows contributing zeros
(§5.2). Never diff across window boundaries — the $100K restarts would inject
seam spikes. Two seam accounting rules, fixed now: (a) each window's final
open structure is charged a **synthetic close on the window's last day** at
bid/ask plus per-leg commission (no window ends holding an unpriced
liability); (b) the one-day entry-mark omission inherent in the diff (its
docstring's known caveat) occurs once per window — the accumulated bound
(≤ 23 entry spread marks) is computed and reported beside the verdict rather
than silently accepted, because its sign is in the pass direction.

---

## 6. Exit-criteria optimization (the answer to "how may exit criteria be optimized?")

### 6.1 Exits are lattice axes, not a second pass

The exit rules enter as the 7 variants of §3.2, optimized **jointly** with the
entry parameters inside the same §5 walk-forward. A staged design (fix entry,
then tune exits, or vice versa) is two searches wearing one accounting — the
order of the stages is itself a fork, and the interaction the practitioner
lore cares about (wider wings want wider stops; 45-DTE entries want the 21-DTE
time exit) is exactly what a staged search cannot see. The price of the joint
design is the larger lattice (120 vs 18), which §5.2's trade floor and §7.4's
single-verdict rule absorb.

### 6.2 The pre-stated expectation (from Experiment 4)

The pinned exit exploration says exits reshape risk without flipping signs:
stops truncate the MAE tail at expectancy cost or benefit depending on whether
the book is hedged (the CC's unhedged stop was whipsaw machinery, t −1.58 and
monotonically worse; the hedged short vol's 2× stop improved expectancy
−0.54R → −0.18R). The spread sits between those precedents — defined-risk like
neither. The design therefore pre-commits the reading: **if the verdict fails
but exit variants reproduce the shape pattern** (truncated worst MAE-R, lower
P(ruin) at fixed f, expectancy still ≤ 0), that is reported in Gap A ledger /
Gap C+B sizing terms as risk-shaping — explicitly not an edge, not a
near-miss, not "promising."

### 6.3 How exit optimization is judged

There is no separate exit verdict. The single §2.2 statistic judges the whole
pipeline; exits earn their place only by improving the *selected* stream out
of sample. Three exit-specific diagnostics are reported: the exit-reason
composition of the OOS trade ledger (`target` / `stop` / `time` counts — the
engine's frozen taxonomy); the **entry-only ablation** (the pipeline
restricted to the 18 hold cells — parameters optimized, exits off); and the
**exit-only ablation** (the pipeline restricted to the 7 exit variants at the
committed 30 / 0.25Δ / 0.10Δ entry — exits optimized, parameters fixed).
Together with the primary the three runs answer the question of record's
three axes — entry, exit, joint — each through the same window arithmetic,
each reported beside the primary, none promotable over it.

### 6.4 The measurement honesty rails (carried verbatim from Gap E)

The stop is a stop-market evaluated once per day on the close quote —
gap-throughs fill at the day's quote, not the stop level, so measured stop
costs flatter the stop. Triggers under-fire relative to a live book (a day
where one wing does not print cannot close). Rolls carry a minimum one-day
re-entry gap. All three ride with every reported number. One structural
mitigant is genuine and stated: **the spread's max loss is capped at
(width − credit) by construction**, so the EOD approximation error on any
single cycle is bounded — materially better measurement conditions than the
naked put's unbounded gap risk.

### 6.5 What exit optimization cannot see here

American early exercise and assignment (the engine settles at expiry
intrinsic; a deep-ITM short put carrying early-exercise premium near ex-div
is not modeled); pin risk at expiration; intraday stop fills. The \~2-year
intraday options window on the connected Massive/Polygon source could bound
the EOD-vs-intraday stop gap in a *future* calibration exercise; it is out of
scope here and named so the omission is a recorded choice, not an oversight.

---

## 7. The null hypothesis (the answer to "what should the null be?")

### 7.1 The economic null

**H0: the strategy earns nothing beyond its replication — cash at the
risk-free rate plus its own equity delta.** Equivalently: E[daily hedged
excess] ≤ 0. Two tempting nulls are rejected ex ante:

- *"Mean strategy return = 0"* is wrong because an unhedged spread seller
  collects rf on \~$100K of cash plus \~+0.1–0.2 delta of equity premium —
  both positive in expectation over this span and both available without
  touching an option. Under that null the strategy "works" before the options
  add anything; the registered put wing's $150,666-of-$155,853 interest share
  is the pinned demonstration.
- *"Underperforms buy-and-hold"* is wrong in the other direction: it charges
  the strategy for holding less equity risk than 100% SPY, which is a risk
  preference, not an inefficiency.

The hedged-excess formulation prices both confounds out mechanically, is
rate-invariant (the rf-netting), and is the same measure every pinned
short-vol number in this repo already uses — so the verdict lands on a
comparable scale to +2.25 (call wing at the same 0.5 bp friction; +2.54
gross), +0.09 (put wing), −0.91 (this structure, one cell).

### 7.2 The statistical test (frozen)

One-sided Newey-West t on the stitched OOS daily hedged-excess stream, Andrews
lag `L = 4(n/100)^(2/9)` via `newey_west_summary`, asymptotic p =
`erfc(t/√2)/2` (the repo's single asymptotic-p convention). **Pass iff
t > 2** (strict; a t of exactly 2.00 fails). The naive t, Sharpe, lag, and
the §7.3 block-bootstrap companion are reported beside it.

### 7.3 Structural nulls reported beside the parametric one

- **Random-entry band (arm E):** the verdict t located in its 20-career
  jitter distribution, with the §4 arm-E pre-committed qualifier rules
  (above-all-20 ⇒ "placement-fragile"; ≥ 10 of 20 above t = 2 ⇒
  "placement-robust"; otherwise no qualifier).
- **Stationary block bootstrap** on the OOS excess stream — expected block
  length **21 trading days** (the monthly cycle), **B = 10,000**
  replications, seed fixed in the analysis script at commit time, add-one
  p = `(1 + #{i : mean_i ≤ 0}) / (1 + B)` — guarding the HAC assumptions on
  overlapping cycles; the `prereg_vol_premium` §7 pattern, reported never
  verdict-bearing.
- **Leave-one-year-out** on the OOS stream: if dropping any single calendar
  year moves the verdict across the t = 2 line, every reporting surface
  carries a "single-year-dependent" qualifier.

### 7.4 Multiplicity accounting (stated, not hand-waved)

This registration runs **one test**: the pre-committed pipeline's single OOS
statistic (plus arm D's one confirmation statistic on IWM). The 120-cell
lattice is selection machinery *inside* the pipeline — cells are never
individually judged, so no per-cell FDR spend occurs and nothing enters the
e-LOND ledger (this is a manual registered experiment in the trend-gate /
VRP lineage, not an automated campaign batch). The two SPY spread cells
already in `idea_ledger.jsonl` remain the family's exploratory record. If
this experiment passes, promotion still runs through the human gate: a pinned
registered result and, if the family re-enters the automated search, a
grammar widening — never a silent upgrade of this document.

---

## 8. Outcome language (pre-committed)

Fixed while the outcome is unknown. Each row is the sentence published
verbatim in that case.

| Outcome | Registered language |
|---|---|
| SPY t > 2 and IWM confirms | "Walk-forward-selected put credit spreads harvest a premium beyond cash and delta on SPY, replicated on a naive index (IWM); scoped to a no-GFC span with daily-close exits. Promote to a registered pin; the §2.3 mechanism clause applies as met or unmet, verbatim." |
| SPY t > 2, IWM does not confirm | "A cost-surviving SPY result that does not replicate out of sample — treat as index-specific selection until a third index exists. Not confirmed." |
| 0 < t ≤ 2 | "Consistent with, not evidence for. The house prior stands unrefuted; the campaign's kill is unchallenged at this power." |
| t ≤ 0 | "Null. Joint entry-and-exit optimization does not rescue the put-credit-spread family on these chains; the campaign's one-cell kill generalizes to the optimized lattice." |

**Binding clause (the repackaged-beta clause):** any surface reporting arm
B's raw scoreboard must state, in the same breath, the decomposition — rf
interest, delta P&L, and hedged residual — whenever the raw P&L is positive
and the residual is not. A high win rate is never reported without its
expectancy. The §7.3 qualifiers ("placement-fragile," "single-year-dependent")
attach wherever triggered.

No result of this experiment supports trading decisions; the repo's standard
disclaimer applies.

---

## 9. Power / minimum detectable effect

The OOS stream spans \~11.5 years. For a daily excess stream judged at t = 2,
the detectable annualized Sharpe is roughly `2/√11.5 ≈ 0.6` — comfortably
above the call wing's pinned 0.52, which is the repo's one surviving premium.
Read plainly: **an effect the size of the best premium this repo has ever
measured would be marginal here.** A null therefore means "no premium ≥ \~0.6
Sharpe of hedged excess," not "no premium." Per-window selection noise (the
unpinned \~+0.13 rank-correlation measurement, §5.1) further widens the
verdict's variance in both directions. One more caveat, fixed now: contract
quantization (`num_contracts = capital // (100 × price)`) deploys roughly
50–100% of the $100K depending on each window's entry price, so the stitched
stream is deliberately heteroskedastic across windows (the HAC variance
absorbs this for validity) and the MDE above — quoted on the $100K base — is
optimistic in fully-deployed units; per-window deployed notional is reported
in the winner table. The modal expected outcome, given §1's priors: arm B's
raw curve positive and seductive, the verdict t ≤ 0, row 4 language
published.
The registration exists so that sentence gets written even when a more
exciting one is available.

---

## 10. Implementation constraints

- **New analysis code, committed before any number:** a
  `walk_forward_structure` driver (the `walk_forward_real` window arithmetic
  around `run_structure_via_spec('credit_spread', …)`), the exit knobs passed
  through `params` (they are engine params, not grammar coordinates — the
  campaign's `_validate_grammar` rightly rejects them, which is why this is a
  registered experiment and not a campaign batch); and, for arm B, a
  spec-override switch letting the `credit_spread` spec run at the engine's
  *existing* `hedge_mode='none'` (the iron condor's committed setting —
  today `run_structure_via_spec` hardcodes the spec's hedge mode; no new
  hedge mode is added).
- **Pin protection:** every existing pinned regression — the vol-premium
  suite, the campaign ledger, the Gap A/D/E/F pins — passes byte-identical
  before any experiment run. The drift alarm (arm C1) must reproduce the
  campaign's SPY −0.91 ± 0.02 at the campaign's exact coordinates (§4)
  before any other number is read.
- **Seeds:** random-entry ensemble 20260717; block-bootstrap seed fixed in the
  analysis script at commit time. No other randomness exists in the design.
- **Ordering:** no §5 run before this file's merge commit to `main`; results
  land in a separate PR citing this registration's merge commit (not any
  later amendment commit) and the analysis-code commit; the §8 row is
  published verbatim.
- **Compute note:** \~120 cells × \~23 windows × (train + test) engine runs per
  arm is the expensive object (\~thousands of overlay runs). The one-store
  memory budget (`portfolio_scout` precedent) and per-ticker sequential
  loading apply; a cached per-cell cycle index inside a window is an
  implementation optimization, never a change to selection semantics.

---

## 11. Amendments

Any change to this document after its registration merge — whether or not
any result has been computed yet — must be recorded in an "Amendments"
section appended here, with date, what changed, and why; every claim affected
by an amendment is demoted to exploratory. Silent edits void the
registration.

---

## 12. Lineage and references

- Internal, the family's record: `select_credit_spread` /
  `run_real_credit_spread_overlay` (`realchains/vol_premium.py`); the
  campaign cell and Widening 3 narrative (`docs/edge_search.md`,
  `test_credit_spread_all_wrong_signed`); the registered put wing
  (`docs/prereg_vol_premium.md`, `docs/vol_premium.md`,
  `TestSpyShortPutRegression` / `TestIwmShortPutRegression`); the call-wing
  benchmark (`TestSpyShortVolRegression`, +2.54 / +2.25 at 0.5 bp).
- Internal, the machinery: Gap E exit seams and Experiment 4
  (`docs/van_tharp_gap_e.md`, `TestSpyExitVariantExploration`); Gap F random
  entry and Experiment 2 (`docs/van_tharp_gap_f.md`, `TestRandomEntryScout`);
  Gap A/D ledger and regimes (`common/trade_ledger.py`); Gap C+B sizing
  (`common/position_sizing.py`); the walk-forward precedents
  (`realchains/walk_forward_real.py`, `TestSpyRealWalkForwardRegression`,
  `TestMsftRealWalkForwardRegression`); the significance block
  (`common/stats.py`); the closed grammar (`STRUCTURE_GRAMMAR` /
  `ALLOWED_GRID`, `search/edge_search.py`).
- Method lineage: walk-forward and the degrees-of-freedom floors follow Pardo
  (2008); the delta-hedged-gain measure follows Bakshi & Kapadia (2003); the
  R-multiple / expectancy frame follows Van Tharp; the add-one Monte Carlo
  convention is Davison & Hinkley (1997); HAC inference is Newey-West with
  the Andrews lag as implemented in `common/stats.py`; the
  selection-vs-verdict discipline follows the data-snooping reality-check
  family (White, 2000).
