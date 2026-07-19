# SPY Covered Call in R-Multiples — Hedged and Unhedged, Exits and Sizing

**Status: DESIGN. No measurement has been run. Every definition below is
frozen before any outcome number is computed.**

**Epistemic class: exploratory measurement** — the Gap E precedent
(`docs/van_tharp_gap_e.md`): sample-spending, kill-or-justify, never a
registered verdict. Nothing enters the idea ledger and no e-value is spent
(the covered call is not a structure-campaign cell). Results pin via the
standard three surfaces. Any right-signed cell clearing the escalation bar
(§8) earns at most a *registration proposal*, never a strategy claim.

Date: 2026-07-19.

---

## 0. Reader's guide — what question this answers

A session of QQQ covered-call dissection (2026-07-19, chat-level, unpinned)
ended at a decomposition: selling a call bundles an **insurance bet** (will
the premium exceed what the wiggling costs?) with a **direction bet** (every
dollar of rally above the strike is forgone). The delta-hedged short call
isolates the first; the covered call carries both. On QQQ the insurance
margin measured \~zero (`TestQqqRealRiskManagedRegression`, Newey-West
t +0.18) and the covered book was dominated by the direction bill.

SPY is the interesting ticker because it is the one place the repo has a
surviving exploratory insurance margin: the delta-hedged short call at
Newey-West t **+2.54** gross / +2.25 at 0.5 basis point hedge friction
(`TestSpyShortVolRegression`, frozen registered span) — never confirmed at
any registered rung, entry-jitter band 0.98–3.58, but alive. Meanwhile the
risk-managed (delta-hedged) **covered call** re-measurement that flipped
MSFT (−0.23) and QQQ (+0.18) was never run on SPY. This experiment fills
that cell and re-measures the whole SPY covered-call object in one
consistent accounting frame:

1. The unhedged covered call and its delta-hedged twin, measured in
   **R-multiples** (per-cycle profit divided by premium collected), on the
   same cycles, so the difference *is* the direction bill per cycle.
2. A frozen exit grid over both books.
3. A position-sizing battery over both books' R-streams.
4. Van Tharp's six market types over both books — with a committed
   prediction that hedging *flattens* the regime structure.

**Terms used throughout** (stated once, per the house plain-language rule):
an **R-multiple** is a trade's profit or loss divided by the risk taken on
at entry — here, the premium collected; a win that keeps the whole premium
is +1R by construction, and losses are open-ended. **Newey-West t** is a
t-statistic (signal divided by noise) robust to overlapping/correlated
returns. **SQN** (System Quality Number) is Tharp's per-trade t-statistic.
**MAE** (maximum adverse excursion) is the worst mark-to-market a trade
touches before it closes. **Kelly fraction** is the bet size that maximizes
long-run growth of a given R-stream; it is zero or negative for a
negative-expectancy stream.

---

## 1. Prior work this design extends (and must not silently re-pin)

- `TestSpyShortVolRegression` — the +2.54 / +2.25 hedged short-call
  benchmark, frozen span. Untouched; this experiment reads it only as a
  comparison point.
- `TestMsftRealRiskManagedRegression` / `TestQqqRealRiskManagedRegression`
  — the hedged-CC daily Newey-West t's (−0.23 / +0.18). SPY is the missing
  third measurement.
- `TestMsftStopLossRegression` — the covered-call stop-loss whipsaw verdict
  (stops truncate the tail, worsen expectancy).
- Gap E (`docs/van_tharp_gap_e.md`, `TestSpyExitVariantExploration`) — the
  six-variant exit grid on the SPY *cash-backed short vol* overlay. Two
  things carry over: its convention caveats (§7) and its half-contradiction
  of the whipsaw prior — on the *hedged* book the 2× stop improved
  expectancy (−0.5407R → −0.1848R) and truncated worst MAE (−11.41R →
  −3.12R). One thing must NOT carry over: its accounting seam. Gap E's
  per-cycle R excluded the hedge's own P&L (the "raw option-cycle basis"),
  which understates a hedged book and was flagged as a trap in the
  jitter exploration. §3 fixes this by construction.

---

## 2. Books, parameters, data, spans (frozen)

**Book U (unhedged):** `run_real_cc_overlay` on SPY — the plain covered
call. Parameters: `call_delta 0.25`, `dte 30`, `close_at_pct 0.75`,
`capital $100K`, bid/ask fills. (30 DTE chosen to match the session's QQQ
book and the 30-DTE short-vol benchmark; stated here because the module's
`__main__` default is 21.)

**Book H (hedged):** identical parameters plus `delta_hedge=True` — the
engine's existing Israelov-Nielsen risk-managed path (same semantics as the
MSFT/QQQ risk-managed pins): short the call's delta in shares, rebalanced
daily at the close.

**Data:** canonical `spy_option_dailies.csv` (calls only — both books need
only calls), as-traded SPY prices via `load_unadjusted_prices`.

**Spans, two arms:**

- **Primary arm — live span:** `CHAIN_CLEAN_START['SPY'] = 2010-05-17` →
  store end (2026-06-05). Exploratory work uses the corrected live hygiene
  boundary.
- **Comparability arm — frozen span:** `REGISTERED_CLEAN_START['SPY'] =
  2010-12-01` → store end, baseline exits only (no grid), so Book H's daily
  statistic can sit directly beside the +2.54 pin and Gap E's numbers
  without span mismatch.

---

## 3. R-multiple accounting (frozen — the seam this design exists to fix)

Both books run through the Gap A trade ledger (`build_trade_ledger`,
`risk_basis='premium_collected'`): a cycle's R = its P&L divided by the
premium collected at entry. Wins cap at +1R; losses are open-ended.

**Book H cycle attribution:** a hedged cycle's P&L **includes the hedge's
share-trade P&L**, attributed by date interval — every hedge trade (and the
mark-to-market of the hedge position) between cycle i's entry and close
belongs to cycle i. The denominator stays the premium collected, identical
to Book U, so per-cycle R is comparable across books and the paired
difference `R_U − R_H` per shared cycle is the direction bill in R units.

If entry days diverge between books mid-sample (an exit variant closing one
book earlier than the other), pairing is by cycle index only while entry
dates match, and the paired table reports its matched count. The baseline
exit produces identical entry sequences by construction (the hedge never
changes entry or exit triggers).

**Reported per book, per cell:** n, expectancy (mean R), win rate, average
win R / average loss R, worst R, MAE distribution, SQN, trade-order
Newey-West t of the R series, total P&L, final equity — plus, for Book H
and the buy-and-hold comparator, the **daily** hedged-excess Newey-West t
(the +2.54-style estimator), so both statistics the repo has historically
used appear side by side and any divergence between them is visible rather
than confusing.

---

## 4. Exit grid (frozen)

The engine's existing exit knobs, a 3 × 3 grid applied to **both** books on
the primary arm:

- `close_at_pct` ∈ { 1.0 (hold to settlement), 0.75 (baseline), 0.50 }.
- `stop_loss_mult` ∈ { none, 2.0, 1.5 } (buy the call back when its cost
  reaches that multiple of the premium collected).

18 engine cells total (plus the two baseline comparability-arm runs).
Convention caveats carried verbatim from Gap E: stops fire as daily-close
stop-markets (this flatters the stop — no intraday fills), triggers
evaluate only on days with all legs quoted, re-entry rolls to the next
chain day.

---

## 5. Position-sizing battery (frozen; post-hoc replay, no engine re-runs)

Applied to each book's **baseline-exit** R-stream on the primary arm:

- **Fixed-fractional replay** (`simulate_sizing` / `sizing_sweep`) at
  f ∈ { 0.5%, 1%, 2% } of equity risked per trade, marble-bag resampling
  with `n_trades` = the book's own cycle count (the corrected frame from
  the call-spread exploration — never the pooled-bag length), seed
  `CC_R_SEED = 20260719`. Reported: P(ruin), P(25% drawdown), median
  terminal equity.
- **Kelly fraction** per book.
- **Overwrite ratio** ∈ { 100%, 50%, 25% } — the covered-call-specific
  dial: scale the overlay to that fraction of the shares (deterministic
  blend of buy-and-hold and the full book). Reported: final equity, max
  drawdown. Stated in advance: per-cycle R is invariant to this dial (it
  scales dollars, not the win/loss exchange rate), and ratio → 0 converges
  to buy-and-hold; the dial *bounds* the overlay, it cannot repair it.

---

## 6. Regime read (frozen)

Both books' ledgers through `six_regime_map` (Van Tharp's six market types
— direction × volatility, lookahead-clean, warmed with prices from
2009-06-01) and `regime_ledger_statistics` with Tharp's 30-trade floor.

**Committed structural prediction:** hedging removes the direction bet, so
Book H's cross-cell spread (max cell expectancy minus min cell expectancy,
floor-passing cells only) is **smaller** than Book U's, and Book U's worst
floor-passing cell is a bull cell (the QQQ pattern: `bull_quiet` at −5.08
trade-order t, unpinned chat measurement). If Book H retains large regime
structure, the decomposition story is wrong in a way worth knowing.

---

## 7. Committed expectations (priors stated before the run)

| # | Claim | Basis |
| --- | --- | --- |
| 1 | Book U expectancy < 0 on the primary arm. | QQQ −0.39R (chat, unpinned); MSFT ledger pins; 0/63 campaign. |
| 2 | Book U's stop cells worsen expectancy vs. their no-stop twins. | The MSFT whipsaw verdict (`TestMsftStopLossRegression`). |
| 3 | Book H's stop cells are a genuine open question — Gap E found stops *improved* the hedged short-vol book. Committed two-sided. | Gap E half-contradiction. |
| 4 | Book H daily hedged-excess t lands between QQQ's +0.18 and the frozen-span +2.54; the frozen comparability arm reconciles with the +2.54 pin to within estimator convention. | The risk-managed pair; `TestSpyShortVolRegression`. |
| 5 | No exit or sizing variant flips either book's expectancy sign. | Three prior exit/sizing explorations. |
| 6 | Kelly ≤ 0 for Book U; Book H's Kelly reported without a committed sign. | Kelly of negative streams. |
| 7 | Hedging flattens the regime structure (§6). | The decomposition. |

Contradictions of these priors are findings, not failures — they get pinned
with the same weight as confirmations.

---

## 8. Multiplicity honesty and the escalation bar

This is a search over 18 exit cells plus a sizing grid; the batch is
reported **in full** — every cell, no cherry-picking, wrong-signed and
boring cells included. The pre-committed escalation bar, mirroring the
call-spread widening precedent: a cell interests us only if it is
right-signed (positive expectancy, or positive daily hedged-excess t for
Book H) with trade-order Newey-West **t > 2** on the primary arm. At or
below the bar it records as closed. Above the bar it escalates to a
human-signed registration proposal — nothing in this experiment promotes
directly.

---

## 9. What this experiment is NOT

- Not a registered experiment; no registered pin moves, and the frozen-span
  arm exists precisely so published pins are compared against, not re-pinned.
- Not an edge hunt: the committed priors say both books stay non-positive
  under every variant; the value is the SPY hedged-CC missing cell, the
  paired direction-bill table, and closing the exit/sizing question for the
  CC family on SPY with pinned numbers.
- Not a Gap E re-run: Gap E's book was the cash-backed short call with
  raw-cycle R accounting; this measures the covered call, both books, with
  hedge-inclusive cycle R.

---

## 10. Build plan

- **Module:** `realchains/cc_r_experiment.py` — grid runner (both books ×
  exit grid × two arms), the Book-H hedge-P&L cycle-attribution helper,
  the paired-difference table, sizing battery calls, regime read, and a
  print-only report (no file outputs beyond what the tests pin).
- **Tests:** `tests/test_cc_r_experiment.py` — an always-run synthetic
  layer (the attribution helper conserves total P&L: summed cycle P&L
  including hedge equals the book's net overlay P&L to the cent; pairing
  logic; grid enumeration) plus a dataset-gated `TestSpyCcRExperiment`
  pinning the decisive numbers: both books' baseline expectancy/worst
  R/trade-order t, the Book H daily t on both arms, the 3 × 3 expectancy
  grids, Kelly fractions, sizing-sweep ruin numbers, and the two regime
  spreads.
- **Results surface:** a `docs/explorations.md` entry (or its
  positive-read equivalent per §8).
- **Plumbing in the build PR:** `ci.yml` pytest bucket, CLAUDE.md
  symbol-regex additions, README file-table row.
- **Runtime estimate:** \~20 engine passes over the SPY store — tens of
  minutes; the sizing battery is a fast post-hoc replay.

---

## 11. Order of operations

1. This design doc merges; the definitions above are frozen.
2. The build PR lands the module and synthetic tests — no measurement
   numbers.
3. One run executes both arms; decisive numbers pin in the dataset-gated
   test and the exploration-log entry in the same PR.
4. §8 governs any escalation; otherwise the SPY covered-call exit/sizing
   question closes with the pins.
