# The Cup-and-Handle Scan — O'Neil's Pattern, Counted Across the S&P 500

**Status: DESIGN. No measurement has been run. Every definition below is
frozen before any outcome number is computed. The scan additionally
waits on a data dependency: the S&P 500 1-minute archive fetch
(in flight, \~4–5 days at the measured cadence) and its coverage
diagnostic (§2).**

**Epistemic class: exploratory replication** — the support/resistance
replication's sibling (`docs/tharp_sr_replication_plan.md`):
sample-spending, kill-or-justify, never a registered verdict. Nothing
enters the idea ledger (the committed guess-counter for automated
structure searches) and no e-value is spent (the running evidence
budget that counter's false-discovery control draws down). Results pin
via the standard three surfaces (module, dataset-gated tests,
exploration-log entry — the pins are **dataset-gated and skip in CI**,
unlike the support/resistance pins: the input archive is personal,
gitignored, and unpublished). A surviving read earns at most a
proposal, never a claim.

Date: 2026-07-20. Owner-directed follow-up to the cup-and-handle
feasibility discussion.

---

## 0. Reader's guide — what question this answers

The **cup and handle** is William O'Neil's flagship chart pattern
(*How to Make Money in Stocks*; the franchise behind IBD — Investor's
Business Daily, his newspaper — and CANSLIM, his stock-picking system):
a stock in an uptrend carves a months-long rounded, U-shaped dip (the
cup), rallies back near its old high, drifts down for a week or two in
quiet volume (the handle), then breaks above the handle's high on a
volume surge — the buy signal. The claim to test is the pattern's
*entry* claim: do these breakouts, detected by frozen mechanical rules,
beat entering on random days?

**The committed prior says no, and says why**: strip the story and the
tradable event is a breakout through local resistance, conditioned on
the shape of the preceding months. The support/resistance replication
just measured the neighborhood — the whole six-signal batch, breakouts
at three lookbacks included, at 0-of-384 cells against random entry —
and the entry-conditioning family is dead at six-for-six (the wheel's
up-day gate being the sixth). The pattern's one honest hope, and the
reason this scan exists, is **universe**: O'Neil's mechanism (an
institutional-accumulation base in a single growth stock) is claimed
for individual names, not the mean-reverting index ETFs everything so
far ran on. The S&P 500 minute archive gives the scan \~500 single
names × \~26 years — pooled, real statistical power.

**Two labels the result wears no matter what it shows**, stated before
any number exists:

1. **Survivorship**: the universe is TODAY'S 502 constituents — the
   winners. Point-in-time membership is a separate licensed dataset the
   repo does not own. Whatever the pooled scan measures is
   survivorship-flattered by construction and stays capped at
   exploratory.
2. **A detector, not THE pattern**: O'Neil's description is prose. The
   rules below adopt his published coordinates where they exist and
   freeze the rest; the result speaks for *this* detector, one point in
   a large definition space, and that cap is carried on every surface.

**Source honesty**: the repo holds no committed O'Neil text. Two of his
coordinates ARE anchored by a committed secondary source — the owner's
Tharp notes quote O'Neil's breakout-volume rule at "at least 50
percent" above average (Kindle Location 5559 in
`research/book-notes/trade-your-way-to-financial-freedom.md` — a passage
that names the cup and handle itself) — and
those carry citations below. The remaining **[O'Neil]** tags are the
values as commonly published in *How to Make Money in Stocks* (4th ed.)
and IBD educational material, cited as practitioner canon and hedged
accordingly; where the canon gives a band, the frozen endpoint is
labeled as our choice from it. If the owner's Kindle library holds the
book, a notes extraction (the Tharp/Pardo pattern) can upgrade the
anchors; the frozen values do not move either way.

**Terms used throughout** (stated once, per the house plain-language
rule): **OHLCV** is a day's open, high, low, close, and volume. A
**session** is one trading day. **OLS slope** is the fitted
straight-line trend through a window of closes (ordinary least
squares). **argmax/argmin** pick the day with the largest/smallest
value (ties: the earliest day, frozen here once). The **base rate** is
the unconditional probability of the same outcome over the same
eligible days. A **flat-only book** takes a new trade only when no
trade is open (the support/resistance harness's convention), and
**add-one Monte-Carlo p** is that harness's `(1 + count) / (1 + B)`
p-value form. A **cluster** is all pooled entries sharing one calendar
date — the §5 judging unit. **Pooled** means all tickers judged as one
stream.

---

## 1. Prior work this design extends (and must not silently re-pin)

- The support/resistance replication
  (`docs/tharp_sr_replication_plan.md`, `TestSrReplicationPins`) — the
  harness whose *conventions* this scan reuses (the flat-only trade
  builder, the add-one Monte-Carlo p, NO-VERDICT floors; the pooled
  cluster null of §5 is new code following those conventions) and the
  standing verdicts: the six-signal batch 0-of-384 against random
  entry, breakouts included; Turtle Soup failure rates \~48–65% in
  every era.
- Tharp's own objection to chart patterns — "difficult to objectively
  describe so that they can be computerized" (the owner's book notes,
  Location 4401) — which §3 answers by freezing one computable
  definition and §4 answers by validating the detector before any
  return is computed.
- The six entry-conditioning nulls (trend gate, cooldown, IV-richness,
  the CC R-experiment's splits, the wing diagnostic, the wheel's
  up-day gate) — the family this pattern's "context filters which
  breakouts work" claim belongs to.
- **The QQQQ rename lesson** (CLAUDE.md, option-chain pipeline): a
  renamed symbol's pre-rename history can be silently absent under the
  new symbol. Directly applicable — e.g. META traded as FB until 2022;
  its archive fetched as META may start at the rename. §2's coverage
  diagnostic exists because of this precedent.
- **The XLE split lesson**: as-traded prices versus corporate actions.
  The minute archive is deliberately as-traded; a cup detector run on
  as-traded prices would read every split as a crash. §2 freezes the
  split adjustment — with the split table SNAPSHOTTED, not live-fetched
  (the yfinance silent-empty failure mode).

---

## 2. Data, aggregation, and coverage (frozen)

**Input**: the S&P 500 1-minute archive
(`data/sp500_intraday_1min/{ticker}_intraday_1min.csv.gz`, fetch in
flight), plus the nine existing archives at the data root for overlap
tickers (MSFT, NVDA). Personal, gitignored, unpublished — hence the
dataset-gated pins. **The universe list IS committed**:
`data/sp500_tickers_2026-07.txt` (502 current constituents, the
Wikipedia constituents table as of 2026-07-20; public strings, tiny) —
the frozen universe lives in git even though the price data cannot.

- **Daily aggregation (frozen)**: minute rows are sorted by timestamp
  with exact-duplicate timestamps collapsed to the last row; one
  derived daily bar per session from bars with timestamps
  `09:30:00 ≤ t ≤ 16:00:00` (the 16:00 bar carries the closing auction
  print; pre-market and after-hours bars are excluded). Open = first
  such bar's open, high = max, low = min, close = last bar's close,
  volume = sum. A session with zero regular-session bars contributes no
  daily bar (the day is simply absent; absences are counted in the
  coverage diagnostic). Cached as a regenerable artifact in the same
  ignored workspace; the scan reads the cache.
- **Split adjustment (frozen)**: the scan runs on **split-adjusted**
  closes. The split table is a **committed snapshot**
  (`data/sp500_splits_2026-07.csv`: ticker, ex-date, ratio, as-of
  date — built once in the build PR from yfinance and committed; an
  empty fetch for any ticker with a visible price cliff is a HARD
  error, never a silent no-split). Each as-traded price is divided by
  the product of ratios of splits dated **strictly after** that day;
  volume is multiplied by the same factor. Dividends are NOT adjusted
  (standard chart convention). **Cliff guard (frozen)**: any adjusted
  day-over-day close ratio outside `[0.5, 2.0]` not explained by a
  committed split is flagged in the coverage diagnostic and that
  ticker is excluded until resolved by hand.
- **Coverage diagnostic (frozen; runs BEFORE the scan and gates it)**:
  per ticker — first/last session, session count vs. the trading
  calendar over its span, cliff-guard flags, and a **late-start flag**
  for any ticker whose first bar is later than 2010 (IPO or rename?).
  Late starts are cross-checked by hand against known renames (META/FB
  class); a rename victim's missing era is a *coverage gap logged in
  the exploration entry*, never silently pooled — its detections count
  only over the span the archive actually holds. A ticker failing the
  fetch entirely (`failed_tickers.txt`) is excluded and listed. The
  gate: the scan runs only after the diagnostic's flag list is
  resolved into excluded/included-with-logged-gap per ticker, recorded
  in the exploration entry.
- **Eras**: the support/resistance replication's fixed panels
  (1999–2009 / 2010–2019 / 2020–2026), assigned by the breakout date,
  reported beside the pooled read.

---

## 3. The detector (frozen)

Scanned per ticker on split-adjusted daily closes. All comparisons are
as written (strict where strict, inclusive where inclusive); an exact
tie on a strict comparison fails the claim. Coordinates marked
**[O'Neil]** are practitioner-canon values (§0 source note);
**[O'Neil band]** marks a canon range whose frozen endpoint is our
choice; **[ours]** is frozen here.

**The iteration, frozen first**: `t` ranges over every session past
warm-up in chronological order. For each `t`, handle-window lengths are
tried in ASCENDING order (5, 6, … 25); the FIRST window passing all of
rules 1–6 is THE recorded detection (its `(W, r, l, b)` is the pinned
anatomy) and the search for that `t` stops. A rule failure inside one
window rejects that window only; a rim-band failure (rule 2) rejects
that window's candidate — no second-best `l` is ever tried (the argmax
`l` is the only `l`). After a detection at `t`, the next candidate is
`t + 25` or later **[ours]**.

1. **Handle**: a window `W = [h0, t-1]` of the current trial length
   **[ours: 5–25 sessions; the canon says "one to two weeks or more"]**
   whose highest close (the **handle high**, and the cup's **right
   rim** `r`; argmax ties → earliest) has zero-based index `i` in `W`
   with `3i < len(W)` (the first third) **[ours]**; the OLS slope of
   `W`'s closes is `≤ 0` (the handle drifts down) **[O'Neil,
   direction; the OLS form ours]**; the handle's lowest close is
   `≥ 0.85 ×` the handle high **[O'Neil band: "10–15% in normal
   markets"; the 15% endpoint ours]**; mean volume over `W` is below
   mean volume over the cup sessions `[l, r]` inclusive (volume dries
   up) **[O'Neil, direction]**.
2. **Cup**: `l = argmax(close)` over `[r−325, r−35]` (duration 7–65
   weeks **[O'Neil]**); the right rim must sit within the band
   `close[r] ∈ [0.85, 1.05] × close[l]` **[ours]**; the bottom
   `b = argmin(close)` over `(l, r)` with depth
   `0.12 ≤ (max(close[l], close[r]) − close[b]) / max(close[l], close[r]) ≤ 0.33`
   **[O'Neil]**; `b`'s position satisfies
   `0.2 < (b − l) / (r − l) < 0.8` **[ours]**; no interior close in
   `(b, r)` exceeds `1.02 × close[r]` **[ours]**.
3. **Roundness — the U-vs-V gate (primary, [ours])**: the fraction of
   sessions in `[l, r]` whose close sits within the bottom quartile of
   the cup's depth (at or below `close[b] + 0.25 × (max(close[l],
   close[r]) − close[b])`) is `≥ 0.15` of the cup's session count. A U
   lingers at the bottom; a V visits for a day. **Labeled variant**: a
   quadratic fit to the cup's normalized closes with R² (the fit's
   explained-variance share) `≥ 0.70` and vertex in the middle third —
   reported, never gating.
4. **Handle in the upper half**: the handle's lowest close
   `≥ close[b] + 0.5 × (max(close[l], close[r]) − close[b])`
   **[O'Neil]**.
5. **Prior uptrend**: `close[l] ≥ 1.3 × min(close over [l−90, l))`
   **[O'Neil, the 30%; the 90-session lookback ours]**.
6. **Trigger**: `close[t] > handle high` (strictly; **[ours]** — the
   canon buy point is the handle high plus ten cents crossed
   *intraday*; this detector substitutes the close-above rule, the
   harness's close-only convention) AND `volume[t] ≥ 1.5 × mean(volume
   over [t−50, t))` **[O'Neil via the committed Tharp notes, Location
   5559: "at least 50 percent above the daily average"; the 50-session
   average ours]**.

**Warm-up (frozen, behavioral)**: a candidate is rejected whenever any
index its rules reference precedes the ticker's first session — in
practice the earliest possible detection sits near session
`440` (`325` cup + `90` uptrend + the handle) — and the null's
drawable universe (§5) starts at the same behavioral boundary,
evaluated per ticker.

---

## 4. Detector validation (frozen; BEFORE any return is computed)

The detector is falsified first, as its own deliverable:

- **Synthetic battery** (always-run tests): a hand-built textbook cup
  fires; each rule's violation is rejected in isolation — a V-bottom
  (roundness), a too-shallow saucer and a too-deep crash (depth), a
  handle below mid-cup, an upward-sloping handle, a quiet-volume
  breakout, a missing prior uptrend, an interior overshoot. Plus
  determinism, the shortest-window-wins iteration rule, and the
  warm-up boundary.
- **Detection-rate sanity (a validity gate, not a tuned target)**: the
  committed statistic is `total detections ÷ Σ per-ticker span in
  decades`, band **0.5–4 per ticker-decade**. Outside the band the
  detector is broken. **The fix path is bounded (frozen)**: at most ONE
  documented amendment round; only **[ours]**-tagged constants may
  move; the symptom, each old→new value, and owner sign-off are
  recorded as an amendment to this doc BEFORE §10 step 4 runs. Priors
  touching the band are restated in the same amendment.
- **The eyeball pass**: a regenerable figure rendering the twenty
  highest-volume-surge detections — price with the cup, handle, and
  breakout annotated, **clipped at the breakout day** (the right edge
  is `t`; no post-breakout path is rendered, so no return is seen
  before §10 step 4). A human looks at it; a detector nobody has
  looked at is not a detector. **DONE 2026-07-23**
  (`engine/cup_handle_figure.py` → `docs/figures/cup_handle_eyeball.png`,
  owner-signed): all twenty are recognizable cups — a rounded U-dip, a
  recovery to the rim, a short quiet handle, a volume breakout — with no
  split-jump, corrupt-spike, or noise artifact among the top surges (the
  strongest confirmation the split hygiene held). The shallowest two
  (AXON 2023, EME 2017) sit at the roundness floor, admissible bases
  rather than failures. The detection count is 432, unchanged by the
  split reclassification — the scale-invariance of the shape rules made
  the evaluation set stable through convention (a).

  ![A 5x4 grid of the twenty highest-volume-surge cup-and-handle detections across the S&P 500, from ISRG (12.4x) to GM (5.3x). Each panel plots price ending at the breakout day with a green up-triangle, the cup span shaded blue and the handle span shaded orange, the left rim / bottom / right rim marked and the handle-high breakout level dashed. Every formation shows a prior uptrend, a rounded U-dip, a recovery to the rim and a short quiet handle; none is a vertical split-jump or a corrupt spike.](figures/cup_handle_eyeball.png)

  *The §4 eyeball pass. Twenty cups, no artifacts: after the split
  reclassification the highest-surge breakouts are real O'Neil bases, not
  adjustment glitches — the check that clears the detector for step 4.
  Each panel stops AT its breakout day, so the figure shows what the
  buyer saw and nothing after, keeping returns unseen until the
  `--evaluate` gate.*

### Amendment 1 — the band was mis-specified; no constant moves (owner-signed 2026-07-22)

This is the one sanctioned amendment round. **It moves zero constants.**

**Symptom.** The S&P 500 archive scan (501 tickers, \~1,101 ticker-decades)
returned **432 detections, a rate of \~0.40 per ticker-decade** — below the
0.5 floor.

**The prior it violates was never self-consistent.** Prior 1 asserts two
things at once: a rate inside 0.5–4, *and* a total detection count inside
1,000–3,500. Against this archive's span those describe different regions:

| Half of prior 1 | Implies |
| --- | --- |
| rate 0.5–4 | 550–4,404 detections |
| count 1,000–3,500 | rate 0.91–3.18 |

They overlap only on a rate of **0.91–3.18**. Any outcome between 0.50 and
0.91 satisfies the rate band while violating the count band, so a run
landing there would have been simultaneously in-band and out-of-band. The
prior could not have been jointly satisfied by a single number it did not
already name; that is a specification defect, not a detector defect.

**Why the detector is not the thing at fault.** The dominant rejector is
the breakout volume trigger, `VOL_SURGE = 1.5` — roughly seven of every
eight otherwise-qualifying candidates die on it. That constant is **not**
`[ours]`: it is O'Neil's, carried in through the committed Tharp notes at
Loc 5559, and it is frozen by attribution. A low detection rate driven
mostly by a constant this amendment is forbidden to touch is evidence that
the *predicted* rate was wrong, not that the *detector* is broken.

**What was considered and rejected.** Loosening `ROUNDNESS_MIN` (0.15),
widening `RIM_BAND` (0.85–1.05), or raising `HANDLE_DEPTH_MAX` (0.15) —
all `[ours]`, all individually sufficient to close a 1.27× gap. Every one
was rejected for the same reason: **the observed rate is already known.**
Moving a shape constant now to land inside a band is tuning to a target
with the answer visible, which is the exact failure the gate exists to
catch. A gate that gets adjusted until it passes has stopped being a gate.

**The restatement.** Prior 1 is **WRONG and withdrawn**, in both halves. It
was an unanchored guess about how often a fully-specified formation with a
frozen volume trigger occurs; nothing derived it, and the two halves were
never checked against each other. The replacement is not another band —
committing a new one after seeing the number would launder the same error.
Instead: **the observed rate is a measured property of this detector on
this archive, and it is reported, not gated.** The remaining §4 defences do
the validation work — the synthetic battery (each rule falsified in
isolation) and the eyeball pass, which is now the load-bearing check.

**Consequence for §10.** Step 3's gate is **discharged by this amendment,
not by passing**. The amendment budget is spent: no further amendment
round exists, and every detector constant is frozen as written for the run.
Any future change to a detector constant requires a new registration, not
an amendment to this doc.

---

## 5. Evaluation frame (frozen — cluster-level judging)

**Entries** at the breakout day's close; **time exits** at
`H ∈ {5, 10, 15, 20}` sessions (the LeBeau–Lucas horizons, for
comparability with the support/resistance pins) plus `H ∈ {60, 120}`
labeled the O'Neil-horizon extension (his holding frame is
weeks-to-months). Long side only. **Per-ticker flat-only books** as in
the harness: within a ticker, a detection during an open trade's `H`
sessions is not traded (at `H = 60/120` the lockout exceeds the §3
dedup, so traded entries are a strict subset of detections — disclosed,
and the null matches TRADED entries, below). An entry whose exit falls
past the ticker's span end is skipped. Win = exit close strictly above
entry close; per-trade simple returns.

**Return breaks are dropped from both sides (added 2026-07-22).** Three
retained sessions are value *detachments* rather than price moves — MO's
2008 Philip Morris spin-off, ROK's 2001 Rockwell Collins distribution, WY's
2010 purge dividend — where the holder was made whole in cash or stock. The
price series is correct; a return measured across one is not, and would
book a 57–69% "loss" that nobody took. Any entry whose window `(t, t + H]`
crosses one is **dropped and counted**, in the real book and in the
matched-count null alike (applying it to one side only would bias the
comparison). Entry *on* the detachment day is kept — `close[t]` already
reflects it, so the forward return is honest. The table lives with the
other hygiene rulings in `pipeline/minute_archive.py`. **Stated cap:** the
cliff guard only sees moves that roughly halve the price, so a 20–30%
spin-off is invisible to it and this archive carries no distribution feed —
this is a floor on known contamination, not a guarantee of clean returns.

**The judging unit is the calendar-day cluster.** Detections
synchronize across hundreds of names on the same market-wave days, so
pooled per-trade counting would understate the noise (many "independent"
trades are one market bet). Frozen: all TRADED entries sharing one
calendar date form one **cluster-trade** whose return is the
equal-weight mean of its members' returns; cluster win = cluster return
`> 0`. Members per cluster are reported.

**The matched-count cluster null (frozen)**: strata are calendar
**months** for `H ≤ 20`, calendar **quarters** for `H = 60`, and
calendar **half-years** for `H = 120` (the stratum must exceed the
lockout for within-stratum matching to be implementable). Per stratum
with `k` clusters, each resample draws `k` distinct random calendar
sessions from that stratum; a null cluster at drawn day `d` takes the
SAME member-ticker set as the real cluster it replaces (matched in
descending member-count order within the stratum) and each member's
`H`-session return from `d`; a member for which `d` is ineligible
(before its behavioral warm-up, or `d + H` past its span) is dropped
from that null cluster and counted in a dilution diagnostic.
`B = 10,000`; one derived stream per `(variant, H, stratum)` with the
derivation string `f'{CUP_SEED}:{variant}|H{H}|{stratum}'`,
`CUP_SEED = 20260720`; strata are consumed in calendar order.

**The survival rule (frozen)**: the scan survives only if the pooled
**cluster-level** win rate beats its null at add-one `p ≤ 0.01` on
BOTH committed headline horizons `H = 20` and `H = 60`. (The two
horizons share entries and are strongly dependent — the conjunction is
a robustness screen, not two independent tests; stated so the bar
isn't over-read.) The pooled per-trade win rate, per-ticker splits,
per-era splits, and the cluster mean-return percentile are reported as
diagnostics, never gating. Fewer than `100` clusters of traded entries
at a horizon → UNDERPOWERED for that horizon (shown, never judged).

**Ablations (single runs, never crossed, reported without verdicts)**:
the volume-surge trigger removed (does O'Neil's volume rule change the
detection set or the read); the quadratic roundness variant; and the
**survivorship bracket** — the pooled base rate (random entry on these
winners' histories) beside every number, so the level flattery is a
visible quantity. The bracket bounds the *level*, not the
detections-minus-random gap; the gap's own survivorship channel is
acknowledged as unmeasurable with this universe (§0 label 1).

---

## 6. Committed expectations (priors stated before the run)

| # | Claim | Basis |
| --- | --- | --- |
| 1 | ~~The detection rate lands inside 0.5–4 per ticker-decade; total detections land in the 1,000–3,500 range.~~ **WRONG — withdrawn, both halves (§4 Amendment 1, 2026-07-22).** Measured: 432 detections, \~0.40/ticker-decade. The two halves were never jointly satisfiable; the rate is now reported, not gated. | The §3 rules' tightness; canon formations are rare. |
| 2 | The pooled cluster book does NOT survive the §5 bar — the breakout does not beat matched random entry even on single names. | The support/resistance replication's 0-of-384; six conditioning nulls; the drift wall. |
| 3 | The volume-trigger ablation does not change the read. | The same replication's volume-free breakouts were equally dead. |
| 4 | The survivorship bracket is visibly fat: random entry on these histories wins comfortably above 50% at long horizons — reported as the flattery bound, not evidence. | Today's members are the winners, by construction. |
| 5 | The coverage diagnostic finds a nonzero set of rename-victim gaps (META-class) and they are logged, not pooled. | The QQQQ rename precedent. |
| 6 | Per-era diagnostics show the 2010s bull friendliest to the pattern's absolute returns without changing any verdict. | Drift dominates absolute reads; the verdict is drift-matched by construction. |

Contradictions of these priors are findings, not failures. A §5
survival would contradict prior 2 and escalate per §7 — while staying
exploratory, survivorship-capped, and a proposal at most.

---

## 7. Multiplicity honesty and the escalation path

One detector (frozen), six horizons of which two are committed headline
horizons, three ablations — every number reported, nothing dropped, and
the non-headline horizons and ablations carry no verdicts at all (so
the priced test count is one two-horizon conjunctive read). The §0
detector cap is restated wherever a number is quoted. **Escalation**: a
§5 survival earns a proposal for a follow-up design (out-of-sample
confirmation on non-S&P names and/or the point-in-time membership
purchase), never a strategy, never a gate. Anything else closes the
pattern with pins.

---

## 8. What this experiment is NOT

- Not a registered experiment; no e-value is spent.
- Not CANSLIM: O'Neil's system is the pattern PLUS fundamentals
  (earnings growth, institutional sponsorship) plus stop-managed exits.
  This scan isolates the pattern's entry claim in the LeBeau–Lucas
  frame; the rest of the system is out of scope and the log entry says
  so.
- Not free of survivorship — and not fixable with this data. The cap
  is permanent for this universe.
- Not the minute-level variant: intraday volume pacing at the breakout
  (and the canon's intraday ten-cent pivot) stay closed dials for v1;
  the minute archive's role here is supplying clean daily bars.
- Not a general pattern-mining license: this is ONE frozen detector.
  New patterns or re-tuned dials arrive as new frozen designs, and the
  §4 amendment path is the only sanctioned in-flight change.

---

## 9. Build plan

- **Module**: `engine/cup_handle_scan.py` — the daily aggregator (and
  its cache), the split-adjustment step reading the committed snapshot,
  the coverage diagnostic, the detector, the cluster-level evaluation
  (following the harness's conventions; the pooled cluster null is new
  code), and a print-only report. Deterministic; the only seed is
  `CUP_SEED`.
  - **Location note (2026-07-21, not a rule change).** The §2 data
    layer named above — aggregation and its cache, split adjustment,
    the coverage diagnostic, the cliff guard, the owner-signed hygiene
    rulings and the reference cross-check — now lives in
    `pipeline/minute_archive.py`, consumed here via `load_clean_daily`.
    The rules are unchanged and the move was verified byte-identical on
    the then-complete 271-ticker archive; it happened because those
    answers are facts about the vendor tape rather than about this
    hypothesis, and a second study must not end up with a second,
    divergent set of them. `engine/cup_handle_scan.py` keeps §3 and §5.
- **Committed data in the build PR**: `data/sp500_tickers_2026-07.txt`
  (the frozen universe) and `data/sp500_splits_2026-07.csv` (the split
  snapshot with as-of date).
- **Tests**: `tests/test_cup_handle_scan.py` — the §4 synthetic battery
  always-run; a dataset-gated pin class (requires the local archive;
  skips in CI) pinning the coverage summary, detection counts, the
  cluster verdicts, and the ablations. The data-layer half (aggregation,
  split adjustment, the rulings, the cross-check) moved with its code to
  `tests/test_minute_archive.py`; both run in CI.
- **Figures**: the §4 eyeball figure via `engine/cup_handle_figure.py`
  (regenerable, committed PNG, clipped at the breakout day). Located
  beside the detector rather than in the plan's originally-named
  `search/make_exploration_figures.py`, which lives in the chain-store
  data world; keeping the minute-archive figure with its own detector
  avoids coupling two disjoint domains (a location note, not a
  methodology change — the §2-to-`pipeline` move set the precedent).
- **Results surface**: a `docs/explorations.md` entry.
- **Plumbing**: CLAUDE.md symbol regex, README rows; the synthetic
  layer rides the CI engine job.
- **Runtime**: aggregation one-time \~minutes over the gz archive; the
  scan seconds per ticker; the pooled nulls minutes.

---

## 10. Order of operations

1. This design doc merges; the definitions above are frozen (the §4
   amendment path is the only sanctioned change, and it is bounded).
2. The fetch completes; the §2 coverage diagnostic runs and its
   findings (rename gaps, cliff flags, failed tickers) are resolved
   and logged.
3. The build PR lands the module, the committed universe and split
   snapshots, and the synthetic tests — no measurement numbers. The §4
   validity gate runs; an out-of-band detection rate triggers the
   bounded amendment path BEFORE any return is computed. **DONE
   2026-07-22: the rate came in at \~0.40, below the 0.5 floor, and §4
   Amendment 1 discharged the gate by withdrawing the mis-specified
   prior rather than moving a constant. The amendment budget is now
   spent and every detector constant is frozen.**
4. One run executes the scan and evaluation; decisive numbers pin in
   the dataset-gated class and the exploration-log entry in the same
   PR. **DONE 2026-07-23: SURVIVES = False. The breakout's cluster win
   rate is BELOW the matched-count random-entry null at every horizon
   (54.9% vs 65.8% at 20 days), p = 1.0 — worse than random. The volume
   ablation is the same story. Pinned by `TestCupHandleSp500Pins`; the
   verdict is in `docs/explorations.md`.**
5. §7 governs any escalation; otherwise the cup and handle closes with
   the pins. **DONE 2026-07-23: no escalation — it is a null (a passing
   read would earn a registration; this is the opposite). The study is
   CLOSED. The pattern is the seventh member of the entry-conditioning
   family to fall to random entry, and it fell on its best-hope
   single-name universe, so the universe hypothesis is refuted rather
   than left open.**
