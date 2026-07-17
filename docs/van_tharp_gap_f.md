# Gap F — the random-entry seam (DESIGN / build spec)

## Status

**Implementation status (2026-07): BUILT, and the measurement is in** — `random_entry_selector` /
`random_entry_scout` / `RANDOM_ENTRY_SEED` in search/explorations.py, pinned by
`TestRandomEntryMechanics` (five synthetic tests including the k=0 baseline-identity anchor and the
emission-keyed desync convention) and `TestRandomEntryScout` in tests/test_explorations.py, with the
verdict logged in [docs/explorations.md](explorations.md). **No envelope exclusion on either metric — no
entry-skill claim in either direction.** The two pinned locates: on raw per-cycle `expectancy_r` the
baseline sits at the **5th percentile** of its own band (worse than 19/20 jittered careers; the band
spans −0.58R to −0.03R, so the raw option-cycle number is placement-fragile); on the hedged NW t it sits
inside at the **85th** (+2.54 in a 0.98–3.58 band) — the premium isolator is placement-robust. The
pre-stated cooldown-texture mechanism predicted the opposite tail and is thereby not supported; the low
expectancy locate is recorded without a mechanism story. One estimate corrected by measurement: the
scout runs in \~20 seconds total (the store load dominates; an engine pass is \~0.05 s), not the \~7
minutes inferred below — the CI-bucket concern dissolves. Career trade counts ran 141–147 against the
baseline's 174. Code file:line references below describe the tree as of the design commit.

This was written as a **DESIGN document — a build spec, PLAN-level**, ahead of the code. It designs Gap F from
[docs/van_tharp_test_plan.md](van_tharp_test_plan.md): the entry-selection seam
(docs/van_tharp_test_plan.md:202-218). The parent plan sizes it Small — "the abstraction point already
exists for structures" (:218) — and sequences it fifth, after the heavier work (:267). The recon sharpens
that sizing to its limit: **Gap F needs zero engine changes.** The structure engine's `select=` seam not
only exists but is already exercised by custom selectors — Gap E's exit-mechanics tests pass inline
closures through it today — so what Gap F adds is a seeded random selector plus one measurement. It
enables **Experiment 2** (docs/van_tharp_test_plan.md:275): hold the exits, the hedge, and the sizing
fixed, randomize the entry, and measure whether the system's character survives.

Predecessors are all in: Gap A (the R-multiple ledger, [docs/van_tharp_gap_a.md](van_tharp_gap_a.md),
merged in #125), Gap D (the six-regime R-distributions, [docs/van_tharp_gap_d.md](van_tharp_gap_d.md),
merged in #126), Gaps C+B (the fixed-fractional replay + marble-bag resampler,
[docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md), designed in #127 and built in #128), and Gap E
([docs/van_tharp_gap_e.md](van_tharp_gap_e.md), BUILT per its Status — its `TestExitMechanics` classes
are part of this doc's seam evidence).

Every number this design will produce is **EXPLORATORY** — sample-spending, kill-or-justify, never a
registered verdict. The work is descriptive research measurement of historical simulation output; it is
not investment advice, and no figure in it is a recommendation to trade any instrument, any entry rule,
or any size.

**Location convention.** Book references are Kindle Locations, per the notes file's own `### Location N`
headers (research/book-notes/trade-your-way-to-financial-freedom.md; Van K. Tharp, *Trade Your Way to
Financial Freedom*, 2nd ed., McGraw-Hill, 2007, Kindle). The parent plan's pointers for this gap (Loc
202, 1801, 1806) are true Locations; the coin-flip experiment itself resolves to Loc 4834–4837, with the
realized-reliability figure at Loc 5775 and the chapter-summary restatement at Loc 5648. Two attribution
caveats travel with every citation below: Loc 202 is the second-edition foreword (third-person about
Tharp, recovered from the Kindle Cloud Reader per the notes file's inline comment), and Loc 1381's
first-person entry-is-least-important line sits amid trader-profile material, so its speaker may be a
trader Tharp is profiling — which is why the least-important claim below cites Loc 890, not Loc 1381.
The notes never name Tom Basso in connection with the random-entry study, so no Basso attribution is
made here.

## Why — Experiment 2 and the book's entry thesis

Tharp ran the experiment this gap enables. His random-entry system — coin-flip entry, an initial and
trailing stop at three times a 10-day exponential moving average of average true range, the stop only
ever tightening, and 1 percent fixed-risk sizing — "made money on 100 percent of the runs" (Loc 4834;
the mechanics at Loc 4837). That is the repo's Experiment 2 shape exactly: random entry, plus the exit
family Gap E built, plus the sizing family Gaps C+B built. The book's supporting claims: the entry
signal is probably the least important of a system's roughly ten components (Loc 890); entry controls
reliability, and reliability is not expectancy (Loc 4138); few entry techniques beat random, especially
past 20 days (Loc 5648); and the foreword's summary — with sound psychology, exits that cap losses near
1R and let winners run to large R-multiples, the right market, and position sizing, the entry just
isn't that important (Loc 202).

The LeBeau reliability frame (Loc 1801) gives the test its calibration: a random entry should be right
about half the time (45–55 percent), a good one needs 55 percent or better, and the protocol tests
without stops or costs so the entry's edge is visible before frictions eat it (Loc 1806), with the
false-positive rate as the companion check (Loc 1812). Tharp's own run is the cautionary footnote: the
realized reliability was about 38 percent, not 50 — transaction costs plus the stop took 12 points (Loc
5775). The repo already owns a live demonstration of the reliability-is-not-expectancy lesson: the
pinned SPY short-vol ledger wins 65.5 percent of its trades at −0.54R expectancy
(tests/test_trade_ledger.py:381-387).

The repo also owns a strong prior on the entry question specifically — three entry-conditioning kills,
detailed in the measurement section below. But all three tested *conditional* timing: wait when a signal
says wait. Experiment 2 asks the unconditional complement: replace the engine's deterministic entry
timing with random timing and measure whether anything is lost. Together the two answers would state
Tharp's claim in full — neither intelligence nor randomness in the entry moves this system.

## The mapping — what random entry means for a premium seller

Tharp's coin flip randomizes direction and timing on futures. Direction does not exist here — shorting
the call is the strategy. Timing does, in a weak but real form: the engine re-enters **immediately**
whenever it is flat and a chain exists. Nobody chose that cadence as a signal, but it is a timing rule —
and not a trivially neutral one, since immediate re-entry after an assignment buyback re-sells into
whatever the market just did, and the cooldown scout's exploratory read says that texture may be real (post-rip cycles
measured *better*, docs/explorations.md:30). The timing decision is therefore the entry axis this engine
actually has, and it is Tharp's axis.

v1 randomizes the entry day and nothing else: **at the start of each flat stretch, wait a random
`J ~ uniform{0..K}` chain-days before entering, with `K = 10` pre-committed** (roughly half the
\~21-trading-day cycle — enough jitter to matter, without gutting the per-career trade count). The pick
itself, once the wait expires, is the *deterministic* baseline logic unchanged: the band filter, the
nearest-|dte−30| expiration, the nearest-0.25-delta strike
(realchains/real_cc_backtest.py:220-229 — band at :224, nearest-DTE at :227, cohort at :228, the
nearest-delta pick at :229). Only the calendar moves.

Held fixed across every career: the strike rule, the DTE rule, the daily delta hedge, the hold-to-expiry
exit, and the sizing.

**Why not randomize the strike instead.** An earlier draft of this design did, and the choice was wrong
for two reasons worth recording. First, the strike is not an entry-timing decision — it is a structural
coordinate of the strategy: `target_delta` is literally a grammar axis (the short_vol campaign grid is
`target_delta × dte`, search/edge_search.py:618-621), searched as committed cells under the e-LOND
stream. A randomized-delta ensemble would blur that governed hypothesis space with an ungoverned look —
the exact boundary Gap E drew for exits. Second, different deltas are different risk objects: the vol
smile prices them systematically differently, so a baseline sitting off-center in a random-strike band
would confound entry skill with smile structure and be uninterpretable as either. Strike-randomization
is therefore a **named widening carrying both caveats**, not a v1 axis.

Two further widenings, also out of v1's look count:

- **The CC-path selector override** — `run_real_cc_overlay` has no selector seam: its signature carries
  no `select` parameter (realchains/real_cc_backtest.py:260-266; the only entry seam is
  `suspended_dates`, an on/off gate that cannot change which contract is picked, :267-274), and the
  selectors are hardwired module-level calls (`select_entry` at :345, `select_cap_leg` at :354-355).
  Making the CC entry pluggable requires an engine signature change, so it is its own widening, not a
  v1 rider.
- **Random direction** — meaningless for a defined short-premium overlay; named only to mark the
  distance from Tharp's coin flip.

The honest scope limit, stated once and carried by every pin: **this tests the re-entry cadence — the
unconditional timing axis this engine actually has** — not Tharp's full direction-and-timing coin flip,
and deliberately not the strike. A verdict here is about whether the engine's deterministic
enter-immediately rule adds or costs anything against random timing, nothing more.

## Zero engine changes — the seam evidence

The seam is already general. `run_real_structure_overlay` takes the selector as a keyword-only callable
— `select: Callable[[dict, dict], list[dict] | None]` (realchains/vol_premium.py:777; the bare `*` at
:776 forces keyword passing) — calls it exactly once per flat chain-day as `picked = select(day, params)`
(:870), treats a `None` return as a no-entry day (:871), and applies the entry guard to whatever comes
back (:872-876). The leg contract is stated in the docstring: each leg is
`{sign, right, strike, contract, entry_net, mid, delta, expiration}` (:783-785). (The parent plan cited
the seam at :785/:869; Gap E's build inserted lines, so the anchors above are the current tree's.)

Custom selectors already flow through it in two ways. The spec path binds the built-in `_legs_*`
wrappers via `select=spec['select']` (realchains/vol_premium.py:708-711). And Gap E's mechanics tests
drive the seam with hand-built selectors: `_two_leg_scenario` defines an inline `def select(day,
params)` returning hand-built leg dicts (tests/test_vol_premium.py:1715-1743, the closure at
:1733-1739), and every `TestExitMechanics` case passes it straight in as `select=select`
(:1771-1776, :1802-1807, :1829-1834, :1849-1853, :1875-1880, :1914-1918). A seeded random selector is
one more callable through a proven seam.

The measurement side needs nothing new either. `run_real_short_vol_overlay` is itself a thin delegate
to this same generic engine (realchains/vol_premium.py:125, the delegate return at :138), so the
baseline and the random careers run literally one code path. `short_vol_statistics` consumes only the
returned `daily_equity` — reading its `rf_credit` column — plus the capital and rf (:141, the column
read at :185-189). `build_trade_ledger` consumes the returned trades list plus
`shares = 100 × num_contracts`, and `num_contracts` already rides the engine's summary
(common/trade_ledger.py:150; the idiom at tests/test_trade_ledger.py:81). No summary field, statistic,
or engine branch is missing.

Consequences: no engine diff, no new knob, no default change, and no `STRUCTURE_ENGINE_VERSION` bump —
the bump rule governs changes to overlay or scoring mechanics, and nothing scored moves.

## The selector — jitter, then delegate the pick

The proposed factory:

```python
def random_entry_selector(seed, k=10):
    """Return a STATEFUL select(day, params) callable: wait J ~ uniform{0..k}
    chain-days at the start of each flat stretch, then delegate the pick to
    the deterministic baseline logic."""
```

**The wait.** The closure carries one stdlib `random.Random(seed)` and an invocation counter. The engine
invokes the selector exactly once per flat day that has a chain (`if legs is None: if day is not None:
picked = select(day, params)` — realchains/vol_premium.py:870, so chainless days never reach it). At the
start of each flat stretch — the career's first invocation, and the first invocation after each
non-`None` emission (the closure cannot observe the engine's acceptance, so emission, not acceptance, is
the boundary it keys on; the guard paragraph below records why the two coincide) — the closure draws
`J ~ uniform{0..k}` and returns `None` for its first `J` invocations; a `None`
return is already a plain no-entry day (:871). The wait is therefore counted in **chain-days**, stated
as a convention: a chainless calendar day consumes none of it. A new `J` is drawn per stretch from the
same career RNG, so a career is deterministic in (seed, day sequence).

**The pick, once the wait expires, is the baseline's own.** The closure delegates to the deterministic
`select_entry` logic — the band filter (`bid > 0`, `0.05 < delta < 0.60`,
realchains/real_cc_backtest.py:224; the same band pipeline/validate_dailies.py names, constants at
:49-50), the nearest-|dte−30| expiration (:227), the cohort restriction (:228), the nearest-0.25-delta
pick (:229) — and emits the leg exactly as `_legs_short_vol` would, reading `dte`, `target_delta`, and `fill` from
`params` at call time so no knob is duplicated in the factory, all eight documented keys with
`entry_net = fill − COMMISSION_PER_SHARE` (realchains/vol_premium.py:393-407 — the dispatch at :400,
the leg dict at :405-407). The strike choice is **byte-identical to the baseline's for the same day**;
the careers differ from the baseline only in *which days* they enter. Everything downstream — marking,
settlement, hedging, the ledger — runs identical machinery on genuine chain rows.

**The `k = 0` degenerate is the off-equivalence anchor.** With `k = 0` every draw is `J = 0`, the wait
never fires, and the career reproduces the deterministic baseline trade-for-trade — a synthetic test
pins that equivalence, and the dataset-gated scout re-runs the baseline through the same harness to
assert the published pins before any percentile is computed.

Two no-entry paths are inherited rather than invented. A post-wait day whose band filter comes up empty
returns `None`, exactly as `select_entry` does (realchains/real_cc_backtest.py:225-226) — the career
simply tries the next chain day, with the wait already spent. And the entry guard still applies after
the selector returns: the short-vol spec's `each_short_positive` requires every short leg's
`entry_net > 0` (realchains/vol_premium.py:872-876), so a pick whose fill minus commission is
non-positive skips that day. The closure cannot see that rejection — it keys stretch boundaries off its
own emissions — so its next invocation would open a new stretch and draw a fresh `J` while the engine is
still flat in the same stretch: a bounded, seed-deterministic desync, not a silent redraw of the same
wait. In practice the band makes that path unreachable: a
penny-quoted in-band bid (at least $0.01) clears the $0.0065 per-share commission
(realchains/real_cc_backtest.py:59), so only a sub-penny bid could ever trip it — recorded as a
convention, not patched. The put-side mirror (`select_put_entry`, realchains/vol_premium.py:74-95, band
at :90) gives a future widening the sign-flipped analog for free; v1 is the call wing only.

RNG: one stdlib `random.Random(seed)` per career, per the engine-adjacent precedent (open question 1
records the alternative and the reasoning).

## The measurement plan — Experiment 2 as an ensemble null

**Coordinates.** The pinned SPY short-vol registered coordinates, unchanged: target_delta 0.25, dte 30,
capital $100,000, rf 0.045 (tests/test_vol_premium.py:427-431), the span frozen at
`REGISTERED_CLEAN_START['SPY']` (:420-421), and the frictionless hedge (`hedge_cost_bps=0.0`, :450) so
the hedged statistic shares the +2.54 pin's basis (:451). Each career calls
`run_real_structure_overlay` with the short_vol spec's non-select knobs passed verbatim —
`entry_guard='each_short_positive'`, `hedge_mode='per_leg_sign'`, `management='early_close_single'`
(realchains/vol_premium.py:674-676) — so the selector is the only coordinate that moves.

**The pre-committed ensemble.** N = 20 random-entry careers. The seed constant is committed here, before
any run: `RANDOM_ENTRY_SEED = 20260714`, following the scouts' design-date convention
(`PERMUTATION_SEED = 20260613`, search/explorations.py:74; `CAMPAIGN_SEED = 20260613`,
search/edge_search.py:104). Career *i* uses seed `RANDOM_ENTRY_SEED + i` for *i* in 0..19, recorded per
row — the edge-search per-candidate idiom (search/edge_search.py:447, :449).

**Per-career measures.** Each career's events feed the Gap A ledger (`build_trade_ledger` →
`ledger_statistics`: expectancy_r, win rate, worst MAE-R) and `short_vol_statistics` (the hedged
Newey-West t). One additional baseline pass runs the deterministic selector through the same harness —
\~21 engine passes total — and the scout asserts it reproduces the pins (expectancy_r −0.5407 on n = 174
closed cycles, the 175th being an open dangler the ledger drops, tests/test_trade_ledger.py:381-382;
hedged NW t +2.54, tests/test_vol_premium.py:451), so drift raises an alarm instead of skewing the
percentile.

**The verdict statistic.** Where the pinned baseline sits in its own random band: the percentile of the
delta-targeted run's expectancy_r and hedged NW t among the 20 random careers, computed as
count(career ≤ baseline) / 20 — the cooldown scout's percentile idiom (`perm_percentile`,
search/explorations.py:211) applied to entry skill. The scout returns one cooldown-shaped summary dict
(the precedent at search/explorations.py:216-220): career rows, ensemble mean/min/max, and the two
baseline percentiles.

**The prior, pre-stated so the measurement cannot be spun.** The baseline sits INSIDE the random band —
the deterministic enter-immediately cadence is not where the edge (or anti-edge) lives. Grounds, in
order of weight: Tharp's thesis that entry is the least important component (Loc 890, Loc 4138); the
registered trend gate killed at Stage 1 (D_A measured +$439.44/cycle against a predicted negative, p_A
0.736; D_B −3.07 pp against a predicted positive, p_B 0.763; fails all three gate conditions —
docs/trend_gate_results.md:3, :125-126, :129-131); the post-rip cooldown scout killed with D_A
wrong-signed at every horizon (docs/explorations.md:30, pinned at tests/test_explorations.py:151-183);
and the IV-richness gate killed (docs/explorations.md:89). All three kills were *conditional*-timing
tests, so they ground the weaker unconditional claim only indirectly — which is exactly why this
measurement is worth running.

The honest uncertainty, also pre-stated — and this time it is **directional**. The cooldown scout's
decisive number cuts the other way here: post-rip cycles measured *better* than the rest (D_A +$376 at
N=30 rising to +$1,770 at N=90, the real arrangement in the 0.94–1.00 high tail of its permutation
null — docs/explorations.md:30). The baseline's immediate re-entry is precisely what captures those
post-event cycles; a jittered career delays some of them out of the window. If that texture is real
rather than sampled noise, the baseline should sit in the *upper* part of the band — a mechanism named
before the run, so finding it would confirm pinned prior work rather than invite a story. The
counter-consideration: the same scout found no return memory to time (lag-1 acf −0.126), which argues
the texture is thin. The measurement decides; it is allowed to contradict the prior — Gap E's
measurement contradicted half of its own.

**What this cannot show, pre-stated.** Twenty careers give 5-percent percentile resolution: a coarse
locate, not a significance claim. Under exchangeability the baseline lands above the 20-career maximum
with probability 1/21 (\~4.8 percent), so envelope exclusion is roughly a 5-percent one-sided event —
scout resolution, nothing finer. The per-career NW t stays descriptive throughout: one hypothesis is
tested (does the deterministic cadence sit outside its own random band?), never twenty. And the
cadence-differs-by-design fact carries two stated consequences. First, jittered careers trade **fewer
cycles** than the baseline's 174 — with `K = 10` against a \~21-trading-day cycle, an expected wait of
\~5 chain-days per stretch cuts roughly a fifth off the trade count (the exact per-career `n` is
recorded and pinned as a range at run time) — so the comparison units are per-trade (`expectancy_r`) and
per-day (the NW t), never cumulative dollars: the abstinence-confound trap the cooldown entry documents
(docs/explorations.md:30) is real here, and the statistic choice is what keeps it inert. Second, the
hedged NW t has a mechanical dilution component across careers — more flat days means more zero
vol-P&L days in the daily series — so `expectancy_r` carries the primary percentile and the NW t
percentile is reported as the secondary, descriptive locate. Careers run one span of one ticker, so the
verdict is about this overlay on these chains. The LeBeau 45–55 band itself does not transfer — it is a
directional-entry frame; the short-premium analog is the ledger win rate, where the prior expects all
21 runs to cluster in the same high-win-rate, negative-expectancy corner (the Loc 4138 lesson made
visible).

## Not an FDR search — the boundary argument, reused

Gap E set the precedent that this boundary is an argument, not a label
(docs/van_tharp_gap_e.md:350-359), and the argument transfers whole. The e-LOND stream's unit of account
is a grammar cell: a `StructureCandidate` recorded to the committed `idea_ledger.jsonl`, drawn from the
countable pre-specified lattice. A random selector cannot be such a cell — it never constructs a
`StructureCandidate`, and its seed is not a grammar coordinate (`_validate_grammar` raises on any
unknown knob), so nothing here can enter the stream even by accident. Nothing enters
`idea_ledger.jsonl`; no e-value is spent.

The sharper point: the 20 careers are a **null distribution, not 20 hypotheses** — the same epistemic
object as the cooldown scout's 1,000 permutation draws (search/explorations.py:75), which never counted
against the stream either. The look count is one.

The v1 axis choice reinforces the boundary rather than straining it: the jitter varies a quantity no
grammar coordinate touches (the entry calendar), while the rejected strike-randomization would have
varied `target_delta` — a governed grammar axis whose committed cells the campaign already searches
under e-LOND (search/edge_search.py:618-621). Keeping v1 off that axis is what lets the
null-distribution argument stand without an asterisk.

One deviation from the parent plan is named rather than papered over: the plan's honesty rails list "the
random-entry batch (Exp 2)" among sweeps that ride the e-LOND stream (docs/van_tharp_test_plan.md:294).
That rail anticipated a batch with selection — many random variants, winners chosen. This design commits
to the narrower object: a fixed 20-seed ensemble, no selection on results, every outcome pinned. The
reclassification trigger is explicit: if the ensemble grows after results are seen, seeds are re-drawn
or cherry-picked, or careers are selected on outcome, the exercise becomes a campaign and belongs under
the e-LOND stream like any other automated search.

## Code home and pinning home

**Choose by code home, not run type.** Gap E's pinning-home precedent divides on re-tag versus engine
re-run (docs/van_tharp_gap_e.md:361-370), and an ensemble is engine re-runs — read literally, that
points at a dataset-gated class in tests/test_vol_premium.py beside the short-vol pins. But unlike Gap
E's six variant cells, an ensemble needs an orchestrator — the seed derivation, the N careers, the
percentile aggregation — and the repo's only precedent for orchestration code is the scout module:
`cooldown_scout`'s logic lives in search/explorations.py (:167) and its test pins only the decisive
outputs (tests/test_explorations.py:136-191). So the code goes where scout code goes, and the pins live
beside the code:

- `random_entry_selector(seed, ...)` and `random_entry_scout(...)` in search/explorations.py, the scout
  reusing the module's loading conventions and returning the pinned summary dict.
- A dataset-gated class in tests/test_explorations.py, on the module-fixture pattern the cooldown class
  models (the dataset-gated fixtures at tests/test_explorations.py:115 and :123).
- A docs/explorations.md entry in the idea / how-tested / verdict / trap shape.

The cost is named: the ensemble pin sits one file away from the +2.54 baseline
(tests/test_vol_premium.py:451). The scout's baseline re-run assertion is the mitigation — the
comparison is recomputed and cross-checked against the pin inside the scout, not assumed across files.
The offsetting benefit: tests/test_trade_ledger.py, the other half of the baseline (−0.5407, n = 174),
is already co-bucketed with tests/test_explorations.py in CI.

**The CI runtime consequence.** The new pins ride the trend-explore bucket (tests/test_trend_gate.py +
tests/test_explorations.py + tests/test_trade_ledger.py, .github/workflows/ci.yml:195), the lightest of
the three scout buckets; ci.yml's balancing comment says overall wall-clock is the slowest bucket,
\~test_vol_premium (:175-179), and the measured 2026-06-24 figures from the #87/#88 CI-perf work put the
slow jobs bunched at \~6.5–8 minutes (vol-premium \~7m45s). The ensemble is \~21 registered-span SPY
passes; at \~20 s per pass — an inference from the vol-premium bucket's pace, not a measured per-pass
figure — that is \~7 minutes added to the light bucket (the store loads once, in the module fixture, on
the \~7 GB single-store runner, ci.yml:24-26). Whether trend-explore absorbs that without becoming the
new bound cannot be settled from here: no measured figure for the trend-explore bucket exists. The
commitment is procedural — **measure the bucket when the pins land**, and rebalance ci.yml only if
trend-explore becomes the slowest bucket.

## Test plan

Always-run synthetic tests (a crafted day sequence, no dataset):

- The `k = 0` degenerate reproduces the deterministic baseline trade-for-trade on a synthetic scenario —
  identical trades and equity (the off-equivalence anchor).
- The wait is honored and chain-day-counted: a career with `J = 2` drawn enters on its third flat
  chain-day, and a chainless calendar day mid-wait consumes none of the wait (the selector is never
  invoked, realchains/vol_premium.py:870).
- A new flat stretch draws a new `J`: two cycles under one seed can wait different lengths, and the
  sequence is deterministic in (seed, day sequence) — the same seed reproduces it, a different seed
  breaks it.
- The post-wait pick equals `select_entry`'s pick field for field — the eight-key leg contract
  (realchains/vol_premium.py:783-785) with a genuine contractID, the band and DTE cohort logic
  delegated, not reimplemented divergently.
- A post-wait day with no band candidates yields `None`, exactly as `select_entry` does
  (realchains/real_cc_backtest.py:225-226), and the engine records a plain no-entry day with the wait
  already spent.
- A guard-rejected emission (a crafted sub-penny-bid day) re-arms a fresh `J` on the next invocation —
  the emission-keyed desync convention, pinned synthetically since the real-chain band cannot reach it.

Dataset-gated ensemble pins (proposed class name `TestRandomEntryScout`, final at build time):

- The baseline re-run matches the existing pins (expectancy_r −0.5407 / n 174; hedged NW t +2.54).
- The 20 career expectancies' mean, min, and max are pinned, and the per-career trade counts are pinned
  as a range.
- The two baseline percentiles (expectancy_r primary, hedged NW t secondary) are pinned.
- The ensemble's hedged-NW-t band (min and max) is pinned.

The ensemble size and seeds are pre-committed in this document: N = 20, `RANDOM_ENTRY_SEED = 20260714`,
career seeds `RANDOM_ENTRY_SEED + i` for i in 0..19.

## Cross-surface obligations (when code lands, not now)

- Symbol-sweep regex: add
  `random_entry_selector|random_entry_scout|RANDOM_ENTRY_SEED|TestRandomEntryScout` (plus the synthetic
  class name if separate — final names at build time) in the same change that lands the symbols, per the
  CLAUDE.md rule.
- README: a project-layout row for this doc; the explorations trio's rows already exist.
- ci.yml: no edit expected — the new tests ride tests/test_explorations.py, already on the trend-explore
  include line (:195). The bucket-rebalance decision happens at land time with measured numbers, per the
  runtime section above.
- Notebook: no regen — nothing here touches tutorial_covered_call_backtest.md or engine/make_figures.py.
  If either is somehow touched, the regen rule applies as usual.
- `STRUCTURE_ENGINE_VERSION`: unchanged, with the reasoning stated in the PR — zero engine edits, so no
  scored quantity can move.
- docs/explorations.md: the verdict entry lands with the pins, whatever the verdict is.

## Honesty rails

- **Exploratory throughout.** Every number is sample-spending, kill-or-justify; the prior is pre-stated
  above so the result cannot be narrated after the fact; the percentile is a locate, not a significance
  claim.
- **The look count is bounded by pre-commitment.** Twenty seeds, fixed in this doc, all outcomes pinned
  regardless of verdict. Growth, re-draws, or outcome-selection reclassifies the work as a campaign
  under the e-LOND stream.
- **No FDR interaction in v1.** Nothing enters `idea_ledger.jsonl`; a baseline that lands outside the
  band escalates to a human-signed registration, never to a headline.
- **One significance authority, used descriptively.** The hedged Newey-West t from
  `short_vol_statistics` is reported per career and never gated on; the ledger statistics are reported,
  never gates — the ledger's own posture.
- **Convention caveats travel with the pins.** The wait is drawn uniform{0..K} and counted in
  chain-days, so chainless calendar days consume none of it. Jittered careers trade fewer cycles by
  design — per-trade and per-day units carry every comparison, never cumulative dollars. A pick failing
  the entry guard skips the day, and the emission-keyed closure would then re-arm a fresh `J` for the
  same engine-side stretch — a bounded desync the band makes unreachable (only a sub-penny bid could
  trip the guard). The hedged statistic is frictionless to match the
  +2.54 pin's basis, and the ledger-versus-hedged basis caveat carried by the baseline pins
  (tests/test_trade_ledger.py:348-355) applies to every career identically.
- **Scope honesty.** The pins say what was randomized: the entry day, within 0..K chain-days of each
  flat stretch — at fixed strike rule, DTE, direction, exit, and sizing. Not the strike (a governed
  grammar axis, deliberately untouched), and not Tharp's coin flip.
- **The disclaimer is stated once, in Status,** and covers every number this design will produce.

## Open questions

1. **RNG convention — stdlib `random.Random` versus `np.random.default_rng`.** The scout side uses
   `default_rng` exclusively (search/explorations.py:187, :323; search/edge_search.py:447); the
   engine-adjacent side uses one `random.Random(seed)` per call, documented as the `monte_carlo_shuffle`
   precedent (common/position_sizing.py:31, :102). Leaning: **stdlib for the selector** — it rides the
   engine seam and is engine-adjacent code, the draw is a single uniform choice with no need for numpy's
   generator, and the precedent for engine-adjacent randomness is already stdlib. The cost is that
   search/explorations.py will carry both styles in one module; a comment at `RANDOM_ENTRY_SEED` should
   say why, so the split reads as a decision rather than drift.
2. **Ensemble size — N = 20 versus 50.** Leaning: **20.** Five-percent percentile resolution suffices
   for a scout verdict (the envelope-exclusion arithmetic above), and \~21 passes ≈ \~7 estimated
   minutes is an addition the light CI bucket has a chance of absorbing; 50 careers would roughly
   triple that (\~17 minutes), likely making trend-explore the CI bound outright, for one notch of
   resolution on a kill-or-justify question.
3. **The jitter bound K — 10 versus other values.** Leaning: **10.** Roughly half the
   \~21-trading-day cycle: an expected \~5-chain-day wait moves entries far enough off the baseline
   calendar to matter while costing only \~a fifth of the trade count. A small K (2–3) would barely
   perturb the calendar and risk a vacuous null; a large K (20+) guts per-career n and turns the
   comparison into a mostly-flat-account artifact. If the ensemble band comes back suspiciously tight,
   re-running at a larger K is a NEW pre-committed exercise, not a quiet retune of this one.
4. **Per-career intratrade-ruin replay (the C+B machinery).** Leaning: **no for v1.** The ledger
   statistics and the hedged NW t carry the entry-skill verdict; ruin curves add runtime without
   changing the question. The per-career ledgers are recorded, so the replay stays available to a
   follow-up without re-running the engine.

## Related

- [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the parent plan whose Gap F section
  (:202-218), sequencing row (:267), and Experiment 2 row (:275) this doc designs, and whose e-LOND rail
  (:294) the boundary section narrows
- [docs/van_tharp_gap_e.md](van_tharp_gap_e.md) — the seam's mechanics-test evidence and the
  boundary-argument and pinning-home precedents this design reuses
- [docs/van_tharp_gap_a.md](van_tharp_gap_a.md) — the R-multiple ledger every career is measured through
  (merged, #125)
- [docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md) — the sizing replay open question 3 defers (BUILT per
  its Status)
- [docs/van_tharp_gap_d.md](van_tharp_gap_d.md) — the six-regime bucketing available to slice career
  ledgers later (merged, #126)
- [docs/explorations.md](explorations.md) — the scout pattern, the recurring entry-conditioning lesson,
  and the log-entry home for the verdict
- [docs/trend_gate_results.md](trend_gate_results.md) — the registered entry-conditioning kill in the
  prior
- [docs/edge_search.md](edge_search.md) — the campaign and e-LOND rules the boundary section respects
- research/book-notes/trade-your-way-to-financial-freedom.md — the highlights file every Location above
  cites
