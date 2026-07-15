# Gap E — exit mechanics beyond hold-to-expiry (DESIGN / build spec)

## Status

**Implementation status (2026-07): E1 and E2 are BUILT** — the general exit branch in
`run_real_structure_overlay` (dispatch, per-trade arm rule, all-legs-quoted triggers, target→stop→time
priority, the `reason` key) with the synthetic `TestExitMechanics` layer, and the pre-committed six-variant
Experiment 4 grid pinned by `TestSpyExitVariantExploration` plus a
[docs/explorations.md](explorations.md) entry. Byte-identity held: the full pinned vol-premium,
edge-search, and ledger suites pass unchanged, and `STRUCTURE_ENGINE_VERSION` stays `'v1'` per the bump
rule. **The measurement contradicted the pre-stated prior in half**, as the design allowed: on the
delta-hedged short call the 2× stop improves expectancy (−0.54R → −0.18R), truncates the worst MAE
(−11.41R → −3.12R), and lowers intratrade P(ruin) at f=2% (0.992 → 0.835) — the hedge had already absorbed
the trend the CC's stop kept firing into. The other half held: no variant flips the sign — every exit
leaves the game negative. E3 (escalation) remains unexercised and human-gated. Code file:line references
in the sections below describe the tree as of the design commit; the implementation inserted lines, so
some have shifted.

This was written as a **DESIGN document — a build spec, PLAN-level**, ahead of the code. It designs Gap E from
[docs/van_tharp_test_plan.md](van_tharp_test_plan.md): exit mechanics beyond hold-to-expiry for the
structure engine. The parent plan sizes it Large and its sequencing table calls it "the heaviest lift"
(docs/van_tharp_test_plan.md:200, :266): it is the first Van Tharp change that must enter the structure
engine's hot loop — `run_real_structure_overlay` (realchains/vol_premium.py:770), the function every
pinned structure number runs through — where Gaps C+B could land as a zero-engine replay layer
([docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md), "The central design decision"). It enables
**Experiment 4**: does varying the exit on a fixed entry change expectancy, and does a tail-stop convert
the negative-skew MAE tail into something survivable?

Predecessors are all in: Gap A (the R-multiple ledger, [docs/van_tharp_gap_a.md](van_tharp_gap_a.md),
merged in #125), Gap D (the six-regime R-distributions, [docs/van_tharp_gap_d.md](van_tharp_gap_d.md),
merged in #126), and Gaps C+B (the fixed-fractional replay + marble-bag resampler,
[docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md), BUILT per its Status).

Every number this design will produce is **EXPLORATORY** — sample-spending, kill-or-justify, never a
registered verdict. The work is descriptive research measurement of historical simulation output; it is
not investment advice, and no figure in it is a recommendation to trade any instrument, any exit rule, or
any size.

**The honest prior is pre-stated.** The repo has already tested an exit of exactly this family once, on
the real-chain covered call, and rejected it: `TestMsftStopLossRegression`
(tests/test_real_cc_backtest.py:1150) pins a premium-multiple stop leaving the CC worse than the no-stop
baseline at every level swept, monotonically worse as it tightens — the same lesson blog post 6 narrates
for the delta-hedge refinement (blog/06_real_chains_flip_the_268000.md). The expectation this design
commits to before running anything: a daily-close stop should truncate the MAE tail — it bounds further
adverse excursion once it fires, though a gap through the level still fills at that day's quote, so
truncation is expected rather than guaranteed — while the CC evidence says the whipsaw cost exceeds the
tail protection on these chains, Tharp's own warning that whipsaw losses add up until the trader abandons
the strategy (Loc 2043). The measurement records whether the premium structures repeat that verdict; it
is allowed to contradict the prior.

**Location convention.** Book references are Kindle Locations, per the notes file's own `### Location N`
headers (research/book-notes/trade-your-way-to-financial-freedom.md; Van K. Tharp, *Trade Your Way to
Financial Freedom*, 2nd ed., McGraw-Hill, 2007, Kindle). The parent plan's approximate pointers were a
mix of Kindle Locations and notes-file line numbers: the exit-thesis and whipsaw pointers (1867, 2043)
are true Locations, the trailing-stop pointer resolves to Loc 1392 (a Basso interview line) with the
retracement mechanics at Loc 6314, and the scale-out anti-pattern resolves to Loc 6430.

## Why — Experiment 4 and the book's exit thesis

Exits are the half of a system Tharp says actually carries the expectancy. His claim is direct:
"expectancy is controlled by your exits" (Tharp, Loc 1867), and the best systems in his view run three or
four different exits (same passage). He repeats the theme from every angle: most of a trader's emphasis
belongs on stops and exits rather than entries (Loc 1481); the cut-losses-let-profits-run rule is about
exits, not entry (Loc 1129, restated at Loc 1131); the foreword's summary of the whole book puts making
money in how you exit — limiting losses when wrong, managing winners when right (Loc 198); and exits
decide both whether a trade profits and by how much (Loc 6231).

The engine currently tests almost none of that. Six of the seven `STRUCTURE_SPECS` entries carry
`management='hold'` — straddle, iron_condor, strangle, risk_reversal, credit_spread, calendar
(realchains/vol_premium.py:672-694; the six hold entries at :677, :680, :683, :686, :689, :692). The
seventh, short_vol, carries `management='early_close_single'` (:674), but that branch evaluates only
`legs[0]` (:939) and never fires in the committed campaign anyway: the kill gate passes only grammar
coordinates plus capital (`{**cand.params_dict(), 'capital': capital}`, search/edge_search.py:988-989),
so `close_at_pct` stays at its `None` default (vol_premium.py:820) and `manage_deep_itm` stays `False`
(:821), and both the profit-target test and the deep-ITM test are inert. No test or exploratory script
passes `close_at_pct` to a structure overlay either — the knob's only callers in the repo are the CC
engine and its walk-forward grid — so every pinned structure result in the repo is hold-to-expiry in
practice. Experiment 4 asks whether that omission matters: holding the pinned entry fixed and varying
only the exit, does the ledger's expectancy move, and does the MAE tail truncate?

The one existing data point runs against the book, and it is pinned. `TestMsftStopLossRegression`
(tests/test_real_cc_backtest.py:1150-1254) sweeps a premium-multiple stop on the real-chain MSFT covered
call: the 10-year 2x stop turns the no-stop baseline's −$183,552.34 net overlay P&L into −$251,775.94
(:1193; the ordering against the baseline is asserted at :1229), fires 118 stop closes in ten years
(:1198) against the baseline's 54 deep-ITM closes, *raises* max drawdown to 50.74% (:1202 — the
docstring names the stock leg, not the short call, as the drawdown driver), and leaves the Newey-West t
at −1.58, still no edge (:1203). Every swept level is worse than the no-stop baseline, and tightening is
monotonically worse — 1.5x pins −$374,845.50 and 3.0x pins −$232,950.60, with the full ordering asserted
(:1226-1229) — and the 16-year run repeats the shape (−$514,187.18, 168 stops, t −1.14, :1212-1215).
Even the walk-forward cannot tune around it: chained OOS drops from +184.97% to +154.53% and the
optimizer retreats to `close_at_pct` 1.00 — the loosest exit on its grid — in 11 of 11 windows
(:1231-1254). The test's docstring names the mechanism: on a relentlessly trending stock the stop is
"whipsaw machinery" — a 0.25-delta call doubles on a moderate rally, each fire locks in \~1x premium
plus a spread crossing, and the engine re-sells into the same rally. That is Tharp's own trend-follower
warning (Loc 2043) plus his transaction-cost half: tight stops raise costs sharply (Loc 6015), missing
re-entry compounds the damage (Loc 6024), and larger stops are generally better on costs (Kaufman's
observation, Loc 6155).

The book still argues the other side — the initial stop is what defines R and the benchmark for gains
(Loc 5740, 5756, 5770; restated at Loc 6177), with a stated tight-versus-wide tradeoff (Loc 6182, 5775,
6014) — so the honest framing is a measurement, not a foregone kill: the stop's tail truncation is
expected, its net cost on these chains is the question, and the CC prior says the cost wins.

## The central design constraint — default-off, byte-identical-when-off

Every pinned number on the repo's most valuable surfaces runs through `run_real_structure_overlay`
(realchains/vol_premium.py:770, called via `run_structure_via_spec` :697-711):

- The 56-cell campaign ledger and the lifetime e-LOND stream are pinned by `TestStructureCampaign`
  (tests/test_edge_search.py:1426-1459, 0 of 56 flagged) and the always-run `TestStructurePhase`
  (:1311).
- The registered and exploratory VRP/straddle regressions run through it: `TestSpyShortVolRegression`
  (tests/test_vol_premium.py:392; the frozen `REGISTERED_CLEAN_START` span is noted at :394-398),
  `TestSpyShortPutRegression` (:481), `TestIwmShortPutRegression` (:564), `TestSpyStraddleSecondary`
  (:734), `TestIwmStraddleSecondary` (:793), `TestQqqShortVolRegression` (:861),
  `TestIwmShortVolRegression` (:909), `TestMsftShortVolRegression` (:956), and
  `TestSpyIronCondorExploratory` (:1194).
- The equivalence and spec-table pins cover every delegate: `TestGenericStructureEngineEquivalence`
  (tests/test_vol_premium.py:1543-1581, one test per structure plus the NVDA iron condor at
  :1578-1581) and the always-run `TestGenericStructureEngineSpecs` (:1427), which pins the
  `management` values themselves (:1433-1447).
- The grammar-signature cross-check `TestGrammarSignatureMatchesEngine`
  (tests/test_vol_premium.py:1668-1686) re-derives each declared signature from engine greeks.

The exit machinery therefore lands as **new optional params that no pinned caller passes**, with a guard
structure that makes the off path byte-identical. The precedent is already in the engine twice: the
calendar's staggered-settlement branch is guarded by a chained comparison that is unreachable for
single-expiration structures (`min(leg expirations) == expiration` makes the condition false,
vol_premium.py:894 with the note at :899-902), and the combined-hedge generalization kept the all-short
callers byte-identical — both landed with byte-identity verified by the existing pins.

Three hard rules follow:

- **`early_close_single` is not touched or generalized in place.** The new general manage branch is
  parallel to it inside the same mark+manage arm, and short_vol's pinned spec keeps
  `'management': 'early_close_single'` verbatim (realchains/vol_premium.py:673-675) — its pins protect
  it, including the spec-table pin at tests/test_vol_premium.py:1433-1447.
- **The dispatch rule is explicit, because `close_at_pct` already has a meaning.** The general branch
  arms iff `stop_loss_mult` or `exit_dte` is set (the two new knobs), or `close_at_pct` is set on a
  `management='hold'` structure — a combination that today is a silent no-op (the knob is read at :820
  but nothing under `'hold'` consumes it). When it arms on an `early_close_single` structure —
  Experiment 4's own SPY short_vol runs do exactly this — the general branch takes over the whole exit
  evaluation for that run and the legacy single-leg block is bypassed, so the two paths can never
  double-fire. When it does not arm, every line of today's code runs verbatim, including
  `early_close_single`'s single-leg `close_at_pct` semantics. Byte-identity for every pinned run
  follows from recon, not hope: the campaign passes grammar coordinates plus capital only
  (search/edge_search.py:771-785, :988-989), and no test or script passes `close_at_pct` to any
  structure overlay, so no pinned caller can reach the armed path.
- **`STRUCTURE_ENGINE_VERSION` stays `'v1'` for the landing change** (search/edge_search.py:554). The
  bump rule (:548-553; docs/edge_search.md:180) targets changes that recompute a different t-stat for
  the same data at the frozen defaults. A default-off knob leaves every committed cell's result
  byte-identical, so no bump. An exit-variant *run* is a new measurement, not a re-scored old one.

The existing pinned suites are the real-data byte-identity guard; an explicit synthetic off-equivalence
test is added anyway (test plan below), so the guarantee does not depend on datasets being present.

## The exit set

Tharp's prescription is multiple exits kept individually simple (Loc 6409, with the simplicity rationale
at Loc 6407 and a worked three-exit example at Loc 6417). Mapped onto premium structures, v1 is three
close-triggers plus a roll convention, with two deliberate exclusions.

### Profit target — `close_at_pct`

Buy the whole structure back when the net close cost falls to `(1 - close_at_pct)` times the entry
credit or below. This is the multi-leg generalization of two existing single-leg tests: the structure
engine's own `short_buy <= leg['entry_net'] * (1 - close_at_pct)` (realchains/vol_premium.py:943-944)
and the real CC's `hit_target` against `premium_collected * (1 - close_at_pct)`
(realchains/real_cc_backtest.py:470; the CC's 0.75 default is set at :276). Tharp's frame: profit-taking
exits exist to raise the reward-to-risk ratio of the system (Loc 4147), with a take-or-tighten rule once
a multiple-of-R objective is reached (Loc 6339).

### Stop loss — `stop_loss_mult`

Close when the net close cost rises to `stop_loss_mult` times the entry credit or above — the CC's
`hit_stop` (realchains/real_cc_backtest.py:472-473, `None`/absent = off per :287-288) generalized to
the whole structure. Tharp's frame: the protective stop is what defines the initial risk R and the
benchmark against which gains are measured (Loc 5740, 5756, 5770; Chapter 10 summary at Loc 6177), and
every system needs a disaster stop to preserve capital (Loc 4144). The CC pin's convention caveat
travels with any new pin: this is a stop-market evaluated on daily closes — a gap through the level
fills at that day's quote, not the stop level — so measured stop costs flatter the stop if anything.

For a net-debit structure the credit-referenced trigger is ill-defined (there is no positive entry
credit for the multiple to reference). **Resolution: triggers arm per trade only when the booked
`entry_credit` (vol_premium.py:868) is positive.** That excludes the calendar structurally — the one
committed structure that is net-debit by construction (the long far-month leg costs more than the
near-month credit), and the only one whose mid-life staggered settlement (:894-916) skips the manage arm
on the settle day. It also covers the edge the spec table hides: the risk reversal's
`each_short_positive` guard checks only its short put (:685-687), so a net-debit risk-reversal entry is
possible in principle — the arm rule skips such a trade rather than referencing a negative credit. v1's
measured scope is the net-credit structures; the eventual debit-side basis is the absolute entry
quantity Gap A already computes as the R floor (`_premium_collected_per_share`, which reads `premium`,
`credit`, or `legs_detail` off the entry event, common/trade_ledger.py:124-147), and it is a named
widening.

### Time exit — `exit_dte`

Close N calendar days before the structure's final expiration (`expiration` is the max leg expiration,
booked at entry, vol_premium.py:880). Cheap, deterministic, and the building block for the roll. The
name follows the grammar's DTE vocabulary; `max_hold_days` (anchored to entry instead of expiry) is the
rejected alternative, since the premium cycle is expiry-anchored throughout the engine. Tharp treats
time stops as a standard tool: exit after a fixed time without profit (Loc 6095), the each-day-a-new-day
formulation (Loc 6097), his caveat that they suit short-term traders — which a 30-DTE premium cycle is —
rather than long-term position holders (Loc 6099), plus a loss-side two-day variant (Loc 6248) and their
place in his stop taxonomy (Loc 6185).

### The roll — a time exit plus the existing re-entry cadence

No new machinery. Step 2 of the day loop is one mutually exclusive if/elif chain, so the close day
itself cannot re-enter; once a close sets `legs = None`, the entry branch (`if legs is None:` at
vol_premium.py:859) fires on the next chain day where the selector returns a pick and the entry guard
passes (:860-866). A roll is therefore `exit_dte` days early plus re-entry at the next eligible day's
chain. The minimum one-day gap is a stated convention of the measurement; same-day roll machinery is a
named widening, not v1 scope. This single decision removes most of Gap E's original "Large" sizing in
the parent plan: exits are close-triggers only, and entry logic is untouched.

### Exclusions

- **Trailing stops** (Basso's usage, Loc 1392; the percent-retracement mechanics, Loc 6314, with
  Tharp's buy-and-hold-substitute endorsement at Loc 6322 and the trailing-volatility worked example at
  Loc 6417) are out of scope for v1 and a named widening. They require per-structure favorable-excursion
  high-water state in the hot loop, and the premium structures' winning path is credit decay rather than
  a trending price the trail was designed for.
- **Scale-outs** are deliberately excluded on Tharp's own reasoning: he calls the scale-out exit a
  reversal of the golden rule — it guarantees the largest position is on when the largest losses hit and
  the smallest when the largest gains arrive (Loc 6430). It would also break the
  one-position-at-a-time invariant that Gap C+B's replay validity rests on
  (docs/van_tharp_gap_cb.md, "Replay validity preconditions").

## Multi-leg close mechanics

The close is the entry's mirror, leg by leg.

- **Close cost.** The signed per-share sum: buy each short leg back at the ask (`q[1]`) and sell each
  long leg at the bid (`q[0]`) under `fill='bid_ask'`, mids (`q[2]`) for both under mid fill, with
  `COMMISSION_PER_SHARE` (0.0065, real_cc_backtest.py:59) charged per leg — so
  `close_cost = Σ_shorts(ask + commission) − Σ_longs(bid − commission)`. The marks tuple is
  `(bid, ask, mid, delta)` keyed by contractID (realchains/real_cc_backtest.py:164, :216), and the side
  convention mirrors the entry selectors — shorts fill at bid, longs at ask on the way in (the docstring
  at realchains/vol_premium.py:433, fills at :442-445; per-leg commission is sign-dependent at entry,
  :448-454) — flipped for the close. Triggers compare the *ex-commission* sum to the entry credit,
  mirroring both precedents: the CC's `close_ref` excludes commission (real_cc_backtest.py:452-466, the
  separate `close_commission` at :462-466), and `early_close_single` tests raw `short_buy` (:942-944)
  and adds commission only on the fill (:948). P&L books as `(entry_credit − close cost) × shares`, the
  sign convention of :950-953.
- **Trigger evaluation rule.** Triggers are evaluated only on days when *every* leg has a live quote in
  `day['marks']` — `all(day['marks'].get(leg['contract']) is not None for leg in legs)` — the
  conservative generalization of `early_close_single`'s `q is not None` guard (vol_premium.py:941) and
  the per-leg refresh guard (:934-937, under which an unquoted leg carries its prior mid forward). The
  refresh stores only mid and delta on the leg, so the branch re-fetches each leg's full quote from
  `day['marks']` exactly as the single-leg block re-fetches its own (:940); the time exit needs no quote
  to be *true* but the same all-legs-quoted rule gates its fill. The alternative precedent is the CC cap
  leg's carried-quote fill (`position['cap_quote']` refreshed when the cap prints and otherwise carried,
  real_cc_backtest.py:439-446); it is rejected for v1 because a carried quote on a far wing can be days
  stale, and a trigger firing on stale marks manufactures fills that never existed. The cost of the rule
  is stated: triggers under-fire relative to a live book (a day where one wing does not print cannot
  close), a conservative bias carried with the pins.
- **Same-day priority.** Target, then stop, then time — deterministic, following the CC's precedent
  `action = 'close' if hit_target else 'close_stop' if hit_stop else 'close_itm'`
  (real_cc_backtest.py:479-480).
- **Event payload.** The action stays `'close'`, with a new `'reason'` key (`'target'` | `'stop'` |
  `'time'`). Verified against the ledger: `TERMINAL_ACTIONS` already contains `'close'`
  (common/trade_ledger.py:57), and `build_trade_ledger` reads only `action`, `pnl`, `mae`, and `date`
  from a terminal event (:181, :187, :206, :209) — unrecognized keys are never read, so the reducer and
  every Gap A/D statistic flow unchanged. The inverse is the hazard and is why the action string does
  not change: an action outside `TERMINAL_ACTIONS` is silently skipped (:185-186), which would orphan
  the entry.
- **Loop placement.** The new branch lives exactly where `early_close_single` lives — inside the
  mark+manage arm (`elif day is not None:` at vol_premium.py:933), the last branch of the day loop's
  mutually exclusive step-2 chain (entry :859, staggered settle :894, final settle :917, mark+manage
  :933) — so expiry-day settlement always preempts it, structurally rather than by convention. A
  staggered-settle day likewise skips the arm entirely, so a two-expiration structure cannot be
  exit-evaluated that day — moot for v1, which excludes the calendar, but recorded for the widening.
- **Bookkeeping.** Cash, wins/losses, and the trade event mirror the existing close path: cash debited
  by the close cost times shares (:949), wins/losses on the sign of net P&L (:950-951), the `'close'`
  event carrying `pnl` and the running-MAE `'mae'` (:952-954, `worst_unrealized` maintained at
  :985-990), then `legs = None` so the hedge step immediately unwinds to target zero (:955-956, hedge
  :958-978).

## The grammar boundary — exits are engine params, not grammar coordinates

The recon check the parent plan asked for comes back clean: `close_at_pct` and `manage_deep_itm` are
*not* grammar axes for short_vol or any overlay — short_vol's grid is `target_delta × dte` only
(search/edge_search.py:618-621) — and both knobs already live in the engine-param tier, read via
`params.get` alongside `fill`, `capital`, `risk_free_rate`, and `hedge_cost_bps`
(realchains/vol_premium.py:816-821). The new exit knobs (`stop_loss_mult`, `exit_dte`) join that tier.
The boundary is hard, not stylistic: `_validate_grammar` requires a candidate's params to match the
overlay's grid knobs exactly — none missing, none extra, type-strict grid membership
(edge_search.py:719-743, the exact-knob rule at :735-740) — so a campaign candidate carrying an exit
knob raises at construction.

One honest asterisk: "exits are never searched" is not a repo-wide law. The real-CC walk-forward grid
*does* treat `close_at_pct` as an optimizable axis ({0.50, 0.75, 1.00},
realchains/walk_forward_real.py:61-64) — which is exactly how the stop-loss pin could show the optimizer
retreating to 1.00. The boundary this section draws is specific to the structure campaign: the grammar
lattice carries no exit axis for any overlay, and the e-LOND stream counts grammar cells, so exit knobs
sit outside the governed hypothesis space by construction rather than by convention.

Consequences:

- No grammar widening, no new campaign cells, no e-LOND spend. Experiment 4 runs as exploratory variant
  measurements on the Gap A ledger overlays, pinned under the explorations pattern rather than judged in
  the FDR stream.
- The distinction is honest because the two tiers are different epistemic objects: grammar coordinates
  are the countable, pre-specified hypothesis space the lifetime FDR stream governs
  (`grid_universe_size` plus the committed batch), while exit variants are diagnostics on samples the
  pinned baselines already spent, loudly labeled and pinned as exploration.
- The escalation path is explicit: a variant that looks promising escalates to a grammar widening plus a
  registered cell — human-signed, spending e-LOND budget in the lifetime stream — never silently
  (phase E3 below).

## Experiment 4 — the measurement plan

**Fixed entry.** The pinned SPY short vol at `target_delta 0.25 / dte 30` — the committed
`short_call_25` coordinates (search/edge_search.py:772) and the exact run behind the Gap A SPY ledger
(`run_real_short_vol_overlay` at tests/test_trade_ledger.py:102-104; the regression is
`TestSpyShortVolRegression`, tests/test_vol_premium.py:392). Under the dispatch rule above, the stop and
time variants arm the general branch; the `close_at_pct`-only variants ride the legacy single-leg path
(identical semantics for the one-leg short_vol — the dispatch-edges mechanics test proves the same close
date and P&L on both paths). The MSFT CC side needs no new stop
runs — its stop grid is already pinned by `TestMsftStopLossRegression` — so the CC contributes at most a
time-exit variant if E2 finds it worth a run; the primary subject is the structure engine.

**Pre-committed grid, one exit varied at a time** (six variants plus the pinned baseline, committed here
to bound the look count):

- `close_at_pct` in {0.50, 0.75} — 0.75 is the CC engine's own default (real_cc_backtest.py:276); 0.50
  is the half-decay convention.
- `stop_loss_mult` in {2, 3} — these are the CC pin's central and loose levels
  (tests/test_real_cc_backtest.py:1187-1229); 1.5 is omitted because the CC already pinned tightening as
  monotonically worse.
- `exit_dte` in {7, 14} — these cover the final week and the final half of a 30-DTE cycle.

**Measured through the Gap A ledger** (`build_trade_ledger` / `ledger_statistics`,
common/trade_ledger.py), against the pinned baseline: the change in `expectancy_r`, the change in win
rate, the MAE-R distribution — does the tail truncate below the pinned worst intratrade excursion of
−11.41R on the SPY ledger (tests/test_trade_ledger.py:387)? — and the Gap C+B ruin curves replayed on
each variant ledger (`simulate_sizing` / `sizing_sweep`, common/position_sizing.py): does the stop
actually lower P(ruin) at fixed `f`?

**The prior, pre-stated so the measurement cannot be spun.** Both pinned ledgers are negative
expectancy (−0.39R MSFT CC, −0.54R SPY short vol; tests/test_trade_ledger.py:370, :382), so expectancy
deltas move within negative territory and no variant is expected to flip the sign. From the CC evidence
and Tharp's whipsaw warning (Loc 2043), the stop should truncate the MAE tail in the typical case —
gap-throughs keep that from being a guarantee — while worsening expectancy and win rate: each fire pays
a spread crossing and re-enters the same move. The genuinely open question is the ruin one: a truncated
tail can lower P(ruin) at fixed `f` even at worse expectancy, and that is exactly what the C+B replay on
the variant ledgers measures. The pins record the verdict either way.

**This is sample-spending exploration, and it says so — but it is not an FDR-governed search, and the
boundary is an argument, not a label.** The e-LOND stream's unit of account is a grammar cell: a
`StructureCandidate` recorded to the committed `idea_ledger.jsonl`, drawn from the countable
pre-specified lattice. An exit variant cannot be such a cell — `_validate_grammar` raises on any exit
knob — so nothing here can enter the stream even by accident. What the stream actually guards against,
an automated proposer multiplying looks faster than the discount sequence pays for them, is bounded here
by two commitments: the grid is six variants, one knob at a time, fixed in this doc before any run; and
all six results are pinned regardless of outcome, so every look stays auditable. If either commitment
breaks — the grid grows, or variants start being selected on their results — the exercise becomes a
campaign and belongs under the e-LOND stream like any other automated search.

**Pinning home.** The explorations pattern has two homes (docs/explorations.md:7-13 names both, and the
delta-hedged CC entry at :128 is the precedent for the second). The scout home
(search/explorations.py + tests/test_explorations.py, the cooldown exemplar: `load_naked_run` at
search/explorations.py:91, `cooldown_scout` at :167, module-scoped dataset-gated fixtures at
tests/test_explorations.py:114-129, decisive pins at :145-191) fits re-tag measurements of pinned runs.
Experiment 4 is an engine re-run, so it takes the other home, the stop-loss precedent's: a dataset-gated
variant class (proposed name `TestSpyExitVariantExploration`) in tests/test_vol_premium.py next to the
short-vol pins, with the `TestMsftStopLossRegression` docstring shape — verdict, mechanism, convention
caveat carried with the pin — plus a docs/explorations.md entry in the idea / how-tested / verdict /
trap shape its cooldown entry models (docs/explorations.md:30-86).

## Phasing

- **E1 — the engine change.** E1 lands the general manage branch (parallel to `early_close_single`),
  the synthetic off-equivalence test, and the synthetic trigger-mechanics tests. No dataset is required,
  and no pinned number moves.
- **E2 — the measurement.** E2 runs the Experiment 4 variants on the pre-committed grid, pins them as
  dataset-gated exploratory tests, replays the ruin curves, and writes the docs/explorations.md entry.
- **E3 — optional, needs sign-off.** A variant that earns escalation gets a grammar widening plus a
  registered cell — human-signed, like every widening; nothing escalates by default.

## Test plan

Synthetic mechanics ride the test_vol_premium `_scenario` pattern (tests/test_vol_premium.py:78) —
hand-crafted chain days, no dataset:

- Each trigger fires on a day built to fire it, and the asserted close fills are side-appropriate
  (shorts bought at ask, longs sold at bid under `bid_ask`; mids under mid fill) with per-leg
  commissions and the ex-commission trigger comparison.
- The `'reason'` label matches the trigger, and a day where two triggers are true records the priority
  winner (target over stop over time).
- A day where one leg has no quote does not trigger, even when the quoted legs alone would cross the
  threshold (the all-legs-quoted rule).
- A trigger condition true on the expiry day settles instead of closing (the elif-chain preemption,
  vol_premium.py:917 before :933).
- The dispatch rule holds at both edges: an `early_close_single` structure with only `close_at_pct` set
  takes the legacy single-leg path byte-identically, and one with a new knob set takes the general
  branch without the legacy block also firing.
- A synthetic net-debit entry (booked `entry_credit <= 0`) never arms a trigger.
- A `'close'` event carrying `'reason'` produces a `TradeRecord` identical to one without it (the
  extra-key invariance of `build_trade_ledger`, common/trade_ledger.py:181-209).
- **Off-equivalence:** with every new knob at its default, trades and daily equity are byte-identical to
  a pre-change golden run on the same synthetic data.

The real-data byte-identity guard is the existing pinned suites — the campaign, registered, equivalence,
and signature classes listed in the central-constraint section — run unchanged and reported in the E1
PR. E2 adds the dataset-gated Experiment 4 pins.

## Cross-surface obligations (when code lands, not now)

- Symbol-sweep regex: `TestMsftStopLossRegression` is already in the CLAUDE.md sweep regex;
  `close_at_pct` and `manage_deep_itm` are not, despite being cited in this doc (docs/*.md is a sweep
  surface). Add them together with the new symbols (e.g.
  `stop_loss_mult|exit_dte|close_at_pct|manage_deep_itm|TestSpyExitVariantExploration` — final names at
  build time) in the same change that lands the symbols, per the CLAUDE.md rule.
- README: a project-layout row for this doc, and any guarantees-line touch the new test classes need.
- Notebook: no regen expected — nothing here touches tutorial_covered_call_backtest.md or
  engine/make_figures.py. If either is somehow touched, the regen rule applies as usual.
- `STRUCTURE_ENGINE_VERSION`: unchanged, with the reasoning stated in the PR — a default-off knob moves
  no scored quantity at the frozen defaults, per the bump rule (search/edge_search.py:548-553).
- ci.yml: no edit expected, provided the new tests ride existing test files already on the pytest line —
  verify at land time.

## Honesty rails

- **Exploratory throughout.** Every Experiment 4 number is sample-spending, kill-or-justify, never a
  registered verdict; the prior is pre-stated above so the result cannot be narrated after the fact.
- **The look count is bounded by pre-commitment.** Six variants, one knob at a time, committed in this
  doc, with all six outcomes pinned. Growth into a sweep — or result-driven variant selection —
  reclassifies the work as a campaign under the e-LOND stream.
- **No FDR interaction in v1.** Nothing enters `idea_ledger.jsonl` and no e-value is spent; escalation
  goes only through E3's human-signed registration.
- **One significance authority.** The daily Newey-West HAC t (`short_vol_statistics`) is unchanged;
  every ledger and ruin statistic is reported, never a gate — the posture the ledger pins in its own
  docstring (common/trade_ledger.py:9-13).
- **Byte-identity is proven, not asserted.** The synthetic off-equivalence test plus the full pinned
  suites, run and reported in the E1 PR.
- **Convention caveats travel with pins:** the stop is a stop-market on daily closes (it flatters the
  stop), the all-legs-quoted trigger rule under-fires relative to a live book, and the roll carries a
  minimum one-day re-entry gap.
- **The disclaimer is stated once, in Status,** and covers every number this design will produce.

## Open questions

1. **Debit-structure trigger basis (for the widening, not v1).** v1 arms triggers only on a positive
   booked `entry_credit` (vol_premium.py:868), which excludes the calendar structurally and skips any
   net-debit risk-reversal entry per trade. The open half is the eventual debit-side basis. Leaning: the
   absolute entry quantity Gap A already floors R with (`_premium_collected_per_share`,
   common/trade_ledger.py:124-147).
2. **Deep-ITM management for multi-leg structures** (per-leg delta tests, generalizing
   `manage_deep_itm`'s single-leg test at vol_premium.py:945-946 and the CC's short-delta-only test at
   real_cc_backtest.py:471). Leaning: defer — assignment risk is a hedged-book concern the daily delta
   hedge already absorbs (the hedge retargets every day, vol_premium.py:958-978), and no Experiment 4
   variant needs it.
3. **The Experiment 4 grid's exact values.** Leaning: the grid above, pre-committed —
   `close_at_pct` {0.50, 0.75}, `stop_loss_mult` {2, 3}, `exit_dte` {7, 14}.

## Related

- [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the parent plan whose Gap E row ("the
  heaviest lift") and Experiment 4 this doc designs
- [docs/van_tharp_gap_a.md](van_tharp_gap_a.md) — the R-multiple ledger every variant is measured
  through (merged, #125)
- [docs/van_tharp_gap_d.md](van_tharp_gap_d.md) — the six-regime bucketing available to slice variant
  ledgers (merged, #126)
- [docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md) — the sizing replay whose ruin curves close the
  Experiment 4 loop (BUILT per its Status)
- [docs/explorations.md](explorations.md) — the exploratory pattern and the log-entry home for the E2
  verdict
- [docs/edge_search.md](edge_search.md) — the campaign, grammar, and lineage rules the boundary section
  respects
- blog/06_real_chains_flip_the_268000.md — the risk-managed collapse the CC prior echoes
- research/book-notes/trade-your-way-to-financial-freedom.md — the highlights file every Location above
  cites
