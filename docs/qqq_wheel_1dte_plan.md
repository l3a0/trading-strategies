# QQQ 1-DTE Wheel — −0.20Δ Put / +0.20Δ Call, Never Below Basis

**Status: DESIGN. No measurement has been run. Every definition below is
frozen before any outcome number is computed.**

**Epistemic class: exploratory measurement** — the Gap E / CC-R
(covered-call R-multiple) precedent: sample-spending, kill-or-justify,
never a registered verdict. Nothing enters the idea ledger (the committed
guess-counter for automated structure searches) and no e-value is spent
(the running evidence budget that counter's false-discovery control draws
down — the wheel is not a structure-campaign cell). Results pin via the
standard three surfaces (module, dataset-gated test, exploration-log
entry). Any cell clearing the escalation bar (§11) earns at most a
*registration proposal*, never a strategy claim.

Date: 2026-07-20. Owner-specified strategy; owner-directed gate (§4).

---

## 0. Reader's guide — what question this answers

The **wheel** is a two-state income strategy. Start in cash: sell a
**cash-secured put** — a promise to buy 100 shares at a chosen strike if
the price finishes below it, with the full purchase price held in reserve
— and collect a premium for making the promise. If the put expires
worthless, keep the premium and repeat. If it finishes in the money you
are **assigned**: you buy the shares at the strike. Now you hold stock,
so you switch sides and sell **covered calls** — a promise to hand the
shares over at a higher strike — collecting premiums until the shares are
called away, which puts you back in cash. Round and round: the wheel.

The owner's specification, tested exactly as given:

1. Sell the put nearest **−0.20 delta** (delta \~0.20 ≈ the market prices
   roughly a one-in-five chance of finishing in the money) and the call
   nearest **+0.20 delta**, both expiring the **next trading session**
   ("1-DTE" throughout).
2. **Never sell a call struck below the cost basis of the shares** — the
   rule that prevents locking in a realized stock loss via call-away.
3. **Only sell the put when QQQ is up on the day** — don't sell into a
   down day (the owner-directed entry gate, §4).
4. Idle cash earns **nothing** — the owner's brokerage (Schwab) sweeps
   cash at effectively zero, so the primary run credits 0% (§6).
5. The verdict is the head-to-head against **buy-and-hold QQQ** on the
   same $100K (§7).

Why 1-DTE and not the 0-DTE version originally asked about: a 0-DTE
option is sold in the morning and dies at that afternoon's close, and
every option dataset this repo owns is an end-of-day snapshot — the
morning entry simply does not exist in the data. The 1-DTE version is the
closest wheel the data supports honestly: entry at today's close (a real
quoted moment), settlement at tomorrow's close (another real quoted
moment), nothing in between assumed.

**Terms used throughout** (stated once, per the house plain-language
rule): **delta** (written Δ) is the option's price sensitivity to a $1
move in the stock; for out-of-the-money options it also approximates the
market's probability of finishing in the money. **DTE** is days to
expiration; "1-DTE" here means expiring the next trading *session* (one
to four calendar days — a Friday sale carries the whole weekend, §5). An
**R-multiple** is a trade's profit or loss divided by the risk taken at
entry — here the premium collected, so keeping the whole premium is +1R
and losses are open-ended. **Newey-West t** (NW t below) is a
t-statistic (signal divided by noise) robust to overlapping/correlated
returns; the **daily** NW t on the gap to buy-and-hold is this repo's
senior judge and the only significance authority. **Assignment** is
being made to honor the promise: buying at the put strike, or delivering
shares at the call strike. **Cost basis** is what the shares cost you —
here the strike you were assigned at (a premium-adjusted variant runs in
§8).

---

## 1. Prior work this design extends (and must not silently re-pin)

- The registered put-spread experiment (`docs/put_credit_spread_results.md`)
  — the put side's premium measured wrong-signed on SPY (−2.26) and IWM
  (−2.51) at the registered rung. Different structure and tenor, but the
  closest measured cousin of "sell puts for income" on these chains.
- The CC R-multiple experiment (`docs/spy_cc_r_experiment_plan.md`,
  `TestSpyCcRExperiment`) — the decomposition frame (option income vs. the
  direction bill), the junior/senior judge hierarchy, and the exit lesson:
  stops mostly *manufacture cycles* — they reshape the per-cycle ledger
  while the daily authority barely moves.
- The five conditioning nulls — trend gate, cooldown, IV-richness, the
  CC R-experiment's splits, and the wing diagnostic (the pinned
  enumeration in the exploration log) — every "only sell when ___" gate
  measured on this program has died, and gates keyed to recent strength
  have measured with the sign *backwards* (rips mean-revert). §4's gate
  is the sixth; it enters with that prior stated, at the owner's
  direction, with its ablation twin (the identical run with the gate
  removed) so its effect is a pinned number either way.
- The Tharp replication (`docs/tharp_random_entry_plan.md`) — position
  sizing multiplies expectancy; it cannot create it. The sizing layer here
  is deliberately small (§9).
- The sealed-vault note: QQQ is sealed for the *re-tag* edge-search phase
  only. Engine-run explorations on QQQ have precedent (the wing-premium
  diagnostic used QQQ as a verdict ticker); this experiment is the same
  epistemic object.

---

## 2. Strategy definition — the state machine (frozen)

Two states, one position at a time, evaluated once per trading day at the
close. A day is **eligible** when the next trading session has a listed
expiration (measured calendar in §5).

A **qualifying row** — the concept every selection below draws from, and
the same filter the §5 calendar was measured with — is a next-session
contract with `bid > 0` and vendor delta (the delta precomputed by the
data provider, used exactly as stored, never recomputed) within ±0.05 of
the 0.20 target: `[−0.25, −0.15]` for puts, `[0.15, 0.25]` for calls.
This implements the owner's frozen rule (directed 2026-07-20): **if
there are no bids at \~0.20 delta, don't sell** — the engine never
stretches to a far-away strike to stay busy. With no qualifying row, the
answer is always **sell nothing tonight**. Measured cost of the rule on
the primary arm: five skipped call days and three skipped put days of
its 858; the full-span cost is larger and era-skewed (§5).

**CASH state**, on an eligible day, if the entry gate (§4) passes: sell
`N` contracts of the qualifying put whose delta is nearest −0.20 (a
distance tie takes the lower strike). The sale is cash-secured: `N` is
clamped so `strike × 100 × N` never exceeds cash on hand *before*
tonight's premium, net of fees (§9).

**SHARES state**, on an eligible day (no gate — §4 applies to put entries
only): sell `N` contracts (one per 100 shares held) of the qualifying
call nearest +0.20Δ **among strikes at or above cost basis**. If no
qualifying strike honors the floor — the basis rule binding after a fall
— sell nothing that day and hold. The selection degrades gracefully:
when basis sits below the 0.20Δ strike the constraint is inert; when
basis sits above it, the rule picks the closest qualifying strike that
still honors the floor; when even those fall out of the band (deep
underwater), it sells nothing rather than collecting pennies on a
far-out strike.

**Settlement**, at the next session's close, in this order:

1. The expiring option settles against that close. Put: assigned exactly
   when `close < strike` — buy `100 × N` shares at the strike; **cost
   basis = the assignment strike** (the premium-adjusted variant is §8).
   Call: called away exactly when `close > strike` — deliver the shares
   at the strike, return to CASH. Exact ties expire worthless.
2. If still holding shares and a stop variant (§8) is active: `close ≤
   basis × (1 − stop)` → sell all shares at that close, return to CASH.
   (The EOD stop-market convention, which flatters the stop — no intraday
   fills — carried verbatim from the CC-R caveats.)
3. The entry logic above runs for tonight's position.

This ordering makes every option trade **atomic**: sold at one close,
settled at the next, never bought back — at 1-DTE on end-of-day data
there is no observable moment in between, so the entire intra-trade exit
menu (stops on the option, early closes, DTE management) is out of scope
*by construction*, not by choice. Exits live on the share side only (§8).
One residual simultaneity is accepted and disclosed: which side tonight's
sale takes (put or call) can be decided by the same close that prices it,
because an assignment resolved at that close flips the state. That is the
ordinary close-execution convention; §3 removes it only where it could
flatter a *signal*.

Early assignment is ignored (options settle only at expiration). QQQ
options are American-style, so this is an approximation; at a one-session
tenor the case that matters — a call driven deep in the money the day
before an ex-dividend date — is rare, and it is disclosed rather than
modeled.

---

## 3. Why the naive gate would flatter itself — and the fix (frozen)

The obvious "up day" definition — today's close above yesterday's,
entering at today's close — makes the signal and the fill simultaneous.
That is the standard close-execution convention, but it quietly grants
the backtest perfect knowledge of the closing print at the moment it
trades on it: on a knife-edge day (up 0.03% at 3:58, down 0.01% at 4:00)
the backtest always calls the sign correctly and a real trader cannot.
The flattery is small and always in the gate's favor — so it is designed
out rather than shrugged at.

**Frozen gate signal**: QQQ's **3:55pm ET print** — the last 1-minute
bar at or before 15:55, and no older than 15 minutes, from the local
`qqq_intraday_1min.csv` archive (bars are US/Eastern, bar-start labeled)
— versus **yesterday's official close** (unadjusted price file).
Strictly greater → the day counts as up. The signal is measured five
minutes *before* the fill, so it is strictly prior.

On a **shortened session** (the exchange's published 1:00pm-ET half days
— the day after Thanksgiving, and the Jul 3 / Dec 24 halves when they
occur; a static calendar in the module), the archive still carries
15:40–15:55 bars, but they are thin after-hours prints from three hours
*past* the close — reading them would postdate the fill. On those days
the signal bar is the last bar at or before **12:55pm ET** instead.

- **Fallback**: if no bar exists in the 15 minutes up to the signal
  cutoff, or the signal bar's price sits more than 5% from that day's
  official close (a scale or adjustment mismatch between the two
  sources), fall back to the close-over-close sign for that day and
  count it; the fallback count is reported.
- **Variant**: the naive close-over-close definition runs as its own
  gate arm.
- **Diagnostic**: the disagreement rate — the share of days where the
  3:55 sign differs from the closing sign. That number *is* the size of
  the flattery the naive convention would have enjoyed. If the gate's
  verdict differs between the two definitions, the gate was never a real
  signal: a rule that only works with five extra minutes of future
  knowledge has told you where its edge lives.

The intraday file is a **signal input only** — it never prices a fill.
It is a local archive (not a published canonical store); the dataset-gated
test skips unless it is present alongside the chain stores, and publishing
it to the data release stays a separate, human-gated decision. The build
PR records the archive's sha256 and runs a frozen sanity battery before
first use: signal-bar coverage across the trading calendar, and the 5%
scale cross-check against the official close file.

---

## 4. The entry gate (frozen; owner-directed)

**Rule**: in CASH state, sell the put only if the day is up per §3.
Down or flat → no entry tonight; cash sits idle at the arm's cash yield.
The gate applies to **put entries only** — once assigned, waiting for up
days would only lengthen the premium-less stretches the basis rule
already creates.

**Prior, stated before the run**: this is the program's sixth entry-
conditioning gate. The claim it encodes — yesterday's direction predicts
tonight's — is a daily-horizon timing signal; QQQ's day-over-day
autocorrelation is near zero and historically leans slightly negative,
the same backwards sign the cooldown scout measured on multi-week rips.
QQQ closed up on \~56–57% of days over the store span (57.6% on the
primary arm, measured from the unadjusted price file), so the expected
first-order effect is simply \~42–43% fewer put sales and about that
much less premium. The committed expectation (§10, prior 3) is that the
gate does not improve the daily verdict. The ablation twin (gate off)
runs in every grid cell so the gate's effect is a pinned number either
way.

---

## 5. Data, spans, and the expiration calendar (frozen)

**Chains**: canonical `qqq_option_dailies.csv` (calls, 2016-06-06 →
2026-06-05) merged with `qqq_option_dailies_puts.csv` at load — the house
calls-only-canonical pattern — window-clipped to call days. QQQ has no
`CHAIN_CLEAN_START` entry — its canonical calls store starts 2016-06,
past the placeholder-greeks era; the build PR adds QQQ to
`TestNewChainsClean` so the `validate_dailies` CLEAN verdict is pinned
rather than asserted. Vendor deltas and real bid/ask are used as stored;
the canonical files are never appended to.

**Prices**: as-traded QQQ closes via `load_unadjusted_prices`; the 1-min
intraday archive feeds the gate signal only (§3).

**The expiration calendar is era-dependent**, and it was measured, not
assumed (scan of the stores, 2026-07-20). Days per year with a
next-session expiration *and* a qualifying row (§2's band, `bid > 0`) on
both sides:

| era | next-session expiries per year | what a "1-DTE wheel" means there |
| --- | --- | --- |
| 2016-06 → 2020 | \~31–56 | weeklies only — Thursday sales expiring Friday |
| 2021 | 125 | Mon/Wed/Fri listings |
| 2022 | 170 | Tue/Thu added late in the year |
| 2023 → 2026-06 | every trading day (250, 252, 250, 106) | the true daily wheel |

(In-window, the no-stretch rule (§2) skips 60 call days and 12 put days,
three-quarters of them in the 2016–2017 weekly era, whose coarser strike
listings often leave the exact 0.15–0.25 window empty. The primary arm
loses only five call days and three put days of its 858. One notable
secondary-arm skip: 2020-03-12, the COVID crash — deltas gapped past the
band, so the wheel would have sat out exactly the night a stretched
seller got hurt worst. The build PR pins the calendar counts in the
dataset-gated test, so the table above stops being a session
measurement.)

**Two arms:**

- **Primary — the daily era**: 2023-01-01 → 2026-06-05, \~858 eligible
  days. This is the window where the strategy as specified actually
  exists five days a week. It contains no bear market — a stated
  limitation the secondary arm exists to offset.
- **Secondary — the full span**: 2016-06-06 → 2026-06-05, primary-cell
  configuration and its gate-off twin only (2 runs, no grid). In the
  early era this is a *Thursday-overnight* wheel — same legs, \~one-fifth
  the cadence — and it is reported as such, never pooled with the primary
  arm. Its value is that it contains the 2018, 2020, and 2022 selloffs:
  a wheel judged only on 2023–2026 never gets assigned in anger.

**The weekend is part of the strategy**: a Friday sale (and every
Thursday sale in the early era) carries up to three calendar nights of
gap risk for one session's premium. The Friday-entry share of trades and
their outcomes are reported separately as a diagnostic.

---

## 6. Accounting (frozen)

- **Fills at the bid.** Premiums at this tenor are small (roughly $0.50–
  $1.50 per share) and the bid-ask spread is a real fraction of them;
  selling at the bid is the conservative floor. A mid-price twin of the
  primary cell is reported as a diagnostic so the spread's cost is a
  visible dollar figure, not an assumption.
- **Fees**: $0.65 per contract sold (the owner's brokerage rate); $0 for
  assignment, exercise, and stock trades.
- **Idle cash earns 0% in the primary arm.** The owner's brokerage sweep
  pays effectively nothing — the interest on the collateral is real, but
  the broker keeps it. The **risk-free variant** (§8) credits the house
  frozen rate (4.5% simple, matching the engines' `risk_free_rate`
  convention) as if collateral were manually parked in a money-market
  fund. The gap between the two arms is a single dollar figure: what the
  sweep account costs per year. The put-spread experiment's pinned
  decomposition put interest at +$51.8K of the book's +$55.7K apparent
  profit — \~93%; this pair makes the same decomposition explicit for
  the wheel.
- **Dividends are omitted from both books.** The buy-and-hold comparator
  runs on unadjusted prices, so it forfeits QQQ's \~0.6%/yr dividend
  yield continuously, while the wheel forfeits it only during
  share-holding stretches. The omission therefore *flatters the wheel*
  in the head-to-head, by roughly the dividend yield times the share of
  time buy-and-hold-only would have collected it — disclosed, not
  repaired, and bounded in the report.
- **Capital**: $100,000. Buy-and-hold: the same $100,000 fully into QQQ
  at the arm's first close, fractional shares, no interest.
- **Window ends**: the last sale occurs on the last day whose settlement
  lands inside the arm; both books mark to the arm's final close, and
  shares still held at the end are marked, not force-sold.

---

## 7. Verdict and judges (frozen)

**Senior judge — the only significance authority**: the daily Newey-West
t (`newey_west_summary`) on the daily gap between the wheel book and the
buy-and-hold book, per cell. Comparing against buy-and-hold prices both
sides of the cash question at once: any interest the wheel earns and any
QQQ return its idle cash forgoes both land in the same daily series.

**Junior judges — descriptive, never gating** (the frozen hierarchy from
the CC-R experiment):

- **Per-overnight-trade R ledger** (`build_trade_ledger`,
  `risk_basis='premium_collected'`): every put and call sale scored on
  its own; +1R = kept the whole premium. Committed expectation on shape:
  many +1R wins and rare, large negative multiples — a 3% overnight gap
  through a 0.7%-out-of-the-money strike against a \~$1 premium is a
  −10R to −20R print.
- **Per-rotation ledger**: one cycle = CASH → assignment → holding →
  back to CASH, attributed from the daily gap series by date interval
  (the `attribute_cycles` frame), so the basis rule's premium-less
  holding stretches live *inside* the cycle that caused them.
- **Cash-plus-interest gap** and the descriptive dollar decomposition
  (premium collected, assignment losses, holding-period share P&L, fees,
  interest), reported per cell. The buckets partition: an assignment
  loss is `(strike − that day's close) × shares` on the assignment day,
  and holding-period share P&L starts the following day — nothing is
  double-counted.

**Decomposition companion** (primary cell only): the primary cell's
realized leg sequence replayed exactly — same entries, strikes, and
fills, no re-selection — with a static overnight hedge bolted on: at
each entry, offset the position's net delta in shares at the same close,
unwound at settlement.
The unhedged-minus-hedged difference is the **direction bill** in
dollars, the same instrument that closed the covered-call question: it
separates "the premium was real but underinvestment cost more" from
"there was no premium at all."

---

## 8. The grid (frozen)

Five dials, all crossed, on the primary arm:

| dial | values | what it tests |
| --- | --- | --- |
| entry gate (§4) | on, off | the owner's up-day rule vs. its ablation |
| basis rule | on, off | never-below-basis vs. the unconstrained 0.20Δ call |
| stock stop (§2) | none, −5%, −10% | dumping shares below basis vs. holding to call-away |
| contracts | 1, 2 | half-committed vs. fully-committed collateral |
| cash yield | 0%, 4.5% | the owner's sweep reality vs. the money-market twin |

2 × 2 × 3 × 2 × 2 = **48 cells**, every one reported (§11).

**The primary cell** — the coordinate every single-run variant, the
secondary arm, the decomposition companion (§7), the sizing battery
(§9), and the headline pin (§13) anchor to — is: **gate on, basis rule
on, no stop, 1 contract, cash 0%**. That is the owner's specification
verbatim, at half-committed collateral.

Beyond the grid: the two secondary-arm runs (§5), the basis-definition
variant (cost basis = assignment strike minus *all* premiums collected
this rotation so far, the assigning put's included, recomputed as each
premium arrives — a floor that ratchets down; primary cell only), the
close-over-close gate variant (§3), and the mid-fill diagnostic (§6) —
each a single additional run, none crossed into the grid.

Notes fixed in advance: with the basis rule **off**, the call is simply
the nearest-to-+0.20Δ qualifying row — the wheel that will realize
stock losses via call-away; that is the point of the ablation. At 2
contracts, `strike × 200` exceeds $100K once QQQ trades above \~$500, so
the affordability clamp (§2) binds late in the window — the share of
clamped days is reported. Double assignment at 2 contracts puts the book
\~100% in stock, at which point the wheel *is* buy-and-hold until called
away.

---

## 9. Position sizing (frozen; post-hoc replay, no engine re-runs)

Cash-secured selling is inherently chunky: one QQQ contract wants
\~$50K of collateral, so on $100K the dial has notches 0, 1, 2 — there
is no "risk 1% per trade" granularity to tune, and that constraint is
itself a finding worth stating. The battery runs on the primary cell's
**per-rotation** dollar stream — deliberately not the per-overnight-trade
stream, which records only option sales and would omit the wheel's
dominant risk: the share P&L a rotation carries while holding through a
selloff. A rotation row contains everything from the assigning put to the
final call-away, so the resampled careers can actually reproduce the
book's real drawdowns.

- **Marble-bag resampling** (`common/position_sizing.py`, seed
  `WHEEL_SEED = 20260720`, `n_trades` = the book's own rotation count):
  draw rotations at random with replacement to replay thousands of
  alternate careers, and report the probability of ruin, the probability
  of a 25% drawdown, and median terminal equity, at each contract notch.
- **Kelly fraction** of the same stream (the bet size that maximizes
  long-run growth of a given trade stream), reported with the standing
  prior: it is zero or negative for a non-positive-expectancy stream,
  and sizing multiplies expectancy — it cannot create it.

---

## 10. Committed expectations (priors stated before the run)

| # | Claim | Basis |
| --- | --- | --- |
| 1 | The primary cell trails buy-and-hold (daily NW t < 0) on 2023–2026. | A cash-secured −0.20Δ put is \~one-fifth as invested as buy-and-hold in a bull tape; the direction bill dominates. |
| 2 | Put-sale win rate ≥ 75% while rotation expectancy is \~0 or negative. | −0.20Δ ≈ 80% expire-worthless by construction; the CC-R shape. |
| 3 | The up-day gate does not improve the daily verdict; its main effect is \~42–43% fewer sales. | Five conditioning nulls; near-zero/negative daily autocorrelation; the cooldown scout's backwards sign. |
| 4 | The basis rule's frozen stretches concentrate after assignments in drawdowns; the rule-off twin collects more premium but realizes stock losses; neither flips the verdict sign. | The CC-R exit lesson — exits manufacture cycles. |
| 5 | Stock stops improve per-rotation R while the daily authority barely moves. | `TestSpyCcRExperiment`'s stop cells. |
| 6 | The 4.5% arm beats the 0% arm by roughly the interest on average idle cash, and that interest exceeds the entire premium harvest. | The put-spread finding: interest was \~93% of apparent profit. |
| 7 | The decomposition companion shows overnight put premium alone \~0 or negative. | The registered put-side NULL (SPY −2.26 / IWM −2.51) — a cousin, not the same cell; stated as analogy. |
| 8 | The 3:55-vs-close gate disagreement rate is low single-digit percent, and the gate verdict does not flip between definitions. | Knife-edge days are rare; if it flips, the gate's edge was the last five minutes (§3). |

Contradictions of these priors are findings, not failures — they get
pinned with the same weight as confirmations.

---

## 11. Multiplicity honesty and the escalation bar

This is a 48-cell search plus named variants; the batch is reported **in
full** — every cell, wrong-signed and boring cells included, and any
"best cell" is read knowing it won a 48-way selection. The pre-committed
escalation bar, mirroring the CC-R and call-spread precedents: a cell
interests us only if its daily Newey-West t against buy-and-hold exceeds
**+2** on the primary arm. At or below the bar it records as closed.
Above the bar it escalates to a human-signed registration proposal —
nothing in this experiment promotes directly.

---

## 12. What this experiment is NOT

- Not a registered experiment: no registered pin moves, no e-value is
  spent, and a surviving cell earns a proposal, not a claim.
- Not the 0-DTE strategy: that requires intraday option quotes the repo
  does not own (and \~2-year vendor windows that miss every bear market);
  this design records why the 1-DTE form is the honest substitute.
- Not a re-opening of the conditioning family on the program's
  initiative: the up-day gate is owner-directed, enters with the family's
  five-null prior stated, and carries its ablation in every cell.
- Not a Tharp sizing study: the collateral chunkiness (§9) reduces
  sizing to three notches; the marble-bag supplies the risk numbers.

---

## 13. Build plan

- **Module**: `realchains/wheel_1dte.py` — the state machine (§2), the
  gate reader (§3), the grid runner (§8), the sizing battery calls (§9),
  and a print-only report. Deterministic; the only seed is the
  marble-bag's.
- **Tests**: `tests/test_wheel_1dte.py` — an always-run synthetic layer
  (state transitions; the basis rule binding and inert cases; gate
  application including the fallback; settlement order and tie handling;
  the affordability clamp; conservation — summed daily P&L equals final
  minus initial equity to the cent) plus a dataset-gated
  `TestQqqWheel1dteExploration` pinning the decisive numbers: the
  primary cell's daily NW t, final equities (wheel vs. buy-and-hold),
  rotation count and expectancy, the gate-on/off pair, the basis-rule
  pair, the disagreement rate, the eligible-day calendar counts (§5),
  and the sizing-battery outputs. The
  dataset gate requires both chain stores *and* the intraday archive;
  absent any of them the class skips (it cannot run in CI until the
  intraday file is published — a separate, human-gated decision).
- **Results surface**: a `docs/explorations.md` entry (or the §11
  escalation path).
- **Plumbing in the build PR**: `ci.yml` pytest bucket, CLAUDE.md
  symbol-regex additions, README file-table rows.
- **Runtime estimate**: a daily loop over \~858 days × 48 cells plus two
  full-span runs — minutes, not hours; the heavy lifting is one pass of
  chain-store loading.

---

## 14. Order of operations

1. This design doc merges; the definitions above are frozen.
2. The build PR lands the module and synthetic tests — no measurement
   numbers.
3. One run executes both arms; decisive numbers pin in the dataset-gated
   test and the exploration-log entry in the same PR.
4. §11 governs any escalation; otherwise the QQQ 1-DTE wheel question
   closes with the pins.
