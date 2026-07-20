# Tharp's Support/Resistance Catalog, Replicated — Extremes Claims and Entries-vs-Random

**Status: DESIGN. No measurement has been run. Every definition below is
frozen before any outcome number is computed.**

**Epistemic class: exploratory replication** — the Tharp random-entry
precedent (`docs/tharp_random_entry_plan.md`): sample-spending,
kill-or-justify, never a registered verdict. Nothing enters the idea
ledger (the committed guess-counter for automated structure searches) and
no e-value is spent (the running evidence budget that counter's
false-discovery control draws down — these are price-only counting
experiments, not structure-campaign cells). Results pin via the standard
three surfaces: module, result-pin tests (which RUN IN CI — the inputs
are committed CSVs, so no dataset gate applies), and an exploration-log
entry. A claim that *survives* earns at most a gate-design proposal on
the option books, under the conditioning family's rules; never a
strategy claim.

Date: 2026-07-20. Owner-directed follow-up to the support/resistance
feasibility discussion; source claims quoted from the owner's book notes
(`research/book-notes/trade-your-way-to-financial-freedom.md`, Kindle
locations cited per claim).

---

## 0. Reader's guide — what question this answers

*Trade Your Way to Financial Freedom* catalogs the classic
support-and-resistance toolkit — breakouts through old highs,
oscillator "oversold" bounces, moving-average crosses — and attaches a
handful of **checkable claims**: a quoted reliability number for one
setup, and a cited study concluding most entry signals beat nothing.
This experiment replicates the checkable subset on our own committed
price data, so the catalog stops being lore we re-litigate and becomes
numbers we cite.

Two phases, both cheap, both price-only:

1. **Phase 1 — the extremes claims.** The book's short-term-setups
   passage (Location 4374, in the chapter discussing Connors and
   Raschke's setups; the note itself carries no explicit attribution)
   says a market that closes in the top part of its daily range has
   "70 to 80 percent reliability for a **more extreme opening in the
   same direction** the next day" — with two companions: the odds it
   *closes* in that direction are "much less," and a full trending day
   raises the odds of a next-day reversal. Three countable overnight
   claims. The wheel connection, stated precisely: C1 concerns the
   next *open*; the wheel settles against the next *close*, so C2 —
   the close version — is the wheel-tenor claim, which is why the two
   are always reported side by side.
2. **Phase 2 — the entries-vs-random meta-test.** The book (Location
   4850) cites LeBeau & Lucas: run classic entry signals with nothing
   but a fixed time exit, and "most of the indicators failed to perform
   any better than random — including all the oscillators and various
   moving-average crossover combinations." We rerun that design on our
   ETFs across a frozen six-signal menu, with the Turtle Soup decay
   claim (Location 4355 — "today, most of these 20-day breakout signals
   are false breakouts") as a counted by-product.

**The honesty rail that decides Phase 1**: the comparison is never
against a coin. QQQ opens higher on well over half of all mornings
unconditionally, so the conditional rates must beat the
**unconditional base rate**, not 50% — otherwise the setup is the base
rate wearing a costume. Phase 2's version of the same rail is the
**drift wall**: on long-drifting ETFs, any long entry followed by a
time exit wins often; the question is always *more often than entering
on a random day?*

**Terms used throughout** (stated once, per the house plain-language
rule): **OHLC** is a day's open, high, low, and close. The **range
position** of a day is where its close sits inside the day's high–low
span, from 0 (closed at the low) to 1 (closed at the high). A
**channel breakout** buys when price exceeds the highest high of the
last N days — breaking through resistance. **SMA** is the simple
moving average (the plain mean of the last N closes). **RSI**
(relative strength index) is a 0–100 oscillator that reads low after
persistent down-closes; "oversold" is the classic reading below 30. A
**golden cross** is the 50-day SMA crossing above the 200-day SMA. A
**time exit** closes a trade a fixed number of sessions after entry, no
stops, no targets — the LeBeau–Lucas frame, which isolates what the
*entry* alone contributes. **Flat** means holding no open trade in the
cell's own book. The **base rate** is the unconditional probability of
the same outcome measured over the same eligible days; every
conditional claim is judged against it. All price comparisons in this
document are **strict** inequalities; an exact tie counts against the
claim (a tie is not the predicted outcome), stated here once and
applied everywhere.

---

## 1. Prior work this design extends (and must not silently re-pin)

- The Tharp random-entry replication
  (`docs/tharp_random_entry_plan.md`, `TestTharpReplicationEnsemble`) —
  the replication template (his coordinates frozen, translation caveats
  named, pre-committed nulls) and the standing result these tests must
  respect: on our ETFs the drift twin beat his full system in 98% of
  careers.
- The six conditioning nulls (trend gate, cooldown, IV-richness, the CC
  R-experiment's splits, the wing diagnostic, the wheel's up-day gate)
  — every "only trade when ___" signal measured on this program has
  died. These phases test *claims*, not gates; a surviving claim earns
  a gate design as a NEW doc, it does not create one here.
- The registered trend gate (`docs/trend_gate_results.md`) — the
  moving-average-as-support cousin, killed at Stage 1 wrong-signed. The
  MA entries in Phase 2 re-enter that neighborhood deliberately, in the
  cheaper counting frame.
- The house placebo pattern (seeded matched-count resampled nulls — the
  wing diagnostic's circular shifts, the wheel's marble-bag): both
  phases judge against that family, not against textbook formulas whose
  independence assumptions the data violates.

---

## 2. Data and spans (frozen)

**Inputs**: the committed split-adjusted daily OHLC files fetched for
the Tharp replication (`pipeline/download_ohlc.py` outputs,
`data/{ticker}_daily_ohlc.csv`). Measured spans (2026-07-20):
QQQ/SPY/MSFT/NVDA/XLE 1999-11-01 → 2026-06-30 (\~6,700 sessions each),
IWM 2000-05, TLT 2002-07, EEM 2003-04, GLD 2004-11 → same end.

- **Primary tickers: QQQ and SPY** (the program's home instruments,
  longest spans). **Robustness: MSFT, NVDA, XLE, IWM, EEM, GLD** —
  reported, never pooled with the primary read. **TLT is excluded, as
  a choice**: the random-entry replication's basket did include TLT,
  but the option-side explorations since (the wing diagnostic, the
  NVDA ladder) have kept it out as the structure campaign's holdout,
  and this design extends that conservative practice to the price-only
  frame rather than re-litigate the boundary.
- **Era panels**: every count reports the full span plus fixed panels
  (1999–2009, 2010–2019, 2020–2026). A day belongs to the panel of its
  **conditioning date** (the day the setup or signal fires), even when
  its outcome window crosses a boundary. Late-start tickers report
  truncated first panels, labeled with their actual start. The panels
  differ in length (roughly 10, 10, and 6.5 years) — acknowledged;
  they are calendar eras, not equal samples, and per-panel counts are
  always pinned beside per-panel rates.
- Split-adjusted OHLC is safe here by construction: range position is a
  same-day ratio, and every cross-day comparison compares prices on
  the same adjustment basis. Days with `high == low` (no range) are
  excluded from range-position conditioning and counted in a
  diagnostic.
- No option data, no intraday data, no fills, no fees: these are
  *claim* replications (did the market do what the book says), not
  strategy backtests. Dollar realism enters only if a claim survives
  and graduates to a gate design.

---

## 3. Phase 1 — the extremes claims (frozen)

Source: Location 4374, three claims. All counts use **range position**
`rp = (close − low) / (high − low)`; "top quartile" is `rp ≥ 0.75`,
"bottom quartile" `rp ≤ 0.25`. The **eligible universe** for every
Phase-1 count (conditional and base rate alike) is: non-degenerate
range (`high > low`), a next session exists in the file.

| # | Book claim (quoted reading) | Frozen measurement |
| --- | --- | --- |
| C1 | Top-part close → "70 to 80 percent reliability for a **more extreme opening** in the same direction." | **Quote-bearing headline**: top-quartile day → `P(next open > high)` (the opening is more extreme than the day's whole range). **Loose companion**: `P(next open > close)`. Mirrors: bottom-quartile → `P(next open < low)` / `P(next open < close)`. Each variant is judged against its own-universe base rate. |
| C2 | The odds it *closes* in that direction are "much less." | Same conditioning: `P(next close > close)` (mirror `<`), vs. its own base rate, reported beside C1 so the open-vs-close gap is one visible number. This close-to-close version is the wheel-tenor claim. |
| C3 | A trending day → "an even greater probability of a reversal." | Trending-up day = same-day `open ≤ low + 0.25 × (high − low)` AND `close ≥ low + 0.75 × (high − low)` (mirror for down; a hindsight classification — this is claim counting, not a tradable signal). Frozen headline: `P(next close < next open)` vs. the same probability unconditionally. A labeled variant conditions additionally on `next open > close` (the gap-continuation reading); a second labeled variant counts `P(next close < close)`. Variants are reported, non-gating. |

**Ambiguity rules, frozen for both C1 and C3**: C1's headline is the
strict quote-bearing reading; if the strict and loose variants disagree
about survival, C1 records as AMBIGUOUS, not survived. C3's passage
never names what reverses; if its headline and either labeled variant
disagree about survival, C3 records as AMBIGUOUS. C2 has one reading.

**Judging (frozen — the placebo pattern, not the textbook binomial)**:
conditioning days cluster in volatility regimes and overnight odds vary
by regime, so an analytic binomial understates the noise. Each claim's
p comes from a **matched-count date-resampled null**: draw `n_cond`
dates from the same eligible universe, matching the conditional set's
per-era-panel counts, uniformly without replacement within each panel;
compute the resampled rate; repeat B = 10,000 times under seed
`SR_SEED = 20260720` (one derived stream per claim × ticker, from the
claim's label); the two-sided add-one Monte-Carlo p is
`2 × min(P_null(rate ≥ observed), P_null(rate ≤ observed))` with the
add-one convention `(1 + count) / (1 + B)`, capped at 1. The exact
binomial against the base rate is computed and reported as a
diagnostic only.

**The survival bar (frozen)**: a claim survives only if its headline
measurement has p < 0.01 on **each** primary ticker (QQQ and SPY,
full span), AND the effect direction (conditional minus base rate)
agrees within each primary ticker on at least two of that ticker's
three era panels. Variants and mirrors never gate; robustness tickers
never gate. QQQ and SPY overlap heavily (correlated indexes), so the
both-tickers clause is closer to one test than two — acknowledged; it
screens instrument-idiosyncrasy, not independence.

**The deflation prediction, committed**: the quoted 70–80% likely
reflects the loose reading's base rate plus gap continuation, and the
strict quote-bearing rate likely sits far below 70–80%. The pinned
deliverable is the *pair of gaps* — strict and loose, each vs. its
base rate — the numbers the book never states.

---

## 4. Phase 2 — entries vs. random, the LeBeau–Lucas frame (frozen)

The design their study froze and the book endorses (Location 4850):
enter on the signal, exit at the close **H sessions later**, no stops,
no targets. Entry at the **signal day's close** (the signal is computed
from that close; the standard close-execution convention, same as the
wheel's). Win = exit close strictly above entry close (mirrored for
shorts). **Each (signal, ticker, H, side) cell keeps its own
independent book**: entries only while flat in that book, a signal
firing during an open window is ignored, re-entry is permitted at the
exit close, and an entry whose exit would fall past the span's end is
skipped. Windows within a book never overlap; they are still draws
from one autocorrelated price path, which is exactly why the judging
below uses matched-count nulls rather than treating trades as
independent.

**The frozen six-signal menu** (long-side definitions; nothing off
this menu runs):

| signal | definition (all inputs are daily closes unless stated) | book anchor |
| --- | --- | --- |
| CB-20 | close > highest high of the prior 20 sessions (the entry day's own high excluded) | the Turtles' original (Location 4888) |
| CB-40 | close > highest high of the prior 40 sessions | the Turtles' later band — "they simply moved up to 40-day breakouts" (Location 4888) |
| CB-100 | close > highest high of the prior 100 sessions | "breakouts between 40 and 100+ days still work fairly well" (Location 4888) |
| MA-200 | close crosses above the 200-day SMA: `close_{t−1} ≤ SMA200_{t−1}` and `close_t > SMA200_t` | the trend-gate cousin (a menu coordinate of ours, not a book-quoted parameter) |
| GX | golden cross: `SMA50_{t−1} ≤ SMA200_{t−1}` and `SMA50_t > SMA200_t` | the crossover family the cited study dismissed (Location 4850; the 50/200 pair is our frozen choice) |
| RSI-30 | Wilder's RSI(14) crosses back up through 30: `RSI_{t−1} < 30 ≤ RSI_t` | the oscillator family, dismissed wholesale (Location 4850; the 14/30 coordinates are the classic defaults, our frozen choice) |

**Wilder's RSI, fully frozen**: gains `g_t = max(close_t − close_{t−1},
0)`, losses `l_t = max(close_{t−1} − close_t, 0)`; the smoothed
averages seed as the plain mean of the first 14 gains/losses and then
follow Wilder's recurrence `avg_t = (13 × avg_{t−1} + x_t) / 14`;
`RSI = 100 − 100 / (1 + avg_gain / avg_loss)` (all-loss windows read
0, all-gain 100). The first 14 sessions are warm-up and cannot fire.
Every signal's warm-up (20/40/100 sessions of highs; 200 SMA sessions;
14 RSI sessions) excludes those days from both firing and the null's
drawable universe.

**Short mirrors, frozen** (a labeled secondary — the book itself says
short entries need different speeds): CB mirrors fire on close <
lowest low of the prior N sessions; MA-200 and GX mirror on the
downward cross; RSI mirrors on `RSI_{t−1} > 70 ≥ RSI_t`. Wins mirror
(exit close strictly below entry close).

Horizons **H ∈ {5, 10, 15, 20}** sessions (theirs, verbatim).

**Baselines, two, both frozen**:

1. **The base rate (the drift wall)**: unconditional
   `P(close_{t+H} > close_t)` over the cell's drawable universe, per
   ticker, per era panel — what "entering for no reason" wins.
2. **The matched-count random null**: for each cell, the drawable
   universe is every session past the signal's warm-up with `t+H`
   inside the span. Algorithm, frozen: shuffle the drawable days under
   the cell's derived seed (`SR_SEED` + a hash of the cell key),
   accept days greedily in shuffled order, rejecting any within H
   sessions of an accepted day, until the cell's own trade count is
   reached; that is one resample. B = 10,000 resamples per cell. The
   cell's win rate gets the one-sided add-one Monte-Carlo p
   `(1 + #{null rate ≥ observed}) / (1 + B)`; **a cell "beats random"
   exactly when p ≤ 0.01**. Cells with fewer than 15 trades report
   NO-VERDICT (shown, never judged — a percentile bar on a handful of
   discrete outcomes is noise wearing a decimal). The cell's **mean
   per-trade return** is placed on the same null's mean-return
   distribution as a reported diagnostic, never a gate — this replaces
   any separate bootstrap-against-zero, which on drifting ETFs would
   be the coin comparison §0 outlaws.

**The Phase-2 survival rule (frozen)**: a signal survives only if **at
least 3 of its 8 primary cells** (2 primary tickers × 4 horizons, long
side) beat random AND the beating cells include at least one on each
primary ticker. Anything less records as closed. Expected false
positives, stated in advance: at p ≤ 0.01 across \~500 reported cells,
\~5 spuriously beating cells are expected under the global null
(clustered, since cells within a signal share entries across
horizons); a lone beating cell is scatter, which is what the 3-of-8
bar encodes.

**The Turtle Soup by-product (frozen)**: entry set = every day CB-20
fires (the raw signal-day set, no flat-only filter — this is a claim
count, not a trade set). The breakout level is **fixed at entry**: the
highest high of the 20 sessions strictly before the entry day. A
breakout **fails** if any close in sessions `t+1 … t+5` is strictly
below that level. The 5-session window is our frozen choice — the book
(Location 4355) gives none, saying only "they don't work and the
market falls back." Verdicts, frozen: the book's "most … are false
breakouts" reads as a modern-panel (2020–2026) failure rate above 50%;
the *decay* claim (they used to work) reads as the earliest panel's
failure rate being lower than the latest panel's by a two-proportion
exact test at p < 0.01. Wilson 95% intervals and per-panel entry
counts are pinned beside every rate.

---

## 5. Committed expectations (priors stated before the run)

| # | Claim | Basis |
| --- | --- | --- |
| 1 | C1's strict quote-bearing rate (next open beyond the day's high) lands far below 70–80%; the loose rate lands near its own base rate plus gap continuation. Survival at the frozen bar is not expected for either reading. | The up-day gate's near-zero daily autocorrelation; the base-rate rail. |
| 2 | C2 confirms trivially (close odds well below open odds) — a caution the book got right. | Overnight drift converts to intraday noise; the wheel's own overnight economics. |
| 3 | Most Phase 2 cells fail to beat the matched-count random null; the oscillator (RSI) and MA-cross families specifically produce zero beating cells. | LeBeau–Lucas as quoted (Location 4850); the trend gate's wrong-signed kill; six conditioning nulls. |
| 4 | The channel breakouts (CB-40/CB-100) do NOT survive the 3-of-8 bar on single long-drifting ETFs, contra the book's futures-era endorsement — the drift wall absorbs them. | The Tharp replication: the drift twin beat the full trend-following system in 98% of careers. |
| 5 | The Turtle Soup failure rate is above 50% in every era panel, and the earliest-vs-latest decay test does not reach p < 0.01 — "breakouts stopped working" is mostly "breakouts on a mean-reverting index mostly always failed." | Rips mean-revert (the cooldown scout's sign); index vs. futures universe mismatch. |
| 6 | No robustness ticker flips any primary conclusion; scattered beating cells appear (≈5 expected across the batch) without any signal reaching the survival bar. | The wing contrast sweep's sign instability; the stated false-positive arithmetic. |

Contradictions of these priors are findings, not failures — they get
pinned with the same weight as confirmations.

---

## 6. Multiplicity honesty and the escalation path

Phase 1 is 3 claims × variants × mirrors; Phase 2 is 6 signals ×
4 horizons × 2 sides × 8 tickers plus two baselines — roughly 500
reported cells. Every cell is reported; nothing is dropped; any "best
cell" is read against the batch it won and the \~5 expected false
positives stated in §4. The survival bars (Phase 1 §3, Phase 2 §4) are
pre-committed and deliberately strict because the batch is large and
the priors are single-digit.

**Escalation**: a surviving claim does not become a strategy or a gate.
It earns a *proposal* for a gate design on the wheel or covered-call
books — a new frozen doc, owner-signed, under the conditioning family's
rules, with the survivor's exact frozen definition carried over
unchanged. The wheel's up-day gate showed the additional hurdle any
survivor faces there: the state machine buffers entry gates, so even a
real overnight edge must survive dilution to a handful of re-timed
sales.

---

## 7. What this experiment is NOT

- Not a registered experiment: no registered pin moves, no e-value is
  spent, and survivors earn proposals, not claims.
- Not a strategy backtest: no fills, fees, sizing, or equity curves —
  claim counting only. Dollar realism enters at the gate-design stage,
  if ever.
- Not a band-trading search: the band family (envelopes, Bollinger,
  ATR bands) has no pinned book claim to check and the widest free-dial
  space in the catalog; it stays unbuilt unless a Phase 1/2 survivor
  motivates it as its own frozen doc.
- Not a re-opening of the conditioning family: no option book is gated
  by anything here; the six-null prior stands.
- Not a futures replication: LeBeau–Lucas and the Turtles traded
  futures universes. The translation to long-drifting ETFs is the same
  disclosed caveat the Tharp replication carried — a deliberate test of
  whether the claims survive the owner's actual instruments.

---

## 8. Build plan

- **Module**: `engine/tharp_sr_replication.py` — the range-position
  counter (Phase 1), the signal library and per-cell flat-only trade
  builder (Phase 2), the matched-count null engine (one derived seed
  per claim/cell), the Turtle Soup counter, and a print-only report.
  Deterministic; the only seed is `SR_SEED`.
- **Tests**: `tests/test_tharp_sr_replication.py` — an always-run
  synthetic layer (range-position edge cases incl. `high == low` and
  exact ties; each signal's firing rule on hand-built series incl.
  warm-up exclusion and the Wilder recurrence against hand-computed
  values; flat-only non-overlap and end-of-span skip; the
  fixed-at-entry Turtle Soup level; null determinism and per-cell seed
  derivation) plus a result-pin class on the committed OHLC CSVs
  (runs in CI): the C1 strict/loose conditional-vs-base pairs and
  their p's on QQQ and SPY, the C2 and C3 headline pairs, the
  per-signal beating-cell counts and survival verdicts, and the
  Turtle Soup per-panel rates, counts, and both verdict tests.
- **Results surface**: a `docs/explorations.md` entry (or the §6
  escalation path).
- **Plumbing in the build PR**: `ci.yml` pytest list, CLAUDE.md
  symbol-regex additions, README file-table rows.
- **Runtime estimate**: pure-python counting over \~6,700-row CSVs is
  seconds; the 10,000-resample nulls across \~500 cells are the only
  real loop — minutes, and the CI pin class re-runs only the primary
  tickers' cells to stay inside the engine job's budget (the full
  robustness sweep is the module's `__main__`, pinned by its printed
  report where decisive).

---

## 9. Order of operations

1. This design doc merges; the definitions above are frozen.
2. The build PR lands the module and synthetic tests — no measurement
   numbers.
3. One run executes both phases; decisive numbers pin in the test and
   the exploration-log entry in the same PR.
4. §6 governs any escalation; otherwise the support/resistance catalog
   closes with the pins.
