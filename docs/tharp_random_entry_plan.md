# Replication design: Tharp's random-entry experiment on the repo's universe

**Status:** DESIGN — a replication *study*, exploratory track. Nothing here is
an edge hunt: Phase 1 tests whether Tharp's reported result reproduces on this
repo's data at his own coordinates; Phase 2 subjects it to the nulls his era
did not run. Sample-spending, kill-or-justify; a Phase-2 survivor earns a
registration, never a headline from this doc.

**Date:** 2026-07-18.

**The claim under test.** Van Tharp's random-entry experiment (with Tom
Basso; *Trade Your Way to Financial Freedom*, exits chapter — practitioner
lore, hedged as such): enter each market in a coin-flip direction, always in
the market, exit on a **3×ATR trailing stop**, size at **1% equity risk per
position**, on a diversified \~10-market futures basket — and, as he reports
it, the system made money consistently across runs. His moral: entries are
the least important component; exits and position sizing carry the system.
Phase 1 tests the sentence; Phase 2 tests the moral.

---

## 1. The system, frozen (his coordinates, our conventions)

- **Direction:** per instrument, a fair coin flip at entry (seeded RNG,
  stdlib `random.Random` — the engine-adjacent convention).
- **Exit:** a trailing stop at `3.0 × ATR(20)` from the daily close
  (long: `trail = max(trail, close − 3·ATR)`; short mirrored). The exact ATR
  period in the original study is lore; **20 is fixed here by choice**, on
  the record. Evaluated once per day on the close; fills at that close — the
  repo's EOD stop-market convention (flatters the stop; carried caveat).
- **Re-entry:** the next trading day after an exit, fresh coin flip — the
  one-day-gap convention (a documented deviation from his literal
  always-in-the-market; the gap is one day of the \~40-day median hold).
- **Sizing:** risk 1% of *current total portfolio equity* per position;
  initial risk per share = the entry stop distance (`3·ATR`); shares =
  `floor(0.01·E / (3·ATR))`. This is his percent-risk model, and the entry
  stop distance is his R — the ledger measures in exactly those units.
- **Portfolio:** all nine instruments run concurrently against one equity
  stream, marked daily. Starting equity $100,000.
- **Careers:** **100 seeded careers**, career `i` seeded `20260719 + i`
  (20260717/-18 are taken by the jitter ensemble and the bootstrap; the
  base is the planned run date). One career = one full coin-flip history
  across the basket and span.

## 2. The universe and data translation

- **Basket (9):** SPY, QQQ, IWM, GLD, TLT, XLE, EEM, MSFT, NVDA — the repo's
  onboarded universe as the futures-basket proxy: equity beta three ways,
  gold, long bonds, energy, EM, two single names. **Named honestly:** his
  basket was more independent than three-equity-ETFs-plus-friends; the
  effective breadth here is maybe 5–6, not 9, and that caveat rides every
  surface.
- **Span:** 2000-01 → 2026-06 — \~26 years that **contain two real bear
  markets** (2000–02, 2008–09), unlike every option experiment in this repo.
  The price data has no GFC hole; this study finally uses that.
- **Data:** daily **OHLC** per ticker from yfinance (free, split-adjusted
  price series — a small `pipeline/download_prices.py` extension or sibling
  fetcher; git-tracked like the other price CSVs). True range needs
  highs/lows the current close-only files lack. The 1-minute archive can
  cross-check the derived daily H/L on any disputed day, but the committed
  source is the daily fetch (the minute bars are as-traded and would need
  split re-scaling — a complication with no payoff here).
- **Dividends:** price-return series, dividends ignored on both sides
  (longs forgo income, shorts forgo liability — roughly netting across coin
  flips; committed simplification, direction \~neutral, stated).
- **Costs:** headline **0.5 bp** of traded share notional per fill, with the
  0 / 1 bp curve reported beside it, plus **0.5%/yr borrow on short
  notional** (0 / 1% curve) — the friction futures never charged him.

## 3. Phase 1 — the replication proper

Run the 100 careers; measure in his units through the existing machinery
(Gap A ledger with R = initial stop distance; expectancy-R, win rate, SQN;
Gap D regime cells; equity curves):

- **R1 (his sentence):** the fraction of careers with positive net
  expectancy, and the career-expectancy distribution (median, p10–p90). His
  reported version implies \~all careers profitable over long runs; anything
  materially below that is a failed replication *on this universe*, scoped
  as such.
- **R2 (the shape check):** win rate in the trend-follower's \~30–40% band
  with avg-win/avg-loss well above 2 — the signature his mechanism predicts.
  A profitable ensemble with the *wrong shape* is already evidence the
  profits come from somewhere else.

## 4. Phase 2 — the nulls his era did not run

Four arms, each answering one deflation hypothesis, all on the same careers:

1. **The drift twin (replication null).** Per career × instrument, hold the
   career's *realized average signed exposure* as a constant position over
   the span; same costs. If the system ≈ its twin, "exits and sizing"
   contributed a costly way to hold an average tilt. This is the put-spread
   registration's replication-null pattern, aimed at trend-following.
2. **Placebo exits (the mechanism null).** Keep each career's entry
   sequence and directions; replace exit times with draws from the pooled
   holding-period multiset (structure-matched, seeded; 1,000 placebo
   careers). The trailing stop's specific claim — cut losers, ride winners —
   must beat skill-free exits of the same cadence, or the mechanism is
   empty. (The trend-gate's §5.1 machinery, pointed at exits.)
3. **The no-stop control.** Coin-flip, hold a fixed H = the real ensemble's
   median holding period, re-flip; same sizing. Tharp's implicit baseline,
   made explicit.
4. **The sizing ablation.** The same trade streams replayed under
   fixed-fractional, his percent-risk, and percent-volatility (the Gap C+B
   replay) — separating "sizing prevented ruin" from "sizing created
   return," the distinction his narrative blurs. Prior from the call-spread
   exploration: on a book whose R already *is* the risk unit, vol-scaled
   sizing was redundant.

The endogenous-tilt diagnostic is reported with all arms: the ensemble's
average net long exposure by regime (a trailing stop on drifting markets
stops shorts fast and lets longs run — the system manufactures long tilt
from randomness; how much of R1 is that?).

## 5. Machinery and homes

- **New:** `engine/tharp_random_entry.py` — a \~150-line shares-only
  simulator (ATR, trail, coin flips, percent-risk sizing, portfolio
  marking), no options, no chains; plus the daily-OHLC fetch addition; plus
  `tests/test_tharp_random_entry.py` — always-run synthetic mechanics
  (hand-derived trail/sizing/flip determinism, cost accounting) and the
  dataset-gated ensemble pins once run. CI: the engine job's pytest line.
- **Reused:** Gap A ledger (R = initial risk — his definition, natively),
  Gap C+B sweeps, Gap D regime bucketing, the placebo-family inference
  pattern, the exploration-log pinning discipline.

## 6. Priors, split and on the record

**For replication:** the loss-cut/ride asymmetry is a real convexity
mechanism; diversified trend-following was genuinely profitable in his era;
two bear markets in-span give shorts somewhere to win. **Against the
mechanism surviving Phase 2:** post-2000 equity-heavy baskets are where
trend-following degraded; the endogenous long tilt plus secular drift can
manufacture most of R1 without any exit skill; borrow and spread costs bite
ETFs harder than futures. **Modal expectation:** Phase 1 replicates in
soft form (most careers profitable, right shape), and Phase 2 deflates it —
the drift twin and placebo exits absorb the result, and the honest sentence
becomes "the profits were diversification plus drift harvested by an
endogenous tilt, not exit skill." Either outcome pins; a Phase-2 survivor
would be the most interesting result this repo has produced and would go to
registration, not to a headline.

## 7. Order of operations

1. Daily-OHLC fetch + the simulator + synthetic mechanics tests (green
   before any ensemble run).
2. Phase 1: the 100-career ensemble; pin R1/R2 + the exploration-log entry.
3. Phase 2: the four arms; pin; extend the entry.
4. Cross-surface: symbol regex, README rows, CI line — the standard sweep,
   **including the generative layer check** (n/a here — no grammar change —
   but the checklist item exists now for a reason).

Out of scope, each its own design: any registered verdict; ATR stops on the
options books (a Gap E widening); optimizing any parameter of this system
(the point of a replication is running *his* numbers, not better ones).
