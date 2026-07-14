# Gap D — six-regime R-distributions (DESIGN + AS-BUILT)

## Status

**BUILT in the same change as this doc** — `six_regime_map` (engine/cc_backtest.py) +
`regime_ledger_statistics` / `SIX_REGIME_CELLS` (common/trade_ledger.py), pinned by the
`TestRegimeLedgerMechanics` / `TestRegimeLedgerRegression` classes in
[tests/test_trade_ledger.py](../tests/test_trade_ledger.py). Gap D from
[docs/van_tharp_test_plan.md](van_tharp_test_plan.md), the first consumer of the Gap A ledger
([docs/van_tharp_gap_a.md](van_tharp_gap_a.md)). Every number is **EXPLORATORY** — sample-spending,
kill-or-justify, never a registered verdict; per-cell `sqn` / `r_newey_west_t` are reported, never gates,
and the daily Newey-West t stays the sole significance authority.

## Why

Van Tharp's system-characterization rule (Loc 1885–1888): know your system's R-distribution in each of six
market types — up, sideways, down, each quiet or volatile — with \~30 or more completed R-multiples per type
before trusting any of them. The engine had a three-bucket directional `regime_analysis` (bull/bear/sideways,
CC-only, P&L summed over a day denominator, no trade counts) and an unwired volatility axis: `detect_regime`
classifies rolling vol but only feeds IV estimation. Gap D crosses the two axes and reports per-cell trade
counts and R-distributions — the shape Experiment 5 needs.

## Design

Two functions, composed at the caller so `common/` stays a leaf:

- **`six_regime_map(dates, prices)`** (engine/cc_backtest.py) — the per-day cell label. Direction is
  `classify_regime` (price vs 200-day SMA, ±5% band — the pinned classifier `regime_analysis` already uses).
  Volatility is `calc_rolling_volatility` (30-day) fed to `detect_regime`, binarized on the engine's pinned
  high-vol boundary: **volatile iff rolling vol > 25%**; `normal` and `low` are both quiet. A fixed threshold
  keeps the axis lookahead-clean — a data-relative split (a sample median) would classify early days with
  future data, a postdictive error. Both axes carry start-of-day semantics (`.shift(1)` on direction; the vol
  window ends at the prior close), matching `regime_analysis`'s no-peek convention. Warmup days (either axis)
  are `unknown`.
- **`regime_ledger_statistics(records, regime_by_date, min_trades=30)`** (common/trade_ledger.py) — buckets
  `TradeRecord`s by **close date** (the regime in force when the outcome was realized, the `regime_analysis`
  convention) and runs `ledger_statistics` per cell, plus `meets_floor`, Tharp's \~30-trade sample-adequacy
  flag. Every cell is always present — a zero-trade cell reports empty statistics, so under-sampling is
  visible, not silent. The engine never imports `common`'s consumer and `common` never imports the engine;
  the caller composes them.

The price for the fixed 25% split is bluntness: a name whose vol regime lives mostly below 25% (post-2010
SPY) concentrates in the quiet cells. That is a property of the era being measured, not a bug — and the
pinned empty cell records it.

## What the first measurement showed (pinned)

The two Gap A ledgers, bucketed:

| Cell | MSFT CC (182 trades) | SPY short vol (174 trades) |
| --- | --- | --- |
| bull_quiet | n=85, **−0.61R**, floor ✓ | n=93, **−1.18R**, 50.5% wins, worst MAE −11.4R, floor ✓ |
| bull_volatile | n=27, −0.62R | **n=0** — no trade closed on one of the span's 8 bull_volatile days |
| sideways_quiet | n=13, +0.30R | n=51, +0.24R, floor ✓ |
| sideways_volatile | n=16, −0.88R | n=9, −0.93R |
| bear_quiet | n=3, +0.81R | n=3, +1.00R |
| bear_volatile | n=25, **+0.40R**, 88% wins | n=9, **+0.58R**, 89% wins |

The remaining 13 (MSFT) and 9 (SPY) trades closed on warmup-era `unknown` days — pinned in the test with
everything else: `TestRegimeLedgerRegression` asserts every cell's trade count and expectancy (plus the
quoted win rates and the SPY worst MAE), so each figure above traces to an assertion.

Two findings:

- **The 30-trade floor fails almost everywhere.** One MSFT cell and two SPY cells clear it. On a single
  ticker, most of Tharp's six types are too thin to read — the sample-adequacy verdict the test plan
  predicted, and the concrete argument for the multi-ticker widening below.
- **The one readable bleed is `bull_quiet` — and the crash cells' sign points the other way.** The quiet
  grind *up* through the strike is what bleeds a call seller (assignment after assignment), and `bull_quiet`
  clears the floor on both tickers. The positive `bear_volatile` cells (\~88% wins) sit **under the floor**
  (n=25 and n=9), so their sign is a sample observation supported by the payoff mechanics — a crash moves
  price away from a short call and the elevated premium cushions it — not a readable expectancy. Read
  narrowly: the naive "vol spikes hurt vol sellers" story finds no support here for the call wing, but the
  counter-sign is not yet measured to the floor; the put wing would need its own measurement either way.

## What this does not claim

No cell's expectancy is a verdict. The under-floor cells are explicitly unreadable; the two floor-clearing
negative cells restate the Gap A headline (negative per-cycle expectancy on the raw option-cycle basis) with
location information added. Promotion unchanged: a regime-conditioned trading rule built on these numbers
would be a new exploratory scout under the usual rails (the explorations pattern, FDR if it becomes a
campaign), not a registered finding.

## Next

- The obvious widening is **more tickers** (the structure-campaign cross-section already loads seven), which
  is also Tharp's own prescription for thin cells (Loc 1929: more independent markets, more opportunity).
- Gap C+B (the fixed-fractional sizer + marble-bag resampler) can now draw per-regime R-distributions
  instead of pooled ones.
