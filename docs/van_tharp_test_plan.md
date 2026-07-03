# Van Tharp test plan — the measurement infrastructure the engine lacks to run his diagnostics

## Status

**PLAN — design only. Nothing measured, nothing built, nothing registered.** No Van Tharp lesson has yet
been tested against this engine. This doc enumerates the infrastructure gaps that block those tests and
sequences the build; it produces no numbers and pins nothing. Any experiment it proposes would be
**EXPLORATORY** — sample-spending, kill-or-justify, never a confirmatory verdict — until it earns a
pre-registration ([docs/prereg_trend_gate.md](prereg_trend_gate.md) is the exemplar). Pinning a future
measurement does **not** promote it to a registered finding; promotion stays CLOSED. This is the same
boilerplate the exploration logs lead with ([docs/explorations.md](explorations.md),
[docs/edge_search.md](edge_search.md)), applied forward to work not yet done.

## Why

The book *Trade Your Way to Financial Freedom* (Van K. Tharp, 2nd ed.; notes at
[research/book-notes/trade-your-way-to-financial-freedom.md](../research/book-notes/trade-your-way-to-financial-freedom.md))
is written from a **trend-following, positive-skew** worldview: a low win rate, a few enormous winners, and
expectancy carried entirely by the right tail — "cut losses short and let profits run." Almost every
strategy in this repo is the **opposite**: short-volatility premium collection — covered calls, short
straddles and strangles, iron condors, credit spreads, the put-side VRP. Those are **negative-skew**: a
high win rate, capped upside, and an occasional fat left tail. The repo's own phrasing for the trade is
"the premium you collect is the upside you give up."

That mismatch is the value, not a problem. Most of Tharp's lessons are **diagnostics** aimed at exactly the
failure mode a negative-skew book structurally carries — the rare large loss that a high win rate hides.
Running his diagnostics against these strategies is a stress test the engine cannot perform today, because
it does not speak his units. This doc lays out what is missing, in dependency order, so the cheapest
high-value diagnostics come first.

## Scope

In scope — the Tharp lessons that are a backtest question this engine could answer:

- Position sizing dominates system choice (Loc 184, 1038, 1047, 727).
- Expectancy is the mean R-multiple; win rate alone lies (Loc 259, 695, 1894).
- Exits dominate entries (Loc 198, 1481, 1867).
- A random entry with good exits still makes money (Loc 202, 1801, 1806).
- Six market types, each needing \~30 or more R-multiples to characterize (Loc 1885–1888).
- Degrees of freedom held to roughly 4–5 (Loc 1025).
- Postdictive and lookahead errors invalidate a backtest (Loc 1028).
- Non-correlated systems lower combined drawdown (Loc 1929, 1932).

Out of scope — not a backtest question for this engine:

- Macro and big-picture framing (P/E regimes, Bretton Woods II, the mutual-fund critique).
- Value investing.
- Trading psychology — the one Tharp weights at \~60% (Loc 525), and the one this engine cannot touch.

Seasonality (Loc 2360–2436) is the lone macro item that could enter as a backtest feature — as a calendar
predicate in the factor-search grammar — and is noted as a possible future factor, not a gap here.

## Step 0 — the measurement substrate (the keystone)

The engine speaks Sharpe, win rate, max drawdown, and the Newey-West HAC t-statistic. It does **not** speak
Tharp's native units: the **R-multiple** (a trade's P&L expressed in multiples of its initial risk),
**expectancy** (the mean R-multiple), and the **SQN** (System Quality Number).

These are closer than they look. The SQN is `sqrt(N) * mean(R) / std(R)` — literally the one-sample
t-statistic of the R-multiple distribution. The repo's Newey-West t-stat
(`short_vol_statistics`, realchains/vol_premium.py:139; `compute_statistics`, engine/cc_backtest.py:600) is
already a stronger, HAC-robust version of the same quantity: it corrects for the serial dependence a naive
SQN ignores. So the engine does not need a new significance authority. It needs the two **inputs** that the
SQN and expectancy require and the engine never computes:

- R-normalized per-trade P&L — a trade ledger with an initial-risk basis (Gap A).
- A per-regime breakdown of those R-multiples (Gap D).

Everything downstream — expectancy, the SQN convenience metric, the win-rate-versus-expectancy flip, the
six-regime R-distributions, the marble-bag resample — depends on Gap A existing first. That is why Gap A is
the keystone.

## The gaps

Each gap below lists what exists today (with file:line citations into the engine), what is missing, what it
blocks, and a rough build size.

### Gap A (keystone) — a trade-level R-multiple ledger

**What exists.** Both engines log trades as an event stream keyed off a single `action` discriminator, with
a *different* payload shape per action — there is no fixed columnar trade schema. The simulated CC engine
appends four shapes in `run_cc_overlay`: `sell` (engine/cc_backtest.py:355), `expiration`
(engine/cc_backtest.py:407), `close` (engine/cc_backtest.py:432), and `close_itm`
(engine/cc_backtest.py:456). The structure engine appends four in `run_real_structure_overlay`: `enter`
(realchains/vol_premium.py:890), `settle_leg` (realchains/vol_premium.py:908), `settle`
(realchains/vol_premium.py:925), and `close` (realchains/vol_premium.py:949). The only per-trade economic
fields recorded are realized P&L (plus `realized_pnl`, `call_value`, `profit_pct` on the CC side; `credit`
on the structure side). The in-memory CC `position` dict (engine/cc_backtest.py:345) holds state, not a
recorded record.

**What's missing.** No per-trade initial-risk basis is recorded anywhere — a grep for
`initial_risk|max_loss|stop_dist|width|risk_basis` returns nothing across engine/cc_backtest.py and
realchains/vol_premium.py. Max adverse excursion (MAE), the worst intratrade mark, is tracked nowhere
either: `worst` appears only in `sensitivity_analysis` (a param-sweep statistic, engine/cc_backtest.py:1289)
and a drawdown comment; the structure mark loop updates `leg['mid']` per day (realchains/vol_premium.py:934)
but never records a running low-water mark. Both statistics functions discard the trade list entirely and
reconstruct everything from the daily equity DataFrame (`short_vol_statistics` reads `daily_equity['equity']`
at realchains/vol_premium.py:181). The need is a common per-trade record — `{entry_date, close_date, pnl,
initial_risk_R, mae}` — emitted by every overlay. `initial_risk_R` is free for the defined-risk structures
(the credit spread and iron condor: long-wing width minus net credit), though that width is never computed
today; undefined-risk overlays need a declared stop or Tharp's average-loss-as-1R fallback (Loc 739).
Because the ledger is non-uniform per action, the R and MAE fields must be threaded through every append
site and through the per-day mark loop where intratrade marks are available but never min-tracked.

**Blocks.** Expectancy, the SQN, the win-rate flip, per-regime R-distributions, and the marble-bag resample
— effectively every experiment in this doc.

**Size.** Medium.

### Gap B — a trade-level resampler

**What exists.** `monte_carlo_shuffle` (engine/cc_backtest.py:1185) resamples the **underlying's daily
returns** to rebuild a synthetic price path and re-runs the overlay on it. The daily simple returns are
computed at engine/cc_backtest.py:1220, shuffled and compounded into a synthetic path, and the overlay is
re-run on that path at engine/cc_backtest.py:1244; the metric is `total_return_pct` and the statistic is the
real path's percentile rank among the shuffles. The trades list is discarded each iteration. This is a
sequence-randomization test on the underlying — it preserves the return distribution and destroys serial
order — and it exists only in the simulated engine (there is no `monte_carlo_shuffle` in realchains/).

**What's missing.** Tharp's marble bag (Loc 671, 747) resamples **trade R-multiples**, not underlying daily
returns — a different axis. The need is a new bootstrap that draws with replacement from Gap A's per-trade R
ledger to build terminal-wealth and drawdown distributions. The existing shuffle answers "is the real
path's order special?"; the marble bag answers "given this trade-outcome distribution, what is the spread of
account outcomes?"

**Blocks.** The risk-of-ruin and terminal-wealth distribution (Experiment 1).

**Size.** Medium (depends on Gap A).

### Gap C — a position-sizing layer

**What exists.** Flat fixed-notional sizing, identical across all three engines. Capital is a flat dollar
input (default $100,000 — engine/cc_backtest.py:1357, realchains/vol_premium.py:826). Contract count is
computed once at t=0 from the first day's price and never reassigned: the CC path does
`num_contracts = int(capital // contract_cost)` (engine/cc_backtest.py:257), the real CC path mirrors it
(realchains/real_cc_backtest.py:313), and every structure overlay routes through the same floor-division in
`run_real_structure_overlay` (realchains/vol_premium.py:834). Returns are measured against the constant
`capital` base, not grown equity (engine/cc_backtest.py:524; realchains/vol_premium.py:182). A grep for
`kelly|fixed.?fraction|percent.?of.?equity|risk.?per.?trade|position.?siz` returns zero matches in the
sizing path.

**What's missing.** Any money-management layer: fixed-fractional percent-of-equity sizing, vol-targeting,
or a stop-based risk unit. There is no equity-curve feedback into size and no per-trade risk amount —
`capital` is a budget, not a risk allowance. The need is a fixed-fractional `%`-of-equity sizer that scales
the unit count by current equity and a declared per-trade risk (which is Gap A's `initial_risk_R`).

**Blocks.** Experiment 1. This is the lesson Tharp weights highest — position sizing \~30% versus system
\~10% (Loc 525) — so it is the engine's largest blind spot relative to the book.

**Size.** Medium (depends on Gap A for the risk unit).

### Gap D — six-regime (vol × direction) bucketing with per-cell counts and R-distributions

**What exists.** `regime_analysis` (engine/cc_backtest.py, returning at engine/cc_backtest.py:855) buckets
CC trade P&L into four directional regimes — bull, bear, sideways, unknown — from `classify_regime`
(engine/cc_backtest.py:725), a price-versus-200-day-SMA classifier with a `±5%` band. Each bucket carries
exactly three fields: `days` (a regime-**day** count, not a trade count), `total_pnl` (summed), and
`avg_pnl_per_day` (total_pnl divided by days). It reads only the `date` and `pnl` columns
(engine/cc_backtest.py:839) and is CC-only — a grep over realchains/ and search/ for `regime_analysis` or
`classify_regime` returns nothing. The volatility axis raw material exists but is unwired: `detect_regime`
(engine/cc_backtest.py:162) returns high/low/normal from rolling vol with `25%` and `15%` thresholds, and
`estimate_iv` (engine/cc_backtest.py:171) multiplies rolling vol by `1.1`/`1.3`/`1.5` per regime — both
live inside IV estimation in `run_cc_overlay`, never used to bucket realized P&L.

**What's missing.** A six-cell bucketing that crosses the vol axis (`detect_regime`, already pinned) with
direction, reporting per-cell **trade counts** and the full per-trade **R-distribution** — not a summed
P&L over a day denominator. The current output gives neither a trade count nor a distribution; pnls are
`.sum()`'d via groupby. The wiring is to cross the existing `detect_regime` output into `regime_analysis`'s
buckets and emit R-multiples (from Gap A) per cell. A structure-overlay extension also needs a `(date, pnl)`
trade list compatible with the CC schema the function hard-codes.

**Blocks.** Experiment 5, including its \~30-trades-per-cell floor (Loc 1885–1888).

**Size.** Small — reuses Gap A and the already-pinned vol classifier.

### Gap E — exit mechanics beyond hold-to-expiry

**What exists.** More exit variety than a single hold-to-expiry, but unevenly. The simulated CC engine has a
profit-target close (action `close`, engine/cc_backtest.py:422) and a deep-ITM close
(action `close_itm`, engine/cc_backtest.py:447). The real CC engine mirrors all three exit triggers —
`hit_target` (realchains/real_cc_backtest.py:467), `deep_itm` (realchains/real_cc_backtest.py:468), and
`hit_stop` (realchains/real_cc_backtest.py:469), the last gated by `stop_loss_mult`
(realchains/real_cc_backtest.py:288) emitting a `close_stop` action — and is pinned by
`TestMsftStopLossRegression` (tests/test_real_cc_backtest.py:1150). Among the seven structures, only
`short_vol` carries early-close management (`early_close_single`, realchains/vol_premium.py:935); the other
six (straddle, iron condor, strangle, risk reversal, credit spread, calendar) carry `management: hold` and
settle only at expiry (the `date >= expiration` branch, realchains/vol_premium.py:915; the calendar adds
staggered per-leg settlement but no discretionary early exit).

**What's missing.** Roll logic exists nowhere — a grep for `roll` finds only comments. No profit-target,
stop, or roll exists for the six `hold` structures. Tharp's claim that exits dominate entries (Loc 198,
1481, 1867) needs the engine to vary the exit on a fixed entry — roll, profit-target, and stop variants
across the structures, not just the CC path. One data point already exists: the stop-loss variant is a
tested-and-rejected exit on real chains — `TestMsftStopLossRegression` pins the stop making results
monotonically worse (no edge), echoing the real-chain collapse of the risk-managed refinement narrated in
[the blog series](../blog/06_real_chains_flip_the_268000.md).

**Blocks.** Experiment 4.

**Size.** Large — it touches engine settlement and mark mechanics.

### Gap F — an entry-selection seam

**What exists.** A partial injection point. The generic `run_real_structure_overlay` takes the selector as a
keyword-only callable — `select` (realchains/vol_premium.py:785), invoked as `picked = select(day, params)`
(realchains/vol_premium.py:869). But every named overlay binds it to a fixed `_legs_*` selector via
`STRUCTURE_SPECS` (realchains/vol_premium.py:718), and the CC path hardwires `select_entry`
(realchains/real_cc_backtest.py:345) and `select_cap_leg` (realchains/real_cc_backtest.py:354) with no
override. No caller, test, or harness ever passes an alternative selector.

**What's missing.** A seeded random-selector that the harness can swap in at the existing `select=` seam
for structures (and an analogous override for the CC path, which lacks one). Tharp's random-entry claim
(Loc 202, 1801, 1806) is exactly this swap: hold the exits fixed, randomize the entry, and measure whether
the edge survives.

**Blocks.** Experiment 2.

**Size.** Small — the abstraction point already exists for structures.

### Gap G — a multi-overlay portfolio harness

**What exists.** Every overlay emits a daily equity stream — the structure engine builds
`daily_equity` with columns `['date', 'equity', 'price', 'rf_credit']` (realchains/vol_premium.py:984), the
real CC engine emits `['date', 'equity', 'price']` (realchains/real_cc_backtest.py:514), and the simulated
CC engine returns a daily curve for `compute_statistics`. Every consumer is single-strategy.

**What's missing.** A harness exists nowhere — a repo-wide grep for
`corrcoef|\.corr(|cov_matrix|correlation_matrix|portfolio_weight|risk_parity|combine_streams` finds only a
single-factor Information Coefficient (factor/factor_backend.py) and a scout's Spearman
(search/explorations.py:335), neither of which aligns multiple overlay streams. The closest portfolio
construct, `long_short_returns` (factor/factor_mechanism.py), builds one cross-sectional equity book from a
single signal — orthogonal to combining option-overlay streams. The need is a harness that aligns several
overlays' `daily_equity` on a common date index, builds a cross-overlay correlation/covariance matrix, and
sizes a combined book — Tharp's non-correlated-systems lesson (Loc 1929, 1932).

**Blocks.** Experiment 6.

**Size.** Moderate.

## What is not a gap

The honesty plumbing is heavily built out in this repo, and **none of it needs to change** for these
experiments. The pinned regression tests, the `explorations.py` / `test_explorations.py` /
`docs/explorations.md` trio, the FDR control (Benjamini-Yekutieli and the e-LOND lifetime stream;
[docs/edge_search.md](edge_search.md)), and the sealed vault (QQQ held out) are all in place. What these
experiments need is new **measurement primitives** — a trade ledger, a sizer, a regime cross, a portfolio
harness — not new epistemics. Every experiment below rides the existing governance unchanged. State this
plainly so the build does not reinvent the rails: the work is downstream of the honest core, not inside it.

Two confirmations fall out for free, needing no new test:

- Degrees of freedom held to \~4–5 (Loc 1025) is already enforced by the closed `ALLOWED_GRID` /
  `grid_universe_size` and the walk-forward window discipline.
- Postdictive and lookahead errors (Loc 1028) are already guarded by settlement-at-or-before-expiry and the
  `CHAIN_CLEAN_START` era clip.

## Build order

Dependency-sorted. The cheapest high-value path is A → D → (C+B).

| Order | Gap | Why here | Size |
| --- | --- | --- | --- |
| 1 | A — trade-level R ledger | Keystone; every other measurement depends on it. | Medium |
| 2 | D — six-regime R-buckets | Reuses A plus the already-pinned vol classifier; cheapest follow-on. | Small |
| 3 | C + B — sizing + trade resampler | Tharp's central thesis (sizing > system); together they enable the risk-of-ruin Monte Carlo. | Medium |
| 4 | E — exit mechanics | Engine settlement and mark changes; the heaviest lift. | Large |
| 5 | F — entry seam | Small swap at an existing seam; sequenced after the heavier work. | Small |
| 6 | G — portfolio harness | Aligns existing per-overlay streams; independent of the rest. | Moderate |

## Gap-to-experiment mapping

| Experiment | Tharp lesson | Gaps required | Existing machinery it reuses |
| --- | --- | --- | --- |
| 1 — position-sizing Monte Carlo | Sizing dominates system (Loc 184, 525) | A, B, C | The marble-bag resample over the new R ledger; fixed-fractional sizer |
| 2 — random entry, good exits | Entries don't matter (Loc 202) | F | The `select=` seam plus the edge-search re-tag idiom, fixed seeds |
| 3 — win-rate vs. expectancy flip | Win rate lies (Loc 259, 695) | A | Pure reporting over the R ledger — no new engine run |
| 4 — exits change expectancy | Exits dominate (Loc 198) | E | The stop-loss data point already exists (tested-and-rejected on real chains) |
| 5 — six-regime R-distributions | Six market types, \~30 R each (Loc 1885) | D | `detect_regime` × direction, with the \~30-trade-per-cell floor |
| 6 — non-correlated portfolio | Diversification lowers drawdown (Loc 1929) | G | Per-overlay `daily_equity` streams aligned on a date index |

Two items are **confirmations, not new experiments**: degrees of freedom (≤5) is already enforced by the
closed grammar and walk-forward, and lookahead safety is already guarded by settlement ≤ expiry plus
`CHAIN_CLEAN_START`.

## Honesty rails

Every experiment here is **EXPLORATORY** — sample-spending, kill-or-justify, never a registered verdict.
The rails are the repo's existing ones, applied unchanged:

- A pass earns a **pre-registration** (`docs/prereg_*.md`), not a headline. A passing scout is a candidate
  for registration, not a finding.
- A kill lands in the `explorations.py` / `test_explorations.py` / `docs/explorations.md` trio, pinned so
  the dead end stays settled ([docs/explorations.md](explorations.md)).
- Multi-comparison sweeps — the random-entry batch (Exp 2) and the sizing sweep (Exp 1) — ride the e-LOND
  lifetime stream and the BY diagnostic, with QQQ held out in the sealed vault
  ([docs/edge_search.md](edge_search.md)).
- Per-ticker tagging, the `CHAIN_CLEAN_START` era clip, and fixed RNG seeds apply, exactly as the scouts
  already do them.
- Do **not** re-pin any canonical number. New measurements get new tests; the existing regression pins are
  untouched.

## Open questions

- **R for undefined-risk structures.** Defined-risk overlays (credit spread, iron condor) get
  `initial_risk_R` for free as width minus net credit. The undefined-risk ones (covered call, short
  straddle/strangle) need a convention: a declared stop distance, or Tharp's average-loss-as-1R fallback
  (Loc 739). Pick one and pin it before Gap A's ledger emits R — the choice changes every R-multiple
  downstream.
- **README registration.** Plan docs do get a `## Project layout` file-table row in
  [README.md](../README.md) (the `_plan` docs are all listed there). Add a
  `| [docs/van_tharp_test_plan.md](docs/van_tharp_test_plan.md) | … Design/plan only — nothing tested or
  registered yet |` row when this lands, per the cross-surface rule for any new public doc.
- **SQN versus the Newey-West t-stat.** The SQN is the naive one-sample t of the R-distribution, which the
  repo's HAC-robust t already dominates. Compute the SQN as a labelled convenience for readers fluent in
  Tharp's units — not as a second significance authority. The Newey-West t stays the judge.
