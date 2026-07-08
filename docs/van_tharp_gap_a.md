# Gap A — the trade-level R-multiple ledger (DESIGN / build spec)

## Status

This is a **DESIGN document — a build spec, PLAN-level. No code is written yet.** It designs Gap A
from [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the keystone measurement primitive the rest of
that plan depends on. It adds a measurement substrate, not new epistemics: it rides the repo's existing
honesty governance unchanged (no new FDR control, no new registration). Any number this ledger eventually
pins is **EXPLORATORY** — sample-spending, kill-or-justify — in the same sense as the negative-results log
([docs/explorations.md](explorations.md)); pinning a number prevents re-work, it does not promote
the ledger to a confirmatory finding. No canonical, registered number is re-pinned by this work. It is new
measurement backed by new tests.

## Why — what this unblocks

Today the overlays log an **event stream**, not a trade ledger. Both engines append action-keyed event
dicts with a different payload shape per action — the simulated CC engine emits `sell`
(engine/cc_backtest.py:355), `expiration` (engine/cc_backtest.py:407), `close` (engine/cc_backtest.py:432),
and `close_itm` (engine/cc_backtest.py:456); the structure engine emits `enter`
(realchains/vol_premium.py:890), `settle_leg` (realchains/vol_premium.py:908), `settle`
(realchains/vol_premium.py:925), and `close` (realchains/vol_premium.py:949). The only per-trade economic
field recorded is realized dollar P&L. No trade carries an initial-risk basis, and max adverse excursion
(MAE) is tracked nowhere — the structure mark loop refreshes `leg['mid']` every day
(realchains/vol_premium.py:934) but never keeps a running low-water mark. Both statistics functions then
discard the trade list and rebuild everything from the daily equity curve: `compute_statistics`
reconstructs its return series from `daily_equity['equity']` and `daily_equity['price']`
(engine/cc_backtest.py:656-668), and `short_vol_statistics` does `np.diff(eq)/capital` on
`daily_equity['equity']` (realchains/vol_premium.py:181-187). Neither ever sees `trades`.

Gap A supplies the missing substrate: a uniform per-trade record carrying dollar P&L, an initial-risk basis
`R`, the R-multiple, and MAE. From that one substrate the downstream Van Tharp measurements follow:

- Expectancy and the System Quality Number (SQN) are means and t-like ratios over the R-multiple column.
- The win-rate-versus-expectancy flip — a high win rate hiding a negative expectancy — needs per-trade R,
  not an equity curve.
- Per-regime R-distributions (Gap D) reuse the ledger's `(date, pnl)` pair, which
  `regime_analysis` already consumes.
- The marble-bag resample (Gap B) draws from the R-multiple column instead of the underlying's returns.

## Trade record schema

A uniform dataclass, `TradeRecord`, is the single columnar shape every overlay reduces to. It keeps
`(entry_date, close_date, pnl)` so it is drop-in compatible with `regime_analysis`, which reads a trade
list as `list[dict]` and projects it to `['date', 'pnl']` (engine/cc_backtest.py:839-847). The added fields
carry the risk basis, R, the R-multiple, MAE, and the win/loss outcome.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TradeRecord:
    strategy: str          # e.g. "covered_call", "short_straddle", "credit_spread"
    ticker: str
    entry_date: str        # kept for regime_analysis (date, pnl) compatibility
    close_date: str
    pnl: float             # realized dollars, already computed by the engine
    risk_basis: str        # WHICH R convention was used (audit trail): one of
                           # "defined_max_loss" | "stop_distance" | "premium_collected"
                           # (ex-post "avg_loss_1R" is applied at the statistics layer, not here)
    initial_risk: float    # R, dollars, > 0
    r_multiple: float      # pnl / initial_risk
    mae: float             # worst intratrade unrealized P&L, dollars, <= 0
    mae_r: float           # mae / initial_risk
    outcome: str           # "win" | "loss", from sign of pnl
```

The `pnl` field is the same rounded dollar P&L the engines already compute at every settle/close site
(`round((entry_credit + structure_flow) * shares, 2)` at realchains/vol_premium.py:926; the CC assignment
and close P&L at engine/cc_backtest.py:407-413 and :432-440). The ledger never recomputes P&L — it reads
what the engine already booked.

## Initial-risk (R) derivation

`R` is set by a **pluggable, per-strategy `risk_basis`**, and every trade records which basis produced it.
The R convention changes every R-multiple downstream, so it is a **declared, pinned choice — not an implicit
default**. Three entry-time bases are built in; a fourth (ex-post) normalizer lives at the statistics layer.

This resolves the open question from the plan (docs/van_tharp_test_plan.md:304-308): defined-risk families
get an exact, entry-time R; undefined-risk families default to `premium_collected` (entry-time, honest about
the fat left tail), with `avg_loss_1R` offered as a labelled ex-post cross-check.

Note on units: `entry_net` and the derived `entry_credit` are **per-share** quantities, and leg `strike` is
in price points, so every dollar R below scales the per-share basis by the loop-level `shares`
(= `100 * num_contracts`, realchains/vol_premium.py:837). `shares` is not a leg-dict field — it is a scalar
each overlay already holds — so it must be threaded to the ledger alongside the per-share leg data.

| Family | Overlays | `risk_basis` | R formula | Source fields |
| --- | --- | --- | --- | --- |
| Defined-risk | credit spread, iron condor | `defined_max_loss` | `(wing width − net credit per share) × shares` | Leg `strike` + `sign` (width per share); `entry_net` (credit per share); `shares` (dollar scaling) |
| Stopped seller | real covered call with `stop_loss_mult` set | `stop_distance` | loss realized at the stop = `(stop_loss_mult − 1) × premium_collected × shares` | `stop_loss_mult`, `premium_collected`, `shares` |
| Undefined-risk | short straddle / strangle / risk reversal / calendar, and unstopped covered call | `premium_collected` | `net premium collected (entry credit) × shares` | `entry_credit` / `premium_collected`, `shares` |

**Defined-risk — `defined_max_loss`.** Both wing width and net credit are computable from the leg dicts at
entry, and the dollar figure scales them by `shares`. Each leg carries `strike` and `sign`, so
`width = abs(short_strike − long_strike)` is direct; the credit-spread selector returns a short put and a
long put wing (realchains/vol_premium.py:586-593) and the iron-condor selector returns four legs with two
call and two put strikes (realchains/vol_premium.py:455-464). The net credit per share is the same quantity
the engine already sums, `sum(-leg['sign'] * leg['entry_net'])` (realchains/vol_premium.py:876), and
`entry_net` already bakes in per-leg commission (realchains/vol_premium.py:413-415). Both families use the
`net_positive` entry guard (realchains/vol_premium.py:874), so the net credit is guaranteed positive at
entry. The max-loss basis is exact and entry-time; it is reconstructable from the leg strikes and
`entry_net` plus the loop-level `shares`, so the ledger must receive the leg strikes/signs (not just the
scalar credit the `enter` event carries today — see A1 below).

**Stopped seller — `stop_distance`.** The real covered-call stop is a multiple of the net premium collected,
keyed on the buyback cost: `hit_stop = close_ref >= premium_collected * float(stop_loss_mult)`
(realchains/real_cc_backtest.py:469-470), with `stop_loss_mult` read from params
(realchains/real_cc_backtest.py:288). So the overlay's initial risk is the credit given back when the
buyback reaches the stop level: `R = (stop_loss_mult − 1) × premium_collected × shares`. For the classic
2×-entry stop this is exactly one premium's worth. The stock notional
(`shares × entry_price`) is **not** the overlay's R — it is the buy-and-hold capital base, whose downside the
covered call barely caps; using it as R would conflate the equity position with the option overlay.

**Undefined-risk — `premium_collected`.** For undefined-risk sellers the only entry-time premium quantity is
`entry_credit` — the net per-share premium recorded to the `enter` event as `credit`
(realchains/vol_premium.py:890-891) and scaled to dollars by `shares` (= `100 * num_contracts`,
realchains/vol_premium.py:837). For all-short structures (straddle, strangle — two short legs each) this is
an unambiguous positive credit. The mixed-sign risk reversal (short put + long call,
realchains/vol_premium.py:557-563) and the calendar (short near + long far,
realchains/vol_premium.py:653-660) use the `each_short_positive` guard
(realchains/vol_premium.py:872), which only checks each short leg — so their `entry_credit` is signed and can
be a net debit. **The R basis for those two is the absolute net premium at risk, floored positive**, and the
`risk_basis` string records that the debit case was normalized. Premium-collected R makes the fat left tail
**visible**: an undefined-risk tail loss surfaces as an R-multiple well past `−1R`, exactly the behavior the
Van Tharp view wants to expose.

**Ex-post normalizer — `avg_loss_1R` (statistics layer only).** Tharp's fallback (Loc 739) sets 1R to the
mean of the strategy's own losing trades. It is applied inside `ledger_statistics`, **not** at record-build
time, and it is **loud about being ex-post — not knowable at entry.** It is offered as the Tharp-comparison
view, never the primary. The primary R for every trade is the entry-time basis above; `avg_loss_1R` is a
labelled cross-check the reports may show alongside it.

## MAE — threading the worst intratrade mark

MAE is the worst intratrade unrealized P&L (`<= 0`), threaded as a running `worst_unrealized` on the
in-flight position through the daily mark loop. At entry it is 0; on each marking day it becomes
`min(worst_unrealized, current_unrealized_pnl)`. The current unrealized P&L is already computed in both
engines, so no new mark is fetched — only a running min is stored and read off at the exit sites.

**Simulated CC engine.** The daily-equity open-position block recomputes the call value with `bs_price` each
day (engine/cc_backtest.py:504) and forms the open-overlay unrealized P&L as
`(position['premium_collected'] - call_value) * shares` (engine/cc_backtest.py:506). Track
`min` of that expression, reset at entry (engine/cc_backtest.py:345), read at each exit/append site. Because
the proxy engine recomputes the mark fresh every day, its MAE has no stale-mark issue.

**Real CC engine.** The same shape holds: `spread_mark` is the current close cost
(realchains/real_cc_backtest.py:510-512) and the open-overlay unrealized P&L is
`(position['premium_collected'] - spread_mark) * shares` (realchains/real_cc_backtest.py:513). Reset at entry
(realchains/real_cc_backtest.py:362), min-track daily.

**Structure engine.** Thread MAE at the mark step (realchains/vol_premium.py:976-981), after per-leg
`mid` is refreshed (realchains/vol_premium.py:934). The running unrealized P&L is
`(entry_credit + sum(-leg['sign'] * leg['mid'])) * shares` — the current cost to close against the credit
collected — using only `entry_credit` (a stable loop scalar, realchains/vol_premium.py:876) and the current
mids, both already in hand. Reset MAE at entry (realchains/vol_premium.py:879) and finalize it at the
settle/close sites (realchains/vol_premium.py:925-929, :949-952).

**Do entry marks need retaining?** No. The structure engine mutates `leg['mid']` in place each day
(realchains/vol_premium.py:934), so the original entry mid is not preserved — but the running unrealized P&L
needs only `entry_credit` plus the current mids, never the entry mid, so MAE threads cleanly with existing
state. A per-leg MAE (as opposed to per-structure) would require a new `leg['entry_mid']` field in the
selectors, which is out of scope for Gap A; the leg schema stays unchanged.

**Two conventions to document, not fix.** (1) MAE is a **daily-bar** measure — it uses closing marks, not
intraday extremes, so it understates the true worst excursion. (2) On the real path, missing-quote days
carry the prior mark forward (`last_mid`/`real_delta` refresh only when a quote prints,
realchains/real_cc_backtest.py:445-448; carry-forward noted at realchains/real_cc_backtest.py:482), so
real-engine MAE only updates on real-quote days and inherits the same stale-mark convention as its daily
equity. Both are stated plainly in the record and the doc; neither is silently repaired.

## Architecture and module layout

The reduction lives in **one testable place**: a new dependency-light module, `common/trade_ledger.py`,
importable by both `engine/` and `realchains/`.

`common/` is the single shared seam both packages already depend on — engine imports it at
engine/cc_backtest.py:2 and engine/make_figures.py:44, realchains at
realchains/real_cc_backtest.py:45, realchains/run_registered_vrp.py:16, and
realchains/walk_forward_real.py:38, all via `from common.paths import data_path`. `common/` today holds only
`__init__.py` and `paths.py`, and `paths.py` imports nothing beyond `pathlib` — so `common/` is a leaf
module with no import into `engine/` or `realchains/`. Placing the ledger there keeps the dependency
direction clean (no cycle) and lets both engines feed it the same trades list they already return.

The module exposes three public symbols:

- `TradeRecord` — the frozen dataclass above.
- `build_trade_ledger(events, risk_basis, ...)` — reduces one overlay's event stream into a list of
  `TradeRecord`, pairing each `enter`/`sell` with its matching `settle`/`close`/`expiration`, computing
  `pnl`, `initial_risk`, `r_multiple`, and `mae` per the declared `risk_basis`.
- `ledger_statistics(records)` — returns `{n, expectancy_r, sqn, r_newey_west_t, win_rate, avg_win_r,
  avg_loss_r, mae_r_distribution}`.

**The `(summary, trades, daily_equity)` tuple is the universal overlay return shape across both engines**, so
`build_trade_ledger` reads the same `trades` list `regime_analysis` already consumes, keying only off the
fields it needs and ignoring the rest — exactly as `regime_analysis` projects to `['date', 'pnl']`
(engine/cc_backtest.py:839-847). One caveat the reducer must handle: the `enter` event carries only `credit`
(per-share net) and `legs` (an integer count), not the leg strikes/signs
(realchains/vol_premium.py:890-891), so the defined-risk width is **not** recoverable from the event stream
as it stands — the A1 payload add must surface the leg strikes/signs (or a precomputed R) on entry. Wins and
losses are counted today per-trade inside the overlay loops (engine/cc_backtest.py:588-590;
realchains/vol_premium.py:923-924, :947-948) and surfaced in the summary; the ledger recomputes `outcome`
from the sign of `pnl` so the ledger is self-contained and does not depend on the loop-level counters.

**Three statistics, one judge.** `ledger_statistics` reports the trade-level significance family as three
labelled columns; only the last is an authority.

- `sqn` — Tharp's `SQN = sqrt(N) * mean(R) / std(R)`, the naive one-sample t of the R-distribution. It is
  kept solely because Tharp's interpretation bands are calibrated to this exact formula; an HAC-corrected
  value would no longer map to them. It is labelled anti-conservative: short-vol trade outcomes are
  positively autocorrelated through regime persistence, so the naive t overstates.
- `r_newey_west_t` — the HAC-honest sibling: the same Bartlett-weighted Newey-West correction the repo
  already uses (`_newey_west_t`, factor/factor_backend.py:123, auto-lag `L = 4·(n/100)^(2/9)`), applied to
  the R-multiple column. R-normalization suits it — dividing each trade by its own initial risk strips
  cross-trade scale differences, so the series is better behaved than raw dollar P&L. Its lag lives in
  trade-index units: lag 1 is one \~30-day cycle, not one day. Dependency note: `common/trade_ledger.py`
  cannot import from `factor/` without inverting the leaf-module direction, so the \~12-line function is
  either duplicated with a cross-reference comment or hoisted into `common/` — an implementation-time
  choice.
- The **daily Newey-West HAC t** (`short_vol_statistics` / `compute_statistics`) remains the sole
  significance authority, unchanged, for three reasons. The daily series carries thousands of observations
  against the ledger's low-hundreds of trade cycles, and NW is asymptotic — its standard errors bias down
  in small samples, so the trade-level t is the weaker instrument. The daily series sees the intratrade
  mark-to-market path, which trade space collapses to one number per cycle. And the FDR pipeline
  (`_asymptotic_p` → e-LOND) is keyed to the daily t, so gating on a second t would recreate the
  two-authorities problem. `sqn` and `r_newey_west_t` are reported, never gates.

**The only engine-hot-loop change** is recording `worst_unrealized` and emitting it (plus the entry legs /
credit needed for R) on the close/settle events. Everything else — pairing, R, the R-multiple, the
statistics — lives in the reduction, off the hot loop, in one place.

## Phased plan

The phasing lands the cheap expectancy/SQN part before touching the MAE hot loop.

- **A1 — the reducer and stats, no hot-loop change.** Build `TradeRecord`, `build_trade_ledger`, and
  `ledger_statistics` with the defined-risk and premium R bases. The premium basis needs only `credit`
  (already on the `enter` event) and `shares`; the defined-risk basis additionally needs the leg
  strikes/signs for the wing width, which the `enter` event does not carry today, so A1 adds those (or a
  precomputed per-trade R) to the entry payload. Both are one-time entry-event additions, so this phase
  touches no daily loop.
- **A2 — thread MAE.** Add the running `worst_unrealized` min to the three daily mark loops
  (engine/cc_backtest.py:504-506, realchains/real_cc_backtest.py:510-513, realchains/vol_premium.py:976-981)
  and emit it on the settle/close events. This is the one hot-loop touch.
- **A3 — wire and pin.** Wire `ledger_statistics` into the reports and pin the tests.

## Test plan

Follow the repo's two-layer pattern: an always-run synthetic layer plus a dataset-gated regression class.
The best exemplar is `tests/test_vol_premium.py` (the `_scenario` helper at :78, always-run `*Mechanics`
classes at :99/:148/:184, dataset-gated `*Regression` classes with `@pytest.mark.skipif` + a `scope='class'`
`market` fixture at :391/:416).

- **Always-run synthetic layer.** A plain `class TestTradeLedgerMechanics:` (no `skipif`, no fixture) builds
  a small hand-computable event stream plus marks inline — the idiom used by the literal trade-event list at
  tests/test_explorations.py:52-63 — and asserts that `build_trade_ledger` produces the right records, the
  right `initial_risk` per basis, the right MAE, and that `ledger_statistics` computes expectancy, the SQN,
  and `r_newey_west_t` correctly (the last against a hand-computed Bartlett-weighted value). Deterministic,
  no data, `pytest.approx` against hand-derived values, one invariant per method.
- **Dataset-gated regression class.** A `class TestTradeLedgerRegression:` gated by
  `@pytest.mark.skipif(not _HAVE_*, ...)` with a module- or class-scoped fixture that loads and runs each
  overlay once (the `_HAVE_*` probe at tests/test_vol_premium.py:49-60, the `scope='class'` fixture at
  :416-423, the run-once `data`/`result` fixtures at tests/test_cc_backtest.py:1345-1360). It pins
  expectancy, the SQN, `r_newey_west_t`, and the MAE-R distribution for one real overlay per engine — an
  MSFT covered call and a SPY short straddle. Skip-not-fail is the rule: real-chain tests skip when datasets are absent.

## Cross-surface and honesty

The R-basis convention is the honesty rail here: it is a declared, pinned choice, and the pin is what keeps
every downstream R-multiple honest. No canonical number is re-pinned — this is new measurement with new
tests, not a re-measurement of a registered result.

New public symbols must be added to the CLAUDE.md symbol-sweep regex **in the same change** that introduces
them, per the repo rule "keep the symbol-list regex in sync with the code's public surface." The Gap A
symbols to append are `TradeRecord`, `build_trade_ledger`, `ledger_statistics`, and the test classes
`TestTradeLedgerMechanics` / `TestTradeLedgerRegression` — none of these appear in the regex today. Insert
them before the closing `)` of the alternation in the "Every symbol name cited in prose" sweep block.

The README `## Project layout` table gets a `common/trade_ledger.py` row when the module lands. Plan docs
also get a `## Project layout` row (the `_plan` docs are all listed), so this design doc and the
still-open `docs/van_tharp_test_plan.md` row should be added there per the cross-surface rule for any new
public doc.

## Related

- [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the parent plan; Gap A is its keystone, and the
  undefined-risk R open question (its "Open questions" section) is resolved here.
- [docs/explorations.md](explorations.md) — the exploratory / kill-or-justify honesty pattern this ledger's
  eventual numbers ride under, unchanged.
