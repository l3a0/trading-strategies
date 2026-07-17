# Implementation plan: the put-credit-spread analysis code

**Status:** PLAN — no code written, no data touched. This document captures the
implementation design for the analysis-code PR that
[docs/prereg_put_credit_spread.md](prereg_put_credit_spread.md) §10 requires
before any number is produced. The registration became effective at merge
commit `4ddbbbe` (PR #133); this plan changes nothing in that document (an
edit there would be a §11 amendment) — it only decides *how* the frozen rules
become code.

**Date:** 2026-07-17.

---

## 0. The §10 traceability table

Every §10 requirement, mapped to the artifact that satisfies it:

| §10 requirement | Artifact |
|---|---|
| `walk_forward_structure` driver | `realchains/walk_forward_structure.py` (new) |
| Exit knobs through `params` | the driver's exit-variant dicts (engine reads them natively) |
| Arm-B spec-override switch at existing `hedge_mode='none'` | the driver's `_run_cell(..., hedged=)` seam (D1) |
| Pin protection, byte-identical suites | zero engine-file diff (D1) + full-suite run pre-PR |
| C1 drift alarm at campaign coordinates, −0.91 ± 0.02 | `run_prereg_put_spread.py` first step, fatal on miss (D6) |
| Seeds: ensemble 20260717; bootstrap fixed at commit | career `i` = 20260717 + i (D8); bootstrap 20260718 (D9) |
| Ordering: code committed before any run | this PR carries synthetic tests only; runs are post-merge |
| Compute: one-store budget, cached-index allowance | sequential SPY→IWM loads; per-window runs are cheap (§5) |

---

## 1. Modules

### 1.1 `realchains/walk_forward_structure.py` — the driver library

The `walk_forward_real.py` architecture (frozen constants + a pure library
function + no I/O), specialized to the registered pipeline:

- Constants: `ENTRY_LATTICE` (9 cells — `dte ∈ {21, 30, 45}` ×
  `short_delta ∈ {0.20, 0.25, 0.30}`, `wing_delta = short_delta − 0.05`
  derived), `EXIT_VARIANTS` (8 params dicts: hold, target50, target75,
  stop2x, stop3x, dte21, bracket, bracket75 — the last per prereg
  Amendment 1: `close_at_pct = 0.75` + `stop_loss_mult = 1.5`), the 69-cell
  valid joint enumeration in frozen lattice order (dte21-exit ×
  21-DTE-entry cells excluded),
  `TRAIN_YEARS = 4`, `TEST_MONTHS = ROLL_MONTHS = 6`, `MIN_TRADES = 30`,
  `CENTRAL_CELL` = (30 / 0.25Δ / 0.20Δ-wing, hold).
- `_run_cell(dates, prices, store, params, *, hedged)` — replicates
  `run_structure_via_spec`'s three lines (defaults-merge under params →
  `run_real_structure_overlay` with the credit-spread spec's `select`,
  `entry_guard='net_positive'`, `management='hold'` → the spec's summary
  reassembly), passing `hedge_mode='combined'` when hedged else the engine's
  existing `'none'`.
- `_dte21_guarded_select` — wraps `_legs_credit_spread`; on a pick, reads the
  short leg's actual DTE by matching its contract ID into
  `day['candidates'][…][0]`; returns None when ≤ 22, else the baseline's leg
  list unmodified (delegation-pure, the Gap F invariance pattern).
- `_excess(daily_eq, capital)` — `np.diff(equity)/capital` minus
  `rf_credit[1:]/capital` (the `short_vol_statistics` recipe; the array is
  not exposed by that function and `common/portfolio.py`'s extractor is
  private, so the driver owns its three-line copy, equivalence-tested).
- `walk_forward_structure(dates, prices, store, *, cells, train_years,
  test_months, roll_months, min_trades, forced_cell=None)` — the window loop
  copied from `walk_forward_real` (pandas `DateOffset`, half-open
  boundaries, the `len(train) < 30 or len(test) < 5` skip); per window: each
  cell runs hedged on the train slice, is disqualified below 30 entries
  (`num_credit_spreads_sold`), and is scored by the unrounded annualized
  Sharpe of `_excess`; strict `>` over frozen lattice order implements the
  registered tie-break. `forced_cell` implements arm C2 and the ablations
  through the identical machinery. A no-winner window trades nothing,
  contributes `len(test_days) − 1` zeros, reports `SKIPPED`, and counts in
  every denominator.
- Stitching per the frozen §5.5 rule: concatenated per-window excess arrays;
  the **seam charge** synthetically closes any structure open on a window's
  last day at bid/ask + per-leg commission, quotes located by matching
  `(expiration, strike, delta-sign)` into `candidates` on the last
  within-window day the leg is quoted; the **day-0 bound** (first entry's
  mid-vs-fill spread + commissions per window) is summed and returned for
  reporting.
- Arm E: `jittered_replay(...)` — careers replay arm A's realized per-window
  winners, an emission-keyed `random_entry_selector`-style wrapper jittering
  only the entry calendar (`k = 10`), selection never re-run.
- `stationary_bootstrap(x, block=21, B=10_000, seed)` and
  `loyo_nw(excess, dates)` — both new (the repo has neither; the trend
  gate's LOYO is cycle-level). They live here, not `common/stats.py`, so the
  pinned shared significance block is untouched.

### 1.2 `realchains/run_prereg_put_spread.py` — the registered runner

The `run_registered_vrp.py` template: §-by-§ docstring citing the prereg and
merge commit `4ddbbbe`; print-only (full dicts so the results tests can pin
every field); verdict booleans inline with §-references; run as
`python -m realchains.run_prereg_put_spread`. Order: C1 (fatal) → arm A →
B replay → C2 → ablations → arm E → bootstrap/LOYO → the §8 verdict block.
Data: SPY = `load_chain_store(calls, extra_paths=[puts],
start=REGISTERED_CLEAN_START['SPY'])` with the day grid clipped to call days
in `[2010-12-01, 2026-06-05]` (the merged union otherwise leaks 3 calls-only
days past the puts end); IWM = its single both-wings file (no puts merge
exists or is needed); stores load sequentially and are freed (`del store`)
between tickers.

### 1.3 `tests/test_walk_forward_structure.py` — always-run, synthetic only

`_two_leg_scenario`-style fixtures; **no dataset-gated class in this PR**
(result pins belong to the results PR; no real-data run may precede merge).
Classes: `TestPutSpreadLattice` (enumeration, wing derivation, 69-count,
order), `TestSpreadWfSelection` (metric, floor on entry count, tie-break,
no-winner branch), `TestSpreadWfStitching` (concat, zeros, hand-derived seam
charge, day-0 bound), `TestDte21Guard` (skip at ≤ 22, delegation
byte-identity), `TestHedgeOverrideEquivalence` (hedged path ==
`run_real_credit_spread_overlay` on summary/trades/equity; unhedged path ==
direct overlay at `hedge_mode='none'`), `TestSpreadJitterMechanics` (k=0
identity, seed determinism, per-window reset), `TestStationaryBootstrap`
(determinism, add-one p, block behavior), `TestLoyoStream` (hand-computed
year drop).

---

## 2. Frozen implementation decisions

- **D1 — driver-side hedge override, zero engine diff.** `run_structure_via_spec`
  hardcodes `spec['hedge_mode']`; rather than adding an engine kwarg, the
  driver binds the spec's knobs by hand and passes `hedge_mode` itself (the
  `random_entry_scout` precedent). §10's switch exists, the engine's existing
  `'none'` path is used, and byte-identity of every pinned suite is
  structural rather than argued. (Owner-overridable: a 3-line
  `vol_premium.py` kwarg is the alternative.)
- **D2 — the `dte21` guard matches contract IDs, not re-selects.** Actual DTE
  comes from the candidate tuple of the already-picked short leg, so accepted
  entries are field-for-field the baseline's.
- **D3 — `min_trades` on the entry count** (`num_credit_spreads_sold`), the
  `walk_forward_real` convention the prereg §5.2 names.
- **D4 — no-winner windows contribute `len(test_days) − 1` zeros** (each
  window's excess array is a diff, one shorter than its day count).
- **D5 — seam-charge quotes walk back** to the last within-window day the leg
  is quoted (the all-legs-quoted honesty rail applied to the seam; never a
  manufactured fill on an unquoted day).
- **D6 — C1 reuses `search.edge_search._load_ticker_data('SPY')`** so the
  campaign coordinates (live `CHAIN_CLEAN_START`, `STRUCTURE_END`, puts
  merge, call-day clip) are exact by construction; tolerance ± 0.02; miss
  aborts the entire run.
- **D7 — arm B replays, never re-selects.** The unhedged arm reuses arm A's
  recorded per-window winners; selection happens exactly once, hedged.
- **D8 — career seeds 20260717 + i** (i = 0..19), `k = 10` — the Gap F
  seed+i convention under the prereg's fixed base.
- **D9 — bootstrap seed 20260718**, block 21 trading days, B = 10,000,
  add-one p `(1 + #{i : mean_i ≤ 0}) / (1 + B)` — committed here per §10's
  "fixed in the analysis script at commit time."

---

## 3. Cross-surface obligations (same PR)

- CLAUDE.md symbol-sweep regex: add `walk_forward_structure`,
  `run_prereg_put_spread`, `stationary_bootstrap`, `loyo_nw`, and the eight
  new Test classes.
- README file table: rows for the two `realchains/` modules and the test
  file.
- `ci.yml`: add `tests/test_walk_forward_structure.py` to the
  `trend-explore` bucket (currently the lightest; re-balance if measured
  timing disagrees).
- Not in this PR: any edit to `docs/prereg_put_credit_spread.md` (§11
  amendment territory), any dataset-gated test, any real-data number.

---

## 4. Compute budget

An engine pass is \~0.05 s full-span (Gap F measurement), so 69 cells × 23
four-year train windows ≈ 1,590 passes lands well under two minutes of
engine time — the existing SPY walk-forward regression already runs \~700
passes in CI. Store loads dominate (\~20 s each); SPY then IWM, one at a
time, freed between. The §10 cached-per-cell-index allowance is unlikely to
be needed; if used, it changes no selection semantics.

---

## 5. Post-merge runbook (the results PR)

1. Full suite byte-identical on the merged analysis code.
2. `python -m realchains.run_prereg_put_spread` — once. C1 gates everything.
3. Results PR: `docs/put_credit_spread_results.md` with the §8 row verbatim,
   dataset-gated pin classes (`TestSpyPutSpreadWfRegression`-style, class
   docstrings carrying the verdict narrative), citing `4ddbbbe` and the
   analysis-code commit. Any value the registration deferred to the data is
   recorded as a §11 amendment, the `prereg_vol_premium` Amendment-1 pattern.
