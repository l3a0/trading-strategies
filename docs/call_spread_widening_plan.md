# Widening 5 design: the bear call credit spread + its unconditional baselines

**Status:** DESIGN — a grammar widening on the exploratory campaign track
(human-signed per the standing widening rule), not a registration. Every cell
it runs is EXPLORATORY: sample-spending, kill-or-justify, charged to the
lifetime e-LOND stream, never a registered verdict. A strong survivor
escalates to manual pre-registration — never auto-promotes.

**Date:** 2026-07-18.

**Why now:** both trend-switched spread designs (put spreads in uptrends /
call spreads in downtrends, at either tilt) are blocked on the same missing
measurement — the **unconditional bear call credit spread baseline**. The
put half of the family is measured three ways (campaign −0.91, menu-walker
+0.05, registered walk-forward −2.26); the call half has never run. This
widening produces that baseline in the family's canonical form.

---

## 1. Priors, stated before any number (deliberately split)

**For (+1, the committed prediction):** the delta-hedged short **call wing**
is the repo's only surviving premium (SPY +2.54 gross / +2.25 at 0.5 bp,
QQQ +2.07), and the call spread's skew geometry is the favorable one: on the
equity call smile the far-OTM long wing should carry *lower* IV than the
nearer short leg — the structure sells rich and buys cheap
(predicted `short_rich`; §3 verifies before declaring). The wing's absolute
cost is small, so the spread should retain most of the wing premium while
capping the tail.

**Against:** the campaign is 0-for-56 with every credit-structure family
wrong-signed somewhere; a crude iron-condor triangulation
(`t_call ≈ √2·t_IC − t_put`, halves treated as independent — a rough read,
the halves share the hedge) scatters widely by ticker (MSFT ≈ +2.8,
GLD ≈ +1.4, QQQ ≈ +0.9, SPY ≈ −0.6, EEM ≈ −0.4, NVDA ≈ −2.0, XLE ≈ −2.2)
and averages ≈ 0; MSFT's naked call wing was negative single-name; and the
far-OTM wing's *relative* bid/ask is the friction hotspot. The committed
`predicted_sign` is **+1** on the mechanism claim, with this counter-prior
on the record.

---

## 2. The structure (frozen)

Bear call credit spread, single expiration: SHORT a `short_delta` call
(nearer the money) + LONG a `wing_delta` call at a strictly **higher**
strike (the defined-risk wing) — the CALL HALF of the iron condor, exactly
as `select_credit_spread` is its put half. Combined-delta-hedged at the
campaign convention (0.5 bp), hold-to-expiry, `net_positive` entry guard.
Net position delta is short (the mirror of the put spread's long): for the
0.25Δ/0.10Δ cell the `combined` hedge target is **+0.15 × shares of long
stock** — making this the grammar's first short-vega-AND-short-delta
overlay, the diagonal opposite of the put spread's short-vega-AND-long-delta.

- **Selector `select_call_credit_spread(day, target_dte, short_delta,
  wing_delta)`:** short leg via `select_entry`'s call band (`bid > 0`,
  `0.05 < δ < 0.60`), nearest-DTE then nearest-delta; wing = same
  expiration, strike strictly above the short's, buyable ask (`> 0`),
  nearest `|δ − wing_delta|` — byte-for-byte the `select_iron_condor` call
  half, factored rather than duplicated where practical.
- **Leg builder `_legs_call_credit_spread`:** short call fills at bid −
  commission, long wing at ask + commission (`fill='mid'` uses marks), the
  `_legs_credit_spread` mirror.
- **Spec:** `STRUCTURE_SPECS['call_credit_spread'] = {select, entry_guard:
  'net_positive', hedge_mode: 'combined', management: 'hold', defaults:
  {'hedge_cost_bps': 0.5}, summary}` + the one-line
  `run_real_call_credit_spread_overlay` delegate. **Zero engine diff** — the
  strangle/credit-spread widening shape; the Gap E knobs arm naturally
  (net-credit entry).

## 3. Grammar and signature (verify, then declare)

- `STRUCTURE_GRAMMAR['call_credit_spread']`: lattice `dte (21, 30, 45) ×
  short_delta (0.20, 0.25, 0.30) × wing_delta (0.05, 0.10)` — 18 templates,
  `grid_universe_size` **70 → 88**; family **CARRY** (theta-positive,
  defined-risk, net credit — the put spread's family, opposite wing).
- Declared signature `{expirations: 1, legs: 2, net_vega: 'short',
  net_delta: 'short', net_skew: 'short_rich' (predicted)}` — but the
  `net_skew` value is **written only after the engine verifies it**, on real
  SPY chains via the `TestGrammarSignatureMatchesEngine` addition. Widening
  3's lesson is binding here: the put spread's "obvious" `short_rich` was
  engine-corrected to `long_rich`; if SPY's call smile upticks at the far
  wing, this declaration changes to what the engine says, and §1's
  mechanism prior weakens accordingly — recorded either way.
- **Data:** calls only. Every canonical store carries the call wing, so —
  unlike widening 3 — no puts merge, no separate file, no lineage change
  beyond the 7 new cells' own rows. TLT stays sealed by omission.

## 4. The committed batch and its accounting

- **+7 cells:** `StructureTemplate('call_credit_spread',
  'call_credit_spread', (('dte', 30), ('short_delta', 0.25),
  ('wing_delta', 0.10)), +1)` across `STRUCTURE_SEARCH` — the exact mirror
  of the put spread's committed cell. Batch **8×7 = 56 → 9×7 = 63**.
- Scored by `structure_kill_gate` (NW t, asymptotic one-sided p), recorded
  via `run_structure_campaign --record`, judged by
  `judge_against_lifetime_stream` over the committed prior ledger (75 rows,
  0 survivors) — the appended batch never restarts the discount sequence.
- **Power honesty, computed now:** at stream position 76 the e-LOND flag
  bar is `t ≈ 6.35` (p ≈ 1.1e-10). **No realistic cell flags.** The
  widening's product is the recorded measurement — sign, magnitude, and the
  closed ledger row — not a plausible flag. Pre-committed escalation rule:
  any cell with t > +2 (right-signed, unflagged) is a candidate for
  *manual* registration in the trend-gate/VRP lineage; anything less joins
  the corpus as KILLED.

## 5. What the baselines unblock

The trend-switched designs (both tilts) require the unconditional call-side
baseline before any switching claim is testable; the §1 skew story gets its
first direct measurement; and the six-regime descriptive read (Gap D) gains
the call-side ledger for the bear-volatile cell watch. None of that
promotes anything by itself.

## 6. Exit-variant exploration (the Experiment 4 pattern)

Once the baseline exists, exits are tested the way Gap E tested the naked
short vol — an Experiment-4-style pre-committed variant grid on the pinned
SPY campaign cell, run once, EXPLORATORY (sample-spending, kill-or-justify;
nothing enters the idea ledger and no e-value is spent — the measurement
axes are risk-shape, not significance):

- **Variants, committed now:** stop 1.5× / 2× / 3× credit; target 50% / 75%;
  `exit_dte 21`; the two brackets (50%+2×, 75%+1.5×) — the Gap E seams arm
  naturally on the spread's net credit, no new engine code.
- **Axes:** `expectancy_r`, win rate, worst MAE-R, exit-reason counts (Gap A
  ledger) plus the Gap C+B intratrade ruin replay at f = 2% — the exact
  measurement set `TestSpyExitVariantExploration` pinned.
- **The question this structure makes new:** the spread's max loss is
  already capped at width − credit *structurally*. On the naked hedged book
  the 2× stop earned its keep by truncating an uncapped tail
  (−11.4R → −3.1R); here the cap pre-exists, so the committed question is
  whether stops add anything **on top of a structural cap** or just pay
  whipsaw for redundant protection. Prior, stated: exit choice moves risk
  shape, not sign (Experiment 4; re-confirmed at family scale by the
  registered exit-only ablation, −3.49).
- The Gap E honesty rails ride verbatim: daily-close stop-markets (flatter
  the stop), all-legs-quoted triggers (under-fire), one-day re-entry gap.
- If run, it pins as the three-surface exploration (scout code, a
  dataset-gated `Test…Exploration` class, an explorations.md entry). Any
  *verdict-grade* exit claim needs a registration in the put-side
  experiment's joint-lattice mold — not this widening.

## 7. Position-sizing exploration (the Gap C+B pattern)

The defined-risk structure finally gives the trade ledger's
`defined_max_loss` risk basis its natural use — the naked-book explorations
had to size on `premium_collected`:

- **R-definition:** R = width − credit per spread (the true worst case, not
  a proxy). Build the Gap A ledger off the baseline run's trades on that
  basis; R-multiples are honest by construction.
- **The sweep:** `sizing_sweep` at the Tharp-sourced fractions (0.25%, 0.5%,
  1%, 2%, 3%) with `mae_r` supplied (three-tier ruin accounting), plus
  `kelly_fraction` as the labeled reference-point-never-recommendation.
- **The contrast worth measuring:** on defined risk, fixed-fractional
  sizing bounds each trade's loss at f by construction — absorption is
  structurally impossible at these fractions, unlike the naked short vol
  whose pinned P(ruin) at f = 2% ran 0.83–0.99. The committed comparison:
  the same sweep, same seeds, defined-risk spread vs the naked book —
  measuring what the width cap is worth in ruin/drawdown terms at matched
  fractions.
- **Percent-volatility sizing (Tharp's model)** enters as the named Gap C+B
  widening it already is: replay the same trade sequence sizing each
  position so one unit of entry volatility risks a fixed equity slice, and
  report it beside fixed-fractional at matched average exposure. The IID
  marble-bag caveat (serial dependence destroyed, regime clustering lost)
  carries on every surface.
- Epistemic frame, non-negotiable: sizing reshapes distributions and can
  never flip expectancy sign (the Gap C+B Jensen lesson) — descriptive risk
  accounting, never an edge claim, never advice.

## 8. Mechanics tests (always-run, hand-derived)

`TestCallCreditSpreadMechanics` in `tests/test_vol_premium.py`, mirroring
`TestCreditSpreadMechanics`: the selector picks a strictly-higher-strike
buyable wing at the short's expiration (None when absent); fills short at
bid − c / long at ask + c; the `net_positive` guard rejects a net-debit
pick; settlement at call intrinsic `max(0, S − K)` per leg; the `combined`
hedge goes **long** stock (+0.15 × shares at 0.25/0.10 — the sign is the
test's point); a Gap E bracket arms on the net credit. Plus the
dataset-gated additions: the signature row in
`TestGrammarSignatureMatchesEngine` (with the `must_trade` guard), the
campaign re-pin (`TestStructureCampaign`, expected 0/63 unless the data
says otherwise), and `TestClosedGrammar` at 88.

## 9. Cross-surface checklist (same change)

Symbol regex: `select_call_credit_spread`,
`run_real_call_credit_spread_overlay`, `TestCallCreditSpreadMechanics`.
README: the `realchains/vol_premium.py` row gains the overlay; the
`tests/test_vol_premium.py` row gains the mechanics class.
`docs/edge_search.md`: the Widening 5 campaign-log entry with the 7 cells'
verdicts. Ledger: the 7 rows via `--record`. CI: no bucket change
(`test_vol_premium.py` and `test_edge_search.py` already run). CLAUDE.md's
widening narrative gains the 70→88 / 56→63 update.

## 10. Order of operations

1. Selector + leg builder + spec + delegate + grammar entry (predicted
   signature), mechanics tests green.
2. Engine signature verification on SPY; declare `net_skew` as measured.
3. `run_structure_campaign --record` — the 7 cells, lifetime-judged; re-pin
   `TestStructureCampaign` / `TestClosedGrammar`.
4. Campaign-log entry + cross-surface sweep; PR.
5. *(Optional, post-baseline, each pinned as a three-surface exploration if
   run:)* the §6 exit-variant grid on the pinned SPY cell; the §7 sizing
   sweep + defined-vs-naked ruin contrast + the percent-volatility
   comparison.

Out of scope, each needing its own design: the trend-switched composites
(blocked on this), any registered call-spread walk-forward or registered
exit verdict (the put side's registration is the template if the baseline
earns it), ATR-scaled stop variants (a Gap E widening of its own), and any
change to the put-side family's record.
