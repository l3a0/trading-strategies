# Registered put-credit-spread experiment — the results

**Registration:** [docs/prereg_put_credit_spread.md](prereg_put_credit_spread.md),
effective at merge commit `4ddbbbe` (PR #133), with Amendment 1 (the
`bracket75` exit variant; 69-cell lattice) recorded pre-computation at PR
#134. **Analysis code:** `realchains/walk_forward_structure.py` /
`realchains/run_prereg_put_spread.py`, merged at `dd8c428` (PR #135) before
any number existed, per the §10 ordering rule. **The run:** executed once,
2026-07-17 21:08 UTC, `python -m realchains.run_prereg_put_spread`, exit 0;
the C1 drift alarm reproduced the campaign's committed cell at **−0.91**
(bar: ±0.02) before any other number was read. No registered value was
deferred to the data, so no §11 amendment accompanies these results.

## The verdict (§8 row 4, published verbatim)

> Null. Joint entry-and-exit optimization does not rescue the
> put-credit-spread family on these chains; the campaign's one-cell kill
> generalizes to the optimized lattice.

Scope, carried on every reporting surface per §2.4: **no-GFC span;
daily-close exits; EOD stop-markets.** No result of this experiment supports
trading decisions; the repo's standard disclaimer applies.

## The numbers

The verdict statistic is the stitched out-of-sample daily hedged-excess
Newey-West t (one-sided, pass bar t > 2 strict) over 23 walk-forward windows,
2014-12-01 → 2026-05-29, per-window $100K restarts:

| Statistic | SPY (primary) | IWM (confirmation) |
| --- | --- | --- |
| Stitched OOS hedged-excess NW t (0.5 bp) | **−2.26** | **−2.51** |
| One-sided p | 0.988 | 0.994 |
| Annualized Sharpe of excess | −0.58 | −0.70 |
| Cost curve t (0 / 0.2 / 0.5 / 1 bp) | −2.14 / −2.19 / −2.26 / −2.38 | −2.40 / −2.44 / −2.51 / −2.62 |
| C2 fixed-defaults t (hedged) | −1.87 | −2.52 |
| OOS days n (NW lag) | 2,862 (8) | 2,861 (8) |

Both fail the bar — and not as near-misses: the optimized strategy is
**wrong-signed with conviction** on both indices. Selling walk-forward-tuned
put credit spreads *lost* to the cash-plus-delta replication at every cost
level including frictionless, the §2.3 mechanism clause fails on both prongs
(C2 t < 0; no modal `short_delta` — the winners split 8/8/7 across the three
deltas), and IWM independently replicates the negative.

## The seduction, decomposed (arm B and the binding clause)

The §9 power section predicted the modal outcome verbatim: "arm B's raw
curve positive and seductive, the verdict t ≤ 0, row 4 language published."
Arm B delivered exactly that. The unhedged retail replay earned **+$55,689
raw P&L at an 82.6% win rate** on SPY (+$55,141 at 80.2% on IWM), beating
cash — and decomposes, in the same breath the §8 binding clause requires:

| Component | SPY | IWM |
| --- | --- | --- |
| Interest on collateral | +$51,754 | +$51,800 |
| Delta P&L (the embedded long-SPY tilt) | +$7,197 | +$7,434 |
| Options residual (arm A, net of 0.5 bp) | **−$3,262** | **−$4,093** |
| Chained scoreboard | B +73.3% vs cash +66.8% vs B&H +253.0% | B +72.3% vs cash +66.9% vs B&H +138.2% |

The per-spread mean was +$438.50 across 127 SPY spreads (worst per-window
drawdown 3.02%) — the smooth, high-win-rate "income" profile the strategy is
sold on, produced entirely by T-bill interest plus a small equity tilt,
minus a fee paid to the options market.

## What optimization chose, and what it bought

- **Per-axis winners (SPY):** `dte` 45 in 15 of 23 windows; `short_delta`
  split 8/8/7 (no modal value — mechanism prong (b) fails); exits: plain
  `hold` won **19 of 23 windows**, `target75` 3, `stop2x` 1. On IWM: `hold`
  18 of 23, `stop3x` 3, `target75` 2.
- **Amendment 1's `bracket75` never won a window on either ticker.** The
  wide-target / tight-stop corner was searched and rejected by the data.
- **The exit-only ablation was the worst arm of the experiment** (t −3.49 on
  SPY vs −1.64 entry-only): exit rules alone, tuned at the fixed central
  entry, made things decisively worse — Experiment 4's "exit choice moves
  risk shape, not sign" holds at the family level.
- **Walk-forward efficiency was negative** (OOS Sharpe −0.58 vs mean winner
  in-sample +0.12 on SPY; −0.70 vs +0.07 on IWM): in-sample selection
  anti-predicted out-of-sample performance — the brutal end of the \~+0.13
  IS→OOS rank-correlation measurement the registration cited as its prior.
- Every window cleared the Pardo floor (`n_below_30` = 0 throughout;
  minimum grid entry count 32); no window was skipped, no cell failed.

## Robustness (§7.3 companions, all agreeing)

- **Entry-jitter ensemble (20 careers, seeds 20260717+i):** career t band
  −2.46 … +0.19, median −1.29; **0 of 20 careers clear t = 2**, and the
  verdict's own entry calendar sits *below* 18 of 20 skill-free calendars.
  There is no entry-timing story hiding in the null.
- **Stationary block bootstrap** (seed 20260718, block 21, B = 10,000):
  p = 0.990, agreeing with the parametric 0.988.
- **Leave-one-year-out:** t range −2.74 … −1.39 with **no verdict-flipping
  year** — the negative is distributed across the span, not an artifact of
  2020 or 2022.
- **Seam accounting:** total window-boundary seam charges $157.70 (SPY) /
  $420.50 (IWM); day-0 omission bound $265.00 / $702.80 — both orders of
  magnitude below the residual being measured, so the §5.5 accounting
  choices cannot have made the verdict.

## What this settles, and what it does not

Combined with the family's prior record — the registered naked put wing
(+0.09, null), the campaign's committed cell (−0.91), and the menu-walker
cell (+0.05) — this experiment closes the question it registered: **on
these chains, over this span, at retail-observable EOD granularity, there
is no put-credit-spread configuration in the committed 69-cell lattice that
earns anything beyond cash and delta, and honest optimization makes the
family look worse, not better.** It does not speak to: GFC-scale regimes
(absent from the sample), intraday exit execution (EOD stop-markets flatter
stops), single-name underlyings, or lattices outside the registered menu —
each of those is a new registration, not an extension of this one.

## Pins and lineage

- **Pinned by** `TestSpyPutSpreadWfRegression` / `TestIwmPutSpreadWfRegression`
  (`tests/test_walk_forward_structure.py`, dataset-gated): the verdict
  statistics, cost curves, axis/exit compositions, arm-B decompositions, C2
  both ways, and the SPY ablations — re-run through the identical pipeline.
  The always-run mechanics layer (34 tests) pins the machinery itself.
- **Run of record:** the §10-ordered chain — registration `4ddbbbe` →
  Amendment 1 (pre-computation) → analysis code `dd8c428` → one execution.
  The arm-E ensemble, bootstrap, and LOYO figures above are
  seed-deterministic (20260717+i / 20260718) and reproducible by re-running
  `python -m realchains.run_prereg_put_spread`.
- **Method lineage:** as registered — Pardo (walk-forward, the floors),
  Bakshi-Kapadia (the hedged-gain measure), Politis-Romano (the bootstrap),
  Newey-West/Andrews via `common/stats.py`, and the White reality-check
  discipline of judging only the pre-committed statistic.
