# Exploration log — ideas that didn't survive

This is the repo's record of **dead ends**: cheap kill-gate scouts on strategy
ideas that were measured and rejected, kept so they aren't re-explored from
scratch.

**Read this first — what these are and aren't.** Most entries are
*exploratory scouts*; the last is a real-chain *robustness check* on a
published refinement. Neither is a registered experiment. Each runs on data
that has already been used, so it **spends the sample**: it can only *kill* an
idea or *justify* taking it to a pre-registration. It is never itself a
confirmatory verdict. The numbers are pinned (`tests/test_explorations.py`, or for
the real-chain check `tests/test_real_cc_backtest.py`) so a dead end stays settled
and so a future change can't silently revive a buried result — but pinning a
result does **not** promote it to a registered finding.
That line is the whole point of [the pre-registration discipline](prereg_trend_gate.md):
a result that conditions on outcome data it also generated cannot claim a
p-value. A scout that *passes* would earn a registration, not a headline.

The recurring lesson across the entry-conditioning scouts: **conditioning
call-selling entry on recent upward price action has the sign backwards on
these names.** The
damage a covered-call seller takes comes from the sharp *rebounds* out of
selloffs (2020-03, 2025-04) — moves the signal reacts to *after* they've
happened, not ones it anticipates. Every gate built on "the stock just went
up, so be cautious" therefore skips the wrong cycles.

---

## Post-rip cooldown — KILLED (2026-06-13)

**The idea.** After a "rip" that causes a deep-in-the-money buyback or a
loss-making assignment, suspend covered-call selling for N days, then resume.
The intuition: a rip means the stock is running, so sit out the continuation.

**How it was tested.** A two-part scout on the naked baseline runs
(`run_real_cc_overlay`, published params) pooled across MSFT / QQQ / SPY — 705
cycles, 243 rip triggers, on the clean canonical chains (`CHAIN_CLEAN_START`
era clip applied; SPY on the corrected 2010-05-17 boundary). Both parts pinned
in `TestCooldownScout`.

1. **Does the mechanism exist?** For each cooldown horizon N, tag every cycle
   as *post-rip* if it was entered within N days of a prior rip **on its own
   ticker** (per-ticker — a rip on one name can't cool down another), and
   compare the per-cycle P&L of post-rip cycles to the rest:
   `D_A = mean(post-rip) − mean(other)`. The hypothesis predicts `D_A < 0`
   (post-rip entries do worse). A seed-pinned trigger-placement permutation
   (1,000 draws) gives the null.

2. **Is there a length to set?** On the price series alone (no strategy P&L),
   measure the forward return after the actual rip dates versus the
   unconditional baseline, and the daily-return autocorrelation — the "memory"
   a cooldown would need to ride.

**The verdict — wrong-signed, and no memory to time it to.** `D_A` is
**positive at every horizon** — post-rip cycles *lose less*, the opposite of
the hypothesis — and the real arrangement sits in the high tail of the
permutation null, never the low tail a real effect needs:

| Cooldown N | D_A (per cycle) | permutation percentile |
| --- | --- | --- |
| 7d | +$57 | 0.56 |
| 30d | +$376 | 0.94 |
| 60d | +$623 | 0.95 |
| 90d | +$1,770 | 1.00 |

And there is no return memory to anchor a cooldown length to: forward returns
after a rip are **below** baseline at every horizon (−0.46pp at 21 trading
days widening to −1.06pp at 120), and the pooled daily-return lag-1
autocorrelation is **−0.126**. A rip is a weakly *mean-reverting* event, not a
momentum-igniting one — so the window after it is, if anything, a slightly
*safer* time to sell, and any nonzero N is pure abstinence.

**The trap this avoids.** Skipping post-rip cycles "improves" net P&L by
+$157K (N=7) rising to +$451K (N=180) — but only because the naked strategy
loses money, so skipping any growing slice of cycles helps regardless of
skill. Sweeping N against net P&L and picking the best would have "found" a
brilliant long cooldown that is nothing but *not trading*. The per-cycle
`D_A`, immune to that abstinence confound, is the honest statistic — and it
says no.

**How to determine the cooldown length** (the question that motivated the
scout): you don't search for it — you measure the memory of the rip in the
return series, and that measurement either hands you N or tells you none
exists. Here it told you none exists.

---

## IV-richness gate — KILLED (2026-06-11, pinned 2026-06-13)

**The idea.** Sell a call only when its implied volatility is *rich* relative
to recent realized volatility — the classic volatility-risk-premium play. If
options are systematically overpriced, gate entry to the rich days and harvest
the premium.

**How it was tested.** For each naked cycle, read the entry contract's vendor
implied volatility (the engine's loader discards the IV column as unreliable,
so the scout reads it directly, with a fail-closed IV < 0.05 floor for the
lattice/placeholder rows). Pooled across MSFT / QQQ / SPY — 694 cycles carry a
usable entry IV. Three measurements, all pinned in `TestIvRichnessScout`:

1. **Is there premium to harvest?** The ex-post VRP at the sold ~25-delta /
   30-day contract = entry IV minus the realized vol over the option's life.
2. **Does the signal predict P&L?** The rank-correlation (Spearman) of entry
   richness — entry IV minus trailing realized vol, the thing you'd gate on —
   against cycle P&L.
3. **Does a rich/not split separate outcomes?** A binary `D_A` on
   IV > trailing-RV, with a permutation null.

**The verdict — no premium to gate on.** The three measurements:

| Measurement | Value | Reading |
| --- | --- | --- |
| Ex-post VRP at the sold contract (median / mean) | −0.27% / −2.30% | ~0 — options not systematically overpriced |
| Spearman(entry richness, cycle P&L) | +0.03 | richness doesn't predict P&L |
| Binary IV>RV split, `D_A` | +$646/cycle (95th pct) | looks like signal — but see below |

The one positive-looking number is a **confound, not a premium.** "Rich"
entries (IV above trailing realized vol) cluster where trailing vol is *low* —
mean trailing RV 0.15 for rich entries vs 0.23 for the rest — i.e. in calm
markets, where covered calls do better regardless of any volatility premium.
With the ex-post VRP at \~0 and the rank-correlation at \~0, the binary split
is gating on the *vol level*, not harvesting a premium. There is no
volatility-risk-premium edge to condition on at these strikes on these names.

---

## Delta-hedged covered call on real chains — KILLED (2026-06-14)

**A different kind of dead end.** The scouts above kill *entry-timing* ideas.
This one kills a *P&L-reconstruction* refinement, and it's the only entry here
whose victim was a **published** number rather than a prospective hypothesis.
Blog post 6 had already flagged that the proxy's delta-hedged t-stats (MSFT
1.63, QQQ 1.58) "need re-measuring against real quotes before they mean
anything"; this entry is that measurement. It is pinned in
`tests/test_real_cc_backtest.py` (`TestMsftRealRiskManagedRegression`,
`TestQqqRealRiskManagedRegression`), not `tests/test_explorations.py`.

**The idea.** Delta-hedge the covered call (Israelov & Nielsen, 2015): each day
hold extra long stock equal to the short call's delta × base shares, pinning
the portfolio's net delta at the buy-and-hold level. This strips the
equity-*timing* swing the short call injects — variance that adds no return —
and should leave the pure volatility-risk premium the call sale harvests. On
the **synthetic** IV-proxy engine it was the strongest refinement the proxy
produced: it lifted the overlay's Newey-West t-stat from 0.46 → 1.63 on MSFT
and 0.10 → 1.58 on QQQ (MSFT's lift quoted in blog post 4, QQQ's in post 5;
pinned in `TestMsftRiskManagedRegression` / `TestQqqRiskManagedRegression`).

**How it was tested.** Re-run the identical hedge on **real option premiums** —
`run_real_cc_overlay(..., delta_hedge=1.0)` over each name's clean canonical
chain (2016-06 → 2026-06, bid/ask fills) — and re-measure the excess-return
Newey-West t-stat. A proxy twin runs on the same unadjusted price series, so
the real-vs-proxy gap isolates the only thing that changed: where the option
price came from.

**The verdict — the edge was a proxy artifact.** On real premiums the hedged
excess falls to noise of zero on both names; the proxy's t-stat does not
survive:

| Ticker | Proxy twin NW t (same series) | Real NW t — bid/ask | Real NW t — mid | Hedged net overlay (real) |
| --- | --- | --- | --- | --- |
| MSFT | +1.76 | −0.23 | +0.73 | −$82.4K |
| QQQ | +1.52 | +0.18 | +0.30 | −$14.9K |

The proxy was minting premiums richer than the market paid (\~1.6× on MSFT), so
the premium the hedge "isolated" lived only in simulation. The hedge still does
its mechanical job on real chains — identical trades, excess vol cut
(MSFT 6.64% → 4.80%, QQQ 5.30% → 3.06%), and it removes the naive run's
near-significant directional *harm* (MSFT −1.73 → −0.23, QQQ −1.78 → +0.18).
What it cannot do is conjure a premium that isn't there. The price is paid in
tail risk: max drawdown *rises* under the hedge (MSFT 41.00% → 44.34%,
QQQ 38.22% → 40.92%), because the extra stock sits on negative cash — a levered
long in selloffs.

**Same conclusion as the IV-richness scout, from the other side.** That scout
measured the *statistical* premium — ex-post IV minus realized vol at the sold
contract — and found ≈0. This builds the strategy that would *capture* that
premium and earns nothing significant. The statistical premium is absent and
the capturable premium is absent: no harvestable volatility-risk premium at
these strikes on these names, at real quotes.

---

## Related, recorded elsewhere

- **Trend gate** (suspend selling during a 200-day uptrend) — a *registered*
  experiment, killed at Stage 1 with the same wrong sign (`D_A = +$439`). Full
  results in [trend_gate_results.md](trend_gate_results.md); it is not a scout,
  so it lives there with its own pins, not here.
- **Delta-neutral / put-side VRP experiment** — the follow-up the delta-hedge
  entry above pointed to, now run (an unpinned scaffold). The clean delta-neutral
  isolator (net delta → 0, not buy-and-hold) confirms the covered-call \~0 was a
  *structure* artifact: at the same strike and span the signal flips positive
  (MSFT −0.23 → +0.87, QQQ +0.18 → +0.90; SPY 0.25Δ NW t +2.54). On SPY the premium
  is real and marginally significant and **survives that name's realistic
  transaction costs** (\~+2.0 at 1bp); the risk-free *financing* nets out and is
  not a drag — an earlier "+0.93 / doesn't beat T-bills" reading was a base
  mismatch (the helper subtracted rf on capital, not on the smaller cash base the
  engine credited), now corrected. It stays thin and single-wing: QQQ/MSFT are
  sub-t=2, the daily-close backtest hides the short-gamma tail, and it is the weak
  call wing — promising, not a confirmed edge. Corrected numbers and the rf-base
  fix are in [vol_premium.md](vol_premium.md); the put side — where the premium
  actually lives — is blocked on a put-inclusive data fetch.
