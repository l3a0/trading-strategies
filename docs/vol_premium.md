# Delta-neutral / put-side VRP experiment — design and scaffold

**Status: pinned & audited; rf-base accounting corrected.** The engine
credits rf on the cash collateral and charges the share-hedge half-spread. The
significance helper now nets rf out on the *same* base the engine earned it (the
recorded per-day cash credit) — fixing a benchmark bug that a flat rf-on-capital
subtraction had only half-addressed (see **Results** and **Lesson 2**). The
re-derived finding on the call wing: a **real, marginally-significant
delta-neutral volatility premium on SPY** (Newey-West t \~+2.5) that **survives
that name's realistic transaction costs**. The risk-free *financing* is not a
separate drag — that earlier reading was the base-mismatch artifact. It stays a
thin, single-index, call-wing signal whose backtested risk understates the
short-gamma tail — promising, not a confirmed edge — and the put-side phase (where
the literature locates the premium) is still blocked on a data fetch. The SPY
headline below is pinned (`TestSpyShortVolRegression`) and the accounting was
adversarially audited.

## Why this experiment exists

The repo's earlier VRP measurements used a covered call: a single 0.25-delta
**call**, mostly equity beta, "hedged" by pinning net delta to the
**buy-and-hold** level rather than to zero. On real MSFT/QQQ chains the captured
premium came back \~0 (Newey-West t −0.23 / +0.18). The reconciliation against
the literature ([exploration log](explorations.md), delta-hedge entry) concluded
that result is *consistent with* — not a contradiction of — research, because the
documented, robust VRP is a different object: a **whole-strip, delta-neutral,
put-heavy, index-level** premium. The covered call sampled its weakest corner.

This experiment builds the missing clean isolator and asks one question:

> **Does a properly delta-NEUTRAL short-vol position surface the volatility risk
> premium that the covered-call (hedged-to-buy-and-hold) construction did not?**

## The instrument

A daily **delta-neutral short option** — the Bakshi-Kapadia (2003) "delta-hedged
gains" construction. Sell an option, hold the offsetting stock so net delta \~ 0,
rebalance daily on the vendor delta. With direction removed, the residual P&L is
the gamma/vega P&L,

`≈ ½ · Γ · S² · (σ_implied² − σ_realized²)`

— the variance risk premium itself. A significantly **positive** mean P&L means
the seller was paid for bearing variance risk; \~0 means the premium isn't there
at these strikes/names/era.

Three deliberate differences from `run_real_cc_overlay` (the covered call), all
in `run_real_short_vol_overlay`:

- **No base long-stock leg.** Capital is collateral; the only stock held is the
  hedge. The covered call is \~93% equity beta; this is \~0%.
- **Net delta targets ZERO**, not the buy-and-hold level. This is the single
  change the reconciliation flagged as untested.
- **Default strike is ATM** (0.50 delta), where gamma/vega — and thus
  variance-premium signal — peak, and **hold-to-expiry** (no early profit-take or
  deep-ITM management, which would truncate the variance exposure being measured).
  Set `target_delta=0.25` / `close_at_pct=0.75` to reproduce the covered call's
  strike and exit for an apples-to-apples comparison of the hedge-target change
  alone.

## Two phases

### Phase A — the call leg (runs today)

`vol_premium.py::run_real_short_vol_overlay`, on the existing call-only datasets.
Pinned mechanics in `test_vol_premium.py` (synthetic: the hedge offsets
direction, a flat market harvests the premium, the NW helper signs correctly);
a dataset-gated structural-invariant check on real SPY.

Runs today across SPY/QQQ/MSFT at ATM and 0.25-delta strikes (rf credited, hedge
cost charged). The re-derived finding: a real, marginally-significant
delta-neutral premium (SPY 0.25Δ NW t +2.54) that survives SPY's realistic
transaction costs; the risk-free financing nets out (the earlier +0.93 "doesn't
beat T-bills" was a base-mismatch artifact, now fixed). Full numbers and the
cross-section are in **Results** below.

### Phase B — the put side (run) and the ATM straddle (pending)

The equity-index VRP is concentrated in **OTM puts** (the skew / crash-insurance
premium; Constantinides-Jackwerth-Savov find index *call* alphas \~0 while put
alphas stay large). The call leg is the weakest wing to harvest. Testing the put
side is the point of the experiment. At the time of the scaffold it was **blocked**:
`download_option_dailies.py` fetched calls only. It has since been pre-registered
(`prereg_vol_premium.md`), the put-inclusive SPY and IWM data fetched, and the run
completed — the null verdict is in *The registered put-side run* below, and the
fetch plan that follows is the one that was executed. The ATM straddle (a two-leg
engine extension) is the one piece still pending.

The fetch plan (premium Alpha Vantage `HISTORICAL_OPTIONS`, which returns both
wings per day):

1. Extend the fetcher to keep put rows (drop the call-only filter; `infer_spot`'s
   strike band must accept negative deltas).
2. Re-run the data lifecycle for SPY first (sequential, one ticker to completion):
   validation battery, `gzip -9`, sha256, release upload, CI cache glob, checksum
   round-trip, cold-storage copy — per the Option-Chain Data Pipeline rules.
3. Add `select_put_entry` (nearest negative target delta) and a straddle mode to
   the engine; the delta-neutral loop is unchanged (it already hedges on signed
   vendor delta).

## Results — what running the corrected scaffold shows

The engine credits rf on the cash collateral and charges the share-hedge
half-spread (commission-free shares, per Schwab). The significance helper,
`short_vol_statistics`, had a benchmark bug: it subtracted a flat risk-free rate
on the *deployed capital* ($100K), but the engine only ever credits rf on the
*cash* balance — which the hedge holds far below capital (mean \~$68K on the SPY
run, and negative on the days the hedge drains it). Removing rf on a base larger
than the cash that earned it strips out interest the account never saw, and that
artifact crushed — and at the extreme flipped — a genuinely positive signal. The
fix records the engine's *actual* per-day rf credit and nets *that*, so rf cancels
on the base it was earned on and the verdict is rate-invariant (the same whether
the engine charges rf=0 or rf=4.5%). The numbers below are the re-derived ones.

### The bug, in one line

On real SPY 0.25Δ (2010–2026, hold-to-expiry, frictionless) the *same* equity
curve scores wildly differently depending only on what the helper subtracts:

| Benchmark the helper subtracts | NW t | what it is |
| --- | --- | --- |
| Actual per-day rf credit (cash, \~$9.36/day) | **+2.54** | the fix — rf cancels exactly |
| Flat rf on capital ($100K → $17.86/day) | +0.24 | over-removes — the prior half-fix |
| Flat rf on grown equity (\~$129K) | −1.41 | over-removes more — the original bug |

The engine credits rf identically in all three; they differ only in the helper's
subtraction. Three accounting choices were on the table: (a) test the raw vol-P&L
with rf credited and debited on the same base, (b) net the *actual* per-day credit,
(c) accrue engine rf on the full equity so a flat subtraction becomes correct. We
took **(b)**, which subsumes (a): netting the actual credit makes the excess
identical to the rf=0 run's raw vol-P&L. It needs no change to the engine's
economics — you cannot earn rf on hedge *stock*, which rules out (c) — and it makes
the excess sum to `alpha_vs_cash` up to the day-0 entry-spread mark (`eq[0]` is
already struck at the entry bid/ask mid before the first daily diff, so the summed
series omits that one day-0 cost — a near-exact conservation that, by dropping a
cost, slightly flatters the premium rather than deflating it).

### The result on SPY (call wing)

Two questions, not three — the old "beat T-bills on the full $100K" row was the
artifact (it charged a financing penalty on the hedge sleeve, the very base
mismatch the fix removes):

| Question (SPY 0.25Δ, 2010–2026) | Result | Verdict |
| --- | --- | --- |
| Is the delta-neutral vol premium positive? (rf netted) | NW **t +2.54**, +$36.5K vol-P&L | yes — real, marginally significant |
| Does it survive a realistic share-hedge cost? | +2.25 @0.5bp, +1.97 @1bp, +1.39 @2bp, −0.35 @5bp | yes at SPY's \~0.1–1bp; no by \~4–5bp |

The premium clears t=2 gross and, at SPY's penny-wide share spread (\~0.1–1bp),
stays \~+2.0 to +2.5 (the same on the matched 2016–2026 window: +2.02 @1bp). It
does not survive a 4–5bp hedge cost, but SPY shares do not trade that wide. The
risk-free financing is **not** a drag once rf is netted on the right base — the
verdict is identical whether the engine charges rf=0 or rf=4.5%.

### The cross-section (rf-netted vol-P&L t-stats)

| Underlying | Strike | Span | NW t |
| --- | --- | --- | --- |
| SPY (index) | 0.25Δ | 2010–2026 | +2.54 |
| SPY | ATM | 2010–2026 | +2.03 |
| SPY | 0.25Δ | 2016–2026 | +2.51 |
| QQQ (index) | ATM | 2016–2026 | +1.23 |
| QQQ | 0.25Δ | 2016–2026 | +0.90 |
| MSFT (single) | ATM | 2016–2026 | +0.87 |
| MSFT | 0.25Δ | 2016–2026 | +0.87 |

S&P > Nasdaq > single name — the literature's ordering (broad-index VRP is a
correlation risk premium; single-name variance is barely priced). Only SPY clears
t=2; QQQ and MSFT are positive but sub-significant. SPY's lead is not a span
artifact — on the matched 2016–2026 window SPY 0.25Δ is +2.51, essentially its
full-span +2.54. (SPY's canonical chain store reaches back to 2010; the MSFT/QQQ
canonical stores start 2016.)

### Against buy-and-hold

A delta-neutral carry is not a substitute for owning the index, but the contrast
shows what kind of object it is (SPY, 2010–2026, $100K; buy-and-hold is price-only,
so it omits \~1.4%/yr of dividends and is slightly *understated*):

| | Buy & hold | Short-vol 0.25Δ (0bp) | Short-vol (1bp) |
| --- | --- | --- | --- |
| Net P&L on $100K | **+$509,495** | +$73,000 | +$62,111 |
| of which rf interest | — | +$36,505 | +$33,914 |
| **Vol-P&L (rf netted)** | — | **+$36,495** | +$28,196 |
| Ann return (total) | 13.1% | 4.7% | 4.0% |
| Annual volatility | 17.1% | 4.5% | 4.5% |
| Max drawdown (daily close) | −34.1% | −4.1% | −4.5% |
| Sharpe | 0.51 (excess/rf) | 0.52 (vol-P&L) | 0.40 |
| NW t | — | +2.54 | +1.97 |
| Correlation to SPY | 1.00 | +0.21 | +0.21 |

On risk-adjusted terms the two are in the same ballpark (Sharpe \~0.5), but they
are different animals: buy-and-hold is full equity beta (17% vol, −34% drawdown);
the short-vol leg is a near-market-neutral carry (4.5% vol, +0.21 correlation). Two
caveats keep the \~0.5 Sharpe from being a buy signal. First, the daily-close max
drawdown (−4%) badly understates a short-gamma book's true left tail — the
intraday/overnight spike that vaporized short-vol products in Feb 2018 is invisible
to a once-a-day hedge, and the +0.21 SPY correlation means the losses cluster
*with* market crashes. Second, this is the weak call wing, one index, one decade.
Buy-and-hold also wins outright on absolute return (\~7×), which matters if you can
hold through the drawdown.

### Lesson 1 — the hedge target is the driver, not the strike

At the *same* 0.25-delta strike and span (2016–2026), switching the hedge target
from buy-and-hold (the covered call) to net-zero (this engine) moves the signal
from MSFT −0.23 → +0.87 and QQQ +0.18 → +0.90. The covered call's \~0 was a
*structure* artifact — equity beta plus the buy-and-hold hedge swamping a thin
short-vol sliver. What the net-zero hedge surfaces is positive but still thin
(sub-t=2 on the single names; only SPY clears the bar), so the lesson is about
*where the signal was hiding*, not that the single names are now tradeable.

### Lesson 2 (methodological) — net rf on the base the engine credited

Credit rf and debit it on the *same* base. The engine credits rf on **cash**, and
the hedge keeps cash well below the deployed capital, so two tempting shortcuts
both bias the verdict. Subtract nothing (test the rf>0 run's raw return against
zero) and you count T-bill interest as if it were vol premium. Subtract a flat rf
on the *capital* — or worse, the *grown equity* — and you remove more interest than
the account ever earned, crushing or flipping the signal (the +2.54 → +0.24 → −1.41
ladder above). The fix records the engine's actual per-day credit and nets exactly
that. Guarded by `test_excess_nets_actual_rf_with_open_position` (the engine path,
rf>0 with a position open) and `test_flat_rf_fallback_columnless_curve` (the
synthetic fallback).

### Where this leaves the experiment

On the call wing, SPY shows a real, marginally-significant delta-neutral premium
(+2.54, +$36.5K vol-P&L over the decade) that survives its own realistic
transaction costs; QQQ and MSFT are positive but sub-significant. The risk-free
financing is not a drag (that was the accounting artifact), so the call-wing
hypothesis is **not** killed the way the pre-fix doc claimed — it is a thin,
promising carry. What keeps it from being a confirmed edge: the daily-close
backtest cannot see the short-gamma crash tail (the +0.21 SPY correlation says it
bleeds when the index falls hard), it is the weak call wing, and it rests on one
index over one decade. The live question was the richer **put side**, which needed
the blocked put-data fetch — and because this phase had already peeked at SPY, that
wing was pre-registered before the data was spent (`prereg_vol_premium.md`). That
registered run is now done; its verdict is next.

## The registered put-side run — the verdict

The put side was the point of the experiment: the equity-index premium is supposed
to live in the OTM puts, not the call wing the covered-call work had measured. It was
pre-registered before the put data existed — registration effective at PR #23's merge
— precisely because the call phase had already seen the SPY price path. The run is now
complete (`run_registered_vrp.py`), and the pre-committed outcome language (§6, row 4)
is published verbatim:

> **Null on the put wing: no significant delta-hedged premium on SPY over this span,
> even gross.** (IWM reported beside.)

The mechanism clause (§1.3) is not met either: the SPY put t does not reach the +2.54
call wing, so the put wing is no stronger than the call — the reverse of the
skew-premium prediction.

| Short put, −0.25Δ, 30 DTE | gross | 0.2 bp | **0.5 bp** | 1 bp | call wing (0.5 bp) |
| --- | --- | --- | --- | --- | --- |
| SPY Newey-West t | +0.20 | +0.16 | **+0.09** | −0.02 | +2.25 |
| IWM Newey-West t | +1.00 | +0.96 | **+0.91** | +0.81 | — |

SPY's headline verdict — the 0.5 bp net-of-cost t — is **+0.09**, and even the
frictionless gross t is **+0.20**. The premium is absent before transaction costs get
a vote. IWM, the naive out-of-sample index, harvests more (+$25.1K gross vol-P&L vs
SPY's +$5.2K) but its t is still only +1.00; it does not confirm. By the §5
conjunction — both indices must clear t = 2 — the finding is **not confirmed**.

**Why the put wing came in weaker, not stronger.** A short OTM put is short gamma on
the downside, and equity down-moves are the violent, vol-expanding ones. The book
collects premium in calm years and gives it back in the vol events the span contains:
on SPY, +$9.9K in 2021 and +$5.2K in 2016 against −$8.1K in 2018, −$6.7K in 2022, and
−$6.4K in 2025 — netting near zero. Its 13.3% max drawdown, against the call wing's
4.1% on the identical span, is that same skew tail. The call wing kept its +2.54
because its gamma is on the upside, where moves are smaller and there is no "crash up."

**The registration anticipated this.** The house prior (§1.5) was explicitly split:
the skew literature predicts a *larger* put premium, but Dew-Becker & Giglio (2025)
find the post-2010 tradeable index premium \~0 net of costs, and the call wing had
only barely survived. The null is the post-2010 result — now shown on the put wing
too, the wing the premium was supposed to favor.

**Verified before pinning.** Because the put path is newer code and a sign error could
fake a plausible null, the result was checked by five independent adversarial lenses
(hedge sign, rf-netting, entry/settlement, economic coherence, and a catch-all), each
tasked with *refuting* it. None could. The run is rate-invariant (identical t at rf = 0
and 4.5%), delta-neutral (the daily vol-P&L correlates −0.06 with SPY, the signature of
a neutralized book — a flipped hedge would show \~+0.5), and reproduces the +2.54 call
pin on the same shared engine, confirming no regression. The result is pinned by
`TestSpyShortPutRegression` and `TestIwmShortPutRegression`; IWM's clean span
(2010-12-01 → 2026-06-05, set by the validation battery, clean from row one) is recorded
as the §10 amendment to the registration.

## Remaining limitations

The hedge cost is modeled (commission-free shares, half-spread) and the rf-base
accounting is corrected (Lesson 2). What still bounds the result:

1. **Daily-close hedging understates the tail.** The hedge rebalances once a day at
   the close, so the backtest cannot see the intraday/overnight spikes that
   vaporized short-vol products in Feb 2018. A real short-gamma book's left tail is
   fatter than the \~4% max drawdown here, and its market correlation (+0.21, not
   \~0) means it loses when SPY falls hard. This is the main reason the \~0.5
   backtested Sharpe should not be read as a clean edge.
2. **The put side is now tested — and null.** The registered short-put run (above)
   found no significant delta-hedged premium on SPY (gross t +0.20) or IWM (+1.00);
   the wing the premium was supposed to favor did not deliver. What remains untested
   is the ATM straddle, a §7 secondary.
3. **The verdict nets financing out by design.** The vol-P&L answers "is the
   variance premium there," not "what does a *levered* book net after margin." The
   engine charges rf on the cash the hedge drives negative, but a real book pays
   margin *above* rf; that cost is real but separate from the VRP-existence question
   measured here.
4. **One index, \~10 years.** A short, single-asset sample on a skewed payoff has
   wide error bars; the +2.54 rests on one decade of one underlying.

## Epistemic status

**Exploratory, sample-spending** (SPY/MSFT/QQQ are already used). It answered its
question: a clean delta-neutral position surfaces a real, marginally-significant
gross premium on SPY that survives that name's realistic transaction costs (with
financing netted out), positive on the call wing and strongest on the broad index,
thin and sub-significant on the single names. That is a promising but unproven,
single-wing, single-decade signal whose backtested risk understates the short-gamma
tail — **not** a confirmatory verdict and not a tradeable edge yet. The follow-up —
the put side — was **pre-registered** (`prereg_vol_premium.md`, the way
`prereg_trend_gate.md` does) and run: a **null** on both SPY and IWM (above), the
informative post-2010 result. The ATM straddle remains the one unspent secondary.
See the [exploration log](explorations.md) for the prior covered-call dead end this
builds on.
