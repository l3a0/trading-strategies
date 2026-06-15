# Pre-registration: put-side volatility-risk-premium experiment

**Status:** DRAFT — not yet registered. Registration becomes effective at the
merge commit of this file to `main`. No put-side data may be fetched, and no
put-side run may be executed, before that commit exists; the engine changes in
§9 must be committed before any put number is produced.

**Date drafted:** 2026-06-15.

**Question of record:** On the broad equity index, does a daily delta-neutral
short *put* — the wing the literature says carries the equity volatility premium —
harvest a delta-hedged gain that is (a) significant and (b) survives realistic
transaction costs, and is it at least as strong as the already-pinned *call*-wing
result?

---

## 0. Reader's guide — why this document exists

This registration doubles as a teaching example, so it is more annotated than a
minimal spec. The operative rules an implementer must follow live in §2–§5 and
§9; the surrounding sentences explain why those rules are what they are.

**Why pre-register at all.** The call-wing phase of this experiment already ran
on SPY and is pinned (`TestSpyShortVolRegression`: a rate-invariant Bakshi-Kapadia
delta-hedged gain of Newey-West **t +2.54**, surviving SPY's realistic costs).
That means the SPY price path is no longer naive — I have seen it. Without a
commitment made *before* the put data exists, every choice left open (which delta,
which span, which cost band, put vs. straddle) is a fork that could be settled
after seeing the number, silently inflating the false-positive rate. Writing every
choice down first — and letting the git history prove the ordering — is what makes
the eventual t-statistic mean what it claims. It is the same discipline this repo
applies to code through pinned regression tests: commitments first, then evidence.

**Why the put wing.** The covered-call line of work measured the *call* wing and
found it the weakest slice of the premium (Constantinides-Jackwerth-Savov: index
*call* alphas \~0 under jump factors; the premium and the steep index skew live in
OTM *puts* — Bondarenko, Broadie-Chernov-Johannes). Selling a 25-delta call,
delta-hedged, still cleared the bar on SPY; the directional prediction here is that
selling a 25-delta *put*, the crash-insurance wing, clears it by **more**. This is
the one test that could turn "a real but thin call-wing premium" into a robust
index VRP — or fail to, which would itself be the informative post-2010 result.

**Why the Newey-West t is the verdict, not a placebo null.** Unlike the trend-gate
registration (`prereg_trend_gate.md`), this statistic has a standard inferential
framework. The verdict is the mean of the daily delta-hedged-gain series tested
against zero; consecutive 30-day cycles overlap, so the dependence is absorbed by
the Newey-West HAC correction (Bartlett weights, the Andrews lag `short_vol_statistics`
already uses) — the same correction every other significance number in this repo
uses, and the one the call pin uses. A stationary block bootstrap is added as a
reported robustness check (§7), not the verdict.

**The honest limitation, and the fix that is part of this registration.** SPY's
price path was seen on the call side, so the SPY primary is a *new-instrument* test
on a *seen* underlying — the put option prices and the put-skew hypothesis are new,
the daily returns of the stock are not. The fix is built in: an out-of-sample
replication on **IWM** (Russell 2000), an index this project has never run, so its
underlying is genuinely naive. IWM is fetched and run alongside SPY under the
identical instrument and cost rule (§3.1, §5.2); the finding is "confirmed" only if
**both** clear the bar (§6). SPY decides the primary hypothesis; IWM decides whether
a SPY pass is confirmed out-of-sample.

---

## 1. Hypothesis

### 1.1 The measure that makes it precise

The statistic is the engine's existing rate-invariant delta-hedged gain: build the
daily equity series of `run_real_short_vol_overlay` with the §2 short-put
instrument, and pass it to `short_vol_statistics`, which nets the engine's *actual*
per-day risk-free credit on the cash base it was earned on (the audited rf-base
fix) and returns the Newey-West t-statistic of the resulting vol-P&L. This measures
the Bakshi-Kapadia (2003) delta-hedged gain — "was the seller paid for bearing
variance risk?" — **not** "did the \$100K beat T-bills"; the latter charges a
financing penalty on the hedge sleeve that the rf-base fix deliberately removes
(see `docs/vol_premium.md`, the audit). The verdict is computed *net of the §3.3
cost band*.

### 1.2 Registered hypothesis (H1, primary)

On SPY over the §3.1 span, the daily delta-neutral short-put instrument (§2),
net of the §3.3 costs, has a delta-hedged-gain Newey-West t-statistic **> 2**.
The gross (frictionless) t must also be > 2. Both are required — a gross signal
that dies at realistic cost is a "premium that isn't tradeable," reported as such.

### 1.3 Mechanism claim (additional, declared now)

H1's *mechanism* reading — "the equity premium is concentrated on the put wing" —
additionally requires the SPY short-put net-of-cost t to **equal or exceed the
pinned call-wing +2.54**. A put result that passes H1 but lands *below* the call
wing passes the significance bar without confirming the skew mechanism, and is
reported as "the put wing harvests a premium, but not a larger one than the call
wing on this sample" — nothing more.

### 1.4 What is explicitly NOT claimed or tested

- **No "beats T-bills" claim.** The verdict is the delta-hedged gain (the
  variance premium), not a capital-efficiency or financing-charged return; the
  hedge's financing drag is a separate lens (`docs/vol_premium.md`), not this test.
- **No GFC-scale claim.** The clean SPY chain span starts 2010-12-01
  (`CHAIN_CLEAN_START`); 2008–09 is excluded. The span contains the 2020 COVID
  crash and the 2018/2025 vol spikes, but nothing like the GFC; claims are scoped
  to the regimes the sample contains.
- **No tail-risk claim.** A daily-close-hedged short-gamma book understates its
  intraday/overnight tail (Feb-2018 vaporized short-vol products in a session);
  the experiment measures the harvested premium, not the survivability of the
  position through an un-hedged gap.

### 1.5 House prior

Two forces pull opposite ways, which is why the test is worth running. The
literature's skew premium predicts the put wing's *gross* gain exceeds the call's
(+2.54) — argues for PASS and for the §1.3 mechanism. But Dew-Becker & Giglio
(2025) find the post-2010 *tradeable* index premium \~0 net of costs, and the call
wing already only barely survived (marginal at a 1bp hedge spread) — argues that
the put gain, gross-larger but on a wider-spread wing, may still fail H1 net of
cost. The registered prior is genuinely split: gross-larger likely, net-survival
uncertain. The experiment is run to settle it either way.

---

## 2. Instrument definition (fixed before any outcome is viewed)

### 2.1 Definition of record

The primary instrument is a **daily delta-neutral short put at target delta
−0.25**, 30 calendar-day DTE, hold-to-expiry, on real SPY option chains:

- **Entry:** `select_put_entry(day, 30, -0.25)` — among put candidates with
  `bid > 0` and `-0.60 < delta < -0.05`, the nearest-DTE expiration, then the put
  nearest `|delta − (−0.25)|`. (The mirror of `select_entry`'s call band.)
- **Fill:** the put is sold at the **bid** (`bid_ask`), the published convention;
  this models the option bid/ask directly.
- **Hedge:** a short put has position delta `−put_delta ≈ +0.25`, so net delta is
  driven to \~0 by holding `round(put_delta × shares)` **short** stock,
  rebalanced daily at the close on the signed vendor delta. (For the call wing the
  hedge was long stock; the put wing's is short — the only sign change.)
- **Close:** hold to expiration (`close_at_pct = None`, `manage_deep_itm = False`),
  then settle and re-enter — identical cadence to the pinned call run.

### 2.2 Why this definition

- **−0.25 delta, fixed:** it is the exact mirror of the pinned call wing (0.25Δ),
  so the put-vs-call comparison (§1.3) is like-for-like — same moneyness, same DTE,
  same engine, only the wing flipped.
- **Short put, not straddle, as primary:** one leg, one hedge sign, and a *direct*
  wing comparison. The ATM straddle (the canonical Coval-Shumway / AQR variance
  harvester) is a pre-registered secondary (§7), not the primary verdict.
- **Hold-to-expiry:** early profit-taking truncates the variance exposure the
  position exists to measure; it matches the pinned call convention.

### 2.3 Hard constraints

- **One delta.** Only −0.25 decides the verdict. Other deltas (−0.10, −0.15,
  −0.40) are exploratory (§7) and can never be promoted to primary within this
  registration; a wing swept for its best delta after seeing results is a fork no
  t-statistic could detect.
- **One instrument.** The short put decides H1. The straddle is reported, never
  promoted.
- **One span, one cost band.** §3.1 and §3.3, fixed below. No post-hoc span trim
  or cost retune.
- **No re-tuning the engine.** The delta-neutral loop, the rf-netting measure, and
  the Newey-West lag rule are frozen at their pinned (`TestSpyShortVolRegression`)
  form; the only new code is the put-entry selector and the signed hedge (§9).

---

## 3. Data, span, and configuration

### 3.1 Analysis span

SPY, **2010-12-01 → 2026-06-05** — the `CHAIN_CLEAN_START['SPY']` clean-chain
span, identical to the pinned call-wing run, so put-vs-call is on the same dates.
The put data is fetched from the same `HISTORICAL_OPTIONS` responses that already
supply the calls (the endpoint returns both wings per day; the current store is
calls-only because `download_option_dailies.py` filters puts out at fetch time —
§9). The fetched put file is git-tracked / release-pinned at the results commit; a
data refresh after registration is an amendment (§10).

**IWM (Russell 2000 ETF)** is fetched fresh for this experiment — never run by this
project, so its underlying is genuinely naive. Its analysis span is the **clean
span of the fetched IWM chains**, set by the same validation battery and
era-clip rule (`CHAIN_CLEAN_START`-style: the first trading day past the last
placeholder-greeks row) that defined SPY's 2010-12-01 boundary — determined by the
data, **not** chosen after seeing a result. The exact IWM span and cycle count are
unknown at registration and are reported with the results, recorded against this
registration as an amendment (§10). IWM's lower price and wider option spreads are
absorbed automatically: the option leg is sold at its own bid (§3.3), and the share
hedge uses the same committed half-spread.

### 3.2 Engine configuration (fixed)

- Engine: `run_real_short_vol_overlay` with the §2 short-put parameters,
  `capital = $100,000`, `dte = 30`, `risk_free_rate = 0.045` (the measure is
  rate-invariant, so this value does not affect the verdict — it is fixed only for
  reproducibility).
- Significance: `short_vol_statistics` unchanged — the rf-credit netting and the
  Andrews Newey-West lag, exactly as pinned.

### 3.3 Cost band (committed now, before any put number)

The verdict is computed **net of**:

- **Share hedge:** a **0.5 bp half-spread** of the share notional traded per daily
  rebalance (`hedge_cost_bps = 0.5`), commission-free per Schwab. SPY's actual
  penny half-spread is \~0.1–0.2 bp, so 0.5 bp is a conservative-realistic level —
  the same level at which the call wing scored +2.25. **0.5 bp is the headline
  cost.** A 0.2 bp run (SPY's literal spread) and a 1 bp run (conservative) are
  reported beside it as the cost curve, but 0.5 bp decides H1.
- **Option entry:** sold at the bid (modelled directly by the `bid_ask` fill).
- **Option commission:** `COMMISSION_PER_SHARE`, as in the engine.

Fixing the cost band before the number forecloses the most tempting fork — reading
off whichever spread assumption clears t = 2.

---

## 4. The verdict statistic

`t_put = short_vol_statistics(...)['t_stat_newey_west']` on the §2 SPY short-put
run at `hedge_cost_bps = 0.5`. Reported alongside, ex ante: the gross (0 bp) t, the
0.2 bp and 1 bp t, the Sharpe, the vol-P&L dollars, the Newey-West lag, and the
pinned call-wing comparison (+2.54 gross; +2.42 / +2.25 / +1.97 at 0.2 / 0.5 / 1 bp).

---

## 5. Pass rule (pre-committed, one-sided)

### 5.1 Primary hypothesis (SPY)

H1 **passes** iff both:

1. the net-of-cost (0.5 bp) SPY `t_put > 2`, **and**
2. the gross (0 bp) SPY `t_put > 2`.

The **mechanism** clause (§1.3) additionally requires the SPY `t_put(0.5 bp) ≥ 2.54`
(the pinned call-wing gross t) — i.e. the put wing is at least as strong as the call
wing. A pass of (1)–(2) without the mechanism clause is reported as "a significant,
cost-surviving put-wing premium, not larger than the call wing."

A point estimate of exactly 2.00 fails (strict `>`). No re-running at a different
delta, span, or cost band to recover a near-miss; a near-miss is the §6 "consistent
with, not evidence for" outcome.

### 5.2 Out-of-sample confirmation (IWM)

IWM is run under the **identical** §2 instrument and §3.3 cost band over its §3.1
clean span. IWM **confirms** iff its net-of-cost (0.5 bp) `t_put > 2` **and** its
gross `t_put > 2` — the same bar as SPY.

The finding is **CONFIRMED** iff SPY passes §5.1 **and** IWM confirms §5.2. This is
a conjunction, deliberately stricter than "either index passes": requiring both to
clear the bar — one on a seen underlying, one on a naive out-of-sample one — is what
upgrades a SPY pass into a result that generalizes. SPY alone decides the primary
hypothesis (H1); IWM decides whether a SPY pass is *confirmed*. IWM is run and
reported whatever SPY does.

---

## 6. Outcome language (pre-committed)

Fixed while the outcome is unknown, because after results exist every author reads
t = 1.9 as "nearly significant." Each row is the sentence published verbatim.

| Outcome | Registered language |
|---|---|
| SPY passes §5.1 **and** IWM confirms §5.2 | "The index volatility premium is significant on the put wing, survives realistic costs, and **replicates out-of-sample on a naive index (IWM)** — a confirmed put-wing VRP. Scoped to the in-sample regimes (no GFC); promote to a registered finding and pin." |
| SPY passes §5.1, IWM does **not** confirm §5.2 | "A significant, cost-surviving put-wing premium on SPY that does **not** replicate out-of-sample on IWM — treat as index-specific, not confirmed, pending a third index." |
| SPY gross t > 2 but net (0.5 bp) t ≤ 2 | "The SPY put-wing premium is real but not tradeable net of realistic hedge cost — the post-2010 picture (Dew-Becker & Giglio), now on the put wing too. (IWM reported beside.)" |
| SPY gross t ≤ 2 | "Null on the put wing: no significant delta-hedged premium on SPY over this span, even gross. (IWM reported beside.)" |

The **mechanism clause** (§1.3, put t ≥ the call's +2.54) is appended verbatim to
the first two rows when met — "and is at least as strong as the call wing,
consistent with the skew premium" — and its absence is stated explicitly when not.

No result of this experiment supports trading decisions; the repo's standard
disclaimer applies.

---

## 7. Secondaries and robustness (reported, never promoted)

Run only if the §2 primary runs; none can change the §5 verdict.

- **ATM straddle.** Short call + short put at \~0.50 / −0.50 delta, net-delta
  hedged — the canonical Coval-Shumway / AQR variance harvester. Reported for the
  symmetric variance premium; a two-leg engine extension (§9).
- **Put delta sweep.** −0.10, −0.15, −0.40 — the wing's shape. Exploratory only.
- **Mid fills.** The option leg at the mark instead of the bid, isolating the
  option spread's contribution.
- **Stationary block bootstrap.** A non-parametric companion to the Newey-West t
  on the daily vol-P&L (seed fixed in the analysis script), guarding the HAC
  correction's assumptions on overlapping cycles. Reported beside the t; the t is
  the verdict.
- **IWM is not here.** The out-of-sample IWM replication is a *committed
  confirmation arm* (§5.2), fetched and run alongside SPY and able to promote a SPY
  pass to "confirmed" — not a reported-never-promoted secondary. It is named in this
  list only to point at §5.2.

---

## 8. Power / what a null means

The SPY span is \~3,900 trading days, \~175 monthly cycles — the same sample that
gave the call wing +2.54, so the put wing has comparable power. The published call
cost curve (+2.54 → +1.97 across 0 → 1 bp) shows the realistic detectable range: an
effect that needs a sub-penny spread to clear t = 2 is at the edge of this sample's
resolution. A SPY null therefore means "no premium detectable at this size on one
index over \~15 years," not "no premium." IWM's span and cycle count are set by its
fetch (§3.1) and reported with the results; its power is lower if its clean chains
start later. The confirmation rule (§5.2) is a **conjunction** — both indices must
clear the bar — so it is conservative against false positives, and on the flip side
an IWM that is merely underpowered (not negative) can block a "confirmed" claim;
that asymmetry is accepted ex ante, and the §6 row-2 language covers it. The modal
expected outcome, per §1.5, is a gross-significant put gain whose net-of-cost
survival is marginal; the §6 language for every branch is committed.

---

## 9. Implementation constraints

- **Data fetch:** extend `download_option_dailies.py` to keep put rows (drop the
  call-only filter; `infer_spot`'s strike band must accept negative deltas), then
  run the full data lifecycle **sequentially — one ticker to completion before the
  next** (shared rate budget) — for **SPY** (puts added to the existing call file)
  and **IWM** (a fresh ticker, calls + puts, needing its own trading-day calendar
  from `download_prices.py` and the era clip its validation battery sets): resumable
  fetch wrapped in the retry loop; validation battery; `gzip -9`; sha256 into
  `data_checksums.sha256`; `gh release upload data-2026-06`; CI cache + fetch-glob
  update; checksum round-trip; cold-storage copy — per the Option-Chain Data
  Pipeline rules.
- **Engine — put entry:** add `select_put_entry` (the §2.1 band and nearest-delta
  rule). The naked-call settlement/close branches mirror to puts
  (`intrinsic = max(0, strike − settle_price)`).
- **Engine — signed hedge:** the delta-neutral rebalance currently clamps the
  hedge to `[0, 1] × shares` (long only). Extend it to signed deltas so a short
  put's `+0.25` position delta is neutralized by **short** stock
  (`hedge_shares` may be negative); the half-spread cost applies to
  `|hedge_trade|` regardless of sign. No other change to the loop.
- **Pin protection:** the put path arrives behind the entry selector / hedge sign;
  every existing `test_vol_premium.py` pin (the call wing, the mechanics, the
  audit invariants) must pass byte-identical before any put run.
- **Ordering:** no put fetch or run before this file's merge commit to `main`; the
  §9 engine code must be committed before any put number is produced. Results land
  in a separate, later PR citing this registration's merge commit and the
  analysis-code commit.

---

## 10. Amendments

Any change to this document after its registration merge — whether or not any
result has been computed — must be recorded in an "Amendments" section appended
here, with date, what changed, and why; every claim affected is demoted to
exploratory. Silent edits void the registration.

---

## 11. Lineage and references

- Internal: the pinned call-wing result (`TestSpyShortVolRegression`) and the
  engine it shares (`vol_premium.py`, `run_real_short_vol_overlay` /
  `short_vol_statistics`); the audit and caveats in `docs/vol_premium.md`; the
  covered-call null this builds on (`TestMsftRealRiskManagedRegression` /
  `TestQqqRealRiskManagedRegression`, delta-hedged t \~0) and its exploration-log
  entry; the data pipeline in `download_option_dailies.py`.
- Method lineage: the delta-hedged-gain measure follows Bakshi & Kapadia (2003,
  *Journal of Derivatives*); the put-wing / skew premium follows Bondarenko,
  Broadie-Chernov-Johannes, and Constantinides-Jackwerth-Savov (2013); the
  post-2010 tradeable-premium decline follows Dew-Becker & Giglio (2025); the
  Newey-West HAC convention is as implemented in `compute_statistics`.
- Precedent: the call-wing phase of this same experiment (merged in the
  delta-neutral short-vol PR), which established the engine, the rf-base audit, and
  the cost-curve method this registration reuses.
