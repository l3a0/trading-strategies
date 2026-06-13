# Results: the trend-gated covered-call experiment

**Status:** FINAL — killed at Stage 1. Per [the registration](prereg_trend_gate.md)
§10, this report cites the registration merge commit `4d2239b`
([#12](https://github.com/l3a0/trading-strategies/pull/12)) and the
analysis-code commit `d9ddb43`
([#14](https://github.com/l3a0/trading-strategies/pull/14)). Stage 1 ran on
2026-06-12 from a clean `main` checkout at `d9ddb43`, in the registered
environment (Python 3.9.6, numpy 2.0.2, pandas 2.3.3). The registration
document was never amended; no exploratory demotions apply.

**Question of record (restated):** Does suspending covered-call selling while
the underlying is in an uptrend — holding the shares uncovered — produce a
positive call-selling residual, and does the trend signal select suspension
days better than structure-matched random gates?

---

## 0. Reader's guide — the experiment in plain words

This report doubles as a teaching example, like the registration it answers.
The operative content — exact numbers, exact rules — is §1–§3; this section
tells the same story in plain language first, then explains how to read a
null.

**The idea being tested.** Selling covered calls on these stocks loses money
one way above all others: the stock takes off, and the call you sold has to
be bought back at a painful price. The fix under test sounded obvious —
don't sell calls while the stock is trending up. Sell only when it looks
flat or falling, and just hold the shares the rest of the time.

**The sealed envelope.** Before any outcome data was touched, every rule was
written down and merged: the exact definition of "trending up," the data,
the statistics, the pass thresholds — and the verbatim sentence to publish
for each possible result (§7). Once results exist, every author reads a
near-miss as a hit (in the standard scoring, a p-value of 0.07 sits just shy
of the usual 0.05 bar and becomes "nearly significant"), and every
bookkeeping choice bends toward the desired answer. Writing the rules first
removes the option. The sentence in §1 was chosen while the outcome was
unknown; this report just opens the envelope.

**The cheap first checkpoint.** Stage 1 is a kill-gate: two quick checks on
trading records that already existed, able only to stop the experiment.
Passing would have proven nothing — the hypothesis came from staring at
these very records, so the same records cannot also confirm it — but failing
ends it, because an idea whose mechanism is absent in-sample has no claim to
an expensive confirmatory test. The two questions: did call trades *started*
on uptrend days actually lose more than the rest? On uptrend days, did the
stock go on to end up above the would-be strike more often? If either answer
is no, stop.

**The answer came back backwards.** Trades started during uptrends made
about $439 *more* per trade, not less. After uptrend days the stock went on
to clear the strike slightly *less* often, not more. Measured against
10,000 fake skip-schedules — random gates with the same rhythm of streaks
but no knowledge of the market — the real signal sat mid-pack. It picked
days to skip no better than a coin with the same calendar habits. So the
experiment stopped itself: Stage 2 (the full thousand-fake-gate comparison)
never ran, the real trend gate was never run through the simulator at all,
and the no-promotion rule (§2.3) forbids retreating to a friendlier signal
definition or bookkeeping convention.

**Why the hunch felt right anyway: two bookkeepings, opposite verdicts.** The
chart that inspired the hypothesis (the tutorial's figure 10) files each
dollar under the day it *arrived*, and on that view uptrend days look nearly
worthless for the call-selling side. But a gate acts on exactly one day, the
day you are about to sell, and the trade then lives for weeks. The "uptrend"
label is a rear-view mirror: the close sitting 5% above its own 200-day
average, refreshed a day late. It summarizes most of the past year; the
trade's fate hangs on the next 30 days.

The two filings disagree most at turning points. In the spring of 2020 the
label read "downtrend" for months while the market ripped upward; the
registration names that exact window as the known failure mode (§6.4),
because a gate keyed to this label keeps selling calls straight into the
recovery. By the time a long climb finally certifies the label as "bull,"
the explosive move is already behind it, and what follows is more often the
grind that an out-of-the-money call survives.

The registration even names how the two filings split a single disaster
(§2.3): a call sold under a "downtrend" label that gets run over as the
climb flips the label books its loss to a *bull* day under arrival-day
filing — the day that made the decision gets the alibi, and the day that
inherited the trade gets the blame. File each trade under the day it was
*decided*, the only day a gate can act, and the chart's picture inverts.
That is why the decision-day convention was pinned before any outcome was
viewed, and §2 below reports what it found.

**Why "better than before" was never the bar.** One registered
side-measurement ran regardless of the verdict: the first 100 fake
skip-schedules were pushed through the full simulator, to put a dollar
figure on a trap the registration had flagged (§1.3). Those skill-free
gates, each skipping just as many days as the real one would, still lose
about $71 for every day spent with a call sold against the shares (§3.2).
Trading less cannot, by itself, rescue a strategy that loses on the days it
does trade — any abstinence "improves" it, the way skipping some lottery
tickets does. The gate had to make call-selling actually *profitable*, and
the mechanism tests say it would not have come close.

---

## 1. Verdict of record

Stage 1 failed the §5.4 gate, so the registered outcome language for
"`T ≤ 0`, or Stage 1 kill" applies, verbatim:

> "Null. The engine's no-trend-filter design choice is validated empirically
> (tutorial, What We'd Add Next, item 10)."

The house prior (§1.4) — that selling into downtrends is desirable for call
sellers and the filter would not help — stands confirmed rather than merely
unrefuted. No result of this experiment supports trading decisions; the
repo's standard disclaimer applies.

---

## 2. Stage 1 numbers

Both tests tagged the same fixed baseline records (884 closed cycles; 11,452
exceedance days) with each ticker's own §2.1 signal, then re-tagged them
under 10,000 placebo sequences from the registered seed-20260611 stream.

| Test | Registered prediction | Measured | Add-one p | Placebo percentile |
| --- | --- | --- | --- | --- |
| A — entry-state cycle split, `D_A` | `D_A < 0` (bull entries do worse) | **+$439.44** per cycle | `p_A` = 0.736 | \~74th |
| B — price-path exceedance, `D_B` | `D_B > 0` (bull days finish above the strike more often) | **−3.07 pp** | `p_B` = 0.763 | \~24th |

The §5.4 gate required strict `D_A < 0` AND strict `D_B > 0` AND
`min(p_A, p_B) ≤ 0.10`. The result fails all three conditions: both point
estimates carry the wrong sign, and both sit comfortably inside the placebo
distribution (the percentiles above locate the real arrangement among 10,000
skill-free ones — neither is within reach of either tail). Zero degenerate
sequences needed replacement in either test, so the §5.1/§11 amendment
trigger (2%) was never approached.

Two readings of the reversal, and why neither is a finding:

- **Cycles entered during uptrends made more money, not less.** This is the
  opposite of the figure-10 intuition that generated the hypothesis — and
  the registration anticipated the gap: figure 10 attributes P&L to the
  regime on each trade's *close date*, while the registered test
  attributes each cycle to its *entry-date* state (§2.3, fixed before any
  outcome was viewed). The two views also differ in data — figure 10 is the
  proxy engine's single-ticker MSFT run; this test pools real-chain cycles
  across three tickers — but the convention is the difference the
  registration anticipated and pinned in advance (§2.3), and §0 walks
  through it slowly. The contrast is the report's main teaching exhibit: an
  attribution convention can manufacture a signal, which is exactly why the
  convention had to be pinned before any outcome was viewed.
- **A momentum-assisted-selling hypothesis is not licensed by this.** At the
  74th placebo percentile the positive `D_A` is unremarkable among random
  arrangements, the registered complement-arm test that could have spoken to
  it (§7, row 4) lives in Stage 2 and never ran, and any follow-up would
  need its own registration.

---

## 3. The minimum detectable effect, before and after

The registration published an MDE forecast (§9) so a null would read as "no
effect this size or larger," not "no effect." Both registered artifacts are
published here with the Stage 1 report, as required.

### 3.1 Artifact (a): the exact MDE table from the baseline records

| Quantity | Registered forecast | Measured |
| --- | --- | --- |
| Per-cycle σ (pooled baseline) | plausibly $7,000–13,000 | **$9,073** |
| Expected record-arm cycles | \~325 | **331** |
| Per-cycle mean needed for t = 2 | \~$780–$1,440 (by σ) | **+$997** |

The forecast arithmetic survives contact with the data almost unchanged: the
measured σ lands mid-band and the expected sample size within 2% of the
registered estimate. The bar the gate needed to clear was a swing of roughly
$1,000 per cycle from the baseline's \~−$1,000 mean — and Test A measured the
bull/non-bull difference pointing the wrong way entirely.

### 3.2 Artifact (b): the placebo-space MDE

The first 100 Family R sequences (they remain the first 100 of the 1,000
that Stage 2 would have used) were pushed through full engine re-runs on all
three tickers — placebo gates only; no record-arm run was computed and
nothing was unblinded. Their §6.1 statistic `T_i` (net overlay P&L per
short-call day, equal ticker weights):

| Statistic | Value |
| --- | --- |
| Mean | **−$71.45** per short-call day |
| Standard deviation | $29.30 |
| 5th–95th percentile | −$115.22 to −$18.27 |
| Replacements | 0 of 100 |

Two registered design points become concrete here. First, the abstinence
confound (§1.3) in dollars: a skill-free, exposure-matched suspension gate
still loses about $71 per day of short-call exposure on these chains —
"better than the unconditional overlay" was never a meaningful bar, because
every gate clears it. Second, the §6.3 pass rule's two conditions were not
equally binding: even a break-even record arm (`T = 0`) would have beaten
roughly every placebo in this sample, so the rank condition was nearly free
and the experiment always hinged on `T > 0` — the gate had to make
call-selling *profitable*, not merely less unprofitable. Stage 1's mechanism
tests say it would not have come close.

---

## 4. What this kills, and what it leaves standing

Killed, within the registered scope (three correlated secular-bull
underlyings, 2010–2026 spans, the published 25-delta / 30-day / 0.75-close
operating point, entry-only gating):

- The registered hypothesis H1, at its own pre-committed thresholds.
- The figure-10 close-date intuition as a tradable entry signal — it does
  not survive entry-date attribution.
- The last open conditioning lever from the June 2026 scans: the IV-richness
  gate was found mechanism-free before registration, and the trend gate now
  joins it with a registered null.

Left standing, explicitly unclaimed (§1.3 and §8 were never reached):

- Anything at GFC scale — the 2008–2010 era is untestable on these chains.
- Other signal definitions, underlyings, or option parameters: the §8
  robustness grid runs only if Stage 2 runs, and the gate-as-grid-axis
  question was reserved for a separate registration.
- Momentum-assisted call selling, per §2 above.

---

## 5. Reproduction

From a clean checkout at or after `d9ddb43`:

```bash
./fetch_option_data.sh             # chain datasets (checksum-verified)
python trend_gate.py stage1        # Tests A + B, gate rule, MDE artifact (a)
python trend_gate.py placebo-mde   # MDE artifact (b): 100 Family R re-runs
```

Every number above is deterministic: the placebo stream is
`numpy.random.default_rng(20260611)` consumed in registration order (its
first accepted sequence is fingerprint-pinned in `test_trend_gate.py`), the
engine is deterministic given the committed data, and the analysis CLI
refuses to run stages from a dirty working tree so any reproduction carries
the same provenance this report does. Family R checkpoints land in
`trend_gate_runs/` (gitignored; safe to delete — re-running recomputes
byte-identical records).

The published figures are pinned in CI by `TestTrendGateStage1Regression`
(`test_trend_gate.py`), so an engine or generator change that silently
shifted the verdict fails the build rather than quietly invalidating this
report. The class pins the deterministic core (D_A, D_B, the counts, the
§9(a) MDE) from three baseline runs, the gate verdict (p_A, p_B, the FAIL)
from the 10,000-sequence re-tag, and one Family R record (the first
accepted sequence's `T`) as a cheap drift-check on the placebo-MDE pipeline
— the full 100-record summary in §3.2 stays reproducible via `placebo-mde`.

---

## 6. Lineage

- Registration: [docs/prereg_trend_gate.md](prereg_trend_gate.md), effective
  at merge commit `4d2239b`. Operative sections cited above: §1.3, §1.4,
  §2.1, §2.3, §5, §6.1–§6.4, §7, §8, §9, §10, §11.
- Analysis code: `trend_gate.py` and `test_trend_gate.py` at `d9ddb43`,
  committed before any Stage 1 number existed (§10 ordering; the git
  history is the proof).
- The design choice this null validates: the tutorial's "What We'd Add
  Next," item 10 (*Entry trend filter*), which predicted exactly this
  outcome for the call side.
- Method lineage: placebo-calibrated inference per White (2000); add-one
  Monte Carlo p-values per Davison & Hinkley (1997) — both cited in full in
  the registration's §12.
