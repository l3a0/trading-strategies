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

## 0. Reader's guide — how to read a null

This report doubles as a teaching example, like the registration it answers.
The operative content is §1–§3; the surrounding sentences explain what each
number is allowed to mean.

**Why the experiment ended at Stage 1.** The registration's Stage 1 (§5) is a
kill-gate: two cheap mechanism checks on records that already existed, able
only to stop the experiment. Passing would have proven nothing — the same
mined sample generated the hypothesis — but failing ends it, because a gate
whose mechanism is absent in-sample has no claim to a confirmatory test.
Both checks failed, each with the wrong sign. Stage 2 (the 1,000
placebo-gate engine re-runs) never ran, no record-arm backtest was ever
computed, and the §2.3 no-promotion rule forbids retreating to a friendlier
signal definition or attribution convention.

**Why the verdict sentence below was written before the result existed.** After
results exist, every author reads a p of 0.07 as "nearly significant."
The registration pre-committed one sentence per outcome (§7) while the
outcome was unknown; the row that fired is quoted verbatim in §1. That is
the entire point of the exercise: the conclusion was constrained before the
evidence could argue.

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
  regime on each day's *close date*, while the registered test attributes
  each cycle to its *entry-date* state (§2.3, fixed before any outcome was
  viewed). Same engine, same decade, two attribution conventions, opposite
  signs. That contrast is the report's main teaching exhibit: an attribution
  convention can manufacture a signal, which is exactly why the convention
  had to be pinned in advance.
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

---

## 6. Lineage

- Registration: [docs/prereg_trend_gate.md](prereg_trend_gate.md), effective
  at merge commit `4d2239b`. Operative sections cited above: §1.3, §1.4,
  §2.1, §2.3, §5, §6.1–§6.3, §7, §8, §9, §10, §11.
- Analysis code: `trend_gate.py` and `test_trend_gate.py` at `d9ddb43`,
  committed before any Stage 1 number existed (§10 ordering; the git
  history is the proof).
- The design choice this null validates: the tutorial's "What We'd Add
  Next," item 10 (*Entry trend filter*), which predicted exactly this
  outcome for the call side.
- Method lineage: placebo-calibrated inference per White (2000); add-one
  Monte Carlo p-values per Davison & Hinkley (1997) — both cited in full in
  the registration's §12.
