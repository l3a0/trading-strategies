# Generative search — a recorded null result

> **Epistemic status: EXPLORATORY.** This is a sample-spending, kill-or-justify scout, **not** a
> registered verdict. Recording the null only stops it from being re-derived every session. A cell that
> *passed* would earn a manual pre-registration, never a headline — and would still stay exploratory
> until the Phase-C time-axis holdout exists. (See [docs/read_gate.md](read_gate.md),
> [docs/edge_search.md](edge_search.md).)

## The one-sentence finding

Run the generative menu-walker proposer on real option chains and it adds **zero survivors** — because
the search was **already saturated before it began**: the published lifetime ledger sits past the e-LOND
bar, the mechanism gate rejects most of what the grammar can express, and the coherent remainder carries
no edge at honest significance. Every comparison is recorded to the committed `gen_ledger.jsonl` audit log.

## What was run, and recorded

`generative_search.run_generative_search` walks a bounded slice of the production grammar — the first 5
single-leg structures and the first 5 same-expiration two-leg structures in canonical-key order — across
the **seven onboarded search tickers** (MSFT, SPY, QQQ, GLD, XLE, EEM, NVDA), with **TLT sealed by
omission**: **70 generative cells**. Each is scored by the generative kill-gate (`score_composition`, with
its inline mechanism gate) and judged over the lifetime e-LOND stream with the committed 75-cell
`idea_ledger.jsonl` as the read-only head (design A, [generative/generative_engine.py](../generative/generative_engine.py)).
Every judged cell — its t-stat, e-LOND verdict, and data lineage — is **recorded to `gen_ledger.jsonl`**,
the generative twin of the closed-grammar ledger and the audit log of this run. "Exploratory" means *not a
registered verdict*, not *unrecorded*: the lifetime e-LOND budget is only honest if every look is on the
record (the same reason the structure campaign records its exploratory 56-cell batch).

Reproduce: `python -m generative.generative_search --record` (the 7-ticker run; slow, one engine pass per cell).

## The three results

### 1. Zero survivors — and it was decided before the search

The headline is not a close miss. Of the 145 cells now in the lifetime stream (75 published named + 70
generative), **none** clears the e-LOND control. More pointedly, the **published 75-cell stream alone was
already past the bar**, with no chain data required to see it:

| quantity | value |
| --- | --- |
| strongest cell ever observed (empirical ceiling) | t ≈ 2.17 (published) → 3.35 (a generative cell) |
| bar the *next* cell must clear (at the published head / after recording) | t ≥ 6.35 → 6.63 |
| discoveries so far (R) | 0 |

The bar is what the **registered e-LOND discount sequence requires**, not a threshold imposed by hand.
With `Σγ_t ≤ 1` spread over the stream and **R = 0** discoveries to loosen it (the e-LOND reward is
`α·γ_t·(R+1)`), the bar only **rises** as cells accumulate — from t ≈ 3.11 at position 1 to t ≈ 6.63 by
position 145. It had already overtaken the empirical ceiling at stream position 2. The saturation readout
says it plainly: *"more rounds cannot flag — widen the grammar or move to the time-axis holdout."*

**And the strongest cell is exactly what noise predicts.** The 2.17 → 3.35 jump is not the search getting
warmer — it is the search getting more chances to draw a high `t` by luck. The two ceilings are different
*snapshots*: 2.17 is the best of the 75 published cells (SPY's `Δ0.25` short call); 3.35 is the best of all
145, and that cell is itself generative — SPY's deep-OTM `Δ0.05` short call (`p ≈ 0.0004`). The grammar's
finer delta granularity (0.05 is a strike bucket; the published campaign walked only 0.25 and ATM)
surfaced a structure that collects tiny premium and is almost never assigned: steady small wins, a higher
`t`. It clears the conventional `t ≥ 2` bar *and* the HLZ `t ≥ 3` multiple-testing bar, and as a
*first-and-only* hypothesis it would even clear the e-LOND control (that bar is `t ≥ 3.11`, and
`3.35 > 3.11`). But it is the best of **131 scored cells**, and the largest of about 131 independent noise
draws is `≈ √(2·ln 131) ≈ 3.1` — almost exactly what was observed. The rising e-LOND bar (`3.11 → 6.63`)
is precisely that accounting: a result that looks significant on its own evaporates once you count the
looks it took to find it. That is the entire reason for the lifetime stream.

### 2. The mechanism gate rejects two-thirds of the proposals

Of the 70 generative cells, **47 never reached the FDR pool**:

- **33 are mechanism-incoherent** — `derive_family` returns `None` for a structure that harvests no
  registered premium (a lone `long call` is long vega; a net-long-vega put spread sells nothing). These
  fail *closed* (`p = None`, never flags) regardless of their t-stat. SPY's `long call Δ0.30` ran a t of
  −3.53, but it is unclassifiable, so it cannot — and must not — flag. This is the contrast-paper defense
  applied **per composition**: a structure cannot survive on a lucky t if its *mechanism* is incoherent.
- **14 do not trade** — all-long structures collect no short premium, so the must-trade guard flags them
  `measurement_invalid` rather than scoring an idle curve.

Only **23 cells** were coherent and traded (21 `variance`, 2 `carry`), and their t-stats are weak — the
best, SPY's 5-delta short call, reaches t ≈ 3.35, still far below the bar.

### 3. More search is the wrong lever

The apparatus does not let the search keep mining — it reports its own futility and names the next move.
The binding constraint was never the proposer's cleverness. It is that (a) there is no volatility-risk
premium at honest significance on these names once spreads and delta-hedging are paid, and (b) even a real
survivor would stay **exploratory** until the **Phase-C time-axis holdout** exists to defend it against
in-sample luck (and, for an LLM author, training-data recall). Widening the grammar or holding out a
post-cutoff span are levers; recording more same-era cells is not.

## A curiosity worth a future tightening

The grammar can still express a **degenerate** structure: a same-strike short + long put nets to \~flat
(the legs cancel), yet it "trades" and `derive_family` typed it `carry` from the tiny residual signature.
It is economically null and non-significant, but it is the same family of edge case as the duplicate-leg
scale-multiple the Phase-4 seal verification caught ([generative/generative_grammar.py](../generative/generative_grammar.py)
`validate_composition`). A candidate next tightening: forbid a same-strike short+long pair of the same
right (a true zero, like the duplicate-leg rule).

## How it is pinned

| Surface | What it holds |
| --- | --- |
| [generative/generative_search.py](../generative/generative_search.py) | the runner (`run_generative_search`, `--record`) that walks the slice, judges over the lifetime stream, and records to the audit log |
| `gen_ledger.jsonl` | **the committed audit log** — the 70 recorded generative comparisons, e-LOND verdicts, and lineage (design A's generative twin of `idea_ledger.jsonl`) |
| [tests/test_generative_search.py](../tests/test_generative_search.py) | always-run: the published stream is past the bar (no chains) + the recorded ledger is well-formed, 0 survivors, distinct cells, TLT absent; dataset-gated: re-running the menu-walker reproduces the null |
| this doc | the human-readable negative-results record |

_Last updated: 2026-06-26._
