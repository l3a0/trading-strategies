# The first real factor exploration — US large-caps, 2016–2026

The factor stack (F1–F4, H1, H2 of [the integration plan](integration_plan.md)) was built and pinned on a
**synthetic** panel. This is its first run on **real** equities.

**Result: 0 of 63 grammar factors survive the e-LOND bar. The machinery works on real data; the verdict is
no edge.** Exploratory, and the standing limits hold — promotion is CLOSED and any survivor would be a
pre-registration candidate, not a verdict (there are none here).

## The panel

`factor/factor_panel.py` assembles **US_LARGE_CAP**: a committed, hand-selected universe of **42 large-cap US
equities** (`FACTOR_UNIVERSE` — tech, financials, consumer, health, industrials/energy), their
split/dividend-adjusted daily closes downloaded from Yahoo Finance into a dates×tickers panel. The
committed snapshot is **42 tickers × 2514 trading days, 2016-06-27 → 2026-06-26**, zero gaps.

The prices are free and regenerable (yfinance), so the panel CSV lives in git like the repo's other price
CSVs — it is not premium option-chain data.

## Honest caveats

These bound what a result here can claim, so they are stated loudly. Survivorship biases *toward* finding
an edge, so a **null despite it is conservative** — the result is robust, not fragile.

1. **Survivorship.** The universe is *current* large-caps — today's survivors, hand-selected, with no
   delisted or failed names (the list is committed but **not** independently pre-registered). That biases
   momentum and low-volatility signals **upward** (the panel is the winners). Fine for "does the machinery
   find real cross-sectional signal," not for a tradeable claim.
2. **One-day forward horizon.** The IC is measured at `fwd=1` (next-day returns) — high-frequency and
   microstructure-sensitive; classic factor research uses 10–20-day horizons. The engine supports any
   horizon via the `fwd` parameter; sweeping it is deferred.
3. **Frozen at download.** `period='10y'` is relative to the download date, so the committed CSV is a frozen
   snapshot; regenerating gives a newer panel (a different end date) that would re-pin. The committed CSV is
   the reproducible artifact — tests pin against it, not a live re-download. (`FACTOR_END` is the
   synthetic-default as-of; the real backend stamps each row's `end` with the panel's actual last date,
   2026-06-26, via `make_factor_backend`.)

## What the run found

- **All 63 factors type coherently** — 35 load on `trend`, 28 on `lowvol`. Every expression in the bounded
  grammar slice loads on a registered premium via the H1b loading regression — real equities have real
  trend and volatility structure, so nothing is mechanism-incoherent. (On the synthetic panel, 5 of 63 were
  incoherent; on real data, 0 are.)
- **No factor has a significant Information Coefficient.** The strongest factor's t-stat is |t| = 1.52 —
  the **Newey-West HAC-corrected** IC t-stat (the daily IC series is autocorrelated; the correction is the
  same Bartlett-weighted convention the option path uses, so the shared `t_stat_newey_west` field is
  literally accurate for factors), comfortably below the conventional t=2 bar. Under the e-LOND
  false-discovery control (α = 0.10, [docs/prereg_fdr_budget.md](prereg_fdr_budget.md)), a factor survives
  only if its e-value clears a position-dependent bar — loosest at the head of the stream, tightening with
  each comparison. The strongest factor's one-sided p (\~0.06 at |t| = 1.52) calibrates to an e-value far
  below even the head-of-stream bar, so across the 63-factor cross-section **0 survive**.

This is the honest outcome of a small, bounded grammar on a single survivor-biased universe: the scoring,
the mechanism gate, and the FDR control all run correctly on real data, and they find nothing that clears
the bar. A null is the expected result for a first pass — it is what the apparatus is built to deliver
without flinching.

## Reproduce

```bash
python -m factor.factor_panel --build      # download the universe panel + run the search
python -m factor.factor_panel              # re-run on the committed snapshot (no re-download)
```

The dataset-gated `tests/test_factor_panel.py::TestRealPanelExploration` pins the full 63-factor headline (0/63
survive, all coherent, strongest |t| < 2) against the committed snapshot — the vectorized search is fast
enough (\~2s) to pin the real result directly.

## Performance

The full 63-factor search on \~2500 days runs in **\~2 seconds**. The Information Coefficient and the
long-short returns are **vectorized** (`information_coefficient`, `long_short_returns` — rank-based, no
per-date Python loops), verified `allclose` to 1e-12 against the original loops, so the speedup is
behavior-preserving. That makes a real-data factor search fast enough to run the proposer's many rounds
routinely.
