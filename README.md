# Covered Call Backtester

[![CI](https://github.com/l3a0/covered-call-backtesting/actions/workflows/ci.yml/badge.svg)](https://github.com/l3a0/covered-call-backtesting/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/l3a0/covered-call-backtesting/blob/main/covered_call_backtest.ipynb)

A from-scratch Python backtester for the covered call overlay strategy. Prices options with Black-Scholes (using `math.erf` for high-precision CDF), estimates IV from rolling historical volatility with regime-based multipliers, and simulates day-by-day trade decisions over multi-year price histories. A companion adapter re-runs the same strategy on ten years of real option chains — and reverses the headline result; see [Reality check](#reality-check-real-option-chains) below.

## Quick start

```bash
# 1. Set up the environment (one time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Download fresh price data — there's already an MSFT CSV in the repo
python download_prices.py                   # default: MSFT, 10y
python download_prices.py --ticker AAPL     # any ticker
python download_prices.py --ticker SPY --period 5y

# 3. Run the backtest
python cc_backtest.py
```

Sample output (MSFT 2016-04 → 2026-04, $100K portfolio):

```text
Capital:                         $  100,000.00
Contracts (100 shares each):               20    ($95,573.55 stock + $4,426.45 cash)

Returns
    Buy & Hold Final:            $  746,166.44     +646.17%
  + Net Overlay P&L:             $  268,424.87     +268.42 pp
  = CC Overlay Final:            $1,014,591.31     +914.59%

Overlay P&L Breakdown
    Gross Premium Collected:     $  998,518.91    (income from 181 calls sold)
  - Buybacks + Assignment Costs: $  730,094.04    (paid to close ITM calls + capped upside on assignment)
  = Net Overlay P&L:             $  268,424.87    (26.9% retained)

Activity
    Calls Sold:                            181
    Win Rate:                             81.1%
    Max Drawdown:                        22.86%

Statistical Significance (H0: overlay adds zero value vs. buy-and-hold)
    Days in Sample:                      2514    (9.98 years)
    Annualized Excess Return:          +1.249%
    Annualized Excess Vol:               9.90%
    Sharpe of Excess Return:           +0.126
    t-stat (naive, IID):                +0.40    (assumes independence — inflated for overlays)
    t-stat (Newey-West, L=8 ):          +0.46    (correct: accounts for position autocorrelation)
    Clears t=2 bar?                     False    (conventional significance)
    Clears t=3 bar (HLZ 2016)?          False    (multiple-testing adjusted)

Degrees of Freedom — 3-year in-sample window (Pardo 2008)
    Observations (trading days):          756
    Consumed (3 params + 30 LB):           33
    Remaining (free):                     723    (95.6% — Pardo floor 90%)
    Bar-level DOF adequate?              True    (necessary, not sufficient)
    Independent trades (median):           36    (grid range 17-73)
    >= 30 trades for inference?          True    (clears it; 2-year window would not)
```

The portfolio is sized into whole 100-share contracts at the initial price; any leftover (here, $4,426 of $100K with MSFT at ~$48) sits as 0%-yield cash. Returns are measured against `capital`, so the cash drag is included. To run a single-contract simulation, omit `capital` from `params`.

The bottom block tests whether the overlay's excess return over buy-and-hold is statistically distinguishable from zero, using Newey-West HAC standard errors that correct for the autocorrelation introduced by holding the same option position across multiple days. On this MSFT sample the t-stat is 0.46 — well below the conventional significance bar of 2 — meaning the $268K of headline overlay P&L isn't reliably distinguishable from noise. On real option chains the verdict is harsher still: the same overlay run on traded quotes *loses* $183,552 (see [Reality check](#reality-check-real-option-chains)). See the [tutorial's Part 5](tutorial_covered_call_backtest.md#part-5-robustness-checks--proving-its-not-luck) for the full significance reasoning.

The final block reports Robert Pardo's degrees-of-freedom check for the default 3-year walk-forward training window. Both checks pass: the bar-level test (756 observations minus 3 free parameters and a 30-bar indicator lookback leaves 95.6% free, above Pardo's ~90% floor) and — the binding one — the ~30-trade sample-size floor, which the 3-year window clears (median 36 trades). The window is sized to 3 years precisely for that: a 2-year window leaves the median grid fit at ~24 trades, short of the floor. Note this is necessary, not sufficient — a clean DOF check means the model isn't over-parameterized, not that the edge is real (the t-stat above settles that). See [tutorial Part 4](tutorial_covered_call_backtest.md#part-4-walk-forward-optimization).

For an explanation of each output line — including what "assignment loss" means and why buybacks can dominate the overlay's gross premium income — see the [tutorial](tutorial_covered_call_backtest.md) (its Glossary defines the terms; Part 3 walks through the trade-by-trade math).

## Reality check: real option chains

The engine above has never seen an option chain — it manufactures implied volatility from realized volatility. To measure what that assumption costs, the repo carries real daily chain slices (\~24M quotes via Alpha Vantage) — fifteen years for QQQ (2011–2026) and, for MSFT and SPY, sixteen years the runs use (2010–2026, including the 2010–2013 sideways era), plus put wings on SPY/MSFT/QQQ and a both-wing IWM set for the put-side VRP experiment and its cross-section ([docs/vol_premium.md](docs/vol_premium.md)) — per-roll entry snapshots for six underlyings, and an adapter ([real_cc_backtest.py](real_cc_backtest.py#L225)) that re-runs the identical strategy on traded quotes: sell at the bid, buy back at the ask, real deltas and expirations, unadjusted closes. The MSFT/SPY datasets reach back to 2008, but the 2008 → mid/late-2010 era carries vendor placeholder greeks (lattice-quantized IVs, deltas unrelated to moneyness) throughout the entry band — on MSFT, 2008–2009 offers \~2 trustworthy entry days a year — so every run excludes that era outright (per-ticker boundaries in `CHAIN_CLEAN_START`). The honest consequence: the GFC itself is untestable on these chains; no run claims it.

The proxy's results do not survive the trial:

| Net overlay P&L, 2016–2026 | Proxy (same series) | Real chains | NW t-stat (real) |
| --- | --- | --- | --- |
| MSFT | +$269,948 | **−$183,552** | −1.73 |
| QQQ | +$120,217 | **−$156,628** | −1.78 |

(The proxy column re-runs the engine on the unadjusted series the chains require. The sample output above — the published $268,424.87 — is the same engine on the dividend-adjusted series; the dollar proximity between the two is partly coincidental, the verdict identical.)

The loss anatomy is the same on both underlyings: profit-target wins (MSFT: 122 closes, +$429K) overwhelmed by deep-ITM forced buybacks (54 closes, −$611K). The driver differs by instrument. On QQQ, mid-quote fills recover only \~$12K of the loss — the premium economics are simply absent. On MSFT, mid fills recover \~$108K (single-name spreads are expensive), but the overlay still loses $76K with the spread given away for free. Across all six underlyings sampled (QQQ, MSFT, SPY, IWM, GLD, TLT), the proxy inflates IV 1.27–1.56× and same-contract premiums 1.55–2.33×, worst where realized volatility is lowest.

The delta-hedged rescue doesn't survive the trial either. On simulated chains, hedging the short call's delta (the Israelov–Nielsen risk-managed variant, `delta_hedge` in both engines) was the strongest signal the proxy ever produced: it lifted the MSFT overlay's Newey-West t-stat from 0.46 to 1.63. Re-measured on real quotes, that signal collapses to −0.23 at bid/ask fills and +0.73 at mid (the hedged proxy twin on the same unadjusted series sits at +1.76). The hedge still does its mechanical job — identical 183 calls sold, excess vol cut from 6.64% to 4.80%, \~$101K of the naive loss recovered (net −$82,372) — but the volatility premium it was built to isolate isn't there at real quote levels. The 1.63 was the proxy's inflated premiums talking. (One accounting note: hedge shares are marked on the unadjusted series, so their dividends — roughly $12K over the decade, about the size of the measured hedged excess — go uncredited. The proxy twin shares the omission, so the collapse comparison is apples-to-apples; the absolute hedged figures are modestly understated, and the bid/ask run's negative *sign* sits within that error band. Its t-stat does not: −0.23 vs +1.76 is a gap no dividend credit closes.)

```bash
./fetch_option_data.sh            # pull the chain datasets from the data release (checksum-verified)
python real_cc_backtest.py MSFT   # REAL vs PROXY side by side (also: QQQ)
```

Every number in this section is pinned by `TestMsftRealChainRegression`, `TestMsftRealRiskManagedRegression`, and `TestQqqRealChainRegression` in [test_real_cc_backtest.py](test_real_cc_backtest.py), which CI runs against the same checksum-verified datasets (the simulated-chain 0.46 and 1.63 trace to `TestMsftTenYearRegression` and `TestMsftRiskManagedRegression` in [test_cc_backtest.py](test_cc_backtest.py); the dividend estimate in the accounting note is deliberately unpinned).

## Tests

```bash
pytest test_cc_backtest.py          # the engine suite
pytest test_real_cc_backtest.py     # the real-chain adapter (pins skip without the datasets)
pytest --cov=. --cov-branch         # everything, with coverage
```

CI runs `ruff`, `pyright`, all three test suites (fetching the checksum-verified chain datasets so the real-chain pins run, not skip), a backtest smoke test, a figure-regeneration check, and a notebook drift check on every PR — see [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Project layout

| File | What it is |
| --- | --- |
| [cc_backtest.py](cc_backtest.py#L201) | Backtest engine: Black-Scholes pricing, rolling vol, regime-based IV, day-by-day overlay state machine, Newey-West t-stat reporting on excess returns |
| [test_cc_backtest.py](test_cc_backtest.py#L38) | Unit and scenario tests covering pricing, the overlay state machine, and the statistics helper |
| [real_cc_backtest.py](real_cc_backtest.py#L225) | Real-chain adapter: the same overlay on traded option quotes (bid entries, ask buybacks, real deltas and expirations; the 2008→2010 placeholder-greeks era excluded via `CHAIN_CLEAN_START`), printed REAL vs PROXY |
| [test_real_cc_backtest.py](test_real_cc_backtest.py) | Adapter unit tests (entry selection, era clip + mark clamp, fill models, delta-hedge mechanics) plus the MSFT/SPY/QQQ real-chain regression and walk-forward pins (skip when the datasets are absent) |
| [walk_forward_real.py](walk_forward_real.py) | Walk-forward optimization driving the real-chain adapter (or, via `--prices proxy`, the proxy engine on the same series/windows/calendar-day grid): per-window Pardo trade stats, chained OOS vs fixed-defaults vs buy-and-hold on one convention |
| [docs/prereg_trend_gate.md](docs/prereg_trend_gate.md) | Pre-registration of the trend-gated covered-call experiment (registered at its merge commit): signal, spans, placebo design, two-stage pass rules, pre-committed outcome language |
| [docs/prereg_vol_premium.md](docs/prereg_vol_premium.md) | Pre-registration of the put-side VRP experiment (the follow-up to the pinned call wing): the short-put instrument, SPY span, committed cost band, the Newey-West pass rule, pre-committed outcome language, a committed out-of-sample IWM confirmation arm, and the engine/data changes it needs (registered; the run came back null — §10 amendment) |
| [docs/trend_gate_results.md](docs/trend_gate_results.md) | The experiment's final report: killed at Stage 1 (both mechanism tests wrong-signed, mid-placebo), null verdict per the registration's pre-committed language, both MDE artifacts |
| [docs/explorations.md](docs/explorations.md) | Exploration log — cheap kill-gate scouts on ideas that didn't survive (the post-rip cooldown, the IV-richness gate), pinned so dead ends aren't re-explored; exploratory, not registered verdicts |
| [explorations.py](explorations.py) | The scout code behind the exploration log (reuses the pinned naked runs + fixed seeds), printed via `python explorations.py` |
| [test_explorations.py](test_explorations.py) | Pins the killed scouts' key outputs (wrong-signed statistic, permutation percentile, no-memory measurement) plus always-run cycle/tagging logic |
| [edge_search.py](edge_search.py) | Automated FDR-controlled edge-search harness — two phases + sealed vault: the cheap **re-tag** class (entry-conditioning templates, MSFT+SPY, QQQ sealed; BY-gated) and the **engine-re-run** class (`run_structure_campaign` — short-vol/straddle/condor/strangle/risk-reversal/credit-spread/calendar overlays across seven tickers scored by `short_vol_statistics`' HAC-t asymptotic null, TLT sealed; **e-LOND control**, BY diagnostic) (`python edge_search.py [structure]`) |
| [test_edge_search.py](test_edge_search.py) | Pins both campaigns (Campaign 1 re-tag — no survivor under BY; Campaign 2 structure — 0/56 flagged under e-LOND, the control) plus always-run FDR / enumerator / kill-gate / seal / scale-guard logic |
| [evalue_fdr.py](evalue_fdr.py) | E-value FDR control (interlock #3b, registered in `docs/prereg_fdr_budget.md`): Vovk-Wang p→e calibration + e-LOND over the lifetime ledger stream (online FDR under arbitrary dependence, peek-whenever) + e-BH diagnostic. Ported from the published recurrences, oracle-validated against the `online-fdr` package |
| [test_evalue_fdr.py](test_evalue_fdr.py) | Pins the calibrator / e-LOND recurrence / e-BH against hardcoded `online-fdr` oracle values (always-run) + an optional live-parity check that skips unless `online-fdr` is installed |
| [read_gate_wire.py](read_gate_wire.py) | The dependency-free read-gate wire contract (`WIRE_VERSION`, `BANNED_RESULT_FIELDS`, `assert_numberless`, the field lists) shared by the trusted oracle (`edge_search.score_and_record`) and the sandboxed proposer — one source of truth, so neither side vendors a drifting copy (docs/read_gate.md) |
| [oracle_server.py](oracle_server.py) | Read-gate transport — the TRUSTED oracle: an NDJSON `serve` loop wrapping `score_and_record` (records before replying, returns only the one-bit scoreboard) + `launch`/`launch_in_container`/`prepare_sandbox` that spawn a sandboxed proposer (fail-closed `assert_sandbox_clean`). `launch` is the soft same-machine MVP (cwd, not a kernel jail); `launch_in_container` spawns the proposer INSIDE the sealed image under `CONTAINER_SEAL_FLAGS` (read-only seed mount, no host env), the real seal (docs/read_gate.md) |
| [proposer_client.py](proposer_client.py) | Read-gate transport — the UNTRUSTED proposer half: `build_request` / `run_proposer_loop`, importing only `read_gate_wire` (no engine — pinned by `test_import_is_engine_free`), so it physically cannot recompute a score. The real model author is a later PR (stubbed/injected here) |
| [Dockerfile.proposer](Dockerfile.proposer) | The read-gate **sealed proposer image** — bakes in only the two engine-free files (`proposer_client.py` + `read_gate_wire.py`), no engine/chains/ledger. `oracle_server.launch_in_container` now spawns the proposer INSIDE it under the hardened seal, making the engine + answer key kernel-unreachable (mount namespace) and the network dead (`--network none`) (docs/read_gate.md) |
| [test_read_gate_container.py](test_read_gate_container.py) | Docker-gated CI proof of the container seal: `TestProposerImageSeal` (inside the image the engine is unimportable, the engine source + answer-key ledger unreadable by host abspath, the network unreachable, the proposer's own code still runs engine-free) + `TestContainerRoundTrip` (a live `launch_in_container`↔`proposer_client` round-trip + read-only-mount / no-docker-socket hardening pins). Skips where docker is absent (the dev box); CI fails if docker is missing rather than silently skipping |
| [docs/edge_search.md](docs/edge_search.md) | Edge-search log — Campaign 1 emptied the cheap entry-conditioning class, Campaign 2 the structure class (56 cells: short-vol/straddle/condor + the strangle, risk-reversal, credit-spread, and calendar grammar widenings; with the XLE split-bug catch); exploratory, not registered verdicts |
| [docs/term_backfill_plan.md](docs/term_backfill_plan.md) | Proposed (not executed) pin-safe plan to backfill longer-dated 60–180 DTE call chains so the TERM-family calendar/diagonal can be measured off the 60-DTE data edge; a costed, human-gated data project |
| [docs/read_gate.md](docs/read_gate.md) | Why the file-hiding read-gate for the LLM-proposer answer key is theater (red-teamed: the answer key is a recomputation — `python -c` / `git show` / pinned-test t-stats / self-record all bypass any file-fence), and the process-boundary architecture (sandboxed proposer + recording oracle) that actually enforces honest search; a pinned design dead-end + the real blueprint |
| [docs/llm_proposer_plan.md](docs/llm_proposer_plan.md) | The item-4 (LLM proposer author) design, recorded before any build. The simplification: the model runs **oracle-side and in-process**, so the container/transport is **off the LLM path** (optional infra for a hypothetical code-proposer, not a gate); the seal is a *correctness argument* (numberless prompt + coordinate-only output + every-look-recorded), with `assert_numberless` the load-bearing guard; the activation gate needs redesign (the engine-absent check is sandbox-specific); the training-leak (time-axis-holdout) blocker keeps survivors exploratory; and a cautionary foil (Huang & Fan, arXiv:2603.14288 — the honor-system version, missing the four interlocks). Design only — nothing activated |
| [vol_premium.py](vol_premium.py) | Delta-neutral short-vol VRP engine: the clean isolator (`run_real_short_vol_overlay` for one wing, `run_real_straddle_overlay` for the two-leg ATM straddle, net delta → 0) the covered-call runs never built, plus the defined-risk `run_real_iron_condor_overlay` and the grammar widenings on the one generic engine (`run_real_strangle_overlay`, `run_real_risk_reversal_overlay`, `run_real_credit_spread_overlay` — all exploratory) and `short_vol_statistics` for the rate-invariant delta-hedged-gain t-stat (`python vol_premium.py SPY`) |
| [test_vol_premium.py](test_vol_premium.py) | Always-run delta-neutral mechanics (hedge offsets direction, flat market harvests premium, the audited rf-base netting) plus the pinned real-chain regressions: `TestSpyShortVolRegression` (the call wing, +2.54) and the registered put-side `TestSpyShortPutRegression` / `TestIwmShortPutRegression` (null — gross t +0.20 / +1.00) plus the §7 straddle secondary `TestSpyStraddleSecondary` / `TestIwmStraddleSecondary` (+0.90 / +1.42, also null); and the exploratory call-wing cross-section `TestQqqShortVolRegression` / `TestIwmShortVolRegression` / `TestMsftShortVolRegression` (index-only & cost-fragile: QQQ +2.07 gross, MSFT a −$58K single-name loss); and the exploratory `TestSpyIronCondorExploratory` (defined-risk SPY iron condor — loses vs cash, NW t −1.08) with `TestIronCondorMechanics`; and the put + straddle cross-section on MSFT/QQQ (`TestMsftShortPutExploratory` / `TestQqqShortPutExploratory`, `TestMsftStraddleExploratory` / `TestQqqStraddleExploratory` — put −0.84 / −1.00, straddle −1.36💥 / +0.21) |
| [docs/vol_premium.md](docs/vol_premium.md) | Design + results for the delta-neutral / put-side VRP experiment: why it exists, the call-wing premium (+2.54), the registered put-wing verdict (a null on SPY and IWM, reported against the pre-committed outcome language), and the §7 ATM-straddle secondary (also null) |
| [run_registered_vrp.py](run_registered_vrp.py) | The one-shot registered put-side VRP run (docs/prereg_vol_premium.md): SPY short-put −0.25Δ + IWM out-of-sample at the committed cost band, the cost curve, the §5 verdict, and the §7 ATM-straddle secondary — the analysis code the result cites |
| [trend_gate.py](trend_gate.py) | Analysis machinery for the registered experiment — signal/spans, the seeded placebo-sequence generator, Stage 1 kill-gate tests, Stage 2 placebo families and verdict (checkpointed, resumable) |
| [test_trend_gate.py](test_trend_gate.py) | Pure-logic tests of that machinery plus dataset-gated pins of the registration's signal-side tables (treatment-side only — no outcome data before the registered ordering allows it) |
| [download_prices.py](download_prices.py#L11) | yfinance data downloader |
| [download_option_chains.py](download_option_chains.py) | Alpha Vantage fetcher for per-roll entry snapshots (one target call per monthly roll, six tickers) |
| [download_option_dailies.py](download_option_dailies.py) | Alpha Vantage fetcher for daily chain slices — the datasets `real_cc_backtest.py` consumes; `--keep {both,call,put}` selects the wing(s) (puts added for the put-side VRP experiment) |
| [test_download_option_dailies.py](test_download_option_dailies.py) | Pure-function tests of the fetcher's `filter_chain` (put support, legacy call-only parity, spot inference) |
| [validate_dailies.py](validate_dailies.py) | Data-hygiene validator + `CHAIN_CLEAN_START` proposer — streams a chain store, flags the placeholder-greeks era (constant-mark / lattice-IV / step-delta rows that leave the entry band defect-free), and proposes the clip or fails closed to "UNVERIFIED"; also a price-vs-chain **scale guard** (`scale_ratio` — catches a split-mismatched price file, the XLE case); calibrated to reproduce MSFT's `2010-05-10` (`python validate_dailies.py GLD TLT XLE EEM`) |
| [test_validate_dailies.py](test_validate_dailies.py) | Always-run synthetic pins of the classifier + boundary scan + scale guard, plus dataset-gated calibration (MSFT backfill → `2010-05-10`; GLD/TLT/XLE/EEM → CLEAN; XLE price-vs-chain scale → ~1.0 after the split fix) |
| [fetch_option_data.sh](fetch_option_data.sh) | Pulls the big chain datasets from the `data-2026-06` release and verifies them against [data_checksums.sha256](data_checksums.sha256) |
| [fetch_batch.sh](fetch_batch.sh) | Generic sequential both-wing fetcher (`./fetch_batch.sh GLD TLT XLE EEM`) — the parameterized replacement for the per-batch `fetch_*.sh` wrappers; ensures a price calendar, then retry-wraps `download_option_dailies.py` |
| [publish_dailies.sh](publish_dailies.sh) | Publishes a validated store to the release: gzip → sha256 → `gh release upload` → round-trip verify; prints the ci.yml/cold-storage follow-up (`DRY_RUN=1` rehearses) |
| [onboard_ticker.sh](onboard_ticker.sh) | End-to-end onboarding orchestrator — fetch → validate → **sign-off gate** → publish → single-ticker structure-campaign smoke test (`./onboard_ticker.sh NVDA [--publish]`) |
| [.claude/workflows/onboard.js](.claude/workflows/onboard.js) | The agentic version of that pipeline as a Claude-Code Workflow — clean-gate agent per ticker → structure campaign → triage (kill / adversarially-vetted survivor flag). SAFE (read-only) by default; `args.live=true` adds fetch + auto-apply of **known** repairs (proposed clip, split back-out) + publish. Novel pathologies and survivors always flag to a human |
| [make_figures.py](make_figures.py#L888) | Regenerates the tutorial and blog figures (`fig1`–`fig13`) into `docs/figures/` |
| [make_notebook.py](make_notebook.py#L1) | Regenerates the runnable notebook from the tutorial markdown + figure script |
| [msft_10yr_prices.csv](msft_10yr_prices.csv) | Sample MSFT price data, 2016-04 to 2026-04 |
| [msft_10yr_prices_unadjusted.csv](msft_10yr_prices_unadjusted.csv) / [qqq_10yr_prices_unadjusted.csv](qqq_10yr_prices_unadjusted.csv) | Unadjusted closes (actual traded prices, matching option strikes) for the real-chain runs |
| [tutorial_covered_call_backtest.md](tutorial_covered_call_backtest.md) | Long-form tutorial — theory, math, code walkthrough, and statistical-significance testing |
| [covered_call_backtest.ipynb](covered_call_backtest.ipynb) | Runnable notebook companion to the tutorial — open in Colab via the badge above, or generate locally with `python make_notebook.py` |
| [docs/figures/](docs/figures/) | Generated PNGs embedded in the tutorial; regenerable from `make_figures.py` |
| [requirements.txt](requirements.txt) | Runtime + dev dependencies |

**Where to start:** the [tutorial](tutorial_covered_call_backtest.md) is the source of truth for *why* every part works the way it does (Black-Scholes math, rolling vol, the overlay state machine, walk-forward optimization, robustness checks). For *what* a function actually does, read [cc_backtest.py](cc_backtest.py#L201) end-to-end — it's heavily commented and the link jumps to `run_cc_overlay`, the engine entry point. For the behavior the engine guarantees, see the scenario tests in [test_cc_backtest.py](test_cc_backtest.py#L477) covering the major trade flows: sell + expire OTM, called away, profit-target close, and multi-cycle accumulation.

## Strategy parameters

Edit the `params` dict at the bottom of [cc_backtest.py](cc_backtest.py#L1351):

| Param | Default | Meaning |
| --- | --- | --- |
| `call_delta` | 0.25 | Target delta for strike selection (≈25% chance ITM at expiry) |
| `close_at_pct` | 0.75 | Close when 75% of premium has been captured |
| `dte` | 21 | Days to expiration when opening a new call |
| `risk_free_rate` | 0.045 | Annual risk-free rate used in Black-Scholes |
| `capital` | cost of 1 contract | Total dollars committed; sized into whole 100-share contracts (leftover sits as 0%-yield cash) |

IV is no longer a tunable param — it's derived from rolling 30-day historical vol times a regime-based multiplier (1.1× / 1.3× / 1.5× for high / normal / low vol).

## Caveats

This is an educational backtester, not a production trading system. Notable limitations:

- The engine's IV is estimated, not real — and the [Reality check](#reality-check-real-option-chains) above quantifies the cost: on real chains the proxy's premiums prove 1.55–2.33× too rich, and both flagship results flip from profit to loss.
- No earnings-week avoidance, no dividend handling, no rolling logic.
- Single-stock, single-period results — see the tutorial's robustness section for how to evaluate generalizability.
