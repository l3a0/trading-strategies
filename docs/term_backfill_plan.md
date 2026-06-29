# Far-DTE backfill plan — measuring the TERM family off the data edge

**Status: EXECUTED 2026-06-23.** The fetch ran, the backfill is published to the `data-2026-06` release, and the calendar was re-measured at `far_dte=90`. This document is the original plan of record — the **Why** and **What gets fetched** sections below describe the *pre-backfill* state and the *intended* approach (a couple of details changed at build time; see Outcome). For the live result see [`docs/edge_search.md`](edge_search.md) (Widening 4) and the Outcome section immediately below.

## Outcome (executed 2026-06-23)

- **Fetched:** `pipeline/download_option_dailies.py --max-dte 180 --keep call` for all 8 tickers (MSFT/SPY/QQQ/GLD/XLE/EEM/NVDA + sealed TLT) into `{ticker}_option_dailies_180dte.csv`. Calls only, the **full** `1 <= DTE <= 180` band — a superset of the canonical `<= 60`, not the `60 < DTE <= 180` slice this plan first sketched. \~23M rows / 2.1 GB raw, 480 MB gzipped.
- **Published:** gzip → 8 sha256s appended to `data/data_checksums.sha256` → uploaded to the `data-2026-06` release → added to both `ci.yml` cache lists → round-trip verified → copied to cold storage. CI fetches them through the same `*_option_dailies*.csv.gz` glob as every canonical store.
- **Wired (pin-safe):** `_far_chain_paths` / `_load_ticker_data(include_far=True)` merge the far store ONLY into the TERM (calendar) data path; `_far_store_sha` folds its `.gz` checksum into the calendar lineage and only the calendar's. The committed calendar template moved `far_dte=60 -> 90`. Regenerating the ledger moved exactly the 7 calendar `data_lineage_hash`es; all 49 single-expiration cells stayed byte-identical, and the full suite (207 tests) passed.
- **Result:** all 7 calendar cells now trade (MSFT is no longer `measurement_invalid`): MSFT −0.45 / SPY −3.02 / QQQ −1.80 / GLD −4.24 / XLE −0.12 / EEM −2.47 / NVDA +0.67 (the lone positive, p\~0.25). Verdict unchanged: **0/56**. The honest expectation below held — better-measured, still a clean null.

## Why

The real-chain datasets were fetched with `pipeline/download_option_dailies.py --max-dte 60` — the chains stop at **60 DTE** on every ticker, because they were captured for the \~30-DTE entry-band experiments (short vol, covered call), which never needed anything longer.

The calendar (the first `TERM`-family structure) is the first overlay to ask for a far leg, and it asks right at that edge. Its far leg needs `DTE >= near_dte + min_gap_dte = 30 + 30 = 60`, so:

- The only reachable geometry is `near=30 / far=60` (gap = the 30-day floor), and `far_dte=90` in the grammar is **literally unreachable**.
- A 60-DTE expiry exists on only \~4–6% of days. The six traded calendar cells scrape that thin tail; **MSFT** falls below the one-entry threshold (its \~30-day roll cadence never lands on enough 60-DTE-expiry days), so it flags `measurement_invalid`.

So the TERM verdict on the current chains is a **clean null on thin data**, not a fully-powered test. This is a dataset-wide limit, not an MSFT bug.

## What gets fetched

- **Endpoint reality.** `HISTORICAL_OPTIONS` returns the *entire* chain for a date in one request; `--max-dte` is only a client-side filter in `filter_chain`. Re-fetching a day to grab far legs is the **same one request/day** as the original fetch — we just keep a wider DTE band.
- **Scope.** Re-run `pipeline/download_option_dailies.py --max-dte 180 --keep call` per ticker into **new** files `{ticker}_option_dailies_fardte.csv`, keeping only `60 < DTE <= 180` (the canonical store already holds `<= 60`). Calls only — the calendar is a call structure, so no puts re-fetch.
- **Tickers.** The seven search names (MSFT/SPY/QQQ/GLD/XLE/EEM/NVDA) plus TLT (sealed, for symmetry).

## Cost / time

- **Volume:** \~25,000–28,000 day-requests (\~8 tickers x \~2,500–4,000 trading days).
- **Wall-clock:** \~**6–8 hours**, sequential one-ticker-at-a-time (the standing fetch preference), at the premium rate (\~75 req/min, 0.85 s sleep) plus retry-loop overhead.
- **Dollar cost:** Alpha Vantage premium is subscription-based, not per-request — the cost is the **rate budget + time**, not incremental dollars (assuming the subscription is live). Storage is modest (the 60–180 band is a slice of each chain).

## Pin-safety (the binding constraint)

The canonical `{ticker}_option_dailies.csv` files are pin-protected — **never appended to**. The backfill lives in separate files and is merged at load, mirroring the `_put_chain_paths` pattern, with one rule:

- **Merge the far backfill ONLY into the TERM-family overlays' data path** (calendar / future diagonal). The single-expiration structures (short vol, straddle, strangle, iron condor, risk reversal, credit spread) load the canonical store **unchanged**, so their lineage, checksums, and every pinned number stay **byte-identical and untouched** — zero blast radius on the existing suite.
- Every far leg is DTE > 60 and every near-band selector targets \~21–45 DTE, so even a merge-for-all would leave near selections byte-identical in *value*. Isolating the merge to TERM overlays also keeps their *lineage* clean. The exact mechanism (a TERM-only second store vs. a conditional merge) is settled at build time and verified by the full suite asserting nothing but the calendar moves.
- **What re-measures (expected, human-signed):** the 7 calendar cells. The 6 currently-trading cells get better/longer far legs; **MSFT becomes a real measurement** instead of `measurement_invalid`. Re-pin `TestStructureCampaign`'s calendar rows + the ledger's calendar rows.

## Build + re-pin steps (after the fetch lands) — all completed 2026-06-23 (see Outcome)

1. `_far_chain_paths` (analog of `_put_chain_paths`) + the TERM-only merge in the data path.
2. Optionally move the committed calendar template `far_dte=60 -> 90` to sit off the data edge — a grammar/template decision (re-measures the calendar; decide after seeing the fetched data).
3. `python -m search.edge_search structure --record` → re-measures the 7 calendar cells (56 rows, calendar lineage re-recorded).
4. Re-pin `TestStructureCampaign`; update `docs/edge_search.md` Widening-4 section (calendar trades on all 7, new per-cell t-stats, drop the MSFT-invalid note); sweep `CLAUDE.md` if counts shift.
5. **Publish** the backfill files through the full ceremony: `gzip -9` → sha256 to `data/data_checksums.sha256` → `gh release upload` → add to the `ci.yml` cache list + confirm the `_fardte` suffix matches the `*_option_dailies*.csv.gz` fetch glob → round-trip verify → cold storage.

## The honest expectation

The calendar is currently `0 / 56`, and all six traded cells are wrong-signed. **The backfill is very likely to keep it a null** — better-measured, still no harvestable term-structure edge on these large-caps. The value is a *clean, defensible* measurement of the TERM family (and unlocking the diagonal), **not** an expectation of finding alpha. The work is framed that way so a marginal result is not motivated-reasoned into significance.
