# Gap G — the multi-overlay portfolio harness (DESIGN / build spec)

## Status

**Implementation status (2026-07): BUILT, and both verdicts are in** — `common/portfolio.py`
(`align_streams` / `stream_correlations` / `combine_streams` / `max_drawdown_pct`) plus
`portfolio_scout` in search/explorations.py, pinned by `TestPortfolioMechanics` /
`TestPortfolioCombos` in tests/test_explorations.py and logged in
[docs/explorations.md](explorations.md). **Both expected verdicts held; neither combo earned its keep.** Combo A:
the correlation came back low (0.1975 — the different-drivers construction held) but the dead leg
earned nothing — combined NW t 1.54 against the better leg's common-span 2.47, and the combined
percent-of-peak DD (34.31%) sat ABOVE the 50/50 weighted average (24.04%): the dollar-space
subadditivity did not survive the percent transform, because the CC leg's compounding beta dominates
the book's later, larger peaks — the design's own "only approximately" caveat, realized. Combo B:
SPY\~QQQ correlation 0.656 (the shared vol factor), combined NW t 1.01 against best-single 2.56 — the
independent-markets claim killed on this cross-section as pre-stated, though the subadditive DD gap is
visible there (19.53% vs the 27.07% weighted average) when no leg compounds a stock position. Spans
realized: Combo A 2016-04-12 → 2026-04-10 (2,514 days), Combo B 2011-03-24 → 2026-04-10 (3,784 days) —
each one trading day past the design's expected store starts, because the panel is diff-labelled (each
per-capital P&L day carries the LATER date of its diff), squarely inside the design's
endpoints-pinned-at-run-time hedge. The parent plan's two stalenesses are fixed in the same change.

This was written as a **DESIGN document — a build spec, PLAN-level**, ahead of the code. It designs Gap G
from [docs/van_tharp_test_plan.md](van_tharp_test_plan.md): the multi-overlay portfolio harness
(docs/van_tharp_test_plan.md:220-238). The parent plan sizes it Moderate and sequences it last,
independent of the other gaps (:268). It enables **Experiment 6** (:279): Tharp's two portfolio claims —
more independent markets improve a positive-expectancy system, and noncorrelated systems combined
improve the whole. **This is the plan's last gap.** With this document, all seven gaps (A through G)
carry designs and six carry builds; Gap G's build is the last open item.

Predecessors are all in: Gap A ([docs/van_tharp_gap_a.md](van_tharp_gap_a.md), merged #125), Gap D
([docs/van_tharp_gap_d.md](van_tharp_gap_d.md), merged #126), Gaps C+B
([docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md), designed in #127 and built in #128), Gap E
([docs/van_tharp_gap_e.md](van_tharp_gap_e.md), designed in #129 and built in #130), and Gap F
([docs/van_tharp_gap_f.md](van_tharp_gap_f.md), BUILT per its Status with the random-entry measurement
in, docs/van_tharp_gap_f.md:5-18). The streams Experiment 6 consumes already exist.

**Zero engine changes are expected — the Gap F pattern repeated.** Every overlay already returns a
daily equity stream (the three producers are cited in the harness section below); Gap G is pure
consumption — alignment, correlation, combination — plus one committed measurement. The parent plan
already states Experiment 6's input contract: per-overlay `daily_equity` streams aligned on a date
index (docs/van_tharp_test_plan.md:279).

Every number this design will produce is **EXPLORATORY** — sample-spending, kill-or-justify, never a
registered verdict. The work is descriptive research measurement of historical simulation output; it is
not investment advice, and no figure in it is a recommendation to trade any instrument, any
combination, or any weighting.

**Location convention.** Book references are Kindle Locations, per the notes file's own `### Location
N` headers (research/book-notes/trade-your-way-to-financial-freedom.md; Van K. Tharp, *Trade Your Way
to Financial Freedom*, 2nd ed., McGraw-Hill, 2007, Kindle). Two attribution caveats travel with the
citations below. Loc 1307 and Loc 1393 are an interviewed trader's first-person voice — the adjacent
Loc 1303 identifies him as a long-term automated trend follower — so they are attributed to the
profiled trader, not to Tharp. And the phrase "portfolio heat" appears nowhere in the notes file, so
this design does not use it; the closest concepts the notes actually carry are cited instead. Code
file:line references below describe the tree as of the design commit.

## Why — Experiment 6 and the book's two portfolio claims

**Claim (a) — independent markets.** A system with a good positive expectancy generally improves as
more trades are taken per unit time, and adding independent markets is how you get more trades: a good
system performs well in many markets, so "adding many markets simply gives you more opportunity" (Loc
1929). The arithmetic behind it is the expectancy-times-opportunity rule (Loc 3791): judge a game by
expectancy per play multiplied by plays per period — his worked example rates a 0.2R game at 60
opportunities an hour (12R) above a 0.78R game at 12 (9.36R). Two boundary conditions ride the claim,
both pre-stated here because Experiment 6 will test them: the multiplication multiplies negative
expectancy just as readily, and the claim's own premise is a system that performs well across markets.

**Claim (b) — noncorrelated systems.** Performance can usually be improved by combining noncorrelated
systems, each with its own position-sizing model (Loc 1932). His exemplar is a long-term trend follower
paired with a short-term system for consolidating markets: the second makes money when nothing trends,
lessening the first's drawdowns. The subtle point is that a diversifier can earn its place through
variance reduction even when its standalone line is unremarkable — the combination, not the leg, is the
unit being judged.

The supporting passages: prefer relatively independent markets, and avoid a book of correlated
positions that can all move against you at once (Loc 4292). The profiled trend follower keeps per-trade
risk low precisely because he can hold up to 20 markets simultaneously (Loc 1307), and monitors ongoing
risk and volatility at fixed percentages of equity (Loc 1393) — the aggregate-exposure logic a
portfolio harness exists to measure.

### What this repo can and cannot test — scope honesty up front

Claim (a) tests cleanly: the same short-vol system across tickers is exactly the shape the structure
campaign's cross-section already runs, and three of those runs carry pinned regression coordinates.

Claim (b) is bounded by the repo's system diversity. Every overlay here is premium-selling or
equity-long; **no standalone trend system exists.** The trend gate was an entry gate on the covered
call — its question of record was suspending covered-call selling while the underlying is in an
uptrend, holding the shares uncovered (docs/trend_gate_results.md:12-15) — and it was killed at Stage 1
(docs/trend_gate_results.md:3; D_A came back +$439 per trade against a predicted negative, so the real
gate was never even run through the simulator, :52-59). The module carries only experiment machinery —
`run_length_multisets`, `SequencePool`, `run_arm` are trendgate/trend_gate.py's only top-level
definitions (:210, :423, :536) — and a repo-wide grep finds no trend-following overlay function.

The pair claim (b) gets, chosen for return drivers that differ by construction: the **MSFT real
covered call** (the full stream carries the stock leg — realchains/real_cc_backtest.py:510 — so it is
beta-dominated, with a pinned 41.00% max drawdown, tests/test_real_cc_backtest.py:753-755) versus the
**SPY delta-hedged short vol** (market-neutral by construction; its pinned correlation to SPY is +0.26,
tests/test_vol_premium.py:413-414). The drivers differ by construction — equity beta against a hedged
vol premium — and the cross-overlay correlation between them is TO BE MEASURED, not assumed; no such
number exists anywhere in the repo today. Stated plainly: **Experiment 6 tests Tharp's claims on the
pairs this repo owns, not on his trend-plus-consolidation exemplar.**

No harness exists to run either test. The repo-wide sweep for correlation machinery finds four
correlation-adjacent sites, none of which aligns two overlay streams: the factor IC is per-date
cross-sectional Spearman across names (factor/factor_backend.py:101-120), the IV-richness scout's
Spearman pools cycles within one stream (search/explorations.py:344-346), the ACF figure computes
single-series autocovariances (engine/make_figures.py:275), and the closest align-two-series precedent
joins a factor's long-short returns to the registered premia, never overlay equity
(factor/factor_mechanism.py:102-103). The walk-forward's concat chains OOS windows of one strategy
(engine/cc_backtest.py:930). The parent plan's Gap G claim is thereby confirmed — with one stale anchor
noted for the build PR: docs/van_tharp_test_plan.md:230 cites the explorations Spearman at
search/explorations.py:335, and it now sits at :346.

## The harness — three pure functions over existing streams

**Inputs.** The three engines' daily streams, taken as they are. The structure engine emits
`['date', 'equity', 'price', 'rf_credit']` (realchains/vol_premium.py:1053; the row append at
:1049-1050); the real CC engine emits `['date', 'equity', 'price']`
(realchains/real_cc_backtest.py:526; the append at :523); the simulated CC engine emits the same CC
schema (engine/cc_backtest.py:241, the DataFrame build at :524-527). All three round each equity row to
two decimals and store the date as an ISO string, so joins key on the string. All three start from the
same deployed base — cash/capital initialization at realchains/vol_premium.py:839,
realchains/real_cc_backtest.py:312-317, and engine/cc_backtest.py:257-270 — with $100,000 the pinned
convention on every run Experiment 6 touches. v1 consumes the two real-chain engines; the simulated
engine's stream is schema-compatible but out of Experiment 6's scope.

The home is **common/portfolio.py**, a new leaf module beside the Gap A ledger and the Gaps C+B sizer —
the parent plan names the portfolio harness among the measurement primitives these experiments need
(docs/van_tharp_test_plan.md:246-247). Three pure functions:

### align_streams — the inner-join panel

`align_streams({name: daily_equity}) -> panel`: a per-leg daily P&L panel on the **inner join** of
dates. The policy, stated once: the pinned runs have different spans, so they overlap only on their
common era; an inner join measures the combination where all legs actually ran, and the surviving span
is REPORTED with every result the panel feeds.

The span facts that make the policy binding. The SPY short-vol leg is frozen at
`REGISTERED_CLEAN_START['SPY'] = '2010-12-01'` (realchains/real_cc_backtest.py:89; the registered load
at tests/test_vol_premium.py:420) and runs the full 2010-12 → 2026-06 span (:486). The MSFT canonical
store runs 2016-04-11 → 2026-04-10 (verified store min/max; the CC fixture runs the bare store span
with a '2026-06-06' end cap, tests/test_real_cc_backtest.py:734-737), which makes MSFT the binding
right endpoint — its store ends \\~2 months before every other ticker's. The QQQ short-vol pin runs
2011-03-23 → 2026-06-05 (tests/test_vol_premium.py:864-866), and the MSFT short-vol pin runs 2010-05-10
→ 2026-04-10 (:959-961) — both spans start before their canonical stores' first days because both pins
load era backfills, an accounting the scout section below carries. Every leg trades the NYSE calendar,
so inside a joint window the binding differences are span endpoints, not calendar holes; any
leg-specific missing date simply drops from the join — alignment is exact, never interpolated.

### stream_correlations — pairwise Pearson on per-capital daily P&L

The per-leg basis is the `short_vol_statistics` normalization: daily dollar P&L over the FIXED deployed
capital, `np.diff(eq) / capital` (realchains/vol_premium.py:184 — the inline comment reads "FIXED
deployed-capital base (not grown equity)"). The alternative convention — prior-day-equity returns, as
`compute_statistics` uses (`np.diff(equity) / equity[:-1]`, engine/cc_backtest.py:678-679) — is the
wrong basis for a portfolio: each stream's denominator embeds its own compounding history, so a sum of
such returns is the return of no actual portfolio. Dollars add; compounding returns do not. The Gap C+B
doc already contrasts exactly these two conventions (docs/van_tharp_gap_cb.md:84-85); Gap G picks the
additive one.

Why naive cross-stream equity summing lies, in three parts, each grounded in the stream contracts:

- **Mixed denominators.** A 1% day on the MSFT CC's grown \\~$486K equity is a different dollar than 1%
  on a $100K base; per-capital P&L puts every leg on its own committed $100K. Contract counts make the
  same point: identical $100K commitments size different notionals — SPY short vol 8 contracts
  (tests/test_vol_premium.py:438), QQQ 17 (:888), MSFT CC 18 (tests/test_real_cc_backtest.py:748) — and
  the count is span-sensitive (MSFT CC 34 on the extended span, :1084, the same ticker at a lower
  initial price). Normalization is by capital, never by contracts.
- **The stock-leg asymmetry.** CC equity carries the full stock position
  (realchains/real_cc_backtest.py:510) while the structure stream carries none — it is cash plus hedge
  stock plus the structure mark (realchains/vol_premium.py:1039-1042). Summing raw curves stacks stock
  exposure and capital; the per-capital P&L convention keeps each leg a self-contained system instead.
- **The rf asymmetry.** The structure engine accrues daily risk-free interest on cash and records each
  day's credit in the `rf_credit` column (realchains/vol_premium.py:861-865); both CC engines credit
  zero interest — the real engine's hedge financing is an explicit zero-interest simplification
  (realchains/real_cc_backtest.py:293-294) and the simulated engine pins leftover cash at 0% yield,
  consuming `risk_free_rate` only as the Black-Scholes pricing rate (engine/cc_backtest.py:257-270,
  :250). A naive sum silently blends rf-inclusive and rf-free P&L.

**The rf convention, decided:** a structure leg's per-day credit is netted out —
`rf_credit[1:] / capital` subtracted from the diffs, with the same off-by-one `short_vol_statistics`
uses, because the credit lands at the start of the following day (realchains/vol_premium.py:185-189).
The presence of the `rf_credit` column is itself the switch: a leg carrying it is netted, a leg without
it passes through raw — no per-engine flags, so the schema asymmetry is implementable exactly as
stated. This is the pinned +2.54's own basis: the docstring records that netting the ACTUAL per-day
interest makes rf cancel exactly and the verdict rate-invariant, with the identity pinned at
realchains/vol_premium.py:169-171 (the accounting-choice discussion at :148-176). CC legs are already
rf-free by engine construction. Net effect: **every leg is daily P&L above zero-yield cash on its own
$100K**, and the structure legs sit on their published statistical basis. The caveat travels with the
pins: the CC engines' idle cash genuinely earned nothing, an engine simplification the harness inherits
rather than introduces. One inherited day-0 convention is also kept: diffs start at day 1, so each
leg's day-0 entry half-spread (the structure stream's `eq[0]` is already struck at the entry mid,
realchains/vol_premium.py:164-173) stays out of the summed P&L, exactly as `short_vol_statistics`
treats it.

On that basis, `stream_correlations(panel)` is pairwise Pearson over the aligned per-capital series.
Full-sample, descriptive — the regime-conditional variant is a named widening (see cannot-show).

### combine_streams — pre-committed weights only

`combine_streams(panel, weights) -> combined per-capital daily P&L`: the weighted sum of leg
per-capital P&L, weights summing to 1. The interpretation is a $100K book allocating `weight × $100K`
per leg with positions scaled linearly; each leg is normalized by its own deployed `capital` first
(every v1 leg deploys the pinned $100,000), which is what makes the units add. Hand-checked on Combo
A's legs: a day where the SPY leg makes $500 and the CC leg loses $300 combines at 50/50 to
`0.5 × (+0.005) + 0.5 × (−0.003) = +0.001` — +$100 on the $100K book — the identical arithmetic
whether the leg ran 8 contracts or 18, because the divisor is capital, never contracts. The caveat is
named: this is a **linear scaling of measured streams, not a re-run at split capital** —
integer-contract granularity (`int(capital // (initial_price * 100))` at realchains/vol_premium.py:834;
the CC engine's equivalent `int(capital // contract_cost)` at realchains/real_cc_backtest.py:313) is
ignored by construction, and it would bite: one-third of the QQQ leg's 17 pinned contracts is not an
integer.

**Equal weights only in v1 — no weight optimization.** Optimized weights are an in-sample search on the
very numbers being measured; efficient frontiers, vol targeting, or risk parity are a NEW pre-committed
exercise or a campaign, never a quiet extension of this one (the reclassification trigger below).

Consumers of the combined stream:

- `newey_west_summary` for the combined stream's descriptive t — the single home of the repo's
  naive-vs-Newey-West arithmetic (common/stats.py:3-8; `NeweyWestSummary` at :41-49, the function at
  :52), with lag units in the caller's series index, calendar days here (:20-23).
- Max drawdown on the combined cumulative curve, built at the fixed-capital base:
  `capital × (1 + cumsum(pnl))`, drawdown as percent of running peak. Leg drawdowns on the common span
  are computed the same way, so the comparison shares one definition — which also means the common-span
  leg DDs will NOT equal the engines' published full-span equity-curve DDs (41.00% and 4.09% are
  full-span pins on the engines' own curves; new numbers get new pins).
- The legs-vs-combo comparison, defined precisely: the combined max DD against the **weighted average
  of the leg max DDs** on the same span, all under the one fixed-base definition, with the best single
  leg's DD reported beside it. Max drawdowns do not add: on an additive book the dollar drawdown is
  subadditive — combined at or below the weighted average always, with equality only when the legs'
  drawdown windows coincide — so the direction of this comparison is near-automatic and the measurement
  is the **size** of the gap. The percent-of-peak form inherits the dollar inequality only
  approximately, since each curve carries its own running peak.

### The C+B mismatch, named honestly

The marble bag is TRADE-level and a portfolio stream is DAILY, so **the ruin replay does not apply to
the combined stream in v1.** `simulate_sizing` draws per-trade R-multiples from the Gap A ledger and
folds each through `equity *= (1 + fraction * r)` (common/position_sizing.py:50-61, the fold at :121;
"a REPLAY layer over the Gap A ledger", :4), and its input comes from `build_trade_ledger`, which
reduces ONE overlay's event stream to per-trade records (common/trade_ledger.py:150-159). A combined
portfolio curve has simultaneous overlapping positions across overlays and no per-trade R
decomposition, so it cannot enter the bag. The module already names the adjacent limitation — the bag
is IID, and block bootstraps are named widenings (common/position_sizing.py:21-26); a daily-block
bootstrap over the combined per-capital stream is the portfolio analog, a NEW object in that family,
not a reuse. Significance stays with the daily Newey-West t: `ledger_statistics` itself disclaims
significance authority (common/trade_ledger.py:238-243).

### The pandas decision

`common/portfolio.py` will be the first pandas import in the leaf package, and that is acceptable — the
leaf rule is about import direction, not third-party dependencies. common/stats.py's docstring defines
`common/` as the leaf everything else imports without a dependency inversion (common/stats.py:1-11);
nothing forbids external libraries. pandas is a first-class repo dependency (requirements.txt carries
pandas and pandas-stubs), and `daily_equity` is already a DataFrame from all three producers, so a
DataFrame-taking harness inverts nothing. The convention cost is named: today's `common/` interfaces
are stdlib and numpy — the ledger takes list-of-dict trades, the sizer takes sequences and imports no
numpy at all (common/trade_ledger.py:48-54; common/position_sizing.py:38-45) — so this heavies the leaf
slightly, in practice a no-op since every consumer package already imports pandas. The numpy-only
alternative (arrays in, alignment pushed to callers) would preserve the current leaf style at the cost
of re-implementing the date join at every call site; the join is the harness's whole job, so the
DataFrame interface wins.

## Experiment 6 — two pre-committed combos

The measurement lives in the scout home, per Gap F's choose-by-code-home precedent
(docs/van_tharp_gap_f.md:314-329): the orchestration function in search/explorations.py, dataset-gated
pins in tests/test_explorations.py, the verdict entry in [docs/explorations.md](explorations.md).

**The drift alarm comes first.** Before any combo number is computed, the scout re-runs each leg at its
pinned coordinates on its full pinned span and asserts the published pins — the Gap F baseline-re-run
mitigation (docs/van_tharp_gap_f.md:331-335) — then windows the streams to the inner join. The stated
consequence: each leg's common-span standalone statistics are NEW numbers (a t measured on 2016–2026 is
not the pinned full-span t), pinned by the new test, while the published full-span pins stay untouched
(the parent plan's do-not-re-pin rail, docs/van_tharp_test_plan.md:299-300).

### Combo A — the noncorrelated-systems claim (Loc 1932)

The legs, coordinates verbatim from their pins:

- **SPY hedged short call vol** — target_delta 0.25, dte 30, capital 100_000, risk_free_rate 0.045,
  hedge_cost_bps 0.0 (tests/test_vol_premium.py:430-431), span frozen at
  `REGISTERED_CLEAN_START['SPY'] = '2010-12-01'` (:420). Published basis: NW t +2.54 (:451), Sharpe
  0.52 (:452), max DD 4.09% (:444), alpha over cash +$36,495.14 within net P&L +$72,999.90 (:441-443),
  cost-robust to +2.42 at 0.2 bp and +2.25 at 0.5 bp (:470, :406-407), correlation to SPY +0.26
  (:413-414). The frictionless hedge basis is deliberately kept so the leg matches the +2.54 pin.
- **MSFT real covered call** — call_delta 0.25, close_at_pct 0.75, dte 30, risk_free_rate 0.045,
  capital 100_000, bid/ask fills (tests/test_real_cc_backtest.py:108-114), canonical-store span with
  the fixture's '2026-06-06' end cap (:734-737). Published basis: net overlay P&L −$183,552.34 (:749),
  total return 386.26% versus buy-and-hold 569.81%, max DD 41.00% (:753-755), excess NW t −1.73 and
  sharpe_excess −0.49 (:784-785).

The CC leg is the **full system stream** — long stock plus short calls — not the overlay-only excess.
That is a deliberate choice: Tharp's claim (b) combines systems, the CC system is the equity-long one,
and its beta-dominance is the entire point of the pair. Replacing it with the overlay excess would pit
two premium streams against each other and dissolve the question.

Weights: **50/50, pre-committed.** Span: the inner join — expected 2016-04-11 → 2026-04-10, \\~10 years
(the MSFT store endpoints bind on both sides; exact trading-day endpoints are pinned at run time). The
join's cost is named: it spends the SPY leg's first \\~5.4 registered years (2010-12 → 2016-04) plus its
post-2026-04-10 tail to buy the comparison.

Measurements: the pairwise correlation (the repo's first measured cross-overlay correlation); each
leg's standalone NW t and max DD on the common span; the combo's NW t and max DD; combined DD versus
the 50/50-weighted average of the leg DDs.

**The prior, pre-stated so the result cannot be spun.** Expect LOW correlation — MSFT beta versus a
delta-hedged vol premium on a different underlying; the short-vol leg's own correlation to SPY is only
+0.26. Expect drawdown reduction: the 50/50 weighted-average reference is \\~22.5% against the CC leg's
41.00% (both full-span pins on the engines' own curves, used here only to set the expectation — the run
compares common-span values under the one fixed-base definition), and low correlation should pull the
combined DD well under that reference. And expect the drag, stated precisely: the CC leg's overlay
component is a pinned negative (−$183,552.34, excess NW t −1.73) — the leg's positive total return is
MSFT beta in a bull era, not system edge — so a combo win on raw return would be an era artifact, which
is why the pre-committed verdict metrics are risk-adjusted. The open question is exactly Tharp's Loc
1932 point: does variance reduction earn a leg its place on the risk-adjusted metric when its
standalone line does not?

What counts as each answer, fixed before the run: the combined NW t against the better leg's
common-span t (improvement of the whole, or mere averaging), and the combined DD against the
50/50-weighted average of the leg DDs — where subadditivity makes the direction near-automatic, so the
gap's size, not its sign, is the evidence. Both moving favorably supports the claim on this pair; DD
reduction without a t improvement reads as variance reduction that did not earn the dead leg its place;
a correlation that comes back high refutes the different-drivers construction itself. No answer
promotes anything — descriptive, kill-or-justify.

### Combo B — the independent-markets claim (Loc 1929)

The legs: the SAME short-vol system on the three canonical call stores with pinned coordinates, all at
target_delta 0.25, dte 30, capital $100,000, frictionless hedge — the published bases:

- **SPY** — as in Combo A: NW t +2.54, the only cost-surviving positive leg
  (tests/test_vol_premium.py:451, :406-407).
- **QQQ** (EXPLORATORY pin) — span 2011-03-23 → 2026-06-05; gross NW t +2.07, dying at 0.5 bp hedge
  friction (+1.88); alpha over cash +$69,381.23 (tests/test_vol_premium.py:864-866, :896, :899, :891).
- **MSFT** (EXPLORATORY pin, the single-name kill) — span 2010-05-10 → 2026-04-10; frictionless net
  P&L −$48,198.61, alpha over cash −$18,202.17, gross NW t −0.26, max DD 74.58%
  (tests/test_vol_premium.py:959-961, :988-990, :994, :997-998). One figure trap flagged now: the class
  docstring's "net P&L −$58K" is the net-0.5bp narrative number (:959-961), not the frictionless pin.

Weights: **one-third each, pre-committed.** Span: the inner join — expected 2011-03-23 → 2026-04-10,
\\~15 years (QQQ's backfill start binds on the left, MSFT's store end on the right; exact endpoints
pinned at run time). The join's cost is small here: SPY loses its 2010-12 → 2011-03 head, MSFT short
vol its 2010-05 → 2011-03 head, and every leg the post-2026-04-10 tail.

Measurements: the 3×3 correlation matrix (three pairwise values — the repo's first cross-market
correlation numbers); the combined NW t and max DD versus the best single leg's common-span values.

**The prior, pre-stated.** The three streams sell the same vol premium and share the vol factor —
expect substantial positive pairwise correlation and at best a modest diversification benefit. These
markets are not that independent, and that is the honest expected verdict against Loc 1929's framing on
THESE markets. The claim's own precondition also fails on one leg: Loc 1929 presumes a system that
performs well across markets, and the same coordinates on MSFT are a pinned negative — Loc 3791's
arithmetic multiplies whatever expectancy the market supplies, negative included. So the expected
verdict is a combined t below SPY-alone, with the measurement quantifying how much correlation and one
negative leg each cost. What counts: combined NW t above the best single leg supports the claim despite
the drag (pre-stated as the surprising outcome); combined at or below the best single leg kills the
claim on this cross-section. Either way the matrix gets pinned.

### What Experiment 6 cannot show

- One era per combo, each a single mostly-rising span; nothing regime-conditional. The correlations are
  full-sample descriptive — regime-conditional correlation, the crisis-convergence problem of
  correlations rising in stress (the Loc 4292 warning), is a named widening via Gap D's six-regime map
  ([docs/van_tharp_gap_d.md](van_tharp_gap_d.md), merged #126).
- Two and three legs at pre-committed equal weights; no frontier, no weight claim, no sizing claim.
- No significance claim on DD differences — DD comparisons are point-descriptive gap sizes; only the NW
  t carries a significance shape, and it stays descriptive throughout.
- Per-capital units only, with integer-contract granularity ignored by the linear combination.
- Combo A's CC leg conflates MSFT beta with overlay P&L by construction — that is the design, but no
  conclusion about the CC overlay alone follows; the overlay-only excess is a different stream with its
  own pinned verdict (−1.73, tests/test_real_cc_backtest.py:784-785).
- No ruin replay on the combined stream (the C+B mismatch above); the daily-block variant is a named
  widening, not a v1 deliverable.

## Not an FDR search — the boundary argument, reused

Gap E set the precedent that this boundary is an argument, not a label
(docs/van_tharp_gap_e.md:350-359), Gap F reused it (docs/van_tharp_gap_f.md:286-312), and it transfers
whole a third time:

1. **Unit of account.** The e-LOND stream counts grammar cells — `StructureCandidate`s recorded to the
   committed `idea_ledger.jsonl`. A portfolio combination constructs no `StructureCandidate`; legs and
   weights are not grammar coordinates (`_validate_grammar` raises on any unknown knob), so nothing
   here can enter the stream even by accident. Nothing enters `idea_ledger.jsonl`; no e-value is spent.
2. **Look count.** Two pre-committed combos with fixed legs, fixed weights, and fixed spans (the inner
   joins), every outcome pinned regardless of verdict. The correlation matrix is a measurement, not a
   hypothesis menu. The parent plan's e-LOND rail names the Experiment 1 and 2 sweeps, not Experiment 6
   (docs/van_tharp_test_plan.md:294-296); this design keeps Experiment 6 outside by pre-commitment
   rather than by omission.
3. **The reclassification trigger, explicit.** Weight sweeps, leg shopping — adding, dropping, or
   substituting legs after seeing results — span shopping, or selecting which combo to report: any of
   these makes the exercise a campaign under the e-LOND lifetime stream like any other automated
   search.

## Code home and pinning home

**The harness: common/portfolio.py.** Three pure functions, no I/O, no engine imports — a measurement
primitive in the trade_ledger / position_sizing family, with the pandas decision argued above.

**The measurement: search/explorations.py**, per Gap F's choose-by-code-home rule
(docs/van_tharp_gap_f.md:314-329) — Experiment 6 needs an orchestrator (leg re-runs, windowing, the two
combos), and orchestration code goes where scout code goes. One operational constraint is designed in
now: the scout loads chain stores **strictly sequentially**, retaining only each leg's `daily_equity`
and summary and releasing the store before the next load. The CI budget is one chain store at a time on
a \\~7 GB runner (.github/workflows/ci.yml:24-26; the 2026-06-24 CI-perf measurements put stores at
\\~2.3 GB), and Combo B touches three tickers' stores. Runtime is not a concern on Gap F's measured
figures — a full scout ran in \\~20 seconds with the store load dominating and an engine pass at
\\~0.05 s (docs/van_tharp_gap_f.md:16) — so Gap G's four sequential leg loads plus four engine passes
land in the same tens-of-seconds class. The accounting, since it is not one-load-per-ticker: SPY's
single short-vol run serves both combos; the QQQ pin loads the canonical store merged with its 2011
backfill (its 2011-03-23 span start predates the canonical store's 2016-06-06 first day); and MSFT
loads twice — the bare canonical store for the CC pin, the canonical plus the 2008–2016 era backfill
for the short-vol pin whose 2010-05-10 start the canonical store (first day 2016-04-11) cannot supply.
The Gap F procedural rule applies anyway: measure the bucket when the pins land.

**The pins: tests/test_explorations.py**, both layers — the always-run synthetic mechanics and the
dataset-gated Experiment 6 class — riding the trend-explore CI bucket unchanged (tests/test_trend_gate.py
+ tests/test_explorations.py + tests/test_trade_ledger.py, .github/workflows/ci.yml:192-195, the
lightest of the three scout buckets; wall-clock is bounded by the slowest bucket, \\~test_vol_premium,
:176-178). The alternative — a dedicated tests/test_portfolio.py — would force a ci.yml include edit
for one new file; riding test_explorations.py keeps ci.yml untouched, at the named cost that a
`common/` module's mechanics tests live in the explorations file. If the build prefers the dedicated
file, the ci.yml edit lands in the same change.

**The log:** a docs/explorations.md entry in the idea / how-tested / verdict / trap shape, whatever the
verdict says.

## Test plan

Always-run synthetic tests (hand-built two- and three-stream panels, no dataset):

- Alignment: two crafted streams with mismatched spans inner-join to exactly their overlap, the
  returned span matches the surviving dates, and a leg-specific missing day drops from the join.
- Correlation exactness: a crafted co-moving pair measures +1.0, an anti-moving pair measures −1.0, and
  an orthogonal pair measures 0.0, exactly.
- The rf convention: a synthetic stream carrying an `rf_credit` column has `rf_credit[1:]` netted with
  the off-by-one honored (the realchains/vol_premium.py:185-189 model), and a stream without the column
  passes through raw — the column's presence is the switch.
- Combination: a 50/50 combine of a tiny panel reproduces hand arithmetic row for row, and weights that
  do not sum to 1 are rejected.
- Drawdown: the max DD of a crafted V-shaped combined curve equals the hand-computed value, and a
  crafted anti-correlated pair shows combined DD strictly below the weighted average of the leg DDs —
  the subadditivity fixture that makes the gap-size comparison concrete.

Dataset-gated Experiment 6 pins (proposed class name `TestPortfolioCombos`, final at build time):

- The four leg re-runs reproduce their published pins before any combo number is computed: SPY +2.54
  (tests/test_vol_premium.py:451), QQQ +2.07 (:896), MSFT short vol −0.26 (:994), and the MSFT CC
  −$183,552.34 with excess NW t −1.73 (tests/test_real_cc_backtest.py:749, :784-785).
- Combo A: the pairwise correlation, each leg's common-span NW t and max DD, the combo's NW t and max
  DD, and the realized span endpoints.
- Combo B: the three pairwise correlations, the combined and best-single NW t and max DD, and the
  realized span endpoints.

Pre-committed in this document: the legs and their coordinates (verbatim above), the weights (50/50 for
Combo A; one-third each for Combo B), the spans (each combo's inner join), and the basis conventions
(per-capital daily P&L, structure legs rf-netted, CC legs raw, frictionless hedge matching the pinned
bases). Unlike Gap F there is no RNG anywhere in v1, so there is no seed constant to commit — the
pre-commitment IS the leg/weight/span block.

## Cross-surface obligations (when code lands, not now)

- Symbol-sweep regex: add `align_streams|stream_correlations|combine_streams` plus the test class names
  (final at build time) in the same change that lands the symbols, per the CLAUDE.md rule.
- README: project-layout rows for common/portfolio.py and this doc; the explorations trio's rows
  already exist.
- ci.yml: no edit expected — the new tests ride tests/test_explorations.py, already on the
  trend-explore include line (.github/workflows/ci.yml:192-195). The dedicated-test-file alternative
  carries the ci.yml edit with it if the build takes it.
- Notebook: no regen — nothing here touches tutorial_covered_call_backtest.md or
  engine/make_figures.py.
- `STRUCTURE_ENGINE_VERSION`: unchanged, with the reasoning stated in the PR — zero engine edits, so
  nothing scored can move.
- docs/explorations.md: the verdict entry lands with the pins, whatever the verdict is.
- Two test-plan stalenesses folded into the build PR: docs/van_tharp_test_plan.md:230's anchor for the
  explorations Spearman (search/explorations.py:335) now points at :346; and the plan's What's-missing
  grep list (:227-230) includes `combine_streams`, a name this build makes real — that sentence gets
  past-tensed in the same sweep.

## Honesty rails

- **Exploratory throughout.** Every number is sample-spending, kill-or-justify; both priors are
  pre-stated above so no result can be narrated after the fact; a surprising pass earns a pre-registration,
  never a headline.
- **The look count is bounded by pre-commitment.** Two combos, fixed legs, fixed weights, fixed spans,
  all outcomes pinned regardless of verdict. Weight sweeps, leg shopping, or span shopping reclassify
  the work as a campaign under the e-LOND stream.
- **No FDR interaction in v1.** Nothing enters `idea_ledger.jsonl`.
- **One significance authority, used descriptively.** `newey_west_summary` over per-capital daily
  streams carries every t; the ledger statistics never gate (their own posture,
  common/trade_ledger.py:238-243); DD comparisons carry no significance claim at all.
- **Do not re-pin.** Common-span leg statistics are NEW pins beside the new code; every published
  full-span pin stays untouched (docs/van_tharp_test_plan.md:299-300), and the scout's leg re-runs
  alarm on drift instead of absorbing it.
- **Convention caveats travel with the pins.** The inner-join span rides every number. Units are
  per-capital on a fixed $100K base, never contracts, never compounding returns. Structure legs are
  rf-netted to their published basis; CC legs are zero-interest by engine construction. The combination
  linearly scales measured streams and ignores integer-contract granularity. The DD comparison reports
  a gap size against a subadditive reference, not a significance verdict. Combo A's CC leg carries
  MSFT beta by design.
- **Scope honesty.** The pins say what was combined: the pairs this repo owns — premium sellers and one
  equity-long CC — at equal weights on one era. Not Tharp's trend-plus-consolidation exemplar, which
  this repo cannot field (no standalone trend system exists; the trend gate was an entry gate, and it
  was killed).
- **The disclaimer is stated once, in Status,** and covers every number this design will produce.

## Open questions

1. **rf handling in the correlation basis.** Leaning: as decided above — net the structure legs'
   `rf_credit` (the pinned +2.54's own excess basis, realchains/vol_premium.py:185-189) so every leg is
   P&L over zero-yield cash, with CC legs already rf-free by construction
   (realchains/real_cc_backtest.py:293-294). The raw alternative avoids the column-keyed behavior but
   blends rf-inclusive and rf-free P&L in the combined stream and takes the structure legs off their
   published basis. The practical asymmetry, named: the slow rf drip barely moves a daily Pearson, but
   it materially moves the combined stream's mean and t — the netting matters for the verdict metric
   more than for the matrix.
2. **Weights beyond equal.** Leaning: **v1 equal only.** Any optimized weighting is an in-sample search
   on the measured streams; frontiers, vol targeting, and risk parity are each a NEW pre-committed
   exercise or a campaign under the stream — named widenings, not v1 riders.
3. **Whether Combo B includes GLD/XLE/EEM.** Leaning: **no.** Three legs with pinned regression
   coordinates suffice for the claim's first measurement, and each has a published baseline the drift
   alarm can assert. The campaign tickers' short-vol cells exist only as exploratory ledger rows, so
   adding them multiplies the look surface without a pinned baseline to alarm against. A wider
   cross-section is a named widening.

## Related

- [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the parent plan whose Gap G section
  (:220-238), build-order row (:268), and Experiment 6 row (:279) this doc designs; this is the plan's
  last gap — all seven now carry designs, six carry builds
- [docs/van_tharp_gap_f.md](van_tharp_gap_f.md) — the zero-engine-change pattern, the
  choose-by-code-home rule, the boundary argument, and the measured runtime this design leans on
- [docs/van_tharp_gap_e.md](van_tharp_gap_e.md) — the boundary-is-an-argument precedent
- [docs/van_tharp_gap_cb.md](van_tharp_gap_cb.md) — the returns-convention contrast (:84-85) and the
  trade-level replay the C+B-mismatch section keeps out of v1
- [docs/van_tharp_gap_a.md](van_tharp_gap_a.md) — the trade-level ledger whose per-overlay scope is
  exactly what the daily portfolio stream cannot reuse
- [docs/van_tharp_gap_d.md](van_tharp_gap_d.md) — the six-regime map behind the
  regime-conditional-correlation widening
- [docs/explorations.md](explorations.md) — the scout pattern and the log-entry home for the verdict
- [docs/trend_gate_results.md](trend_gate_results.md) — the killed entry gate that bounds claim (b)'s
  testable pairs
- [docs/edge_search.md](edge_search.md) — the campaign and e-LOND rules the boundary section respects
- research/book-notes/trade-your-way-to-financial-freedom.md — the highlights file every Location above
  cites
