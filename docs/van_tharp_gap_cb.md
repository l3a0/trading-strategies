# Gaps C+B — position sizing + the marble-bag resampler (DESIGN / build spec)

## Status

**DESIGN document — a build spec, PLAN-level, no code yet.** It designs Gap C (the position-sizing layer)
and Gap B (the trade-level resampler) from [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) as one
change, because they are one mechanism: the marble bag replays trades *through* a sizing rule. Together
they enable **Experiment 1** — the risk-of-ruin / terminal-wealth Monte Carlo. Predecessors, both merged:
Gap A ([docs/van_tharp_gap_a.md](van_tharp_gap_a.md), the R-multiple ledger) and Gap D
([docs/van_tharp_gap_d.md](van_tharp_gap_d.md), per-regime R-distributions).

Every number this design will produce is **EXPLORATORY** — sample-spending, kill-or-justify, never a
registered verdict — and Experiment 1 produces **descriptive risk distributions, not edge claims**: no
hypothesis is flagged, nothing enters the idea ledger, and the daily Newey-West t stays the repo's sole
significance authority (the same posture `common/trade_ledger.py` pins in its module docstring,
common/trade_ledger.py:9-13). This work is descriptive research measurement of historical simulation
output; it is not investment advice, and no figure in it is a recommendation to trade any instrument or
any size.

**Location convention.** Book references are Kindle Locations, per the notes file's own `### Location N`
headers (research/book-notes/trade-your-way-to-financial-freedom.md; Van K. Tharp, *Trade Your Way to
Financial Freedom*, 2nd ed., McGraw-Hill, 2007, Kindle). The parent plan cites the two marble-bag passages
by notes-file line number ("Loc 671, 747"); their Kindle Locations are 3728 and 4098 — the same passages.

## Why — Experiment 1 and the thesis it tests

Position sizing is the lesson Tharp weights far above system development. The ranking he reports puts
psychology at roughly 60 percent, position sizing at roughly 30, and system development at roughly 10
(Loc 525), and his practice mandate is to spend more time on the sizing part of a system than on
everything else combined (Loc 4537). The book's foreword compresses the survival half of the claim: bet
too big and you blow out, and a low-risk idea is one traded at a risk level that survives worst-case
contingencies long enough to realize the system's expectancy (Loc 184 — the foreword's words, not Tharp's
chapter text). Tharp's own formulation matches: a methodology must be traded at a sizing level that
protects against the worst short-run conditions while still letting the long-term expectancy arrive
(Loc 1038).

The instrument he uses to teach this is the marble bag. Represent each of a system's R-multiples as a
marble; draw with replacement; watch equity. He reports running 10,000 simulated 20-trade months against
one R-multiple distribution and finding it lost money in \~12 percent of months (Loc 3728), and he
prescribes the same exercise to readers: draw out a year's worth of trades, replacing each marble, at
least 100 times (Loc 4098). His workshop version makes the sizing point directly — a bag of seven 1R
losers, one 5R loser, and two 10R winners (expectancy +0.8R despite 80 percent losers), where every
player faces the identical 40 draws and chooses only the bet size, ends with equities ranging from
bankrupt to up 1,000 percent (Loc 8233). He attributes that spread, citing Brinson, to the how-much
decision dominating performance variance.

That is why Gaps C and B are one design. The bag without a sizer only shuffles outcome order; a sizer
without the bag replays a single realized sequence. Experiment 1 needs the distribution of equity paths
under a sizing rule — the bag drawn through the sizer, which is exactly Tharp's object.

## The central design decision — a replay layer, not an engine change

The sizer does not enter the engines' hot loops. Fixed-fractional sizing is applied by **replaying the
ledger's R-multiple sequence**: risking fraction `f` of current equity on a trade that returns `r`
R-multiples multiplies equity by `(1 + f*r)`. That is the standard fixed-fractional identity, and it is
the reason Tharp expresses systems as R-distributions in the first place — normalizing each trade by its
declared initial risk (`r_multiple = pnl / initial_risk`, common/trade_ledger.py:84) makes sizing
separable from the system, so the how-much question can be answered after the fact from the ledger alone
(Loc 3761: expectancy is realized long-term only if positions are sized to equity).

Consequences of the replay design:

- **Zero pin risk.** No engine file is edited at all, so no pinned regression can move. Contrast Gap A's
  A2 phase, which had to touch all three daily mark loops to thread MAE
  (docs/van_tharp_gap_a.md, "Phased plan").
- **The engines keep their flat-notional convention.** All three size once from `prices[0]` before the
  daily loop and never re-size: `num_contracts = int(capital // contract_cost)` and
  `shares = 100 * num_contracts` at engine/cc_backtest.py:258-264, realchains/real_cc_backtest.py:313-316,
  and realchains/vol_premium.py:825-828, with the loops starting at :308, :338, and :848 respectively.
  Position exposure therefore never compounds — each trade's dollar P&L comes off the same contract
  count. One denominator nuance: `short_vol_statistics` measures returns against constant capital
  (`np.diff(eq) / capital`, the "FIXED deployed-capital base" comment, realchains/vol_premium.py:183),
  while `compute_statistics` divides by prior-day equity (engine/cc_backtest.py:678-679) — but the
  flat-exposure fact holds in both. The engines' convention is the constant-dollar-risk baseline (a fixed
  dollar R every trade, P&L adding rather than compounding); the replay generalizes it to risk that
  scales with equity.
- **The whole of C+B lands in one new leaf module plus tests.** No cross-package surgery.

The price for the replay design is that sizing can never feed back into trade selection. A rule that
skips entries after a drawdown, or resizes an open position, changes the trade sequence itself and would
need the engine. The replay is valid today precisely because the engines' entry and exit decisions never
read equity — they are price- and chain-driven, and equity only accumulates P&L. Equity-feedback sizing
rules are out of scope here and would be a separate, engine-touching design.

## Replay validity preconditions

1. **The overlays trade one position at a time, so a single equity path is well-defined.** Every engine
   holds one scalar position state, gates entry on it being empty, and resets it at each terminal:
   `if position is None:` at engine/cc_backtest.py:330 (state declared :287; resets :408, :435, :461) and
   realchains/real_cc_backtest.py:341 (declared :324; resets :434, :482), and `if legs is None:` at
   realchains/vol_premium.py:859 (declared :832; resets :930, :955 — the staggered branch at :895 settles
   the near leg of the one open calendar, never a second entry). The ledger reducer mirrors the same
   strict alternation with a single `entry` slot cleared after each terminal
   (common/trade_ledger.py:178-186, :216).
2. **Per-unit P&L is treated as size-invariant.** No market-impact model: replaying at a larger size
   assumes fills at the same prices. At retail scale on liquid listed options this is a reasonable
   simplification, and it is stated rather than hidden.
3. **Positions are treated as perfectly divisible.** Real contracts are integers — the engines' own
   floor-division sizing shows the granularity. An integer-rounding variant of the replay is a possible
   widening, not in scope.
4. **The bag is IID, and that caveat stays loud.** Drawing with replacement destroys serial dependence,
   including regime persistence. Gap D showed outcomes cluster by regime (docs/van_tharp_gap_d.md — the
   pinned per-cell table; SPY's largest cell, bull_quiet, holds 93 of its 174 trades), and clustered
   losses mean IID resampling can **understate** drawdown and ruin risk. Tharp's bag is IID by
   construction (Loc 3728, 4098) — this design implements his object and labels the limitation. Two
   widenings are named now: a block bootstrap, and a per-regime bag drawing from Gap D's cell
   distributions. One empirical wrinkle: on the MSFT ledger the trade-level HAC t is slightly larger in
   magnitude than the naive SQN (−2.797 vs −2.70, tests/test_trade_ledger.py:356-357), consistent with
   mild negative short-lag autocovariance in trade order there — so the direction of the IID bias is not
   assumed, only not modeled.

## Module and API — `common/position_sizing.py`

A new stdlib-only leaf module, sibling of `common/trade_ledger.py`, with the same dependency direction
(`common/` imports nothing above `common/` — the rule the ledger's docstring states,
common/trade_ledger.py:15-23) and the same import discipline (the ledger's imports are stdlib plus the
sibling `common.stats`, common/trade_ledger.py:48-54; the sizer needs only `random`, `math`, and
`statistics`). Its module docstring carries the same epistemics language the ledger pins: measurement
substrate only, every output exploratory, reported never a gate.

### `simulate_sizing`

```python
def simulate_sizing(
    r_multiples,            # the bag: a ledger's r_multiple column (or any R list)
    *,
    fraction,               # f: fraction of current equity risked per trade
    n_paths=10_000,
    n_trades=None,          # None -> len(r_multiples): one same-length career
    seed=42,
    ruin_threshold=0.5,     # fraction of starting equity
    mae_r=None,             # optional parallel MAE-R column -> intratrade ruin
) -> dict
```

Each path draws `n_trades` R-multiples with replacement and compounds equity from 1.0 via
`equity *= (1 + fraction * r)`. Defaults and their sources:

- `n_paths=10_000` matches Tharp's own repetition count in the 20-trade-month exercise (Loc 3728).
- `n_trades=None` replays one career the same length as the input ledger; a fixed horizon is available
  through the parameter (open question 3).
- `seed=42` drives a stdlib `random.Random(seed)` — the `monte_carlo_shuffle` convention
  (engine/cc_backtest.py:1182, :1217), chosen over the search-side numpy convention
  (`np.random.default_rng(20260613)`: `PERMUTATION_SEED` at search/explorations.py:74, `CAMPAIGN_SEED`
  at search/edge_search.py:104) because this is a descriptive, engine-adjacent Monte Carlo like the
  shuffle test, not a kill-gate permutation null — and stdlib `random` keeps the module numpy-free.

Proposed output shape (a dict, exact keys settled at build time): `terminal` percentiles
(`median`, `p10`, `p90`), a max-drawdown distribution (`median`, `p90`, `worst`), `p_ruin`,
`p_negative_terminal` (the fraction of paths ending below starting equity), and a `ruin_basis` label.

Ruin accounting is three-tiered:

- **Close-only (default).** With `mae_r` absent, a path is ruined if post-trade equity ever falls below
  `ruin_threshold` of starting equity. The output labels itself `close_only`.
- **Intratrade (when `mae_r` is supplied).** Each draw carries its trade's MAE-R alongside its R, and the
  trough `pre_trade_equity * (1 + fraction * mae_r)` is what tests the threshold — tested before the
  close multiplication `equity *= (1 + fraction * r)` is applied. Because `mae <= min(pnl, 0)` by
  construction (common/trade_ledger.py:206), the trough never sits above the close equity, so testing
  trough-then-close cannot miss a breach. The trough enters only the ruin test; the max-drawdown
  distribution is computed peak-to-trough on post-trade close equity in both modes. Gap A's MAE column is
  what makes intratrade ruin measurable — a trade that recovers by the close can still have killed the
  account at its worst mark. The column is the Sweeney-convention worst intratrade unrealized P&L
  (`mae <= 0`, `mae_r = mae / initial_risk`; common/trade_ledger.py:85-86, :206, docstring :40-45), and
  it inherits MAE's stated conventions: daily closing marks (so true intraday excursions are understated)
  and stale-mark carry-forward on the real path.
- **Absorption.** Fixed-fractional equity is multiplicative and never literally reaches zero **unless**
  a single draw has `f*r <= -1` (or an intratrade trough has `f*mae_r <= -1`). That is possible here:
  the pinned worst intratrade excursion on the SPY short-vol ledger is −11.41R
  (tests/test_trade_ledger.py:372), so any `f` at or above \~8.8 percent (1/11.41) turns that one draw
  into a wipeout. A path hitting equity `<= 0` is clamped to zero, marked ruined, and stops compounding —
  nothing compounds from zero, and an intratrade absorption stops the path even when the close R would
  have recovered. This is Tharp's bet-too-big lesson surfacing in the arithmetic: his player who stakes
  the whole $100 and draws the black marble is out of the game and can never realize the expectancy
  (Loc 3765). Threshold breach alone is a flag; only absorption stops the path.

Three rules are fixed now, so the build has no discretion:

- **Determinism and common random numbers.** One call builds one `random.Random(seed)`, and every path's
  draw indices are generated up front, independent of the equity fold — an absorbed path stops
  compounding, not drawing. The same seed therefore produces identical draw sequences at every
  `fraction`, so a sweep compares fractions on common random numbers, path by path.
- **Draws are a bootstrap, not a permutation.** Each path draws `n_trades` marbles with replacement
  (`n_trades >= 1`); `n_trades=None` means a career the same length as the ledger, resampled — not a
  reshuffle of it.
- **An empty bag raises `ValueError`.** There is nothing to draw, and a silent empty result would read
  as a zero-risk system.

### `sizing_sweep`

```python
def sizing_sweep(r_multiples, *, fractions=(0.0025, 0.005, 0.01, 0.02, 0.03), **kwargs) -> dict
```

One `simulate_sizing` result per fraction. The grid is sourced from the book rather than invented — Tharp
runs his own version of this exercise, taking one 0.8R-expectancy system through 0.5, 1, and 3 percent
risk and getting three different careers, from steady growth to abandon-trading drawdowns (Loc 8903):

| Fraction | Source in the book |
| --- | --- |
| 0.25% | The wide-stop example: a 10-ATR stop with total risk held to 0.25 percent of equity (Loc 5826). |
| 0.5% | The bottom of Tharp's own-money guideline band, 0.5 to 2.5 percent (Loc 8559). |
| 1% | An interviewed fund manager's actual 0.8 to 1.0 percent sizing (Loc 1307 — a practitioner's words, not Tharp's); Tharp's guideline of 1 percent or less for other people's money (Loc 8559). |
| 2% | The same manager's escalation path: 2 to 3 percent "would push the envelope" (Loc 1307). |
| 3% | Basso's threshold — risking 3 percent on one position is being a "gunslinger" (Loc 959); Tharp's over-2.5-percent band accepts a high probability of ruin (Loc 8559). |

### `kelly_fraction`

```python
def kelly_fraction(r_multiples) -> float
```

The log-optimal fraction by numeric maximization of `mean(log(1 + f*r))` over `f` in `[0, 1/|min r|)` —
the upper bound is the absorption boundary, past which one draw is fatal. **Reported as a reference point
only, never a recommendation.** For a negative-expectancy bag the maximum sits at `f = 0`, which is
itself the informative answer: no positive size is log-optimal on a losing game. The Status disclaimer
applies to this function with particular force.

Edge rules, fixed now: a bag with `mean(r) <= 0` returns exactly `0.0` without searching — the growth
curve is concave with nonpositive slope at the origin, so the maximum sits on the boundary, and the
function never returns a negative fraction (an all-loser bag is the extreme case of the same rule). A bag
with no negative R has no absorption boundary and an unbounded log-optimal fraction, so the function
raises `ValueError` rather than inventing a cap; an empty bag raises too. The maximization itself is a
fixed, seed-free grid over the open interval — deterministic, no RNG anywhere in the function.

### Per-regime bags

They compose for free. Filter a ledger by Gap D's bucketing (`regime_ledger_statistics` /
`SIX_REGIME_CELLS`, common/trade_ledger.py:300-341) and pass a cell's `r_multiple` column as the bag. The
sizer adds no regime code; the caller composes, keeping `common/` a leaf — the Gap D pattern.

## The synthetic ground-truth game

Tharp's game 1 is a bag with 60 percent winners and +0.2R expectancy (Loc 3758). The highlights state the
win rate and the expectancy; the ±1R payoff structure is implied by the arithmetic
(`0.6*(+1R) + 0.4*(-1R) = +0.2R`) rather than stated verbatim, so this doc pins ±1R as the standard
reading, not as quoted text. With ±1R payoffs the game has closed-form ground truth:

- Expected log growth is `G(f) = 0.6*ln(1+f) + 0.4*ln(1-f)`.
- The Kelly optimum is `f* = 0.6 - 0.4 = 0.2`.
- `G(f) > 0` for `f` below \~0.39 — roughly twice Kelly. Beyond that, the median path shrinks despite
  the positive expectancy.
- `f = 1` busts on the first losing draw — Tharp's whole-stake example (Loc 3765) — and his summary rule
  follows: position size must stay low enough that the long-term expectancy remains realizable
  (Loc 3777).

This game is the always-run test bed: every mechanics assertion checks the simulator against these
analytic values, no dataset required. The workshop bag (Loc 8233: seven −1R, one −5R, two +10R) is a
second fully specified fixture — unlike game 1, its composition is verbatim in the notes — available if
the mechanics tests want a skewed bag; game 1 stays primary because of the closed form.

## Expected first results — pre-stated

The expected shape is stated before any code runs, so the measurement cannot be spun after the fact.

**On the real ledgers, every positive fraction loses.** Both pinned ledgers are negative expectancy:
−0.39R per trade on the MSFT covered call and −0.54R on the SPY short-vol overlay
(tests/test_trade_ledger.py:355, :367). Jensen's inequality settles the growth half before the sweep
runs: for a negative-mean bag, `E[ln(1 + f*r)] <= ln(1 + f*E[r]) < 0` for every `f > 0`, so long-run
growth is negative at every fraction. The ruin half is the prediction: P(ruin) should rise monotonically
with `f`, and the pinned regression records whether it does. Experiment 1 on these ledgers therefore
demonstrates one half of Tharp's thesis: **no sizing rescues a negative game.** That is his own example
— the 90-percent-win system with −0.1R expectancy that eventually loses everything (Loc 3810) — and his
definition: a negative net impact per dollar risked makes the account disappear (Loc 3650). Sizing
realizes expectancy; it cannot create it (Loc 3761).

**The keep-what-you-win half runs on the synthetic game.** Sizing determining how much of a positive game
you keep is demonstrated where the ground truth is analytic: median terminal wealth should rise from
`f = 0.02` to the Kelly optimum `f = 0.2` and fall by `f = 0.5` (where `G(0.5) < 0`), while P(ruin)
stays monotone in `f` throughout. The drawdown distribution makes Tharp's losing-streak passages
quantitative — larger sizing ruins when the streaks come early (Loc 4059), and his players' urge to bet
bigger mid-streak (Loc 4041) is exactly what the per-path drawdowns price. The recovery arithmetic
anchors the ruin threshold: beyond a 50 percent drawdown, required gains grow improbable (Loc 8308).

The real ledgers are the dataset-gated measurement; the synthetic game is the always-run bed.

## Test plan

The new classes ride tests/test_trade_ledger.py rather than a new file. The module-scoped `msft_run` /
`spy_run` fixtures live there (tests/test_trade_ledger.py:45, :71) and are file-local — no
tests/conftest.py or repo-root conftest.py exists — so reusing them costs zero additional engine passes,
while a new file would force hoisting them into a new conftest (a larger structural change that would
also expose them to unrelated files). CI needs no edit either: the trend-explore bucket already runs the
whole file (.github/workflows/ci.yml:195, executed by the pytest line at :240).

**`TestPositionSizingMechanics`** (always-run, synthetic, no dataset):

- Hand-computable replay: the bag `[+1R]` at `f = 0.10` gives terminal `1.1**n` to float tolerance; a
  −12R draw at `f = 0.10` gives `f*r = -1.2`, an absorbed ruin.
- Determinism: the same seed yields the same output dict, and every fraction in a sweep sees the
  identical draw sequences (the common-random-numbers rule above).
- The Tharp game: `kelly_fraction` returns \~0.2; median terminal at `f = 0.2` beats both `f = 0.02` and
  `f = 0.5` on long paths (the Kelly hump); P(ruin) is monotone in `f`.
- Intratrade ruin: a trade whose `mae_r` breaches the threshold while its `r` recovers by the close marks
  the path ruined when `mae_r` is passed, and does not when it is omitted.
- Edge rules: an empty bag raises in both functions; `kelly_fraction` returns `0.0` on an all-loser bag
  and raises on a bag with no negative R.

**`TestPositionSizingRegression`** (dataset-gated with the same skipif as the existing regression
classes, tests/test_trade_ledger.py:322-323; reuses `msft_run` / `spy_run`): pins the default sweep on
both real ledgers — P(ruin) at the default threshold per fraction, median terminal wealth per fraction,
and the monotonicity verdict. These are Experiment 1's first pinned measurements.

## Cross-surface obligations (when the code lands, not now)

- Symbol-sweep regex: append
  `simulate_sizing|sizing_sweep|kelly_fraction|TestPositionSizingMechanics|TestPositionSizingRegression`
  in the same change that introduces the symbols, per the CLAUDE.md rule.
- README `## Project layout`: add rows for `common/position_sizing.py` and this doc.
- ci.yml: unchanged, provided the tests ride tests/test_trade_ledger.py — the file is already in the
  trend-explore bucket, and only a brand-new test file would need a bucket edit.
- Notebook: no regen — nothing touches tutorial_covered_call_backtest.md or engine/make_figures.py.
- Line anchors: no engine edit, so no engine anchors move.

## Honesty rails

- **Descriptive, not edge.** Experiment 1 measures the risk distributions of already-measured systems. It
  proposes no trade, flags no hypothesis, and claims no edge.
- **No FDR interaction.** Nothing enters `idea_ledger.jsonl` and no e-value is spent: e-LOND/BY govern
  flagged hypotheses, and none is flagged here. If a sizing-conditioned strategy ever emerges from these
  distributions, it becomes a new exploratory scout under the usual rails (the explorations pattern, the
  campaign FDR if automated) — never a promoted finding of this work.
- **One significance authority.** The daily Newey-West HAC t (`compute_statistics` /
  `short_vol_statistics`) remains the sole authority, unchanged. Every sizing output is reported, never a
  gate — the ledger's own pinned posture (common/trade_ledger.py:9-13).
- **Kelly is a reference point.** `kelly_fraction` locates a bag on the growth curve; it is never a
  recommendation.
- **The disclaimer is stated once, in Status,** and covers every number this design will produce.

## Open questions

1. **Ruin-threshold default.** `ruin_threshold=0.5` (equity falling below half its starting value) is
   grounded in the book's recovery arithmetic — beyond 50 percent down, the gains required to recover
   grow improbable (Loc 8308). The notes also carry two practitioner 25-percent figures: an interviewed
   manager plans around "worst-case drawdowns of 25 percent" (Loc 1332 — the practitioner's words, not
   Tharp's), and Gallacher's largest-expected-equity-drop exercise names 25 or 50 percent as the
   tolerance to plan against (Loc 8812). Leaning: report P(ruin) at both thresholds (0.5, and 0.75 for
   the 25-percent tolerance) in the sweep output and headline the 0.5 default.
2. **An `avg_loss_1R`-normalized bag as a labelled cross-check.** `ledger_statistics` already offers
   Tharp's ex-post normalizer (`r_normalizer='avg_loss_1r'`, common/trade_ledger.py:246-250, :259-269).
   Consistency argues for accepting an optional bag normalized the same way, carrying the same loud
   ex-post label — never the primary. Leaning yes.
3. **Horizon convention.** The default is one same-length career (`n_trades=None`). Tharp's own exercises
   use fixed horizons — 20-trade months (Loc 3728) and 40 draws (Loc 8233) — plus at least 100
   repetitions of a year's worth of trades (Loc 4098), and his comparison metric multiplies expectancy by
   opportunity (Loc 3791), which fixed horizons mirror. The `n_trades` parameter covers all of these; the
   only open choice is which convention the pinned regression uses. Leaning: same-length career, so the
   pin needs no extra convention argument.

## Related

- [docs/van_tharp_test_plan.md](van_tharp_test_plan.md) — the parent plan; Gaps C and B are its rows, and
  Experiment 1 is its first experiment.
- [docs/van_tharp_gap_a.md](van_tharp_gap_a.md) — the R-multiple ledger this design replays (merged).
- [docs/van_tharp_gap_d.md](van_tharp_gap_d.md) — the per-regime R-distributions the per-regime bags draw
  from (merged).
- [docs/explorations.md](explorations.md) — the exploratory, kill-or-justify pattern any follow-on
  strategy idea would ride under.
- research/book-notes/trade-your-way-to-financial-freedom.md — the highlights file every Location above
  cites.
