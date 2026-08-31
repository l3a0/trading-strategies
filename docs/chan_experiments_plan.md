# Chan's *Quantitative Trading*: experiment reproduction plan

A catalog of the worked examples in Ernest Chan, *Quantitative Trading* (revised
ed., 2021), captured as Kindle highlights in
[research/book-notes/quantitative-trading.md](../research/book-notes/quantitative-trading.md).
This is the roadmap for reproducing them in this repo, one at a time, as a
follow-on to the GLD/GDX cointegration reproduction that is already done.

These are the **book's own examples**, not backtests this repo invented. The book
teaches through numbered "Examples," and those are the experiments. Every figure
below is the figure Chan reports. It is the target to reproduce, not a repo
result, unless a row says "Done" and cites the repo's number next to it.

## How to read this

Each experiment carries a status:

- **Done**: reproduced in the repo and pinned by a regression test.
- **Ready**: the data is on hand and the build is small (a day or less).
- **Buildable**: the data is on hand but the build is larger (a new model or panel).
- **Blocked**: the repo does not have the data, and it is not free to fetch.

Data the repo holds today: daily price CSVs for ten tickers (EEM, GDX, GLD, IWM,
MSFT, NVDA, QQQ, SPY, TLT, XLE), option-chain dailies for most of them, and a
personal S&P 500 one-minute archive (\~913 tickers, cold storage). Any US equity's
daily prices are free to add from yfinance, the way GLD and GDX were. The repo
does **not** hold earnings dates or surprises, commodity-futures data, or
fundamentals (book value, market cap), so the experiments that need those are
blocked until that data is sourced.

## Progress at a glance

Ordered by the recommended work sequence: done items first, then Steps 1-7, then
the data-blocked ones. Each row maps to a detail section below by name. The detail
sections stay grouped by the book's categories, so the section numbers (§) run in
book order, not work order.

| Step | Experiment | Book Example | Data | Status |
| --- | --- | --- | --- | --- |
| ✓ | GLD/GDX cointegration test | Ch.3 / Ch.7 | on hand | **Done** |
| ✓ | GLD/GDX half-life | Ch.3 | on hand | **Done** |
| ✓ | Python-vs-MATLAB stat detour | Ch.3 | on hand | **Done** |
| 1 | KO vs PEP non-cointegration | Ch.2 | free to fetch | **Ready** |
| 2 | Kelly / SPY leverage | Ex. 6.2 | on hand | **Ready** |
| 3 | The coin-flip game | Ex. 6.1 | synthetic | **Ready** |
| 4 | Risk parity | Ch.6 | on hand | **Ready** |
| 5 | Khandani-Lo linear reversal | Ex. 3.6 / 3.7 | daily panel needed | Buildable |
| 6 | Equity seasonals | Ex. 7.6 / 7.7 | on hand | Buildable |
| 7 | Conditional Parameter Optimization | new-edition centerpiece | 1-min archive | Buildable |
| — | Other stationary-spread candidates | Ch.2 | mixed | Blocked / Ready |
| — | Post-earnings announcement drift | Ch.7 | earnings data | Blocked |
| — | PCA statistical factor model | Ch.7 | S&P 600 panel | Blocked |
| — | Fama-French three-factor + WML | Ch.5 | fundamentals | Blocked (WML partial) |
| — | Commodity-futures seasonals | Ch.7 | futures data | Blocked |

## 1. The flagship: GLD vs GDX pair trade

GLD tracks spot gold. GDX is a basket of gold-mining stocks. Their prices should
move together, so a long/short combination should stay range-bound. This example
runs through the whole book.

### 1.1 Cointegration test: Done

A cointegration test asks whether a long/short combination of two prices stays
range-bound. The book's CADF (cointegrating augmented Dickey-Fuller) test reports
a t-statistic of −3.18 and a hedge ratio of \~1.68 (long GLD, short 1.68x GDX).

Repo result: the reproduction lands the CADF t-statistic at −3.09 on the Ch.3
training window and −3.45 on the full Ch.7 window, with a through-origin hedge of
1.6379. The exact 1.6766 the book prints is a lost data vintage. Independent
re-runs converge near 1.6379. Pinned by `TestGldGdxReproduction` in
[tests/test_pair_cointegration.py](../tests/test_pair_cointegration.py). The
engine is [search/pair_cointegration.py](../search/pair_cointegration.py) on the
statsmodels-backed [common/timeseries.py](../common/timeseries.py).

### 1.2 Half-life: Done

The Ornstein-Uhlenbeck half-life sets the expected holding period. The book
reports \~10 days.

Repo result: 10.3 days on the training window, 10.6 on the full window. Same test
class as 1.1.

### 1.3 The Python-vs-MATLAB stat detour: Done

Chan reports that Python's statistics libraries disagreed with MATLAB and R on the
cointegration verdict, and concludes "do not trust Python's statistics and
econometrics packages."

Repo result: the disagreement is real and explained. Python's `statsmodels`
defaults to `autolag='aic'`, which picks a different lag count and a different
statistic. Pinning the lag (`maxlag=1, autolag=None`) reproduces the fixed-lag
convention MATLAB and R use. Written up in
[blog/gld-gdx-cointegration-lessons.md](../blog/gld-gdx-cointegration-lessons.md).

### 1.4 Conditional Parameter Optimization (CPO): Buildable

The new edition's centerpiece. A machine-learning method (random forest with
boosting) re-tunes the strategy's three parameters daily from 115
market-condition features, trained on one-minute bars 2006-2020, split 80/20.
It predicts the *strategy's* return, not gold's price. That is metalabeling.

Data: the repo's one-minute S&P 500 archive covers the bar frequency, though not
GLD/GDX specifically (those would need a one-minute fetch). The build is the large
piece: a feature pipeline, a random-forest model, and a metalabel target. This is
the most involved experiment in the book.

## 2. Mean-reversion / reversal

### 2.1 Khandani-Lo linear reversal: Buildable

Buy the stocks with the worst one-day returns, short the best (Example 3.6 / 3.7).
The original paper reported a Sharpe of 4.47 in 2006, but that came from
small-cap and microcap stocks. On S&P 500 large caps the Sharpe drops to 0.25, and
to −3.19 after a 5-basis-point transaction cost. A variation trades at the open
instead of the close to try to recover it.

Data: needs a daily cross-section of S&P 500 large caps. The one-minute archive
can be aggregated to daily bars for \~913 names, or a daily panel fetched from
yfinance. The build is a cross-sectional ranking backtest with a cost model. The
lesson to reproduce is the collapse from 4.47 to 0.25 to −3.19, not a headline
edge.

### 2.2 KO vs PEP non-cointegration: Ready

A deliberate counter-example. Coca-Cola and Pepsi are correlated (0.4849,
significant) but do **not** cointegrate. It shows that correlation and
cointegration are different things.

Data: KO and PEP daily prices, free to fetch from yfinance like GLD and GDX.
Build: reuse [search/pair_cointegration.py](../search/pair_cointegration.py)
directly. Run the correlation and the cointegration test on the pair and confirm
the pair correlates but fails to cointegrate. This is the smallest next step and
reuses the flagship's engine.

### 2.3 Other stationary-spread candidates: Blocked / Ready

The book names CAD/AUD, calendar spreads, and bond-maturity pairs as other
stationary-spread candidates beyond stocks. CAD/AUD (currency ETFs like FXC/FXA)
and bond-maturity pairs (Treasury ETFs) are fetchable and reuse the pair engine.
Futures calendar spreads need futures data the repo lacks.

## 3. Momentum

### 3.1 Post-earnings announcement drift (PEAD): Blocked

Buy when earnings beat expectations, short when they miss. Needs an earnings
calendar with surprise data, which the repo does not have.

### 3.2 PCA statistical factor model: Blocked

A principal-component factor model on S&P 600 small caps, assuming factor returns
have momentum: buy the top expected returns, short the bottom. The book's result
is modest, \~2-4% a year, and only before costs. Needs an S&P 600 small-cap daily
panel, which the repo does not have (its minute archive is S&P 500).

## 4. Factor models

### 4.1 Fama-French three-factor + WML: Blocked (WML partial)

The Fama-French three factors (market, SMB = small-minus-big, HML =
high-minus-low value) plus WML (winners-minus-losers) momentum as an extension.
SMB and HML need fundamentals (market cap and book-to-market), which the repo does
not have. The WML momentum leg is buildable from prices alone, and the repo's
[factor/factor_mechanism.py](../factor/factor_mechanism.py) already builds a
price-based `trend` factor of the same shape, so a partial reproduction of the
momentum leg is within reach.

## 5. Seasonal / calendar

### 5.1 Equity seasonals: Buildable

Mostly dead, per the book. The January effect and the Heston-Sadka monthly
strategy (buy each month's prior-year winners) returned more than 13% a year
before 2002, but the effect has since disappeared (Example 7.7). A reader-suggested
seasonal stock trade also failed on backtest (Example 7.6).

Data: long daily equity history, which the repo has for its ten tickers and can
extend by fetching more. The lesson to reproduce is the disappearance of the
effect after 2002, which is a "this used to work and no longer does" result.

### 5.2 Commodity-futures seasonals: Blocked

Still alive, per the book. The gasoline trade (buy RB's May contract \~April 13,
sell \~April 25) was profitable 19 of 21 years. The natural-gas trade (buy the June
NG contract \~Feb 25, sell \~April 15) hit 14 straight years but held up worse out
of sample. Both need continuous commodity-futures data the repo does not have.

## 6. Money management

### 6.1 Kelly / SPY leverage: Ready

With SPY at 11.23% mean return, 16.91% volatility, and a 4% risk-free rate, the
Sharpe is 0.4275 and the optimal Kelly leverage is 2.528x, for 13.14% compounded
growth (Example 6.2). The book then stress-tests it against Black Monday's 20.47%
one-day loss.

Data: SPY daily returns, on hand. Build: the continuous-Kelly leverage
`f* = (mu - r) / sigma^2` plus the leverage stress test. This is a small new build,
not a rerun of what the repo has. The repo's
[common/position_sizing.py](../common/position_sizing.py) carries a
`kelly_fraction`, but it is the DISCRETE form over a bag of per-trade R-multiples
(grid-searched, capped at the absorption boundary), applied to the option
strategies' trade ledgers as a descriptive reference. Chan 6.2 is the CONTINUOUS
form on one asset's return moments, and it yields a LEVERAGE above 1x (2.528x), not
a risk fraction. Same idea, different formula and object. So this reuses the
concept, not the code.

### 6.2 The coin-flip game: Ready

Win $110 or lose $100 on a fair coin (Example 6.1). Positive expected value, yet
negative *compound* growth. This is the ensemble-average versus time-average
argument for why loss aversion is rational.

Data: none, it is synthetic. Build: a small simulation contrasting the ensemble
mean with the time-average growth rate. It pairs naturally with the repo's
marble-bag resampler in `common/position_sizing.py`.

### 6.3 Risk parity: Ready

Qian's 23/77 stock/bond mix, levered 1.8x, beats the classic 60/40.

Data: SPY and TLT daily returns, both on hand. Build: a risk-parity weighting plus
the levered comparison. The repo's
[common/portfolio.py](../common/portfolio.py) already aligns and combines daily
equity streams, so the plumbing exists.

## Notes on the order

The table's Step column is the recommended sequence. Steps 1-4 are quick. KO/PEP
reuses `pair_cointegration` directly, and risk parity reuses `portfolio`'s
stream-combining. Kelly and the coin-flip are small standalone builds. They share
the log-growth idea behind `position_sizing`'s `kelly_fraction`, but reproduce
Chan's continuous-leverage and ensemble-vs-time forms, which the repo does not
carry. Steps 5-7 are larger builds whose data is on hand. The blocked rows wait
until their data is sourced.

Each experiment carries the same epistemic label as the rest of the repo's
exploratory work. Reproducing the book's number is the goal. A reproduced edge is a
candidate, never a promotion, until it clears an out-of-sample, net-of-cost
holdout.
