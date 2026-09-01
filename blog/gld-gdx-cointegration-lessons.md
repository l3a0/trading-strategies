# Lessons from testing GLD/GDX for cointegration

The published hedge ratio in a 2009 quant-trading book never reproduced exactly, no matter whose data went in. The reason is a data-vintage story, not a mistake. The pair itself had a shelf life.

Ernest Chan's *Quantitative Trading* opens its pairs-trading chapter with an example. Buy gold, short the gold miners, and the two prices tend to drift back toward each other.

The book runs a **cointegration test** on the pair. That is a statistical check for whether a long-short combination of two prices stays range-bound instead of wandering off on its own. It reports two numbers:

1. A **hedge ratio** of 1.6766, meaning short 1.6766 dollars of miners per dollar of gold.
2. A test statistic of −3.18, where more negative means the spread mean-reverts more convincingly.

Using Python and the latest data reproduced the test statistic, but not the hedge ratio. The hedge ratio kept landing on 1.6379, not the book's 1.6766. So did [the independent reproduction](https://aushaff.github.io/2018/04/06/e_chan_ex7.2.html) and [Chan's own data](https://github.com/burakbayramli/books/tree/master/Quantitative_Trading_Chan) when re-run today.

Here is the ledger of what reproduced and what didn't, then the six detours that explain why.

| Quantity | Book → reproduced | Status |
| --- | --- | --- |
| Hedge ratio (through-origin) | 1.6766 → 1.6379 | vintage drift |
| CADF test statistic (full / training) | −3.36 / −3.18 → −3.45 / −3.09 | reproduced |
| Mean-reversion half-life | \~10 days → \~10 days | reproduced |
| Cointegration verdict | >90% conf. → rejects null | reproduced |

## Six detours

### 1. The test statistic and hedge ratio came from two different runs

The hedge ratio and the test statistic do not come from the same computation. The hedge ratio is from a full-window run in Chapter 7. The test statistic is from a Chapter 3 example that drops the last 60 days and tests only the first 252.

The book prints them near each other. It is easy to read them as one result. They are two, on two different windows. Trace every quoted number to its own run before trying to match it.

### 2. Two regressions, two different hedge ratios

The hedge ratio from the test's own regression came out at 1.3905, nowhere near the book's 1.6766. The answer was buried in Chan's `example7_2.m`.

Chan's hedge ratio comes from a regression forced through the origin, with no intercept. His test statistic comes from a *separate* regression that includes one. Same pair, two regressions, two different slopes. Through the origin the slope is 1.6379. With an intercept it is 1.3905. Read the spec, not just the result.

### 3. "Adjusted close" is not a fixed number

The exact 1.6766 cannot be reproduced from any download made today.

Prices come in two flavors. **Raw** prices are what traded. **Adjusted** prices are re-scaled backward to fold dividends in, as if every payout were reinvested, so a chart reflects total return. The catch is that the series is pinned to the latest price. So every new dividend scales the whole earlier history down another notch.

GDX has paid dividends for nineteen years since 2007. So today's adjusted 2006 price sits about 15% below the number Chan saw. His 2007-vintage adjusted price is closest to today's *raw* price, because the miners had barely paid a dividend yet. Adjusted close is a function of the download date.

Anchoring the series at a fixed start, instead of today, sidesteps the drift. That gives a total-return index that never rewrites its past. Simpler still, a raw as-traded price is already fixed. A given day's close is a historical fact, whatever dividends come later. That is the series this reproduction uses.

| Reproduction | Hedge ratio | Note |
| --- | --- | --- |
| Book, 2007 vintage | 1.6766 | not reproducible from any modern download |
| Everyone since | \~1.6379 | independent reproductions converge here, this one included |

### 4. Separate the fragile estimate from the robust conclusion

Not everything drifted. The hedge ratio moved with the data vintage. The **half-life**, which is how long the spread takes to close half its gap, reproduced at about 10 days, matching the book. So did the verdict. The spread is cointegrated with better than 90% confidence.

The trade signal survived. The hedge ratio slipped about two percent, from 1.6766 to 1.6379, while the half-life and the verdict held.

### 5. An independent implementation separates data bugs from code bugs

The test ran twice. Once by hand, in fifteen lines of numpy. Once through a standard statistics library. The two agreed to four decimal places. That agreement was the proof that the remaining gap was data, not a bug in the arithmetic.

Chan hit the same fork and drew the opposite lesson. His Python test disagreed with his MATLAB and R tests on this pair, and he concluded that Python's statistics packages could not be trusted. They were fine. All three ran the same test on different default settings.

| Setting | Lags | CADF t |
| --- | --- | --- |
| Python default (autolag) | 6 | −2.30 |
| MATLAB spec (fixed) | 1 | −3.09 |

Same data, same library, one setting. Chan's own numbers were −2.4 and −3.2, reproduced here.

The setting is the number of lags the test adds to absorb autocorrelation. MATLAB fixes it at one. Python's default reads it from the data, and on the shorter window it chose six. Each extra lag pulls the statistic toward zero, and six was enough to push Python's result across the line into "not cointegrated." Pin the lag and all three agree.

### 6. Cointegration is a property of a window, not a pair

The last detour outlives the reproduction. Gold and gold miners cointegrated cleanly from 2006 to 2008. Run the same test over 2006 to 2026 and it falls apart.

The statistic drops to −1.45, well short of significance. The spread's half-life balloons from 10 days to over 800. The miners detached from gold somewhere in the 2010s and never fully came back.

| Window | t-statistic | Verdict |
| --- | --- | --- |
| 2006–2008 | −3.18 | cointegrated |
| 2006–2026 | −1.45 | not cointegrated |

The relationship the book documented was real. It also had a shelf life.

The two-window table is the compressed version. The full picture is a rolling test. Slide a one-year window across the whole history and compute the statistic in each. Where it dips below the 10% critical line, the pair cointegrates in that window.

[![Two-panel regime map of GLD versus GDX from 2007 to 2026. The top panel plots the rolling one-year CADF t-statistic, which dips below the −3.04 critical line only in scattered windows clustered in the early years. The bottom panel shows Chan's through-origin hedge drifting upward from about 1.64 to a peak near 6.6 around 2016.](../docs/figures/reproduction_regime_map.png)](../docs/figures/reproduction_regime_map.png)

*Rolling one-year cointegration test on as-traded GLD/GDX closes, 2007–2026. Green bands mark the windows that clear the 10% critical value (−3.04). Only 31 of 231 windows clear it, and they cluster before 2015. Below, Chan's through-origin hedge drifts from \~1.64 to above 4, so there is no single ratio a fixed pair trade could have held.*

Cointegration flickers on and off, and just 31 of the 231 windows clear even the 10% bar. They cluster in the early years, around Chan's own window. In the decade from 2015, only 10 of 139 windows reject. The lower panel shows why the fixed trade was doomed regardless. The hedge that balances the spread climbs from Chan's \~1.64 to above 4, and briefly past 6. There was never one ratio to hold.

## The two runs, reproduced

| Run | Window | Hedge (origin) | CADF t | Verdict |
| --- | --- | --- | --- | --- |
| Ch. 7 full | 2006-05 – 2007-11 | 1.6766 → 1.6379 | −3.36 → −3.45 | \~95% |
| Ch. 3 (p.63) | first 252 days | 1.6766 → 1.6283 | −3.18 → −3.09 | \~90% |

*Exploratory reproduction, not investment advice. Figures are from a from-scratch Python re-run on yfinance data and will not match the book to the last digit, by design.*

## So what

When backtesting from a paper or a book, budget more time for data provenance than method. The method is usually in the open. The exact prices are not. Which vintage, which adjustment, which vendor's later revisions. A reproduction that lands the conclusion but misses the last digit is usually a data-vintage story, not a mistake.

The clean version of this project would have printed 1.6766 and moved on. The messy version shows how a published number ages. The method holds. The data drifts. And the relationship itself can quietly dissolve. Reproducing a result is less about matching digits than understanding why they move.

*Reproduced with a numpy-only Engle-Granger / CADF test, cross-checked against statsmodels. The implementation is open source: [l3a0/trading-strategies](https://github.com/l3a0/trading-strategies), with the [pair engine](https://github.com/l3a0/trading-strategies/blob/main/search/pair_cointegration.py) and its [pinned tests](https://github.com/l3a0/trading-strategies/blob/main/tests/test_pair_cointegration.py). Source: Ernest P. Chan, Quantitative Trading, rev. ed., `example7_2.m` and `example3_6_1.m`, GLD & GDX daily closes.*
