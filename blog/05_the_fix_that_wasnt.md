# The Index Was Supposed to Be the Fix

*I ran the covered-call backtest on QQQ expecting a diversified ETF to clear the bar a single stock couldn't. It came back weaker — and the reason is a blind spot in the backtest itself.*

The last post ended with a prescription. If you want to know whether a covered call has a real edge, don't run a longer backtest on one stock. Run it on a broad index, where the volatility premium covered calls harvest has been measured for decades and shows up far more reliably than it does on any single name.

QQQ is the obvious place to start — the Nasdaq-100 in one ticker, the name every options-income article reaches for when it wants to sound prudent. So I pointed the same engine at it: same parameters, same ten years, same hostile battery of tests. I expected the t-statistic that buried the Microsoft version, 0.46 against a bar of 2, to finally climb.

It came back **0.10.**

## A quick recap, for anyone joining here

Across the [Microsoft series](04_one_number_that_killed_it.md), a covered-call overlay added $268,000 over a decade and survived every robustness check I could build — except the one that counts. The honest t-statistic on its excess return *over simply holding the stock* was 0.46: indistinguishable from luck. The profit was real; the edge was not.

The explanation pointed somewhere specific. The premium covered calls collect is compensation for bearing volatility risk, and that compensation is reliably measurable at the index level while getting lost in the idiosyncratic noise of any single stock. Microsoft was the hard case. QQQ — a hundred names averaged into one quieter index — was supposed to be the upgrade.

Here's what the run actually showed: the same seductive headline, a thinner edge beneath it, a delta-hedged signal that survives where the raw one dies, and a caveat about the engine that the rest of the post has to be read against.

## Same seduction, half the fuel

The top line looks familiar. The overlay turned $100,000 into $745,000 — up 645% — against 542% for buying and holding QQQ outright. That's $103,000 of apparent outperformance, a 77.5% win rate across 182 calls sold, and a 22% maximum drawdown. Read only that, and you'd open your brokerage app.

Then look at what the overlay had to work with. It collected $493,000 in gross premium over the decade. The Microsoft version pulled in $999,000 — more than double. That gap is diversification doing exactly what it's designed to do. QQQ holds a hundred names whose idiosyncratic moves partly cancel, so the index swings less than its average component, and option premium is priced off swings. A calmer underlying is a thinner vein to mine. The "safe" candidate is safe precisely because there's less to harvest.

## The deflation, wider

Now the honest number. Subtract QQQ's own 542% climb and ask what the overlay added on top — the excess return, which measures the strategy rather than the index it rode. Annualized, that excess is 0.22%. Its Sharpe ratio is **0.027**, which rounds to nothing. The Newey-West t-statistic — which corrects for the fact that holding a one-month option makes consecutive days far from independent — is **0.10.**

Microsoft's 0.46 was already a failing grade. QQQ fails by a wider margin. The $103,000 of "outperformance" is almost entirely the equity-timing wiggle, the noise that comes from the call's exposure drifting as the index moves, riding on top of a signal too faint to separate from it.

A shuffle test agrees. Scramble the order of QQQ's daily returns five hundred times, rebuild a synthetic price path from each, and re-run the overlay: the real ordered path lands at the 80th percentile of that distribution. Better than a coin flip, but a long way from the Microsoft overlay, which beat all five hundred. There's a flicker of real structure in QQQ's price path. There is no edge in the strategy that trades it.

## What the hedge reveals

This is where QQQ earns the comparison. The overlay's excess return tangles two things together: the volatility premium you actually want, and that equity-timing wiggle, which has zero expected value and a lot of variance. Hedge the wiggle away — rebalance the share position daily so the portfolio's net exposure stays pinned to plain buy-and-hold — and the same 182 calls isolate the premium.

On Microsoft, that hedge tripled the Sharpe of the excess return. On QQQ it does far more. The Sharpe jumps from 0.027 to **0.405**, and the Newey-West t-statistic climbs from 0.10 to **1.58.** Net overlay profit roughly doubles, $103,000 to $217,000; premium retention rises from 21% to 44%. The fainter the naive signal, the more of it was buried under the wiggle — and QQQ's was nearly buried entirely, so clearing the noise does proportionally more work.

Notice where that lands. QQQ's hedged t-statistic of 1.58 is essentially Microsoft's 1.63. Two very different underlyings — one jumpy single stock, one placid index — produce almost the same risk-adjusted premium once the direction is stripped out. That sameness is the fingerprint of something real and shared: a roughly constant price for bearing volatility, wherever you point the strategy. It is also, on both, still short of the t = 2 bar.

## What the engine can't see

Now the caveat the whole result hinges on, and it deserves its own section because it reframes everything above.

This engine has never seen an option chain. It prices every call with Black-Scholes off an *implied-volatility proxy* it estimates from the index's own recent realized volatility, nudged up by a regime multiplier. No real QQQ option prices enter the calculation anywhere.

That matters for what the QQQ result can claim. The reason the literature says the index volatility premium is strong is that index options trade *rich*: their implied volatility sits persistently above what the index goes on to realize, because institutions pay up for portfolio insurance. That persistent gap between implied and realized *is* the premium. And it is precisely what a realized-vol proxy can only assume, never measure: the engine manufactures implied volatility out of realized volatility, baking the premium into a fixed multiplier rather than reading it off real prices.

So the QQQ run isn't a measurement of the real index premium. It's a measurement of how the overlay's mechanics behave on QQQ's price path, with synthetic premiums standing in for traded ones. The honest reading of "QQQ came back weaker" is narrow: with less realized volatility to scale, the proxy mints less premium, so the modeled edge shrinks. Whether the *actual* QQQ premium — the one living in real option prices — would clear the bar is a question this backtest is structurally unable to answer. The instrument has a blind spot, and naming it beats narrating around it.

## So what

Two things survive the caveat, and they point the same way.

First, switching tickers was never the fix, and now there's a number for it. A single underlying — jumpy stock or placid index, real premium or proxied — can't manufacture significance out of one decade. The naive overlay is noise on both. The hedged remainder is a real but small premium, small enough that confirming it on one index would take decades of clean data — which is exactly why the academic work leans on multi-decade histories. The naive version would never get there on any horizon worth discussing.

Second, the route the last post pointed at holds, sharpened. The thing worth harvesting is a *cross-asset* premium, so the fast way to a real t-statistic is breadth. Combine many weakly-correlated underlyings into one position and your statistical power grows with the number of *independent* bets, not the calendar — a basket of names that don't move together reaches significance years before any single ticker could. Pair that with real option data, so the implied-minus-realized gap is measured instead of assumed, and you'd finally be testing the premium the literature actually describes. QQQ holding Microsoft at roughly 8% of the index is a small reminder that stacking correlated tickers doesn't buy independent bets.

## What the run was actually worth

The covered call isn't a free lunch on QQQ any more than it was on Microsoft. But the most useful thing this run produced wasn't the 0.10, or even the 1.58. It was the reminder that the backtest can only interrogate the world it can see — and a volatility strategy priced off a volatility proxy is, in the most literal sense, grading its own homework on exactly the question it exists to answer.

The next honest step was never going to be another ticker. It's real option chains, and a basket.

---

*This is an educational walkthrough, not investment advice. Every number here is produced by the same engine as the rest of the series, run on QQQ instead of Microsoft, and pinned by a regression test so it can't quietly drift.*
