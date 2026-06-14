# Real Option Chains Flip the $268,000 to a Loss

*Ten years of traded quotes for six underlyings, and a re-run of the whole series: the engine had been minting premiums 1.5–2.3× richer than anything the market ever paid.*

Five posts ago, this series opened with a covered-call backtest that turned $100,000 into just over a million dollars — $268,000 of it credited to the option overlay. I just re-ran that exact strategy against ten years of real Microsoft option prices. The overlay loses $183,552.

This post is the bill for the caveat at the end of the [last one](05_the_fix_that_wasnt.md). The engine that produced every number in this series prices its options with Black-Scholes off a volatility proxy — it had never seen an actual option chain. I wrote then that a volatility strategy priced off a volatility proxy is grading its own homework. So I bought the homework's answer key: real daily option chains, every trading day from April 2016 through April 2026, and re-ran everything.

What follows is the gap between the world the backtest modeled and the one the market traded, in three steps: how big the pricing error was, what it did to QQQ, and what it does to the $268,000 this series was built on.

## Measuring the blind spot

The data is the kind of thing you can just buy now. For each of six underlyings — Microsoft, QQQ, SPY, IWM, GLD, and TLT — I pulled the real quote for the exact call the strategy would have sold at each monthly roll over the decade: bid, ask, implied volatility, delta. Then, for QQQ and Microsoft, the full daily treatment: a slice of the whole chain every single trading day, almost four million quotes between the two names, so the strategy could be marked to market on real prices every day it held a position.

The first finding came from the six-name comparison, before any backtest ran. For every underlying, on the same contract — same strike, same expiration, same day — the engine's proxy quoted a richer price than the market did. The implied volatility it manufactured ran 1.27× to 1.56× the real figure. The premiums that volatility fed ran 1.55× to 2.33× the traded quotes, and the inflation was worst exactly where realized volatility is lowest — SPY, the calmest of the six, at 2.33×.

The mechanism is the one the last post flagged. The engine estimates volatility from the underlying's own recent price swings, then nudges it up with a fixed multiplier to stand in for the markup option sellers usually command. That markup — the gap between what options imply and what the underlying goes on to do — is the entire crop a covered call harvests. The proxy didn't measure it. It assumed it, generously, on every contract for ten years.

## QQQ flipped first

QQQ got the full re-run before Microsoft did, and the result set the pattern. On the same unadjusted price series, the proxy engine scores the QQQ covered call at **+$120,217** of net overlay profit. On real chains — selling each call at the bid, buying it back at the ask, real deltas deciding when a position is too deep in the money to hold — the same decade produces **−$156,628**.

The anatomy of the loss matters more than its size. The strategy's bread-and-butter trades still worked: 122 calls hit their profit target and banked $234,000. But 71 positions blew through their strikes and had to be bought back deep in the money, at a cost of $398,000. The wins were real and frequent; the losses were rarer and ruinous. That asymmetry was always in the strategy. The proxy's inflated premiums had been papering over it.

And the spread wasn't the culprit. Re-run the QQQ decade with every trade filled at the quote midpoint — no bid/ask cost at all — and the loss only shrinks by about $12,000. The premium the strategy was supposed to collect simply wasn't there at the prices QQQ options actually traded.

## Then the number the series was named for

Microsoft is where this series started: the backtest that made $268,000. That figure was computed on a dividend-adjusted price series, and option data forces a switch to *unadjusted* prices — the ones whose dollars match actual strikes — so the apples-to-apples check comes first. Run the proxy engine on the unadjusted series and it reports +$269,948. The two dollar figures land close partly by coincidence (the unadjusted series buys fewer, pricier shares), but the verdict is the same run twice: same strategy, same decade, same comfortable six-figure profit. The headline stands or falls on the premiums.

It falls.

| MSFT, 2016–2026 | Proxy engine | Real chains |
| --- | --- | --- |
| Net overlay P&L | +$269,948 | **−$183,552** |
| Gross premium collected | $937,324 | $729,055 |
| Win rate | 80% | 68% |
| Max drawdown | 24% | 41% |

Every line moves the same direction. The market paid $729,000 for the calls the proxy thought were worth $937,000. The win rate drops because real deltas flag trouble earlier and more often than the proxy's synthetic ones did. The drawdown nearly doubles because the overlay keeps surrendering upside in the recoveries without collecting enough premium to pay for it. Net result: 122 profit-target wins earn $429,000, and 54 deep-in-the-money buybacks give back $611,000.

The statistics agree with the accounting. The published result's Newey-West t-statistic was +0.46 — profit indistinguishable from luck, as [post 4](04_one_number_that_killed_it.md) conceded. The real-chain run lands at **−1.73** under worst-case fills, which still doesn't clear the t = 2 bar this series holds every result to. The defensible reading is the modest one: there is no detectable edge in either direction, and the dollar sign now points down.

## Where Microsoft differs from the index

One honest wrinkle separates the two flips. For QQQ, mid-quote fills barely dented the loss; the failure was pure premium economics. For Microsoft, filling every trade at the midpoint recovers about $108,000 — roughly six-tenths of the loss. Single-stock options trade with much wider spreads than index ETF options, and a strategy that sells at the bid and buys back at the ask pays that toll on every round trip, 183 times in this run.

So a generous reader could say the Microsoft covered call doesn't lose $184,000; it loses $76,000 plus transaction costs. That is the most charitable version of this result, and it still has the strategy ten years and $76,000 underwater before commissions on the half of the cost the market lets no one skip entirely. The spread explains part of the damage. It rescues nothing.

## What's left standing

The honest scoreboard for the series, then. The stock profits were always real — Microsoft itself rose 570% over the decade, and no option overlay touched that engine. The statistical verdicts survive too, and this is worth pausing on: the t-statistic refused to bless the edge back when the backtest still showed a profit. The methodology held up. What collapsed is the modeled P&L itself, because the model's premiums were fiction, and a covered call is nothing but its premiums.

That collapse reaches further than the headline. Post 4's delta-hedged refinement — the version that isolated the volatility premium and earned a tantalizing t of 1.63 — was priced by the same proxy. So was QQQ's 1.58. So I re-measured both on real chains, and the edge was the proxy's, not the market's: Microsoft's hedged Newey-West t falls from 1.63 to **−0.23**, QQQ's from 1.58 to **+0.18**, each indistinguishable from zero. The hedge still does its mechanical job; what it cannot do is isolate a premium that was never in the real quotes. The basket idea from the last post inherits the same warning, still unmeasured. I no longer trust any dollar figure this engine prints, and the engine is mine. That's the right amount of trust to extend a backtest that assumes the one quantity it exists to measure.

The price of finding this out was a few hundred dollars of data and a weekend of plumbing. Against ten years of imaginary premium income, it's the best trade in the series.

## The takeaway worth keeping

Every backtest is a small world, and the results it prints are facts about that world, not ours. The only way to learn which facts transfer is to replace each assumption with a measurement and watch what breaks. Realized volatility plus a multiplier felt close enough to implied volatility. It was off by enough to turn a million-dollar decade into a loss.

If you run backtests of your own, the transferable lesson is mechanical: find the input your strategy's profit is most sensitive to, and check whether your simulator measures it or assumes it. If the answer is "assumes," your headline number is a hypothesis wearing a conclusion's clothes.

The next stop hasn't changed — a basket of underlyings, now with real premiums end to end. The bar it has to clear is wherever t = 2 lives. Real prices have already shown which side of zero this strategy starts from.

---

*This is an educational walkthrough, not investment advice. Every backtest number here — both flips, the fill-model variants, the trade-level loss anatomy — is pinned by regression tests that run in CI against the checksummed option-chain datasets, so none of it can quietly drift. The premium-inflation ranges come from the per-roll quote snapshots committed in the same repo.*
