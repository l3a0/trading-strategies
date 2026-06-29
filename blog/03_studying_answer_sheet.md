# The Cheapest Way to Fool Yourself in Quant Investing

*Four independent attacks every backtest should survive before you believe it.*

If you study for an exam by reading the exact questions that will be on it, you'll get a perfect score. You'll also have learned nothing about whether you understand the subject. The grade is real and completely uninformative at the same time.

That's the most common way a backtest lies, and it has a technical name: **in-sample optimization**. Last post I showed that even an honestly priced engine — every option valued the way the real market would have — can still be tortured into a fragile, fictional result. This is how that happens, and how you stop doing it to yourself.

## The cheapest way to fool yourself

A covered-call strategy has a few knobs: how far out-of-the-money to sell, how long until the option expires, how aggressively to mark up the volatility estimate. Each knob has a handful of plausible settings. Multiply them together and you have a few dozen variants of the same strategy.

Now run all of them on 2016–2020 and keep the best one. You will find a combination that returned something absurd — call it 1,000%. Publish that number and you have a sensation. Trade it in 2021 and it dies.

It dies because you didn't discover a strategy. You discovered the settings that best fit the *noise* in one specific stretch of history — which storms hit which weeks, which dips happened to reverse. That pattern was real in 2016–2020 and carries no information about 2021, the same way memorizing last year's exam *answers* tells you nothing about this year's *questions*. The more knobs you turn and the more combinations you try, the more confidently you fit noise and the more spectacular — and more fictional — the headline number gets.

## Studying for a different test

The fix is simple to state: **never evaluate a strategy on data you used to choose its settings.**

The disciplined version is called **walk-forward validation**, and it works like a rolling exam. Take the first three years of history as a training window. Search every parameter combination on *those three years only*, pick the best one, then lock it — no more tuning allowed. Now run that locked strategy on the next six months, which the search never saw. Record the result. Roll the whole apparatus forward six months and repeat: retrain, lock, test on fresh data, record. On the Microsoft history this produces thirteen of these train-then-test cycles.

At the end you stitch together only the out-of-sample pieces — the six-month stretches the strategy was never tuned on — into a single equity curve. That curve is the honest one. It's what you would actually have earned running the strategy in real time, making each parameter choice with the information you'd really have had, never with hindsight. You studied on one set of questions and were graded on a different set. The grade means something now.

The search itself is a grid of twenty-seven combinations — three choices for strike distance, three for expiration length, three for the profit-taking rule — and on each training window it simply keeps the combination with the best risk-adjusted return. The discipline isn't in the search. It's in the locking.

![A schedule of thirteen stacked rows, one per walk-forward cycle, time on the horizontal axis from 2016 to 2026. Each row shows a three-year training bar followed immediately by a six-month test bar; successive rows shift six months later, so the windows march diagonally down and to the right.](../docs/figures/07_walk_forward_schematic.png)

*Thirteen rolling exams. Each blue bar is three years the search was free to optimize on; the orange bar right after it is six months of fresh data where the rules were frozen. Only the orange stretches count toward the honest score.*

## What the honest number looks like

As a practitioner rule of thumb — *walk-forward efficiency*, popularized by Robert Pardo — a strategy with a genuine edge keeps most of its in-sample return when you measure it honestly out-of-sample; the figure usually quoted is somewhere around two-thirds. That band is lore, not a law, but the order of magnitude is the point. Some give-back is expected and healthy — it's the price of not having hindsight. What you're watching for is the *collapse*: 1,000% in-sample, 50% out-of-sample. That gap is the signature of a strategy that was never real, and walk-forward is the test that exposes it.

The covered-call overlay does not collapse. Over the walk-forward span — April 2019 to October 2025 — the honest, no-peeking curve compounded to about **324%**. Running the fixed default settings over that same window returned about **378%**. The strategy kept roughly 86% of its in-sample performance: comfortably above the healthy band, nowhere near a collapse.[^1]

There's a subtler piece of evidence buried in those thirteen retraining cycles. The search was free to pick any of twenty-seven combinations each time, on thirteen very different market windows — and it kept gravitating to the same neighborhood. The strike-distance dial in particular locked onto the boring middle setting — a 0.25 delta — in all thirteen; the other two knobs wandered a little but never far from the configuration the rest of the analysis already used (you can see the wandering in the labels on the chart above). A search that keeps independently returning to the same region across a correction, a crash, and a sideways grind is telling you that region reflects something structural, not a fluke of one window.

Now the part that matters. Retaining 86% of an in-sample number is reassuring, but it's the wrong yardstick. The right one is the simplest possible alternative: just buying Microsoft and holding it. Over that same April 2019–October 2025 window, buy-and-hold returned about **317%**. So the honest, no-hindsight overlay beat doing nothing clever at all by roughly **7 points over six and a half years**.[^2] The optimized 378% looked like a comfortable 61-point win over the stock; strip out the hindsight and almost the entire margin evaporates. The strategy is robust, general, survives every test in this post — and clears a buy-and-hold investor by a sliver. Even the sliver's width depends on a bookkeeping choice (the footnote has the details), so treat the 7 as a rough size, not a measurement. Whether even that sliver is real is exactly the question the final post answers.

![Three cumulative-return curves over the 2019–2025 walk-forward span. The optimized fixed-defaults curve ends highest at +378%, the honest out-of-sample curve at +324%, and buy-and-hold Microsoft just below at +317%. All three track closely for most of the span and only fan apart late.](../docs/figures/08_is_vs_oos.png)

*The honest blue curve doesn't collapse — but it doesn't separate either. It spends the whole span a hair above the gray buy-and-hold line. The win that looked like 61 points is a 7-point sliver once the hindsight is gone.*

## Why three years, not two

Why train on *three* years per window rather than two? Robert Pardo, who popularized walk-forward testing, has a rule of thumb for whether a window holds enough data: tally your data points, subtract what the strategy's moving parts consume, and keep most of the total — north of 90% — free. By that count both windows pass easily; two years leaves about 93% free, three years about 96%.

But days are the wrong unit when a single position sits on the books for weeks. One call I sell drives the profit and loss for the next month, so those daily observations aren't independent. The honest unit is the *trade*. A two-year window produces only about two dozen of them — short of the ~30 that's the conventional floor for drawing a statistical conclusion at all. A three-year window clears it comfortably: every one of its thirteen retraining cycles lands above the floor.

## Three more ways to attack it

Walk-forward catches overfitting. It doesn't catch everything, so you attack the strategy from three more directions.

The first is **Monte Carlo**. Take the actual daily returns, shuffle their order, and rebuild a synthetic price path from the scrambled sequence. The set of returns is identical — same mean, same volatility — but the *order* is destroyed: no more trends, no more volatility clusters. Do this five hundred times and run the strategy on every scrambled path. If it only made money on the real ordering, it was exploiting a lucky sequence. On the Microsoft data the real path's return lands far out in the right tail of the scrambled distribution — around the 99th percentile. The batch below cleared all five hundred shuffles; rerun the scramble and a handful edge past, so the real ordering isn't unbeatable, just rare. Either way, the strategy is harvesting a statistical property of the returns, not memorizing their sequence.

![A histogram of total returns from 500 shuffled price paths, roughly bell-shaped and centered near 657%. A dotted line marks the shuffle mean, a dashed line the best shuffle at about 870%, and a solid red line far to the right marks the real ordered path at about 915%, beyond every shuffled outcome.](../docs/figures/09_monte_carlo.png)

*Five hundred scrambles, and in this batch the real path's return sits to the right of all of them, out around the 99th percentile of orderings. If the strategy depended on the exact sequence of history, destroying the order would have killed it. It didn't.*

The second is **sensitivity analysis**. Take the chosen settings and nudge one at a time — strike distance a little tighter, then looser; profit target a little higher, then lower. A strategy perched on a fragile optimum falls apart under small perturbations. This one doesn't: every nudge moves the return by single-digit percentages. The optimum is a plateau, not a needle.

The third is **regime analysis**. Bucket every trade's profit by the market state it closed in — bull, bear, or sideways — and check whether the strategy secretly depends on one of them. The covered-call overlay is profitable in all three. It earns roughly $23 of premium per day in bull markets and $300–$400 per day in bear and sideways ones. It's structurally defensive — it earns most of its keep precisely when the market isn't going straight up, which is the entire point of selling insurance.

![A bar chart of average overlay P&L per day by market regime. The bull bar is tiny at about $23 per day; the sideways bar towers at about $400 per day and the bear bar at about $300 per day.](../docs/figures/10_regime_pnl.png)

*The shape of a defensive strategy. A bull-market strategy in disguise would have its tall bar on the left. This one barely registers when Microsoft is ripping and does its real work in the flat and falling stretches — selling insurance pays best when buyers are scared.*

## Robustness beats optimization

A strategy that survives four independent attacks — out-of-sample testing, scrambled price paths, parameter perturbation, and every market regime — is worth far more than one that posted a bigger number on a single decade it was tuned to fit. The optimized number is the one you want to be true. The robust number is the one you can act on.

Most strategies you'll be pitched report the first kind and quietly skip all four tests. Now you know exactly which questions to ask, and why a confident answer to "how did it do?" without an answer to "how did you keep from fooling yourself?" is no answer at all.

## The one test it still fails

So the covered-call overlay clears every bar in this post. It doesn't collapse out-of-sample, it beats five hundred scrambled histories, it shrugs off parameter nudges, it works in every regime. By the standards most backtests are held to, it's bulletproof.

It still fails one test — the single statistic from the first post, the one that came back at 0.46. Walk-forward and robustness checks ask "is this strategy stable and general?" They never ask "is its edge distinguishable from zero?" Those are different questions, and the second one has a brutal answer here. That's the next, and final, post — including why the naive way of computing that statistic is itself a fourth way to lie to yourself.

---

*This is an educational walkthrough, not investment advice. The walk-forward, Monte Carlo, sensitivity, and regime routines described here are [open-source on GitHub](https://github.com/l3a0/trading-strategies) if you want to run the four attacks yourself.*

[^1]: Span housekeeping, since these numbers invite cross-comparison with the first post. The headline ~915% there is the fixed-settings return over the *full* ten-year sample (2016–2026); the 324% and 378% here are the narrower walk-forward window (2019–2025), which trims the first three years off the front and ~6 months off the end, so the span-matched pair is 324% against 378%, not 324% against 915%. Likewise the ~646% buy-and-hold figure from the first post is the full sample; the ~317% used above is buy-and-hold over the matched walk-forward window.

[^2]: Bookkeeping behind that 7, for the careful reader. The 324% walk-forward figure restarts every six-month window at the same $100,000 and compounds the thirteen window returns; the 378% and 317% comparison figures run once, straight through the span, with no restarts. Recompute all three on a single consistent convention and the overlay's margin over buy-and-hold lands anywhere from roughly 19 points (restart everything each window) to roughly 44 points (let capital ride and reinvest throughout) — the mixed accounting above is what yields the 7. A margin that swings several-fold with bookkeeping choices isn't a measurement of edge; it's a warning that endpoint gaps are the wrong instrument. The right instrument is the final post's: the day-by-day excess return over buy-and-hold, with an honest uncertainty estimate attached — a measurement none of this bookkeeping touches.
