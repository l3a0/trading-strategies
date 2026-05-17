# The Cheapest Way to Fool Yourself in Quant Investing

*Four independent attacks every backtest should survive before you believe it.*

If you study for an exam by reading the exact questions that will be on it, you'll get a perfect score. You'll also have learned nothing about whether you understand the subject. The grade is real and completely uninformative at the same time.

That is the single most common way a backtest lies, and it has a technical name: **in-sample optimization**. Last post I showed that even an honestly priced engine — every option valued the way the real market would have — can still be tortured into a beautiful, fragile, fictional result. This is how the torturing happens, and how you stop doing it to yourself.

## The cheapest way to fool yourself

A covered-call strategy has a few knobs: how far out-of-the-money to sell, how long until the option expires, how aggressively to mark up the volatility estimate. Each knob has a handful of plausible settings. Multiply them together and you have a few dozen variants of the same strategy.

Now run all of them on 2016–2020 and keep the best one. You will find a combination that returned something absurd — call it 1,000%. Publish that number and you have a sensation. Trade it in 2021 and it dies.

It dies because you didn't discover a strategy. You discovered the settings that best fit the *noise* in one specific stretch of history — which storms hit which weeks, which dips happened to reverse. That pattern was real in 2016–2020 and carries no information about 2021, the same way memorizing last year's exam tells you nothing about this year's. The more knobs you turn and the more combinations you try, the more confidently you fit noise and the more spectacular — and more fictional — the headline number gets.

## Studying for a different test

The fix is almost insultingly simple to state: never evaluate a strategy on data you used to choose its settings.

The disciplined version is called **walk-forward validation**, and it works like a rolling exam. Take the first two years of history as a training window. Search every parameter combination on *those two years only*, pick the best one, then lock it — no more tuning allowed. Now run that locked strategy on the next six months, which the search never saw. Record the result. Roll the whole apparatus forward six months and repeat: retrain, lock, test on fresh data, record. On the Microsoft history this produces fifteen of these train-then-test cycles.

At the end you stitch together only the out-of-sample pieces — the six-month stretches the strategy was never tuned on — into a single equity curve. That curve is the honest one. It's what you would actually have earned running the strategy in real time, making each parameter choice with the information you'd really have had, never with hindsight. You studied on one set of questions and were graded on a different set. The grade means something now.

The search itself isn't exotic. It's a grid of twenty-seven combinations — three choices for strike distance, three for expiration length, three for the profit-taking rule — and on each training window it simply keeps the combination with the best risk-adjusted return. The discipline isn't in the search. It's in the locking.

## What the honest number looks like

Here's the rule of thumb. A strategy with a genuine edge typically retains 60–70% of its in-sample return when you measure it honestly out-of-sample. Some give-back is expected and healthy — it's the price of not having hindsight. What you're watching for is the *collapse*: 1,000% in-sample, 50% out-of-sample. That gap is the signature of a strategy that was never real, and walk-forward is the test that exposes it.

The covered-call overlay does not collapse. Over the walk-forward span — April 2018 to October 2025 — the honest, no-peeking curve compounded to about **483%**. Running the fixed default settings over that same window returned about **563%**. The strategy kept roughly 86% of its in-sample performance: comfortably above the healthy band, nowhere near a collapse.[^1]

There's a subtler piece of evidence buried in those fifteen retraining cycles. The search was free to pick any of twenty-seven combinations each time, on fifteen very different market windows — and it kept landing on the same one, the boring middle setting the rest of the analysis already used. A search that keeps independently rediscovering the same configuration across a correction, a crash, and a sideways grind is telling you that configuration reflects something structural, not a fluke of one window.

Now the sobering part, and it's the one that actually matters. Retaining 86% of an in-sample number is reassuring, but it's the wrong yardstick. The right one is the simplest possible alternative: just buying Microsoft and holding it. Over that same April 2018–October 2025 window, buy-and-hold returned about **467%**. So the honest, no-hindsight overlay beat doing nothing clever at all by roughly **16 points over seven and a half years**. The optimized 563% looked like a comfortable 96-point win over the stock; strip out the hindsight and almost the entire margin evaporates. The strategy is robust, general, survives every test in this post — and clears a buy-and-hold investor by a sliver. Whether even that sliver is real is exactly the question the final post answers.

## Three more ways to attack it

Walk-forward catches overfitting. It doesn't catch everything, so you attack the strategy from three more directions.

The first is **Monte Carlo**. Take the actual daily returns, shuffle their order, and rebuild a synthetic price path from the scrambled sequence. The set of returns is identical — same mean, same volatility — but the *order* is destroyed: no more trends, no more volatility clusters. Do this five hundred times and run the strategy on every scrambled path. If it only made money on the real ordering, it was exploiting a lucky sequence. On the Microsoft data the real path's return beat all five hundred shuffles — every single one, with the best scramble topping out below it. The strategy is harvesting a statistical property of the returns, not memorizing their sequence.

The second is **sensitivity analysis**. Take the chosen settings and nudge one at a time — strike distance a little tighter, then looser; profit target a little higher, then lower. A strategy perched on a fragile optimum falls apart under small perturbations. This one doesn't: every nudge moves the return by single-digit percentages. The optimum is a plateau, not a needle.

The third is **regime analysis**. Bucket every trade's profit by the market state it closed in — bull, bear, or sideways — and check whether the strategy secretly depends on one of them. The covered-call overlay is profitable in all three, and the breakdown is the most interesting result in the whole project: it earns roughly $23 of premium per day in bull markets and $300–$400 per day in bear and sideways ones. It isn't a bull-market strategy wearing a disguise. It's structurally defensive, earning most of its keep precisely when the market isn't going straight up — which is the entire point of selling insurance.

## Why this is the real skill

A strategy that survives four independent attacks — out-of-sample testing, scrambled price paths, parameter perturbation, and every market regime — is worth far more than one that posted a bigger number on a single decade it was tuned to fit. Robustness beats optimization, and it isn't close. The optimized number is the one you want to be true. The robust number is the one you can act on.

Most strategies you'll be pitched report the first kind and quietly skip all four tests. Now you know exactly which questions to ask, and why a confident answer to "how did it do?" without an answer to "how did you keep from fooling yourself?" is no answer at all.

## The one test it still fails

So the covered-call overlay clears every bar in this post. It doesn't collapse out-of-sample, it beats five hundred scrambled histories, it shrugs off parameter nudges, it works in every regime. By the standards most backtests are held to, it's bulletproof.

It still fails one test — the single statistic from the first post, the one that came back at 0.46. Walk-forward and robustness checks ask "is this strategy stable and general?" They never ask "is its edge distinguishable from zero?" Those are different questions, and the second one has a brutal answer here. That's the next, and final, post — including why the naive way of computing that statistic is itself a fifth way to lie to yourself.

---

*This is an educational walkthrough, not investment advice. The walk-forward, Monte Carlo, sensitivity, and regime routines described here are [open-source on GitHub](https://github.com/l3a0/covered-call-backtesting) if you want to run the four attacks yourself.*

[^1]: Span housekeeping, since these numbers invite cross-comparison with the first post. The headline ~915% there is the fixed-settings return over the *full* ten-year sample (2016–2026); the 483% and 563% here are the narrower walk-forward window (2018–2025), which trims roughly two years off each end — so the apples-to-apples pair is 483% against 563%, not 483% against 915%. Likewise the ~646% buy-and-hold figure from the first post is the full sample; the ~467% used above is buy-and-hold over the matched walk-forward window.
