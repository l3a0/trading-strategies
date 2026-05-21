# Black-Scholes Is a Recipe, Not a Crystal Ball

*Four of the five inputs to an option's price are facts you can look up. The fifth is a guess — and that's where almost every options backtest quietly cheats.*

Suppose I asked you to price one year of car insurance for a 25-year-old. You don't need an actuarial license to know what you'd want: how much the car costs to replace (the stock price), what the coverage limit is (the strike), how long the policy runs (the time to expiration), what you can earn on the premium while you hold it (the risk-free rate), and how wild a driver we're talking about (the volatility). Five questions. Answer them honestly and you can name a fair price.

That is the entire idea behind the most famous equation in finance.

Last post I showed a backtest that "made" $268,000 layering covered calls on Microsoft, and then a statistic suggesting the edge was indistinguishable from luck. Before we can judge whether that profit was real, we have to open the engine and look at where its option prices came from — because the backtest never bought a single real option. It calculated them. This is the part of the machine where optimism gets in.

## The five ingredients

The Black-Scholes model is the insurance recipe, ported to options. It takes five inputs and returns a fair premium. Mapping them onto the car-insurance questions:

- **Stock price** — the current value of the asset, like the replacement cost of the car.
- **Strike price** — the price at which the shares could be called away, the equivalent of the coverage limit.
- **Time to expiration** — how long the contract runs, measured as a fraction of a trading year.
- **Risk-free rate** — what you'd earn parking money in Treasuries while the contract is open, roughly 4% lately.
- **Volatility** — how much the stock bounces around. This is the wild-driver question, and it is the one that matters here.

Four of those five are facts. The stock price is on the screen. The strike and the expiration are terms you choose. The risk-free rate is a published Treasury yield. You can look all of them up to the penny on any given historical day.

Volatility is the exception, and the whole post turns on it.

## Delta is a probability dial

Before the volatility problem, one piece of vocabulary, because it's the knob a covered-call seller actually turns. **Delta** is, roughly, the probability that an option finishes in-the-money — that the stock ends up past the strike and the shares get called away.[^1]

It runs from 0 to 1, and you can read it directly as a risk setting. Sell a 0.20-delta call and you've said "I'm fine with about a one-in-five chance these shares get called away." A 0.50-delta call is a coin flip. A 0.80-delta call is asking for it. Income-oriented sellers live in the 0.20–0.40 band, and the backtest from Post 1 used 0.25 — roughly a one-in-four chance of assignment on each contract.

![A downward-sloping curve of call delta against how far out-of-the-money the strike is set. Delta falls from about 0.54 at the money toward zero far out-of-the-money. A shaded horizontal band marks the 0.20–0.40 income-seller zone, and a red dot marks the backtest's 0.25-delta setting at roughly 7% out-of-the-money.](../docs/figures/06_delta_dial.png)

*The dial, drawn out. Pushing the strike further out-of-the-money slides you down the curve to a lower delta — a smaller chance of assignment, and a smaller premium. The backtest sits at the red dot.*

That assignment isn't a malfunction when it happens. It's the deal you signed for the premium. Selling a call means selling away the far upside on purpose; getting called away on the one-in-four is the price of getting paid the other three times. A covered-call strategy that never gets assigned isn't being careful — it's leaving money it was promised on the table.

## The input you can't look up

Here is the honest problem. To price that 0.25-delta call on a given day in 2018, Black-Scholes needs the volatility the *market* expected on that day — **implied volatility**, the figure baked into what option buyers were actually willing to pay. And implied volatility is not something you can recover after the fact unless you have the historical option prices it was baked into.

You don't have them. For two separate reasons, and both are worth naming because most amateur backtests skip past both.

The first is that the data barely exists in usable form for free. Daily closing stock prices going back decades are a click away and effectively free. Historical option chains — every strike, every expiration, every day — are spotty, inconsistent, and unreliable wherever they're free, when they're available at all.

The second is that the *clean* version costs real money. Raw historical option prices are cheap enough now — a few hundred dollars buys years of end-of-day quotes from budget vendors. The expensive part is the rigorously cleaned, research-grade implied-volatility data that funds and universities actually trust. The OptionMetrics tier is sold by quote and licensed through institutional subscriptions; even mid-market vendors charge into the thousands for a full clean history, five figures for the deepest sets.[^2]

So an individual building an educational backtest faces a fork. The cheap data needs cleaning and validation work most people won't do, and the trustworthy version is priced for funds. Either way you end up estimating volatility — and that's the quiet reason so many retail backtests of options strategies never say where their volatility came from. Pinning it down honestly costs money or labor, so they skip it.

So the backtest substitutes a proxy. It measures **historical volatility** — how much the stock actually bounced over the trailing 30 days — and scales it up by a multiplier that depends on the market regime: about 1.1× when volatility is already high, 1.3× in normal conditions, 1.5× when volatility is unusually low.

The multipliers run the way they do for a real reason. Implied volatility tends to sit roughly 20–40% above realized volatility — the *volatility risk premium*, option buyers paying up for insurance the way drivers overpay for peace of mind. (The premium is well documented; the exact band is a practitioner rule of thumb, not a precise constant.) But that gap is widest exactly when markets are calm and complacent, and it compresses when volatility is already elevated and implied vol has nowhere left to climb. A flat markup would miss that; the regime dial approximates it.

It is a reasonable approximation. It is still a guess, and it breaks in predictable places: implied vol explodes before earnings in a way trailing realized vol can't see coming, it normalizes faster than realized vol after a crash, and in a strong one-directional trend a stock can be moving a lot while options stay cheap. Name those weak spots out loud, because a model that hides its softest input is more dangerous than one that doesn't.

![Two volatility lines over 2016–2026 for Microsoft: trailing realized volatility in gray and the regime-scaled implied-volatility proxy in blue, the proxy sitting consistently above realized with the gap shaded. Horizontal bands at 15% and 25% mark the low, normal, and high regimes with their 1.5×, 1.3×, and 1.1× markups labeled. A volatility spike to about 110% appears in early 2020.](../docs/figures/05_implied_vs_realized_vol.png)

*Four of the five Black-Scholes inputs are the gray line's worth of fact. The blue line — what the proxy assumes option buyers would have paid — is the guess, and the shaded gap is the size of it. Notice the gap narrows in the 2020 panic: that's the 1.1× high-vol multiplier kicking in, exactly where implied vol has the least room left to run.*

## Why you should care even if you never sell an option

This generalizes well past covered calls. Any backtest of any options strategy rests on an assumed volatility, because the alternative — paying for clean historical option data — is expensive enough that most people doing it for a blog post or a pitch deck didn't.

So when someone shows you the returns of an options strategy, the first question isn't how good the returns are. It's: where did your volatility come from — did you pay for the real thing, or estimate it? If they estimated it, how, and where does the estimate fail? A backtest that answers those questions cleanly has earned a look. One that can't answer them is showing you the output of a recipe whose main ingredient was assumed into existence.

## What this doesn't fix

Suppose the volatility proxy were perfect — every option in the backtest priced exactly as the real market would have. The strategy still wouldn't be proven, because there's a second way to fool yourself that has nothing to do with pricing and everything to do with how you tune the rules around it. You can take an honest engine and still torture it into a beautiful, fragile, fictional result.

That's the next post.

Black-Scholes earns the "recipe" label honestly: give it good ingredients and it returns a fair price every time, fast. The danger was never the math. It's that one ingredient has to be guessed, and the guess is invisible in the final number — which is exactly why the honest move is to say it out loud.

---

*This is an educational walkthrough, not investment advice. The pricing functions, the volatility proxy, and the regime logic described here are [open-source on GitHub](https://github.com/l3a0/covered-call-backtesting) if you want to see precisely what got estimated and swap in your own assumptions.*

[^1]: "Roughly" is doing real work. Delta and the probability of finishing in-the-money are closely related but not identical quantities; for setting a covered-call strike, the difference is small enough to ignore and the intuition is what matters. The tutorial in the repo walks through the precise version for anyone who wants it.

[^2]: Budget vendors list full multi-year U.S. option histories for roughly $69–$245 ([discountoptiondata.com](https://www.discountoptiondata.com/)). Research-grade vendors price a basic full history around $1,150 and a cleaned implied-volatility-surface set around $10,600 ([historicaloptiondata.com](https://historicaloptiondata.com/shop/)). [OptionMetrics / IvyDB](https://optionmetrics.com/), the dataset most academic options research runs on, doesn't publish prices — it's licensed through institutional subscriptions like [WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics/). Figures current as of May 2026.
