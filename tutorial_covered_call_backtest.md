# Building a Covered Call Backtester From Scratch

## Theory, Code, and Hard-Won Lessons

**For:** Bao, actively learning options trading  
**Goal:** Understand both the WHY and the HOW of backtesting covered calls  
**Time to read:** 60 minutes (code walkthrough: another 60)  
**Last updated:** May 2026

---

## How to Read This Tutorial

This is a long document. You almost certainly don't need every word of it. Pick the path that matches where you're starting from:

- **Total beginner (new to options, new to backtesting):** Read straight through. Each part builds on the last, and the [glossary](#appendix-c-glossary-of-key-terms) at the back has every term you'll need.
- **Coder, no finance background:** Skim the [Glossary](#appendix-c-glossary-of-key-terms) and [Part 1](#part-1-foundations--what-are-we-actually-doing) for vocabulary, then spend your time in [Part 2](#part-2-option-pricing-with-black-scholes) (Black-Scholes) and [Part 3](#part-3-the-covered-call-overlay-engine) (the overlay engine). That's where the finance becomes mechanical.
- **Quant learning Python options:** Jump to [Part 4](#part-4-walk-forward-optimization) (walk-forward) and [Part 5](#part-5-robustness-checks--proving-its-not-luck) (Newey-West, bootstraps, regime splits), then browse the rest as reference.

---

## Table of Contents

1. [Part 1: Foundations — What Are We Actually Doing?](#part-1-foundations--what-are-we-actually-doing)
2. [Part 2: Option Pricing with Black-Scholes](#part-2-option-pricing-with-black-scholes)
3. [Part 3: The Covered Call Overlay Engine](#part-3-the-covered-call-overlay-engine)
4. [Part 4: Walk-Forward Optimization](#part-4-walk-forward-optimization)
5. [Part 5: Robustness Checks — Proving It's Not Luck](#part-5-robustness-checks--proving-its-not-luck)
6. [Part 6: Putting It All Together](#part-6-putting-it-all-together)
7. [Part 7: Key Takeaways & Cheat Sheet](#part-7-key-takeaways--cheat-sheet)
8. [Appendix A: The Code](#appendix-a-the-code)
9. [Appendix B: Common Pitfalls and How to Avoid Them](#appendix-b-common-pitfalls-and-how-to-avoid-them)
10. [Appendix C: Glossary of Key Terms](#appendix-c-glossary-of-key-terms)
11. [References](#references)
12. [Provenance & Disclaimer](#provenance--disclaimer)

---

## Part 1: Foundations — What Are We Actually Doing?

### The Core Idea: Own Shares + Sell Insurance = Extra Income

Let me start with the simplest possible explanation.

Imagine you own a house worth $300,000. You could:

1. **Do nothing** — hope it appreciates, sit and wait
2. **Rent it out** — get monthly income while keeping the house
3. **Sell homeowner's insurance to your neighbors** — collect premiums if nothing bad happens

A covered call is like option #2 and #3 combined. You own the stock (like owning a house). You sell call options (like selling insurance: "I'll let you buy my stock at $50 anytime in the next 30 days, and you pay me $2 for that right"). If the stock goes up, sometimes your buyer exercises the option and buys your shares (you "lose" them, but at a fixed price). If it stays flat or goes down, the buyer doesn't exercise, and you keep both the stock AND the premium ($2).

> **Key insight:** You're not trying to hit a home run. You're trying to collect small, frequent premiums while the stock does its normal thing.

### Why Backtesting Matters (And Why Most Backtests Lie)

Before you risk real money, you want to ask: "Does this actually work? How much could I make? What could go wrong?"

A backtest is a time machine. You rewind to the past, follow your rules perfectly, and measure what would have happened. It's not perfect, but it's way better than guessing.

**Why most backtests are misleading:**

- People test only when things are going well (lucky timing)
- They peek at future prices while building today's decision (look-ahead bias)
- They only count the survivors — the stocks that didn't go bankrupt (survivorship bias)
- They tweak parameters so much that they overfit to random noise (overfitting)

We'll avoid all three of these traps.

### The Three Enemies of Backtesting

| Enemy | What It Means | Example in CC Trading | How We'll Stop It |
| --- | --- | --- | --- |
| **Look-ahead bias** | You use tomorrow's price to make today's decision | "I'll sell a call because I know the price will drop tomorrow" | Only use data available on the decision date; never peek forward |
| **Survivorship bias** | You only test stocks that survived (ignoring the ones that died) | Only test Apple, Google, Microsoft (tech survived 2000s); ignore Blockbuster | Test a diverse index; accept all stocks |
| **Overfitting** | You tune your strategy so it's perfect for 2010-2020, then it fails 2021-2026 | Tweak the delta, expiration, and volatility multiplier until you get 1000% returns | Use walk-forward validation: train on one period, test on a different period |

### Mental Model: Think of Backtesting Like a Time Machine With Rules

Here's how we'll build it:

1. **Rewind** to April 2016
2. **Load** 10 years of daily price data for a stock
3. **Each day**, check: Should I open a covered call? Should I close it? What's my profit?
4. **Simulate** the entire history
5. **Measure** the results (return, drawdown, Sharpe ratio, etc.)

The output? A graph showing: "If you'd done this from 2016–2026, you'd have made $X" — and how much of that was luck vs. skill.

---

## Part 2: Option Pricing with Black-Scholes

### The Analogy: Black-Scholes Is Like a Recipe

If I asked you, "How much should I charge for car insurance for a 25-year-old?" you'd need to know:

- How likely is a crash? (volatility)
- How much will it cost when it happens? (strike)
- How long is the policy? (time)
- How much will I earn from interest on the premiums? (interest rate)
- What's the current car value? (stock price)

The **Black-Scholes model** ([Black & Scholes, 1973](https://www.jstor.org/stable/1831029)) is exactly that recipe. It takes five ingredients and spits out a fair premium.

For covered calls, we'll use Black-Scholes to *estimate* what an option should cost when we can't buy real option data.

### The Five Ingredients

| Ingredient | Symbol | Meaning in CC Context | Example |
| --- | --- | --- | --- |
| **Stock price** | S | Current price of the stock we own | $50 |
| **Strike price** | K | The price at which we'll sell shares | $52 |
| **Time to expiration** | T | Trading days until the contract ends, as a fraction of a year | 30 days = 30/252 |
| **Risk-free rate** | r | What we'd earn if we put money in Treasury bonds | 4% per year |
| **Volatility** | σ (sigma) | How much the stock bounces around | 25% per year |

For a stock at $50:

- If **T = 30 days** and **σ = 25%**, the stock might swing $1–2
- If **T = 365 days** and **σ = 25%**, it might swing $8–12
- Higher volatility = bigger swings = more valuable insurance = higher premium

### Step-by-Step Intuition (Not the Math)

The Black-Scholes formula looks scary:

```text
C = S₀·N(d₁) - K·e^(-rT)·N(d₂)
```

But here's what's *actually* happening:

#### Step 1: Calculate "d₁" — the distance the stock needs to move

```text
d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
```

Translation: "If the stock is at $50 and the strike is $52, how many standard deviations away is that? And how much time do we have?"

A high σ (volatility) makes the denominator huge, so d₁ stays closer to zero. This means the market expects big moves, so the option is worth more.

#### Step 2: Use the normal distribution to convert d₁ to a probability

```text
N(d₁) = probability that the stock ends up in-the-money
```

This requires the **cumulative normal distribution function (CDF)** — the function that converts a z-score into a probability (see [Glossary](#appendix-c-glossary-of-key-terms)). We'll use the [Abramowitz & Stegun approximation](https://en.wikipedia.org/wiki/Abramowitz_and_Stegun).

#### Step 3: Calculate the call option price

```text
C = S·N(d₁) - K·e^(-rT)·N(d₂)
```

Think of it this way:

- **S·N(d₁)** = "If I were to buy the stock outright, but only for the probability it goes up, how much would I pay?"
- **K·e^(-rT)·N(d₂)** = "If I were to pay the strike price, but discounted for interest and adjusted for risk, how much is that?"
- The **difference** is what the option is worth.

### Why We Need It: No Historical Option Data

Here's the hard truth: **We don't have historical option prices.**

If I want to test, "Would I have profited selling SPY calls on January 15, 2015?", I can't just look up what SPY calls cost that day—at least not easily or reliably. The data is expensive or incomplete.

So we *estimate* option prices using Black-Scholes, assuming a volatility level. This is a simplification, but it's the standard approach.

### The Normal CDF (Cumulative Distribution Function) Approximation

The Black-Scholes formula requires the cumulative normal distribution function N(x) — the function that tells us "given a bell curve, what fraction of the area falls to the left of x?"

**The problem:** There's no simple equation you can write down that directly computes this. Unlike, say, the area of a circle (πr²), the area under a bell curve can't be expressed as a neat formula using basic math operations — addition, multiplication, exponents, etc. Mathematicians call this "no closed-form solution." The exact answer requires calculus (computing an integral), which is slow for a computer to do thousands of times per backtest.

**The workaround:** Instead of computing the exact integral every time, mathematicians found that a **polynomial** — a simple expression like `a·t + b·t² + c·t³ + d·t⁴ + e·t⁵` — can mimic the real CDF closely enough. Polynomials are just multiplication and addition, which computers do extremely fast. The trick is picking the right coefficients (a, b, c, d, e) so the polynomial matches the true answer to many decimal places.

The **Abramowitz & Stegun approximation** (from their 1964 math reference handbook) does exactly this — it's accurate to 7 decimal places, which is far more precision than we need for option pricing.

But first, we need one helper — the **PDF (probability density function)**, which gives the *height* of the bell curve at any point. The CDF approximation uses the PDF as a building block:

```python
import math

def normal_pdf(x):
    """
    The height of the bell curve at point x.
    
    CDF = area under the curve (cumulative probability).
    PDF = height of the curve (how likely this exact value is).
    
    The CDF approximation below multiplies the PDF by a polynomial
    to estimate the area.
    """
    # Step by step:
    #   x**2        → always positive (squaring kills the sign)
    #   -x**2       → always negative (flip it)
    #   -x**2 / 2.0 → still negative, just smaller
    #   math.exp(-x**2 / 2.0) → e^(negative) → always between 0 and 1
    #     At x=0: e^0 = 1 (peak of bell curve)
    #     At x=3: e^(-4.5) ≈ 0.011 (curve nearly flat)
    #   / math.sqrt(2 * math.pi) → scale so total area under curve = 1
    return math.exp(-x**2 / 2.0) / math.sqrt(2 * math.pi)

def normal_cdf(x):
    """
    Approximates the cumulative standard normal distribution.
    Accurate to ~7 decimal places (max absolute error ~7.5e-8).
    Uses Abramowitz & Stegun (1964) Formula 26.2.17.
    
    NOTE: These are the same coefficients used in black_scholes.py.
    """
    # Polynomial coefficients (Abramowitz & Stegun, 1964, Formula 26.2.17)
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    
    # For negative x, use symmetry: Φ(-x) = 1 - Φ(x)
    sign = 1 if x >= 0 else -1
    x_abs = abs(x)
    
    t = 1.0 / (1.0 + p * x_abs)
    # Multiply the bell curve height (PDF) by the polynomial to estimate the area (CDF)
    # PDF is symmetric, so pdf(x_abs) == pdf(-x_abs) — the x² kills the sign
    y = 1.0 - normal_pdf(x_abs) * (b1*t + b2*t**2 + b3*t**3 + b4*t**4 + b5*t**5)
    
    return y if sign == 1 else 1.0 - y
```

**Why this works:** The CDF is the area under the bell curve. The approximation says: "take the height of the curve at this point (PDF), multiply by a polynomial correction, and you get a good estimate of the area." It's like estimating the area of a hill by measuring its height and applying a shape factor.

> **Production note:** The A&S polynomial above is shown here because you can read it and *understand* how a CDF approximation works. In actual production code (and in the runnable scripts later in this tutorial) we use Python's built-in `math.erf` instead, via the identity:
>
> ```python
> def normal_cdf(x):
>     return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))
> ```
>
> The C standard library's `erf` is good to ~15 decimals (vs A&S's ~7), which matters when you stack hundreds of thousands of CDF calls in a backtest — the accumulated rounding error in A&S can shift final equity by a few cents. Same algorithm, more precise pipes. We keep the polynomial here for teaching purposes only.

### Delta: The Probability Dial — What 0.20 Δ Actually Means

Delta (Δ) is one of the most misunderstood Greek letters in finance.

**Simple definition:** Delta is the probability that your option ends in-the-money at expiration.

- **Δ = 0.20** → 20% chance the stock rises past the strike → sell a call with 20% ITM risk
- **Δ = 0.50** → 50% chance the stock rises past the strike → sell a call with 50% ITM risk
- **Δ = 0.80** → 80% chance the stock rises past the strike → sell a call with 80% ITM risk (dangerous!)

**In covered call terms:**

- Sell **0.30Δ** = "I'm okay with a 30% chance the stock gets called away"
- Sell **0.50Δ** = "50/50 shot the shares get called away"
- Sell **0.70Δ** = "High chance the shares get called away; this is aggressive"

For income strategies, we typically sell 0.20Δ to 0.40Δ strikes (low probability of assignment).

### Finding a Strike for a Target Delta: The Brute-Force Search Approach

Now the practical question: "I want to sell a 0.25Δ call. Which strike should I pick?"

We work **backwards** from delta:

1. Start with a guess strike (e.g., stock price + 5%)
2. Calculate delta at that strike using Black-Scholes
3. Compare to our target (0.25)
4. If delta is too high, move the strike up (safer)
5. If delta is too low, move the strike down (more aggressive)
6. Repeat until delta ≈ target delta

This is called a **grid search**. It checks every whole-dollar strike in a range and picks the one whose delta is closest to the target. With only ~30–50 candidates to check, it runs instantly — and it naturally returns whole-dollar strikes that match real option chains.

The production implementation is [`cc_backtest.py::find_strike_for_delta`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L83). It scans whole-dollar strikes in `[0.80·S, 1.02·S]` for puts and `[0.98·S, 1.25·S]` for calls — the asymmetric ranges cover the relevant out-of-the-money zone for each option type while keeping the candidate count small. For each candidate it calls `bs_delta(...)` and tracks the strike whose computed delta is closest to `target_delta`.

**Example run:**

- Stock at $100, want 0.25Δ, 30 days out, σ=20%
- Grid search checks every whole-dollar strike from $98 to $125
- $105 has delta ≈ 0.28, **$106** has delta ≈ 0.23 — $106 is closest to 0.25
- Returns strike = $106, delta = 0.23

### Code Walkthrough: bs_price(), bs_delta(), find_strike_for_delta()

The full Black-Scholes toolkit — `normal_pdf`, `normal_cdf`, `bs_price`, `bs_delta`, and `find_strike_for_delta` — lives in [`cc_backtest.py`'s section 1](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L11). What each one does:

| Function | What it computes |
| --- | --- |
| `normal_pdf(x)` | Height of the standard-normal bell curve at `x`. Used inside the A&S polynomial CDF approximation shown earlier. |
| `normal_cdf(x)` | Area under the bell curve from `-∞` to `x` — converts a z-score into a probability. Production uses `math.erf` (~15 decimals) rather than the polynomial (~7 decimals); the educational section above shows the polynomial so you can see *how* a CDF approximation works, and the production docstring explains why the switch matters at scale (A&S's 8th-decimal error compounds into a few cents of equity drift across hundreds of thousands of CDF calls). |
| `bs_price(S, K, T, r, sigma, option_type='put')` | The Black-Scholes formula itself — returns the option premium given stock, strike, time, rate, vol, and option type. |
| `bs_delta(S, K, T, r, sigma, option_type='put')` | Just `N(d1)` for calls or `N(d1) − 1` for puts — the probability of finishing ITM (and the option's first-derivative sensitivity to stock price). |
| `find_strike_for_delta(S, T, r, sigma, target_delta, option_type='put')` | Grid search across whole-dollar strikes; returns the one whose Black-Scholes delta is closest to `target_delta`. Whole-dollar because real option chains list whole-dollar strikes. |

**How to use it:**

```python
# Current stock price is $100
S = 100
# We want to sell a 0.25-delta call, 30 days out
target_delta = 0.25
T = 30 / 252  # 30 trading days as fraction of year (252 trading days/year)
r = 0.04  # 4% interest rate
sigma = 0.20  # 20% volatility

# Find the strike (note: arg order matches black_scholes.py)
strike = find_strike_for_delta(S, T, r, sigma, target_delta, option_type='call')
actual_delta = bs_delta(S, strike, T, r, sigma, option_type='call')
print(f"Strike: ${strike:.0f}, Delta: {actual_delta:.4f}")

# Calculate the premium for that strike
premium = bs_price(S, strike, T, r, sigma, option_type='call')
print(f"Premium: ${premium:.2f} per share (${premium*100:.0f} per contract)")

```

**Output:**

```text
Strike: $106, Delta: 0.2294
Premium: $0.89 per share ($89 per contract)
```

Note: The delta won't be exactly 0.25 after rounding to a whole dollar — that's normal. Real strikes come in fixed increments, so you pick the closest one to your target delta.

### Common Mistake: Confusing Historical Volatility with Implied Volatility

Here's where most people get confused.

**Historical volatility (HV)** = How much the stock bounced around in the past

- Example: "SPY moved ±1% per day on average over the last 30 days"
- That's a daily standard deviation of 0.01 (1%)
- Annualize it: 0.01 × √252 = 0.01 × 15.87 = **0.159 ≈ 16%**
- (Why √252? Standard deviations scale with the square root of time, not linearly. 252 = trading days per year.)

**Implied volatility (IV)** = What the market *expects* will happen

- Example: "Call buyers are willing to pay premiums that assume 18% volatility" → IV ≈ 18%

In our backtest, we **don't have historical option prices**, so we can't extract IV. Instead, we use HV as a proxy, then adjust it.

**The relationship:**

- When the market is calm, IV might be *lower* than HV (people expect calm)
- When the market is nervous, IV might be *higher* than HV (people expect chaos)

### The IV Proxy: Why a Regime-Based Multiplier Works

In practice, we calculate **rolling historical volatility** and multiply by a **regime-dependent** factor — higher when vol is low (markets underpricing risk), lower when vol is already elevated (IV converges toward HV).

The three helpers — [`calc_rolling_volatility`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L126), [`detect_regime`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L161), and [`estimate_iv`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L170) — are implemented in `cc_backtest.py`. Part 3 walks through both in detail (see *Rolling Historical Volatility* for the log-returns identity, Bessel's correction, and the √252 annualization derivation; see *The Dynamic IV Multiplier* for the regime → multiplier table). The skeleton in plain English:

- For each day, `rolling_vol = std_dev(last 30 log returns) × √252` (annualized).
- Classify the regime: `"high"` if `rolling_vol > 25%`, `"low"` if `< 15%`, else `"normal"`.
- Apply the multiplier: `iv_estimate = rolling_vol × {high: 1.1, normal: 1.3, low: 1.5}[regime]`.

In the production engine `estimate_iv(rolling_vol)` does steps 2 and 3 together — pass it a vol, get back an IV estimate.

**Why these multipliers?**

- Empirically, implied volatility tends to be 20–40% higher than realized volatility
- But the gap **varies by regime**: when vol is already high, IV doesn't spike as much above HV; when vol is low, IV tends to stay well above HV (mean-reversion pricing)
- The regime-based approach (1.1×/1.3×/1.5×) captures this dynamic better than a flat constant

**When this still breaks down:**

- **Before earnings:** IV spikes way above HV (the market expects a big move)
- **After a crash:** HV explodes but IV might normalize faster (panic recedes)
- **In persistent trends:** HV might be high (the stock is moving a lot) but IV might be low (it's moving in one direction, so options are more predictable)

We implement this regime-based approach in Part 3's `run_cc_overlay()` engine.

---

## Part 3: The Covered Call Overlay Engine

### The Key Insight That Changed Everything: "Never Sell Your Shares"

This is the most important rule of covered call backtesting.

**Mistake:** Selling a 0.60Δ call, hoping the stock goes down, so you keep the premium AND the shares. If it rises above the strike, the shares get called away at a loss.

**Right approach:** Sell a 0.25Δ call. You *expect* the shares to get called away 25% of the time. That's fine — it's built into the premium you collect. You own the upside up to the strike, then the shares leave. You're okay with this.

> **Analogy:** You own a rental property worth $300k. You lease it for $2k/month, expecting a 10% chance per year the lease breaks early. The early-break risk is built into your rental decision. You don't twist the terms hoping the tenant never leaves — that's not the business you're in.

In a covered call overlay, you're in the **income business**, not the **capital appreciation business**.

### How the Overlay Works Day by Day: Walk Through a Week

Let's simulate a realistic week:

**Monday, Jan 6, 2025:**

- Own 100 shares of ABC at $50
- ABC is up 5% over the last month → HV ≈ 18%
- IV estimate = 18% × 1.3 = 23.4%
- **Decision:** Sell a call
  - Target delta = 0.25 (25% chance of ITM)
  - Strike = $53 (found via grid search — closest whole-dollar strike to 0.25Δ)
  - Premium = $0.64
  - Net credit = $0.64 × 100 = $64, minus $0.65 commission = **$63.35**
- **State:** OPEN (call is active)

**Tuesday, Jan 7:**

- ABC closes at $51
- Call is still OTM ($51 < $53 strike), delta ≈ 0.30
- Option has decayed slightly — time decay is working in our favor
- Check profit target: not yet at 75% of premium captured
- **Decision:** Hold (not at target yet)

**Wednesday, Jan 8:**

- ABC rallies to $51.50
- The call is more expensive now — stock moved toward the strike
- **Decision:** Hold (waiting for time decay to work in our favor)

**How "close at 75% profit" works:**

When you sell a call:

- You receive the premium now: $64
- You *might* have to buy it back later at a higher price (loss)
- You *might* get assigned at expiration (shares sold at the strike)

If we want a "75% return on the premium," we close when the option has lost 75% of its value:

- Collected: $64
- Close when option worth: $64 × 0.25 = $16 (we keep 75%, buy back for 25%)
- If option is still worth $50, we haven't hit our target yet
- **Decision:** Hold, waiting for more decay

**Thursday, Jan 9:**

- ABC drops back to $50.50
- Check expiration: 23 days left (not close to expiration)
- Option still worth more than $16 target — not at 75% profit yet
- **Decision:** Hold, waiting for target or expiration

**Friday, Jan 10:**

- ABC stays at $50.50
- Check expiration: 22 days left
- Option still decaying but not at 75% profit target
- **Decision:** Hold

**Wednesday, Jan 15 (expiration week, 8 days out):**

- ABC is at $50
- Call is now OTM (delta ≈ 0.10) and worth ~$0.10
- We sold at $0.64; if we buy back now, we pay $0.10
- Profit: $0.64 - $0.10 = $0.54 = **84% return on the premium**
- That exceeds our 75% target — time to close
- **Decision:** Close the position, lock in 84% profit, reset for next month

**Friday, Jan 17 (expiration day):**

- Call expires worthless (ABC is still below $53)
- Shares are still ours
- **State:** RESET (ready to sell another call)

### The State Machine: OPEN → Check → Handle → Reset

Here's the logic:

```text
IDLE (no open call)
  ↓
[Sell call at 0.25Δ]
  ↓
OPEN (call is active)
  ├─ [Expiration reached: days_left ≤ 0?] → YES → settle (assigned if price ≥ strike, else expires worthless)
  ├─ [Check profit target: 75% of premium captured?] → YES → close and RESET
  ├─ [Check ITM assignment risk: delta > 0.70?] → YES → close and RESET
  └─ [Hold and check again tomorrow]
  ↓
RESET (sold and closed; ready for next call)
  └─ [Wait 1 day, then go back to IDLE]
```

Sketched as a per-day handler (the real `run_cc_overlay` inlines this loop body — `run_cc_overlay_day` is illustrative, not a function in the codebase):

```python
def run_cc_overlay_day(
    current_date,
    position_state,  # None, or {'entry_price', 'entry_date', 'dte', ...}
    current_price,
    rolling_vol,
    current_rate,
    call_delta=0.20,
    close_at_pct=0.75,
):
    """
    Decide what to do with the covered call position on this day.
    
    Returns:
        action: 'idle', 'sell', 'close', 'assign' or 'hold'
        premium: if 'sell', the premium collected
        result_price: if 'close' or 'assign', the result price
    """
    
    # Case 1: No open position, check if we should sell
    if position_state is None:
        # Decide on strike
        T = 30 / 252  # 30 DTE is our target (252 trading days/year)
        strike = find_strike_for_delta(
            current_price, T, current_rate, rolling_vol, call_delta, option_type='call'
        )
        premium = bs_price(current_price, strike, T, current_rate, rolling_vol, option_type='call')
        
        return 'sell', premium, strike
    
    # Case 2: Open position; check conditions
    premium_collected = position_state['premium']
    strike = position_state['strike']
    entry_date = position_state['entry_date']
    dte = position_state['dte']
    
    # Recalculate call value today
    T_remaining = dte / 252
    current_call_value = bs_price(current_price, strike, T_remaining, current_rate, rolling_vol, option_type='call')
    
    # Profit check: has enough premium been captured?
    # close_at_pct = 0.75 means close when option worth <= 25% of what we sold it for
    if current_call_value <= premium_collected * (1 - close_at_pct):
        return 'close', current_call_value, strike
    
    # Expiration check
    if dte <= 0:
        if current_price >= strike:
            return 'called_away', current_call_value, strike
        else:
            return 'expired_otm', current_call_value, strike
    
    # Condition: Call is deeply ITM (delta > 0.70)? Consider closing.
    if current_price > strike:
        delta_today = bs_delta(current_price, strike, T_remaining, current_rate, rolling_vol, option_type='call')
        if delta_today > 0.70:
            return 'close', current_call_value, strike
    
    # Otherwise, hold
    return 'hold', None, None
```

### Transaction Costs: Commission ($0.65/contract) + Slippage (3% of Premium)

This is where backtests often lie.

**Reality:**

- You pay $0.65 per contract to open ($65 for a 100-share contract, or $0.65 per share)
- You pay $0.65 per contract to close
- You have slippage: the bid-ask spread might mean you sell the call for 95¢ but it's worth $1.00

In our model:

```python
def apply_transaction_costs(premium, cost_per_contract=0.65, slippage_pct=0.03):
    """
    Reduce premium by transaction costs (per-share basis).
    
    Args:
        premium: option premium per share
        cost_per_contract: commission per contract ($0.65 typical)
        slippage_pct: bid-ask slippage as % of premium
    
    Returns:
        net_premium: premium after costs (per share)
    """
    # Slippage: reduce by 3% of the premium
    slippage_cost = premium * slippage_pct
    
    # Commission: per-contract cost
    # If we're selling 1 contract (100 shares), commission is $0.65
    # Per-share basis: 0.65 / 100 = $0.0065
    commission_per_share = cost_per_contract / 100.0
    
    net_premium = premium - slippage_cost - commission_per_share
    
    return net_premium
```

**Example:**

- Black-Scholes says the call is worth $1.00
- Slippage (3%): lose $0.03
- Commission on open (0.65 per contract = $0.0065 per share): lose $0.0065
- **Net credit:** $1.00 - $0.03 - $0.0065 = **$0.9635**

Over a year with 12 calls sold, transaction costs can eat 5–10% of returns.

### The Dynamic IV Multiplier: Context Matters

We use a simple regime-based IV multiplier. `detect_regime(rolling_vol)` classifies the current 30-day annualized HV into one of three buckets, and `estimate_iv(rolling_vol, regime)` applies the appropriate multiplier:

| Regime | HV range | Multiplier | Why |
| --- | --- | --- | --- |
| **High** | > 25% | 1.1× | IV is already elevated; further expansion is limited. |
| **Normal** | 15–25% | 1.3× | Typical HV-to-IV adjustment in calm markets. |
| **Low** | < 15% | 1.5× | IV is suppressed; expect mean reversion to higher values. |

Implementations: [`cc_backtest.py::detect_regime`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L161) and [`::estimate_iv`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L170).

### Rolling Historical Volatility: 30-Day Window, Log Returns, Annualize

The implementation is [`cc_backtest.py::calc_rolling_volatility`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L126) — for each price index, it computes the standard deviation of the last `window` log returns and annualizes by `√252`.

Four pedagogical notes worth pulling out, because they show up in every volatility-related calculation in this codebase:

1. **Log returns vs. simple returns.** `np.diff(np.log(prices))` computes `ln(price_t / price_{t-1})` for each day. The identity `log(a) − log(b) = log(a/b)` is what makes this work. Log returns are *additive across days* (you can sum them to get multi-day returns) and *symmetric* (a +5% followed by a −5% nets to zero in log space). Note that `log(diff(prices))` is *not* the same thing and will break on any negative price change.

2. **NaN padding for alignment.** The first `window - 1` indices of the output get `NaN` because there aren't enough prior return observations to fill the window yet (with a 30-day window, you need at least 30 returns before the first valid volatility). NaN-padding keeps the output array index-aligned with the input price series, so downstream lookups stay correct.

3. **Bessel's correction (`ddof=1`).** The window's return values are a *sample* from the stock's theoretical distribution, not the population. Dividing by `N-1` instead of `N` corrects for the bias introduced when the sample mean is computed from the same data you're measuring deviation from. For `N = 30` the correction is small (about 3% larger std dev), but it's the statistically correct choice.

4. **Annualize by `√252`, not `252`.** Variance (`σ²`) is additive over independent time periods, so `σ²_annual = σ²_daily × 252`. Taking square roots: `σ_annual = σ_daily × √252`. Standard deviations scale with the *square root* of time, not linearly. This is one of the most-confused identities in finance.

**Example:**

- Last 30 daily log returns have a standard deviation of 1.2%
- Annualized: 1.2% × √252 ≈ 1.2% × 15.87 ≈ **19%**
- IV estimate: 19% × 1.3 = **24.7%**

### The SMA Trend Filter: 50-Day and 200-Day Moving Averages

Some wheel traders use a trend filter to decide *when* to sell options. The key insight: the filter matters more for **which phase** you're in.

**CSP phase (selling puts) — avoid downtrends:** If the stock is falling, you don't want to sell puts and get assigned at a price that keeps dropping. Wait for stabilization (SMA50 > SMA200) before selling puts.

**CC phase (selling calls) — sell in any trend:** If you already hold shares and the stock is declining, selling calls is *exactly* what you want. You collect premium, reduce your cost basis, and the calls expire worthless (the stock isn't rising to your strike). Not selling calls in a downtrend means sitting on losses with no income to cushion them.

**CC phase — strong uptrends are the real risk:** If the stock is surging, your call gets exercised and you're called away, capping your upside. But in the wheel, getting called away just cycles you back to selling puts — so it's not a disaster, just a missed rally.

> **Note — the big tradeoff vs. buy-and-hold:** Premiums are small and steady; rallies are rare and huge. If you repeatedly get called away during strong uptrends, the capped upside compounds against you, and the strategy can materially **underperform a pure buy-and-hold** of the same stock. Covered calls trade lottery-ticket upside for consistent income — that's the deal, and it only looks good if you actually prefer smoother returns to maximizing total return.

```python
def sma(prices, window):
    """Simple moving average: take the last `window` prices, return their average."""
    # prices[-window:] grabs the last N prices from the list
    # e.g. if window=50 and prices has 1000 entries, this averages the last 50
    return np.mean(prices[-window:])

def is_uptrend(prices, sma_short=50, sma_long=200):
    """
    Check if stock is in uptrend using golden cross.
    
    Returns:
        True if SMA50 > SMA200 (uptrend), else False
    """
    if len(prices) < sma_long:
        return True  # Not enough data; assume neutral
    
    sma_50 = sma(prices, sma_short)
    sma_200 = sma(prices, sma_long)
    
    return sma_50 > sma_200
```

**In the overlay:**

```python
if trend_filter_enabled and not is_uptrend(prices):
    # CSP phase: skip selling puts in downtrend (avoid assignment into falling stock)
    # CC phase: you'd typically STILL sell calls here to reduce cost basis
    return 'idle', None, None
```

**Empirical finding (from our walk-forward results):** The trend filter didn't help much. The filter's job is to pause CSPs when SMA50 < SMA200 (a downtrend) so you don't get assigned into a falling stock — but the wheel is defensive enough that even entering in a downtrend works out: premiums cushion the drawdown, and once you're assigned you just start collecting CC income on the way back up. We'll include the filter as an option but not use it by default.

### The Run_cc_overlay() Function: Full Walkthrough

The core backtesting engine lives in [`cc_backtest.py::run_cc_overlay`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L201). It's heavily commented and small enough to read end-to-end. Function signature:

```python
def run_cc_overlay(
    dates: list[str] | NDArray[Any],
    prices: NDArray[np.floating[Any]],
    params: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
```

The function takes the price series and the strategy parameters and returns `(summary, trades, daily_equity)` — a summary dict with the headline metrics, a list of every trade with its action/price/P&L, and the day-by-day equity curve.

**What the function does on each trading day** (the inner loop):

1. **Compute today's rolling volatility** from a 30-day window of log returns. During the first three days fall back to 20% annualized as a warm-up baseline.
2. **Pick an IV estimate** by multiplying the rolling HV by a regime-based factor (1.1× in high vol, 1.3× normal, 1.5× low) — see `detect_regime()` and `estimate_iv()`.
3. **If no position is open:** find the strike whose Black-Scholes delta matches `params['call_delta']` via grid search, price the call, apply transaction costs (3% slippage + $0.65 commission), and open the position. Skip opening if the net premium would be negative.
4. **If a position is open:** check three close conditions, in this order:
    - **Expiration reached** (`days_left ≤ 0`): settle as assigned (if ITM, the buyer exercises and we rebuy shares at market) or expired worthless (we keep the premium and the shares).
    - **Profit target hit** (call has lost `close_at_pct` of its value, default 75%): buy back the call and book the gain.
    - **Deep ITM** (`delta > 0.70`): assignment is now likely; close early to free up capital and limit gamma damage.
5. **Mark-to-market** today's equity = stock value + idle cash + cumulative overlay P&L (plus any unrealized P&L on an open position), and append to the day-by-day equity curve.

That's the whole loop. The state-machine diagram earlier in this section is the visual counterpart; the source code is the executable one.

At the end of the loop, the function tallies summary statistics — total return, buy-and-hold benchmark, gross premium collected, buybacks and assignment costs, premium retention %, calls sold, win rate, max drawdown — and returns the three result objects.

### Common Mistake: Letting Shares Get Called Away vs. Buying Back ITM Calls

**Mistake A:** Sell a 0.50Δ call, hoping to keep the shares. The stock rockets up. You're now forced to sell at the strike, feeling like you "missed out" on the upside.

**Reality:** You were running a 50/50 bet on assignment. It happened. That's not a mistake; it's the business you signed up for. The premium you collected compensated for the upside risk.

**Mistake B:** The call goes ITM (delta > 0.60), and you panic-buy it back even though you have 10 days to expiration.

**Reality:** Let it ride. The closer to expiration, the faster the call loses value. If you wait 5 more days and nothing happens, the call might be worth 50% less.

**Principle:** Treat covered calls like an **income strategy**, not a market-timing strategy. You sold insurance at a price you thought was fair. Let the contract play out unless:

1. You hit your profit target (75% of premium captured), or
2. The call has gone deep ITM (delta > 0.70) and assignment is now very likely — close to free up capital before gamma compounds the damage

---

## Part 4: Walk-Forward Optimization

### The Analogy: It's Like Studying for a Test, Then Taking a Different Test

**Bad approach (in-sample optimization):**

1. Get a practice test (2010–2020)
2. Memorize all the answers (tweak your strategy on this exact data)
3. Take the real test (2010–2020)
4. Pat yourself on the back: "I got 100%!"
5. Take another test (2021–2026) — suddenly you fail

**Right approach (walk-forward validation):**

1. **Split your data into a training window and a test window** (e.g., train on 2010–2017, test on 2018–2019)
2. **Optimize your strategy on the training window** — build intuition, tune parameters, develop rules
3. **Lock those rules and score yourself on the test window** — no peeking, no re-tuning
4. **Roll both windows forward by the same step** (e.g., train on 2012–2019, test on 2020–2021)
5. **Repeat**: optimize on the new training window, then score on the new test window
6. **Keep rolling** until you've exhausted your data (e.g., train on 2014–2021, test on 2022–2023)
7. **Average all the out-of-sample test scores** — that average is your realistic estimate of future performance

Walk-forward is not perfect, but it's the best *single* tool we have for avoiding overfitting. Its limitations:

- **You can overfit the walk-forward itself.** The choice of training window length, test window length, step size, and optimization metric are all meta-parameters. If you try 10 different walk-forward configurations and pick the best one, you've just moved the overfitting up one level.
- **Regime changes break the assumption.** Walk-forward assumes the near future resembles the near past. But markets undergo structural shifts (COVID crash, 2008 crisis, interest rate pivots from 0% to 5%). A strategy optimized on 2016–2018 calm markets has no way to prepare for March 2020.
- **Limited data gets sliced too thin.** With 10 years of daily data, each training window might be too short to capture a full bull/bear cycle, and each test window might be too short for results to be statistically meaningful — you could "pass" a 6-month test by luck.
- **It validates parameters, not strategy design.** Walk-forward tells you "delta 0.25 is better than delta 0.30." But you *chose* to run covered calls with delta-based strike selection — that architectural decision was made with hindsight knowledge about what kinds of strategies tend to work.

That's why Part 5 layers additional robustness checks on top: Monte Carlo, sensitivity analysis, regime testing, and more.

### Why In-Sample Optimization Lies: The "Peeking at the Answer Sheet" Problem

Imagine you're optimizing a covered call strategy. You have 10 years of data (2016–2026). You ask:

"What delta (0.20, 0.25, 0.30) gave the best returns?"

If you test all three on the same 2016–2026 data and pick the winner, you're peeking at the answer sheet. The delta that worked best is partly because of *luck* during those specific 10 years. It might not work as well forward.

**The problem gets worse** with more parameters:

- Delta: 5 choices (0.15, 0.20, 0.25, 0.30, 0.35)
- DTE target: 4 choices (14, 21, 28, 35 days)
- Profit target: 3 choices (10%, 20%, 30%)
- Trend filter: 2 choices (yes/no)
- **Total combinations:** 5 × 4 × 3 × 2 = **120 parameter sets**

Test all 120 on 2016–2026 data, pick the top performer, and you've almost certainly overfit. By random chance alone, some strategies will look great.

Walk-forward prevents this by using *different* data for training and testing.

### The Walk-Forward Structure: 2-Year Train → 6-Month Test → Roll Forward

Here's the idea:

```text
Training window        Testing window
[Apr 2016 - Mar 2018] [Apr 2018 - Sep 2018] (6 months)
                                               ↓ roll forward
                      [Oct 2016 - Sep 2018] [Oct 2018 - Mar 2019]
                                               ↓ roll forward
                                    [Apr 2017 - Mar 2019] [Apr 2019 - Sep 2019]
                                    ... etc ...
```

For each **training window:**

1. Test all parameter combinations
2. Pick the one with best Sharpe ratio
3. Use that parameter set on the **testing window**
4. Record the result

Finally, stitch all testing results together into one equity curve.

### The Parameter Grid: What We Search Over and Why

```python
param_grid = {
    'call_delta': [0.15, 0.20, 0.25],
    'dte': [21, 30, 45],
    'close_at_pct': [0.50, 0.75, 1.0],
}

# Why these ranges?
# call_delta: 0.15 = conservative (rarely called away), 0.25 = aggressive (more premium)
# dte: 21 = fast-moving, frequent sales; 45 = slower, higher premiums
# close_at_pct: 0.50 = close when 50% of premium captured; 1.0 = hold to expiry
#
# Note: put_delta is NOT included here. This is a covered call overlay
# backtest — we already own the shares and are only selling calls. The
# put_delta parameter belongs to the CSP (cash-secured put) entry phase
# of the full wheel strategy, which we aren't testing here.

def param_combinations(grid):
    """
    Turn a dict of lists into every possible combination.
    
    Input:  {'call_delta': [0.15, 0.20], 'dte': [21, 30]}
    Output: [{'call_delta': 0.15, 'dte': 21},
             {'call_delta': 0.15, 'dte': 30},
             {'call_delta': 0.20, 'dte': 21},
             {'call_delta': 0.20, 'dte': 30}]
    
    Each factor in the product is the number of options for one parameter:
      call_delta:   3 choices ([0.15, 0.20, 0.25])
      dte:          3 choices ([21, 30, 45])
      close_at_pct: 3 choices ([0.50, 0.75, 1.0])
    Total: 3 × 3 × 3 = 27 combos. This generates all 27 parameter sets
    so the optimizer can try each one.
    """
    import itertools
    
    keys = list(grid.keys())           # ['call_delta', 'dte', 'close_at_pct']
    values = list(grid.values())       # [[0.15, 0.20, 0.25], [21, 30, 45], ...]
    
    for combo in itertools.product(*values):  # itertools.product gives every combination
        yield dict(zip(keys, combo))          # zip pairs each key with one value from the combo
        # e.g., zip(['call_delta', 'dte'], (0.15, 21)) → {'call_delta': 0.15, 'dte': 21}
```

### How to Stitch Out-of-Sample Results into a Single Equity Curve

The implementation is [`cc_backtest.py::walk_forward_optimization`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L827). Signature:

```python
def walk_forward_optimization(
    dates: list[str],
    prices: NDArray[np.floating[Any]] | list[float],
    param_grid: dict[str, list[float]],
    fixed_params: dict[str, float] | None = None,
    train_years: int = 2,
    test_months: int = 6,
    roll_months: int = 6,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
```

The function takes the price series, a parameter grid (dict mapping parameter name to candidate values), and the window-sizing knobs. It returns `(oos_equity, period_records)` — the stitched out-of-sample daily equity curve, and a list of dicts describing each iteration's train/test bounds (ISO date strings), the chosen `best_params`, and the in-sample training Sharpe that won.

What it does per iteration:

1. Slice `[train_start, train_end)` and `[test_start, test_end)` from the date series. The half-open intervals guarantee the boundary date `current_date` belongs to exactly one window — `test_start == train_end`, never both. This is the central guard against in-sample overfitting: the parameters evaluated on a test window are chosen *without ever seeing* that test window's data.
2. Loop over every combination from `param_grid`. For each combo, run the overlay on the training window and compute the annualized Sharpe of daily returns (`mean / std × √252`). Keep the highest-Sharpe combo.
3. Run those locked params on the out-of-sample test window. Append the resulting daily equity to the stitched curve.
4. Advance `current_date` by `roll_months` and repeat until the next test window would run past `end_date`.

The production implementation in [`cc_backtest.py::walk_forward_optimization`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L827) is heavily commented — it carries the teaching content (window arithmetic diagram, boolean-indexing explainer, Sharpe-built-inside-out walkthrough, Bessel's correction, √252 annualization derivation, "rules are LOCKED — no re-tuning" emphasis) right next to the code that does the work. The fixing test [`test_cc_backtest.py::TestMsftTenYearRegression::test_walk_forward_optimization`](https://github.com/l3a0/covered-call-backtesting/blob/main/test_cc_backtest.py#L1214) pins the 15 walk-forward periods, the most-chosen parameters, and the cumulative OOS compound return on the bundled MSFT data.

### What the Optimizer Chose

Running walk-forward on the bundled MSFT data with the 3×3×3 grid produces 15 OOS test periods. The optimizer's choices per period (pinned by `test_walk_forward_optimization`):

| Parameter | Most-chosen value | Counts across 15 periods |
| --- | --- | --- |
| **call_delta** | **0.25** | 0.25 × 14, 0.20 × 1, 0.15 × 0 |
| **dte** | **21** | 21 × 9, 30 × 4, 45 × 2 |
| **close_at_pct** | **0.75** | 0.75 × 11, 0.50 × 2, 1.00 × 2 |

All three winners match the `__main__` defaults: `0.25Δ`, `21 DTE`, and `0.75 close_at_pct`. The walk-forward optimizer searches 27 combinations across 15 disjoint out-of-sample periods and keeps landing on the same configuration the rest of the tutorial uses. That convergence is a small piece of evidence that these defaults are what an honest, no-peeking search settles on across very different market windows.

**Why these defaults make sense:**

1. **0.25Δ** is the sweet spot:
   - Conservative (0.15Δ) misses too much premium
   - Aggressive (0.35Δ) gets assigned too often
   - 0.25 balances "collect income" with "keep the shares"

2. **21 DTE** is the monthly rhythm:
   - Matches typical options expiration cycles
   - Gives enough time for the trade to work out
   - Allows 4–5 cycles per year for reinvesting premiums

3. **75% profit target** is the sweet spot for closing:
   - Captures most of the premium decay without holding through the gamma-heavy final stretch
   - Closing earlier (50%) leaves real income on the table; holding to expiry (100%) means riding through the period where assignment risk is highest and time decay slows
   - Walk-forward picks 0.75 in 11 of 15 periods — the optimizer keeps choosing it across very different market windows

4. **Deep-ITM close at delta > 0.70** caps assignment damage:
   - When the call goes deep ITM, gamma is steep and a small adverse move can wipe out months of premium income
   - Closing early at the 0.70-delta threshold gives up the last sliver of time value to escape before assignment crystallizes the full upside loss

5. **No trend filter** is surprising:
   - In the CC phase, selling calls in a downtrend is actually *desirable* — it reduces your cost basis and generates income while you wait for recovery
   - Premiums are richest during downtrends (high vol), so that's when call selling is most rewarding
   - A trend filter mainly helps the CSP phase (avoid selling puts into a falling market), but the backtest found it wasn't worth the complexity

### The Key Finding: Walk-Forward Tells the Honest Story

**Result on the bundled MSFT data, over the walk-forward span (2018-04 → 2025-10):**

- **Walk-forward** (params optimized per period, 6-month OOS windows chained): **~483%** cumulative compound return.
- **Fixed params** (`0.25Δ`, `21 DTE`, `0.75 close`) over the same span: **~563%** total return.

Walk-forward **underperformed** fixed-params by about 80 percentage points (~14% relative). That sounds bad until you notice what the fixed-params number actually represents: the return *given that you somehow knew, before seeing any of this data, that those exact three parameters would be the winners on this 7.5-year window*.

The walk-forward number is the return you'd have actually achieved running this strategy in real time, with no peeking. The gap is the cost of not having hindsight — which is to say, the realistic expected return.

**The pedagogical point isn't "walk-forward gets you a better number."** It's the opposite: **fixed-params backtests systematically overestimate the strategy's return; walk-forward gives you the number you'd actually have achieved**. If you see a strategy that "outperforms" in a single full-period backtest, walk-forward will often pull the headline number lower — that's the methodology working, not failing.

This also clarifies the right reading of the headline 915% number reported elsewhere in this tutorial. That number is the fixed-params total return over the *full* 10-year MSFT sample (2016 → 2026), which includes 2 extra years on either side of the walk-forward span. The 483% / 563% comparison above is the apples-to-apples one inside the walk-forward window.

### Common Mistake: Optimizing on Too Many Parameters (Overfitting the Grid)

If you optimize on 500 parameter combinations, some will look amazing by pure luck.

**Red flag:**

- "I tested 500 parameter sets and the best one has 250% returns"
- "The second-best is 120% and the third-best is 45%"
- **There's a huge drop-off between best and second-best → overfitting**

**Healthy pattern:**

- Best: ~1,050% return
- Second-best: ~1,030% return
- Third-best: ~1,010% return
- **Close together → robust, not overfit**

**To avoid overfitting:**

1. Use walk-forward (tested on different data than training)
2. Look for stability across parameter values
3. Test fewer parameters (coarse grid first, then fine-tune)
4. Use cross-validation (train on A, test on B, train on B, test on A, average)

---

## Part 5: Robustness Checks — Proving It's Not Luck

A backtested strategy is only as good as your confidence that it will work forward. These tests check if the returns are **real** or just **lucky**.

### Monte Carlo Simulation: Shuffle Daily Returns, Rebuild Price Paths

**Idea:** If you randomly shuffle the daily price returns and rebuild synthetic price paths, does the strategy still make money?

If yes → the strategy exploits the return *distribution* (robust, real skill)

If no → the strategy exploits a specific *sequence* of returns (could be luck)

**Why this works:** Real prices have a specific order — trends, mean-reversion, volatility clusters. Shuffling destroys that order while keeping the exact same set of daily returns (same mean, same volatility, same distribution). So if your strategy profits on both real and shuffled paths, it's capturing **statistical properties** of the returns (e.g., collecting premium in a volatile market) — those survive shuffling. But if it only works on the real path, it was exploiting the **specific sequence** — like selling calls right before drops and not selling before rallies. That pattern won't repeat, so it's likely overfitting or luck. Think of it like poker: if you win with many random deals, you have real skill. If you only win with the exact hand order you practiced on, you just memorized that deck.

The reference implementation lives in [`test_cc_backtest.py::TestMsftTenYearRegression::test_monte_carlo_shuffle`](https://github.com/l3a0/covered-call-backtesting/blob/main/test_cc_backtest.py#L1029) — a test that re-runs the full shuffle on the bundled MSFT data (500 paths, `seed=42`, `__main__` params) and pins the resulting percentile, MC mean, and best-shuffled return. The algorithm in one line: compute daily returns from the real prices, shuffle their order with a fixed seed, rebuild a synthetic price path from each shuffled sequence, run the overlay backtest on each synthetic path, then compare the real ordered path's total return against the distribution.

**Our result (`__main__` params on the bundled MSFT data, 500 shuffles, seed=42):**

- Real return: ~915%
- MC mean: ~657% (average across 500 shuffled paths)
- MC percentile: 100 (our strategy beat 100% of random shuffles — the real return is higher than every single shuffled path's, with the shuffle max at ~870%)
- This means: 0% of random price orderings produced a better return than our strategy did on the real price path.

**Interpretation:** The strategy beats randomized price paths — it exploits real price patterns, not just luck. A percentile above 80 indicates genuine skill.

### Sensitivity Analysis: Perturb Each Parameter, See If Results Collapse

**Idea:** Unlike a grid search (which tries many combinations to find the *best* params), sensitivity analysis starts from already-chosen params and nudges *one at a time* to check *stability*. Grid search answers "what's optimal?" — sensitivity analysis answers "how fragile is that optimum?" If returns change drastically from a small tweak, you're overfitting that parameter. A robust strategy should stay in a similar range across small perturbations.

The reference implementation lives in [`test_cc_backtest.py::TestMsftTenYearRegression::test_sensitivity_perturbations`](https://github.com/l3a0/covered-call-backtesting/blob/main/test_cc_backtest.py#L977) — a parameterized test that sweeps `call_delta` and `close_at_pct` at ±0.05 / ±0.10 / ±0.20 offsets from base and pins each variant's total return, plus asserts the worst drop from base stays under 10% (the "robust" verdict). The algorithm: hold all params fixed except one, vary that one by a small offset in both directions, measure each variant's total return, then compute the worst drop from base and the full-range swing.

**Example output** (`__main__` params on the bundled MSFT data, run against the current engine):

```text
call_delta sensitivity:
  -0.10: 837%   -0.05: 827%   base: 915%   +0.05: 900%   +0.10: 904%
  Swing: 87 pp (max−min) ≈ 10% of base; worst drop from base is 87 pp.

close_at_pct sensitivity:
  -0.20: 946%   -0.10: 956%   base: 915%   +0.10: 857%   +0.20: 902%
  Swing: 100 pp (max−min) ≈ 11% of base; worst drop from base is 58 pp ≈ 6%.

Strategy is ROBUST: both params produce single-digit-percent drops under
realistic perturbations. Worth noting: the base config isn't always the
optimum — close_at_pct=0.65 outperforms the default 0.75 by ~42 pp here,
hinting at a small in-sample optimization opportunity (which walk-forward
in Part 4 lets you exploit honestly without overfitting).

Math behind the call_delta sensitivity:
  base = 915%, worst variant = 827% (at -0.05 offset, i.e., 0.20Δ)
  Drop = 915 − 827 = 87 percentage points
  Relative drop = 87 / 915 = 9.6% of base return
  → Changing call_delta by 0.05 (from 0.25 to 0.20) costs ~10% of return.

Math behind the close_at_pct sensitivity:
  base = 915%, worst variant = 857% (at +0.10 offset, i.e., 0.85)
  Drop = 915 − 857 = 58 percentage points
  Relative drop = 58 / 915 = 6.3% of base return
  → Changing close_at_pct by 0.10 (from 0.75 to 0.85) costs ~6% of return.
```

**Our result:** ~100-point spread across close_at_pct combos (857–956%). Single-digit-percent relative variation — well inside "robust" territory.

- Spread = max − min = 956% − 857% = 99 percentage points
- Relative spread = 99 / 906 (midpoint) = 10.9% variation
- Compare: if the spread were 400+ pp / 40%+ variation, that'd be a sign of overfitting; we're well below.

### Regime Analysis: Does It Work in Bulls, Bears, and Sideways?

**Idea:** Classify each day as bull, bear, or sideways, then bucket the overlay's trade P&L by regime. If most of the income comes from one regime, the strategy isn't actually market-neutral.

The implementations are [`cc_backtest.py::classify_regime`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L666) and [`::regime_analysis`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L713). `classify_regime` looks at where the last price sits relative to its trailing 200-day SMA — `bull` if it's >5% above, `bear` if >5% below, `sideways` if within the band, `unknown` for the first 199 days when there aren't enough observations yet. `regime_analysis` runs the classifier at each day (using only past prices — no future peeking) and sums each closed trade's P&L into the regime active on its close date.

**Our result** (`__main__` params on the bundled MSFT data, pinned by `test_regime_analysis`):

| Regime | Days | Total P&L | Avg P&L/day |
| --- | ---: | ---: | ---: |
| Bull | 1,690 | $38,917 | $23.03 |
| Bear | 279 | $84,616 | $303.28 |
| Sideways | 346 | $139,032 | $401.83 |
| Unknown (first 200 days) | 200 | $7,916 | $39.58 |

**Interpretation:** Bear and sideways regimes produce **roughly 10× the per-day premium** of bull regimes, even though bull days dominate the day count (1,690 out of 2,515). Two things drive this: (1) volatility is higher in non-bull regimes, so option premium per trade is richer; (2) more positions hit their profit target or assignment threshold when the stock isn't grinding steadily upward. The strategy is structurally defensive — it earns most of its keep when the market is anything other than a one-way bull. That's the point of selling vol.

### Common Mistake: Only Testing in Bull Markets

If you only backtest on 2016–2021 (a strong bull run), you'll overestimate buy-and-hold returns and underestimate the CC overlay's relative value.

**Solution:** Test on multiple regimes. MSFT data from 2016–2026 includes:

- Bull: 2016–2017, 2019–2021 (tech boom)
- Bear: 2018 (correction), 2022 (rate hike sell-off)
- Sideways: 2023–2024 (consolidation)

### Statistical Significance: Is the Excess Return Real?

You can have a backtest with massive dollar P&L *and* zero statistical edge. These aren't contradictory — they're often the same result viewed two ways.

Run the MSFT backtest with `capital=$100,000`. The "Net Overlay P&L" line shows roughly **+$268,000** of excess profit over buy-and-hold. That looks great. It is also, statistically, indistinguishable from zero.

The t-statistic is how we settle the question.

#### Why a Headline Number Can Lie

Picture two coin-flippers trying to prove they have an edge. Player A flips 10 coins and gets 7 heads. Player B flips 10,000 coins and gets 5,200 heads. Player A's *rate* (70%) looks more impressive than Player B's (52%), but Player B's *evidence* is much stronger. With 10 flips, 7 heads is well within what luck produces; with 10,000 flips, 5,200 is improbable under fair-coin luck.

The t-statistic formalizes this. It asks: **how many standard errors above zero is my estimate?** The bigger the t-stat, the harder it is to dismiss the result as luck.

Two thresholds to remember:

- **|t| > 2** — Fisher's traditional bar. ~5% chance of occurring under the null. The textbook line for "statistically significant."
- **|t| > 3** — [Harvey, Liu & Zhu's stricter bar from their 2016 paper](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824). They argue that because finance has tested hundreds of factors, many "significant" |t| ≈ 2 results are just the lucky ones from a wide search. Three is the honest bar once you account for multiple testing.

#### What We're Actually Testing

For a covered call overlay, the relevant null hypothesis is:

> **H₀: The overlay adds zero value compared to simply holding the stock.**

We are *not* testing "does the strategy make money?" That's almost guaranteed because the strategy holds shares of MSFT, which compounded ~27% annualized over our sample. We need to isolate the overlay's contribution.

The clean way to do that is **excess returns**: each day's overlay return minus that day's buy-and-hold return on the same shares. The stock's drift cancels in the subtraction; what's left is the part attributable to selling calls. We then ask whether the *mean* of that excess-return series is reliably different from zero.

#### The IID Trap

Here's where most DIY backtesters get fooled. The standard t-stat formula is:

```text
t = mean / (std_dev / sqrt(n))
```

This assumes daily observations are IID — Independent and Identically Distributed. They almost never are for an overlay.

**Independent fails.** When you sell a 21-DTE call, that *same option position* drives daily P&L for up to 21 days. Today's overlay return and tomorrow's overlay return share a common driver. They aren't independent draws — they're samples from one position lifecycle.

**Identically distributed fails.** Your own `detect_regime()` function explicitly classifies returns into low-vol, normal-vol, and high-vol regimes — acknowledging that returns come from *different* distributions depending on which regime we're in.

When IID is violated, the naive formula systematically *understates* the standard error, which inflates the t-stat by 30–100%. You think you have a real edge when you don't.

#### Autocorrelation and Heteroskedasticity: What They Actually Mean

The two IID failures we just named have proper names in the statistics literature, and those names are baked right into "Newey-West **HAC**" (often abbreviated **NW**) — **Heteroskedasticity and Autocorrelation Consistent**. Worth understanding each one before applying the fix, because both are everywhere in financial data and both inflate naive t-stats in slightly different ways.

**Autocorrelation: when today depends on yesterday.**

A series is *autocorrelated* when one observation gives you information about the next. The data has *memory*. Picture the difference between a coin flip and a thermostat-controlled room: the coin has zero memory (yesterday's result tells you nothing about today's), while the room temperature has lots of memory (if it was 72°F a minute ago, it's almost certainly 71–73°F now). Financial returns aren't quite the thermostat, but they're definitely not the coin flip either.

Why this breaks naive t-stats: every t-statistic is `t = estimate / SE`, where SE is the **standard error** — how much your estimate would wobble across different samples. The textbook formula `SE = σ/√n` quietly assumes each observation contributes one full unit of *independent* information. When observations are correlated, that's a lie — your effective sample size is smaller than `n`. The naive formula produces a too-small SE, which produces a too-large t-stat, which makes you believe in edges that aren't there.

In our backtest, autocorrelation enters in three reinforcing ways:

1. **Position lifecycle.** When we sell a 21-DTE call, that *same* option position drives the overlay's P&L for up to 21 days. Day 5 and day 6 aren't two independent draws — they're samples from one shared position.
2. **Profit-target clustering.** The 75% close threshold tends to fire after stretches of stock-friendly days, which clusters close events and introduces serial correlation in the overlay's daily P&L.
3. **Underlying market memory.** Returns themselves have mild *momentum* (the tendency of returns to keep moving in the same direction) at short lags. Even before the overlay adds its own correlation, the price series isn't IID.

**Heteroskedasticity: when the variance moves.**

A series is *heteroskedastic* when its variance changes over time. Some periods are calmer; some are wilder. Variance isn't a constant property of the series — it's a moving target. Think of the ocean: waves are a different size on a windy day vs. a calm day. Try to summarize "the ocean's wave height" with one number and you'll be roughly right on average but dramatically wrong about both individual days.

Why this also breaks naive t-stats: a standard error that assumes constant variance is doing exactly the ocean thing — averaging over periods that have genuinely different volatility. The composite SE is biased: too small in calm periods, too big in volatile ones. Across the full sample you can over- or under-state significance depending on how the high-vol periods coincide with your signal.

In our backtest, heteroskedasticity is everywhere. Volatility clusters in equities — a documented stylized fact for at least 50 years ([Engle's 1982 ARCH paper](https://www.jstor.org/stable/1912773), [Mandelbrot's 1963 cotton-prices paper](https://www.jstor.org/stable/2350970)). The MSFT data spans calm 2017, the COVID volatility spike of 2020, the 2022 rate-hike sell-off, and 2024's renewed mega-cap bull run, each with materially different return variance. Our own `detect_regime()` function literally encodes this: it classifies each day as low / normal / high vol, explicitly acknowledging that returns come from different distributions. That classification *is* heteroskedasticity in code form.

**Why it matters that HAC handles both.**

Newey-West is built around *sample autocovariances at each lag* rather than a single global variance assumption. The lag-0 autocovariance — i.e., the variance itself — is estimated directly from the data, so heteroskedasticity in the level of variance is handled by simply not assuming the variance is constant. The lag-1, lag-2, ..., lag-L autocovariances handle the autocorrelation. Two violations, one estimator.

That's the promise of the "C" (consistent) in HAC: **as your sample grows, the HAC standard error gets closer and closer to the truth**. The naive standard error doesn't have this property under autocorrelated data — give it a million more observations and it just becomes more confidently wrong. Picture a pollster using a broken sampling method: with 100 voters their estimate is off, with 10 million voters it's still off by the same amount, just with tighter "confidence" around the wrong answer. That's the naive estimator. HAC is the pollster who fixes the method, so each additional observation actually pulls the estimate toward the right number. At any finite sample (like our 2,500 days) HAC has a small wobble that shrinks as you collect more data; the naive formula has a structural bug that more data can't fix.

This is why HAC is the default standard-error tool in modern empirical finance. Any time-series regression that assumes IID errors is making both these mistakes silently. HAC is what you reach for when you want the t-stat to mean what it claims to mean.

#### The Newey-West Fix

The Newey-West correction widens the standard error to account for autocorrelation:

```text
Var(mean) = (1/n) · [γ₀ + 2 · Σₖ wₖ · γₖ]
```

where γₖ is the autocovariance of excess returns at lag k, and wₖ = 1 − k/(L+1) are Bartlett weights — a smoothing kernel that gives lag 0 full weight, then tapers linearly to zero at lag L, ensuring distant noisy lags don't blow up the variance estimate. Intuitively: Newey-West asks "how much *effectively independent* information do I have, given that consecutive observations are correlated?" — then sizes the standard error accordingly.

The lag cutoff follows [Andrews (1991)](https://www.jstor.org/stable/2938229): `L = floor(4 · (n/100)^(2/9))`. For our 10-year MSFT sample (~2,500 days), that's 8 lags.

The constants and exponent in that formula solve a real bias-variance tradeoff in choosing how far back to look for autocorrelation:

- **Bias side.** Set L too small and you cut off lags that still have real autocorrelation. Newey-West then *still* underestimates the variance of the mean — you've fixed the IID assumption only partially.
- **Variance side.** Set L too large and you start including lags where the *true* autocorrelation is essentially zero, but the *sample estimate* is just noise. Each near-zero noisy autocovariance you add jitters the variance estimator from sample to sample, so your standard error becomes unreliable in a different way.

The optimum sits where the marginal reduction in bias equals the marginal increase in variance — equivalently, where the estimator's **mean squared error** (`MSE = bias² + variance`) is minimized. For the Bartlett kernel weights `w_k = 1 − k/(L+1)`, that optimum scales as `n^(2/9)`. Andrews (1991) provided the theoretical framework; [Newey & West (1994)](https://ideas.repec.org/p/att/wimass/9220.html) made it operational with the specific constants `4 · (n/100)^(2/9)`, calibrated to give sensible lag counts across the sample sizes typical in econometrics. The `n/100` term is just a scaling anchor — at n = 100 the formula returns exactly 4, so think of "4 lags at 100 observations" as the calibration point and everything else as a slow extrapolation from there.

![Three curves plotted against the lag cutoff L from 0 to 30. The bias-squared curve starts at 1.0 and decays toward zero by about L equals 10. The variance curve starts at zero and rises monotonically. Their sum, the mean squared error, is U-shaped, reaching a minimum around L equals 8 to 10. A vertical green line at L equals 8 marks the value the Andrews and Newey-West formula chooses for our sample size of 2,500.](docs/figures/03_bias_variance.png)

*Bias-variance decomposition of the Newey-West variance-of-the-mean estimator. Simulated from 2,000 AR(1) paths — a "today depends on yesterday plus noise" time series with `φ = 0.3` autocorrelation (see [Glossary](#appendix-c-glossary-of-key-terms)) — each of length n=2,500. Small L misses real autocorrelation (high bias, dashed blue); large L pulls in noisy near-zero autocovariances (high variance, dotted orange). The black MSE curve's U-shape is shallow but real, and the Andrews/Newey-West choice of L=8 lands essentially at the bottom — exactly the sweet spot the formula was designed to find.*

The 2/9 exponent is small, so L grows slowly with sample size:

| Sample size n | L = floor(4 · (n/100)^(2/9)) |
| --- | --- |
| 100 | 4 |
| 500 | 5 |
| 1,000 | 6 |
| 2,500 | 8 |
| 10,000 | 11 |

Doubling your sample only buys you ~17% more lags (`2^(2/9) ≈ 1.17`). The formula is telling you that with more data, most of the extra information goes into *refining the autocovariances you're already estimating* — not chasing deeper lags whose estimates would be too noisy to trust. The behavior is also self-floored: at n = 2 (the minimum to have any variance at all) the formula returns 1, so you never need a separate `max(1, ...)` guard.

The intuition transfers nicely. If your data has long memory (momentum factors, volatility clusters, slowly mean-reverting overlay P&L), the formula picks up enough lags to handle it. If the data is nearly IID, the few-lag NW correction barely changes the standard error from the naive version. Either way, it auto-adapts — you don't have to tune the bandwidth by hand for each dataset.

**Bottom line: report the Newey-West t-stat, not the naive one.** On near-IID data NW reduces to nearly the naive value at no cost; on the autocorrelated data you actually have, the naive formula quietly inflates the t-stat by 30–100%. There's no scenario where the naive version is the safer call.

#### The Code

The full implementation is [`cc_backtest.py::compute_statistics`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L541). Signature:

```python
def compute_statistics(
    daily_equity: pd.DataFrame,
    num_contracts: int,
    cash: float,
    periods_per_year: int = 252,
) -> dict[str, Any]:
```

Internally it reconstructs the buy-and-hold equity curve from `shares × price + cash`, computes daily excess returns by differencing both curves, then runs both the naive `mean / (std/√n)` t-stat *and* the Newey-West-corrected version (with `L = floor(4·(n/100)^(2/9))` Bartlett-weighted lags via Andrews / Newey-West). Returns the two t-stats, annualized excess return/vol, Sharpe of excess, the chosen lag cutoff, and pass/fail flags for the t=2 and t=3 thresholds.

#### What MSFT Actually Says

Running this on the bundled 10-year MSFT data:

```text
Annualized Excess Return:          +1.249%
Annualized Excess Vol:               9.90%
Sharpe of Excess Return:           +0.126
t-stat (naive, IID):                +0.40
t-stat (Newey-West, L=8 ):          +0.46
Clears t=2 bar?                     False
Clears t=3 bar (HLZ 2016)?          False
```

The +$268K headline P&L is real money in dollar terms, but **it's not statistically distinguishable from buy-and-hold noise**. With a Sharpe of 0.126 and 10 years of data, we'd need ~250 years of comparable data to clear the t = 2 bar at this effect size.

![Histogram of daily excess returns across 2,514 trading days. The distribution is roughly bell-shaped and centered near zero, with most days falling between minus 200 and plus 200 basis points. A red vertical line marks the sample mean at +0.5 basis points per day, sitting just to the right of zero and well within the bulk of the distribution.](docs/figures/02_excess_histogram.png)

*Daily excess returns from the bundled MSFT backtest. The mean (red line) is +0.5 bps/day, annualizing to +1.25%. The daily standard deviation around that mean is 62 bps — well over 100× larger. That noise-to-signal ratio is why the t-statistic is small.*

**Two standard deviations, not one.** The formula `t = mean / (std / √n)` quietly does more work than it looks like. It's the bridge between **two different standard deviations**, and conflating them is how people misread their own backtests.

**σ — the standard deviation of the data itself.** How much do individual daily returns vary day-to-day? In our case, **62 bps/day**. Some days are +200 bps, some are −150 bps, most are within ±60 bps. That's the volatility of the underlying observations.

**SE — the standard deviation of the *estimate* of the mean.** Imagine running this same 10-year backtest in a parallel universe with the same underlying market dynamics but different specific outcomes. The sample mean from each universe would differ slightly. How much it would differ is the *standard error of the mean*. It's much smaller than σ once you have many observations.

When you average independent observations, random ups and downs partially cancel — some are above the true mean, some below — and they average toward the truth. Mathematically:

$$\text{SE} = \frac{\sigma}{\sqrt{n}}$$

So with more data, the SE shrinks — but only as the *square root* of `n`. **To halve your SE you need 4× the data. To shrink it 10× you need 100× the data.** This is the iron law of statistical convergence, and it's what makes large samples expensive to obtain.

**Plugging in our numbers:**

- σ (daily noise) = 62 bps/day
- n = 2,514 trading days
- √n ≈ 50.1
- SE = 62 / 50.1 ≈ **1.2 bps**

After 2,514 days of averaging, the wobble of our sample-mean *estimate* has shrunk from 62 bps to about 1.2 bps — roughly 50× more precise than a single day's observation. The √n machinery did exactly what it was supposed to do.

**The t-stat is just "how many SEs is the mean away from zero?":**

- Sample mean (the edge): 0.5 bps/day
- SE: 1.2 bps
- Ratio: 0.5 / 1.2 ≈ **0.4**

That ratio *is* the t-statistic. A mean less than half an SE from zero is comfortably inside the noise of estimation — exactly what you'd see if the true mean were actually zero and our sample just happened to land slightly above it. To declare the mean "significantly different from zero" we'd need t ≥ 2 (about 2 SEs out).

**The dartboard picture.** A dart-thrower aiming at some target throws 2,514 darts. Individual darts land all over the place with spread σ = 62 bps. You take the average position of all the darts; that average wobbles around the true aim with spread SE ≈ 1.2 bps. You compute the average and find it 0.5 bps to the right of zero. Is the thrower aiming right of zero — or aiming at zero and the dart-average just happens to be slightly off? With a 1.2-bps wobble in your estimate and only 0.5 bps off-center, you can't tell. The honest verdict: "could be either."

**The misconception this clarifies.** People often see the 62 bps daily noise and conclude "the noise is so large, the t-stat is small." That's only half the story. The 62-bps daily noise got averaged down by √2,514 to a 1.2-bps SE on the *mean* — the √n machinery worked. **The reason the t-stat is small isn't that the daily noise is large in absolute terms; it's that the daily edge (0.5 bps) is even smaller than what 10 years of averaging can resolve.** That's why "just run a longer backtest" doesn't help much. To halve the SE from 1.2 to 0.6 bps (matching the signal exactly, getting t ≈ 1), you'd need 4× the data — 40 years. For t = 2, ~25× the data — ~250 years (the exact figure is `(2/0.126)² ≈ 252`). The arithmetic is brutal because of the square root.

**The shortcut.** There's a useful shortcut buried in the math: **t-stat ≈ Sharpe × √(years)**. You can sanity-check the relationship in one line: 0.126 × √10 ≈ 0.40, almost exactly the naive t-stat of 0.40 that `compute_statistics` returns. (The Newey-West-adjusted t-stat of 0.46 includes the autocorrelation correction from earlier in this section, which the shortcut doesn't model.) If your Sharpe and your sample length don't multiply to a healthy t-stat, no amount of fiddling with the strategy will rescue it — you need a bigger effect or more data.

![Line chart on a logarithmic x-axis showing the expected t-statistic as a function of years of data, given a Sharpe ratio of 0.126. The curve rises from about 0.13 at one year through 0.40 at ten years, crosses the conventional significance threshold of 2 at roughly 252 years, and crosses the Harvey-Liu-Zhu threshold of 3 at roughly 567 years. A dot at ten years marks the actual MSFT sample.](docs/figures/04_t_stat_vs_years.png)

*The shortcut, visualized. At this strategy's Sharpe, each additional decade of data buys roughly four-tenths of a t-stat point. Clearing the conventional t=2 bar would take around 250 years; the HLZ bar of 3 would take around 570. The path to a confident conclusion on this setup isn't "run a longer backtest" — it's "find a strategy with a bigger Sharpe" (index VRP, delta-hedged short calls, multi-asset diversification).*

#### Why Naive Is *Smaller* Than Newey-West Here

Usually Newey-West shrinks the t-stat — that's the whole point of the correction. In our case, naive (0.40) is slightly *smaller* than NW (0.46). What gives?

Newey-West can move the t-stat in either direction depending on the *sign* of short-lag autocovariances. If consecutive excess returns are positively correlated (a position held across days produces correlated P&L), NW shrinks the t-stat. If they're *negatively* correlated — *mean reversion*, the tendency for returns to bounce back toward their average — NW *inflates* the t-stat because the data has more "effective" sample than a naive count of days suggests.

Our excess returns show mild day-to-day mean reversion — likely from the way profit-target closes and position re-opens introduce alternation between premium-collection days and gap days. Either way the conclusion stands: t = 0.46 is firmly below any meaningful threshold.

#### Why Is This Lower Than the Volatility Risk Premium Literature?

The academic VRP literature ([Bakshi-Kapadia 2003](https://academic.oup.com/rfs/article-abstract/16/2/527/1605194), [Coval-Shumway 2001](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00352), the [BXM whitepapers](https://www.cboe.com/us/indices/dashboard/BXM/)) reports t-statistics in the range of **5–8**. So why does our well-built MSFT backtest produce 0.46?

The papers test a different null hypothesis. They compare a short-vol portfolio's return to **cash** (the risk-free rate). We compare the overlay's return to **buy-and-hold of the same stock**. Both questions are valid; they isolate different things.

When you compare to buy-and-hold, the stock's own return cancels and you're left measuring just the net premium contribution after assignments and buybacks. That's a small number with substantial noise — especially when your underlying is MSFT during a 10-year bull run that maximized assignment costs. When you compare to cash, you measure the *full* combined return of equity exposure plus premium income, which is much larger relative to its noise.

Three other compounding factors:

1. **Index VRP > single-stock VRP.** SPX options have structural insurance demand from institutional hedgers that single names lack. [Israelov & Nielsen (2015)](https://rpc.cfainstitute.org/research/financial-analysts-journal/2015/covered-calls-uncovered) found single-stock CC strategies underperform index CCs on a risk-adjusted basis.
2. **Covered calls capture only one side.** The richest part of the equity vol surface is *put* premium (skew). Our overlay sells only OTM calls — roughly 30–40% of the full one-leg VRP.
3. **Modeled IV vs. market IV.** Our backtest derives premiums from `HV × multiplier`, which is an *assumed* VRP, not a *measured* one. The academic numbers come from real market option prices spanning decades.

The right way to engage with this gap is to add a parallel test against cash, on an index ETF, with longer data. That comparison isn't built into this backtester yet — see "What We'd Add Next."

#### Common Mistake: Treating Dollar P&L as Evidence of Edge

A backtest that shows "+$268K excess profit" sounds like a win. It is also perfectly consistent with the strategy adding zero value — just lucky enough to land on the positive side of the noise distribution given a single 10-year sample. Without a t-statistic, you can't distinguish a real $268K edge from a zero-edge strategy that happened to flip 10 lucky coins in a row.

Always report the t-stat. Always compute it with Newey-West if you have any time-series data. And always benchmark against the question you actually care about (overlay vs. buy-and-hold *or* strategy vs. cash) — they answer different questions and produce different t-stats.

### Beyond Walk-Forward: The Full Anti-Overfitting Toolkit

Monte Carlo, sensitivity analysis, and regime testing (above) are the robustness checks we implemented in code. But there are several more tools worth knowing about — think of them as layers of defense, not a single wall:

| Layer | What It Catches | How It Works |
| --- | --- | --- |
| **Walk-forward** (Part 4) | Parameter overfitting | Train on one period, test on a different one, roll forward. |
| **Parameter stability** (sensitivity analysis above) | Fragile strategies | Check that nearby parameters give similar results — look for a "plateau" of good performance, not a single lucky peak. |
| **Monte Carlo shuffle** (above) | Sequence-dependent luck | Randomize the order of daily returns, rebuild price paths, see if strategy still works. |
| **Deflated Sharpe Ratio** | Multiple-testing bias | Adjusts your Sharpe ratio for how many strategies you tried. If you tested 120 parameter combos, the best one will look great by pure chance — even on random data, the luckiest combo will have a high Sharpe (same reason flipping 120 coins, at least one lands heads 7+ times in a row). The Deflated Sharpe corrects by asking: "Given N strategies tested, what's the probability my best Sharpe is just the expected maximum of N random trials?" It penalizes based on: (1) how many strategies tested — more trials → higher penalty, (2) variance of Sharpe ratios across trials — wider spread → best is more likely an outlier, (3) skewness/kurtosis of returns — fat tails make lucky outliers more likely. If your adjusted Sharpe is still significant after this penalty, the strategy has genuine edge — not just "I picked the luckiest coin out of 120." Key reference: Marcos López de Prado's work on this. |
| **Newey-West t-stat** (above) | Excess returns indistinguishable from zero | Compute the t-statistic of daily excess returns (overlay minus benchmark) using Newey-West standard errors that correct for the autocorrelation introduced by holding the same option position across multiple days. Conventional bar `\|t\| > 2`; stricter HLZ bar `\|t\| > 3`. Pairs naturally with Deflated Sharpe: t-stat tests whether the strategy's edge survives the *autocorrelation* of its own returns; deflated Sharpe tests whether it survives *multiple-testing bias* across the parameter grid. |
| **Multi-asset testing** | Stock-specific luck | Run the same strategy on MSFT, AAPL, SPY, QQQ, etc. A strategy that works across many tickers is capturing a real market dynamic, not a quirk of one stock. |
| **Regime analysis** (above) | Fair-weather strategies | Verify the strategy works in bull, bear, and sideways markets — not just the regime you happened to backtest on. |
| **Final holdout set** | All-of-the-above leakage | Reserve the last 1–2 years of data and *never touch it* until you're completely done designing and tuning. One shot, no do-overs. **How is this different from walk-forward's test set?** Walk-forward prevents the *code* from peeking at future data, but *you* still see the walk-forward results and make decisions based on them (e.g., "915% looks good, let's keep this approach"). That's information leakage through the human. The holdout prevents that second layer — data you literally never look at during the entire design process. No tuning, no validation, no "let me just check." After you've finalized everything, you run it once on the holdout. That result is your most honest estimate of real-world performance. |
| **Paper trading** | Everything historical testing can't | Run the strategy live with fake money for 3–6 months. No amount of historical testing substitutes for this. |

**The key insight:** No single check is enough. The more layers that agree your strategy works, the more confident you can be that you've found something real rather than a pattern in noise. Our backtest uses six of these layers (walk-forward, parameter stability, Monte Carlo, regime analysis, sensitivity, and the Newey-West t-stat on excess returns). Adding multi-asset testing and paper trading is the next step before risking real money.

> **Rule of thumb:** If your strategy survives walk-forward + Monte Carlo + parameter stability + at least two different tickers, you have something worth paper trading. If it survives 3–6 months of paper trading, you have something worth deploying with small real capital.

---

## Part 6: Putting It All Together

### The Full Pipeline: Data → Volatility → Black-Scholes → Overlay → Walk-Forward → Robustness

Here's the complete process:

```text
1. LOAD DATA
   ↓
   Daily prices from 2016–2026
   
2. CALCULATE ROLLING VOLATILITY
   ↓
   30-day window, log returns, annualize (ddof=1)
   Apply regime-based IV multiplier (1.1×/1.3×/1.5× HV)
   
3. BLACK-SCHOLES PRICING
   ↓
   For each day, calculate option prices
   Find strikes for target delta (0.25)
   
4. OVERLAY ENGINE
   ↓
   Day-by-day simulation:
   - Sell calls when no position open
   - Monitor profit target & expiration
   - Close when target hit or time passes
   - Record P&L
   
5. WALK-FORWARD OPTIMIZATION
   ↓
   Train on 2-year window, test on 6-month window
   Find best params for each window
   Stitch out-of-sample results
   
6. ROBUSTNESS CHECKS
   ├─ Monte Carlo: Shuffle daily returns, rebuild price paths, check percentile
   ├─ Sensitivity: Vary each parameter ±offset, check stability
   ├─ Regime: Split by volatility regime (low/normal/high vol)
   └─ Statistical Significance: Newey-West t-stat on excess returns vs. buy-and-hold
```

### How to Interpret Results Honestly

**Good signs:**

- Walk-forward test shows 60–70% of in-sample returns (realistic)
- Monte Carlo: real return percentile > 80% (beats randomized price paths)
- Sensitivity: nearby parameters give similar results (not overfit)
- Works in all regimes (not just bull markets)
- Sharpe ratio > 0.8 (good risk-adjusted returns)
- Newey-West t-stat on excess returns > 2 (>3 if you tested many parameter combos)

**Red flags:**

- In-sample 500%, out-of-sample 50% (massive overfitting)
- Monte Carlo percentile < 50% (random paths beat you)
- Sensitivity shows wildly different results for small tweaks (unstable)
- Only works in one market regime (not generalizable)
- Sharpe < 0.3 (returns don't justify the risk)
- Newey-West t-stat < 2 even though dollar P&L looks positive (the apparent edge is noise)

**Our strategy:**

![Overlay vs. buy-and-hold equity curves on MSFT 2016–2026, showing the overlay ending at approximately $1,015K and buy-and-hold ending at approximately $746K. Both curves grow substantially; the overlay's lead is small early on and widens noticeably from 2019 onward, ending with a $268K gap.](docs/figures/01_equity_curves.png)

*Overlay vs. buy-and-hold equity on the bundled MSFT data. The overlay finishes about $268K ahead. The gap is small through 2018, then widens through the 2019–2024 stretch and stays near $200–300K through the recent vol-heavy period — accumulating in the volatile middle years rather than in any single regime.*

- ✅ Fixed params: ~915% total return on the bundled `$100K` configuration (final equity ~$1,015K; see Figure 1)
- ✅ Monte Carlo: percentile 100 (real ordered path beats every one of 500 shuffled paths; max shuffled return ~870%)
- ✅ Sensitivity: single-digit-% drops across both `call_delta` and `close_at_pct` perturbations
- ✅ All regimes: bull, bear, sideways all profitable
- ✅ Sharpe ratio vs cash (rf = 4.5%): ~1.12, vs buy-and-hold MSFT's ~0.72 over the same window — risk-adjusted *absolute* returns are strong
- ⚠️ **Newey-West t-stat on excess returns: 0.46** (overlay's *excess* over buy-and-hold is not statistically distinguishable from zero on this single-stock 10-year sample; see Part 5). The dollar P&L is real, but the evidence for "the overlay specifically is adding value beyond holding MSFT" doesn't clear the statistical bar. This is what the literature on single-stock CC underperformance vs. index CC predicts.

### The Limitations We Haven't Solved

1. **IV proxy (regime-based HV multiplier):** Even with regime switching (1.1×/1.3×/1.5×), this is a rough approximation. Real IV can spike 50%+ on bad news, especially around earnings.
2. **No gap risk:** We assume you can always close at model prices. Reality: market gaps at open, especially on earnings.
3. **No dividend handling:** MSFT pays dividends; our model ignores them (small effect, but nonzero).
4. **No earnings avoidance:** Selling calls into earnings is dangerous (IV crush, whipsaws). We should avoid this.
5. **No rolling:** We modeled buying back and selling new as separate events, but real traders often "roll" — a single combined order that closes the old call and opens a new one simultaneously, often for a net credit. Example: your old call costs $0.50 to buy back, the new call sells for $2.00 → rolling combines them into one order for a $1.50 net credit, with one fill instead of two and often better pricing since brokers optimize combo orders. Our backtest treats these as independent transactions, so real performance could be slightly better due to reduced slippage.
6. **Commission simplification:** We assumed $0.65 per contract. Real costs vary by broker.

### What We'd Add Next

**To make this production-ready:**

1. **Actual option prices:** Use OptionMetrics or similar to get real IV and prices
2. **Earnings calendar:** Avoid selling calls in the week before earnings
3. **VIX regime switching:** Adjust delta based on VIX level (high VIX → be more defensive, sell lower deltas further OTM — you still collect decent premium because high IV inflates prices even at lower deltas, and the extra buffer protects against the larger price swings that high-vol environments produce)
4. **Rolling logic:** Model rolling ITM calls for credits, not just buying back
5. **Portfolio optimization:** Test on 10–20 stocks, optimize correlation effects
6. **Slippage modeling:** Account for bid-ask widening on high-volatility days
7. **Strategy-vs-cash significance test:** Add a second mode to `compute_statistics` that benchmarks the CC strategy's *total* return against the risk-free rate (not against buy-and-hold). This is the comparison the academic VRP literature reports, and it's the right way to put our backtest on equal footing with published BXM/PUT t-stats
8. **Index ETF test:** Run the same strategy on SPY or QQQ. Single-stock VRP is structurally weaker than index VRP because index options have richer insurance demand. If the t-stat moves substantially toward the academic range when we switch underlyings, that confirms the gap was about *what* we backtested, not *how* we backtested
9. **Risk-managed (delta-hedged) covered call mode:** Add a `delta_hedge` flag to `params` that, when enabled, buys or sells underlying shares each day to keep the portfolio's net delta pinned at `base_shares` — regardless of where the short call's delta sits. *Delta-hedging* is the practice of continuously trading the underlying to neutralize an option's directional exposure. Conceptual basis in the callout below. Costs ~25–30% more capital (you're holding extra shares to offset the call's negative delta) but produces a meaningfully cleaner test of whether the volatility risk premium is actually being captured
10. **7-DTE close rule:** Close any open position when fewer than 7 days remain to expiration, regardless of profit target or delta. The current engine triggers on expiration itself, profit-target, or deep ITM (`delta > 0.70`) — so a position that drifts into the gamma-heavy final week without hitting either still has to ride through it. Adding a `min_dte_to_close` parameter (e.g., default 7) is the conventional fix and matches the *Gamma risk* warning in the glossary. Effect on results is probably small on bullish underlyings like MSFT (the delta-0.70 trigger already catches most of these positions early) but would be more visible on volatile or sideways tickers where positions can sit near-ATM into the final week without becoming deep ITM

> **Lessons from Israelov & Nielsen (2015), "Covered Calls Uncovered"** ([CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2015/covered-calls-uncovered))
>
> A covered call's return decomposes exactly into three parts: (1) passive equity exposure, (2) short volatility exposure — the VRP, and (3) a hidden *equity-timing* exposure that nobody explicitly chose. The third one is mechanical: as the stock rallies, the short call's delta rises and your *effective* stock exposure shrinks; as the stock sells off, delta falls and your effective exposure grows. You're being forced to lighten up into rallies and add into selloffs, on autopilot.
>
> Components 1 and 2 are real, persistent, harvestable premiums. Component 3 is essentially zero in expectation but adds substantial variance — it's a coin flip you didn't sign up for. The paper's prescriptive fix is the **risk-managed covered call**: dynamically rebalance the underlying share position so the portfolio's net delta stays pinned at the buy-and-hold equivalent (e.g., 100 shares for a 1-contract position). Same equity exposure as buy-and-hold, plus the vol premium, minus the equity-timing wiggle.
>
> The implementation is one extra block in the daily loop: compute `target_shares = base_shares + abs(call_delta) * 100 * num_contracts`, then buy or sell shares to match. The expected effect on our backtest is **a higher Sharpe of excess returns and a higher Newey-West t-stat — not because alpha (excess return beyond what the benchmark explains) increases, but because we've stopped measuring an exposure that contributes variance without contributing return**. That's the cleanest way to test whether the VRP is showing up on this underlying. Item 9 above operationalizes it.

---

## Part 7: Key Takeaways & Cheat Sheet

### The 6 Most Important Lessons

1. **Covered calls are income, not capital appreciation.** Sell 0.25–0.30Δ calls and be happy when they're exercised. The premium is your profit, not the stock appreciation.

2. **Walk-forward validation is essential.** In-sample optimization lies. Always test on different periods to avoid overfitting.

3. **Black-Scholes is a recipe, not a crystal ball.** It estimates fair option value given volatility. The volatility assumption is the biggest source of error.

4. **Transaction costs are real.** Commission + slippage eat 3–5% of premium. Don't ignore them.

5. **Robustness beats optimization.** A strategy that works in bulls, bears, and sideways is better than one that's tuned perfectly for one regime.

6. **Dollar P&L is not the same as statistical edge.** A backtest can show massive excess profits *and* a Newey-West t-stat below 2 on excess returns — meaning the apparent edge is within what noise produces over the sample. Always report the t-stat alongside the dollar P&L, and always compute it on excess returns over a benchmark you actually care about (overlay vs. buy-and-hold answers a different question than strategy vs. cash). Use Newey-West standard errors when the data has time-series autocorrelation, which yours always does.

### Parameter Cheat Sheet

| Parameter | Conservative | Balanced | Aggressive |
| --- | --- | --- | --- |
| **call_delta** | 0.15–0.20 | 0.25 | 0.30–0.35 |
| **put_delta** *(CSP phase only)* | -0.15 to -0.20 | -0.20 | -0.25 to -0.30 |
| **dte** | 35–45 | 30 | 21 |
| **close_at_pct** | 0.50 (close early) | 0.75 | 1.00 (hold to expiry) |
| **Expected assignment freq.** | ~15% | ~25% | ~35% |
| **Capital in use** | Low (less frequent sales) | Medium | High (constant selling) |

**Recommendation for beginners:** Start with "Balanced" (0.20Δ put / 0.20Δ call, 30 DTE, 0.75 close_at_pct). It's tested and robust.

### Decision Flowchart: "Should I Sell a CC Today?"

```text
Is there an open position?
  ├─ YES:
  │    ├─ Has the position reached expiration? → SETTLE (assigned if ITM, else expires worthless)
  │    ├─ Has 75% of premium been captured? → BUY BACK (close)
  │    ├─ Has the call gone deep ITM (delta > 0.70)? → BUY BACK (avoid assignment)
  │    └─ Otherwise → HOLD
  │
  └─ NO:
       ├─ CSP phase + downtrend (SMA50 < SMA200)? → WAIT (optional; avoid put assignment into decline)
       ├─ Calculate rolling vol: σ = 30-day HV × regime multiplier (1.1/1.3/1.5)
       ├─ Find 0.25Δ strike using grid search
       ├─ Calculate premium using Black-Scholes
       ├─ Apply 3% slippage + $0.65 commission
       └─ SELL CALL (if net premium > 1% of stock price)
```

### Resources for Further Learning

**Theory:**

- Hull, *Options, Futures, and Other Derivatives* (textbook, comprehensive)
- Natenberg, *Option Volatility and Pricing* (practical, less math-heavy)
- Taleb, *Dynamic Hedging* (advanced, but teaches intuition)

**Data & Implementation:**

- [QuantConnect](https://www.quantconnect.com) — free backtesting platform
- [OptionMetrics](https://optionmetrics.com) — historical option data (expensive)
- [cboe.com](https://www.cboe.com) — real-time option prices, educational resources

**Strategy-Specific:**

- [CBOE Covered Call Index (BXMD)](https://www.cboe.com/us/indices/dashboard/BXMD/) — real-world CC strategy benchmark
- Jansen, *The Complete Guide to Option Selling* (practical, real trading experience)

---

## Appendix A: The Code

The full implementation lives in this repository. Each file is the source of truth — the snippets and pseudocode in the body of this tutorial are explanatory excerpts, not the canonical implementation.

| File | What it contains |
| --- | --- |
| [`cc_backtest.py`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L201) | Black-Scholes pricing, rolling-volatility helpers, the [`run_cc_overlay`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L201) engine, and [`compute_statistics`](https://github.com/l3a0/covered-call-backtesting/blob/main/cc_backtest.py#L541) for Newey-West t-stats |
| [`test_cc_backtest.py`](https://github.com/l3a0/covered-call-backtesting/blob/main/test_cc_backtest.py#L474) | Pytest suite covering pricing primitives, the overlay state machine, scenario tests, and the statistics helper |
| [`download_prices.py`](https://github.com/l3a0/covered-call-backtesting/blob/main/download_prices.py#L11) | Fetches historical daily closes via yfinance |
| [`msft_10yr_prices.csv`](https://github.com/l3a0/covered-call-backtesting/blob/main/msft_10yr_prices.csv) | Bundled 10-year MSFT daily-close dataset used in the worked examples |
| [`requirements.txt`](https://github.com/l3a0/covered-call-backtesting/blob/main/requirements.txt) | Pinned dependencies |
| [`README.md`](https://github.com/l3a0/covered-call-backtesting/blob/main/README.md) | Quick-start instructions and project summary |

**To run it locally:**

```bash
git clone https://github.com/l3a0/covered-call-backtesting.git
cd covered-call-backtesting
pip install -r requirements.txt
python cc_backtest.py            # runs the backtest on bundled MSFT data
pytest                           # runs the test suite
```

---

## Appendix B: Common Pitfalls and How to Avoid Them

| Pitfall | What Goes Wrong | How to Fix |
| --- | --- | --- |
| **Forgetting transaction costs** | Backtest shows 20% returns; real trading shows 14% | Include 3% slippage + $0.65 commission per open/close in model |
| **Look-ahead bias** | You "know" the stock will drop, so you sell a low-delta call | Only use data available on decision day; never peek ahead |
| **Over-optimizing** | You test 1000 parameter combos, pick the best, then it fails | Use walk-forward validation; test on different data |
| **Ignoring gaps** | Model assumes you can close at Black-Scholes price; market gaps 5% overnight | Add slippage buffer; avoid earnings; use limit orders |
| **Wrong volatility** | You use realized vol from calm period; market gets volatile | Use rolling vol; adjust for regime; don't assume constant vol |
| **Assignment confusion** | You think you're "missing upside" when shares get called away | You EXPECTED 25% chance of assignment; that's the deal you made |
| **Not testing downturns** | You test 2016–2021 (bull market); strategy fails in 2022 | Test across bull, bear, and sideways regimes |
| **Ignoring dividends** | You didn't account for dividend yield | For MSFT, add ~0.7–1% annual yield to returns |
| **Wrong delta interpretation** | You think 0.50Δ = 50% probability of profit | Delta = probability of being ITM at expiration (mathematically) or equivalent stock hedge (practically) |
| **Holding too long (close_at_pct too high)** | You set close_at_pct=1.0 (hold to expiry); miss early profit opportunities | Use close_at_pct of 0.50–0.75; capture most of the premium decay without waiting for expiration risk |
| **Confusing dollar P&L with edge** | Backtest shows "+$268K excess profit"; strategy is actually statistically indistinguishable from buy-and-hold | Compute Newey-West t-stat on daily excess returns (overlay minus benchmark). Aim for `\|t\| > 2` (conventional) or `\|t\| > 3` (Harvey-Liu-Zhu adjusted for multiple testing). Use the `compute_statistics()` helper and read the t-stat alongside the dollar P&L — not in isolation |
| **Naive t-stat on autocorrelated returns** | You compute t = mean / (std/√n) and get an inflated number that disappears in live trading | Always use Newey-West HAC standard errors when measuring t-stats on overlay or any held-position strategy. Same formula otherwise undersizes the standard error by 30–100% because consecutive-day P&Ls share a common driver (the open option position) |

---

## Appendix C: Glossary of Key Terms

Reference glossary for terms used throughout the tutorial. Most are also defined inline where they first appear; this is the place to come back to when you need a refresher.

- **AR(1) (autoregressive, order 1):** The simplest model of a time series that has memory. Each value depends linearly on the previous one plus a fresh random shock: `y_t = φ · y_{t-1} + ε_t`. The single coefficient `φ` (phi) controls how much: `φ = 0` is white noise (no memory — IID coin flips), `φ > 0` is positive autocorrelation (today inherits from yesterday — momentum), `φ < 0` is mean reversion (today fades yesterday). AR(1) is the standard testbed for time-series methods because its long-run variance has a clean closed form (`1 / (1 − φ)²` for unit-variance innovations), so you can measure how well an estimator like Newey-West tracks the truth. Figure 3 in Part 5 uses 2,000 AR(1) paths with `φ = 0.3` to demonstrate the bias-variance tradeoff for the NW lag cutoff.
- **Assignment loss:** What happens when your covered call gets exercised because the stock rallied past the strike. You collected premium up front, but to keep running the overlay you must rebuy the shares at the current (higher) market price. The overlay's net is `premium − (market_price − strike)`. It's a **loss** when the stock rallied past `strike + premium`. Example: you sold a $310 strike call for $1.50 premium and the stock closed at $325 — you keep the $1.50 but pay back $15 of capped upside, netting **−$13.50/share**. The stock appreciation up to the strike is still yours (tracked separately as part of equity), but the *uncapped* portion of the rally is gone. In a strong bull market this is the dominant cost the overlay pays.
- **CC (Covered Call):** A strategy where you own 100 shares of a stock and sell a call option against them. You collect the premium up front; if the stock stays below the strike at expiration, you keep the shares and the premium. If it rises above the strike, your shares may be called away at that price. "Covered" means you already own the shares, so you're not exposed to unlimited upside risk like a naked call.
- **CSP (Cash-Secured Put):** A strategy where you sell a put option and set aside enough cash to buy 100 shares at the strike price if assigned. You collect the premium up front; if the stock stays above the strike, the put expires worthless and you keep the cash. If it falls below, you're obligated to buy the shares at the strike. CSPs are the "entry" half of the wheel — a way to get paid while waiting to buy a stock at a discount.
- **CDF (Cumulative Distribution Function):** Answers the question "what's the probability a value falls at or below X?" Imagine filling a glass of water as you move left to right across a bell curve — at the far left it's nearly empty (0%), at the center it's half full (50%), at the far right it's nearly full (100%). In our context, the CDF converts a stock's distance from the strike price into a probability, which is exactly what Black-Scholes needs to price an option.
- **Closed-form solution:** A formula you can write down using basic math (add, multiply, exponents, etc.) and compute directly — like the area of a circle (πr²). When something has "no closed-form solution," it means you can't write a simple equation for it; you need either calculus or an approximation. The normal CDF has no closed-form, which is why we use a polynomial shortcut.
- **Delta (Δ):** The probability (roughly) that an option expires in-the-money. A 0.25 delta call means ~25% chance the stock ends above the strike. We use delta to choose how aggressive our covered calls are.
- **Drawdown:** The decline from a recent peak to a subsequent trough. *Max drawdown* is the worst peak-to-trough loss in the sample — a common measure of downside risk. A strategy returning 20% annualized with 5% max drawdown is safer than one returning 25% with 40% max drawdown, even though the first has lower expected return.
- **DTE (Days to Expiration):** How many calendar days until the option expires. We use 21 DTE (about 3 weeks) as our default.
- **Excess Return:** The strategy's return *minus* the benchmark's return on the same day. For a covered call overlay, the relevant excess return is the overlay's daily return minus buy-and-hold's daily return on the same shares. Subtracting the benchmark cancels out the stock's own movement, leaving only the value the overlay adds (or destroys). When testing whether a strategy "works," you almost always want to test excess returns, not raw returns — otherwise you're mostly measuring the underlying market.
- **Gamma risk:** The risk that your option's delta changes rapidly as the stock moves. Near expiration, small stock moves can cause big swings in an option's value — a "safe" out-of-the-money call can suddenly become in-the-money. The conventional fix is to close positions before the last week of expiration (the "7-DTE close" rule). The current engine handles this indirectly through its `delta > 0.70` close trigger, which tends to fire more readily near expiration as gamma steepens; an explicit DTE-based threshold is listed in "What We'd Add Next" in Part 6.
- **HV (Historical Volatility):** How much the stock price has actually been bouncing around, measured from past prices.
- **IID (Independent and Identically Distributed):** Two assumptions baked into most introductory statistics formulas. *Independent* means each observation tells you nothing about any others (like fair coin flips). *Identically distributed* means every observation comes from the same probability distribution (same bag of marbles each draw). Financial returns rarely satisfy either — they cluster in volatility regimes (not identical) and they autocorrelate through position holding (not independent). Naive t-stat formulas assume IID and inflate when it's violated.
- **ITM (In the Money):** The opposite of OTM. A call option is ITM when the strike is *below* the current stock price (the buyer would exercise immediately to capture the difference); a put is ITM when the strike is *above* the current stock price. As a covered-call seller you *don't* want your short calls to end up ITM at expiration — that's the assignment loss scenario, where you lose all the upside above the strike. An option's delta is roughly the probability it expires ITM.
- **IV (Implied Volatility):** What the market thinks future volatility will be, baked into the option price. Since we don't have real IV data, we estimate it using a regime-based multiplier on HV (1.1× in high-vol regimes, 1.3× in normal, 1.5× in low-vol).
- **Lag:** How many time periods back you look when comparing observations in a time series. "Lag 1" means yesterday's value, "lag 5" means the value from 5 days ago, and so on. The *autocovariance at lag k* measures how much today's observation correlates with the observation `k` periods earlier — at lag 0 this is just the variance (correlation of a value with itself), at lag 1 it's the same-as-yesterday correlation, etc. Newey-West sums weighted autocovariances from lag 0 up to a chosen *lag cutoff* `L`, picked big enough to capture meaningful autocorrelation but small enough to avoid pulling in noisy near-zero terms.
- **MSE (Mean Squared Error):** The standard way to measure how far an estimator is from the truth on average: `MSE = E[(estimator − true value)²]`. It decomposes cleanly into two pieces — `MSE = bias² + variance` — where the bias² term captures systematic miss (the estimator is centered on the wrong value) and the variance term captures noise (the estimator wobbles a lot across samples). Minimizing MSE means *balancing* those two, sometimes accepting a little bias to reduce a lot of variance, or vice versa. Figure 3 in Part 5 plots both pieces and their sum as a function of the Newey-West lag cutoff `L`; the U-shape comes directly from this tradeoff.
- **Newey-West HAC (NW):** A correction to standard errors that accounts for autocorrelation and heteroskedasticity in time-series data — both common in financial returns. Often abbreviated **NW** in code and prose. HAC stands for "Heteroskedasticity and Autocorrelation Consistent." *Heteroskedasticity* means the variance changes over time (the stock is calm one week, volatile the next); *autocorrelation* means consecutive observations are correlated (today's value tells you something about tomorrow's). For an overlay strategy, where the same option position drives multiple consecutive days of P&L, naive standard errors are too small (consecutive days aren't independent observations) and naive t-stats are inflated. Newey-West fixes both at once by widening the standard error to reflect the actual independent information in the sample. Lag cutoff in our backtest is `L = floor(4 · (n/100)^(2/9))` — the framework comes from Andrews (1991); the specific operational formula is from Newey & West (1994).
- **OTM (Out of the Money):** A call option where the strike price is above the current stock price (the buyer wouldn't exercise yet). We sell OTM calls to collect premium while giving the stock room to grow.
- **PDF (Probability Density Function):** The "height" of the bell curve at a given point. While the CDF measures the area under the curve (a cumulative probability), the PDF measures how tall the curve is at one specific value. We need it inside the CDF approximation formula — the approximation works by multiplying the PDF (height) by a polynomial correction to estimate the CDF (area).
- **P&L (Profit and Loss):** The dollar gain or loss from a position or strategy over some period. *Realized P&L* counts only closed trades; *unrealized P&L* marks open positions to current market prices. In our backtest, the overlay P&L is the cumulative profit from selling and buying back calls, kept separate from the stock's price appreciation so we can isolate what the overlay specifically contributed.
- **Premium:** The price the option buyer pays you. This is your income as a covered call seller.
- **Sharpe Ratio:** Annualized return divided by annualized volatility — return per unit of risk. A Sharpe of 1.0 is excellent; most decent strategies sit at 0.4–0.8. Critical caveat: Sharpe assumes normally distributed returns. Strategies with fat left tails (covered calls, short puts, merger arb) look better on Sharpe than they actually are because the metric ignores the asymmetry between small frequent gains and rare large losses.
- **Standard Error (SE):** How much your *estimate* of a quantity (a mean, a regression coefficient) would wobble if you re-ran the experiment with a fresh sample. Smaller SE = more precise estimate. For the mean of `n` observations with standard deviation `σ`, the textbook formula is `SE = σ/√n` — meaning that to halve your SE you need four times the data. The t-statistic is built on top of SE: `t = estimate / SE`, so the value you compute depends critically on which SE formula you use. Using the wrong SE (naive instead of Newey-West, for instance) produces a wrong t-stat that may look impressive but doesn't survive scrutiny.
- **Strike (Strike Price):** The price at which an option's buyer can choose to buy (for a call) or sell (for a put) the underlying stock. Strikes are usually whole-dollar increments. When you "sell a 0.25-delta call," what you're really choosing is the strike where the call has roughly 25% probability of being ITM at expiration — high enough above the current price that exercise is unlikely, low enough that the premium is worth collecting.
- **T-statistic:** A number that tells you how many standard errors your estimate is away from zero. A t-stat of 2 means there's only a ~5% chance of observing this result if the true effect were zero — Fisher's conventional bar for "statistically significant." For backtests, you want a t-stat well above 2 (Harvey, Liu & Zhu 2016 argue 3 is the honest bar after multiple-testing adjustment) and you want it computed with Newey-West standard errors, not naive ones.
- **VRP (Volatility Risk Premium):** The systematic gap between options' implied volatility (what option prices imply about future moves) and realized volatility (what actually happens). Across decades and asset classes, IV averages above subsequent realized vol — meaning options are, on average, slightly overpriced. Selling options is the textbook way to harvest the VRP. Our covered call overlay captures only a fraction of it: the call side, only out-of-the-money strikes, only on a single stock. The full premium is much richer at the index level (BXM, PUT indices document this).
- **Wheel (Wheel Strategy):** A two-phase options income strategy. Phase 1: sell *cash-secured puts* (CSPs) on a stock you'd be happy to own. If the puts expire worthless, keep the premium; if they get assigned, you now own the stock at a discount. Phase 2: sell *covered calls* (CCs) against those shares. If the calls expire worthless, keep the premium; if they get exercised, you sell at a profit and start the cycle over. This tutorial focuses on the covered-call (Phase 2) half of the wheel.

---

## Final Thoughts: What to Study Next

After reading this tutorial, you understand:

- ✅ **Why** covered calls work (income generation)
- ✅ **How** to price them (Black-Scholes)
- ✅ **How** to simulate them (overlay engine)
- ✅ **How** to validate them (walk-forward, robustness)
- ✅ **How** to assess if the result is real (Newey-West t-statistic on excess returns)
- ✅ **What can go wrong** (limitations, pitfalls)

**Next steps:**

1. **Implement this from scratch** on a stock you own. Use real price data. Compare to the backtest.
2. **Paper trade for 1 month.** Use real-time option prices. See if the model works live.
3. **Read the limitations section again.** Understand which ones matter most for YOUR broker and situation.
4. **Study roll mechanics.** Our model close calls; professionals roll them for extra credit.
5. **Explore earnings avoidance.** Add a function to detect earnings weeks and skip them.

Good luck. Covered call trading is not exciting, but it's one of the most reliable ways to generate steady income from stock ownership.

---

## References

Academic papers cited or built on in this tutorial. URLs link to the publishers' canonical landing pages where I'm reasonably sure of the exact URL; for the rest, the citation alone should be enough to find the paper on Google Scholar.

### Option pricing

- Black, F. & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654. ([JSTOR](https://www.jstor.org/stable/1831029))

### Numerical methods

- Abramowitz, M. & Stegun, I. A. (eds.) (1964). *Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables*. National Bureau of Standards. Source of the polynomial CDF approximation (Formula 26.2.17) used in the educational version of `normal_cdf`. ([Wikipedia](https://en.wikipedia.org/wiki/Abramowitz_and_Stegun))

### Volatility risk premium

- Bakshi, G. & Kapadia, N. (2003). "Delta-Hedged Gains and the Negative Market Volatility Risk Premium." *Review of Financial Studies*, 16(2), 527–566. ([Oxford Academic](https://academic.oup.com/rfs/article-abstract/16/2/527/1605194))
- Coval, J. D. & Shumway, T. (2001). "Expected Option Returns." *Journal of Finance*, 56(3), 983–1009. ([Wiley](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00352))
- Israelov, R. & Nielsen, L. N. (2015). "Covered Calls Uncovered." *Financial Analysts Journal*, 71(6). ([CFA Institute](https://rpc.cfainstitute.org/research/financial-analysts-journal/2015/covered-calls-uncovered))

### Heteroskedasticity and volatility clustering

- Engle, R. F. (1982). "Autoregressive Conditional Heteroskedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987–1007. ([JSTOR](https://www.jstor.org/stable/1912773))
- Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices." *Journal of Business*, 36(4), 394–419. ([JSTOR](https://www.jstor.org/stable/2350970))

### HAC standard errors and t-statistics

- Andrews, D. W. K. (1991). "Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation." *Econometrica*, 59(3), 817–858. ([JSTOR](https://www.jstor.org/stable/2938229))
- Newey, W. K. & West, K. D. (1987). "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*, 55(3), 703–708. ([JSTOR](https://www.jstor.org/stable/1913610))
- Newey, W. K. & West, K. D. (1994). "Automatic Lag Selection in Covariance Matrix Estimation." *Review of Economic Studies*, 61(4), 631–653. ([RePEc](https://ideas.repec.org/p/att/wimass/9220.html))
- Harvey, C. R., Liu, Y. & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies*, 29(1), 5–68. ([Oxford Academic](https://academic.oup.com/rfs/article-abstract/29/1/5/1843824))

### Indices and benchmarks

- CBOE S&P 500 BuyWrite Index (BXM). ([cboe.com](https://www.cboe.com/us/indices/dashboard/BXM/))
- CBOE S&P 500 PutWrite Index (PUT). ([cboe.com](https://www.cboe.com/us/indices/dashboard/PUT/))
- CBOE S&P 500 30-Delta BuyWrite Index (BXMD). ([cboe.com](https://www.cboe.com/us/indices/dashboard/BXMD/))

For textbooks (Hull, Natenberg, Taleb) and educational platforms (QuantConnect, OptionMetrics, CBOE), see the "Resources for Further Learning" section in Part 7.

## Provenance & Disclaimer

This tutorial was drafted in collaboration with Claude (Anthropic) — the prose, structure, and code in `cc_backtest.py` and `test_cc_backtest.py` were generated through extended back-and-forth between the author and the model. The code is unit-tested and runs correctly on the bundled MSFT data, but you should still review it before relying on it for real money: both for correctness on your own data and for whether the modeling choices baked in (Black-Scholes pricing instead of market quotes, regime-based IV multipliers instead of measured IV, fixed slippage and commission assumptions) match your actual broker, instruments, and tolerance for tail risk. AI-generated citations and derivations occasionally drift from the primary sources, so treat everything here as a starting point rather than gospel. The references above are the original published papers — those are what any claim in this tutorial should be checked against.

The author is a learner working through these ideas, not a credentialed quant or financial advisor. Nothing in this tutorial is investment advice.

**Last updated:** May 2026
