# Reading Notes — Pardo, *The Evaluation and Optimization of Trading Strategies*

Highlights extracted from my Kindle clippings (`My Clippings.txt`). Reference material for the backtesting methodology used in this repo (walk-forward analysis, walk-forward efficiency, degrees of freedom).

**Source:** Pardo, Robert. *The Evaluation and Optimization of Trading Strategies: 314 (Wiley Trading)*. John Wiley & Sons, 2008.

**Extracted:** 2026-06-02 — 107 clippings, ordered by Kindle location (deduplicated from 176 raw highlights; exact duplicates and Kindle prefix-extensions collapsed, clipping-limit markers stripped).

> Verbatim highlights from the Kindle edition, for personal reference. Not authored prose and not part of the repo's cross-surface consistency system.

---

## Location 293–295

> The processing time involved is typically so massive with these commercial applications as to make it highly undesirable, if not practically impossible.

## Location 295–296

> Pardo Capital Limited uses in-house proprietary applications for most of the heavy-duty computing that we professional trading firms must do when developing trading platforms.

## Location 380–383

> the existence of such a trading strategy development application with the ability to integrate and apply such technologies at a usable speed would really be a development that would be at least somewhat in proportion to the vast explosion of computer hardware that we have seen over the last 15 years.

## Location 431–438

> If one looks at the history of these developments, one can easily trace its beginnings to an extremely important discovery by the creator of fractal geometry, the mathematical genius Benoit Mandelbrot. He discovered that the distribution of price changes in financial markets follows a fractal distribution, not the standard Gaussian distribution assumed by all financial mathematicians and which is embedded in things like the Black-Sholes options pricing model. This is an earthquake. To a certain extent, I am really not too sure that this is understood or applied by all financial engineers even today. The tides of change were further fueled by the discoveries of things like genetic algorithms, chaos math, fuzzy logic, complexity theory, and concepts like artificial life, which describes the way complex structures form from simple processes, discovered by another mathematical genius, Stephen Wolfram (also the creator of Mathematica).

## Location 452–457

> Trading is always about making a profit and remaining the last person standing, which means that you leave trading on your terms, ahead of the game and because you no longer wish to trade. To do that, one needs to find what the old-time floor traders called an edge. I personally believe that anyone who succeeds at trading does so because he has discovered an edge. I also believe that there are probably as many different edges as there are traders.

## Page 5 · Location 550–553

> In the final analysis, I believe that any successful trader, discretionary or automated, does trade with a systematic trading strategy. In the end, as I think you will see if you work through this book, it is difficult to generate long-term and above-average trading or investment returns without a systematic approach.

## Page 6 · Location 577–582

> These strategies can be on a par with, worse than, or better than those of many professional investment firms. There is nothing to prevent the individual trader working on his own from creating very sophisticated and successful trading strategies. The technology, price history, and the software are all available. In fact, the capabilities of the contemporary investor equipped with testing software and a powerful computer (circa 2007) far exceeds that of the professional strategist working in 1990. In addition to those increased capacities, the availability, range, and sophistication of current technical analytical methods are also many times greater.

## Page 8 · Location 617–621

> Optimization refers to the process whereby a trading strategy is tested and refined so as to produce the best possible real-time trading profits. Optimization then is testing done correctly. Overfitting, which no sane strategist ever does intentionally, is optimization that has gone bad. Overfitting, then, is incorrect testing.

## Page 8 · Location 625–627

> The value of a trading strategy must be evaluated in two interrelated dimensions: profit and risk. One cannot judge these two components of trading performance in isolation. Trading always involves risk. Trading profit can be correctly evaluated only with respect to its risk, which is its major cost.

## Page 10 · Location 674–675

> It has been my experience that WFA is the only nearly fool-proof method (nothing in trading is 100 percent) of trading strategy optimization.

## Page 11 · Location 683–687

> A mechanical trading strategy, called simply a trading strategy, or strategy, is a set of objective and formalized rules external to and independent of the mind and emotions of the trader. The majority of successful traders employ a consistent set of rules, whether or not they are overtly formulated and tested as a formal trading strategy. The use of a consistent set of trading rules is essential to the management of risk and to the creation of trading profit.

## Page 12 · Location 722–724

> The model parameters selected during the optimization of a trading strategy are based on an objective function also known as optimization function and search parameter.

## Page 13 · Location 733–736

> Optimization proceeds through two levels. The first is an optimization of the trading strategy over a variety of different markets and time periods. The main purpose of this stage is to determine to what degree the trading strategy is enhanced by optimization. If the strategy demonstrates better performance under optimization, then it is taken to the final round of optimization and testing: the Walk-Forward Analysis.

## Page 13 · Location 738–746

> The optimization of the trading strategy under an exhaustive WFA measures the trading performance exclusively on the basis of out-of-sample trading, that is, on data other than those used to optimize the strategy. The first, and far and away the most important, objective of the WFA is to determine whether the trading strategy remains effective on unseen or out-of-sample price history. This, of course, is one of the most reliable and major predictors of real-time trading success. If it walks forward well, as we call it, then it is highly likely that it will continue to perform profitably in real-time trading. The second and next most important objective of the WFA analysis is to determine the optimal parameter values to be used with real-time trading. The third objective is to determine the sizes of the optimization window and the periodic rate at which the strategy is to be reoptimized.

## Page 14 · Location 765–767

> I personally believe that the most effective way to avoid overfitting during the optimization process is to perform optimization through a Walk-Forward Analysis.

## Page 14 · Location 772–773

> the most effective way to do this, which is to include out-of-sample testing in your optimization process.

## Page 14 · Location 774–776

> Once the full cycle of trading strategy development has been successfully completed—namely, strategy formulation, testing, optimization, walk-forward analysis, and evaluation—then, and only then, can real-time trading safely begin.

## Page 18 · Location 805–809

> The main reasons that a properly tested and validated systematic trading strategy helps in the pursuit of trading profit are its: • Verifiability • Quantifiability • Consistency • Objectivity • Extensibility

## Page 19 · Location 827–831

> Aside from all of the technical skills that the discretionary trader must master, first and foremost, the successful ones of long-standing tenure are masters of themselves. Remember that the inability to follow a proven strategy is high on the list of reasons for failure of the systematic trader. How much more difficult must it be for the discretionary trader who needs to be on and in control of himself day in and day out?

## Page 21 · Location 883–885

> “Perfect Profit is the sum total of all of the potential profit that could be realized by buying every bottom and selling every top.”

## Page 22 · Location 898–901

> Consider that a sustained annualized rate of return of 25 percent or more places a hedge fund or a commodity trading adviser (CTA) in the upper stratosphere of performance. According to the Barclay Trading Group, as of June 2007, only 16 CTAs (out of a universe of 375) produced a five-year annualized compound rate of return of 25 percent or better.

## Page 22 · Location 916–922

> the first and foremost advantage of a thoroughly tested systematic trading strategy is the determination that it, in fact, has a profit potential. Another way of looking at this is that a successful and fully tested systematic trading strategy is in itself a proof of the trading concept. Without a reasonable estimate of the potential risk-adjusted reward of a trading strategy, it is impossible to know if it is worth trading. Without a reasonable estimate of potential risk, it is impossible to know the true cost of trading with the strategy. How is it that we determine that a trading strategy has a positive profit expectancy? This is done through the construction and evaluation of a historical simulation.

## Page 23 · Location 928–931

> We are also very interested to see what the profit and risk of the trading strategy is over both ever-changing market conditions and different markets. If we find that the trading strategy produces profit over a range of conditions (necessary) and a variety of markets (desirable, but not necessary), we have further validation of the trading concept. We also know that we have a more valuable trading strategy.

## Page 23 · Location 940–942

> The results of such an approach, however, are well-documented and predictable. The likely outcome is an assured entry into that very nonexclusive “90 percent-of-all traders-who-trade-lose” club.

## Page 23 · Location 945–953

> It reminds me a of a W.D. Gann quote (paraphrased): “A doctor goes to medical school for four years before practicing medicine; a lawyer goes to law school for three years before beginning the practice of law; why then does a sufficient amount of money with which to trade qualify an unschooled individual to be a trader?” The Trading Advantage of Verification. When a trading strategy has successfully undergone the full testing cycle from start to finish, it has been verified to have a positive expectancy. It has also been verified that the trading strategy has a reasonably high likelihood of producing real-time trading returns relatively consistent with its historical simulation. Armed with this knowledge, the trader has a sound and rational basis for confidence in the trading strategy sufficient to trade with it and to follow it faithfully.

## Page 26 · Location 956–963

> An accurate measure of profit and the risk required to obtain it are needed for two main reasons. The first is to determine whether the risk-adjusted reward is equal to, inferior to, or superior to other competing trading and investment vehicles. The second is to determine the optimal account capitalization required to obtain the maximum rate of sustainable return. Another tremendous advantage of the quantification or statistical evaluation of a trading strategy is that it makes it possible to accurately compare different trading strategies to one another. Because of the varying profit and risk profiles of different trading systems and the profit potential of different markets, the only thing that can be meaningfully compared from one system to another is the rate of its risk-adjusted rate of return.

## Page 26 · Location 964–967

> A computer-tested trading strategy measures profit and risk. It also provides a large number of other very useful statistics, such as the number of trades, the value of the average trade, statistics about winning and losing runs, and trading performance on a year-by-year basis. These statistics collectively compose the performance profile. Table 2.2 is an example of such a profile.

## Page 26 · Location 974–976

> Let us calculate required capital at three times risk plus margin of $5,000. Strategy One produces a risk-adjusted return of 38.5 percent ($2,500/((3 × $500) + $5,000).

## Page 26 · Location 979–982

> Which strategy is better? If we were to judge by profit alone, Strategy Two would be the hands-down winner ($50,000 versus $2,500.) If we are to judge correctly by the respective risk-adjusted return, however, or what it costs to earn a dollar of profit, Strategy One produces a return that is more than 6 percent better.

## Page 27 · Location 986–988

> More important, it provides both a road map and a benchmark providing what should—and, more important, what should not—be expected from its real-time trading performance. It provides us with a mechanism to check the health of our trading strategy’s real-time performance.

## Page 27 · Location 989–992

> Suffice it to say that this performance profile is essential to the proper management of trading a systematic strategy. Without a precise, statistically reliable measurement of risk-adjusted returns, it is impossible to assess whether future profits and losses are in line with the strategy’s historical performance.

## Page 27 · Location 1000–1005

> The Trading Advantage of Quantification. The quantification of the risk and reward of the trading strategy provides the mathematical basis for the correct capitalization for real-time trading of the strategy. The statistical performance profile that is also a result of this quantification provides a set of milestones by which real-time trading performance can be evaluated. Aside from the value inherent in these features, this knowledge should further enhance the trader’s confidence in the trading strategy and the assurance of a positive outcome.

## Page 29 · Location 1046–1048

> The execution of a trading strategy automatically in real-time trading will produce profits and losses that are relatively consistent with the results of its profitable, verified, and robust historical simulation. The alteration of the rules in the execution of a systematic trading strategy will not.

## Page 29 · Location 1049–1050

> It is certainly true that the more robust and excellent the trading strategy, the less delicate this balance is. But even the most robust trading strategy can and will be

## Page 29 · Location 1050–1052

> even the most robust trading strategy can and will be eventually ruined by the acceptance or rejection of trading signals that can and will result from human interference.

## Page 29 · Location 1055–1058

> What does it mean to “unfaithfully follow” a systematic trading strategy? It means, in general terms, any human—and untested—override, interference, or alteration of the rule of the strategy. One way of looking at these human overrides, or as we’ll call it going forward, trader interference of a strategy, both in theory and in practice, introduces a new and untested variable(s) or rule(s) to the strategy.

## Page 30 · Location 1068–1076

> Now let us consider System Y. Y is a variant—perhaps an aberration—of X. It has a 45 percent accuracy rate, an average win of $1,000, and an unknown (Hmm, can we detect a possible case of trader interference going on here?) loss size. After 100 trades, it can be reasonably predicted that 45 trades will be wins, producing a profit in the vicinity of $45,000, and 55 trades that will be losers. However, the dollar value of these 55 losses is unknown due to Y’s lack of statistics and rules about risk. The net loss may be $11,000 (assuming an average loss of $200) yielding a net profit of $34,000. This would be great. Or, they may yield a net loss of $82,500 (assuming an average loss of $1,500) producing a net loss of $37,500. It could also be worse. As this simple example illustrates, ignorance of the risk of a trading strategy makes it impossible to properly assess the return of the strategy. Even worse, such ignorance of risk makes it impossible to properly capitalize the trading strategy.

## Page 31 · Location 1080–1082

> I have always considered it one of the great ironies of trading that, at least from what I have seen, it is the trade not taken that is so often a winner, and often enough to matter, a big winner. Conversely, the ones taken are typically losers.

## Page 31 · Location 1087–1104

> The trading strategy is on a hot streak and it has produced seven wins in a row and decent size ones, too, trading a unit size of two contracts per trade. Feeling flush from the extended winning streak and the big equity increase, the trader feels like the strategy can do no wrong and increases the unit size to ten contracts per trade. Well, anyone who has traded for any length of time knows that all winning streaks come to an end. So does that of our hot-handed trading strategy and its does so with three typically sized losers in a row at the larger trade size of ten contracts. Of course, a trade size five times larger than that which produced the winning streak will lose money at a rate five times greater. As a result, the trader winds up giving back all of the profits from the last winning run and much more as well. Without consistency of entry, exit, risk, and trade size on each and every trade, it is impossible to estimate the probability of success. It is essential to understand that without a thorough knowledge of strategy-specific risk, it is impossible to intelligently trade with any strategy, whether automatic or discretionary. Consistency also means knowing in advance how to act in any circumstance, based on preestablished and verified rules. Contrast this to the inconsistency and unpredictability of trading responses based on emotions such as fear and greed. Consider the tremendous advantage this confers during those infrequent but oftentimes very dramatic, fast, and either hugely rewarding or painfully costly price moves that occur from time to time driven by sudden, large, and unexpected political or economic events. The Trading Advantage of Consistency. A sound systematic trading strategy is that it consistently—with the uncluttered logic of mathematics and the relentlessness of computers—applies the same entry and exit rules without exception or deviation. As a result of this, and assuming a relative consistency in market activity from period to period, which is not always the case, the size and frequency of profits and losses will remain reasonably in line with that of its performance profile.

## Page 32 · Location 1110–1112

> One of the most valuable but often overlooked advantages of the systematic trading strategy is that, given sufficient trading capital and computing capacity, it can be applied to as many different markets and time frames in which it has proven itself to be effective.

## Page 32 · Location 1126–1127

> The Trading Advantage of Extensibility. A profitable and sound systematic trading strategy run by computer makes it possible to trade virtually as many markets for which the trader has capital.

## Page 33 · Location 1135–1142

> A historical simulation of a trading strategy is a model or representation of the trading performance—a historical profit and loss statement—produced by the rules of the trading model under evaluation. How is a trading simulation produced? Two things are needed: trading simulation software and a historical database of market prices. The first step, then, is to create a precise formulation of all of the rules of the trading strategy in a computer-testable language. Next, this strategy formulation is then processed by a computer application—a trading simulator—that has the ability to trade, or to apply, this strategy on historical data. The trading simulation software then collects all of the trades—buys, sells, and individual profit and loss—produced by the strategy during this historical period and a number of different statistical performance reports are created from them.

## Page 33 · Location 1143–1145

> There are a number of common statistics, however, that are included in all such reports, including, for example, net profit, maximum drawdown, number of trades, percentage of winners, and average trade.

## Page 34 · Location 1156–1156

> An accurate historical simulation of the trading strategy is the only way to determine whether the strategy has a positive expectancy.

## Page 34 · Location 1158–1161

> The historical simulation and its evaluation can and must answer two very important questions. First, the strategist needs to determine how effective the trading strategy has been historically. This needs to be evaluated as an investment comparing it to a variety of competing alternatives. This is explored in Chapter 11: The Evaluation of Performance.

## Page 34 · Location 1162–1163

> The second and far more important question is to determine the likelihood of the strategy producing returns in real-time trading in a manner consistent with that of its historical simulation.

## Page 34 · Location 1165–1166

> if one uses Walk-Forward Analysis to optimize and validate the strategy, arriving at this determination with a high degree of confidence is far more straightforward and mechanical.

## Page 34 · Location 1167–1168

> It is one of the central theses of this book that Walk-Forward Analysis is the most effective remedy for, and method of, avoiding overfitting. The historical simulation of course, is the mode of operation of Walk-Forward Analysis.

## Page 35 · Location 1169–1174

> We can see from this that the historical simulation by itself does not answer this very important question. Rather, this determination is arrived at by the evaluation of the • Historical simulation itself • Development process • Optimization process • Walk-Forward Analysis

## Page 35 · Location 1189–1193

> There are two main applications of this profile. The first is to use this information to determine the minimum proper capitalization of the trading account. The pursuit of consistent, high, and sometimes outsized trading profits is what drives the trader to trade. It is the size of the risk, however, that tells the trader how much it is going to cost to achieve this profit. Reward cannot be properly evaluated in the absence of its attendant risk. The only meaningful and practical measure of performance is in the form of the risk-adjusted return.

## Page 36 · Location 1195–1198

> What concerns us here is that this measure is central to the proper capitalization of a trading account. Proper capitalization means that an account is funded with sufficient capital to absorb the maximum risk, or drawdown, that the trading strategy may endure. More important, not only can it absorb this drawdown, the account must have sufficient capital remaining after drawdown to continue to trade with the strategy.

## Page 36 · Location 1201–1203

> The second and perhaps even more important application of the performance profile is its application to the evaluation of real-time trading performance. With a properly developed systematic trading strategy, real-time trading performance should conform in general to that of the statistical profile of performance.

## Page 36 · Location 1211–1212

> One does need to keep in mind, however, that outsize losses usually follow outsize profits. The sword of volatility cuts both ways.

## Page 37 · Location 1212–1217

> the benefits of the historical simulation are a: • Determination of positive expectancy and a measure thereof • Method for determining the likelihood of real-time trading profit • Method of properly capitalizing a trading account • Yardstick for real-time trading

## Page 37 · Location 1222–1226

> Most trading strategies have rules that accept various different values. For example, a two moving average crossover system will have a value for the length of each of the averages. Whether the analyst chooses to optimize these values, any strategy that can accept different values for its rules is optimizable, if so desired. The first function of optimization then, is to determine the appropriate values for the most robust implementation of the trading strategy.

## Page 37 · Location 1228–1229

> The second, and more important and more difficult, function of optimization is to arrive at an estimation of the robustness of the trading strategy.

## Page 38 · Location 1238–1244

> It is also a function of optimization and Walk-Forward Analysis to adapt a trading strategy to different types of markets. All markets have their own unique personalities. A trading strategy may perform well in one market with one set of values and poorly in another with those same values. I do not believe that one set of parameters should necessarily be sufficient for every market in which the strategy is traded. In my experience, such a situation is rare. In fact, I tend to prefer different values for a trading model for different markets, in that it offers an additional dimension of portfolio diversification. Optimization will identify the best set of parameter values for each market.

## Page 38 · Location 1244–1245

> different traders have different trading capital, time available, computing resources, profit expectations, tolerance for risk, and temperaments. Another application of optimization then, is to adapt the trading strategy to the individual needs of the trader.

## Page 38 · Location 1247–1253

> the benefits of optimization are the: • Achievement of peak performance • Evaluation of one measure of robustness of the strategy • Maintenance of peak performance • Adaptation to changing market conditions • Adaptation to different markets • Adaptation to different traders

## Page 38 · Location 1255–1257

> Walk-Forward Analysis is a systematic and formalized manner of performing what has been referred to as a rolling optimization or a periodic reoptimization.

## Page 39 · Location 1260–1264

> Another important advantage of Walk-Forward Analysis is to produce peak trading performance as markets, trends, and volatility change. Since the Walk-Forward Analysis provides a method of periodic reoptimization with current price action, this often means that it can produce trading performance superior to that of traditional optimization. Since this periodic reoptimization is done with a strategy-appropriate amount of current price data, this also provides an efficient way to continuously adapt a trading model to ongoing changes in market conditions.

## Page 39 · Location 1264–1270

> the main benefits of Walk-Forward Analysis are the: • Evaluation of the likelihood of a trading strategy performing well in real-time trading • Measurement of the robustness of the trading strategy • Achievement of peak trading performance at a level superior to that of traditional optimization • Maintenance of superior trading performance through more effective adaptation to changing market conditions

## Page 39 · Location 1273–1280

> These five important benefits are: 1. A comprehensive and precise knowledge of the strategy’s reward and risk 2. A high degree of confidence that your strategy will perform in real-time trading as it has in historical simulation 3. A basis for a rational and reliable evaluation of the trading strategy’s real-time performance 4. The confidence to stick with the trading strategy during good times and bad 5. A comprehensive knowledge of the trading strategy and of its real-time trading performance, which makes it easier to successfully improve and further refine the trading strategy over time

## Page 40 · Location 1291–1299

> W.D. Gann had a saying of which I have always been particularly fond. He said, “Never trade on hope.” The systematic approach to trading strategy development leads to knowledge. With knowledge we don’t need to hope. The most important knowledge which this process produces is of: • The workings of our strategy • Its performance • Its risk • Its robustness • Its likelihood to produce profit in real-time • A method to evaluate its real-time performance

## Page 43 · Location 1325–1331

> the development and application of a trading strategy follows eight steps: 1. Formulation 2. Specification in computer-testable form 3. Preliminary testing 4. Optimization 5. Evaluation of performance and robustness 6. Trading of the strategy 7. Monitoring of trading performance 8. Refinement and evolution

## Page 44 · Location 1335–1345

> There are two main philosophical approaches to trading strategy development. The first approach applies reason in the original design and conceptualization of the trading strategy. This is followed by the systematic and empirical verification of each component of the trading strategy. Every element of the strategy must make sense before the testing process even begins. I refer to this as the scientific approach to strategy development and it is the approach that I primarily follow in this book and in my trading. The second approach might best be called the empirical approach. To a large extent, the logic and reason of the strategy developer is eclipsed, and to a varying extent, replaced by computer intelligence. The empirical approach uses various forms of computer software technology to search a space comprising a vast library of indicators, patterns, and price action to assemble a profitable set of trading rules that become the trading strategy if accepted and put to use. In other words, the computer picks and optimizes a trading strategy developed in this manner.

## Page 44 · Location 1346–1350

> The main drawback to this approach is that it is relatively easy for this to devolve into a morass of sophisticated overfitting. Also, the actual logic of the trading strategies that emerge from this process is often not visible to the trader. The trader is, typically, and with good reason, unwilling to invest millions of dollars traded on a strategy that is essentially unknown to the trader.

## Page 47 · Location 1406–1407

> It is rather well-known that technologies such as neural nets are often referred to as the ultimate curve-fitting technology.

## Page 47 · Location 1411–1416

> I believe, to a large extent, that most professional traders are reluctant to, and for the most part do not, use empirically derived trading strategies, largely because the rules and trading logic of such strategies are not transparent or visible. The reluctance of the professional trading community to embrace the empirically derived strategy is also due, in part, to the inherent difficulties and high costs associated with the creation and validation of empirically derived strategies. But for many traders, and I include myself, the inability to perform the necessary due diligence on such black-box trading strategies is their biggest drawback.

## Page 89 · Location 2149–2151

> Definition: A position sizing rule determines the number of contracts or shares that are committed to each trade.

## Page 89 · Location 2160–2171

> One of the primary reasons for this difficulty, however, arises from the non-Gaussian (especially fat-tailed) distributions that are typical of financial markets and the returns that are produced by them. Such distributions reduce the effectiveness of traditional statistical estimation methods that are employed in more sophisticated sizing methods. Another level of difficulty in sizing arises from the lack of statistical robustness of many of the measures that are employed in sizing rules. Last, but not least, position sizing takes on another level of complexity when a trading strategy is incorporated in a portfolio of markets. These are complex and highly technical issues that are beyond the scope of this book. It is absolutely essential, however, to realize that position sizing can make or break a trading strategy. We will look at four position sizing examples: 1. Volatility adjusted 2. Martingale 3. Anti-Martingale 4. The Kelly method A volatility adjusted sizing rule uses the size of the account and the size of the risk

## Page 90 · Location 2172–2181

> Definition: Volatility adjusted position sizing determines the number of contracts or shares per trade as a fixed percentage of trading equity divided by the trade risk. For example, assume: 1. A risk size of 3 percent of equity 2. A risk per contract of $1,000 3. An account size of $250,000 The trade unit would be seven contracts and is calculated as follows: Total Equity to Risk = $7,500($250,000 × .03) Number of Contracts = 7($7,500/$1,000 = 7.5 rounded down = 7)

## Page 90 · Location 2182–2184

> Definition: The Martingale sizing rule doubles the trade size after each loss and starts at one unit after each win.

## Page 90 · Location 2185–2189

> Definition: The anti-Martingale sizing rule doubles the number of trading units after each win, and starts at one unit after each loss. Optimal f (fixed fractional trading) was introduced by Ralph Vince in 1990. It is based on a formula derived from the Kelly method, which was, in turn, applied by Professor Edward Thorpe to gambling and trading.

## Page 90 · Location 2190–2200

> Kelly % = (Win % - Loss %)/(Average Profit/Average Loss) For example, assume a strategy that has a winning percentage of 55 percent, an average win of $1,750, and average loss of $1,250. The Kelly percent will tell us what percentage of our trading capital to risk is on the next trade. Kelly % = (55 − 45)/($1,750/$1,250)
> Kelly % = 10/1.4
> Kelly % = 7.14 % Let us use this to calculate our position size. We have our risk size of 7.14 percent of equity from the Kelly formula, a risk per contract of $1,000, and an account size of $250,000. The trade size would be seventeen contracts and is calculated as follows: Total Equity to Risk = $250,000 × .0714
> Total Equity to Risk = $17,850
> Number of Contracts = $17,850/$1,000 = 17.65
> Number of Contracts = 17.85 rounded down = 17 contracts

## Page 91 · Location 2216–2224

> Definition: Scaling into a position adds incrementally to an existing position as the market moves in the profitable direction of the trade. Definition: Scaling out of a position incrementally decreases an existing position. An example of scaling into a position would be to add one trading unit to a position each time open equity profit increases by $1,000. An example of scaling out of a position would be to remove one trading unit every time open equity increases by $1,000. An example of both scaling into and out of a position would be to add one trading unit to a position each time open equity profit increases by $1,000 until a maximum open equity profit of $5,000 on the oldest position is reached, and to then remove one trading unit each time open equity increases by an additional $1,000.

## Page 92 · Location 2227–2229

> Rather, as a valued associate and friend once quipped: “The best trading system is one that relies upon an irresistible force of nature for its profits.”

## Page 106 · Location 2248–2250

> Certain reports are essential, such as the performance summary (see Table 6.1) and the trade listing. Others are extremely valuable, such as performance broken down over various time intervals and a chart of the daily equity curve.

## Page 106 · Location 2252–2255

> over the historical period for which the simulation was created. Key statistics are net profit, maximum drawdown, number of trades, percentage of winners, average trade, and the ratio of average win to average loss. The Sharpe Ratio is extremely valuable because it is a key statistic for many professional investors,

## Page 106 · Location 2261–2262

> This report is a tabulation of the historical performance of the trading strategy on a trade-by-trade basis. This report typically provides the entry date, entry price, label, exit date, exit price, label, the trade profit or loss, and a cumulative total profit and loss

## Page 106 · Location 2268–2272

> This report is typically useful during the preliminary stages of the development of the strategy. It is essential then, as a diagnostic, to confirm that trading is proceeding as specified. The trade list has another more valuable and underused application. That is as a trading simulator, during which the trader can look at each trade and its behavior on a price chart to get an intuitive feel for what it will be like to trade with this strategy in real time when he will be watching these trades on a daily, and perhaps far more frequent, basis.

## Page 107 · Location 2288–2290

> This is typically a graphical report that displays the cumulative profit and loss of the trading strategy. It is typically plotted as line chart in a window below a price chart of the market on which the strategy was simulated (see Figure 6.1).

## Page 107 · Location 2291–2294

> One of the primary benefits of the equity curve is a quick indication of its relative smoothness and consistency, or the lack thereof. Another extremely valuable benefit of the equity curve is the insight that it provides as to the performance of the market under different conditions. Of particular value is to examine market action during periods of suboptimal performance, particularly during maximum drawdown.

## Page 108 · Location 2298–2300

> This is a report that provides some essential information about the tradability and robustness of the trading strategy. It breaks down trading performance on an interval basis. For most strategies, typically, we are interested in performance on an annual basis.

## Page 109 · Location 2319–2321

> The creation of the most perfect simulation possible requires two things: a thorough knowledge and understanding regarding the implications of software limitations and data issues, and the use of conservative assumptions regarding costs and slippage in its various forms.

## Page 109 · Location 2325–2328

> As technical and as innocuous as this may sound, the use of rounding—or its absence—to the proper tick price of data and of entry and exit orders can have the cumulative effect of exaggerating trading performance. It can even lead to another area of error, which is the inclusion or omission of trades that would not be triggered in actual trading.

## Page 109 · Location 2328–2332

> All markets trade in increments of minimum fluctuations, or ticks. For example, soybeans trade in ticks of .25 cents, T-bonds in .03125s and the S&P trades in .10 of a point. Obviously, when placing an order, they are always placed at a price reflecting the proper tick such as 1535.60 for the S&P, 109.08 for T-bonds and 812.5 for soybeans. Why is this a problem for simulation software? Some applications will round both the data and the orders to their proper tick value. This is called doing tick math.

## Page 110 · Location 2333–2335

> If proper tick math is not used in simulations, it can result in orders that are filled and should not have been or not filled that should have been. The second source of error is that it can lead to a subtle, small, and persistent over- or understatement of profits.

## Page 125 · Location 2663–2670

> The perpetual contract introduces three unique problems. First, it does not contain real price history. Every price is transformed. Second, it introduces a new distortion of its own and it tends to somewhat artificially dampen actual price volatility by behaving differently from the actual price data themselves. Third, entry orders for real-time trading derived from it must be transformed. If used to create daily trading signals, these signal prices will need to be adjusted so as to be usable in real-time trading. This added price distortion may be of little consequence, with a very slow system that trades for the big moves. This distortion, however, may prove to be a serious problem with a very active trading system that targets small moves and is highly sensitive to short-term changes in volatility.

## Page 125 · Location 2671–2680

> The adjusted continuous contract combines the best of all of the preceding alternatives. It merges front expiration price data into a continuous price history. It mathematically removes all of the price roll gaps, however. It can be done in two ways. Contracts can be adjusted, keeping the most recent data unchanged and adjusting all preceding data up or down an amount equal to the roll gaps. This is a back-adjusted continuous contract (see Figure 6.4). A front-adjusted continuous contract adjusts from the beginning of the file to the end. This leaves the most distant data in their natural form and the most current data are adjusted. The neutral data transform preserves the relative differences between prices. It introduces a distortion with any calculations that use percentages of price. It cannot be used with charting applications that use absolute prices for support and resistance. Back-adjusted contracts can also have negative prices because of the gap adjustments.

## Page 126 · Location 2689–2693

> Definition: The test window is the length of the historical price data on which a trading strategy is evaluated by historical simulation. Two main considerations must be satisfied when deciding the size of the test window: statistical soundness and relevance to the trading system and to the market.

## Page 127 · Location 2699–2704

> The test window must be large enough to generate statistically sound results and also include a broad sample of data conditions. Statistically sound means two things. There must be a sufficiently large number of trades so as to be able to draw meaningful conclusions. The test window must also be large enough to allow enough degrees of freedom for the number and length of the variables employed by the trading strategy. If these guidelines are not followed, the results of the historical simulation are likely to be deficient in statistical robustness, and are therefore suspect.

## Page 127 · Location 2709–2711

> Standard Error = Standard Deviation/Square Root of the Sample Size

## Page 128 · Location 2716–2720

> Standard error will provide us a measure of reliability of our average win as a function of the number of winning trades, that is, the sample size. For example, if the average win is $200 and has a standard error of $50, then the typical win will be within a range of $150 to $250 ($200 +/- $50.) The wise strategist will always err to the side of conservatism, so he will assume that the average win is likely to be $150 (the pessimistic side of the range of expected wins).

## Page 129 · Location 2734–2734

> The analysis of a historical simulation is a statistical analysis of its trades.

## Page 129 · Location 2738–2741

> This desire for a large trade sample can lead to a problem in the testing of long-term trading systems that trade infrequently. The best way to attempt to get a sufficient number of trades when testing a slower trading strategy is to make the test window as large as possible. Statisticians seem fond of the number 30 as the smallest sample size that can be evaluated statistically with confidence.

## Page 129 · Location 2748–2750

> The stability of a trading system refers to the overall consistency of its trading performance. The more consistent a trading system is in each of its performance dimensions, the more stable and reliable it tends to be in real-time trading.

## Page 129 · Location 2752–2753

> Trades should be relatively evenly distributed throughout the test window. The smaller the standard deviation of the size and the length of wins and losses, the more stable the strategy is likely to be.

## Page 130 · Location 2754–2755

> It is also better to have consistent trading performance on a quarter-by-quarter and year-by-year basis. The more consistent trading performance is from parameter set to parameter set, the more robust the strategy.

## Page 130 · Location 2756–2757

> Stability in historical simulation is one of the more important predictors of reproducibility in real-time trading.

## Page 130 · Location 2758–2766

> It is a fundamental principle of statistics that for a statistical test to produce reliable conclusions, it must begin with sufficient degrees of freedom. The term degrees of freedom is quite descriptive, in the sense that it is the number of observations in the data collection that are free to vary after the sample statistics have been calculated.2 Consider degrees of freedom as the simulation sample size adjusted for the number of conditions and rules placed upon it. The simulation test space is reduced in proportion to the number of degrees of freedom that are consumed by the rules and variables of the trading strategy.

## Page 130 · Location 2773–2775

> The degrees of freedom left in our example are 70 percent, which is not that attractive. As a rule of thumb, we would prefer that the remaining degrees of freedom exceed 90 percent. Therefore, there is no point in performing this simulation as stipulated.

## Page 131 · Location 2778–2784

> This simulation can proceed with 99 percent degrees of freedom. This will produce a more statistically reliable historical simulation. Consider two other applications of this principle. In the first, a trading strategy has 100 rules and a simulation with one hundred 100 days of data is considered. Applying our formula, we see that this leaves us with no degrees of freedom. It is easy to see that this test is absurd. In contrast, consider a simulation of a trading strategy with 1 rule and 100 days of data. Even though it is a small data sample, the 99 percent degrees of freedom is acceptable.

## Page 131 · Location 2789–2796

> Very fast trading strategies on volatile markets such as the S&P index typically benefit from smaller test window sizes. For example, a fast countertrending strategy exploiting a short (three-day) price swing may well benefit from short windows such as one to three years. Conversely, a small test window is generally not capable of producing an adequate trade sample for a longer-term system with a slower trading pace. It is a function of the slower pace of trading and of the disproportionate consumption of degrees of freedom by longer indicators typical of slower-paced strategies. Slower trend-following strategies trading in markets that are more highly trend-persistent like the yen will typically benefit from longer windows in the three-to-six year (and beyond) range.

## Page 132 · Location 2801–2804

> There are four different types of markets: 1. The bull 2. The bear 3. The cycling 4. The congested

## Page 132 · Location 2811–2812

> that are going up for a sustained time. There are typical bull markets. A regression line drawn through a typical bull market will have a slope between roughly 15 and 50 degrees.

## Page 132 · Location 2814–2817

> There can be roaring bull markets as well. A line of regression drawn through such a market may have an angle between 50 and 70 degrees and sometimes beyond. Such a market looks as if it is exploding. Roaring bull markets are rarer than typical bull markets. They are also less sustainable and therefore relatively short-lived (see Figure 6.6).
