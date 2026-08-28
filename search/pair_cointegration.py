"""Daily cointegration test for a pair of price series.

Reproduces the GLD-vs-GDX pair-trading analysis from Ernest Chan,
*Quantitative Trading* (Example 7.2, Ch.7; the training-set variant is Ch.3 /
p.63): a hedge-ratio regression, an Engle-Granger / CADF unit-root test on the
residual spread, and an Ornstein-Uhlenbeck half-life.

EXPLORATORY / a method reproduction -- NOT a pinned result. Nothing here is
frozen by a regression test.

Three steps, numpy-only for the core (statsmodels is an optional cross-check):

1. Hedge ratio. OLS of A's close on B's close, with an intercept -- this is the
   cointegrating regression the CADF test uses. The residual ``z = A - beta*B
   - alpha`` is the candidate stationary spread. ``origin=True`` additionally
   reports a separate through-origin slope (Chan's ``ols(GLD, GDX)`` = 1.6766);
   it is display-only and does not enter the test.
2. Engle-Granger / CADF test. An Augmented Dickey-Fuller unit-root test on that
   residual spread, compared to MacKinnon critical values for the residual-based
   (cointegration) test with N=2 series. More negative than the crit value
   rejects the "no cointegration" null.
3. Ornstein-Uhlenbeck half-life. OLS of the daily change in the spread on the
   lagged spread level gives the mean-reversion speed theta; the half-life
   ln(2)/theta is the expected holding period.

Reproducing the book (``--ch7`` / ``--ch3``): Chan reports two numbers
per pair, from two regressions. His ``cadf(GLD, GDX, 0, 1)`` runs the
with-intercept test (t-stat, AR(1)/half-life), and his ``ols(GLD, GDX)`` quotes
a through-origin hedge of 1.6766. Both come from the GLD-GDX intersection: the
full 2006-05-23 .. 2007-11-30 window (Ch.7: t=-3.357, ~95%) and the
first-252-day training set on p.63 (Ch.3: t=-3.18, ~90%). This module matches
the t-stat and half-life closely (t=-3.45 full / -3.09 train) and lands the
origin hedge at ~1.638. The exact 1.6766 is a lost data vintage -- Yahoo
re-scales adjusted close by every later dividend, so no modern download hits it;
independent reproductions converge on ~1.638 too. Raw close is the closest
modern proxy to Chan's 2007-era adjusted series (GDX had barely any dividends
stripped then), so ``--book`` uses raw.

If ``statsmodels`` is installed, the report prints an independent second
opinion (its ``coint`` and ``adfuller``) next to the hand-rolled numbers.
statsmodels is OPTIONAL and deliberately NOT in ``requirements.txt``, so the
default path and the whole engine test suite stay dependency-free.

Usage:
    python -m search.pair_cointegration                    # GLD vs GDX, full history
    python -m search.pair_cointegration --ch7              # Chan's Ch.7 full-window run
    python -m search.pair_cointegration --ch3              # Chan's Ch.3 / p.63 run
    python -m search.pair_cointegration --a SPY --b IVV --origin
    python -m search.pair_cointegration --selftest         # verify the math
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from common.paths import data_path

# MacKinnon (2010) asymptotic critical values.
#
# Standard ADF unit-root test on a single series, constant, no trend.
ADF_CRIT_CONST: dict[str, float] = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
# Engle-Granger residual-based test: N=2 I(1) variables, a constant in the
# cointegrating regression, no trend. More demanding than the plain ADF
# values because the spread was itself estimated (the hedge ratio is fitted,
# which mechanically makes residuals look more stationary).
EG_CRIT_N2: dict[str, float] = {"1%": -3.90, "5%": -3.34, "10%": -3.04}


@dataclass(frozen=True)
class OLSFit:
    """Coefficients, their standard errors, and residuals from one OLS fit."""

    beta: NDArray[np.float64]
    se: NDArray[np.float64]
    resid: NDArray[np.float64]


def _ols(y: NDArray[np.float64], x: NDArray[np.float64]) -> OLSFit:
    """Ordinary least squares of ``y`` on the columns of ``x`` (no implicit
    intercept -- add a column of ones yourself if you want one)."""
    beta, _res, _rank, _sv = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    dof = len(y) - x.shape[1]
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return OLSFit(beta=beta, se=se, resid=resid)


def adf_tstat(series: NDArray[np.float64], lags: int = 1, *, constant: bool = True) -> tuple[float, int]:
    """Augmented Dickey-Fuller t-statistic on the lagged-level coefficient.

    Regression (with ``lags`` augmenting terms):

        d_y_t = [const] + rho * y_{t-1} + sum_i gamma_i * d_y_{t-i} + eps_t

    Returns ``(tstat, nobs)``. A large-magnitude *negative* ``tstat`` is
    evidence against a unit root (i.e. for stationarity / mean reversion).
    Pass ``constant=False`` for regression residuals, which are mean-zero by
    construction and so take no deterministic term.
    """
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)  # dy[k] = y[k+1]-y[k]; so d_y_t == dy[t-1]
    n = len(dy)
    if n - lags < 10:
        raise ValueError(f"series too short: {len(y)} points, {lags} lags")

    # Response d_y_t for t = lags+1 .. N-1  ->  dy[lags:]
    response = dy[lags:]
    columns: list[NDArray[np.float64]] = [y[lags:-1]]  # y_{t-1}
    for i in range(1, lags + 1):
        columns.append(dy[lags - i : n - i])  # d_y_{t-i}
    if constant:
        columns.append(np.ones(len(response)))
    design = np.column_stack(columns)

    fit = _ols(response, design)
    tstat = float(fit.beta[0] / fit.se[0])  # coefficient on y_{t-1}
    return tstat, len(response)


@dataclass(frozen=True)
class CointResult:
    """Everything the CLI needs to report one pair's cointegration test.

    ``hedge_ratio``/``intercept``/``spread`` come from the with-intercept
    cointegrating regression that drives the CADF test. ``origin_hedge`` is the
    separate through-origin slope Chan quotes (his ``ols(GLD, GDX)`` = 1.6766),
    filled only when asked -- it does not enter the test.
    """

    hedge_ratio: float
    intercept: float
    spread: NDArray[np.float64]
    adf_stat: float
    nobs: int
    half_life: float
    origin_hedge: float | None = None


def engle_granger(
    a: NDArray[np.float64], b: NDArray[np.float64], lags: int = 1, *, origin: bool = False
) -> CointResult:
    """Two-step Engle-Granger cointegration test.

    The test always uses the statistically standard *with-intercept* form: fit
    ``a = alpha + beta*b + z``, run the ADF on the mean-zero residual ``z`` (no
    deterministic term, compared to the with-constant N=2 critical values), and
    measure the OU half-life. This is what Chan's ``cadf(GLD, GDX, 0, 1)``
    computes -- its t-stat and AR(1)/half-life match the book.

    ``origin=True`` ADDITIONALLY reports a through-origin hedge ratio from a
    separate no-constant OLS -- Chan's ``ols(GLD, GDX)``, which he quotes as
    1.6766. It is display-only and does NOT change the test.
    """
    design = np.column_stack([b, np.ones(len(b))])
    fit = _ols(a, design)
    hedge_ratio = float(fit.beta[0])
    intercept = float(fit.beta[1])
    spread = fit.resid
    adf_stat, nobs = adf_tstat(spread, lags=lags, constant=False)
    origin_hedge = float((b @ a) / (b @ b)) if origin else None
    return CointResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
        adf_stat=adf_stat,
        nobs=nobs,
        half_life=ou_half_life(spread),
        origin_hedge=origin_hedge,
    )


def ou_half_life(spread: NDArray[np.float64]) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life, in the series' own time
    step (trading days here). Regress ``d_z_t`` on ``z_{t-1}``; the slope is
    ``-theta``; half-life is ``ln(2)/theta``. Returns ``+inf`` when the spread
    does not mean-revert (non-negative slope)."""
    z = np.asarray(spread, dtype=float)
    dz = np.diff(z)
    zlag = z[:-1]
    design = np.column_stack([zlag, np.ones(len(zlag))])
    fit = _ols(dz, design)
    slope = float(fit.beta[0])  # == -theta
    if slope >= 0:
        return math.inf
    return math.log(2.0) / (-slope)


def load_close(ticker: str, *, unadjusted: bool = False) -> pd.Series:
    """Load a ticker's daily close as a date-indexed Series.

    Reads ``data/{ticker}_20yr_prices.csv`` (Yahoo dividend-adjusted, the repo
    default) or ``..._20yr_prices_unadjusted.csv`` (raw close) when
    ``unadjusted=True``. Both are written by yfinance with a 3-row multi-index
    header (Price/Close, Ticker/SYM, Date/blank); rather than hard-code the
    skip count we drop every leading row whose first field is not a parseable
    date, so either header shape loads.

    The basis matters for cross-ticker levels: GLD pays no dividend, so its
    adjusted close already equals its raw close, but GDX's dividends make its
    adjusted history ~20% below raw. Chan's book used 2008-era (raw) prices,
    so faithful reproduction of the 1.6766 hedge ratio needs ``unadjusted``.
    """
    suffix = "_20yr_prices_unadjusted.csv" if unadjusted else "_20yr_prices.csv"
    path = data_path(f"{ticker.lower()}{suffix}")
    raw = pd.read_csv(path, header=None, names=["date", "close"], usecols=[0, 1])
    with warnings.catch_warnings():
        # The header rows ("Date", "Ticker") don't parse as dates; coerce drops
        # them to NaT. pandas warns about the mixed formats -- expected here.
        warnings.simplefilter("ignore", UserWarning)
        dates = pd.to_datetime(raw["date"], errors="coerce")
    mask = dates.notna()
    series = pd.Series(
        pd.to_numeric(raw["close"][mask], errors="coerce").to_numpy(dtype=float),
        index=pd.DatetimeIndex(dates[mask]),
        name=ticker.upper(),
    )
    return series.sort_index()


def aligned_closes(
    a: str,
    b: str,
    *,
    start: str | None = None,
    end: str | None = None,
    unadjusted: bool = False,
) -> pd.DataFrame:
    """Inner-join two tickers' closes on their common trading days, optionally
    clipped to the inclusive window ``[start, end]`` (both ``YYYY-MM-DD``)."""
    joined = pd.concat(
        [load_close(a, unadjusted=unadjusted), load_close(b, unadjusted=unadjusted)],
        axis=1,
        join="inner",
    ).dropna()
    joined.columns = [a.upper(), b.upper()]
    if start is not None:
        joined = joined.loc[joined.index >= pd.Timestamp(start)]
    if end is not None:
        joined = joined.loc[joined.index <= pd.Timestamp(end)]
    return joined


def _verdict(stat: float, crit: dict[str, float]) -> str:
    """The most demanding level (lowest %) at which the stat rejects."""
    for level in ("1%", "5%", "10%"):
        if stat < crit[level]:
            return f"REJECTS the no-cointegration null at the {level} level"
    return "fails to reject -- no evidence of cointegration"


def run(
    a: str,
    b: str,
    lags: int,
    *,
    start: str | None = None,
    end: str | None = None,
    unadjusted: bool = False,
    origin: bool = False,
    reference: str | None = None,
) -> None:
    """Load the pair, run the test, and print a book-style report."""
    closes = aligned_closes(a, b, start=start, end=end, unadjusted=unadjusted)
    if len(closes) < 30:
        raise SystemExit(
            f"only {len(closes)} common trading days in the window -- need >= 30 "
            f"(check --start/--end and that both price files exist)"
        )
    av = closes[a.upper()].to_numpy(dtype=float)
    bv = closes[b.upper()].to_numpy(dtype=float)
    result = engle_granger(av, bv, lags=lags, origin=origin)

    span_start = closes.index[0].date()
    span_end = closes.index[-1].date()
    au, bu = a.upper(), b.upper()
    basis = "raw / unadjusted closes" if unadjusted else "Yahoo dividend-adjusted closes"

    print("Pair cointegration -- daily closes   (EXPLORATORY, not a pinned result)")
    print(f"  A = {au}   B = {bu}")
    print(f"  Span: {span_start} .. {span_end}   (N = {len(closes)} trading days)")
    print(f"  Price basis: {basis} (both legs)")
    if reference is not None:
        print(f"  Book reference: {reference}")
    print()
    print(f"Cointegrating regression (OLS with intercept):  {au} = alpha + beta*{bu} + z")
    print(f"  hedge ratio beta = {result.hedge_ratio:.4f}    intercept alpha = {result.intercept:.4f}")
    print(f"  => spread z = {au} - {result.hedge_ratio:.4f}*{bu} {-result.intercept:+.4f}")
    if result.origin_hedge is not None:
        note = "   <- Chan quotes this (his ols); book value 1.6766" if reference is not None else ""
        print(f"  through-origin hedge (no intercept, ols({au},{bu})) = {result.origin_hedge:.4f}{note}")
    print()
    print(f"Engle-Granger / CADF test (ADF on residual spread, {lags} lag, no const):")
    print(f"  ADF t-statistic = {result.adf_stat:.4f}   (nobs = {result.nobs})")
    crit = EG_CRIT_N2
    print(f"  MacKinnon crit (N=2, const, asymptotic):  "
          f"1% {crit['1%']}   5% {crit['5%']}   10% {crit['10%']}")
    print(f"  Verdict: {_verdict(result.adf_stat, crit)}")
    print()
    print("Ornstein-Uhlenbeck half-life:")
    if math.isinf(result.half_life):
        print("  spread does not mean-revert (non-negative OU slope) -- half-life undefined")
    else:
        print(f"  half-life = {result.half_life:.1f} trading days")
    print()
    print("Caveats: asymptotic critical values (a fitted hedge ratio biases the")
    print("stat toward rejection) and a single fixed hedge ratio over the whole")
    print("span. Cointegration is window-dependent; a borderline verdict deserves")
    print("a cross-check.")
    _print_crosscheck(av, bv, result.spread, lags)


def _print_crosscheck(
    a: NDArray[np.float64], b: NDArray[np.float64], spread: NDArray[np.float64], lags: int
) -> None:
    """Independent second opinion from statsmodels, if it is installed.

    Optional by design: absence prints a hint and returns, so the primary
    numpy-only path never depends on it. ``coint`` re-fits the whole
    with-intercept Engle-Granger test; ``adfuller`` re-runs the ADF on our own
    residual spread. Both should land close to the hand-rolled t-statistic.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, coint
    except ImportError:
        print()
        print("statsmodels cross-check: not installed -- skipped "
              "(uv pip install statsmodels to enable).")
        return

    with warnings.catch_warnings():
        # adfuller in statsmodels 0.15 warns that its return type will change
        # in a future release; we read only the first two elements, so silence
        # it rather than pin a version-specific kwarg.
        warnings.simplefilter("ignore", FutureWarning)
        eg_t, eg_p, eg_crit = coint(a, b, trend="c", maxlag=lags, autolag=None)
        adf_stat, adf_p = adfuller(spread, maxlag=lags, regression="n", autolag=None)[:2]
    print()
    print("statsmodels cross-check (independent implementation):")
    print(f"  coint()  Engle-Granger t = {eg_t:.4f}   p = {eg_p:.4f}   "
          f"crit 1%/5%/10% = {eg_crit[0]:.3f}/{eg_crit[1]:.3f}/{eg_crit[2]:.3f}")
    print(f"  adfuller() on our spread:  t = {adf_stat:.4f}   p = {adf_p:.4f}")


def selftest() -> None:
    """Verify the ADF and cointegration arithmetic on synthetic series with
    known answers. Deterministic (seeded); prints OK or raises."""
    rng = np.random.default_rng(0)
    n = 2000

    # 1. Pure random walk -> unit root -> ADF should NOT reject.
    walk = np.cumsum(rng.standard_normal(n))
    rw_stat, _ = adf_tstat(walk, lags=1, constant=True)
    assert rw_stat > ADF_CRIT_CONST["10%"], f"random walk wrongly rejected: {rw_stat:.3f}"

    # 2. Stationary AR(1), phi=0.2 -> strong mean reversion -> ADF REJECTS.
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.2 * ar[t - 1] + rng.standard_normal()
    ar_stat, _ = adf_tstat(ar, lags=1, constant=True)
    assert ar_stat < ADF_CRIT_CONST["1%"], f"stationary AR(1) not rejected: {ar_stat:.3f}"

    # 3. Cointegrated pair: shared random-walk factor + small noise -> REJECTS.
    factor = np.cumsum(rng.standard_normal(n))
    a = factor + rng.standard_normal(n) * 0.5
    b = 0.5 * factor + rng.standard_normal(n) * 0.5
    coint = engle_granger(a, b, lags=1)
    assert coint.adf_stat < EG_CRIT_N2["1%"], f"cointegrated pair not detected: {coint.adf_stat:.3f}"
    assert 0 < coint.half_life < 50, f"implausible half-life: {coint.half_life:.2f}"

    # 4. Independent random walks: NOT cointegrated -> fails to reject.
    ind_a = np.cumsum(rng.standard_normal(n))
    ind_b = np.cumsum(rng.standard_normal(n))
    ind = engle_granger(ind_a, ind_b, lags=1)
    assert ind.adf_stat > EG_CRIT_N2["10%"], f"independent walks wrongly cointegrated: {ind.adf_stat:.3f}"

    print("selftest OK")
    print(f"  random walk ADF        = {rw_stat:+.3f}  (not < {ADF_CRIT_CONST['10%']}, correct)")
    print(f"  AR(1) phi=0.2 ADF      = {ar_stat:+.3f}  (< {ADF_CRIT_CONST['1%']}, correct)")
    print(f"  cointegrated pair ADF  = {coint.adf_stat:+.3f}  (< {EG_CRIT_N2['1%']}, correct); "
          f"half-life {coint.half_life:.1f}d")
    print(f"  independent walks ADF  = {ind.adf_stat:+.3f}  (not < {EG_CRIT_N2['10%']}, correct)")


# Chan's GLD/GDX cadf example, reconstructed from his own MATLAB source
# (example7_2.m, example3_6_1.m) and companion GLD.xls/GDX.xls. There are two
# runs on two windows, both through-origin (his ols/cadf add no intercept):
#   Ch.7 (example7_2.m): FULL GLD-GDX intersection 2006-05-23..2007-11-30.
#     Reported hedge 1.6766, cadf t=-3.357, "better than 95%".
#   Ch.3 (example3_6_1.m, book p.63): drops the last 60 days, then cadf on the
#     first 252 days = 2006-05-23..~2007-05-23. Reported t=-3.18, "better than
#     90%" -- the numbers cited on page 63.
# Chan used the ADJUSTED close, but the exact 1.6766 is a lost data vintage:
# Yahoo re-scales adjusted close by every later dividend, so his 2007-era
# series differs from any modern download. Independent reproductions (aushaff,
# our own) converge on hedge ~1.638. Raw close is the closest modern proxy to
# his 2007-vintage adjusted (GDX had barely any dividends stripped back then),
# so --book uses raw + origin over the full window.
BOOK_START = "2006-05-23"
BOOK_END = "2007-11-30"
BOOK_TRAIN_END = "2007-05-23"  # first ~252 trading days -> Ch.3 / p.63 run
BOOK_REF_FULL = "example7_2.m (Ch.7): hedge 1.6766, cadf t=-3.357, ~95%"
BOOK_REF_TRAIN = "example3_6_1.m (Ch.3, p.63): cadf t=-3.18, ~90%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily cointegration test for a pair of tickers")
    parser.add_argument("--a", default="GLD", help="dependent-leg ticker (default: GLD)")
    parser.add_argument("--b", default="GDX", help="independent-leg ticker (default: GDX)")
    parser.add_argument("--lags", type=int, default=1, help="ADF augmenting lags (default: 1, as in the book)")
    parser.add_argument("--start", default=None, help="window start YYYY-MM-DD (default: full history)")
    parser.add_argument("--end", default=None, help="window end YYYY-MM-DD (default: full history)")
    parser.add_argument("--unadjusted", action="store_true",
                        help="use raw closes instead of dividend-adjusted (the closest modern proxy to the book)")
    parser.add_argument("--origin", action="store_true",
                        help="also report the through-origin hedge (Chan's ols convention; test stays with-intercept)")
    parser.add_argument("--ch7", action="store_true",
                        help=f"reproduce Chan's Chapter 7 full-window run ({BOOK_START}..{BOOK_END}, raw, origin)")
    parser.add_argument("--ch3", action="store_true",
                        help=f"reproduce Chan's Chapter 3 (p.63) training-set run ({BOOK_START}..{BOOK_TRAIN_END}, raw, origin)")
    parser.add_argument("--selftest", action="store_true", help="verify the ADF/coint math on synthetic data")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return

    start, end, unadjusted, origin, reference = (
        args.start, args.end, args.unadjusted, args.origin, None
    )
    if args.ch7:
        start, end, unadjusted, origin, reference = (
            BOOK_START, BOOK_END, True, True, BOOK_REF_FULL
        )
    elif args.ch3:
        start, end, unadjusted, origin, reference = (
            BOOK_START, BOOK_TRAIN_END, True, True, BOOK_REF_TRAIN
        )
    run(args.a, args.b, args.lags, start=start, end=end,
        unadjusted=unadjusted, origin=origin, reference=reference)


if __name__ == "__main__":
    main()
