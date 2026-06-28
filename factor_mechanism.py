"""factor_mechanism.py — the factor mechanism gate via a loading regression (H1 of docs/integration_plan.md).

An option structure has a greek signature to read; a factor does NOT. Its mechanism is the LOADING
REGRESSION (the conversation's throughline, and the doc's caveat 1): regress the factor's long-short
returns on a panel of REGISTERED PREMIA, and TYPE it by the premium it actually loads on. This is the
factor's `derive_family` — a MEASUREMENT, not a story. A factor that loads significantly on a known
premium gets that family; one that loads on nothing known is mechanism-INCOHERENT (`None`), the
fail-closed verdict the gate will turn into `measurement_invalid` (the foil-paper defense).

H1a (this module) is the mechanism COMPUTATION, additive and standalone (it takes a factor signal +
the panel, never imports the backends). H1b wires it into `FactorBackend.mechanism` /
`GrammarFactorBackend.mechanism` and the score gate, turning today's `family=None` into a derived family.

DEPENDENCY-LIGHT: a plain OLS by numpy linear algebra (`(X'X)^-1 X'y` + residual standard errors), no
scipy/statsmodels — the loading t-stat is all the gate needs, and a normal-tailed |t| hurdle matches the
repo's `_asymptotic_p` convention.

WHY ONLY `trend` + `lowvol`. They are the base styles a PRICE panel can build; the other canonical premia
need data the panel does not carry — value / quality / investment need fundamentals (book, earnings,
balance sheets), size needs share count, carry needs yields, and the variance risk premium needs options
(this repo's *other* domain). So the set is the dependency-light FLOOR, not a claim that only two premia
matter. It is also a COMMITTED taxonomy — the factor analog of the option grammar's `PremiumFamily` — and
must stay PRE-committed: registering a premium *after* seeing which factor you want to pass would game the
gate (a premium set that grows to fit a favoured factor is the foil-paper move the gate exists to stop).
A widening is therefore a human-signed governance act, like an option grammar widening, sourced when a
fundamentals-bearing equity panel arrives. The price of the small set is paid in the SAFE direction: a
factor whose true premium is not registered loads on nothing and fails closed (`None`) — a false negative,
but a foil-paper defense should reject a coherent factor before admitting an incoherent one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGISTERED_PREMIA: tuple[str, ...] = ('trend', 'lowvol')   # the committed factor families (base styles)
PREMIUM_WINDOW = 20                                          # lookback for the base signals (the momentum window)
# |t| a loading must clear to TYPE the factor — the exposure-typing threshold (does it harvest a known
# premium). NOT the factor's discovery bar: that is its IC t-stat under e-LOND (the doc's HLZ t>3), a
# separate, downstream gate. So this stays the conventional t>2 for a real exposure, not the t>3 hurdle.
LOADING_HURDLE_T = 2.0
MIN_LOADING_OBS = 30                                         # below this, the regression is data-insufficient
PORTFOLIO_Q = 0.3                                            # long/short the top/bottom fraction by signal


def long_short_returns(signal: pd.DataFrame, prices: pd.DataFrame, fwd: int = 1,
                       q: float = PORTFOLIO_Q) -> pd.Series:
    """The daily long-short return of a portfolio formed on `signal`: each date, long the top-`q` fraction
    of names by signal and short the bottom-`q`, realized over the next `fwd` period. LOOK-AHEAD-FREE —
    the position is formed on date t's signal and earns t->t+fwd's return (`shift(-fwd)`)."""
    fwd_ret = prices.shift(-fwd) / prices - 1.0
    out: dict = {}
    for date in signal.index:
        s = signal.loc[date].dropna()
        if len(s) < 5:                                       # need a meaningful cross-section to sort
            continue
        k = max(1, int(len(s) * q))
        order = s.sort_values()
        r = fwd_ret.loc[date]
        long_r, short_r = r[order.index[-k:]].mean(), r[order.index[:k]].mean()
        if pd.notna(long_r) and pd.notna(short_r):
            out[date] = float(long_r - short_r)
    return pd.Series(out, dtype=float)


def registered_premia(prices: pd.DataFrame, window: int = PREMIUM_WINDOW) -> pd.DataFrame:
    """The REGISTERED PREMIA: the daily long-short returns of the base-style factors a price panel can
    build — `trend` (sort by trailing return) and `lowvol` (sort by negative realized vol). The committed
    family set the loading regression types against (`REGISTERED_PREMIA`)."""
    ret = prices.pct_change()
    trend = prices / prices.shift(window) - 1.0
    lowvol = -ret.rolling(window).std()
    return pd.DataFrame({'trend': long_short_returns(trend, prices),
                         'lowvol': long_short_returns(lowvol, prices)})


def _ols_tstats(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Plain OLS t-stats for `y ~ [1, x]` by numpy linear algebra (no scipy): the column t-stats
    `beta / se(beta)` with `se` from the residual variance and `(X'X)^-1`. Returns the t-stat vector
    (index 0 the intercept)."""
    design = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ (design.T @ y)
    resid = y - design @ beta
    dof = len(y) - design.shape[1]
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * xtx_inv))
    return beta / se


def loading_family(signal: pd.DataFrame, prices: pd.DataFrame, premia: pd.DataFrame | None = None,
                   hurdle: float = LOADING_HURDLE_T) -> str | None:
    """Type a factor by a LOADING REGRESSION — the factor's `derive_family`. Form the factor's long-short
    returns, regress them on the registered premia, and return the premium with the largest loading t-stat
    that clears `hurdle` (the factor's economic family). Returns `None` for a mechanism-INCOHERENT factor
    that loads on no registered premium — the fail-closed verdict (the foil-paper defense). `None` also for
    data-insufficiency, or for collinear premia (a singular design the t-stats can't be read from).

    Typing is SIGN-AGNOSTIC (dominant by `|t|`): a factor that loads negatively on `trend` is still typed
    `trend` — the direction rides the regression coefficient, not the family label. A factor that IS a
    registered premium (e.g. momentum == the trend signal) loads perfectly, so its `|t|` is enormous and
    it types as that premium — correct; whether it is a *novel* edge vs. a repackaging of the premium is a
    separate orthogonalized-alpha question, not the mechanism gate's job."""
    prem = registered_premia(prices) if premia is None else premia
    pair = pd.concat([long_short_returns(signal, prices).rename('_factor'), prem], axis=1).dropna()
    if len(pair) < MIN_LOADING_OBS:
        return None
    cols = list(prem.columns)
    try:
        t = _ols_tstats(pair['_factor'].to_numpy(), pair[cols].to_numpy())
    except np.linalg.LinAlgError:
        return None                                                 # collinear premia -> can't type
    loadings = {c: float(t[i + 1]) for i, c in enumerate(cols)}      # skip the intercept t-stat
    dominant = max(loadings, key=lambda c: abs(loadings[c]))         # dominant exposure (sign-agnostic)
    return dominant if abs(loadings[dominant]) > hurdle else None
