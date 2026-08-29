"""Shared time-series / regression primitives (the leaf statistics toolkit).

``common/`` is the leaf everything else imports, so a primitive that more than
one package needs lives here rather than in any one of them. These are the
reusable estimators behind the pair-cointegration reproduction and the factor
mechanism gate, backed by ``statsmodels``:

- ``ols`` — ordinary least squares (coefficients, their standard errors, and
  residuals), via ``statsmodels.api.OLS``. The single least-squares definition;
  ``search/pair_cointegration`` and ``factor/factor_mechanism`` both build on it.
- ``adf_tstat`` — the Augmented Dickey-Fuller unit-root t-statistic, via
  ``statsmodels.tsa.stattools.adfuller`` at a FIXED lag.
- ``ou_half_life`` — the Ornstein-Uhlenbeck mean-reversion half-life (an AR(1)
  regression through ``ols``).
- ``ADF_CRIT_CONST`` / ``EG_CRIT_N2`` — MacKinnon (2010) asymptotic critical
  values for the plain ADF test and the Engle-Granger residual test (N=2).

Leaf discipline: this module imports nothing above ``common/`` (statsmodels,
numpy, stdlib), so both ``search/`` and ``factor/`` can share it without a
dependency inversion.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray
from statsmodels.tsa.stattools import adfuller

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


def ols(y: NDArray[np.float64], x: NDArray[np.float64]) -> OLSFit:
    """Ordinary least squares of ``y`` on the columns of ``x`` (no implicit
    intercept -- add a column of ones yourself if you want one). Backed by
    ``statsmodels.api.OLS``; returns its coefficients, their standard errors,
    and the residuals."""
    res = sm.OLS(np.asarray(y, dtype=float), np.asarray(x, dtype=float)).fit()
    return OLSFit(
        beta=np.asarray(res.params, dtype=float),
        se=np.asarray(res.bse, dtype=float),
        resid=np.asarray(res.resid, dtype=float),
    )


def adf_tstat(series: NDArray[np.float64], lags: int = 1, *, constant: bool = True) -> tuple[float, int]:
    """Augmented Dickey-Fuller t-statistic on the lagged-level coefficient, via
    ``statsmodels.tsa.stattools.adfuller`` at a FIXED lag (``maxlag=lags,
    autolag=None``). The fixed lag is deliberate: statsmodels' ``autolag='aic'``
    default would pick the lag count from the data and return a different
    statistic (the reproduction essay's whole point).

    Returns ``(tstat, nobs)``. A large-magnitude *negative* ``tstat`` is
    evidence against a unit root (i.e. for stationarity / mean reversion).
    Pass ``constant=False`` for regression residuals, which are mean-zero by
    construction and so take no deterministic term.
    """
    with warnings.catch_warnings():
        # adfuller warns that its return type will change in a future release;
        # we read the tuple's stat and nobs, so silence the FutureWarning.
        warnings.simplefilter("ignore", FutureWarning)
        result = adfuller(
            np.asarray(series, dtype=float),
            maxlag=lags,
            autolag=None,  # type: ignore[arg-type]  # statsmodels stub over-narrows autolag to str
            regression="c" if constant else "n",
        )
    return float(result[0]), int(result[3])  # type: ignore[arg-type]  # adfuller's tuple is untyped


def ou_half_life(spread: NDArray[np.float64]) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life, in the series' own time
    step (trading days here). Regress ``d_z_t`` on ``z_{t-1}`` (through the
    shared ``ols``); the slope is ``-theta``; half-life is ``ln(2)/theta``.
    Returns ``+inf`` when the spread does not mean-revert (non-negative slope)."""
    z = np.asarray(spread, dtype=float)
    dz = np.diff(z)
    zlag = z[:-1]
    design = np.column_stack([zlag, np.ones(len(zlag))])
    fit = ols(dz, design)
    slope = float(fit.beta[0])  # == -theta
    if slope >= 0:
        return math.inf
    return math.log(2.0) / (-slope)
