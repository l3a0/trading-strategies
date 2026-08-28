"""Shared time-series / regression primitives (the leaf statistics toolkit).

``common/`` is the leaf everything else imports, so a primitive that more than
one package needs lives here rather than in any one of them. These are the
reusable estimators behind the pair-cointegration reproduction and the factor
mechanism gate:

- ``ols`` — ordinary least squares (coefficients, their standard errors, and
  residuals). The single least-squares definition; ``search/pair_cointegration``
  and ``factor/factor_mechanism`` both build on it rather than each rolling their
  own normal-equations copy.
- ``adf_tstat`` — the Augmented Dickey-Fuller unit-root t-statistic.
- ``ou_half_life`` — the Ornstein-Uhlenbeck mean-reversion half-life.
- ``ADF_CRIT_CONST`` / ``EG_CRIT_N2`` — MacKinnon (2010) asymptotic critical
  values for the plain ADF test and the Engle-Granger residual test (N=2).

Leaf discipline: this module imports nothing above ``common/`` (numpy + stdlib
only), so both ``search/`` and ``factor/`` can share it without a dependency
inversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

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
    intercept -- add a column of ones yourself if you want one).

    ``beta`` comes from ``np.linalg.lstsq`` (SVD-based, numerically stabler than
    the normal equations); the standard errors use the residual variance and
    ``(X'X)^-1`` in the textbook form.
    """
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

    fit = ols(response, design)
    tstat = float(fit.beta[0] / fit.se[0])  # coefficient on y_{t-1}
    return tstat, len(response)


def ou_half_life(spread: NDArray[np.float64]) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life, in the series' own time
    step (trading days here). Regress ``d_z_t`` on ``z_{t-1}``; the slope is
    ``-theta``; half-life is ``ln(2)/theta``. Returns ``+inf`` when the spread
    does not mean-revert (non-negative slope)."""
    z = np.asarray(spread, dtype=float)
    dz = np.diff(z)
    zlag = z[:-1]
    design = np.column_stack([zlag, np.ones(len(zlag))])
    fit = ols(dz, design)
    slope = float(fit.beta[0])  # == -theta
    if slope >= 0:
        return math.inf
    return math.log(2.0) / (-slope)
