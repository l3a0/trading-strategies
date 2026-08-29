# pyright: reportPrivateUsage=false
"""Always-run mechanics for common/timeseries.py — the shared OLS / ADF / OU
primitives that search/pair_cointegration.py and factor/factor_mechanism.py both
build on. Hand fixtures are pinned exactly; the synthetic ADF cases assert the
robust threshold property (the verdict the stat implies) rather than a seeded
value.
"""

from __future__ import annotations

import numpy as np
import pytest

from common.timeseries import ADF_CRIT_CONST, adf_tstat, ols, ou_half_life


class TestTimeseriesPrimitives:
    def test_ols_recovers_exact_line(self) -> None:
        """OLS of y = 2x + 1 on design [x, 1] returns [2, 1] with ~0 residual."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0
        fit = ols(y, np.column_stack([x, np.ones(5)]))
        assert fit.beta[0] == pytest.approx(2.0, abs=1e-9)
        assert fit.beta[1] == pytest.approx(1.0, abs=1e-9)
        assert float(np.abs(fit.resid).max()) == pytest.approx(0.0, abs=1e-9)

    def test_ou_half_life_of_known_decay(self) -> None:
        """A z_{t+1} = 0.5 z_t decay has slope -0.5, so half-life = ln(2)/0.5."""
        z = np.zeros(200)
        z[0] = 1.0
        for t in range(1, 200):
            z[t] = 0.5 * z[t - 1]
        assert ou_half_life(z) == pytest.approx(np.log(2.0) / 0.5, abs=1e-4)

    def test_ou_half_life_infinite_when_not_mean_reverting(self) -> None:
        """A pure random walk does not mean-revert -> half-life is huge or +inf."""
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(500))
        assert ou_half_life(walk) > 100.0

    def test_adf_random_walk_not_rejected(self) -> None:
        """A unit-root random walk should NOT reject the no-stationarity null."""
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(2000))
        tstat, nobs = adf_tstat(walk, lags=1, constant=True)
        assert tstat > ADF_CRIT_CONST["10%"]
        assert nobs == 1998

    def test_adf_stationary_ar1_rejected(self) -> None:
        """A stationary AR(1), phi=0.2, rejects the unit-root null hard."""
        rng = np.random.default_rng(2)
        ar = np.zeros(2000)
        for t in range(1, 2000):
            ar[t] = 0.2 * ar[t - 1] + rng.standard_normal()
        tstat, _ = adf_tstat(ar, lags=1, constant=True)
        assert tstat < ADF_CRIT_CONST["1%"]
