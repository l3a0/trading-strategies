# pyright: reportPrivateUsage=false
"""Regression tests for search/pair_cointegration.py.

Three always-run layers (no dataset gate — the GLD/GDX price CSVs are committed
to git, not release-sized option chains). The shared OLS / ADF / OU primitives
have their own mechanics tests in ``tests/test_timeseries.py``; this file tests
the pair-specific two-step ``engle_granger`` and the reproduction.

1. ``TestEngleGranger`` — the two-step cointegration test on synthetic pairs
   with known answers (a shared-factor pair rejects, independent walks do not),
   plus the module's own ``selftest``.
2. ``TestGldGdxReproduction`` — pins the reproduced GLD/GDX numbers from Chan's
   *Quantitative Trading* (Ch.7 full window, Ch.3 / p.63 training set, and the
   full-history decay). These are the numbers the reproduction essay quotes. The
   test is what freezes the yfinance vintage: a future re-download that shifts
   the adjusted-close basis moves these numbers and fails CI.
3. ``TestStatsmodelsCrossCheck`` — an independent second opinion. With the lag
   convention matched (``autolag=None``), statsmodels agrees with the hand-rolled
   ADF to machine precision, which also encodes the essay's lesson: drop
   ``autolag=None`` and statsmodels reverts to its AIC default and diverges.
   Gated behind ``find_spec`` so it runs wherever statsmodels is installed and
   skips cleanly where it isn't (CI installs it in the unit job).

The reproduced hedge is 1.6379 / 1.6283, not Chan's printed 1.6766 — the exact
book value is a lost data vintage. 1.6766 is a cited book target, never asserted
as a computed result.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

from common.timeseries import EG_CRIT_N2
from search.pair_cointegration import (
    BOOK_END,
    BOOK_START,
    BOOK_TRAIN_END,
    CointResult,
    aligned_closes,
    engle_granger,
    selftest,
)


# ============================================================
# Layer 1 — the two-step test on synthetic pairs (engle_granger)
# ============================================================
class TestEngleGranger:
    """The pair-specific two-step cointegration test on synthetic pairs with
    known answers. The underlying OLS / ADF / OU primitives are tested in
    tests/test_timeseries.py; here the assertions are the verdict engle_granger
    reaches.
    """

    def test_cointegrated_pair_detected(self) -> None:
        """A shared-factor pair rejects the no-cointegration null with a short
        half-life."""
        rng = np.random.default_rng(3)
        factor = np.cumsum(rng.standard_normal(2000))
        a = factor + rng.standard_normal(2000) * 0.5
        b = 0.5 * factor + rng.standard_normal(2000) * 0.5
        res = engle_granger(a, b, lags=1)
        assert res.adf_stat < EG_CRIT_N2["1%"]
        assert 0 < res.half_life < 50

    def test_independent_walks_not_cointegrated(self) -> None:
        """Two unrelated random walks fail to reject -> no spurious pair."""
        rng = np.random.default_rng(4)
        a = np.cumsum(rng.standard_normal(2000))
        b = np.cumsum(rng.standard_normal(2000))
        res = engle_granger(a, b, lags=1)
        assert res.adf_stat > EG_CRIT_N2["10%"]

    def test_selftest_runs_clean(self) -> None:
        """The module's own selftest (all four synthetic checks) must not raise."""
        selftest()


# ============================================================
# Layer 2 — the pinned GLD/GDX reproduction (committed CSVs)
# ============================================================
class TestGldGdxReproduction:
    """Freeze the reproduced Chan GLD/GDX numbers the essay quotes.

    Always-run: the GLD/GDX price CSVs are committed to git, so there is no
    dataset gate. Values are pinned at the 4-dp CLI precision, rounded from the
    true value. 1.6766 (Chan's printed hedge) is NOT asserted — it is a cited
    book target the reproduction cannot hit from modern Yahoo data.
    """

    @pytest.fixture(scope="class")
    def ch7(self) -> CointResult:
        """Chapter 7: the full 2006-05-23 .. 2007-11-30 window, raw closes."""
        df = aligned_closes("GLD", "GDX", start=BOOK_START, end=BOOK_END, unadjusted=True)
        a = df["GLD"].to_numpy(dtype=float)
        b = df["GDX"].to_numpy(dtype=float)
        return engle_granger(a, b, lags=1, origin=True)

    @pytest.fixture(scope="class")
    def ch3(self) -> CointResult:
        """Chapter 3 / p.63: the first ~252-day training set, raw closes."""
        df = aligned_closes("GLD", "GDX", start=BOOK_START, end=BOOK_TRAIN_END, unadjusted=True)
        a = df["GLD"].to_numpy(dtype=float)
        b = df["GDX"].to_numpy(dtype=float)
        return engle_granger(a, b, lags=1, origin=True)

    @pytest.fixture(scope="class")
    def full_span(self) -> CointResult:
        """The whole dividend-adjusted history — the decay the essay ends on."""
        df = aligned_closes("GLD", "GDX")
        a = df["GLD"].to_numpy(dtype=float)
        b = df["GDX"].to_numpy(dtype=float)
        return engle_granger(a, b, lags=1)

    def test_ch7_hedge_and_stat(self, ch7: CointResult) -> None:
        # Hedge ratios at 5 significant figures, t-stats at 2 decimals — the
        # repo's sig-fig convention, matching the essay's quoted values.
        assert ch7.nobs == 383
        assert ch7.origin_hedge == pytest.approx(1.6379, abs=5e-4)
        assert ch7.hedge_ratio == pytest.approx(1.3905, abs=5e-4)
        assert ch7.adf_stat == pytest.approx(-3.45, abs=1e-2)
        assert ch7.half_life == pytest.approx(10.6, abs=0.1)
        # Rejects the no-cointegration null at the 5% level on this window.
        assert ch7.adf_stat < EG_CRIT_N2["5%"]

    def test_ch3_hedge_and_stat(self, ch3: CointResult) -> None:
        assert ch3.nobs == 250
        assert ch3.origin_hedge == pytest.approx(1.6283, abs=5e-4)
        assert ch3.hedge_ratio == pytest.approx(1.1911, abs=5e-4)
        assert ch3.adf_stat == pytest.approx(-3.09, abs=1e-2)
        # Rejects at the 10% level on the shorter training set.
        assert ch3.adf_stat < EG_CRIT_N2["10%"]

    def test_full_span_fails_to_reject(self, full_span: CointResult) -> None:
        """Over the full history the pair no longer looks cointegrated — the
        anti-cointegration point the essay lands on."""
        assert full_span.nobs == 5028
        assert full_span.adf_stat == pytest.approx(-1.45, abs=1e-2)
        assert full_span.adf_stat > EG_CRIT_N2["10%"]


# ============================================================
# Layer 3 — independent statsmodels cross-check (enforced in CI)
# ============================================================
@pytest.mark.skipif(
    find_spec("statsmodels") is None,
    reason="statsmodels is an optional cross-check; the CI unit job installs it",
)
class TestStatsmodelsCrossCheck:
    """The hand-rolled ADF must match statsmodels to machine precision when the
    lag convention is matched (``maxlag=1, autolag=None``). Dropping
    ``autolag=None`` is exactly the divergence the essay is about, so this test
    also guards against a future edit reintroducing the AIC default.
    """

    @staticmethod
    def _legs(end: str) -> tuple[np.ndarray, np.ndarray]:
        df = aligned_closes("GLD", "GDX", start=BOOK_START, end=end, unadjusted=True)
        return df["GLD"].to_numpy(dtype=float), df["GDX"].to_numpy(dtype=float)

    @pytest.mark.parametrize("end", [BOOK_END, BOOK_TRAIN_END])
    def test_matches_statsmodels(self, end: str) -> None:
        import warnings

        from statsmodels.tsa.stattools import adfuller, coint

        a, b = self._legs(end)
        ours = engle_granger(a, b, lags=1, origin=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eg_t = coint(a, b, trend="c", maxlag=1, autolag=None)[0]
            adf_t = adfuller(ours.spread, maxlag=1, regression="n", autolag=None)[0]
        assert ours.adf_stat == pytest.approx(float(adf_t), abs=1e-6)
        assert ours.adf_stat == pytest.approx(float(eg_t), abs=1e-6)
