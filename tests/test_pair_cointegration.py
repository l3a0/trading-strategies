# pyright: reportPrivateUsage=false
"""Regression tests for search/pair_cointegration.py.

Two always-run layers (no dataset gate — the GLD/GDX price CSVs are committed
to git, not release-sized option chains). The shared OLS / ADF / OU primitives
(now statsmodels-backed) have their own mechanics tests in
``tests/test_timeseries.py``; this file tests the pair-specific two-step
``engle_granger`` and the reproduction.

1. ``TestEngleGranger`` — the two-step cointegration test on synthetic pairs
   with known answers (a shared-factor pair rejects, independent walks do not),
   plus the module's own ``selftest``.
2. ``TestGldGdxReproduction`` — pins the reproduced GLD/GDX numbers from Chan's
   *Quantitative Trading* (Ch.7 full window, Ch.3 / p.63 training set, and the
   full-history decay). These are the numbers the reproduction essay quotes. The
   test is what freezes the yfinance vintage: a future re-download that shifts
   the adjusted-close basis moves these numbers and fails CI.
The reproduced hedge is 1.6379 / 1.6283, not Chan's printed 1.6766 — the exact
book value is a lost data vintage. 1.6766 is a cited book target, never asserted
as a computed result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common.timeseries import EG_CRIT_N2
from search.pair_cointegration import (
    BOOK_END,
    BOOK_START,
    BOOK_TRAIN_END,
    CointResult,
    RollingCoint,
    aligned_closes,
    engle_granger,
    rolling_cointegration,
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
# Layer 3 — the rolling-window regime scan (the essay's regime map)
# ============================================================
class TestRollingRegime:
    """Freeze the rolling-window scan behind the essay's regime map
    (docs/figures/reproduction_regime_map.png, drawn by
    search/make_regime_figure.py). Always-run: the raw GLD/GDX CSVs are
    committed. These are the numbers section 6 of the reproduction essay quotes;
    a re-download that shifts the vintage moves the map and these pins together.

    A one-year window (252 trading days) stepped monthly (21 days) over the full
    as-traded history, ``origin=True`` so Chan's through-origin hedge rides along.
    """

    @pytest.fixture(scope="class")
    def scan(self) -> tuple[RollingCoint, pd.DatetimeIndex]:
        df = aligned_closes("GLD", "GDX", unadjusted=True)
        a = df["GLD"].to_numpy(dtype=float)
        b = df["GDX"].to_numpy(dtype=float)
        return rolling_cointegration(a, b, window=252, step=21, lags=1), df.index

    def test_cointegration_is_episodic(self, scan: tuple[RollingCoint, pd.DatetimeIndex]) -> None:
        """Only ~1 window in 8 clears even the 10% bar — the map's headline."""
        roll, _ = scan
        adf = roll.adf_stat
        assert len(adf) == 231
        assert int((adf < EG_CRIT_N2["10%"]).sum()) == 31
        assert int((adf < EG_CRIT_N2["5%"]).sum()) == 14

    def test_first_window_reproduces_chans_era(
        self, scan: tuple[RollingCoint, pd.DatetimeIndex]
    ) -> None:
        """The first rolling window ≈ Chan's Ch.3 training set: it rejects, with
        a through-origin hedge near his ~1.64 (hedge at 5 sig figs)."""
        roll, _ = scan
        assert roll.adf_stat[0] == pytest.approx(-3.18, abs=1e-2)
        assert roll.origin_hedge[0] == pytest.approx(1.6286, abs=5e-4)
        assert roll.adf_stat[0] < EG_CRIT_N2["10%"]

    def test_hedge_drifts_far_past_the_book(
        self, scan: tuple[RollingCoint, pd.DatetimeIndex]
    ) -> None:
        """Chan's hedge does not hold: the through-origin ratio peaks past 6."""
        roll, _ = scan
        assert roll.origin_hedge.max() == pytest.approx(6.6152, abs=5e-4)

    def test_cointegration_fades_after_the_early_years(
        self, scan: tuple[RollingCoint, pd.DatetimeIndex]
    ) -> None:
        """The early cluster is the whole story: in the decade from 2015 only
        10 of 139 windows reject."""
        roll, idx = scan
        years = idx[roll.end_idx].year.to_numpy()
        post = years >= 2015
        assert int(post.sum()) == 139
        assert int((roll.adf_stat[post] < EG_CRIT_N2["10%"]).sum()) == 10
