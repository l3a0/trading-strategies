# pyright: reportPrivateUsage=false
"""Regression tests for search/pair_cointegration.py.

Always-run layers (no dataset gate — the price CSVs are committed to git, not
release-sized option chains). The shared OLS / ADF / OU primitives (now
statsmodels-backed) have their own mechanics tests in
``tests/test_timeseries.py``; this file tests the pair-specific two-step
``engle_granger``, the return correlation, and the reproductions.

1. ``TestEngleGranger`` — the two-step cointegration test on synthetic pairs
   with known answers (a shared-factor pair rejects, independent walks do not),
   plus the module's own ``selftest``.
2. ``TestGldGdxReproduction`` — pins the reproduced GLD/GDX numbers from Chan's
   *Quantitative Trading* (Ch.7 full window, Ch.3 / p.63 training set, and the
   full-history decay). These are the numbers the reproduction essay quotes. The
   test is what freezes the yfinance vintage: a future re-download that shifts
   the adjusted-close basis moves these numbers and fails CI.
3. ``TestRollingRegime`` — the rolling-window scan behind the essay's regime map.
4. ``TestKoPepNonCointegration`` — pins Chan's KO/PEP counter-example
   (example7_3.m): correlated in returns yet not cointegrated in levels. Unlike
   GLD/GDX this runs on Chan's OWN committed companion data, so it reproduces his
   printed figures exactly (origin hedge 1.0114, cadf t -2.14, corr 0.4849).
5. ``TestGldGdxChanArchive`` — Chan's own committed GLD/GDX data (gld_chan.csv /
   gdx_chan.csv). Pins that it gives 1.6395 / -3.52, NOT his printed 1.6766: the
   lost-vintage receipt. The verdict still holds; only the hedge drifted.

The GLD/GDX hedge reproduces at 1.6379 / 1.6283, not Chan's printed 1.6766 — the
exact book value is a lost data vintage (even Chan's own archived GLD.xls, re-run,
lands at 1.6395, pinned by layer 5). 1.6766 is a cited book target, never asserted
as a computed result. KO/PEP is the opposite case: Chan's committed data
reproduces exactly.
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
    return_correlation,
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


# ============================================================
# Layer 4 — Chan's KO/PEP counter-example (committed Chan data)
# ============================================================
class TestKoPepNonCointegration:
    """Freeze Chan's KO/PEP counter-example (example7_3.m, Ch.7): the pair is
    significantly CORRELATED in daily returns yet does NOT cointegrate — the
    demonstration that correlation and cointegration are different things.

    Always-run: data/ko_chan.csv and pep_chan.csv are committed (the
    adjusted-close columns of Chan's own KO.xls/PEP.xls). Unlike GLD/GDX, this
    runs on Chan's exact companion data, so it reproduces his printed figures to
    the digit — the counterpoint to the lost 1.6766 vintage.
    """

    @pytest.fixture(scope="class")
    def kopep(self) -> CointResult:
        """Full KO/PEP intersection, adjusted close — Chan's example7_3.m run."""
        df = aligned_closes("KO", "PEP", chan=True)
        a = df["KO"].to_numpy(dtype=float)
        b = df["PEP"].to_numpy(dtype=float)
        return engle_granger(a, b, lags=1, origin=True)

    @pytest.fixture(scope="class")
    def corr(self) -> tuple[float, float, float]:
        df = aligned_closes("KO", "PEP", chan=True)
        a = df["KO"].to_numpy(dtype=float)
        b = df["PEP"].to_numpy(dtype=float)
        return return_correlation(a, b)

    def test_hedge_matches_chan_exactly(self, kopep: CointResult) -> None:
        """Chan's through-origin hedge 1.0114 reproduces to the digit (his .xls
        data, not a modern download). The with-intercept hedge is 0.9209."""
        assert kopep.nobs == 7833
        assert kopep.origin_hedge == pytest.approx(1.0114, abs=5e-4)
        assert kopep.hedge_ratio == pytest.approx(0.9209, abs=5e-4)

    def test_fails_to_cointegrate(self, kopep: CointResult) -> None:
        """CADF t = -2.14 (Chan's -2.14258438), well above the 10% crit: the
        pair fails to reject the no-cointegration null. The OU 'half-life' of
        ~619 days confirms the spread barely mean-reverts."""
        assert kopep.adf_stat == pytest.approx(-2.14, abs=1e-2)
        assert kopep.adf_stat > EG_CRIT_N2["10%"]
        assert kopep.half_life > 100

    def test_returns_are_correlated(self, corr: tuple[float, float, float]) -> None:
        """The other half of the counter-example: daily returns ARE
        significantly correlated (Chan's r = 0.4849), even though the prices do
        not cointegrate."""
        r, _t, p = corr
        assert r == pytest.approx(0.4849, abs=5e-4)
        assert p < 0.05


# ============================================================
# Layer 5 — Chan's own GLD/GDX archive (the lost-vintage receipt)
# ============================================================
class TestGldGdxChanArchive:
    """Freeze what Chan's OWN archived GLD.xls/GDX.xls produce, and pin the fact
    that they do NOT reproduce his printed 1.6766.

    Always-run: data/gld_chan.csv and gdx_chan.csv are committed (the
    adjusted-close columns of Chan's own companion .xls, egorpe mirror, last
    saved by Ernest Chan 2007-12-02). The essay calls the 1.6766 hedge a lost
    data vintage; this is the receipt. Even Chan's own saved files, re-run, land
    at 1.6395 — essentially the yfinance 1.6379, not the book. The 2007 book-run
    vintage is a still-earlier state that no surviving file carries. The
    cointegration verdict holds (it still rejects at 5%); only the hedge drifted.
    """

    @pytest.fixture(scope="class")
    def arch(self) -> CointResult:
        """Full GLD/GDX intersection from Chan's committed .xls, adjusted close."""
        df = aligned_closes("GLD", "GDX", chan=True)
        a = df["GLD"].to_numpy(dtype=float)
        b = df["GDX"].to_numpy(dtype=float)
        return engle_granger(a, b, lags=1, origin=True)

    def test_reproduces_chans_archive(self, arch: CointResult) -> None:
        """Chan's own data gives 1.6395 / -3.52 over 2006-05-23..2007-11-30."""
        assert arch.nobs == 383
        assert arch.origin_hedge == pytest.approx(1.6395, abs=5e-4)
        assert arch.hedge_ratio == pytest.approx(1.3865, abs=5e-4)
        assert arch.adf_stat == pytest.approx(-3.52, abs=1e-2)
        assert arch.half_life == pytest.approx(10.3, abs=0.1)

    def test_hedge_is_not_the_lost_book_vintage(self, arch: CointResult) -> None:
        """The whole point of committing this archive: even Chan's own saved
        files miss his printed 1.6766. The 2007 book-run vintage is gone."""
        assert arch.origin_hedge is not None
        assert abs(arch.origin_hedge - 1.6766) > 0.03

    def test_verdict_survives_the_drift(self, arch: CointResult) -> None:
        """The vintage moved the hedge but not the conclusion: the pair still
        cointegrates at the 5% level, the same verdict --ch7 reaches on yfinance."""
        assert arch.adf_stat < EG_CRIT_N2["5%"]
