"""
Tests for validate_dailies.py — the option-chain hygiene validator and CHAIN_CLEAN_START proposer.

Two layers:

* An always-run **synthetic** layer that pins the classifier and the boundary scan on
  hand-built chains and per-day metric sequences (no datasets needed — runs in CI).
* A dataset-gated **calibration** layer that pins the validator against real stores:
  the MSFT 2008-2016 backfill must reproduce the published boundary ``2010-05-10``, and
  the structurally-different new chains (GLD/TLT/XLE/EEM) must read CLEAN (no clip). These
  skip when the stores are absent.

The calibration layer is what makes the automated boundary trustworthy: it proves the scan
reproduces a human-decided clip exactly before anyone leans on it for a novel ticker.
"""
from __future__ import annotations

import os
from common.paths import DATA_DIR
from datetime import date, timedelta

import pytest

import pipeline.validate_dailies as vd
from pipeline.validate_dailies import (
    DayMetrics,
    classify_day,
    propose_boundary,
    _on_lattice,
    _mark_inside,
    _parity_spot,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def row(strike, bid, ask, mark, iv, delta, dte=30, expiration="2010-07-17"):
    return {"expiration": expiration, "dte": dte, "strike": strike,
            "bid": bid, "ask": ask, "mark": mark, "iv": iv, "delta": delta}


def clean_call_chain():
    """A sane modern call chain: smooth delta curve, marks at the midpoint."""
    spot = 100.0
    out = []
    for k in range(90, 121, 2):
        # crude smooth call delta that falls through the band as strike rises
        dlt = max(0.01, min(0.99, 1.0 - (k - 80) / 40.0))
        bid, ask = max(0.05, spot - k) + 0.5, max(0.05, spot - k) + 1.5
        out.append(row(float(k), bid, ask, (bid + ask) / 2, 0.20, dlt))
    return out


def placeholder_call_chain():
    """The 2008-2010 pathology: constant 0.01 mark, lattice IV, step-function delta."""
    out = []
    for k in range(90, 121, 2):
        dlt = 1.0 if k <= 100 else 0.02      # step function: nothing lands in 0.05-0.60
        bid, ask = max(0.0, 100.0 - k) + 0.2, max(0.0, 100.0 - k) + 0.4
        out.append(row(float(k), bid, ask, 0.01, vd.PLACEHOLDER_IV, dlt))
    return out


def both_wing_chain(spot=100.0):
    """Calls and puts so put-call parity can recover the spot."""
    out = []
    for k in range(90, 111, 1):
        cmid = max(0.5, spot - k) + 2.0
        pmid = max(0.5, k - spot) + 2.0
        out.append(row(float(k), cmid - 0.5, cmid + 0.5, cmid, 0.20, 0.5))      # call
        out.append(row(float(k), pmid - 0.5, pmid + 0.5, pmid, 0.20, -0.5))     # put
    return out


def seq(n, start=date(2010, 1, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def clean_dm(d):
    return DayMetrics(day=d, n_rows=50, n_call_inband_raw=5, n_call_inband_clean=5, n_call_inband_defect=0)


def defect_dm(d):
    return DayMetrics(day=d, n_rows=50, n_call_inband_raw=5, n_call_inband_clean=0,
                      n_call_inband_defect=3, lattice_iv_frac=0.6, mark_outside_frac=0.8)


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
class TestClassifier:
    def test_clean_chain_is_usable(self):
        m = classify_day(date(2015, 6, 1), clean_call_chain(), spot=100.0, vol=0.20)
        assert m.n_call_inband_clean >= 1
        assert m.n_call_inband_defect == 0
        assert m.usable and not m.defective

    def test_placeholder_chain_is_defective(self):
        m = classify_day(date(2008, 1, 2), placeholder_call_chain(), spot=None, vol=None)
        # step-function deltas put nothing trustworthy in the band, and the chain is mostly lattice IV
        assert not m.usable
        assert m.defective
        assert m.lattice_iv_frac >= 0.5

    def test_in_band_mark_outside_is_defective(self):
        # one in-band call (delta 0.30) whose mark sits below the bid -> a dangerous placeholder-in-band row
        chain = [row(100.0, 1.00, 1.20, 0.01, 0.20, 0.30)]
        m = classify_day(date(2009, 3, 2), chain, spot=None, vol=None)
        assert m.n_call_inband_defect == 1
        assert m.n_call_inband_clean == 0
        assert m.defective

    def test_lattice_detection(self):
        assert _on_lattice(vd.PLACEHOLDER_IV)
        assert _on_lattice(vd.PLACEHOLDER_IV + 5 * vd.LATTICE_STEP)
        assert not _on_lattice(0.2037)       # an ordinary IV off the lattice
        assert not _on_lattice(float("nan"))

    def test_mark_inside(self):
        assert _mark_inside(1.0, 1.2, 1.1)
        assert _mark_inside(1.0, 1.2, 1.0)            # on the bid
        assert not _mark_inside(10.15, 10.35, 0.01)   # the placeholder mark
        assert _mark_inside(1.2, 1.0, 1.1)            # tolerates bid/ask swapped

    def test_parity_spot_recovers_spot(self):
        spot = _parity_spot(both_wing_chain(spot=100.0))
        assert spot is not None and abs(spot - 100.0) <= 1.0

    def test_parity_spot_none_when_calls_only(self):
        calls_only = [r for r in both_wing_chain() if r["delta"] > 0]
        assert _parity_spot(calls_only) is None

    def test_bs_flags_smoothly_wrong_delta(self):
        # both rows are in-band with marks inside the quote; only the delta differs in plausibility
        ok = [row(105.0, 0.8, 1.0, 0.9, 0.20, 0.22)]          # BS(100,105,~30d,.20,call) ~= 0.22 -> matches -> clean
        wrong = [row(130.0, 0.2, 0.4, 0.3, 0.20, 0.55)]       # deep OTM call claiming delta 0.55 -> BS ~0 -> flagged
        m_ok = classify_day(date(2015, 6, 1), ok, spot=100.0, vol=0.20)
        m_bad = classify_day(date(2015, 6, 1), wrong, spot=100.0, vol=0.20)
        assert m_bad.bs_checked == 1 and m_bad.bs_flagged == 1
        assert m_ok.bs_checked == 1 and m_ok.bs_flagged == 0


# --------------------------------------------------------------------------- #
# boundary scan
# --------------------------------------------------------------------------- #
class TestBoundaryScan:
    def test_dense_early_era_clips_after_cluster(self):
        dates = seq(400)
        days = [defect_dm(d) for d in dates[:100]] + [clean_dm(d) for d in dates[100:]]
        v = propose_boundary("TST", days)
        assert v.status == "CLIP"
        assert v.boundary == dates[100]              # first day past the dense cluster
        assert v.post_defect_frac == 0.0

    def test_all_clean_is_clean(self):
        days = [clean_dm(d) for d in seq(300)]
        v = propose_boundary("TST", days)
        assert v.status == "CLEAN"
        assert v.boundary == days[0].day

    def test_isolated_strays_stay_clean(self):
        dates = seq(400)
        days = [clean_dm(d) for d in dates]
        days[250] = defect_dm(dates[250])            # one stray, head stays clean
        v = propose_boundary("TST", days)
        assert v.status == "CLEAN"
        assert v.boundary == dates[0]                # NOT clipped to the stray
        assert dates[250] in v.stray_defects

    def test_pervasive_strays_unverified(self):
        dates = seq(400)
        days = [clean_dm(d) for d in dates]
        for i in range(80, 400, 12):                 # ~3% scattered, head clean -> pervasive
            days[i] = defect_dm(dates[i])
        v = propose_boundary("TST", days)
        assert v.status == "UNVERIFIED"

    def test_defects_to_the_end_unverified(self):
        dates = seq(400)
        days = [defect_dm(d) for d in dates]          # never resolves
        v = propose_boundary("TST", days)
        assert v.status == "UNVERIFIED"

    def test_era_running_too_late_unverified(self):
        dates = seq(400)
        # dense era covering the first 70% (> MAX_BOUNDARY_FRAC) with no gap
        days = [defect_dm(d) for d in dates[:280]] + [clean_dm(d) for d in dates[280:]]
        v = propose_boundary("TST", days)
        assert v.status == "UNVERIFIED"

    def test_smoothly_wrong_post_clip_unverified(self):
        dates = seq(400)
        days = [defect_dm(d) for d in dates[:100]] + [clean_dm(d) for d in dates[100:]]
        for d in days[100:]:                          # post-clip rows fail the BS reconstruction en masse
            d.bs_checked, d.bs_flagged = 10, 1
        v = propose_boundary("TST", days)
        assert v.status == "UNVERIFIED"
        assert "Black-Scholes" in v.reason


# --------------------------------------------------------------------------- #
# dataset-gated calibration
# --------------------------------------------------------------------------- #
def _resolve(basename):
    """Find a store under data/ as plain .csv (local) or .csv.gz (the CI release cache)."""
    for p in (str(DATA_DIR / basename), str(DATA_DIR / (basename + ".gz"))):
        if os.path.exists(p):
            return p
    return None


MSFT_BACKFILL = _resolve("msft_option_dailies_2008_2016.csv")


@pytest.mark.skipif(MSFT_BACKFILL is None, reason="MSFT 2008-2016 backfill not present")
class TestMsftCalibration:
    """The validator must reproduce the published manual boundary exactly."""

    def test_reproduces_published_boundary(self):
        v, _ = vd.validate("MSFT", MSFT_BACKFILL)
        assert v.status == "CLIP"
        assert str(v.boundary) == "2010-05-10"        # == CHAIN_CLEAN_START['MSFT']
        assert v.post_usable_frac >= 0.99             # one thin (non-defective) day in the tail
        assert v.post_defect_frac == 0.0


@pytest.mark.parametrize("ticker", ["GLD", "TLT", "XLE", "EEM", "QQQ"])
class TestNewChainsClean:
    """The structurally-different new chains start past the placeholder era — no clip
    needed. QQQ joined 2026-07-20 (the wheel plan's §5 commitment): its canonical
    calls store starts 2016-06, past the QQQQ-era pathology, so the CLEAN verdict
    is pinned here rather than asserted in prose."""

    def test_clean_no_clip(self, ticker):
        path = _resolve(f"{ticker.lower()}_option_dailies.csv")
        if path is None:
            pytest.skip(f"{ticker} store not present")
        v, _ = vd.validate(ticker, path)
        assert v.status == "CLEAN", f"{ticker}: {v.reason}"
        assert v.boundary == v.span[0]                # usable from the first day
        assert v.post_defect_frac < vd.STRAY_MAX_FRAC


# --------------------------------------------------------------------------- #
# price-vs-chain scale guard
# --------------------------------------------------------------------------- #
class TestScaleGuard:
    """The second hygiene axis: does the unadjusted price file match the chain's
    as-traded strike scale? (A clean chain with a rescaled price file still blows up
    the delta-hedged overlays — the XLE split case.)"""

    def test_scale_ratio_matched(self):
        atm = {f"d{i}": 100.0 for i in range(10)}
        px = {f"d{i}": 100.0 for i in range(10)}
        assert vd.scale_ratio(atm, px) == pytest.approx(1.0)

    def test_scale_ratio_flags_split_mismatch(self):
        # strikes 2x the price file = the XLE halved-price signature
        atm = {f"d{i}": 100.0 for i in range(10)}
        px = {f"d{i}": 50.0 for i in range(10)}
        r = vd.scale_ratio(atm, px)
        assert r == pytest.approx(2.0)
        assert abs(r - 1.0) > vd.SCALE_TOL            # would be flagged

    def test_scale_ratio_uncheckable(self):
        assert vd.scale_ratio({}, {}) is None
        assert vd.scale_ratio({"a": 1.0}, {"b": 1.0}) is None   # no overlapping days


_XLE_STORE = _resolve("xle_option_dailies.csv")
_HAVE_XLE_SCALE = _XLE_STORE is not None and bool(
    __import__("glob").glob(str(DATA_DIR / "xle_*_unadjusted.csv")))


@pytest.mark.skipif(not _HAVE_XLE_SCALE, reason="needs XLE store + unadjusted price file")
class TestXleScaleRepaired:
    """Regression on the load_unadjusted_prices split fix: XLE's 2:1 split (2025-12-05)
    had halved its price file (ratio ~2.0); after backing the split out, the price file
    matches the chain to ~1.0 and the guard reads OK."""

    def test_price_matches_chain_after_split_fix(self):
        res = vd.check_price_chain_scale("XLE", _XLE_STORE)
        assert res["median_ratio"] is not None
        assert res["median_ratio"] == pytest.approx(1.0, abs=vd.SCALE_TOL)
        assert res["ok"] is True
