"""Always-run tests for the e-value FDR control (interlock #3b, evalue_fdr.py).

Pins the calibrator, the e-LOND recurrence, and e-BH against ORACLE values captured
from the `online-fdr` package (its GitHub-main e-value module — ELond, e_bh,
p_to_e_power — which parity-tests itself against the R/Bioconductor onlineFDR). The
oracle values are HARDCODED here so the repo depends on `online-fdr` for nothing at
runtime (port, don't depend). `TestOnlineFdrParity` re-derives them live IF
`online-fdr` is installed (Python 3.10+), and skips otherwise.
"""
from __future__ import annotations

import math

import pytest

from evalue_fdr import (
    CALIBRATOR_KAPPA,
    ELOND_GAMMA_C,
    ONLINE_FDR_ALPHA,
    _assert_calibrator_admissible,
    _calibrator_integral,
    calibrate_p_to_e,
    e_bh,
    elond,
    online_fdr_survivors,
    registered_gamma,
)


# online-fdr's DefaultLondGamma (Javanmard-Montanari), with its alpha factored OUT
# to match this module's alpha_t = alpha * gamma_t * (R+1) convention. Used only to
# reproduce the captured online-fdr e-LOND oracle (the registered gamma is different
# and is pinned separately in TestRegisteredGamma).
_ORACLE_LOND_C = 0.07720838


def _oracle_gamma(t: int) -> float:
    return _ORACLE_LOND_C * math.log(max(t, 2)) / (t * math.exp(math.sqrt(math.log(t))))


class TestCalibrator:
    """The Vovk-Wang p-to-e calibrator e = kappa*p^(kappa-1), kappa=0.5 -> 1/(2*sqrt p)."""

    def test_matches_vovk_wang_oracle(self) -> None:
        # ORACLE: online-fdr p_to_e_power(p, 0.5) — captured values
        for p, e in [(0.5, 0.7071067811865476), (0.25, 1.0), (0.01, 5.0), (0.0001, 50.0)]:
            assert calibrate_p_to_e(p) == pytest.approx(e, rel=1e-12)
            assert calibrate_p_to_e(p) == pytest.approx(1.0 / (2 * math.sqrt(p)), rel=1e-12)

    def test_invalid_calibrates_to_zero(self) -> None:
        # measurement_invalid cell -> p is None -> e=0 (counts toward n, never rejectable)
        assert calibrate_p_to_e(None) == 0.0

    def test_e_exceeds_one_iff_p_below_quarter(self) -> None:
        assert calibrate_p_to_e(0.24) > 1.0
        assert calibrate_p_to_e(0.25) == pytest.approx(1.0)
        assert calibrate_p_to_e(0.26) < 1.0

    def test_admissible_integral_le_one(self) -> None:
        assert _calibrator_integral(0.5) <= 1.0 + 1e-3       # integral f <= 1
        _assert_calibrator_admissible(0.5)                   # does not raise
        for bad in (0.0, 1.0, 1.5, -0.5):
            with pytest.raises(ValueError):
                _assert_calibrator_admissible(bad)
            with pytest.raises(ValueError):
                calibrate_p_to_e(0.1, kappa=bad)

    def test_clamps_extreme_p(self) -> None:
        assert math.isfinite(calibrate_p_to_e(0.0))          # clamped, not inf
        assert calibrate_p_to_e(1.0) == pytest.approx(0.5)   # boundary p=1 -> kappa


class TestRegisteredGamma:
    """The committed e-LOND discount sequence gamma_t ∝ 1/(t log^2(t+1)), Sum <= 1."""

    def test_normalizer_and_first_terms_pinned(self) -> None:
        assert ELOND_GAMMA_C == pytest.approx(0.2951824238711732, abs=1e-12)
        assert registered_gamma(1) == pytest.approx(0.6143835407835092, abs=1e-12)
        assert registered_gamma(2) == pytest.approx(0.12228455115137625, abs=1e-12)

    def test_sum_over_infinite_stream_below_one(self) -> None:
        # The e-LOND requirement is Sum over ALL t>=1, not a finite prefix. Exercise the
        # INFINITE-stream bound: the H-term prefix plus the proven conservative tail
        # Sum_{t>H} gamma_t <= ELOND_GAMMA_C / log(H) (raw(t) <= 1/(t log^2 t),
        # integral_H^inf = 1/log(H)). If THIS <= 1, the whole infinite sum is < 1.
        h = 100_000   # the normalizer's horizon
        prefix = sum(registered_gamma(t) for t in range(1, h + 1))
        tail_bound = ELOND_GAMMA_C / math.log(h)
        assert prefix + tail_bound <= 1.0          # the actual infinite-stream guarantee

    def test_non_increasing(self) -> None:
        prev = registered_gamma(1)
        for t in range(2, 50):
            g = registered_gamma(t)
            assert g < prev
            prev = g


class TestElondRecurrence:
    """e-LOND alpha_t = alpha*gamma_t*(R_{t-1}+1), reject iff e_t >= 1/alpha_t —
    pinned against the captured online-fdr ELond oracle (using online-fdr's gamma)."""

    def test_no_rejection_stream(self) -> None:
        # ORACLE: ELond(0.10, DefaultLondGamma(0.07720838)) on [5,0.5,30,2,0]
        steps = elond([5.0, 0.5, 30.0, 2.0, 0.0], 0.10, _oracle_gamma)
        levels = [0.0053516771, 0.0011638206, 0.00099124988, 0.00082436061, 0.00069888697]
        assert [s.level for s in steps] == pytest.approx(levels, rel=1e-7)
        assert [s.rejected for s in steps] == [False] * 5    # even e=30 < 1/level≈1009

    def test_rejection_and_discovery_counting(self) -> None:
        # ORACLE: same procedure on [200,500,0.5] — t1 & t2 reject; (R+1) raises levels
        steps = elond([200.0, 500.0, 0.5], 0.10, _oracle_gamma)
        assert [s.level for s in steps] == pytest.approx(
            [0.005351677091, 0.002327641157, 0.002973749638], rel=1e-7)
        assert [s.rejected for s in steps] == [True, True, False]
        # level_2 doubled relative to its R=0 value (one prior discovery): (R+1)=2
        assert steps[1].level == pytest.approx(2 * 0.10 * _oracle_gamma(2), rel=1e-9)

    def test_zero_e_value_never_rejected(self) -> None:
        # a measurement_invalid cell (e=0) can never clear any threshold
        steps = elond([0.0, 1e12], 0.10, _oracle_gamma)
        assert steps[0].rejected is False

    def test_registered_gamma_is_the_default(self) -> None:
        # the live default uses the REGISTERED gamma (not the oracle's)
        default = elond([1e9, 0.1], 0.10)
        assert default[0].level == pytest.approx(0.10 * registered_gamma(1), rel=1e-12)


class TestEBH:
    """e-BH diagnostic: reject top k* = largest k with e_(k) >= n/(k*alpha)."""

    def test_reject_top_one(self) -> None:
        assert e_bh([60.0, 5.0, 2.0, 0.5, 0.0], 0.10) == [True, False, False, False, False]

    def test_reject_top_two(self) -> None:
        assert e_bh([300.0, 120.0, 2.0, 0.5, 0.0], 0.10) == [True, True, False, False, False]

    def test_no_rejection(self) -> None:
        assert e_bh([5.0, 0.5, 30.0, 2.0, 0.0], 0.10) == [False] * 5

    def test_empty(self) -> None:
        assert e_bh([], 0.10) == []


class TestLedgerRunner:
    """online_fdr_survivors over a synthetic ledger stream: calibrate -> e-LOND flag."""

    def test_calibrates_and_flags(self) -> None:
        rows = [
            {'template': 'a', 'ticker': 'MSFT', 'p_value': 1e-5},          # tiny p -> big e
            {'template': 'b', 'ticker': 'SPY', 'p_value': 0.5},            # weak -> small e
            {'template': 'c', 'ticker': 'XLE', 'p_value': None,
             'measurement_invalid': True},                                 # invalid -> e=0
        ]
        out = online_fdr_survivors(rows, ONLINE_FDR_ALPHA, CALIBRATOR_KAPPA)
        assert out[0]['e_value'] == pytest.approx(1.0 / (2 * math.sqrt(1e-5)), rel=1e-9)
        # registered gamma_1: a cell at t=1 needs e >= 1/(alpha*gamma_1) ≈ 16.3; e≈158 -> flagged
        assert out[0]['elond_survivor'] is True
        assert out[1]['elond_survivor'] is False
        assert out[2]['e_value'] == 0.0 and out[2]['elond_survivor'] is False
        assert all('elond_level' in r for r in out)

    def test_empty_ledger(self) -> None:
        assert online_fdr_survivors([]) == []


class TestOnlineFdrParity:
    """Optional LIVE parity against the online-fdr package (skips unless installed,
    Python 3.10+). The hardcoded oracle values above came from exactly this."""

    def test_live_parity(self) -> None:
        try:   # skip unless online-fdr's e-value module is importable (GitHub-main, 3.10+)
            from online_fdr.e_values.batch import e_bh as ref_e_bh  # type: ignore[import]
            from online_fdr.e_values.sequential import (  # type: ignore[import]
                DefaultLondGammaSequence, ELond)
            from online_fdr.e_values.toolbox import p_to_e_power  # type: ignore[import]
        except Exception as exc:  # not installed / wrong version / wrong Python / no e_values
            pytest.skip(f'online-fdr e-value module unavailable: {exc}')

        for p in (0.5, 0.25, 0.01, 0.0001):
            assert calibrate_p_to_e(p) == pytest.approx(p_to_e_power(p, 0.5), rel=1e-12)
        for evs in ([60.0, 5.0, 2.0, 0.5, 0.0], [300.0, 120.0, 2.0, 0.5, 0.0]):
            assert e_bh(evs, 0.10) == ref_e_bh(evs, 0.10)
        ref = ELond(alpha=0.10, gamma_seq=DefaultLondGammaSequence(c=_ORACLE_LOND_C))
        mine = elond([200.0, 500.0, 0.5], 0.10, _oracle_gamma)
        for e, m in zip([200.0, 500.0, 0.5], mine):
            d = ref.test_one_detail(e)
            assert m.rejected == d.rejected
            assert m.level == pytest.approx(d.test_level, rel=1e-9)
