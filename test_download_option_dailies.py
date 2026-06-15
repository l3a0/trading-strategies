"""Pure-function tests for the option-dailies fetcher's chain filter.

Guards the put support added for the put-side VRP experiment
(docs/prereg_vol_premium.md §9): `filter_chain` must keep puts in the strike
band when asked, leave the legacy call-only behaviour byte-identical, and still
infer spot from the calls. No network — these run on a synthetic chain.
"""

from __future__ import annotations

from download_option_dailies import filter_chain, infer_spot


def _row(typ: str, strike: float, delta: float, exp: str = '2020-02-21') -> dict:
    return {'type': typ, 'expiration': exp, 'strike': str(strike),
            'delta': str(delta), 'contractID': f'{typ[0].upper()}{strike:g}'}


# spot inferred from the 0.50-delta call = 100; band ±0.35 -> [65, 135].
CHAIN = [
    _row('call', 100, 0.50), _row('call', 110, 0.25), _row('call', 150, 0.02),
    _row('put', 90, -0.25), _row('put', 70, -0.05), _row('put', 50, -0.01),
]
ASOF = '2020-01-22'


def test_infer_spot_from_calls() -> None:
    assert infer_spot([r for r in CHAIN if r['type'] == 'call'], ASOF) == 100.0


def test_keep_both_returns_calls_and_puts_in_band() -> None:
    ids = {r['contractID'] for r in filter_chain(CHAIN, ASOF, 60, 0.35)}
    assert ids == {'C100', 'C110', 'P90', 'P70'}  # 150 and 50 are out of band


def test_keep_put_returns_in_band_puts_only_with_negative_delta() -> None:
    kept = filter_chain(CHAIN, ASOF, 60, 0.35, 'put')
    assert {r['contractID'] for r in kept} == {'P90', 'P70'}
    assert all(float(r['delta']) < 0 for r in kept)


def test_keep_call_matches_legacy_call_only_behaviour() -> None:
    ids = {r['contractID'] for r in filter_chain(CHAIN, ASOF, 60, 0.35, 'call')}
    assert ids == {'C100', 'C110'}


def test_dte_window_excludes_far_expirations() -> None:
    far = CHAIN + [_row('put', 95, -0.30, exp='2021-01-15')]  # ~360 DTE
    kept = filter_chain(far, ASOF, 60, 0.35, 'put')
    assert 'P95' not in {r['contractID'] for r in kept}  # beyond max_dte=60


def test_no_calls_yields_no_spot_and_empty() -> None:
    assert filter_chain([_row('put', 90, -0.25)], ASOF, 60, 0.35) == []
