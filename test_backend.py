"""Pins for the Backend protocol (backend.py, F1 of docs/integration_plan.md).

Two layers:
  * ALWAYS-RUN, no data — `OptionBackend` structurally satisfies `Backend`, and its pure methods
    (enumerate / validate / canonical_key) FORWARD to the existing generative functions unchanged.
  * DATASET-GATED — on real chains, `score` is byte-identical to the existing per-composition row
    (`score_composition` enriched with end + lineage), the row carries the honest-core-facing contract,
    `mechanism` agrees with `score`'s inline family, and `lineage` forwards `_data_lineage_hash`.

F1 is an ADDITIVE adapter — no existing code is edited — so the suite below proves the seam exists and
forwards, while every pre-existing pin proves behavior is unchanged.
"""
from __future__ import annotations

import os

import pytest

from backend import Backend, OptionBackend
from generative_grammar import (Composition, GrammarError, Leg, canonical_key, composition_of,
                                 enumerate_compositions)

# the honest-core-facing contract a VALID (scored) row must emit (read_gate_wire result keys + ids); a
# no-trades / mechanism-incoherent row is measurement_invalid with p_value=None and is tested separately.
CONTRACT = {'ticker', 'predicted_sign', 't_stat_newey_west', 'p_value', 'n_days', 'sign_ok',
            'measurement_invalid', 'family', 'data_lineage_hash', 'end'}


def _have(*tickers: str) -> bool:
    return all(any(os.path.exists(f'{t.lower()}_option_dailies.csv{ext}') for ext in ('', '.gz'))
               for t in tickers)


def _empty_backend() -> OptionBackend:
    """An OptionBackend with no data — enough to exercise the pure (data-independent) methods."""
    return OptionBackend('TEST', [], [], {})


class TestBackendProtocol:
    """ALWAYS-RUN: OptionBackend structurally satisfies the runtime-checkable Backend protocol."""

    def test_option_backend_is_a_backend(self) -> None:
        assert isinstance(_empty_backend(), Backend)        # @runtime_checkable: all six methods present

    def test_has_the_six_methods(self) -> None:
        ob = _empty_backend()
        for m in ('enumerate', 'validate', 'canonical_key', 'mechanism', 'lineage', 'score'):
            assert callable(getattr(ob, m))


class TestOptionBackendForwarding:
    """ALWAYS-RUN: the pure methods forward to the generative functions unchanged (no chains needed)."""

    def test_enumerate_forwards(self) -> None:
        assert _empty_backend().enumerate() == enumerate_compositions(2)

    def test_enumerate_honors_max_legs(self) -> None:
        ob = OptionBackend('TEST', [], [], {}, max_legs=1)
        assert ob.enumerate() == enumerate_compositions(1)

    def test_canonical_key_forwards(self) -> None:
        c = composition_of('iron_condor', {'short_delta': 0.25, 'wing_delta': 0.10, 'dte': 30})
        assert _empty_backend().canonical_key(c) == canonical_key(c)

    def test_validate_returns_a_valid_composition_unchanged(self) -> None:
        c = composition_of('short_vol', {'target_delta': 0.25, 'dte': 30})
        assert _empty_backend().validate(c) is c

    def test_validate_raises_off_grammar(self) -> None:
        bad = Composition(legs=(Leg('short', 'call', ('delta', 0.99), 30),), predicted_sign=1)  # 0.99 off-bucket
        with pytest.raises(GrammarError):
            _empty_backend().validate(bad)

    def test_score_no_trades_branch_is_byte_identical(self) -> None:
        # an EMPTY store -> the composition never enters -> the no-trades / measurement_invalid branch,
        # which the adapter wraps identically (no chains needed). Closes the branch-coverage gap the
        # trading-cell test leaves open; distinct from the mechanism-incoherent branch (which DOES trade).
        from generative_engine import score_composition
        ob = OptionBackend('TEST', ['2020-01-02', '2020-01-03'], [100.0, 101.0], {})
        c = composition_of('short_vol', {'target_delta': 0.25, 'dte': 30})
        row = ob.score(c)
        assert row['no_trades'] is True and row['measurement_invalid'] is True
        assert row['family'] is None and row['p_value'] is None and row['t_stat_newey_west'] is None
        direct = score_composition(c, 'TEST', ['2020-01-02', '2020-01-03'], [100.0, 101.0], {})
        assert row == {**direct, 'end': ob.end, 'data_lineage_hash': ob.lineage(c)}


@pytest.mark.skipif(not _have('MSFT'),
                    reason='needs msft_option_dailies.csv (or .gz twin)')
class TestOptionBackendOnRealChains:
    """DATASET-GATED: on real MSFT chains, the data-bound methods are behavior-identical to the existing
    path — score() is byte-identical to score_composition enriched, carries the contract, and mechanism()
    agrees with the inline family. This is the no-behavior-change proof on real data."""

    def _setup(self):
        from edge_search import _data_lineage_hash, _load_ticker_data
        from generative_engine import score_composition
        store, dates, prices = _load_ticker_data('MSFT')
        ob = OptionBackend('MSFT', dates, prices, store)
        c = composition_of('short_vol', {'target_delta': 0.25, 'dte': 30})
        return ob, c, store, dates, prices, _data_lineage_hash, score_composition

    def test_score_is_byte_identical_to_the_existing_row(self) -> None:
        ob, c, store, dates, prices, lineage, score_composition = self._setup()
        direct = score_composition(c, 'MSFT', dates, prices, store)
        expected = {**direct, 'end': ob.end,
                    'data_lineage_hash': lineage('MSFT', ob.end, ob.capital, ob.checksums)}
        row = ob.score(c)
        assert row == expected                              # the adapter changes nothing
        assert not row.get('no_trades') and row['n_days'] > 0   # non-vacuous: the cell actually traded

    def test_score_emits_the_contract(self) -> None:
        ob, c, *_ = self._setup()
        assert CONTRACT <= set(ob.score(c))

    def test_mechanism_agrees_with_score_family(self) -> None:
        ob, c, *_ = self._setup()
        assert ob.mechanism(c) == ob.score(c)['family']     # same _entry_signature -> derive_family

    def test_lineage_forwards(self) -> None:
        ob, c, store, dates, prices, lineage, _ = self._setup()
        assert ob.lineage(c) == lineage('MSFT', ob.end, ob.capital, ob.checksums)

    def test_mechanism_incoherent_branch_trades_but_fails_closed(self) -> None:
        from edge_search import _load_ticker_data
        from generative_engine import score_composition
        store, dates, prices = _load_ticker_data('MSFT')
        ob = OptionBackend('MSFT', dates, prices, store)
        # a LONG single call is long-vega / single-expiration -> harvests no committed premium ->
        # derive_family None: it TRADES yet fails closed (the foil-paper defense), distinct from no_trades.
        longcall = Composition(legs=(Leg('long', 'call', ('delta', 0.25), 30),), predicted_sign=1)
        row = ob.score(longcall)
        assert not row.get('no_trades') and row['n_days'] > 0                     # it actually traded
        assert row['family'] is None and row['measurement_invalid'] is True and row['p_value'] is None
        assert ob.mechanism(longcall) is None                                     # mechanism agrees
        direct = score_composition(longcall, 'MSFT', dates, prices, store)
        assert row == {**direct, 'end': ob.end, 'data_lineage_hash': ob.lineage(longcall)}
