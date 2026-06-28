"""Pins for the factor backend (factor_backend.py, F2 of docs/integration_plan.md).

ALL ALWAYS-RUN — the proof is on a SYNTHETIC, deterministic equity panel (no chains, no network, no
Qlib). The F2 deliverable is the wiring: a FactorBackend satisfies the SAME `Backend` protocol as the
option path (F1), and its IC-based score row feeds the SAME honest core (`online_fdr_survivors`, the
e-LOND control) unchanged. The mechanism gate (the loading regression) is H1, so `family` is None and
`mechanism` returns None here — by design, not omission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend import Backend
from factor_backend import (FACTOR_NAMES, WINDOWS, Factor, FactorBackend, FactorPrimitiveError,
                            factor_key)

# the honest-core-facing contract a valid factor row emits (the same keys the option path emits)
CONTRACT = {'ticker', 'predicted_sign', 't_stat_newey_west', 'p_value', 'n_days', 'sign_ok',
            'measurement_invalid', 'family', 'data_lineage_hash', 'end'}


def _panel(seed: int = 7, T: int = 400, N: int = 30) -> pd.DataFrame:
    """A synthetic price panel with PERSISTENT per-ticker drift, so a high trailing return (momentum)
    predicts a high forward return cross-sectionally — a deterministic panel where momentum has a real,
    positive IC and reversal (its negation) a negative one."""
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0, 0.002, N)                       # per-ticker persistent drift (the signal)
    shocks = rng.normal(0.0, 0.008, (T, N))
    logp = np.cumsum(drift + shocks, axis=0)
    idx = pd.date_range('2020-01-01', periods=T, freq='B')
    return pd.DataFrame(100.0 * np.exp(logp), index=idx, columns=[f'S{i:02d}' for i in range(N)])


def _backend(**kw) -> FactorBackend:
    return FactorBackend('SYNTH', _panel(), checksum='deadbeef', **kw)


class TestFactorBackendProtocol:
    """FactorBackend satisfies the runtime-checkable Backend protocol — the same seam as OptionBackend."""

    def test_is_a_backend(self) -> None:
        assert isinstance(_backend(), Backend)

    def test_has_the_six_methods(self) -> None:
        fb = _backend()
        for m in ('enumerate', 'validate', 'canonical_key', 'mechanism', 'lineage', 'score'):
            assert callable(getattr(fb, m))


class TestFactorGrammar:
    """The pure grammar methods: enumerate, validate (raises off-grammar), sign-excluded canonical key."""

    def test_enumerate_is_the_primitive_slice(self) -> None:
        fb = _backend()
        fs = fb.enumerate()
        assert len(fs) == len(FACTOR_NAMES) * len(WINDOWS)
        assert all(isinstance(f, Factor) and f.predicted_sign == 1 for f in fs)

    def test_validate_raises_off_grammar(self) -> None:
        fb = _backend()
        with pytest.raises(FactorPrimitiveError):
            fb.validate(Factor('momentum', 7, 1))            # 7 is not a WINDOWS bucket
        with pytest.raises(FactorPrimitiveError):
            fb.validate(Factor('nonsense', 20, 1))           # off-menu name

    def test_canonical_key_excludes_sign(self) -> None:
        # a factor and its sign-flipped twin share one identity (the sign-shopping guard)
        assert factor_key(Factor('momentum', 20, 1)) == factor_key(Factor('momentum', 20, -1))
        assert factor_key(Factor('momentum', 20, 1)) != factor_key(Factor('reversal', 20, 1))

    def test_canonical_key_forwards(self) -> None:
        f = Factor('lowvol', 60, 1)
        assert _backend().canonical_key(f) == factor_key(f)


class TestFactorScoring:
    """The IC scorer on the synthetic panel: momentum is predictive, reversal is its exact negation, and
    the row carries the honest-core contract. `family`/`mechanism` are None (H1 builds the gate)."""

    def test_momentum_is_predictive_and_row_is_valid(self) -> None:
        row = _backend().score(Factor('momentum', 20, 1))
        assert row['measurement_invalid'] is False
        assert row['n_days'] >= 30 and row['t_stat_newey_west'] > 1.0    # real positive IC t-stat
        assert row['sign_ok'] is True and row['p_value'] is not None
        assert CONTRACT <= set(row)

    def test_reversal_is_the_exact_negation_of_momentum(self) -> None:
        fb = _backend()
        for w in WINDOWS:                                    # robust across every window, not just one
            mom = fb.score(Factor('momentum', w, 1))['t_stat_newey_west']
            rev = fb.score(Factor('reversal', w, 1))['t_stat_newey_west']
            assert rev == pytest.approx(-mom)                # reversal == -momentum, period by period

    def test_mechanism_derives_a_family(self) -> None:
        # H1b live: momentum loads on the trend premium -> typed 'trend' (a MEASUREMENT), not None
        fb = _backend()
        f = Factor('momentum', 20, 1)
        assert fb.mechanism(f) == 'trend'
        row = fb.score(f)
        assert row['family'] == 'trend' and row['mechanism_ok'] is True

    def test_too_few_periods_is_measurement_invalid(self) -> None:
        # a panel far shorter than the lookback + min IC periods -> data-insufficiency, fails closed
        short = FactorBackend('SHORT', _panel(T=40), checksum='x')
        row = short.score(Factor('momentum', 60, 1))         # 60-day window on a 40-day panel
        assert row['measurement_invalid'] is True and row['p_value'] is None
        assert CONTRACT <= set(row)                          # still emits the contract

    def test_synthetic_panel_is_actually_momentum_predictive(self) -> None:
        # INDEPENDENT of the IC machinery: trailing returns rank-correlate positively with forward returns
        # on the panel, so the t-stat test above measures a TRUE signal, not a construction artifact an IC
        # bug could also hide. (Computed inline, not via information_coefficient.)
        p = _panel()
        trailing, forward = p / p.shift(20) - 1.0, p.shift(-1) / p - 1.0
        ics = [trailing.loc[d].rank().corr(forward.loc[d].rank()) for d in p.index   # rank-Pearson == Spearman
               if trailing.loc[d].notna().sum() >= 3 and forward.loc[d].notna().sum() >= 3]
        assert np.nanmean(ics) > 0.05                        # the panel really is momentum-predictive

    def test_all_primitives_type_coherently(self) -> None:
        # H1b live: every F2 primitive is price-based, so it loads on a registered premium (trend/lowvol)
        # -> typed, mechanism_ok True. (The fail-closed path needs an INCOHERENT factor — exercised by a
        # None-typing grammar Expr in test_factor_engine.py and the ic_to_row gate test below.)
        from factor_mechanism import REGISTERED_PREMIA
        fb = _backend()
        for f in fb.enumerate():
            row = fb.score(f)
            assert row['family'] in REGISTERED_PREMIA and row['mechanism_ok'] is True
            assert fb.mechanism(f) == row['family']

    def test_ic_to_row_fails_closed_on_incoherent_family(self) -> None:
        # the gate logic directly: family=None keeps the t for transparency but NEVER flags (p=None,
        # measurement_invalid); a typed family scores normally. (Mirrors the option path's family-None.)
        from factor_backend import ic_to_row
        ic = np.linspace(0.05, 0.15, 100)                    # mean 0.1, real variance -> a big t
        coherent = ic_to_row(ic, 'trend', 1, 'k', 'U', '2026', 'lin')
        incoherent = ic_to_row(ic, None, 1, 'k', 'U', '2026', 'lin')
        assert coherent['measurement_invalid'] is False and coherent['p_value'] is not None
        assert incoherent['measurement_invalid'] is True and incoherent['p_value'] is None
        assert incoherent['t_stat_newey_west'] == coherent['t_stat_newey_west']   # t kept for transparency
        assert incoherent['family'] is None and incoherent['mechanism_ok'] is False


class TestFactorBackendFeedsHonestCore:
    """THE F2 DELIVERABLE: a second backend's rows feed the SAME honest core unchanged. The factor score
    rows flow through `online_fdr_survivors` (the e-LOND FDR control of record) and get e-LOND verdicts,
    exactly as the option backend's rows do — the seam is domain-general."""

    def test_factor_rows_get_elond_verdicts(self) -> None:
        from evalue_fdr import online_fdr_survivors
        fb = _backend()
        rows = [fb.score(f) for f in fb.enumerate()]
        judged = online_fdr_survivors(rows)
        assert len(judged) == len(rows)
        for r in judged:
            assert 'e_value' in r and 'elond_level' in r and 'elond_survivor' in r
            assert isinstance(r['elond_survivor'], bool)
        assert any(r['e_value'] > 0 for r in judged)         # calibration actually ran (not a vacuous pass)

    def test_data_insufficient_rows_never_flag(self) -> None:
        # a measurement_invalid factor (p=None -> e=0) feeds through and is never a survivor. A 40-day
        # panel is a MIX: short windows have enough IC periods, long windows don't — so assert the
        # precise claim (invalid rows never flag), not that all are invalid.
        from evalue_fdr import online_fdr_survivors
        short = FactorBackend('SHORT', _panel(T=40), checksum='x')
        judged = online_fdr_survivors([short.score(f) for f in short.enumerate()])
        invalid = [r for r in judged if r['measurement_invalid']]
        assert invalid                                       # the long-window factors are data-insufficient
        assert all(r['e_value'] == 0 for r in invalid)       # p=None calibrates to e=0 (the contract)...
        assert not any(r['elond_survivor'] for r in invalid)    # ...so they never flag
