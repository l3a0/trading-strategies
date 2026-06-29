"""Pins for the factor mechanism gate (factor_mechanism.py, H1a of docs/integration_plan.md).

All always-run on the synthetic panel + constructed signals: the numpy OLS helper, the registered
premia, the long-short return series, and — the throughline — `loading_family` typing a factor by the
premium it loads on (momentum -> `trend`), and returning `None` for a mechanism-incoherent (noise) factor.
This is the mechanism COMPUTATION; H1b wires it into the backends' `mechanism()` + the score gate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor.factor_mechanism import (REGISTERED_PREMIA, _ols_tstats, loading_family, long_short_returns,
                              registered_premia)
from test_factor_backend import _panel


def _momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return prices / prices.shift(window) - 1.0


def _vol_panel(seed: int = 11, T: int = 400, N: int = 30) -> pd.DataFrame:
    """A panel with cross-sectional VOL dispersion (half low-vol, half high-vol) and no drift — so the
    `lowvol` premium is the live signal and `trend` is noise. Lets us type a factor as `lowvol`."""
    rng = np.random.default_rng(seed)
    vol = np.where(np.arange(N) < N // 2, 0.004, 0.016)
    logp = np.cumsum(rng.normal(0.0, 1.0, (T, N)) * vol, axis=0)
    idx = pd.date_range('2020-01-01', periods=T, freq='B')
    return pd.DataFrame(100.0 * np.exp(logp), index=idx, columns=[f'S{i:02d}' for i in range(N)])


class TestOLS:
    """The dependency-light OLS t-stat helper — a strong loading reads large, noise reads small."""

    def test_recovers_a_strong_loading(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 1))
        y = 2.0 * x[:, 0] + rng.normal(scale=0.1, size=200)
        assert abs(_ols_tstats(y, x)[1]) > 10               # y = 2x -> large |t|

    def test_finds_no_loading_in_noise(self) -> None:
        rng = np.random.default_rng(1)
        x, y = rng.normal(size=(200, 1)), rng.normal(size=200)
        assert abs(_ols_tstats(y, x)[1]) < 3                # unrelated -> small |t|


class TestRegisteredPremia:
    """The base-style premium return panel the loading regression types against."""

    def test_columns_are_the_committed_family_set(self) -> None:
        assert tuple(registered_premia(_panel()).columns) == REGISTERED_PREMIA

    def test_premium_returns_are_finite_and_present(self) -> None:
        prem = registered_premia(_panel()).dropna()
        assert prem.shape[0] > 100 and np.isfinite(prem.to_numpy()).all()

    def test_long_short_returns_is_a_clean_series(self) -> None:
        ls = long_short_returns(_momentum(_panel()), _panel())
        assert isinstance(ls, pd.Series) and len(ls) > 100 and ls.notna().all()


class TestLoadingFamily:
    """The factor's derive_family: type by the loaded premium, or None for mechanism-incoherent."""

    def test_momentum_types_as_trend(self) -> None:
        # momentum IS the trend signal, so it loads PERFECTLY on the trend premium (the degenerate
        # |t|->inf case) and types as trend — correct: a copy of a premium harvests that premium.
        p = _panel()
        assert loading_family(_momentum(p), p) == 'trend'

    def test_lowvol_signal_types_as_lowvol(self) -> None:
        p = _vol_panel()                                    # the OTHER registered premium types too
        assert loading_family(-p.pct_change().rolling(20).std(), p) == 'lowvol'

    def test_collinear_premia_return_none(self) -> None:
        # a singular design (two identical premium columns) can't be read -> fail-closed, not a crash
        p = _panel()
        prem = registered_premia(p)
        prem['lowvol'] = prem['trend']
        assert loading_family(_momentum(p), p, premia=prem) is None

    def test_noise_is_mechanism_incoherent(self) -> None:
        p = _panel()
        rng = np.random.default_rng(3)
        noise = pd.DataFrame(rng.normal(size=p.shape), index=p.index, columns=p.columns)
        assert loading_family(noise, p) is None             # loads on no registered premium -> fail-closed

    def test_data_insufficient_returns_none(self) -> None:
        short = _panel(T=20)                                 # shorter than the 20-day premium lookback
        assert loading_family(_momentum(short), short) is None

    def test_hurdle_is_the_gate(self) -> None:
        # the hurdle is the significance threshold: drop it to 0 and even noise types as its dominant
        # (insignificant) loading — proving it's the |t| bar, not the data, that gates incoherence.
        p = _panel()
        rng = np.random.default_rng(3)
        noise = pd.DataFrame(rng.normal(size=p.shape), index=p.index, columns=p.columns)
        assert loading_family(noise, p) is None                     # default hurdle: incoherent
        assert loading_family(noise, p, hurdle=0.0) in REGISTERED_PREMIA   # hurdle 0: types anyway
