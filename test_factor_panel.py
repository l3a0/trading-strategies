"""Pins for the real equity-universe panel (factor_panel.py).

Two layers, like the option real-chain tests: an ALWAYS-RUN layer for the universe / loader / checksum
logic (synthetic, no network, no committed CSV), and a DATASET-GATED layer that loads the committed
`factor_universe_prices.csv` and pins the FULL exploration headline on REAL equities — 0 of 63 grammar
factors survive the e-LOND bar, all 63 coherent. The full 63-factor search runs in ~2s (the IC + long-short
returns are vectorized), so the headline is pinned directly here, not just reproduced by `python
factor_panel.py` / documented in docs/factor_real_panel.md. EXPLORATORY: promotion CLOSED, any survivor a
pre-registration candidate, none here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_panel as fp
from factor_engine import GrammarFactorBackend

_HAVE_PANEL = fp.panel_available()


def _synthetic_panel(seed: int = 3, T: int = 200, N: int = 20) -> pd.DataFrame:
    logp = np.cumsum(np.random.default_rng(seed).normal(0.0, 0.01, (T, N)), axis=0)
    idx = pd.date_range('2020-01-01', periods=T, freq='B')
    return pd.DataFrame(100.0 * np.exp(logp), index=idx, columns=[f'S{i:02d}' for i in range(N)])


class TestFactorUniverse:
    """Always-run: the committed universe + the loader/checksum/backend wiring (no network, no CSV)."""

    def test_universe_is_pre_specified_and_deduped(self) -> None:
        assert len(fp.FACTOR_UNIVERSE) == len(set(fp.FACTOR_UNIVERSE)) >= 30   # a real cross-section
        assert all(t.isupper() and t.isalpha() for t in fp.FACTOR_UNIVERSE)

    def test_checksum_is_deterministic_and_content_addressed(self) -> None:
        p = _synthetic_panel()
        assert fp.panel_checksum(p) == fp.panel_checksum(p.copy())             # deterministic
        assert fp.panel_checksum(p) != fp.panel_checksum(p * 1.01)             # content-addressed

    def test_make_backend_wires_the_panel_with_its_checksum(self) -> None:
        p = _synthetic_panel()
        backend = fp.make_factor_backend(panel=p)
        assert isinstance(backend, GrammarFactorBackend)
        assert backend.universe == fp.FACTOR_PANEL_NAME and backend.checksum == fp.panel_checksum(p)


@pytest.mark.skipif(not _HAVE_PANEL, reason='committed factor_universe_prices.csv not present')
class TestRealPanelExploration:
    """Dataset-gated: the factor stack on the REAL US_LARGE_CAP panel (the committed snapshot)."""

    def test_panel_loads_clean_and_deterministic(self) -> None:
        panel = fp.load_factor_panel()
        assert set(fp.FACTOR_UNIVERSE) <= set(panel.columns)                   # every universe ticker present
        assert panel.shape[0] >= 2000 and panel.isna().to_numpy().sum() == 0   # long history, no gaps
        assert fp.panel_checksum(panel) == fp.panel_checksum(fp.load_factor_panel())

    def test_full_search_is_a_conservative_null(self) -> None:
        # THE HEADLINE, pinned directly (the vectorized search is ~2s): on the committed snapshot, all 63
        # grammar factors type coherently (real equities have real trend/vol structure) and 0 survive the
        # e-LOND bar — a conservative null on real US large-caps. The strongest HAC-corrected IC t-stat is
        # below the conventional t=2 bar, so nothing clears even the head-of-stream e-LOND threshold.
        from factor_search import run_factor_search
        r = run_factor_search(fp.make_factor_backend())
        assert r['scored'] == 63 and r['coherent'] == 63 and r['incoherent'] == 0
        assert r['survivors'] == 0                                              # the null — nothing flagged
        ts = [abs(row['t_stat_newey_west']) for row in r['judged'] if row['t_stat_newey_west'] is not None]
        assert max(ts) < 2.0                                                    # strongest IC t below the t=2 bar
