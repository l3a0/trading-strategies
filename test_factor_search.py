"""Pins for the factor menu-walker proposer (factor_search.py, F4 of docs/integration_plan.md).

All always-run on synthetic panels. The proposer is the search LOOP — enumerate -> score -> e-LOND-judge.
The deliverables: it FLAGS a real signal (a planted-momentum panel -> survivors), REJECTS noise (a
random-walk panel -> 0 survivors, the null), the mechanism gate's incoherent factors never survive, it is
backend-GENERIC (primitives + grammar), and the record path appends + dedups. Two grammar searches run once
(module fixtures) since each walks the full 63-Expr slice.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from factor_backend import FactorBackend
from factor_engine import GrammarFactorBackend
from factor_search import run_factor_search
from test_factor_backend import _panel


def _noise_panel(seed: int = 99, T: int = 200, N: int = 30) -> pd.DataFrame:
    """A pure random walk — no drift, no vol dispersion — so no factor is predictive (the null panel)."""
    logp = np.cumsum(np.random.default_rng(seed).normal(0.0, 0.01, (T, N)), axis=0)
    idx = pd.date_range('2020-01-01', periods=T, freq='B')
    return pd.DataFrame(100.0 * np.exp(logp), index=idx, columns=[f'S{i:02d}' for i in range(N)])


@pytest.fixture(scope='module')
def planted() -> dict:
    """One grammar search over a panel with a REAL planted momentum signal."""
    return run_factor_search(GrammarFactorBackend('SYNTH', _panel(T=200), checksum='x'))


@pytest.fixture(scope='module')
def noise() -> dict:
    """One grammar search over a no-signal random-walk panel."""
    return run_factor_search(GrammarFactorBackend('NOISE', _noise_panel(), checksum='x'))


class TestFactorSearch:
    def test_partitions_the_full_slice(self, planted: dict) -> None:
        assert planted['scored'] == 63 and planted['recorded'] == 0          # the 63-Expr slice, dry
        assert planted['coherent'] + planted['incoherent'] + planted['data_invalid'] == 63
        assert all('elond_survivor' in row for row in planted['judged'])     # every cell judged

    def test_flags_a_planted_signal(self, planted: dict) -> None:
        # the search CAN flag a real factor — the planted momentum is found (survivors > 0). Still
        # EXPLORATORY: a survivor escalates to manual pre-reg + the Phase-C holdout, never auto-promoted.
        assert planted['survivors'] > 0

    def test_rejects_noise(self, noise: dict) -> None:
        # the null: factors still TYPE coherently (the mechanism gate is orthogonal to predictiveness),
        # but NONE survive (e-LOND rejects a non-predictive signal).
        assert noise['survivors'] == 0 and noise['coherent'] > 0

    def test_incoherent_factors_never_survive(self, planted: dict) -> None:
        # the foil-paper defense, live: a mechanism-incoherent factor (family None, p=None -> e=0) can
        # never be a survivor, even amid a panel full of real signal.
        incoherent = [r for r in planted['judged']
                      if not r['mechanism_ok'] and r['t_stat_newey_west'] is not None]
        assert incoherent and not any(r['elond_survivor'] for r in incoherent)

    def test_backend_generic_over_primitives(self) -> None:
        # the loop drives any Backend with enumerate + score — here the named primitives (fast, 9 cells)
        r = run_factor_search(FactorBackend('SYNTH', _panel(T=200), checksum='x'))
        assert r['scored'] == 9 and r['coherent'] == 9                       # all primitives type coherently

    def test_record_path_appends_and_dedups(self, tmp_path) -> None:
        ledger = str(tmp_path / 'factor_ledger.jsonl')
        fb = FactorBackend('SYNTH', _panel(T=200), checksum='x')             # primitives — fast
        first = run_factor_search(fb, record=True, ledger_path=ledger)
        assert first['recorded'] == 9 and os.path.exists(ledger)
        again = run_factor_search(fb, record=True, ledger_path=ledger)
        assert again['recorded'] == 0                                        # same cells -> dedup
