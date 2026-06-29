"""Pins for the generative-search exploration (generative_search.py, docs/generative_search.md).

Three layers, mirroring the scout discipline:
  * ALWAYS-RUN, no data — the committed 75-cell published ledger is already PAST THE BAR (the next cell
    must clear a t the strongest observed cell cannot reach, with zero discoveries to loosen the e-LOND
    discount). This IS the diminishing-returns finding.
  * ALWAYS-RUN, no data — the committed `gen_ledger.jsonl` audit log: the recorded generative run, every
    comparison + its e-LOND verdict, 0 survivors, distinct cells (the way `TestIdeaLedger` pins the
    closed-grammar ledger).
  * DATASET-GATED — re-running the menu-walker on real chains reproduces the null: no survivor, the
    mechanism/trade gate rejects the majority.

EXPLORATORY, not a registered verdict — recording the null prevents re-deriving it; it does not promote
the scout.
"""
from __future__ import annotations

import json
import os
from common.paths import DATA_DIR, data_path

import pytest

GEN_LEDGER = data_path('gen_ledger.jsonl')


def _have(*tickers: str) -> bool:
    return all(any(os.path.exists(os.path.join(DATA_DIR, f'{t.lower()}_option_dailies.csv{ext}')) for ext in ('', '.gz'))
               for t in tickers)


class TestPublishedStreamSaturated:
    """ALWAYS-RUN: the published lifetime ledger (committed idea_ledger.jsonl) is already saturated — the
    diminishing-returns finding, provable from the ledger alone, no chains needed."""

    def test_zero_survivors_and_past_the_bar(self) -> None:
        from search.edge_search import load_idea_ledger, search_saturation
        sat = search_saturation(load_idea_ledger())
        assert sat['survivors'] == 0                       # nothing flagged under lifetime e-LOND
        assert sat['past_bar'] is True                     # the bar overtook the empirical ceiling
        # the gap is not marginal: the next cell needs a t the strongest observed cell cannot reach.
        assert sat['required_t'] > sat['ceiling_t']
        assert sat['required_t'] > 5.0 and sat['ceiling_t'] < 3.0

    def test_more_cells_only_raise_the_bar(self) -> None:
        # R = 0 discoveries, so the registered discount is monotone: a later position faces a higher bar.
        from search.edge_search import load_idea_ledger
        from search.evalue_fdr import next_flag_threshold
        n = len(load_idea_ledger())
        assert next_flag_threshold(n + 50, 0).t_required > next_flag_threshold(n, 0).t_required


class TestGenLedgerRecorded:
    """ALWAYS-RUN: the committed gen_ledger.jsonl audit log — the recorded generative run. No chains: it
    reads the committed artifact, exactly as TestIdeaLedger pins the closed-grammar ledger."""

    def _rows(self):
        assert os.path.exists(GEN_LEDGER), 'gen_ledger.jsonl is committed — run `python generative_search.py --record`'
        return [json.loads(line) for line in open(GEN_LEDGER) if line.strip()]

    def test_audit_log_is_present_and_well_formed(self) -> None:
        rows = self._rows()
        assert len(rows) > 0
        for r in rows:
            assert r['phase'] == 'structure'
            assert isinstance(r['key'], str) and isinstance(r['ticker'], str)
            assert 'elond_survivor' in r and 'data_lineage_hash' in r

    def test_zero_survivors_recorded(self) -> None:
        # the null, on the record: every recorded generative comparison was killed under lifetime e-LOND.
        assert all(not r.get('elond_survivor') for r in self._rows())

    def test_cells_are_distinct_and_sealed_ticker_absent(self) -> None:
        rows = self._rows()
        cells = {(r['key'], r['ticker']) for r in rows}
        assert len(cells) == len(rows)                     # one row per (structure, ticker) cell
        from search.edge_search import STRUCTURE_SEALED
        assert all(r['ticker'] not in STRUCTURE_SEALED for r in rows)   # TLT never ran


@pytest.mark.skipif(not _have('MSFT', 'GLD'),
                    reason='needs msft_option_dailies + gld_option_dailies (or .gz twins)')
class TestGenerativeSearchNull:
    """DATASET-GATED: re-running the menu-walker on real chains reproduces the null — no survivor, the
    mechanism/trade gate rejects the majority. Dry (record=False): the committed ledger is the artifact."""

    def test_search_adds_no_survivor(self) -> None:
        from generative.generative_search import run_generative_search
        b = run_generative_search(('MSFT', 'GLD'))
        assert b['n_generative'] == 20                     # 2 tickers x (5 single + 5 two-leg) slice
        assert b['recorded'] == 0                          # dry by default
        assert b['gen_survivors'] == 0                     # the null: nothing flags
        assert b['gen_invalid'] >= b['gen_coherent']       # the mechanism/trade gate rejects the majority
        assert b['saturation']['past_bar'] is True         # adding the cells keeps the stream past the bar
