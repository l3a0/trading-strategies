"""factor_search.py — the factor menu-walker proposer (F4 of docs/integration_plan.md).

The deterministic search loop over the factor grammar, the factor analog of `edge_search.run_proposer_round`
(option menu-walker) and `generative_search.run_generative_search` (composition menu-walker): enumerate a
backend's bounded candidate slice, SCORE each through the same honest-core path, and JUDGE the batch over
the e-LOND FDR control of record. It is backend-GENERIC — it drives any `backend` with `enumerate()` +
`score()`, so it walks the named primitives (`FactorBackend`) or the expression grammar
(`GrammarFactorBackend`) unchanged.

EXPLORATORY, and PROMOTION STAYS CLOSED. A flagged cell (`elond_survivor`) escalates to manual
pre-registration + the Phase-C time-axis holdout; it is NEVER auto-promoted — the time-axis holdout is the
binding defense, and it does not exist yet (docs/integration_plan.md, the doc's caveat). The mechanism gate
(H1b) already fails closed for a factor that loads on no registered premium (`p_value=None` -> `e=0`), so an
incoherent factor cannot survive on a lucky IC — but a coherent survivor is still only a candidate.

DRY BY DEFAULT. `record=False` scores + judges and mutates nothing — the loop, on demand. The `record=True`
path appends fresh `(canonical_key, universe)` cells to a committed factor ledger; it is gated on a REAL
equity panel (the only panel today is synthetic, so there is no committed `factor_ledger.jsonl` — recording
a synthetic null is not a real exploration). The future LLM author (H2) plugs into this loop, swapping its
coordinate output for the enumerator while the score/judge/record stay identical (the option-domain pattern).
"""
from __future__ import annotations

import json
import os
from typing import Any

from evalue_fdr import online_fdr_survivors

FACTOR_LEDGER_PATH = 'factor_ledger.jsonl'


def factor_search_summary(judged: list[dict]) -> dict[str, int]:
    """Partition the judged rows: `coherent` (the mechanism gate typed a family — these carry a p and can
    flag), `incoherent` (mechanism-incoherent, fail-closed — a t kept for transparency but no p), and
    `data_invalid` (too few IC periods — no t at all), plus the `survivors` count under e-LOND."""
    return {
        'coherent': sum(1 for r in judged if r['mechanism_ok']),
        'incoherent': sum(1 for r in judged if not r['mechanism_ok'] and r['t_stat_newey_west'] is not None),
        'data_invalid': sum(1 for r in judged if r['t_stat_newey_west'] is None),
        'survivors': sum(1 for r in judged if r['elond_survivor']),
    }


def _record_factor_cells(judged: list[dict], ledger_path: str) -> int:
    """Append fresh `(key, ticker)` cells to the committed factor ledger, deduped against what it holds.
    Returns the count newly appended; e-LOND is online, so an appended verdict is permanent (the
    composition/option ledger pattern)."""
    existing: set[tuple[str, str]] = set()
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            existing = {(r['key'], r['ticker']) for r in (json.loads(line) for line in f if line.strip())}
    added = 0
    with open(ledger_path, 'a') as f:
        for r in judged:
            cell = (r['key'], r['ticker'])
            if cell in existing:
                continue
            existing.add(cell)
            f.write(json.dumps(r, sort_keys=True) + '\n')
            added += 1
    return added


def run_factor_search(backend: Any, *, record: bool = False, ledger_path: str = FACTOR_LEDGER_PATH,
                      prior_rows: list[dict] | None = None) -> dict[str, Any]:
    """Walk `backend`'s bounded candidate slice, score each, and judge the batch over the e-LOND stream.
    `prior_rows` is the lifetime factor stream placed AHEAD of the fresh batch (empty at head-of-stream, so
    the batch IS the whole stream and `online_fdr_survivors` is the lifetime judge). `record=False` (the
    default) mutates nothing; `record=True` appends the fresh cells to `ledger_path`. Returns the scored
    count, the recorded count, the judged rows, and the `factor_search_summary` partition."""
    rows = [backend.score(c) for c in backend.enumerate()]
    prior = list(prior_rows) if prior_rows else []
    judged = online_fdr_survivors(prior + rows)[len(prior):]              # prior fixes the discount head
    recorded = _record_factor_cells(judged, ledger_path) if record else 0
    return {'scored': len(rows), 'recorded': recorded, 'judged': judged, **factor_search_summary(judged)}
