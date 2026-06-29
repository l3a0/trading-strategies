"""generative_search.py — the generative menu-walker SEARCH, and the pinned negative result it produced.

Runs the deterministic generative proposer (`enumerate_compositions` -> `score_composition` ->
`judge_compositions_against_published`, Phase 3c) over a bounded slice of the production grammar on the
onboarded search tickers, and reports the lifetime-e-LOND saturation (`search_saturation`). It is the
generative twin of `edge_search`'s structure campaign, and like every scout in this repo it is
EXPLORATORY — sample-spending, kill-or-justify, NEVER a registered verdict (docs/generative_search.md).

THE PINNED FINDING (2026-06-26): the search adds NO survivor and was saturated before it began. The
published 75-cell lifetime ledger already sits PAST THE BAR — the next cell must clear t>=6.35
(p<=1.1e-10) while the strongest cell ever observed is t~=2.17 — so with R=0 discoveries to loosen the
e-LOND discount, more rounds only tighten it. The mechanism gate (`derive_family`, the foil-paper
defense) and the must-trade guard reject the majority of generative compositions BEFORE the FDR pool, and
the coherent remainder is weak. The binding constraint is not search cleverness; it is the Phase-C
time-axis holdout (which would make any survivor meaningful) and the absence of a real volatility-risk
premium at honest significance on these names.

RECORDED to the audit log: with `--record` the judged cells are appended to the committed
`gen_ledger.jsonl` (design A — the generative twin of `idea_ledger.jsonl`), the audit record of every
comparison, its e-LOND verdict, and its data lineage. "Exploratory" means NOT a registered verdict (a
survivor would still need manual pre-registration + the Phase-C holdout) — it does NOT mean unrecorded:
the structure campaign records its exploratory 56-cell batch too, and the lifetime e-LOND budget is only
honest if every look is recorded. The committed `gen_ledger.jsonl` IS the pinned artifact.
"""
from __future__ import annotations
from common.paths import data_path

from typing import Any

GEN_LEDGER_PATH = data_path('gen_ledger.jsonl')


def _composition_slice(n_singles: int, n_doubles: int) -> list:
    """The deterministic bounded slice the search walks: the first `n_singles` single-leg structures and
    the first `n_doubles` same-expiration two-leg structures, in canonical-key order. A representative
    sample of the menu-walk — not the whole 2,030-cell space, which the saturation makes pointless to
    enumerate (the bar is unreachable from cell 1)."""
    from generative.generative_grammar import enumerate_compositions
    comps = enumerate_compositions()
    singles = [c for c in comps if len(c.legs) == 1][:n_singles]
    doubles = [c for c in comps if len(c.legs) == 2][:n_doubles]
    return singles + doubles


def comp_desc(comp) -> str:
    """A human-readable one-line description of a composition's legs (for the CLI + the doc)."""
    parts = []
    for leg in comp.legs:
        k = f'd{leg.strike[1]}' if leg.strike[0] == 'delta' else 'sameK'
        parts.append(f'{leg.side} {leg.right} {k} {leg.dte}d')
    return ' + '.join(parts)


def run_generative_search(tickers: tuple[str, ...] | None = None, *,
                          n_singles: int = 5, n_doubles: int = 5,
                          record: bool = False, gen_path: str = GEN_LEDGER_PATH) -> dict[str, Any]:
    """Walk the bounded composition slice on each onboarded SEARCH ticker (TLT stays sealed by omission),
    score every cell (the generative kill-gate, with its inline mechanism gate), judge the batch over the
    lifetime e-LOND stream with the published ledger as the read-only head, and report the saturation. With
    `record=True` the judged cells are appended to the committed `gen_path` (the audit log) — deduped on
    the (canonical_key, ticker) cell, e-LOND verdict and all. Returns the decisive-outputs bundle. Slow:
    one engine pass per cell."""
    from search.edge_search import (STRUCTURE_CAPITAL, STRUCTURE_END, STRUCTURE_SEARCH, _data_lineage_hash,
                             _load_ticker_data, load_idea_ledger, search_saturation)
    from generative.generative_engine import (judge_compositions_against_published, record_compositions,
                                   score_composition)

    tickers = tuple(tickers) if tickers else STRUCTURE_SEARCH
    slice_ = _composition_slice(n_singles, n_doubles)
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        store, dates, prices = _load_ticker_data(ticker)
        lineage = _data_lineage_hash(ticker, STRUCTURE_END, STRUCTURE_CAPITAL)
        for comp in slice_:
            r = score_composition(comp, ticker, dates, prices, store)
            r['end'], r['data_lineage_hash'], r['desc'] = STRUCTURE_END, lineage, comp_desc(comp)
            rows.append(r)

    fresh = judge_compositions_against_published(rows)
    recorded = record_compositions(fresh, gen_path) if record else 0
    published = load_idea_ledger()
    sat = search_saturation(published + fresh)
    coherent = sum(1 for r in rows if not r.get('measurement_invalid'))
    return {
        'tickers': list(tickers),
        'slice': [comp_desc(c) for c in slice_],
        'n_published': len(published),
        'n_generative': len(rows),
        'n_fresh': len(fresh),
        'recorded': recorded,
        'gen_coherent': coherent,
        'gen_invalid': len(rows) - coherent,
        'gen_survivors': sum(1 for r in fresh if r.get('elond_survivor')),
        'rows': rows,
        'saturation': sat,
    }


def main() -> None:
    import sys
    from search.edge_search import format_saturation
    record = '--record' in sys.argv
    b = run_generative_search(record=record)
    print(f"\ngenerative search: {b['n_generative']} cells on {len(b['tickers'])} tickers "
          f"({b['gen_coherent']} coherent, {b['gen_invalid']} rejected by the mechanism/trade gate); "
          f"{b['n_fresh']} fresh vs the {b['n_published']}-cell published head; "
          f"survivors = {b['gen_survivors']}")
    print(format_saturation(b['saturation']))
    if record:
        print(f"\ngen_ledger: +{b['recorded']} comparison(s) recorded to {GEN_LEDGER_PATH} "
              f"(deduped; e-LOND judged over the lifetime stream — the audit log)")
    elif b['n_fresh']:
        print(f"\n(dry — pass --record to append the {b['n_fresh']} cells to {GEN_LEDGER_PATH})")


if __name__ == '__main__':
    main()
