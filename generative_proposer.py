"""generative_proposer.py — Phase 4: the GENERATIVE LLM author front-end (the seal capstone).

The generative analog of edge_search's closed-grammar proposer (`llm_propose_candidates` +
`build_proposer_prompt`). Where the closed author emits a fixed (overlay, params) lattice point, the
generative author emits a COMPOSITION as coordinates — `{legs, ticker, predicted_sign}` — drawn from the
production grammar (generative_grammar). It passes through the SAME read-gate seal: a NUMBERLESS prompt
(no result statistic in scope), a COORDINATE-ONLY gate (`validate_composition` is the grammar wall), and
recording downstream (`run_composition_round`, Phase 3c). It is sealed by INFORMATION, not isolation —
the author sees only the scrubbed corpus + the grammar primitives, never the engine or a t-stat.

This increment is the GATE + the NUMBERLESS PROMPT + the coordinate parser, stub-tested (no model). The
in-process Claude client that fills the `GenLLMProposer` slot is Phase 4b (OFF by default, env-gated, the
exact shape of edge_search's `ClaudeProposer`); promotion stays CLOSED and survivors stay EXPLORATORY
until the Phase-C time-axis holdout exists (docs/llm_proposer_plan.md, docs/read_gate.md).
"""
from __future__ import annotations

from typing import Any, Callable

from edge_search import (
    STRUCTURE_CAMPAIGN,
    Campaign,
    PremiumFamily,
    ProposalBatch,
    _is_onboarded,
    render_proposer_corpus,
)
from generative_grammar import (
    DELTAS,
    DTES,
    MAX_EXPIRATIONS,
    MAX_LEGS,
    RIGHTS,
    SIDES,
    Composition,
    GrammarError,
    Leg,
    canonical_key,
    validate_composition,
)
from read_gate_wire import assert_numberless

# (scrubbed_corpus, onboarded_search_tickers) -> ProposalBatch. The author builds its own NUMBERLESS
# prompt from these two inputs (via build_composition_prompt); the grammar primitives are static, so —
# unlike the closed LLMProposer — there is no menu argument to pass.
GenLLMProposer = Callable[[list[dict[str, Any]], tuple[str, ...]], ProposalBatch]


# --- the coordinate parser: a proposal dict -> a validated Composition ----------------------------------
def _leg_from_dict(d: dict[str, Any]) -> Leg:
    """Parse one proposal leg `{side, right, dte}` + EITHER `delta: <float>` OR `strike: "same"` into a
    `Leg`. Raises (KeyError / TypeError / ValueError) on a malformed leg; the values themselves are
    type-checked downstream by `validate_composition` (the single grammar wall), so this only shapes the
    tuple — it does NOT coerce a value onto the grid (an off-grid or wrong-typed coordinate must be
    REJECTED, not silently snapped)."""
    if not isinstance(d, dict):
        raise TypeError(f'leg must be a dict, got {type(d).__name__}')
    if 'delta' in d:
        strike: tuple = ('delta', d['delta'])          # validate_composition enforces float in DELTAS
    elif d.get('strike') == 'same':
        strike = ('same',)
    else:
        raise ValueError(f"leg needs 'delta' or strike='same', got keys {sorted(d)}")
    return Leg(d['side'], d['right'], strike, d['dte'])


def _composition_from_proposal(p: dict[str, Any]) -> Composition:
    """Build a Composition from a proposal `{legs, predicted_sign}` and run it through the grammar wall.
    Raises `GrammarError` (or KeyError/TypeError/ValueError on a malformed shape) — the SAME gate the
    menu-walker construction uses, so an off-grammar LLM proposal is rejected, never scored."""
    raw = p['legs']
    if not isinstance(raw, list) or not raw:
        raise ValueError('legs must be a non-empty list')
    legs = tuple(_leg_from_dict(d) for d in raw)
    return validate_composition(Composition(legs=legs, predicted_sign=p['predicted_sign']))


# --- the numberless prompt (the load-bearing seal) ------------------------------------------------------
def _render_production_grammar() -> str:
    """Render the production grammar's PRIMITIVES — the leg buckets, the strike kinds, and the space caps.
    Every value here is a GRAMMAR CONSTANT (generative_grammar), never an engine output, so there is no
    result statistic in scope to leak. This is the generative analog of `_render_grammar_menu`, but it
    lists the COMPOSABLE primitives rather than a fixed template lattice."""
    deltas = ', '.join(str(d) for d in DELTAS)
    dtes = ', '.join(str(d) for d in DTES)
    sides = ', '.join(SIDES)
    rights = ', '.join(RIGHTS)
    fams = ', '.join(sorted(f.value for f in PremiumFamily))
    return (
        f'A COMPOSITION is {1}..{MAX_LEGS} legs spanning at most {MAX_EXPIRATIONS} distinct expirations '
        f'(`dte`), delta-hedged. Each leg is:\n'
        f'  - side  in ({sides})            (short SELLS premium, long BUYS it)\n'
        f'  - right in ({rights})\n'
        f'  - a strike, EITHER  delta in ({deltas})   (the |delta| bucket; 0.5 == ATM)\n'
        f'                OR    strike = "same"        (shares the composition\'s single delta-anchored\n'
        f'                                              strike, at a LATER dte — the calendar far leg)\n'
        f'  - dte   in ({dtes})\n'
        f'The premium families a composition may target: {fams}. `predicted_sign` is +1 (the composition '
        f'EARNS a delta-hedged premium) or -1 (it pays one out); the committed convention is +1.'
    )


def build_composition_prompt(scrubbed_corpus: list[dict[str, Any]], onboarded: tuple[str, ...], *,
                             max_proposals: int = 16) -> str:
    """Assemble the NUMBERLESS prompt the generative author sees — the load-bearing seal (the generative
    twin of `build_proposer_prompt`). THE SEAL is `assert_numberless` on the scrubbed-corpus INPUT, run
    first: raw `load_idea_ledger()` rows carry banned KEYS (`t_stat_newey_west`, `p_value`, ...) and fail
    here, loudly, before any model call — the #1 builder bug. The prompt is then built ONLY from
    allow-listed sources: `_render_production_grammar` (grammar constants), `render_proposer_corpus`
    (SAFE_FIELDS only), the onboarded universe, and static instructions. The signature excludes the engine
    and the answer-key ledger, so no result statistic is in scope to format in (pinned by the always-run
    seal test). It is NOT a regex over the assembled string — a banned NAME is a legitimate substring of
    the instructions ('t-statistic'); the guard is the structural assertion on the answer-key-sourced
    input plus allow-listed assembly, exactly as the closed builder documents."""
    assert_numberless(list(scrubbed_corpus), 'gen_proposer_prompt.corpus')
    grammar = _render_production_grammar()
    # The corpus today is the CLOSED-grammar scrubbed view (idea_ledger's {template, params, ...} rows),
    # which render_proposer_corpus renders. Two FORWARD-LOOKING obligations for when the generative loop's
    # own scrubbed corpus (gen_ledger's {legs, predicted_sign} rows) is built (seal-verification nits):
    # (1) ship a generative renderer — render_proposer_corpus is hardwired to template/params; (2) the
    # generative scrub must allow-list inner params/legs keys to grammar coordinates, closing the
    # value-under-a-non-banned-key boundary the closed builder defers to the upstream scrub (a key-NAME
    # guard like assert_numberless can't catch a number carried as a VALUE).
    tried = render_proposer_corpus(scrubbed_corpus)
    universe = ', '.join(sorted(onboarded)) or '(none onboarded)'
    return f"""You are proposing options-overlay experiments for a systematic short-volatility research \
program. Your job is to nominate (composition, ticker) experiments that should harvest a REAL, \
economically-motivated risk premium when delta-hedged.

You will NOT be shown any performance figure, significance value, or return — by design. Propose from \
economic reasoning about WHY a given structure on a given underlier should earn a premium (variance \
risk, skew, term structure, carry), and from what has already been ruled out below. You cannot see \
results; do not ask for them.

## Grammar — compose structures ONLY from these primitives
{grammar}

## Universe — the onboarded tickers you may propose on
{universe}

Propose only these. Naming any other ticker flags it for a human-gated onboarding step, not a trade.

## Already tried — do NOT re-propose these; treat a KILLED verdict as ruled out
{tried}

## Your task
Propose up to {max_proposals} NEW compositions (not in the tried list) that you can defend on economic \
grounds. Output ONLY a JSON array, each element exactly {{"legs": [{{"side": <str>, "right": <str>, \
"delta": <grid value> OR "strike": "same", "dte": <grid value>}}, ...], "ticker": <str>, \
"predicted_sign": <+1 or -1>, "reasoning": <str>}}, where `reasoning` is ONE sentence naming the risk \
premium you expect to harvest and why. No prose outside the JSON; no fields beyond those four."""


# --- the gate: proposals -> validated (Composition, ticker) cells ---------------------------------------
def gate_compositions(author: GenLLMProposer, campaign: Campaign = STRUCTURE_CAMPAIGN, *,
                      corpus: list[dict[str, Any]] | None = None,
                      tried_keys: set[tuple[str, str]] | None = None,
                      max_batch: int = 16) -> tuple[list[tuple[Composition, str]], list[str],
                                                    list[dict[str, Any]], ProposalBatch]:
    """Gate a generative author's COORDINATE-ONLY proposals into validated `(Composition, ticker)` cells —
    the generative twin of `llm_propose_candidates`, keyed on the `(canonical_key, ticker)` CELL. The
    gating order mirrors the closed gate: malform-check, then SEALED ticker (must never run), then
    OFF-CAMPAIGN ticker (a universe edit is human-gated), then the GRAMMAR wall (`validate_composition`),
    then the ONBOARDING check (an un-onboarded SEARCH ticker is flagged, never auto-fetched), then DEDUP
    against the tried set + within the batch (canonical_key EXCLUDES predicted_sign, so a structure and
    its sign-flip collapse to one cell — the sign-shopping guard), capped at `max_batch` (so the author
    cannot burn the e-LOND budget enumerating the whole untried space).

    Returns `(cells, needs_onboard, rejected, batch)`: `cells` is the accepted `(Composition, ticker)`
    list to feed `run_composition_round`; `rejected` is `[{proposal, reason}]`; `batch` carries the raw
    author output + model identity for the provenance audit. NEVER raises on a bad proposal — a malformed
    or off-grammar proposal becomes a `rejected` entry, so one bad cell can't abort the round."""
    corpus = list(corpus or [])
    tried = set(tried_keys or set())
    onboarded = tuple(t for t in campaign.search if _is_onboarded(t))
    batch = author(corpus, onboarded)
    cells: list[tuple[Composition, str]] = []
    needs_onboard: list[str] = []
    rejected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for p in batch.proposals:
        if not isinstance(p, dict) or 'ticker' not in p:
            rejected.append({'proposal': p, 'reason': 'malformed: not a dict with a ticker'})
            continue
        ticker = p['ticker']
        if ticker in campaign.sealed:
            rejected.append({'proposal': p, 'reason': 'sealed ticker — must never run'})
            continue
        if ticker not in campaign.search:
            rejected.append({'proposal': p, 'reason': 'off-campaign ticker (universe edit is human-gated)'})
            continue
        try:
            comp = _composition_from_proposal(p)
        except (GrammarError, KeyError, TypeError, ValueError) as exc:
            rejected.append({'proposal': p, 'reason': f'off-grammar: {exc}'})
            continue
        if ticker not in onboarded:
            if ticker not in needs_onboard:
                needs_onboard.append(ticker)
            continue
        cell = (canonical_key(comp), ticker)
        if cell in tried or cell in seen:
            continue
        seen.add(cell)
        cells.append((comp, ticker))
        if len(cells) >= max_batch:
            break
    return cells, needs_onboard, rejected, batch
